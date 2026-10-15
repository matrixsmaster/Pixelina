/* This file is a part of 'Pixelina' Image Generator, Editor, and Viewer.
 *
 * (C) Dmitry 'sciloaf' Solovyev, 2026
 *
 * License: MIT
 */
#include <algorithm>
#include <array>
#include <cstring>
#include <map>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <deque>
#include "backend_vulkan.h"

#ifdef PIXELINA_WITH_VULKAN

#include <vulkan/vulkan.h>
#include "tensor.h"
#include "inference.h"
#include "compute_shader.h"
#include "matmul_shader.h"
#include "matvec_shader.h"
#include "im2col_shader.h"
#include "attention_shader.h"
#include "matmul_mmq_shader.h"
#include "quantize_shader.h"
#include "attention_subgroup_shader.h"
#include "matmul_coop_shader.h"
#include "groupnorm_partial_shader.h"
#include "groupnorm_merge_shader.h"
#include "groupnorm_apply_shader.h"

struct VulkanBuffer {
    VkBuffer buffer = VK_NULL_HANDLE; // shader storage buffer
    VkDeviceMemory memory = VK_NULL_HANDLE; // coherent host-visible allocation
    size_t bytes = 0; // descriptor-visible size
    VkDeviceSize allocated = 0; // actual allocation size
    void* pointer = nullptr; // persistent coherent mapping
    uint64_t last_use = 0; // last submission referencing this allocation
};

struct VulkanSample {
    unsigned kind = 0; // kernel family and quantization used for timing prediction
    double work = 0; // shape-dependent estimate of arithmetic or element visits
};

struct VulkanBatch {
    VkCommandBuffer command = VK_NULL_HANDLE; // command storage retained until completion
    VkFence fence = VK_NULL_HANDLE; // completion of this individual submission
    VkQueryPool queries = VK_NULL_HANDLE; // per-dispatch GPU timestamps when supported
    std::array<VkDescriptorSet,VK_DISPATCH_BATCH> descriptors = {}; // immutable sets while queued
    std::array<VulkanSample,VK_DISPATCH_BATCH> samples = {}; // timings used to update predictions
    uint32_t recorded = 0; // physical dispatches in the command buffer
    uint64_t serial = 0; // submission lifetime identity
    uint64_t first = 0,last = 0; // logical operator range for diagnostics
    std::string operation; // last logical operator for failure reporting
    double predicted = 0; // estimated GPU seconds in this submission
    bool pending = false; // submitted resources cannot yet be reset
    uint32_t* marker = nullptr; // coherent completion token checked independently of the fence
    VkBuffer marker_buffer = VK_NULL_HANDLE; // token written after all compute work
    uint32_t expected = 0; // fresh nonzero token for the current submission
    std::chrono::steady_clock::time_point submitted; // host wait diagnostics and timing fallback
};

struct VulkanCachedWeight {
    void* pointer = nullptr; // mapped quantized tensor storage
    size_t bytes = 0; // cache accounting
    uint64_t used = 0; // dispatch counter for eviction
};

using VulkanPush = std::array<uint32_t,VK_PUSH_WORDS>;

static VulkanOptions configured_options;

bool vulkan_configure(const VulkanOptions &options)
{
    if (!std::isfinite(options.batch_seconds) || options.batch_seconds <= 0 || options.batch_seconds > VULKAN_MAX_BATCH_SECONDS) return false;
    if (options.kernels != VULKAN_KERNEL_AUTO && options.kernels != VULKAN_KERNEL_PORTABLE && options.kernels != VULKAN_KERNEL_OPTIMIZED) return false;
    configured_options = options;
    return true;
}

static thread_local std::shared_ptr<VulkanState> active_state;

struct VulkanState {
    VkInstance instance = VK_NULL_HANDLE; // compute-only instance, no surface extensions
    VkPhysicalDevice physical = VK_NULL_HANDLE; // selected compute device
    VkDevice device = VK_NULL_HANDLE; // logical device
    VkQueue queue = VK_NULL_HANDLE; // compute queue
    VkDescriptorSetLayout descriptor_layout = VK_NULL_HANDLE; // five storage bindings
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE; // descriptor and push-constant layout
    VkPipeline compute = VK_NULL_HANDLE; // general model operators
    VkPipeline matrix = VK_NULL_HANDLE; // register-tiled quantized matrix multiplication
    VkPipeline vector = VK_NULL_HANDLE; // matrix-vector projections
    VkPipeline im2col = VK_NULL_HANDLE; // convolution patch extraction
    VkPipeline attention = VK_NULL_HANDLE; // F32 tiled flash attention
    VkPipeline attention_large = VK_NULL_HANDLE; // lazily specialized Qwen VAE head size
    VkPipeline quantize = VK_NULL_HANDLE; // Q8_1 activation conversion
    VkPipeline cooperative = VK_NULL_HANDLE; // FP16 cooperative matrix with F32 accumulation
    VkPipeline group_partial = VK_NULL_HANDLE; // shifted Welford chunk statistics
    VkPipeline group_merge = VK_NULL_HANDLE; // merge chunk statistics per group
    VkPipeline group_apply = VK_NULL_HANDLE; // vectorized normalization with affine output
    std::map<unsigned,VkPipeline> mmq; // integer-dot kernels specialized by weight type
    std::map<uint64_t,VkPipeline> subgroup_attention; // head/tile-specialized attention
    uint32_t matrix_tile = VK_MATRIX_TILE; // selected portable register tile
    uint32_t mmq_tile = VK_MATRIX_TILE; // integer-dot cache tile selected for device limits
    uint32_t subgroup_size = 0; // zero retains portable attention on this device
    uint32_t cooperative_subgroup_size = 0; // supported 16x16x16 FP16/F32 cooperative shape
    bool integer_dot = false; // selected packed signed integer-dot feature
    size_t cache_limit = 0; // heap-dependent quantized weight residency budget
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE; // reusable dispatch descriptors
    std::array<VulkanBatch,VK_INFLIGHT_BATCHES> batches; // bounded asynchronous submission ring
    size_t current = 0; // slot selected for recording
    std::deque<size_t> pending; // submissions in queue order
    uint64_t next_serial = 1,completed_serial = 0; // lifetime boundary for buffer reuse
    std::map<unsigned,double> rates; // measured seconds per work unit by kernel family
    double batch_seconds = VULKAN_BATCH_SECONDS; // conservative GPU time target
    uint32_t timestamp_bits = 0; // zero selects conservative host timing fallback
    bool tracing = false; // optional stderr diagnostics
    uint64_t synchronize_from = UINT64_MAX; // optional first dispatch to isolate for fault diagnosis
    uint64_t submissions = 0; // submitted command batches
    uint64_t weight_uploads = 0; // cache misses
    uint64_t weight_uploaded_bytes = 0; // copied quantized weight bytes
    std::multimap<uint64_t,void*> retired; // released buffers held until their last submission completes
    std::multimap<size_t,VulkanBuffer> reusable; // completed allocations available for reuse
    VkCommandPool command_pool = VK_NULL_HANDLE; // reusable command allocation
    VkPhysicalDeviceProperties properties = {}; // device limits and identity
    uint32_t family = 0; // compute queue family
    uint32_t memory_type = 0; // coherent host-visible memory type
    uint32_t memory_heap = 0; // heap containing the selected memory type
    PFN_vkGetPhysicalDeviceMemoryProperties2KHR memory_properties = nullptr; // optional driver budget query
    VkDeviceSize heap_limit = 0; // conservative allocation bound
    VkDeviceSize allocated = 0; // live allocation accounting
    std::map<uintptr_t,VulkanBuffer> buffers; // owned mapped buffers, indexed by address
    std::map<std::string,VulkanCachedWeight> weights; // bounded quantized weight cache
    size_t cache_bytes = 0; // live cache byte total
    uint64_t dispatches = 0; // eviction age
    void* dummy = nullptr; // valid binding for unused inputs

    ~VulkanState()
    {
        if (device) vkDeviceWaitIdle(device);
        for (auto &entry : buffers) {
            vkUnmapMemory(device,entry.second.memory);
            vkDestroyBuffer(device,entry.second.buffer,nullptr);
            vkFreeMemory(device,entry.second.memory,nullptr);
        }
        for (auto &entry : reusable) {
            vkUnmapMemory(device,entry.second.memory);
            vkDestroyBuffer(device,entry.second.buffer,nullptr);
            vkFreeMemory(device,entry.second.memory,nullptr);
        }
        for (auto &batch : batches) {
            if (batch.fence) vkDestroyFence(device,batch.fence,nullptr);
            if (batch.queries) vkDestroyQueryPool(device,batch.queries,nullptr);
        }
        if (command_pool) vkDestroyCommandPool(device,command_pool,nullptr);
        if (descriptor_pool) vkDestroyDescriptorPool(device,descriptor_pool,nullptr);
        if (compute) vkDestroyPipeline(device,compute,nullptr);
        if (matrix) vkDestroyPipeline(device,matrix,nullptr);
        if (vector) vkDestroyPipeline(device,vector,nullptr);
        if (im2col) vkDestroyPipeline(device,im2col,nullptr);
        if (attention) vkDestroyPipeline(device,attention,nullptr);
        if (attention_large) vkDestroyPipeline(device,attention_large,nullptr);
        if (quantize) vkDestroyPipeline(device,quantize,nullptr);
        if (cooperative) vkDestroyPipeline(device,cooperative,nullptr);
        if (group_partial) vkDestroyPipeline(device,group_partial,nullptr);
        if (group_merge) vkDestroyPipeline(device,group_merge,nullptr);
        if (group_apply) vkDestroyPipeline(device,group_apply,nullptr);
        for (auto &entry : mmq) vkDestroyPipeline(device,entry.second,nullptr);
        for (auto &entry : subgroup_attention) vkDestroyPipeline(device,entry.second,nullptr);
        if (pipeline_layout) vkDestroyPipelineLayout(device,pipeline_layout,nullptr);
        if (descriptor_layout) vkDestroyDescriptorSetLayout(device,descriptor_layout,nullptr);
        if (device) vkDestroyDevice(device,nullptr);
        if (instance) vkDestroyInstance(instance,nullptr);
    }

    VkPipeline pipeline(const unsigned char* bytes, size_t size, const std::vector<uint32_t> &ids = {}, const std::vector<uint32_t> &values = {})
    {
        // Embedded arrays have four-byte alignment; the tool appends one terminator byte.
        VkShaderModuleCreateInfo shader_info = {};
        shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        shader_info.codeSize = size - 1;
        shader_info.pCode = reinterpret_cast<const uint32_t*>(bytes);
        VkShaderModule shader;
        if (vkCreateShaderModule(device,&shader_info,nullptr,&shader) != VK_SUCCESS) return VK_NULL_HANDLE;
        VkComputePipelineCreateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        info.layout = pipeline_layout;
        info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        info.stage.module = shader;
        info.stage.pName = "main";
        std::vector<VkSpecializationMapEntry> entries(ids.size());
        for (uint32_t i = 0; i < entries.size(); i++) entries[i] = {ids[i],uint32_t(i * sizeof(uint32_t)),sizeof(uint32_t)};
        VkSpecializationInfo specialization = {};
        specialization.mapEntryCount = entries.size();
        specialization.pMapEntries = entries.data();
        specialization.dataSize = values.size() * sizeof(uint32_t);
        specialization.pData = values.data();
        if (!ids.empty()) info.stage.pSpecializationInfo = &specialization;
        VkPipeline result = VK_NULL_HANDLE;
        VkResult status = vkCreateComputePipelines(device,VK_NULL_HANDLE,1,&info,nullptr,&result);
        vkDestroyShaderModule(device,shader,nullptr);
        if (status != VK_SUCCESS) return VK_NULL_HANDLE;
        return result;
    }

    VkPipeline matrix_pipeline(const unsigned char* bytes, size_t size, unsigned type, uint32_t &tile)
    {
        std::vector<uint32_t> ids = {0,1,2,4,5,6,7,8};
        std::vector<uint32_t> values = {VK_MATRIX_LANES,VK_MATRIX_TILE,VK_MATRIX_TILE,VK_MATRIX_WARP_ROWS,VK_MATRIX_WARP_COLUMNS,VK_MATRIX_ROW_ITERATIONS,VK_MATRIX_REGISTER_ROWS,VK_MATRIX_REGISTER_COLUMNS};
        size_t shared_bytes = type == MODEL_F32? (VK_MATRIX_TILE * 2) * (VK_MATRIX_INNER_TILE / 2 + 1) * sizeof(float) * 2 : (VK_MATRIX_TILE * 2) * VK_MMQ_K_STEPS * (VK_MMQ_PACKED_WORDS + 2) * sizeof(uint32_t);
        if (properties.limits.maxComputeWorkGroupInvocations < VK_MATRIX_LANES || properties.limits.maxComputeWorkGroupSize[0] < VK_MATRIX_LANES || properties.limits.maxComputeSharedMemorySize < shared_bytes) {
            tile = VK_MATRIX_SMALL_TILE;
            values = {VK_COMPUTE_LANES,VK_MATRIX_SMALL_TILE,VK_MATRIX_SMALL_TILE,VK_MATRIX_SMALL_TILE / 2,VK_MATRIX_WARP_COLUMNS,VK_MATRIX_ROW_ITERATIONS,VK_MATRIX_REGISTER_ROWS / 2,VK_MATRIX_REGISTER_COLUMNS};
        }
        if (type != MODEL_F32) {
            ids.push_back(VK_MMQ_SPECIALIZATION_TYPE);
            values.push_back(type);
        }
        return pipeline(bytes,size,ids,values);
    }

    VkPipeline attention_pipeline(const VulkanPush &push)
    {
        uint32_t dimension = push[2] / push[3],tile = push[VK_PUSH_ATTENTION_TILE];
        uint64_t key = (uint64_t(dimension) << 32) | tile;
        auto existing = subgroup_attention.find(key);
        if (existing != subgroup_attention.end()) return existing->second;
        uint32_t lowest_bit = dimension & (~dimension + 1);
        uint32_t split = std::min(std::min(subgroup_size,uint32_t(VK_FAST_ATTENTION_D_SPLIT)),lowest_bit / VK_ATTENTION_VECTOR);
        std::vector<uint32_t> ids = {0,1,3,6,8};
        std::vector<uint32_t> values = {subgroup_size * VK_FAST_ATTENTION_ROW_SPLIT,tile,dimension,split,subgroup_size};
        VkPipeline result = pipeline(pixelina_attention_subgroup_shader,sizeof(pixelina_attention_subgroup_shader),ids,values);
        subgroup_attention.emplace(key,result);
        return result;
    }

    bool initialize()
    {
        tracing = configured_options.tracing;
        batch_seconds = configured_options.batch_seconds;
        synchronize_from = configured_options.synchronize_from;
        VkApplicationInfo application = {};
        application.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        application.pApplicationName = PIXELINA_NAME;
        application.apiVersion = VK_API_VERSION_1_0;
        auto enumerate_version = reinterpret_cast<PFN_vkEnumerateInstanceVersion>(vkGetInstanceProcAddr(VK_NULL_HANDLE,"vkEnumerateInstanceVersion"));
        if (enumerate_version) {
            uint32_t version = VK_API_VERSION_1_0;
            if (enumerate_version(&version) != VK_SUCCESS) return false;
            application.apiVersion = std::min(version,uint32_t(VK_API_VERSION_1_3));
        }
        VkInstanceCreateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        info.pApplicationInfo = &application;
        const char* properties_extension = VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME;
        if (tracing) {
            uint32_t extension_count = 0;
            if (vkEnumerateInstanceExtensionProperties(nullptr,&extension_count,nullptr) != VK_SUCCESS) return false;
            std::vector<VkExtensionProperties> extensions(extension_count);
            if (vkEnumerateInstanceExtensionProperties(nullptr,&extension_count,extensions.data()) != VK_SUCCESS) return false;
            for (const auto &extension : extensions) {
                if (std::strcmp(extension.extensionName,properties_extension) != 0) continue;
                info.enabledExtensionCount = 1;
                info.ppEnabledExtensionNames = &properties_extension;
                break;
            }
        }
        if (vkCreateInstance(&info,nullptr,&instance) != VK_SUCCESS) return false;
        if (info.enabledExtensionCount) memory_properties = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties2KHR>(vkGetInstanceProcAddr(instance,"vkGetPhysicalDeviceMemoryProperties2KHR"));
        uint32_t count = 0;
        if (vkEnumeratePhysicalDevices(instance,&count,nullptr) != VK_SUCCESS) return false;
        if (!count) return false;
        std::vector<VkPhysicalDevice> devices(count);
        if (vkEnumeratePhysicalDevices(instance,&count,devices.data()) != VK_SUCCESS) return false;
        int best = -1;
        for (VkPhysicalDevice candidate : devices) {
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(candidate,&props);
            uint32_t families = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(candidate,&families,nullptr);
            std::vector<VkQueueFamilyProperties> queues(families);
            vkGetPhysicalDeviceQueueFamilyProperties(candidate,&families,queues.data());
            for (uint32_t i = 0; i < families; i++) {
                if (!(queues[i].queueFlags & VK_QUEUE_COMPUTE_BIT) || !queues[i].queueCount) continue;
                if (props.limits.maxComputeWorkGroupInvocations < VK_COMPUTE_LANES || props.limits.maxComputeWorkGroupSize[0] < VK_COMPUTE_LANES || props.limits.maxPushConstantsSize < sizeof(VulkanPush)) continue;
                int rank = 0;
                if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) rank = VULKAN_DEVICE_RANK_DISCRETE;
                else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) rank = VULKAN_DEVICE_RANK_INTEGRATED;
                else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU) rank = VULKAN_DEVICE_RANK_VIRTUAL;
                else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) rank = VULKAN_DEVICE_RANK_CPU;
                if (rank > best) {
                    physical = candidate;
                    properties = props;
                    family = i;
                    timestamp_bits = queues[i].timestampValidBits;
                    best = rank;
                }
                break;
            }
        }
        if (!physical) return false;
        float priority = 1;
        VkDeviceQueueCreateInfo queue_info = {};
        queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_info.queueFamilyIndex = family;
        queue_info.queueCount = 1;
        queue_info.pQueuePriorities = &priority;
        VkDeviceCreateInfo device_info = {};
        device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        device_info.queueCreateInfoCount = 1;
        device_info.pQueueCreateInfos = &queue_info;
        std::vector<const char*> device_extensions;
        uint32_t extension_count = 0;
        if (vkEnumerateDeviceExtensionProperties(physical,nullptr,&extension_count,nullptr) != VK_SUCCESS) return false;
        std::vector<VkExtensionProperties> extensions(extension_count);
        if (vkEnumerateDeviceExtensionProperties(physical,nullptr,&extension_count,extensions.data()) != VK_SUCCESS) return false;
        auto has_extension = [&](const char* name) {
            return std::any_of(extensions.begin(),extensions.end(),[&](const VkExtensionProperties &extension) { return std::strcmp(extension.extensionName,name) == 0; });
        };
        const char* mode = configured_options.kernels.c_str();
        bool portable = mode && std::strcmp(mode,VULKAN_KERNEL_PORTABLE) == 0;
        bool force_optimized = mode && std::strcmp(mode,VULKAN_KERNEL_OPTIMIZED) == 0;
        VkPhysicalDeviceShaderIntegerDotProductFeatures dot_features = {};
        dot_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES;
        VkPhysicalDeviceShaderFloat16Int8Features half_features = {};
        half_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
        VkPhysicalDeviceVulkanMemoryModelFeatures memory_features = {};
        memory_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES;
        VkPhysicalDeviceCooperativeMatrixFeaturesKHR coop_features = {};
        coop_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
        if (!portable && application.apiVersion >= VK_API_VERSION_1_1 && properties.apiVersion >= VK_API_VERSION_1_1) {
            VkPhysicalDeviceSubgroupProperties subgroup = {};
            subgroup.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
            VkPhysicalDeviceShaderIntegerDotProductProperties dot_properties = {};
            dot_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES;
            bool dot_extension = has_extension(VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME);
            bool dot_supported = dot_extension || (application.apiVersion >= VK_API_VERSION_1_3 && properties.apiVersion >= VK_API_VERSION_1_3);
            subgroup.pNext = dot_supported? &dot_properties : nullptr;
            VkPhysicalDeviceProperties2 props = {};
            props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
            props.pNext = &subgroup;
            vkGetPhysicalDeviceProperties2(physical,&props);
            if ((subgroup.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) && (subgroup.supportedOperations & VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT) && subgroup.subgroupSize >= VK_FAST_ATTENTION_D_SPLIT && subgroup.subgroupSize * VK_FAST_ATTENTION_ROW_SPLIT <= properties.limits.maxComputeWorkGroupInvocations && subgroup.subgroupSize * VK_FAST_ATTENTION_ROW_SPLIT <= properties.limits.maxComputeWorkGroupSize[0]) subgroup_size = subgroup.subgroupSize;
            if (dot_supported) {
                VkPhysicalDeviceFeatures2 features = {};
                features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
                features.pNext = &dot_features;
                vkGetPhysicalDeviceFeatures2(physical,&features);
                integer_dot = dot_features.shaderIntegerDotProduct && (dot_properties.integerDotProduct4x8BitPackedSignedAccelerated || force_optimized);
                if (integer_dot) {
                    if (dot_extension) device_extensions.push_back(VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME);
                    dot_features.pNext = nullptr;
                    device_info.pNext = &dot_features;
                }
            }
            bool coop_supported = has_extension(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME) && application.apiVersion >= VK_API_VERSION_1_2 && properties.apiVersion >= VK_API_VERSION_1_2;
            if (coop_supported && subgroup.subgroupSize >= VK_COOP_MATRIX_SHAPE && subgroup.subgroupSize * VK_COOP_MATRIX_SUBGROUPS <= properties.limits.maxComputeWorkGroupInvocations && subgroup.subgroupSize * VK_COOP_MATRIX_SUBGROUPS <= properties.limits.maxComputeWorkGroupSize[0]) {
                VkPhysicalDeviceFeatures2 features = {};
                features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
                features.pNext = &half_features;
                half_features.pNext = &memory_features;
                memory_features.pNext = &coop_features;
                vkGetPhysicalDeviceFeatures2(physical,&features);
                auto get_shapes = reinterpret_cast<PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR>(vkGetInstanceProcAddr(instance,"vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"));
                VkPhysicalDeviceCooperativeMatrixPropertiesKHR coop_properties = {};
                coop_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
                props.pNext = &coop_properties;
                vkGetPhysicalDeviceProperties2(physical,&props);
                if (half_features.shaderFloat16 && memory_features.vulkanMemoryModel && coop_features.cooperativeMatrix && get_shapes && (coop_properties.cooperativeMatrixSupportedStages & VK_SHADER_STAGE_COMPUTE_BIT)) {
                    uint32_t shape_count = 0;
                    if (get_shapes(physical,&shape_count,nullptr) != VK_SUCCESS) return false;
                    std::vector<VkCooperativeMatrixPropertiesKHR> shapes(shape_count);
                    for (auto &shape : shapes) shape.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
                    if (get_shapes(physical,&shape_count,shapes.data()) != VK_SUCCESS) return false;
                    for (const auto &shape : shapes) {
                        if (shape.scope != VK_SCOPE_SUBGROUP_KHR || shape.MSize != VK_COOP_MATRIX_SHAPE || shape.NSize != VK_COOP_MATRIX_SHAPE || shape.KSize != VK_COOP_MATRIX_SHAPE || shape.AType != VK_COMPONENT_TYPE_FLOAT16_KHR || shape.BType != VK_COMPONENT_TYPE_FLOAT16_KHR || shape.CType != VK_COMPONENT_TYPE_FLOAT32_KHR || shape.ResultType != VK_COMPONENT_TYPE_FLOAT32_KHR) continue;
                        cooperative_subgroup_size = subgroup.subgroupSize;
                    }
                }
                if (cooperative_subgroup_size) {
                    device_extensions.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
                    // Enable only features used by the shader, preserving the integer-dot chain.
                    half_features.shaderInt8 = VK_FALSE;
                    memory_features.vulkanMemoryModelDeviceScope = VK_FALSE;
                    memory_features.vulkanMemoryModelAvailabilityVisibilityChains = VK_FALSE;
                    coop_features.cooperativeMatrixRobustBufferAccess = VK_FALSE;
                    coop_features.pNext = const_cast<void*>(device_info.pNext);
                    device_info.pNext = &half_features;
                }
            }
        }
        const char* budget_extension = VK_EXT_MEMORY_BUDGET_EXTENSION_NAME;
        if (memory_properties) {
            if (has_extension(budget_extension)) device_extensions.push_back(budget_extension);
            else memory_properties = nullptr;
        }
        device_info.enabledExtensionCount = device_extensions.size();
        device_info.ppEnabledExtensionNames = device_extensions.data();
        if (vkCreateDevice(physical,&device_info,nullptr,&device) != VK_SUCCESS) return false;
        vkGetDeviceQueue(device,family,0,&queue);
        VkPhysicalDeviceMemoryProperties memory;
        vkGetPhysicalDeviceMemoryProperties(physical,&memory);
        bool found = false;
        for (uint32_t i = 0; i < memory.memoryTypeCount; i++) {
            VkMemoryPropertyFlags needed = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            if ((memory.memoryTypes[i].propertyFlags & needed) == needed) {
                memory_type = i;
                memory_heap = memory.memoryTypes[i].heapIndex;
                heap_limit = VkDeviceSize(memory.memoryHeaps[memory.memoryTypes[i].heapIndex].size * VULKAN_HEAP_FRACTION);
                found = true;
                if (memory.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) break;
            }
        }
        if (!found) return false;
        cache_limit = size_t(heap_limit - std::min(VkDeviceSize(VULKAN_CACHE_RESERVE_BYTES),heap_limit / 4));
        std::array<VkDescriptorSetLayoutBinding,VK_BUFFER_BINDINGS> bindings = {};
        for (uint32_t i = 0; i < VK_BUFFER_BINDINGS; i++) {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo layout_info = {};
        layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layout_info.bindingCount = VK_BUFFER_BINDINGS;
        layout_info.pBindings = bindings.data();
        if (vkCreateDescriptorSetLayout(device,&layout_info,nullptr,&descriptor_layout) != VK_SUCCESS) return false;
        VkPushConstantRange constants = {};
        constants.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        constants.size = sizeof(VulkanPush);
        VkPipelineLayoutCreateInfo pipeline_info = {};
        pipeline_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipeline_info.setLayoutCount = 1;
        pipeline_info.pSetLayouts = &descriptor_layout;
        pipeline_info.pushConstantRangeCount = 1;
        pipeline_info.pPushConstantRanges = &constants;
        if (vkCreatePipelineLayout(device,&pipeline_info,nullptr,&pipeline_layout) != VK_SUCCESS) return false;
        compute = pipeline(pixelina_compute_shader,sizeof(pixelina_compute_shader));
        matrix = matrix_pipeline(pixelina_matmul_shader,sizeof(pixelina_matmul_shader),MODEL_F32,matrix_tile);
        vector = pipeline(pixelina_matvec_shader,sizeof(pixelina_matvec_shader));
        im2col = pipeline(pixelina_im2col_shader,sizeof(pixelina_im2col_shader));
        attention = pipeline(pixelina_attention_shader,sizeof(pixelina_attention_shader));
        group_partial = pipeline(pixelina_groupnorm_partial_shader,sizeof(pixelina_groupnorm_partial_shader));
        group_merge = pipeline(pixelina_groupnorm_merge_shader,sizeof(pixelina_groupnorm_merge_shader));
        group_apply = pipeline(pixelina_groupnorm_apply_shader,sizeof(pixelina_groupnorm_apply_shader));
        if (!compute || !matrix || !vector || !im2col || !attention || !group_partial || !group_merge || !group_apply) return false;
        if (integer_dot) {
            quantize = pipeline(pixelina_quantize_shader,sizeof(pixelina_quantize_shader));
            if (!quantize) return false;
            for (unsigned type : {MODEL_Q8_0,MODEL_Q5_K,MODEL_Q6_K}) {
                VkPipeline result = matrix_pipeline(pixelina_matmul_mmq_shader,sizeof(pixelina_matmul_mmq_shader),type,mmq_tile);
                if (!result) return false;
                mmq.emplace(type,result);
            }
        }
        if (cooperative_subgroup_size) {
            std::vector<uint32_t> ids = {0,VK_COOP_MATRIX_SUBGROUP_SPECIALIZATION,VK_COOP_MATRIX_PAD_SPECIALIZATION};
            std::vector<uint32_t> values = {cooperative_subgroup_size * VK_COOP_MATRIX_SUBGROUPS,cooperative_subgroup_size,VK_COOP_MATRIX_OPTIMIZED_PAD};
            cooperative = pipeline(pixelina_matmul_coop_shader,sizeof(pixelina_matmul_coop_shader),ids,values);
        }
        if (cooperative_subgroup_size && !cooperative) return false;
        VkDescriptorPoolSize pool_size = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,VK_BUFFER_BINDINGS * VK_DISPATCH_BATCH * VK_INFLIGHT_BATCHES};
        VkDescriptorPoolCreateInfo pool_info = {};
        pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_info.maxSets = VK_DISPATCH_BATCH * VK_INFLIGHT_BATCHES;
        pool_info.poolSizeCount = 1;
        pool_info.pPoolSizes = &pool_size;
        if (vkCreateDescriptorPool(device,&pool_info,nullptr,&descriptor_pool) != VK_SUCCESS) return false;
        VkCommandPoolCreateInfo command_info = {};
        command_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        command_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        command_info.queueFamilyIndex = family;
        if (vkCreateCommandPool(device,&command_info,nullptr,&command_pool) != VK_SUCCESS) return false;
        dummy = allocate(VULKAN_DEFAULT_BUFFER_BYTES);
        std::memset(dummy,0,VULKAN_DEFAULT_BUFFER_BYTES);
        for (auto &batch : batches) {
            VkDescriptorSetAllocateInfo set_info = {};
            set_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            set_info.descriptorPool = descriptor_pool;
            std::array<VkDescriptorSetLayout,VK_DISPATCH_BATCH> layouts;
            layouts.fill(descriptor_layout);
            set_info.descriptorSetCount = VK_DISPATCH_BATCH;
            set_info.pSetLayouts = layouts.data();
            if (vkAllocateDescriptorSets(device,&set_info,batch.descriptors.data()) != VK_SUCCESS) return false;
            VkCommandBufferAllocateInfo allocation = {};
            allocation.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            allocation.commandPool = command_pool;
            allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocation.commandBufferCount = 1;
            if (vkAllocateCommandBuffers(device,&allocation,&batch.command) != VK_SUCCESS) return false;
            VkFenceCreateInfo fence_info = {};
            fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            if (vkCreateFence(device,&fence_info,nullptr,&batch.fence) != VK_SUCCESS) return false;
            if (timestamp_bits) {
                VkQueryPoolCreateInfo query_info = {};
                query_info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
                query_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
                query_info.queryCount = VULKAN_QUERY_PAIR * VK_DISPATCH_BATCH;
                if (vkCreateQueryPool(device,&query_info,nullptr,&batch.queries) != VK_SUCCESS) return false;
            }
            batch.marker = static_cast<uint32_t*>(allocate(sizeof(uint32_t)));
            batch.marker_buffer = buffers.find(reinterpret_cast<uintptr_t>(batch.marker))->second.buffer;
            *batch.marker = 0;
        }
        if (tracing) std::fprintf(stderr,"Vulkan device=%s memory_type=%u heap_limit=%llu cache_limit=%zu matrix_tile=%u batch_seconds=%.3f inflight_slots=%u timestamp_bits=%u\n",properties.deviceName,memory_type,static_cast<unsigned long long>(heap_limit),cache_limit,matrix_tile,batch_seconds,VK_INFLIGHT_BATCHES,timestamp_bits);
        if (tracing) std::fprintf(stderr,"Vulkan kernels integer_dot=%u mmq_tile=%u attention_subgroup_size=%u cooperative_subgroup_size=%u mode=%s\n",integer_dot? 1u : 0u,mmq_tile,subgroup_size,cooperative_subgroup_size,mode? mode : VULKAN_KERNEL_AUTO);
        return true;
    }

    void recycle(void* pointer)
    {
        auto entry = buffers.find(reinterpret_cast<uintptr_t>(pointer));
        if (entry == buffers.end()) return;
        entry->second.last_use = 0;
        reusable.emplace(entry->second.bytes,entry->second);
        buffers.erase(entry);
    }

    void release(void* pointer)
    {
        auto entry = buffers.find(reinterpret_cast<uintptr_t>(pointer));
        if (entry == buffers.end()) return;
        if (entry->second.last_use > completed_serial) retired.emplace(entry->second.last_use,pointer);
        else recycle(pointer);
    }

    void trim()
    {
        for (auto &entry : reusable) {
            vkUnmapMemory(device,entry.second.memory);
            vkDestroyBuffer(device,entry.second.buffer,nullptr);
            vkFreeMemory(device,entry.second.memory,nullptr);
            allocated -= entry.second.allocated;
        }
        reusable.clear();
    }

    VkResult complete_front(bool wait)
    {
        if (pending.empty()) return VK_NOT_READY;
        VulkanBatch &batch = batches[pending.front()];
        VkResult status = wait? vkWaitForFences(device,1,&batch.fence,VK_TRUE,UINT64_MAX) : vkGetFenceStatus(device,batch.fence);
        if (status != VK_SUCCESS) return status;
        // A fence may report success after device loss; require an actual GPU write too.
        if (*batch.marker != batch.expected) return VK_ERROR_DEVICE_LOST;
        double host_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - batch.submitted).count();
        std::array<uint64_t,VULKAN_QUERY_PAIR * VK_DISPATCH_BATCH> timestamps = {};
        if (batch.queries) {
            status = vkGetQueryPoolResults(device,batch.queries,0,batch.recorded * VULKAN_QUERY_PAIR,batch.recorded * VULKAN_QUERY_PAIR * sizeof(uint64_t),timestamps.data(),sizeof(uint64_t),VK_QUERY_RESULT_64_BIT);
            if (status != VK_SUCCESS) return status;
        }
        uint64_t mask = timestamp_bits == sizeof(uint64_t) * 8? UINT64_MAX : (uint64_t(1) << timestamp_bits) - 1;
        double gpu_seconds = 0;
        for (uint32_t i = 0; i < batch.recorded; i++) {
            double seconds = batch.queries? double((timestamps[i * VULKAN_QUERY_PAIR + 1] - timestamps[i * VULKAN_QUERY_PAIR]) & mask) * properties.limits.timestampPeriod / VULKAN_NANOSECONDS : host_seconds;
            gpu_seconds += seconds;
            const VulkanSample &sample = batch.samples[i];
            double observed = std::max(seconds,VULKAN_MIN_SAMPLE_SECONDS) / sample.work;
            auto rate = rates.find(sample.kind);
            if (rate == rates.end()) rates.emplace(sample.kind,observed);
            else rate->second = std::max(observed,rate->second * VULKAN_RATE_DECAY);
        }
        if (tracing) std::fprintf(stderr,"Vulkan batch=%llu dispatches=%u gpu_seconds=%.6f host_seconds=%.6f verified=yes pending=%zu last=%s\n",static_cast<unsigned long long>(batch.serial),batch.recorded,batch.queries? gpu_seconds : 0,host_seconds,pending.size(),batch.operation.c_str());
        completed_serial = batch.serial;
        batch.pending = false;
        batch.recorded = 0;
        pending.pop_front();
        auto entry = retired.begin();
        while (entry != retired.end() && entry->first <= completed_serial) {
            recycle(entry->second);
            entry = retired.erase(entry);
        }
        return VK_SUCCESS;
    }

    VulkanBatch* begin_batch()
    {
        VkResult status;
        while ((status = complete_front(false)) == VK_SUCCESS) continue;
        if (status != VK_NOT_READY) return nullptr;
        while (batches[current].pending) {
            if (complete_front(true) != VK_SUCCESS) return nullptr;
        }
        VulkanBatch &batch = batches[current];
        if (!batch.recorded) {
            batch.serial = next_serial++;
            batch.predicted = 0;
            batch.first = dispatches;
            batch.expected = uint32_t((batch.serial - 1) % UINT32_MAX) + 1;
            *batch.marker = 0;
            if (vkResetCommandBuffer(batch.command,0) != VK_SUCCESS) return nullptr;
            VkCommandBufferBeginInfo begin = {};
            begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            if (vkBeginCommandBuffer(batch.command,&begin) != VK_SUCCESS) return nullptr;
            if (batch.queries) vkCmdResetQueryPool(batch.command,batch.queries,0,VULKAN_QUERY_PAIR * VK_DISPATCH_BATCH);
        }
        return &batch;
    }

    VkResult submit_batch()
    {
        VulkanBatch &batch = batches[current];
        if (batch.pending || !batch.recorded) return VK_SUCCESS;
        VkMemoryBarrier after = {};
        after.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        after.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        after.dstAccessMask = VK_ACCESS_HOST_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(batch.command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,0,1,&after,0,nullptr,0,nullptr);
        vkCmdFillBuffer(batch.command,batch.marker_buffer,0,sizeof(uint32_t),batch.expected);
        VkMemoryBarrier marker = {};
        marker.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        marker.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        marker.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(batch.command,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,1,&marker,0,nullptr,0,nullptr);
        VkResult status = vkEndCommandBuffer(batch.command);
        if (status != VK_SUCCESS) return status;
        status = vkResetFences(device,1,&batch.fence);
        if (status != VK_SUCCESS) return status;
        VkSubmitInfo submit = {};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &batch.command;
        if (tracing) trace_memory();
        if (tracing) std::fprintf(stderr,"Vulkan submitting batch=%llu first=%llu last=%llu dispatches=%u predicted_seconds=%.6f pending=%zu allocated_bytes=%llu cached_weight_bytes=%zu last_operation=%s\n",static_cast<unsigned long long>(batch.serial),static_cast<unsigned long long>(batch.first),static_cast<unsigned long long>(batch.last),batch.recorded,batch.predicted,pending.size(),static_cast<unsigned long long>(allocated),cache_bytes,batch.operation.c_str());
        batch.submitted = std::chrono::steady_clock::now();
        status = vkQueueSubmit(queue,1,&submit,batch.fence);
        if (status != VK_SUCCESS) return status;
        batch.pending = true;
        pending.push_back(current);
        submissions++;
        current = (current + 1) % VK_INFLIGHT_BATCHES;
        return VK_SUCCESS;
    }

    bool synchronize()
    {
        if (submit_batch() != VK_SUCCESS) return false;
        VkResult status;
        while ((status = complete_front(true)) == VK_SUCCESS) continue;
        return status == VK_NOT_READY;
    }

    void trace_memory()
    {
        VkDeviceSize reusable_bytes = 0,retired_bytes = 0;
        for (const auto &entry : reusable) reusable_bytes += entry.second.allocated;
        for (const auto &released : retired) {
            auto entry = buffers.find(reinterpret_cast<uintptr_t>(released.second));
            if (entry != buffers.end()) retired_bytes += entry->second.allocated;
        }
        std::fprintf(stderr,"Vulkan memory batch=%llu allocations=%zu reusable_bytes=%llu retired_bytes=%llu budget_available=%u\n",static_cast<unsigned long long>(submissions),buffers.size() + reusable.size(),static_cast<unsigned long long>(reusable_bytes),static_cast<unsigned long long>(retired_bytes),memory_properties? 1u : 0u);
        if (!memory_properties) return;
        VkPhysicalDeviceMemoryBudgetPropertiesEXT budget = {};
        budget.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
        VkPhysicalDeviceMemoryProperties2KHR memory = {};
        memory.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2_KHR;
        memory.pNext = &budget;
        memory_properties(physical,&memory);
        for (uint32_t heap = 0; heap < memory.memoryProperties.memoryHeapCount; heap++) {
            std::fprintf(stderr,"Vulkan heap batch=%llu heap=%u selected=%u size_bytes=%llu budget_bytes=%llu usage_bytes=%llu\n",static_cast<unsigned long long>(submissions),heap,heap == memory_heap? 1u : 0u,static_cast<unsigned long long>(memory.memoryProperties.memoryHeaps[heap].size),static_cast<unsigned long long>(budget.heapBudget[heap]),static_cast<unsigned long long>(budget.heapUsage[heap]));
        }
    }

    void evict()
    {
        if (weights.empty()) return;
        auto older = [](const std::pair<const std::string,VulkanCachedWeight> &a, const std::pair<const std::string,VulkanCachedWeight> &b) {
            return a.second.used < b.second.used;
        };
        auto oldest = std::min_element(weights.begin(),weights.end(),older);
        cache_bytes -= oldest->second.bytes;
        release(oldest->second.pointer);
        weights.erase(oldest);
    }

    void* allocate(size_t bytes)
    {
        VkResult completion;
        while ((completion = complete_front(false)) == VK_SUCCESS) continue;
        if (completion != VK_NOT_READY) return nullptr;
        if (!bytes) bytes = VULKAN_DEFAULT_BUFFER_BYTES;
        if (bytes > properties.limits.maxStorageBufferRange) return nullptr;
        bytes = (bytes + 3) & ~size_t(3);
        auto free = reusable.lower_bound(bytes);
        if (free != reusable.end() && free->first <= bytes * 2) {
            VulkanBuffer buffer = free->second;
            buffers.emplace(reinterpret_cast<uintptr_t>(buffer.pointer),buffer);
            reusable.erase(free);
            return buffer.pointer;
        }
        VulkanBuffer buffer;
        buffer.bytes = bytes;
        VkBufferCreateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        info.size = bytes;
        info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateBuffer(device,&info,nullptr,&buffer.buffer) != VK_SUCCESS) return nullptr;
        VkMemoryRequirements requirements;
        vkGetBufferMemoryRequirements(device,buffer.buffer,&requirements);
        if (!(requirements.memoryTypeBits & (1u << memory_type))) return nullptr;
        if (allocated + requirements.size > heap_limit || buffers.size() + reusable.size() >= properties.limits.maxMemoryAllocationCount) {
            if (!synchronize()) return nullptr;
            trim();
        }
        while (!weights.empty() && (allocated + requirements.size > heap_limit || buffers.size() + reusable.size() >= properties.limits.maxMemoryAllocationCount)) {
            evict();
            if (!synchronize()) return nullptr;
            trim();
        }
        if (allocated + requirements.size > heap_limit || buffers.size() + reusable.size() >= properties.limits.maxMemoryAllocationCount) {
            return nullptr;
        }
        VkMemoryAllocateInfo memory = {};
        memory.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        memory.allocationSize = requirements.size;
        memory.memoryTypeIndex = memory_type;
        VkResult status = vkAllocateMemory(device,&memory,nullptr,&buffer.memory);
        if (status == VK_SUCCESS) status = vkBindBufferMemory(device,buffer.buffer,buffer.memory,0);
        if (status == VK_SUCCESS) status = vkMapMemory(device,buffer.memory,0,VK_WHOLE_SIZE,0,&buffer.pointer);
        if (status != VK_SUCCESS) return nullptr;
        buffer.allocated = requirements.size;
        allocated += requirements.size;
        buffers.emplace(reinterpret_cast<uintptr_t>(buffer.pointer),buffer);
        return buffer.pointer;
    }

    void* upload(const void* source, size_t bytes)
    {
        void* pointer = allocate(bytes);
        if (bytes) std::memcpy(pointer,source,bytes);
        return pointer;
    }

    void* weight(const PixModelFile &file, const PixWeight &tensor)
    {
        std::string key = file.filename + ":" + tensor.name + ":" + std::to_string(tensor.offset) + ":" + std::to_string(tensor.bytes);
        auto existing = weights.find(key);
        if (existing != weights.end()) {
            existing->second.used = dispatches;
            return existing->second.pointer;
        }
        if (tensor.bytes > cache_limit) return nullptr;
        while (!weights.empty() && cache_bytes + tensor.bytes > cache_limit) evict();
        void* pointer = upload(file.data + tensor.offset,size_t(tensor.bytes));
        weights.emplace(key,VulkanCachedWeight{pointer,size_t(tensor.bytes),dispatches});
        cache_bytes += size_t(tensor.bytes);
        weight_uploads++;
        weight_uploaded_bytes += tensor.bytes;
        return pointer;
    }

    VkDescriptorBufferInfo binding(const void* pointer, uint64_t serial)
    {
        uintptr_t address = reinterpret_cast<uintptr_t>(pointer? pointer : dummy);
        auto entry = buffers.upper_bound(address);
        entry--;
        size_t offset = address - entry->first;
        entry->second.last_use = std::max(entry->second.last_use,serial);
        return {entry->second.buffer,offset,entry->second.bytes - offset};
    }

    unsigned cost_kind(unsigned pipeline_kind, const VulkanPush &push)
    {
        switch (pipeline_kind) {
        case VULKAN_PIPELINE_MATRIX:
            return (push[2] == 1? VULKAN_COST_VECTOR : VULKAN_COST_MATRIX) + push[4];
        case VULKAN_PIPELINE_MMQ:
            return VULKAN_COST_MMQ + push[4];
        case VULKAN_PIPELINE_QUANTIZE:
            return VULKAN_COST_QUANTIZE;
        case VULKAN_PIPELINE_COOPERATIVE:
            return VULKAN_COST_COOPERATIVE + push[4];
        case VULKAN_PIPELINE_ATTENTION_SUBGROUP:
            return VULKAN_COST_ATTENTION_SUBGROUP + (push[2] / push[3]) * VK_FAST_ATTENTION_ROWS + push[VK_PUSH_ATTENTION_TILE];
        case VULKAN_PIPELINE_GROUP_PARTIAL:
            return VULKAN_COST_GROUP_PARTIAL;
        case VULKAN_PIPELINE_GROUP_MERGE:
            return VULKAN_COST_GROUP_MERGE;
        case VULKAN_PIPELINE_GROUP_APPLY:
            return VULKAN_COST_GROUP_APPLY;
        case VULKAN_PIPELINE_IM2COL:
            return VULKAN_COST_IM2COL;
        case VULKAN_PIPELINE_ATTENTION:
            return VULKAN_COST_ATTENTION;
        case VULKAN_PIPELINE_ATTENTION_LARGE:
            return VULKAN_COST_ATTENTION_LARGE;
        default:
            break;
        }
        if (push[0] == VK_OP_DECODE) return VULKAN_COST_DECODE + push[3];
        return push[0];
    }

    double work_units(unsigned pipeline_kind, const VulkanPush &push, uint32_t base, uint32_t groups)
    {
        double work = VK_COMPUTE_LANES;
        if (pipeline_kind == VULKAN_PIPELINE_MATRIX || pipeline_kind == VULKAN_PIPELINE_MMQ || pipeline_kind == VULKAN_PIPELINE_COOPERATIVE) {
            uint32_t tile = pipeline_kind == VULKAN_PIPELINE_MMQ? mmq_tile : (pipeline_kind == VULKAN_PIPELINE_COOPERATIVE? VK_MATRIX_TILE : matrix_tile);
            work = (push[2] == 1? double(VK_MATVEC_ROWS) : double(tile) * tile) * push[3] * 2;
        } else if (pipeline_kind == VULKAN_PIPELINE_QUANTIZE) {
            work = (VK_COMPUTE_LANES / VK_MMQ_PACKED_WORDS) * MODEL_Q8_BLOCK;
        } else if (pipeline_kind == VULKAN_PIPELINE_IM2COL) {
            work = double(push[5]) * push[6] * push[6];
        } else if (pipeline_kind == VULKAN_PIPELINE_ATTENTION || pipeline_kind == VULKAN_PIPELINE_ATTENTION_LARGE || pipeline_kind == VULKAN_PIPELINE_ATTENTION_SUBGROUP) {
            uint32_t tile = pipeline_kind == VULKAN_PIPELINE_ATTENTION_SUBGROUP? push[VK_PUSH_ATTENTION_TILE] : VK_ATTENTION_QUERY_TILE;
            uint64_t last_query = (uint64_t(base) + groups - 1) / push[3] * tile;
            double keys = push[6]? double(std::min(uint64_t(push[5]),last_query + tile)) : push[5];
            work = tile * keys * (push[2] / push[3]) * VULKAN_ATTENTION_WORK_PER_ELEMENT;
        } else if (pipeline_kind == VULKAN_PIPELINE_GROUP_PARTIAL) {
            work = VK_GROUP_CHUNK;
        } else if (pipeline_kind == VULKAN_PIPELINE_GROUP_MERGE) {
            work = push[6];
        } else if (pipeline_kind == VULKAN_PIPELINE_GROUP_APPLY) {
            work = VK_COMPUTE_LANES * VK_GROUP_VECTOR;
        } else if (push[0] == VK_OP_NORM) {
            work = double(push[2]) * VULKAN_NORMALIZATION_PASSES + VK_COMPUTE_LANES;
        } else if (push[0] == VK_OP_GROUP) {
            work = double(push[1]) * (push[2] / push[3]) * VULKAN_NORMALIZATION_PASSES + VK_COMPUTE_LANES;
        }
        return std::max(1.0,work * groups);
    }

    bool dispatch(unsigned pipeline_kind, const VulkanPush &push, const std::array<const void*,VK_BUFFER_BINDINGS> &pointers, uint32_t groups_x, uint32_t groups_y = 1)
    {
        uint64_t total = uint64_t(groups_x) * groups_y;
        if (!total || total > UINT32_MAX) return false;
        bool isolated = dispatches >= synchronize_from;
        unsigned kind = cost_kind(pipeline_kind,push);
        bool calibrating = rates.find(kind) == rates.end();
        if ((isolated || calibrating) && !synchronize()) return false;
        uint32_t base = 0;
        while (base < total) {
            VulkanBatch* recording = begin_batch();
            if (!recording) return false;
            VulkanBatch &batch = *recording;
            double rate = calibrating? 0 : rates.find(kind)->second;
            // Reevaluate queued work when a completed submission updates a timing model.
            batch.predicted = 0;
            for (uint32_t i = 0; i < batch.recorded; i++) batch.predicted += rates.find(batch.samples[i].kind)->second * batch.samples[i].work * VULKAN_PREDICTION_MARGIN;
            double available = batch_seconds - batch.predicted;
            if (batch.recorded && rate * work_units(pipeline_kind,push,base,1) * VULKAN_PREDICTION_MARGIN > available) {
                if (submit_batch() != VK_SUCCESS) return false;
                continue;
            }
            uint32_t count = 1;
            if (!calibrating) {
                uint32_t lower = 1,upper = uint32_t(std::min(total - base,uint64_t(properties.limits.maxComputeWorkGroupCount[0])));
                while (lower < upper) {
                    uint32_t middle = lower + (upper - lower + 1) / 2;
                    double predicted = rate * work_units(pipeline_kind,push,base,middle) * VULKAN_PREDICTION_MARGIN;
                    if (predicted <= available) lower = middle;
                    else upper = middle - 1;
                }
                count = lower;
            }
            VulkanPush parameters = push;
            parameters[VK_PUSH_GROUP_BASE] = base;
            parameters[VK_PUSH_GROUP_STRIDE] = groups_x;
            std::array<VkDescriptorBufferInfo,VK_BUFFER_BINDINGS> infos;
            std::array<VkWriteDescriptorSet,VK_BUFFER_BINDINGS> writes = {};
            for (uint32_t i = 0; i < VK_BUFFER_BINDINGS; i++) {
                infos[i] = binding(pointers[i],batch.serial);
                writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[i].dstBinding = i;
                writes[i].dstSet = batch.descriptors[batch.recorded];
                writes[i].descriptorCount = 1;
                writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                writes[i].pBufferInfo = &infos[i];
            }
            vkUpdateDescriptorSets(device,VK_BUFFER_BINDINGS,writes.data(),0,nullptr);
            VkMemoryBarrier before = {};
            before.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            before.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            before.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(batch.command,VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&before,0,nullptr,0,nullptr);
            VkPipeline selected = compute;
            switch (pipeline_kind) {
                case VULKAN_PIPELINE_MATRIX:
                    selected = parameters[2] == 1? vector : matrix;
                    break;
                case VULKAN_PIPELINE_MMQ:
                    selected = mmq.find(parameters[4])->second;
                    break;
                case VULKAN_PIPELINE_QUANTIZE:
                    selected = quantize;
                    break;
                case VULKAN_PIPELINE_COOPERATIVE:
                    selected = cooperative;
                    break;
                case VULKAN_PIPELINE_ATTENTION_SUBGROUP:
                    selected = attention_pipeline(parameters);
                    break;
                case VULKAN_PIPELINE_IM2COL:
                    selected = im2col;
                    break;
                case VULKAN_PIPELINE_ATTENTION:
                    selected = attention;
                    break;
                case VULKAN_PIPELINE_ATTENTION_LARGE:
                    if (!attention_large) attention_large = pipeline(pixelina_attention_shader,sizeof(pixelina_attention_shader),{VK_ATTENTION_DIM_SPECIALIZATION},{VK_QWEN_ATTENTION_MAX_DIM});
                    selected = attention_large;
                    break;
                case VULKAN_PIPELINE_GROUP_PARTIAL:
                    selected = group_partial;
                    break;
                case VULKAN_PIPELINE_GROUP_MERGE:
                    selected = group_merge;
                    break;
                case VULKAN_PIPELINE_GROUP_APPLY:
                    selected = group_apply;
                    break;
            }
            if (!selected) return false;
            vkCmdBindPipeline(batch.command,VK_PIPELINE_BIND_POINT_COMPUTE,selected);
            VkDescriptorSet descriptor_set = batch.descriptors[batch.recorded];
            vkCmdBindDescriptorSets(batch.command,VK_PIPELINE_BIND_POINT_COMPUTE,pipeline_layout,0,1,&descriptor_set,0,nullptr);
            vkCmdPushConstants(batch.command,pipeline_layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(parameters),parameters.data());
            uint32_t query = batch.recorded * VULKAN_QUERY_PAIR;
            if (batch.queries) vkCmdWriteTimestamp(batch.command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,batch.queries,query);
            vkCmdDispatch(batch.command,count,1,1);
            if (batch.queries) vkCmdWriteTimestamp(batch.command,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,batch.queries,query + 1);
            double work = work_units(pipeline_kind,push,base,count);
            batch.samples[batch.recorded] = {kind,work};
            batch.predicted += rate * work * VULKAN_PREDICTION_MARGIN;
            const char* name = nullptr;
            if (pipeline_kind == VULKAN_PIPELINE_MATRIX) name = "matmul";
            else if (pipeline_kind == VULKAN_PIPELINE_MMQ) name = "matmul_mmq";
            else if (pipeline_kind == VULKAN_PIPELINE_COOPERATIVE) name = "matmul_coop";
            else if (pipeline_kind == VULKAN_PIPELINE_QUANTIZE) name = "quantize_q8_1";
            else if (pipeline_kind == VULKAN_PIPELINE_ATTENTION_SUBGROUP) name = "attention_subgroup";
            else if (pipeline_kind == VULKAN_PIPELINE_ATTENTION || pipeline_kind == VULKAN_PIPELINE_ATTENTION_LARGE) name = "attention";
            else if (pipeline_kind == VULKAN_PIPELINE_IM2COL) name = "im2col";
            else if (pipeline_kind == VULKAN_PIPELINE_GROUP_PARTIAL) name = "groupnorm_partial";
            else if (pipeline_kind == VULKAN_PIPELINE_GROUP_MERGE) name = "groupnorm_merge";
            else if (pipeline_kind == VULKAN_PIPELINE_GROUP_APPLY) name = "groupnorm_apply";
            batch.operation = (name? std::string(name) : "operator " + std::to_string(push[0])) + " rows=" + std::to_string(push[1]) + " columns=" + std::to_string(push[2]) + " inner=" + std::to_string(push[3]);
            batch.last = dispatches;
            if (tracing) std::fprintf(stderr,"Vulkan dispatch=%llu recorded %s group_base=%u groups=%u batch=%llu\n",static_cast<unsigned long long>(dispatches),batch.operation.c_str(),base,count,static_cast<unsigned long long>(batch.serial));
            batch.recorded++;
            base += count;
            if (calibrating) {
                // Measure one workgroup once per kernel family before estimating large workloads.
                if (!synchronize()) return false;
                calibrating = false;
            } else if (batch.recorded == VK_DISPATCH_BATCH || batch.predicted >= batch_seconds || base < total) {
                // The descriptor capacity remains a resource cap, not a time estimate.
                if (submit_batch() != VK_SUCCESS) return false;
            }
        }
        dispatches++;
        if (isolated && !synchronize()) return false;
        return true;
    }

    bool multiply(const VulkanPush &push, const std::array<const void*,VK_BUFFER_BINDINGS> &pointers)
    {
        bool packed = integer_dot && push[2] > 1 && push[3] % MODEL_Q8_BLOCK == 0 && mmq.find(push[4]) != mmq.end();
        bool coop = cooperative && push[2] > 1 && (push[4] == MODEL_F32 || push[4] == MODEL_F16 || push[4] == MODEL_BF16);
        std::vector<uint32_t,VulkanAllocator<uint32_t>> activations;
        auto inputs = pointers;
        if (packed) {
            uint64_t blocks = uint64_t(push[2]) * push[3] / MODEL_Q8_BLOCK;
            if (blocks > UINT32_MAX || blocks > SIZE_MAX / (VK_MMQ_BLOCK_WORDS * sizeof(uint32_t))) return false;
            activations.resize(size_t(blocks) * VK_MMQ_BLOCK_WORDS);
            VulkanPush conversion = {};
            conversion[1] = uint32_t(blocks);
            if (!dispatch(VULKAN_PIPELINE_QUANTIZE,conversion,{pointers[0],nullptr,nullptr,nullptr,activations.data()},uint32_t((blocks + VK_COMPUTE_LANES / VK_MMQ_PACKED_WORDS - 1) / (VK_COMPUTE_LANES / VK_MMQ_PACKED_WORDS)))) return false;
            inputs[0] = activations.data();
        }
        uint32_t tile = packed? mmq_tile : (coop? VK_MATRIX_TILE : matrix_tile);
        uint32_t groups_x = push[2] == 1? (push[1] + VK_MATVEC_ROWS - 1) / VK_MATVEC_ROWS : (push[1] + tile - 1) / tile;
        return dispatch(packed? VULKAN_PIPELINE_MMQ : (coop? VULKAN_PIPELINE_COOPERATIVE : VULKAN_PIPELINE_MATRIX),push,inputs,groups_x,(push[2] + tile - 1) / tile);
    }
};

bool vulkan_synchronize()
{
    return !active_state || active_state->synchronize();
}

std::shared_ptr<VulkanState> vulkan_active_state()
{
    return active_state;
}

void* vulkan_allocate(const std::shared_ptr<VulkanState> &state, size_t bytes)
{
    return state->allocate(bytes);
}

void vulkan_free(const std::shared_ptr<VulkanState> &state, void* pointer)
{
    state->release(pointer);
}

bool VulkanSession::initialize()
{
    if (active_state) return false;
    state = std::make_shared<VulkanState>();
    if (!state->initialize()) return false;
    active_state = state;
    return true;
}

VulkanSession::~VulkanSession()
{
    if (active_state == state) active_state.reset();
}

std::string VulkanSession::device_name() const
{
    return state->properties.deviceName;
}

bool VulkanSession::validate_dimensions(const PixSettings &settings) const
{
    uint64_t largest = uint64_t(settings.width) * settings.height * VULKAN_VAE_FINAL_CHANNELS * sizeof(float);
    return largest <= state->properties.limits.maxStorageBufferRange;
}

bool VulkanSession::memory_limits(uint64_t &bytes, uint64_t &range) const
{
    if (!state) return false;
    bytes = state->heap_limit;
    range = state->properties.limits.maxStorageBufferRange;
    if (state->memory_properties) {
        VkPhysicalDeviceMemoryBudgetPropertiesEXT budget = {};
        budget.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
        VkPhysicalDeviceMemoryProperties2KHR memory = {};
        memory.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2_KHR;
        memory.pNext = &budget;
        state->memory_properties(state->physical,&memory);
        uint64_t capacity = budget.heapBudget[state->memory_heap],usage = budget.heapUsage[state->memory_heap];
        if (capacity) bytes = std::min(bytes,capacity > usage? capacity - usage : 0);
    }
    return bytes != 0;
}

bool VulkanSession::prepare_stage(uint64_t activation_bytes, uint64_t scratch_bytes)
{
    if (!state || activation_bytes > state->heap_limit || scratch_bytes > state->heap_limit - activation_bytes || !state->synchronize()) return false;
    while (!state->weights.empty()) state->evict();
    if (!state->synchronize()) return false;
    state->trim();
    uint64_t reserve = std::max(uint64_t(state->allocated),activation_bytes + scratch_bytes);
    if (reserve >= state->heap_limit) return false;
    state->cache_limit = size_t(std::min(packed_weight_budget,uint64_t(state->heap_limit) - reserve));
    return state->cache_limit != 0;
}

bool vulkan_available(std::string &description)
{
    VulkanSession session;
    if (!session.initialize()) {
        description = "Vulkan compute initialization failed.";
        return false;
    }
    description = session.device_name();
    return true;
}

static uint32_t float_bits(float value)
{
    uint32_t bits;
    std::memcpy(&bits,&value,sizeof(bits));
    return bits;
}

static uint32_t groups_for(size_t values)
{
    if (values > UINT32_MAX) return 0;
    return uint32_t((values + VK_COMPUTE_LANES - 1) / VK_COMPUTE_LANES);
}

bool vulkan_matmul(const PixModelFile &file, const PixWeight &weight, const float* input, size_t columns, float* output)
{
    VulkanPush push = {};
    push[1] = uint32_t(weight.shape[1]);
    push[2] = uint32_t(columns);
    push[3] = uint32_t(weight.shape[0]);
    push[4] = weight.type;
    push[5] = uint32_t(weight_bytes({weight.shape[0]},weight.type));
    push[6] = push[1];
    void* packed = active_state->weight(file,weight);
    return active_state->multiply(push,{input,nullptr,nullptr,packed,output});
}

PixTensor vulkan_norm(const PixTensor &input, const std::vector<float> &weight, float epsilon, bool centered)
{
    PixTensor output(input.rows,input.channels);
    PixTensor affine(1,weight.empty()? 1 : weight.size());
    std::copy(weight.begin(),weight.end(),affine.values.begin());
    VulkanPush push = {};
    push[0] = VK_OP_NORM;
    push[1] = uint32_t(input.rows);
    push[2] = uint32_t(input.channels);
    push[3] = centered;
    push[4] = float_bits(epsilon);
    push[5] = !weight.empty();
    if (!active_state->dispatch(false,push,{input.values.data(),affine.values.data(),nullptr,nullptr,output.values.data()},push[1])) return {};
    return output;
}

void vulkan_pointwise(unsigned operation, PixTensor &destination, const PixTensor* second, const PixTensor* third, unsigned part, float scale, float bias)
{
    if ((second && second->values.empty()) || (third && third->values.empty())) destination = PixTensor();
    if (destination.values.empty()) return;
    VulkanPush push = {};
    push[0] = operation;
    push[1] = uint32_t(destination.rows);
    push[2] = uint32_t(destination.channels);
    push[3] = part;
    push[4] = third != nullptr;
    if (operation == VK_OP_AFFINE || operation == VK_OP_EULER) {
        push[3] = float_bits(scale);
        push[4] = float_bits(bias);
    }
    if (!active_state->dispatch(false,push,{destination.values.data(),second? second->values.data() : nullptr,third? third->values.data() : nullptr,nullptr,destination.values.data()},groups_for(destination.values.size()))) destination = PixTensor();
}

PixTensor vulkan_attention(const PixTensor &q, const PixTensor &k, const PixTensor &v, size_t heads, size_t kv_heads, bool causal)
{
    if (q.channels / heads > VULKAN_MAX_ATTENTION_DIM) return {};
    PixTensor output(q.rows,q.channels);
    VulkanPush push = {};
    push[0] = VK_OP_ATTENTION;
    push[1] = uint32_t(q.rows);
    push[2] = uint32_t(q.channels);
    push[3] = uint32_t(heads);
    push[4] = uint32_t(kv_heads);
    push[5] = uint32_t(k.rows);
    push[6] = causal;
    bool subgroup = active_state->subgroup_size && q.channels / heads <= VK_ATTENTION_MAX_DIM && q.rows >= VK_FAST_ATTENTION_MIN_ROWS && (q.channels / heads) % (VK_ATTENTION_VECTOR * VK_FAST_ATTENTION_D_SPLIT) == 0;
    uint32_t tile = subgroup? (q.rows > VK_FAST_ATTENTION_SMALL_ROWS && k.rows >= VK_FAST_ATTENTION_LARGE_KEYS? VK_FAST_ATTENTION_ROWS : VK_FAST_ATTENTION_SMALL_ROWS) : VK_ATTENTION_QUERY_TILE;
    if (subgroup) {
        // Use the passing profile-sized tiles; retain the existing choice for other dimensions.
        if (q.rows > VK_FAST_ATTENTION_DIFFUSER_ROWS && k.rows >= VK_FAST_ATTENTION_LARGE_KEYS) {
            if (q.channels / heads == VK_FAST_ATTENTION_DEFAULT_DIM) tile = VK_FAST_ATTENTION_DIFFUSER_ROWS;
            else if (q.channels / heads == VULKAN_Z_VAE_ATTENTION_DIM) tile = VK_FAST_ATTENTION_VAE_ROWS;
        }
        size_t shared_per_row = (q.channels / heads / VK_ATTENTION_VECTOR + 1) * VK_ATTENTION_VECTOR * sizeof(float);
        while (tile > VK_FAST_ATTENTION_MIN_ROWS && tile * shared_per_row > active_state->properties.limits.maxComputeSharedMemorySize) tile /= 2;
        if (tile * shared_per_row > active_state->properties.limits.maxComputeSharedMemorySize) {
            subgroup = false;
            tile = VK_ATTENTION_QUERY_TILE;
        }
    }
    push[VK_PUSH_ATTENTION_TILE] = tile;
    unsigned portable = q.channels / heads > VK_ATTENTION_MAX_DIM? VULKAN_PIPELINE_ATTENTION_LARGE : VULKAN_PIPELINE_ATTENTION;
    if (!active_state->dispatch(subgroup? VULKAN_PIPELINE_ATTENTION_SUBGROUP : portable,push,{q.values.data(),k.values.data(),v.values.data(),nullptr,output.values.data()},uint32_t((q.rows + tile - 1) / tile * heads))) return {};
    return output;
}

void vulkan_rope(PixTensor &tensor, size_t heads, const std::vector<float> &cosine, const std::vector<float> &sine, bool interleaved)
{
    PixTensor c(1,cosine.size()),s(1,sine.size()),output(tensor.rows,tensor.channels);
    std::copy(cosine.begin(),cosine.end(),c.values.begin());
    std::copy(sine.begin(),sine.end(),s.values.begin());
    VulkanPush push = {};
    push[0] = VK_OP_ROPE;
    push[1] = uint32_t(tensor.rows);
    push[2] = uint32_t(tensor.channels);
    push[3] = uint32_t(heads);
    push[4] = interleaved;
    if (!active_state->dispatch(false,push,{tensor.values.data(),c.values.data(),s.values.data(),nullptr,output.values.data()},groups_for(tensor.values.size()))) output = PixTensor();
    tensor = std::move(output);
}

PixTensor vulkan_conv(const PixModelFile &model, const std::string &name, const PixTensor &input, size_t width, size_t height)
{
    const PixWeight &weight = model.tensor(name + WEIGHT_SUFFIX);
    size_t inner = size_t(weight.shape[0] * weight.shape[1] * weight.shape[2]);
    PixTensor output(input.rows,size_t(weight.shape[3]));
    std::vector<float> vector = tensor_vector(model,name + BIAS_SUFFIX);
    PixTensor bias(1,vector.size());
    std::copy(vector.begin(),vector.end(),bias.values.begin());
    VulkanPush matrix_push = {};
    matrix_push[1] = uint32_t(output.channels);
    matrix_push[3] = uint32_t(inner);
    matrix_push[4] = weight.type;
    matrix_push[5] = uint32_t(weight_bytes({inner},weight.type));
    matrix_push[6] = uint32_t(output.channels);
    matrix_push[8] = 1;
    for (size_t base = 0; base < input.rows; base += VK_CONV_TILE_PIXELS) {
        size_t count = std::min(size_t(VK_CONV_TILE_PIXELS),input.rows - base);
        PixTensor patches(count,inner);
        VulkanPush patch_push = {};
        patch_push[1] = uint32_t(count);
        patch_push[3] = uint32_t(width);
        patch_push[4] = uint32_t(height);
        patch_push[5] = uint32_t(input.channels);
        patch_push[6] = uint32_t(weight.shape[0]);
        patch_push[7] = uint32_t(base);
        if (!active_state->dispatch(VULKAN_PIPELINE_IM2COL,patch_push,{input.values.data(),nullptr,nullptr,nullptr,patches.values.data()},uint32_t(count))) return {};
        matrix_push[2] = uint32_t(count);
        matrix_push[7] = uint32_t(base * output.channels);
        void* packed = active_state->weight(model,weight);
        if (!active_state->multiply(matrix_push,{patches.values.data(),bias.values.data(),nullptr,packed,output.values.data()})) return {};
    }
    return output;
}

PixTensor vulkan_groupnorm(const PixModelFile &model, const std::string &name, const PixTensor &input)
{
    std::vector<float> w = tensor_vector(model,name + WEIGHT_SUFFIX),b = tensor_vector(model,name + BIAS_SUFFIX);
    PixTensor weight(1,w.size()),bias(1,b.size()),output(input.rows,input.channels);
    std::copy(w.begin(),w.end(),weight.values.begin());
    std::copy(b.begin(),b.end(),bias.values.begin());
    VulkanPush push = {};
    push[0] = VK_OP_GROUP;
    push[1] = uint32_t(input.rows);
    push[2] = uint32_t(input.channels);
    push[3] = COMPUTE_VAE_GROUPS;
    push[4] = float_bits(COMPUTE_NORM_EPSILON);
    push[5] = 1;
    size_t features = input.channels / COMPUTE_VAE_GROUPS;
    if (input.channels % COMPUTE_VAE_GROUPS == 0 && features % VK_GROUP_VECTOR == 0) {
        size_t chunks = (input.rows * features + VK_GROUP_CHUNK - 1) / VK_GROUP_CHUNK;
        PixTensor partials(chunks * COMPUTE_VAE_GROUPS,VK_GROUP_STATS_WORDS),statistics(COMPUTE_VAE_GROUPS,VK_GROUP_FINAL_STATS_WORDS);
        push[6] = uint32_t(chunks);
        if (!active_state->dispatch(VULKAN_PIPELINE_GROUP_PARTIAL,push,{input.values.data(),nullptr,nullptr,nullptr,partials.values.data()},uint32_t(chunks * COMPUTE_VAE_GROUPS))) return {};
        if (!active_state->dispatch(VULKAN_PIPELINE_GROUP_MERGE,push,{partials.values.data(),nullptr,nullptr,nullptr,statistics.values.data()},COMPUTE_VAE_GROUPS)) return {};
        if (!active_state->dispatch(VULKAN_PIPELINE_GROUP_APPLY,push,{input.values.data(),weight.values.data(),bias.values.data(),statistics.values.data(),output.values.data()},groups_for(output.values.size() / VK_GROUP_VECTOR))) return {};
    } else {
        // The scalar kernel handles group widths that cannot use aligned vec4 loads.
        if (!active_state->dispatch(false,push,{input.values.data(),weight.values.data(),bias.values.data(),nullptr,output.values.data()},COMPUTE_VAE_GROUPS)) return {};
    }
    return output;
}

PixTensor vulkan_embedding(const PixModelFile &model, const PixWeight &weight, const std::vector<int> &ids)
{
    size_t stride = size_t(weight_bytes({weight.shape[0]},weight.type));
    void* packed = active_state->allocate(ids.size() * stride);
    for (size_t i = 0; i < ids.size(); i++) std::memcpy(static_cast<unsigned char*>(packed) + i * stride,model.data + weight.offset + size_t(ids[i]) * stride,stride);
    PixTensor output(ids.size(),size_t(weight.shape[0]));
    VulkanPush push = {};
    push[0] = VK_OP_DECODE;
    push[1] = uint32_t(ids.size());
    push[2] = uint32_t(weight.shape[0]);
    push[3] = weight.type;
    push[4] = uint32_t(stride);
    bool result = active_state->dispatch(false,push,{nullptr,nullptr,nullptr,packed,output.values.data()},groups_for(output.values.size()));
    active_state->release(packed);
    if (!result) return {};
    return output;
}

void vulkan_copy_rows(const PixTensor &source, PixTensor &destination, size_t rows, size_t source_row, size_t source_feature, size_t destination_row)
{
    if (source.values.empty()) destination = PixTensor();
    if (destination.values.empty()) return;
    VulkanPush push = {};
    push[0] = VK_OP_COPY;
    push[1] = uint32_t(rows);
    push[2] = uint32_t(destination.channels);
    push[3] = uint32_t(source.channels);
    push[4] = uint32_t(source_row * source.channels + source_feature);
    push[5] = uint32_t(destination_row * destination.channels);
    if (!active_state->dispatch(VULKAN_PIPELINE_GENERAL,push,{source.values.data(),nullptr,nullptr,nullptr,destination.values.data()},groups_for(rows * destination.channels))) destination = PixTensor();
}

bool vulkan_copy_features(const PixTensor &source, PixTensor &destination, size_t destination_feature)
{
    if (source.rows != 1 || destination.rows != 1 || source.values.empty() || destination.values.empty() || destination_feature > destination.channels || source.channels > destination.channels - destination_feature) return false;
    VulkanPush push = {};
    push[0] = VK_OP_COPY;
    push[1] = 1;
    push[2] = uint32_t(source.channels);
    push[3] = uint32_t(source.channels);
    push[5] = uint32_t(destination_feature);
    return active_state->dispatch(VULKAN_PIPELINE_GENERAL,push,{source.values.data(),nullptr,nullptr,nullptr,destination.values.data()},groups_for(source.channels));
}

void vulkan_layout(unsigned operation, const PixTensor &source, PixTensor &destination, size_t width, size_t patch)
{
    if (source.values.empty()) destination = PixTensor();
    if (destination.values.empty()) return;
    VulkanPush push = {};
    push[0] = operation;
    push[1] = uint32_t(destination.rows);
    push[2] = uint32_t(destination.channels);
    push[3] = uint32_t(width);
    push[4] = uint32_t(source.channels);
    push[5] = uint32_t(patch);
    if (!active_state->dispatch(VULKAN_PIPELINE_GENERAL,push,{source.values.data(),nullptr,nullptr,nullptr,destination.values.data()},groups_for(destination.values.size()))) destination = PixTensor();
}

void vulkan_bias(PixTensor &output, const std::vector<float> &bias)
{
    PixTensor affine(1,bias.size());
    std::copy(bias.begin(),bias.end(),affine.values.begin());
    vulkan_pointwise(VK_OP_BIAS,output,&affine,nullptr);
}

PixTensor vulkan_l2norm(const PixTensor &input, const std::vector<float> &weight, float epsilon)
{
    PixTensor output(input.rows,input.channels),affine(1,weight.size());
    std::copy(weight.begin(),weight.end(),affine.values.begin());
    VulkanPush push = {};
    push[0] = VK_OP_L2NORM;
    push[1] = uint32_t(input.rows);
    push[2] = uint32_t(input.channels);
    push[4] = float_bits(epsilon);
    push[5] = !weight.empty();
    if (!active_state->dispatch(VULKAN_PIPELINE_GENERAL,push,{input.values.data(),affine.values.data(),nullptr,nullptr,output.values.data()},uint32_t(input.rows))) return {};
    return output;
}

PixTensor vulkan_attention(const PixTensor &q, const PixTensor &k, const PixTensor &v, size_t heads, size_t kv_heads, const std::vector<PixAttentionRange> &ranges, size_t query_offset)
{
    if (q.channels / heads > VULKAN_MAX_ATTENTION_DIM || query_offset > UINT32_MAX - q.rows) return {};
    PixTensor output(q.rows,q.channels);
    VulkanPush push = {};
    push[0] = VK_OP_ATTENTION;
    push[1] = uint32_t(q.rows);
    push[2] = uint32_t(q.channels);
    push[3] = uint32_t(heads);
    push[4] = uint32_t(kv_heads);
    push[5] = uint32_t(k.rows);
    push[VK_PUSH_QUERY_OFFSET] = uint32_t(query_offset);
    push[VK_PUSH_ATTENTION_TILE] = VK_ATTENTION_QUERY_TILE;
    for (const PixAttentionRange &range : ranges) {
        push[6] = range.causal;
        push[VK_PUSH_QUERY_BEGIN] = uint32_t(range.query_begin);
        push[VK_PUSH_QUERY_END] = uint32_t(range.query_end);
        push[VK_PUSH_KEY_END] = uint32_t(range.key_end);
        size_t tiles = (range.query_end - range.query_begin + VK_ATTENTION_QUERY_TILE - 1) / VK_ATTENTION_QUERY_TILE;
        unsigned pipeline = q.channels / heads > VK_ATTENTION_MAX_DIM? VULKAN_PIPELINE_ATTENTION_LARGE : VULKAN_PIPELINE_ATTENTION;
        if (!active_state->dispatch(pipeline,push,{q.values.data(),k.values.data(),v.values.data(),nullptr,output.values.data()},uint32_t(tiles * heads))) return {};
    }
    return output;
}

bool vulkan_rows(const PixTensor &source, PixTensor &destination, const std::vector<size_t> &rows, bool scatter)
{
    if (rows.empty()) return false;
    void* indices = active_state->allocate(rows.size() * sizeof(uint32_t));
    if (!indices) return false;
    uint32_t* mapped = static_cast<uint32_t*>(indices);
    for (size_t i = 0; i < rows.size(); i++) mapped[i] = uint32_t(rows[i]);
    VulkanPush push = {};
    push[0] = scatter? VK_OP_SCATTER : VK_OP_GATHER;
    push[1] = uint32_t(rows.size());
    push[2] = uint32_t(source.channels);
    bool result = active_state->dispatch(VULKAN_PIPELINE_GENERAL,push,{source.values.data(),indices,nullptr,nullptr,destination.values.data()},groups_for(rows.size() * source.channels));
    active_state->release(indices);
    return result;
}

PixTensor vulkan_conv(const PixModelFile &model, const std::string &name, const PixTensor &input, size_t width, size_t height, const PixConv2D &geometry, bool with_bias)
{
    const PixWeight &weight = model.tensor(name + WEIGHT_SUFFIX);
    size_t inner = size_t(weight.shape[0] * weight.shape[1]) * input.channels;
    size_t output_width = (width + geometry.left + geometry.right - weight.shape[0]) / geometry.stride_x + 1;
    size_t output_height = (height + geometry.top + geometry.bottom - weight.shape[1]) / geometry.stride_y + 1;
    PixTensor output(output_width * output_height,size_t(weight.shape.back()));
    PixWeight flattened = weight;
    flattened.shape = {inner,output.channels};
    std::vector<float> vector;
    if (with_bias) vector = tensor_vector(model,name + BIAS_SUFFIX);
    PixTensor bias(1,vector.size());
    std::copy(vector.begin(),vector.end(),bias.values.begin());
    VulkanPush matrix_push = {};
    matrix_push[1] = uint32_t(output.channels);
    matrix_push[3] = uint32_t(inner);
    matrix_push[4] = weight.type;
    matrix_push[5] = uint32_t(weight_bytes({inner},weight.type));
    matrix_push[6] = uint32_t(output.channels);
    matrix_push[8] = with_bias;
    for (size_t base = 0; base < output.rows; base += VK_CONV_TILE_PIXELS) {
        size_t count = std::min(size_t(VK_CONV_TILE_PIXELS),output.rows - base);
        PixTensor patches(count,inner);
        VulkanPush patch_push = {};
        patch_push[1] = uint32_t(count);
        patch_push[3] = uint32_t(width);
        patch_push[4] = uint32_t(height);
        patch_push[5] = uint32_t(input.channels);
        patch_push[6] = uint32_t(weight.shape[0]);
        patch_push[7] = uint32_t(base);
        patch_push[8] = uint32_t(output_width);
        patch_push[9] = uint32_t(weight.shape[1]);
        patch_push[10] = uint32_t(geometry.stride_x);
        patch_push[11] = uint32_t(geometry.stride_y);
        patch_push[12] = uint32_t(geometry.left);
        patch_push[13] = uint32_t(geometry.top);
        if (!active_state->dispatch(VULKAN_PIPELINE_IM2COL,patch_push,{input.values.data(),nullptr,nullptr,nullptr,patches.values.data()},uint32_t(count))) return {};
        matrix_push[2] = uint32_t(count);
        matrix_push[7] = uint32_t(base * output.channels);
        void* packed = active_state->weight(model,flattened);
        if (!packed || !active_state->multiply(matrix_push,{patches.values.data(),bias.values.data(),nullptr,packed,output.values.data()})) return {};
    }
    return output;
}

bool vulkan_shortcut(const PixTensor &source, PixTensor &destination, size_t width, size_t temporal, bool up, bool spatial)
{
    VulkanPush push = {};
    push[0] = VK_OP_SHORTCUT;
    push[1] = uint32_t(destination.rows);
    push[2] = uint32_t(destination.channels);
    push[3] = uint32_t(source.channels);
    push[4] = uint32_t(width);
    push[5] = spatial? 2 : 1;
    push[6] = uint32_t(temporal);
    push[7] = up;
    return active_state->dispatch(VULKAN_PIPELINE_GENERAL,push,{source.values.data(),nullptr,nullptr,nullptr,destination.values.data()},groups_for(destination.values.size()));
}

void vulkan_qwen_modulate(PixTensor &destination, const PixTensor &parameters, size_t target_begin, size_t part, const PixTensor* residual)
{
    VulkanPush push = {};
    push[0] = residual? VK_OP_QWEN_RESIDUAL : VK_OP_QWEN_SCALE;
    push[1] = uint32_t(destination.rows);
    push[2] = uint32_t(destination.channels);
    push[3] = uint32_t(part);
    push[4] = uint32_t(target_begin);
    push[5] = uint32_t(parameters.channels);
    if (!active_state->dispatch(VULKAN_PIPELINE_GENERAL,push,{destination.values.data(),parameters.values.data(),residual? residual->values.data() : nullptr,nullptr,destination.values.data()},groups_for(destination.values.size()))) destination = PixTensor();
}

#else /* PIXELINA_WITH_VULKAN */

bool vulkan_available(std::string &description)
{
    description = "Vulkan support is not compiled in.";
    return false;
}

#endif /* PIXELINA_WITH_VULKAN */
