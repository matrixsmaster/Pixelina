/* This file is a part of 'Pixelina' Image Generator, Editor, and Viewer.
 *
 * (C) Dmitry 'sciloaf' Solovyev, 2026
 *
 * License: MIT
 */
#ifndef PIXELINA_BACKEND_VULKAN_H
#define PIXELINA_BACKEND_VULKAN_H

#include <string>
#include <memory>
#include <type_traits>
#include <utility>
#include <new>

#ifdef PIXELINA_WITH_VULKAN

#include "model.h"
#include "vulkan_ops.h"

#define VULKAN_BATCH_SECONDS 0.25
#define VULKAN_KERNEL_AUTO "auto"
#define VULKAN_KERNEL_PORTABLE "portable"
#define VULKAN_KERNEL_OPTIMIZED "optimized"
#define VULKAN_DEFAULT_BUFFER_BYTES 4
#define VULKAN_HEAP_FRACTION 0.85
#define VULKAN_DEVICE_RANK_DISCRETE 4
#define VULKAN_DEVICE_RANK_INTEGRATED 3
#define VULKAN_DEVICE_RANK_VIRTUAL 2
#define VULKAN_DEVICE_RANK_CPU 1
#define VULKAN_VAE_FINAL_CHANNELS 128
#define VULKAN_MAX_ATTENTION_DIM VK_ATTENTION_MAX_DIM
#define VULKAN_CACHE_RESERVE_BYTES (512u * 1024u * 1024u)
#define VULKAN_PIPELINE_GENERAL 0
#define VULKAN_PIPELINE_MATRIX 1
#define VULKAN_PIPELINE_IM2COL 2
#define VULKAN_PIPELINE_ATTENTION 3
#define VULKAN_PIPELINE_MMQ 4
#define VULKAN_PIPELINE_QUANTIZE 5
#define VULKAN_PIPELINE_ATTENTION_SUBGROUP 6
#define VULKAN_PIPELINE_COOPERATIVE 7
#define VULKAN_PIPELINE_GROUP_PARTIAL 8
#define VULKAN_PIPELINE_GROUP_MERGE 9
#define VULKAN_PIPELINE_GROUP_APPLY 10
#define VULKAN_MAX_BATCH_SECONDS 0.5
#define VULKAN_PREDICTION_MARGIN 2.0
#define VULKAN_RATE_DECAY 0.75
#define VULKAN_MIN_SAMPLE_SECONDS 0.000001
#define VULKAN_NANOSECONDS 1000000000.0
#define VULKAN_NORMALIZATION_PASSES 3
#define VULKAN_ATTENTION_WORK_PER_ELEMENT 4
#define VULKAN_COST_MATRIX 0x100
#define VULKAN_COST_VECTOR 0x200
#define VULKAN_COST_ATTENTION 0x300
#define VULKAN_COST_IM2COL 0x400
#define VULKAN_COST_DECODE 0x500
#define VULKAN_COST_MMQ 0x600
#define VULKAN_COST_QUANTIZE 0x700
#define VULKAN_COST_ATTENTION_SUBGROUP 0x800
#define VULKAN_COST_COOPERATIVE 0x4000
#define VULKAN_COST_GROUP_PARTIAL 0x5000
#define VULKAN_COST_GROUP_MERGE 0x5001
#define VULKAN_COST_GROUP_APPLY 0x5002
#define VULKAN_QUERY_PAIR 2

struct VulkanOptions {
    bool tracing = false; // explicit diagnostic logging
    double batch_seconds = VULKAN_BATCH_SECONDS; // GPU submission time target
    uint64_t synchronize_from = UINT64_MAX; // first isolated diagnostic dispatch
    std::string kernels = VULKAN_KERNEL_AUTO; // automatic, portable or optimized kernels
};

bool vulkan_configure(const VulkanOptions &options);

struct PixTensor;
struct VulkanState;
std::shared_ptr<VulkanState> vulkan_active_state();
void* vulkan_allocate(const std::shared_ptr<VulkanState> &state, size_t bytes);
void vulkan_free(const std::shared_ptr<VulkanState> &state, void* pointer);
bool vulkan_synchronize();

// Activations use coherent mappings; GPU layouts stay resident and host reads synchronize.
template<typename T> struct VulkanAllocator {
    using value_type = T;
    using propagate_on_container_move_assignment = std::true_type;
    std::shared_ptr<VulkanState> state; // owning device lifetime, or CPU allocation mode
    VulkanAllocator() : state(vulkan_active_state()) {}
    template<typename U> VulkanAllocator(const VulkanAllocator<U> &other) : state(other.state) {}
    T* allocate(size_t count)
    {
        if (state) return static_cast<T*>(vulkan_allocate(state,count * sizeof(T)));
        return static_cast<T*>(::operator new(count * sizeof(T)));
    }
    void deallocate(T* pointer, size_t)
    {
        if (state) vulkan_free(state,pointer);
        else ::operator delete(pointer);
    }
    void construct(T* pointer)
    {
        // Output kernels overwrite every element; avoid clearing their buffers on the CPU.
        if (state) ::new (static_cast<void*>(pointer)) T;
        else ::new (static_cast<void*>(pointer)) T();
    }
    template<typename... Args> void construct(T* pointer, Args&&... args)
    {
        ::new (static_cast<void*>(pointer)) T(std::forward<Args>(args)...);
    }
    template<typename U> bool operator==(const VulkanAllocator<U> &other) const
    {
        return state == other.state;
    }
    template<typename U> bool operator!=(const VulkanAllocator<U> &other) const
    {
        return state != other.state;
    }
};

class VulkanSession {
    std::shared_ptr<VulkanState> state; // independent Vulkan compute context
public:
    VulkanSession() = default;
    bool initialize();
    ~VulkanSession();
    std::string device_name() const;
    bool validate_dimensions(const PixSettings &settings) const;
};

bool vulkan_matmul(const PixModelFile &file, const PixWeight &weight, const float* input, size_t columns, float* output);
PixTensor vulkan_norm(const PixTensor &input, const std::vector<float> &weight, float epsilon, bool centered);
void vulkan_pointwise(unsigned operation, PixTensor &destination, const PixTensor* second, const PixTensor* third, unsigned part = 0, float scale = 0, float bias = 0);
PixTensor vulkan_attention(const PixTensor &q, const PixTensor &k, const PixTensor &v, size_t heads, size_t kv_heads, bool causal);
void vulkan_rope(PixTensor &tensor, size_t heads, const std::vector<float> &cosine, const std::vector<float> &sine, bool interleaved);
PixTensor vulkan_conv(const PixModelFile &model, const std::string &name, const PixTensor &input, size_t width, size_t height);
PixTensor vulkan_groupnorm(const PixModelFile &model, const std::string &name, const PixTensor &input);
PixTensor vulkan_embedding(const PixModelFile &model, const PixWeight &weight, const std::vector<int> &ids);
void vulkan_copy_rows(const PixTensor &source, PixTensor &destination, size_t rows, size_t source_row = 0, size_t source_feature = 0, size_t destination_row = 0);
void vulkan_layout(unsigned operation, const PixTensor &source, PixTensor &destination, size_t width, size_t patch = 0);
void vulkan_bias(PixTensor &output, const std::vector<float> &bias);

#endif /* PIXELINA_WITH_VULKAN */

bool vulkan_available(std::string &description); // always available

#endif /* PIXELINA_BACKEND_VULKAN_H */
