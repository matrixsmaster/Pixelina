/* This file is a part of 'Pixelina' Image Generator, Editor, and Viewer.
 *
 * (C) Dmitry 'sciloaf' Solovyev, 2026
 *
 * License: MIT
 */
#ifndef PIXELINA_QWEN_IMAGE_H
#define PIXELINA_QWEN_IMAGE_H

#include "qwen_vl.h"
#include "qwen_vae.h"

#define QWEN_IMAGE_TEXT_NORM "txt_in.text_norm.weight"
#define QWEN_IMAGE_BLOCK_PREFIX "transformer_blocks."
#define QWEN_IMAGE_QUERY_NORM "attn.norm_q.weight"
#define QWEN_IMAGE_KEY_NORM "attn.norm_k.weight"
#define QWEN_IMAGE_FFN_FUSED_WEIGHT "img_mlp.gate_up.weight"
#define QWEN_IMAGE_CONTAINER_ERROR "Cannot read the selected Qwen Image component containers."
#define QWEN_IMAGE_INPUT_PROJECTION "img_in"
#define QWEN_IMAGE_FFN_FUSED "img_mlp.gate_up"

#define QWEN_IMAGE_HIDDEN 4096
#define QWEN_IMAGE_MLP 12288
#define QWEN_IMAGE_LAYERS 32
#define QWEN_IMAGE_HEADS 32
#define QWEN_IMAGE_HEAD_DIM 128
#define QWEN_IMAGE_CHANNELS 64
#define QWEN_IMAGE_EPSILON 0.000001f
#define QWEN_IMAGE_TIME_DIM 256
#define QWEN_IMAGE_TIME_PERIOD 10000.0
#define QWEN_IMAGE_TIME_SCALE 1000.0f
#define QWEN_IMAGE_THETA 10000.0
#define QWEN_IMAGE_ROPE_T 16
#define QWEN_IMAGE_ROPE_H 56
#define QWEN_IMAGE_ROPE_W 56
#define QWEN_IMAGE_SIGMA_BASE_LENGTH 256.0
#define QWEN_IMAGE_SIGMA_MAX_LENGTH 8192.0
#define QWEN_IMAGE_SIGMA_BASE_SHIFT 0.5
#define QWEN_IMAGE_SIGMA_MAX_SHIFT 0.9
#define QWEN_IMAGE_SIGMA_TERMINAL 0.02f
#define QWEN_IMAGE_DIFFUSION_BEGIN 0.42f
#define QWEN_IMAGE_DIFFUSION_SPAN 0.48f
#define QWEN_IMAGE_MEMORY_LINE 256
#define QWEN_IMAGE_RESIZE_TAP_RESERVE 128
#define QWEN_IMAGE_KIB_BYTES 1024
#define QWEN_IMAGE_HOST_RESERVE (512u * 1024u * 1024u)
#define QWEN_IMAGE_STAGE_COUNT 5
#define QWEN_IMAGE_MODULATION_PARTS 4
#define QWEN_MEMORY_VISION_MAPS 8
#define QWEN_MEMORY_LANGUAGE_ATTN_MAPS 4
#define QWEN_MEMORY_FFN_MAPS 3
#define QWEN_MEMORY_DIFFUSION_ATTN_MAPS 7
#define QWEN_MEMORY_ENCODER_BASE_MAPS 3
#define QWEN_MEMORY_DECODER_BASE_MAPS 6

struct PixQwenImageShape {
    size_t width = 0,height = 0; // latent grid in input order
};

struct PixQwenLayout {
    size_t rows = 0; // complete joint token count
    size_t target_begin = 0,target_end = 0; // appended target interval
    std::vector<size_t> text_sources; // non-image conditioning rows used by text projection
    std::vector<size_t> text_rows; // corresponding joint destinations
    std::vector<std::vector<size_t>> reference_rows; // latent-order joint destinations for each reference
    std::vector<size_t> image_rows; // all reference and target image rows
    std::vector<PixAttentionRange> attention; // complete block-causal query coverage
    PixRotary rotary; // three-axis interleaved rotary factors
};

struct PixQwenKVLayer {
    PixTensor key,value; // F32 normalized/rotated prefix keys and projected values
};

struct PixQwenKVCache {
    std::vector<PixQwenKVLayer> layers; // one fixed prefix per diffuser block
    PixRotary rotary; // absolute target positions for subsequent evaluations
    const PixModelFile* model = nullptr; // immutable job model owning the cached projections
    const PixQwenLayout* layout = nullptr; // immutable job layout and prefix boundaries
    const PixQwenCondition* condition = nullptr; // immutable job text conditioning
    const std::vector<PixTensor>* references = nullptr; // immutable job reference latents
    bool ready = false; // published only after every block and output projection succeed
};

enum PixQwenStage {
    QWEN_STAGE_VISION,QWEN_STAGE_LANGUAGE,QWEN_STAGE_ENCODE,QWEN_STAGE_DIFFUSION,QWEN_STAGE_DECODE
};

struct PixQwenMemoryInput {
    size_t language_tokens = 0; // expanded complete language input length
    std::vector<PixVisionGrid> vision; // processor grids
    std::vector<PixQwenImageShape> references; // fixed-reference latent grids
    PixQwenImageShape target; // target latent grid
    std::array<uint64_t,QWEN_IMAGE_STAGE_COUNT> mapped_weights = {}; // inspected file working sets by stage
    std::array<uint64_t,QWEN_IMAGE_STAGE_COUNT> largest_weight = {}; // largest required packed binding by stage
    uint64_t source_bytes = 0; // immutable original image planes retained by the caller
    size_t cpu_threads = 1; // score-row and operator scratch workers
    bool kv_cache = false; // request F32 prefix caching with automatic budget fallback
};

struct PixQwenMemoryLimits {
    uint64_t host_bytes = 0; // available host memory after the OS/UI reserve
    uint64_t device_bytes = 0; // available application memory on the selected heap
    uint64_t max_buffer_bytes = UINT64_MAX; // largest permitted storage binding
    bool shared_memory = false; // device allocations also consume the host budget
};

struct PixQwenStageMemory {
    PixQwenStage stage; // ordered sequential stage
    uint64_t activation_bytes = 0; // live activation peak including retained tensors
    uint64_t scratch_bytes = 0; // bounded operator and worker scratch
    uint64_t retained_bytes = 0; // source, prepared and conditioning inputs surviving the stage
    uint64_t host_peak_bytes = 0,device_peak_bytes = 0; // checked complete memory peaks
    uint64_t largest_binding_bytes = 0; // maximum individual activation or packed weight buffer
    uint64_t packed_cache_bytes = 0; // weight residency admitted after stage reservation
};

struct PixQwenMemoryPlan {
    std::vector<PixQwenStageMemory> stages; // validated execution-order entries
    uint64_t kv_cache_bytes = 0; // admitted per-layer F32 prefix storage, zero means uncached
};

bool qwen_image_validate(const PixEditSettings &settings, std::string &error);
std::vector<PixImage> qwen_image_prepare(const std::vector<PixImage> &references, int resolution);
PixQwenLayout qwen_image_layout(const PixQwenCondition &condition, const std::vector<PixQwenImageShape> &references, size_t width, size_t height);
PixTensor qwen_image_diffuse(const PixModelFile &diffuser, const PixTensor &target, const std::vector<PixTensor> &references, const PixQwenCondition &condition, const PixQwenLayout &layout, float sigma, const InferenceProgress &progress, PixQwenKVCache* cache = nullptr);
std::vector<float> qwen_image_sigmas(int steps, size_t target_tokens);
void qwen_image_euler(PixTensor &target, const PixTensor &flow, float sigma, float next_sigma);
bool qwen_image_memory_plan(const PixQwenMemoryInput &input, const PixQwenMemoryLimits &limits, PixQwenMemoryPlan &plan, std::string &error);
PixImage qwen_image_edit(const PixEditSettings &settings, const std::vector<PixImage> &references, const std::string &prompt, const InferenceProgress &progress, int &error);

#endif
