/* This file is a part of 'Pixelina' Image Generator, Editor, and Viewer.
 *
 * (C) Dmitry 'sciloaf' Solovyev, 2026
 *
 * License: MIT
 */
#ifndef PIXELINA_TENSOR_H
#define PIXELINA_TENSOR_H

#include "model.h"
#include <functional>

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    #include <immintrin.h>
    #define PIX_KERNEL_AVX2 __attribute__((target("avx2,fma")))
    #define COMPUTE_X86_SIMD 1
#endif

#ifdef PIXELINA_WITH_VULKAN
    #include "backend_vulkan.h"
    using PixTensorVals = std::vector<float,VulkanAllocator<float>>;
#else
    using PixTensorVals = std::vector<float>;
#endif

#define COMPUTE_THREAD_MIN_WORK 16
#define COMPUTE_TILE_ROWS 512
#define COMPUTE_WEIGHT_ROWS 128
#define COMPUTE_MATRIX_ROWS 4
#define COMPUTE_MATRIX_COLUMNS 3
#define COMPUTE_SIMD_LANES 8
#define COMPUTE_NORM_EPSILON 0.000001f
#define COMPUTE_Z_NORM_EPSILON 0.00001f
#define COMPUTE_VAE_GROUPS 32
#define HALF_MANTISSA_BITS 10
#define HALF_EXPONENT_BIAS 15
#define HALF_EXPONENT_MASK 31
#define HALF_MANTISSA_MASK 1023
#define HALF_SIGN_MASK 0x8000
#define COMPUTE_GELU_SCALE 0.7978845608028654f
#define COMPUTE_GELU_CUBIC 0.044715f
#define COMPUTE_SQRT_TWO 1.4142135623730951f
#define K_SCALE_MASK 63
#define Q5_HIGH_OFFSET 16
#define Q5_LOW_OFFSET 48
#define Q6_HIGH_OFFSET 128
#define Q6_SCALE_OFFSET 192
#define Q6_DELTA_OFFSET 208

struct PixTensor {
    size_t rows = 0; // tokens or spatial pixels
    size_t channels = 0; // contiguous features per row
    PixTensorVals values; // row-major float activations
    PixTensor() = default;
    PixTensor(size_t count, size_t features);
};

struct PixAttentionRange {
    size_t query_begin; // first local query row, inclusive
    size_t query_end; // final local query row, exclusive
    size_t key_end; // permitted key prefix, exclusive
    bool causal; // restrict keys to the absolute query row
};

struct PixConv2D {
    size_t stride_x = 1,stride_y = 1; // spatial convolution strides
    size_t left = 0,right = 0,top = 0,bottom = 0; // independent zero-padding extents
};

void cpu_parallel(size_t work, const std::function<void(size_t,size_t)> &function);
void cpu_set_threads(size_t count);
size_t cpu_thread_count();
bool cpu_gemm(const float* weights, const float* input, float* output, size_t rows, size_t columns, size_t inner, size_t output_stride);
bool cpu_quant_matmul(const unsigned char* weights, uint32_t type, const float* input, float* output, size_t rows, size_t columns, size_t inner);
float cpu_dot(const float* a, const float* b, size_t count);
PixTensor tensor_linear(const PixModelFile &model, const std::string &name, const PixTensor &input, bool bias);
PixTensor tensor_linear_rows(const PixModelFile &model, const std::string &name, const PixTensor &input, size_t first, size_t count);
PixTensor tensor_norm(const PixTensor &input, const std::vector<float> &weight, float epsilon, bool centered = false);
std::vector<float> tensor_vector(const PixModelFile &model, const std::string &name);
void tensor_silu(PixTensor &tensor);
void tensor_gelu(PixTensor &tensor, bool approximate);
void tensor_bias(PixTensor &tensor, const std::vector<float> &bias);
PixTensor tensor_l2norm(const PixTensor &input, const std::vector<float> &weight, float epsilon);
PixTensor tensor_slice(const PixTensor &input, size_t row, size_t rows, size_t feature, size_t channels);
PixTensor tensor_gather(const PixTensor &input, const std::vector<size_t> &rows);
bool tensor_scatter(PixTensor &destination, const PixTensor &source, const std::vector<size_t> &rows);
PixTensor tensor_upsample(const PixTensor &input, size_t width, size_t height);
PixTensor tensor_resample_shortcut(const PixTensor &input, size_t width, size_t height, size_t channels, size_t temporal, bool up, bool spatial);
void tensor_add(PixTensor &destination, const PixTensor &source);
void tensor_gate(PixTensor &destination, const PixTensor &gate);
PixTensor tensor_attention(const PixTensor &q, const PixTensor &k, const PixTensor &v, size_t heads, size_t kv_heads, bool causal);
PixTensor tensor_attention(const PixTensor &q, const PixTensor &k, const PixTensor &v, size_t heads, size_t kv_heads, const std::vector<PixAttentionRange> &ranges, size_t query_offset = 0);
void tensor_rope(PixTensor &tensor, size_t heads, const std::vector<float> &cosine, const std::vector<float> &sine, bool interleaved);
PixTensor tensor_conv(const PixModelFile &model, const std::string &name, const PixTensor &input, size_t width, size_t height);
PixTensor tensor_conv(const PixModelFile &model, const std::string &name, const PixTensor &input, size_t width, size_t height, const PixConv2D &geometry, bool bias = true);
PixTensor tensor_groupnorm(const PixModelFile &model, const std::string &name, const PixTensor &input);

#endif
