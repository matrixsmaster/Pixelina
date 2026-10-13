/* This file is a part of 'Pixelina' Image Generator, Editor, and Viewer.
 *
 * (C) Dmitry 'sciloaf' Solovyev, 2026
 *
 * License: MIT
 */
#include <cmath>
#include <cstring>
#include <vector>
#include "tensor.h"

float half_float(uint16_t value)
{
    unsigned sign = value & HALF_SIGN_MASK;
    unsigned exponent = (value >> HALF_MANTISSA_BITS) & HALF_EXPONENT_MASK;
    unsigned mantissa = value & HALF_MANTISSA_MASK;
    float result;
    if (!exponent) result = std::ldexp(float(mantissa),1 - HALF_EXPONENT_BIAS - HALF_MANTISSA_BITS);
    else if (exponent == HALF_EXPONENT_MASK) result = mantissa? NAN : INFINITY;
    else result = std::ldexp(float(mantissa + HALF_MANTISSA_MASK + 1),int(exponent) - HALF_EXPONENT_BIAS - HALF_MANTISSA_BITS);
    return sign? -result : result;
}

void cpu_decode(const unsigned char* data, uint32_t type, size_t count, float* output)
{
    std::vector<uint64_t> shape = {count};
    weight_bytes(shape,type);
    if (type == MODEL_F32) {
        for (size_t i = 0; i < count; i++) {
            uint32_t raw = uint32_t(data[i * 4]) | (uint32_t(data[i * 4 + 1]) << 8) | (uint32_t(data[i * 4 + 2]) << 16) | (uint32_t(data[i * 4 + 3]) << 24);
            std::memcpy(output + i,&raw,4);
        }
        return;
    }
    if (type == MODEL_F16 || type == MODEL_BF16) {
        for (size_t i = 0; i < count; i++) {
            uint16_t raw = uint16_t(data[i * 2]) | (uint16_t(data[i * 2 + 1]) << 8);
            if (type == MODEL_F16) output[i] = half_float(raw);
            else {
                uint32_t bits = uint32_t(raw) << 16;
                std::memcpy(output + i,&bits,4);
            }
        }
        return;
    }
    size_t block = type == MODEL_Q8_0? MODEL_Q8_BLOCK : MODEL_QUANT_BLOCK;
    size_t bytes = type == MODEL_Q8_0? MODEL_Q8_BYTES : (type == MODEL_Q5_K? MODEL_Q5_BYTES : MODEL_Q6_BYTES);
    for (size_t base = 0; base < count; base += block,data += bytes) {
        float* decoded = output + base;
        if (type == MODEL_Q8_0) {
            float delta = half_float(uint16_t(data[0]) | (uint16_t(data[1]) << 8));
            for (unsigned i = 0; i < MODEL_Q8_BLOCK; i++) decoded[i] = int8_t(data[2 + i]) * delta;
        } else if (type == MODEL_Q5_K) {
            float delta = half_float(uint16_t(data[0]) | (uint16_t(data[1]) << 8));
            float minimum = half_float(uint16_t(data[2]) | (uint16_t(data[3]) << 8));
            const unsigned char* scales = data + 4;
            const unsigned char* high = data + Q5_HIGH_OFFSET;
            const unsigned char* low = data + Q5_LOW_OFFSET;
            for (unsigned group = 0; group < 8; group++) {
                unsigned scale,min;
                if (group < 4) {
                    scale = scales[group] & K_SCALE_MASK;
                    min = scales[group + 4] & K_SCALE_MASK;
                } else {
                    scale = (scales[group + 4] & 15) | ((scales[group - 4] >> 6) << 4);
                    min = (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4);
                }
                for (unsigned i = 0; i < MODEL_Q8_BLOCK; i++) {
                    unsigned q = low[(group / 2) * MODEL_Q8_BLOCK + i];
                    q = group % 2? q >> 4 : q & 15;
                    if (high[i] & (1 << group)) q += 16;
                    decoded[group * MODEL_Q8_BLOCK + i] = delta * scale * q - minimum * min;
                }
            }
        } else if (type == MODEL_Q6_K) {
            float delta = half_float(uint16_t(data[Q6_DELTA_OFFSET]) | (uint16_t(data[Q6_DELTA_OFFSET + 1]) << 8));
            for (unsigned half = 0; half < 2; half++) {
                const unsigned char* low = data + half * 64;
                const unsigned char* high = data + Q6_HIGH_OFFSET + half * 32;
                const int8_t* scales = reinterpret_cast<const int8_t*>(data + Q6_SCALE_OFFSET + half * 8);
                for (unsigned group = 0; group < 4; group++) {
                    for (unsigned i = 0; i < MODEL_Q8_BLOCK; i++) {
                        unsigned q = low[i + (group % 2) * 32];
                        q = group < 2? q & 15 : q >> 4;
                        q |= ((high[i] >> (group * 2)) & 3) << 4;
                        decoded[half * 128 + group * 32 + i] = delta * scales[i / 16 + group * 2] * (int(q) - 32);
                    }
                }
            }
        }
    }
}

bool cpu_matmul(const PixModelFile &file, const PixWeight &weight, const float* input, size_t columns, float* output)
{
    size_t inner = size_t(weight.shape[0]),rows = size_t(weight.shape[1]);
    uint64_t stride = weight_bytes({inner},weight.type);
    // Use borrowed native quantized dots or bounded decoded tiles; model mappings stay quantized.
#ifdef PIXELINA_WITH_VULKAN
    if (vulkan_active_state()) {
        return vulkan_matmul(file,weight,input,columns,output);
    }
#endif
    if (cpu_quant_matmul(file.data + weight.offset,weight.type,input,output,rows,columns,inner)) return true;
    if (weight.type == MODEL_F32 && cpu_gemm(reinterpret_cast<const float*>(file.data + weight.offset),input,output,rows,columns,inner,rows)) return true;
    // A bounded scratch tile feeds register-blocked kernel; model mappings stay quantized.
    size_t capacity = std::min(size_t(COMPUTE_WEIGHT_ROWS),rows);
    std::vector<float> decoded(capacity * inner);
    for (size_t base = 0; base < rows; base += capacity) {
        size_t count = std::min(capacity,rows - base);
        auto decode_rows = [&](size_t begin, size_t end) {
            for (size_t r = begin; r < end; r++) cpu_decode(file.data + weight.offset + (base + r) * stride,weight.type,inner,decoded.data() + r * inner);
        };
        cpu_parallel(count,decode_rows);
        if (cpu_gemm(decoded.data(),input,output + base,count,columns,inner,rows)) continue;
        auto run_rows = [&](size_t begin, size_t end) {
            for (size_t r = begin; r < end; r++) {
                for (size_t c = 0; c < columns; c++) output[c * rows + base + r] = cpu_dot(decoded.data() + r * inner,input + c * inner,inner);
            }
        };
        cpu_parallel(count,run_rows);
    }
    return true;
}
