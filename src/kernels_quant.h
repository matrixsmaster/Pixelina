/* This file is a part of 'Pixelina' Image Generator, Editor, and Viewer.
 *
 * (C) Dmitry 'sciloaf' Solovyev, 2026
 *
 * License: MIT
 */
#ifndef PIXELINA_KERNELS_QUANT_H
#define PIXELINA_KERNELS_QUANT_H

#include "tensor.h"

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define PIX_QUANT_AVX2 __attribute__((target("avx2,fma,f16c")))
#endif

#define QK_K MODEL_QUANT_BLOCK
#define QK8_0 MODEL_Q8_BLOCK
#define K_SCALE_SIZE 12
#define PIX_QUANT_BYTE_MAX 127
#define PIX_QUANT_SUM_GROUP 16
#define PIX_QUANT_SIMD_BYTES 32
#define PIX_QUANT_Q5_CHUNK 64
#define PIX_QUANT_Q6_CHUNK 128
#define PIX_QUANT_SCALE_WORDS 4
#define PIX_QUANT_NIBBLE_BITS 4
#define PIX_QUANT_SCALE_BITS 6
#define PIX_QUANT_Q6_SIGN_SHIFT 5
#define PIX_QUANT_NIBBLE_MASK 15
#define PIX_QUANT_PAIR_MASK 3
#define PIX_QUANT_Q6_HIGH_ONE 12
#define PIX_QUANT_Q6_HIGH_TWO 48
#define PIX_QUANT_Q6_HIGH_THREE (-64)
#define PIX_K_SCALE_MASK 0x3f3f3f3f
#define PIX_K_NIBBLE_MASK 0x0f0f0f0f
#define PIX_K_HIGH_MASK 0x03030303
#define PIX_ROUND_BIAS 12582912.0f
#define PIX_ROUND_MASK 0x007fffff
#define PIX_ROUND_OFFSET 0x00400000

#endif
