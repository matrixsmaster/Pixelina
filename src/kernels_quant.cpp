/* This file is a part of 'Pixelina' Image Generator, Editor, and Viewer.
 *
 * (C) Dmitry 'sciloaf' Solovyev, 2026
 *
 * License: MIT
 */
#include "kernels_quant.h"
#include <algorithm>
#include <cmath>
#include <cstring>

#define MM256_SET_M128I(a,b) _mm256_insertf128_si256(_mm256_castsi128_si256(b),(a),1)

struct block_q8_0 {
    uint16_t d; // fp16 block scale
    int8_t qs[QK8_0]; // signed eight-bit weights or activations
};
struct block_q5_K {
    uint16_t d,dmin; // fp16 super-block scales
    uint8_t scales[K_SCALE_SIZE]; // packed six-bit scales and minima
    uint8_t qh[QK_K / 8]; // fifth quantization bit
    uint8_t qs[QK_K / 2]; // packed low nibbles
};
struct block_q6_K {
    uint8_t ql[QK_K / 2]; // low four bits
    uint8_t qh[QK_K / 4]; // high two bits
    int8_t scales[QK_K / PIX_QUANT_SUM_GROUP]; // signed sub-block scales
    uint16_t d; // fp16 super-block scale
};
struct block_q8_K {
    float d; // fp32 activation scale
    int8_t qs[QK_K]; // signed quantized activations
    int16_t bsums[QK_K / PIX_QUANT_SUM_GROUP]; // activation sums used for weight minima
};
static_assert(sizeof(block_q8_0) == MODEL_Q8_BYTES,"Q8_0 layout mismatch");
static_assert(sizeof(block_q5_K) == MODEL_Q5_BYTES,"Q5_K layout mismatch");
static_assert(sizeof(block_q6_K) == MODEL_Q6_BYTES,"Q6_K layout mismatch");

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
static inline int nearest_int(float fval)
{

    float val = fval + PIX_ROUND_BIAS;
    int i;
    memcpy(&i, &val, sizeof(int));
    return (i & PIX_ROUND_MASK) - PIX_ROUND_OFFSET;
}

PIX_QUANT_AVX2 static inline __m256i get_scale_shuffle_k4(int i)
{
    static const uint8_t k_shuffle[QK_K] = {
         0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
         2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
         4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
         6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
         8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
        10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
        12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,
        14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15
    };
    return _mm256_loadu_si256((const __m256i*)k_shuffle + i);
}

PIX_QUANT_AVX2 static inline __m128i get_scale_shuffle(int i)
{
    static const uint8_t k_shuffle[QK_K / 2] = {
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
         2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
         4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,
         6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
         8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9,
        10,10,10,10,10,10,10,10, 11,11,11,11,11,11,11,11,
        12,12,12,12,12,12,12,12, 13,13,13,13,13,13,13,13,
        14,14,14,14,14,14,14,14, 15,15,15,15,15,15,15,15
    };
    return _mm_loadu_si128((const __m128i*)k_shuffle + i);
}

PIX_QUANT_AVX2 static void quantize_row_q8_K_ref(const float* x, block_q8_K* y, int64_t k)
{

    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        float max = 0;
        float amax = 0;
        for (int j = 0; j < QK_K; ++j) {
            float ax = fabsf(x[j]);
            if (ax > amax) {
                amax = ax;
                max = x[j];
            }
        }
        if (!amax) {
            y[i].d = 0;
            memset(y[i].qs,0,QK_K);
            memset(y[i].bsums,0,sizeof(y[i].bsums));
            x += QK_K;
            continue;
        }
        const float iscale = -float(PIX_QUANT_BYTE_MAX) / max;
        for (int j = 0; j < QK_K; ++j) {
            int v = nearest_int(iscale*x[j]);
            y[i].qs[j] = std::min(PIX_QUANT_BYTE_MAX,v);
        }
        for (int j = 0; j < QK_K / PIX_QUANT_SUM_GROUP; ++j) {
            int sum = 0;
            for (int ii = 0; ii < PIX_QUANT_SUM_GROUP; ++ii) {
                sum += y[i].qs[j * PIX_QUANT_SUM_GROUP + ii];
            }
            y[i].bsums[j] = sum;
        }
        y[i].d = 1/iscale;
        x += QK_K;
    }
}

PIX_QUANT_AVX2 static void quantize_row_q8_0_ref(const float* x, block_q8_0* y, int64_t k)
{

    const int nb = k / QK8_0;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max

        for (int j = 0; j < QK8_0; j++) {
            const float v = x[i*QK8_0 + j];
            amax = std::max(amax,std::fabs(v));
        }

        const float d = amax / PIX_QUANT_BYTE_MAX;
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = _cvtss_sh(d,_MM_FROUND_TO_NEAREST_INT);

        for (int j = 0; j < QK8_0; ++j) {
            const float x0 = x[i*QK8_0 + j]*id;

            y[i].qs[j] = roundf(x0);
        }
    }
}

PIX_QUANT_AVX2 static float pix_quant_hsum(__m256 x)
{
    __m128 res = _mm256_extractf128_ps(x,1);
    res = _mm_add_ps(res,_mm256_castps256_ps128(x));
    res = _mm_add_ps(res,_mm_movehl_ps(res,res));
    res = _mm_add_ss(res,_mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

PIX_QUANT_AVX2 static __m256 pix_quant_mul_pairs(__m256i x, __m256i y)
{
    const __m256i ax = _mm256_sign_epi8(x,x);
    const __m256i sy = _mm256_sign_epi8(y,x);
    const __m256i dot = _mm256_maddubs_epi16(ax,sy);
    return _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_set1_epi16(1),dot));
}

PIX_QUANT_AVX2 static float pix_dot_q5_K(int n, const block_q5_K* x, const block_q8_K* y)
{

    const int nb = n / QK_K;

    static const uint32_t kmask1 = PIX_K_SCALE_MASK;
    static const uint32_t kmask2 = PIX_K_NIBBLE_MASK;
    static const uint32_t kmask3 = PIX_K_HIGH_MASK;

    uint32_t utmp[PIX_QUANT_SCALE_WORDS];

    const __m256i m4 = _mm256_set1_epi8(PIX_QUANT_NIBBLE_MASK);
    const __m128i mzero = _mm_setzero_si128();
    const __m256i mone  = _mm256_set1_epi8(1);

    __m256 acc = _mm256_setzero_ps();

    float summs = 0.f;

    for (int i = 0; i < nb; ++i) {
        const uint8_t* q5 = x[i].qs;
        const int8_t* q8 = y[i].qs;

        const float d = y[i].d * _cvtsh_ss(x[i].d);
        const float dmin = -y[i].d * _cvtsh_ss(x[i].dmin);

        memcpy(utmp, x[i].scales,K_SCALE_SIZE);
        utmp[3] = ((utmp[2] >> PIX_QUANT_NIBBLE_BITS) & kmask2) | (((utmp[1] >> PIX_QUANT_SCALE_BITS) & kmask3) << PIX_QUANT_NIBBLE_BITS);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> PIX_QUANT_SCALE_BITS) & kmask3) << PIX_QUANT_NIBBLE_BITS);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]));

        const __m256i q8sums = _mm256_loadu_si256((const __m256i*)y[i].bsums);
        const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
        const __m128i prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
        const __m128i hsum = _mm_hadd_epi32(_mm_hadd_epi32(prod, mzero), mzero);
        summs += dmin * _mm_extract_epi32(hsum, 0);

        const __m128i sc128  = _mm256_extracti128_si256(mins_and_scales, 0);
        const __m256i scales = MM256_SET_M128I(sc128, sc128);

        const __m256i hbits = _mm256_loadu_si256((const __m256i*)x[i].qh);
        __m256i hmask = mone;

        __m256i sumi = _mm256_setzero_si256();

        int bit = 0;

        for (int j = 0; j < QK_K / PIX_QUANT_Q5_CHUNK; ++j) {

            const __m256i scale_0 = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2*j+0));
            const __m256i scale_1 = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2*j+1));

            const __m256i q5bits = _mm256_loadu_si256((const __m256i*)q5);
            q5 += PIX_QUANT_SIMD_BYTES;

            const __m256i q5l_0 = _mm256_and_si256(q5bits, m4);
            const __m256i q5h_0 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), bit++),PIX_QUANT_NIBBLE_BITS);
            const __m256i q5_0  = _mm256_add_epi8(q5l_0, q5h_0);
            hmask = _mm256_slli_epi16(hmask, 1);

            const __m256i q5l_1 = _mm256_and_si256(_mm256_srli_epi16(q5bits,PIX_QUANT_NIBBLE_BITS), m4);
            const __m256i q5h_1 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), bit++),PIX_QUANT_NIBBLE_BITS);
            const __m256i q5_1  = _mm256_add_epi8(q5l_1, q5h_1);
            hmask = _mm256_slli_epi16(hmask, 1);

            const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)q8);
            q8 += PIX_QUANT_SIMD_BYTES;
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8);
            q8 += PIX_QUANT_SIMD_BYTES;

            __m256i p16_0 = _mm256_maddubs_epi16(q5_0, q8_0);
            __m256i p16_1 = _mm256_maddubs_epi16(q5_1, q8_1);

            p16_0 = _mm256_madd_epi16(scale_0, p16_0);
            p16_1 = _mm256_madd_epi16(scale_1, p16_1);

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));

        }

        __m256 vd = _mm256_set1_ps(d);
        acc = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(sumi), acc);

    }

    return pix_quant_hsum(acc) + summs;

}

PIX_QUANT_AVX2 static float pix_dot_q6_K(int n, const block_q6_K* x, const block_q8_K* y)
{

    const int nb = n / QK_K;

    const __m256i m3 = _mm256_set1_epi8(PIX_QUANT_PAIR_MASK);
    const __m256i m15 = _mm256_set1_epi8(PIX_QUANT_NIBBLE_MASK);

    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * _cvtsh_ss(x[i].d);

        const uint8_t* q4 = x[i].ql;
        const uint8_t* qh = x[i].qh;
        const int8_t* q8 = y[i].qs;

        const __m256i q8sums = _mm256_loadu_si256((const __m256i*)y[i].bsums);
        const __m128i scales = _mm_loadu_si128((const __m128i*)x[i].scales);
        const __m256i scales_16 = _mm256_cvtepi8_epi16(scales);
        const __m256i q8sclsub = _mm256_slli_epi32(_mm256_madd_epi16(q8sums, scales_16),PIX_QUANT_Q6_SIGN_SHIFT);

        __m256i sumi = _mm256_setzero_si256();

        int is = 0;

        for (int j = 0; j < QK_K / PIX_QUANT_Q6_CHUNK; ++j) {
            const __m256i q4bits1 = _mm256_loadu_si256((const __m256i*)q4);
            q4 += PIX_QUANT_SIMD_BYTES;
            const __m256i q4bits2 = _mm256_loadu_si256((const __m256i*)q4);
            q4 += PIX_QUANT_SIMD_BYTES;
            const __m256i q4bitsH = _mm256_loadu_si256((const __m256i*)qh);
            qh += PIX_QUANT_SIMD_BYTES;

            const __m256i q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH,m3),PIX_QUANT_NIBBLE_BITS);
            const __m256i q4h_1 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, _mm256_set1_epi8(PIX_QUANT_Q6_HIGH_ONE)), 2);
            const __m256i q4h_2 = _mm256_and_si256(q4bitsH, _mm256_set1_epi8(PIX_QUANT_Q6_HIGH_TWO));
            const __m256i q4h_3 = _mm256_srli_epi16(_mm256_and_si256(q4bitsH, _mm256_set1_epi8(PIX_QUANT_Q6_HIGH_THREE)), 2);

            const __m256i q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m15), q4h_0);
            const __m256i q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m15), q4h_1);
            const __m256i q4_2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1,PIX_QUANT_NIBBLE_BITS), m15), q4h_2);
            const __m256i q4_3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2,PIX_QUANT_NIBBLE_BITS), m15), q4h_3);

            const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)q8);
            q8 += PIX_QUANT_SIMD_BYTES;
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8);
            q8 += PIX_QUANT_SIMD_BYTES;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i*)q8);
            q8 += PIX_QUANT_SIMD_BYTES;
            const __m256i q8_3 = _mm256_loadu_si256((const __m256i*)q8);
            q8 += PIX_QUANT_SIMD_BYTES;

            __m256i p16_0 = _mm256_maddubs_epi16(q4_0, q8_0);
            __m256i p16_1 = _mm256_maddubs_epi16(q4_1, q8_1);
            __m256i p16_2 = _mm256_maddubs_epi16(q4_2, q8_2);
            __m256i p16_3 = _mm256_maddubs_epi16(q4_3, q8_3);

            const __m128i scale_0 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 0));
            const __m128i scale_1 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 1));
            const __m128i scale_2 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 2));
            const __m128i scale_3 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 3));
            is += 4;

            p16_0 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_0), p16_0);
            p16_1 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_1), p16_1);
            p16_2 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_2), p16_2);
            p16_3 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_3), p16_3);

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_2, p16_3));

        }

        sumi = _mm256_sub_epi32(sumi, q8sclsub);
        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
    }

    return pix_quant_hsum(acc);

}

PIX_QUANT_AVX2 static float pix_dot_q8_0(int n, const block_q8_0* x, const block_q8_0* y)
{

    const int qk = QK8_0;
    const int nb = n / qk;

    int ib = 0;

    // Initialize accumulator with zeros
    __m256 acc = _mm256_setzero_ps();

    // Main loop
    for (; ib < nb; ++ib) {
        // Compute combined scale for the block
        const __m256 d = _mm256_set1_ps(_cvtsh_ss(x[ib].d) * _cvtsh_ss(y[ib].d));
        __m256i qx = _mm256_loadu_si256((const __m256i *)x[ib].qs);
        __m256i qy = _mm256_loadu_si256((const __m256i *)y[ib].qs);

        const __m256 q = pix_quant_mul_pairs(qx, qy);

        // Multiply q with scale and accumulate
        acc = _mm256_fmadd_ps( d, q, acc );
    }

    return pix_quant_hsum(acc);

}

#endif

bool cpu_quant_matmul(const unsigned char* weights, uint32_t type, const float* input, float* output, size_t rows, size_t columns, size_t inner)
{
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    static bool supported = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c");
    if (!supported || (type != MODEL_Q8_0 && type != MODEL_Q5_K && type != MODEL_Q6_K)) return false;
    size_t block_size = type == MODEL_Q8_0? QK8_0 : QK_K;
    if (!rows || !columns || !inner || inner % block_size || inner > INT32_MAX) return false;
    size_t blocks = inner / block_size;
    if (blocks > SIZE_MAX / columns) return false;
    std::vector<block_q8_0> q8;
    std::vector<block_q8_K> qk;
    if (type == MODEL_Q8_0) q8.resize(blocks * columns);
    else qk.resize(blocks * columns);
    auto quantize = [&](size_t begin, size_t end) {
        for (size_t c = begin; c < end; c++) {
            if (type == MODEL_Q8_0) quantize_row_q8_0_ref(input + c * inner,q8.data() + c * blocks,inner);
            else quantize_row_q8_K_ref(input + c * inner,qk.data() + c * blocks,inner);
        }
    };
    cpu_parallel(columns,quantize);
    auto multiply = [&](size_t begin, size_t end) {
        for (size_t r = begin; r < end; r++) {
            for (size_t c = 0; c < columns; c++) {
                float value;
                if (type == MODEL_Q8_0) value = pix_dot_q8_0(int(inner),reinterpret_cast<const block_q8_0*>(weights) + r * blocks,q8.data() + c * blocks);
                else if (type == MODEL_Q5_K) value = pix_dot_q5_K(int(inner),reinterpret_cast<const block_q5_K*>(weights) + r * blocks,qk.data() + c * blocks);
                else value = pix_dot_q6_K(int(inner),reinterpret_cast<const block_q6_K*>(weights) + r * blocks,qk.data() + c * blocks);
                output[c * rows + r] = value;
            }
        }
    };
    cpu_parallel(rows,multiply);
    return true;
#else
    (void)weights;
    (void)type;
    (void)input;
    (void)output;
    (void)rows;
    (void)columns;
    (void)inner;
    return false;
#endif
}
