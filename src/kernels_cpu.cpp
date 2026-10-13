/* This file is a part of 'Pixelina' Image Generator, Editor, and Viewer.
 *
 * (C) Dmitry 'sciloaf' Solovyev, 2026
 *
 * License: MIT
 */
#include "tensor.h"
#include <algorithm>

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))

PIX_KERNEL_AVX2 static float pix_hsum(__m256 x)
{
    __m128 sum = _mm_add_ps(_mm256_extractf128_ps(x,1),_mm256_castps256_ps128(x));
    sum = _mm_add_ps(sum,_mm_movehl_ps(sum,sum));
    sum = _mm_add_ss(sum,_mm_movehdup_ps(sum));
    return _mm_cvtss_f32(sum);
}

PIX_KERNEL_AVX2 static __m256 pix_load_partial(const float* p, int n)
{
    const __m256 index = _mm256_setr_ps(0,1,2,3,4,5,6,7);
    return _mm256_maskload_ps(p,_mm256_castps_si256(_mm256_cmp_ps(index,_mm256_set1_ps(float(n)),_CMP_LT_OQ)));
}

template<int RM, int RN> PIX_KERNEL_AVX2 static void pix_gemm_bloc(const float* A, const float* B, float* C, int64_t k, int64_t lda, int64_t ldb, int64_t ldc, int64_t ii, int64_t jj)
{
    __m256 Cv[RN][RM] = {};
    for (int64_t l = 0; l + COMPUTE_SIMD_LANES <= k; l += COMPUTE_SIMD_LANES) {
        // help compiler for op order.
        if (RM <= RN) {
            __m256 Av[RM];
            for (int64_t i = 0; i < RM; ++i) {
                Av[i] = _mm256_loadu_ps(A + lda * (ii + i) + l);
            }
            for (int64_t j = 0; j < RN; ++j) {
                __m256 Bv = _mm256_loadu_ps(B + ldb * (jj + j) + l);
                for (int64_t i = 0; i < RM; ++i) {
                    Cv[j][i] = _mm256_fmadd_ps(Av[i], Bv, Cv[j][i]);
                }
            }
        } else {
            __m256 Bv[RN];
            for (int64_t j = 0; j < RN; ++j) {
                Bv[j] = _mm256_loadu_ps(B + ldb * (jj + j) + l);
            }
            for (int64_t i = 0; i < RM; ++i) {
                __m256 Av = _mm256_loadu_ps(A + lda * (ii + i) + l);
                for (int64_t j = 0; j < RN; ++j) {
                    Cv[j][i] = _mm256_fmadd_ps(Av, Bv[j], Cv[j][i]);
                }
            }
        }
    }
    const int64_t rem = k % COMPUTE_SIMD_LANES;
    if (rem != 0) {
        __m256 Av[RM];
        for (int64_t i = 0; i < RM; ++i) {
            Av[i] = pix_load_partial(A + lda * (ii + i) + k - rem, rem);
        }
        for (int64_t j = 0; j < RN; ++j) {
            __m256 Bv = pix_load_partial(B + ldb * (jj + j) + k - rem, rem);
            for (int64_t i = 0; i < RM; ++i) {
                Cv[j][i] = _mm256_fmadd_ps(Av[i], Bv, Cv[j][i]);
            }
        }
    }
    for (int64_t j = 0; j < RN; ++j) {
        for (int64_t i = 0; i < RM; ++i) C[ldc * (jj + j) + (ii + i)] = pix_hsum(Cv[j][i]);
    }
}

template<int RM> static void pix_gemm_columns(const float* weights, const float* input, float* output, size_t inner, size_t stride, size_t row, size_t column, size_t count)
{
    if (count == 1) pix_gemm_bloc<RM,1>(weights,input,output,inner,inner,inner,stride,row,column);
    else if (count == 2) pix_gemm_bloc<RM,2>(weights,input,output,inner,inner,inner,stride,row,column);
    else if (count == 3) pix_gemm_bloc<RM,3>(weights,input,output,inner,inner,inner,stride,row,column);
}
#endif

bool cpu_gemm(const float* weights, const float* input, float* output, size_t rows, size_t columns, size_t inner, size_t output_stride)
{
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    static bool supported = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    if (!supported) return false;
    size_t row_tiles = (rows + COMPUTE_MATRIX_ROWS - 1) / COMPUTE_MATRIX_ROWS;
    size_t column_tiles = (columns + COMPUTE_MATRIX_COLUMNS - 1) / COMPUTE_MATRIX_COLUMNS;
    auto run = [&](size_t begin, size_t end) {
        for (size_t tile = begin; tile < end; tile++) {
            size_t row = tile % row_tiles * COMPUTE_MATRIX_ROWS,column = tile / row_tiles * COMPUTE_MATRIX_COLUMNS;
            size_t row_count = std::min(size_t(COMPUTE_MATRIX_ROWS),rows - row),column_count = std::min(size_t(COMPUTE_MATRIX_COLUMNS),columns - column);
            if (row_count == 1) pix_gemm_columns<1>(weights,input,output,inner,output_stride,row,column,column_count);
            else if (row_count == 2) pix_gemm_columns<2>(weights,input,output,inner,output_stride,row,column,column_count);
            else if (row_count == 3) pix_gemm_columns<3>(weights,input,output,inner,output_stride,row,column,column_count);
            else pix_gemm_columns<COMPUTE_MATRIX_ROWS>(weights,input,output,inner,output_stride,row,column,column_count);
        }
    };
    cpu_parallel(row_tiles * column_tiles,run);
    return true;
#else
    (void)weights;
    (void)input;
    (void)output;
    (void)rows;
    (void)columns;
    (void)inner;
    (void)output_stride;
    return false;
#endif
}
