/* This file is a part of 'Pixelina' Image Generator, Editor, and Viewer.
 *
 * (C) Dmitry 'sciloaf' Solovyev, 2026
 *
 * License: MIT
 */
#include <algorithm>
#include "tensor.h"
#include "inference.h"

static void pix_im2col_f32(const PixTensor &input, PixTensor &patches, size_t base, size_t width, size_t height, size_t kernel)
{
    size_t inner = input.channels * kernel * kernel;
    auto extract = [&](size_t begin, size_t end) {
        for (size_t pixel = begin; pixel < end; pixel++) {
            int64_t ioh = int64_t((base + pixel) / width),iow = int64_t((base + pixel) % width);
            float* dst_data = patches.values.data() + pixel * inner;
            for (size_t iic = 0; iic < input.channels; iic++) {
                for (size_t ikh = 0; ikh < kernel; ikh++) {
                    for (size_t ikw = 0; ikw < kernel; ikw++) {
                        const int64_t iiw = iow + int64_t(ikw) - int64_t(kernel / 2);
                        const int64_t iih = ioh + int64_t(ikh) - int64_t(kernel / 2);
                        size_t offset = iic * kernel * kernel + ikh * kernel + ikw;
                        if (iih < 0 || iih >= int64_t(height) || iiw < 0 || iiw >= int64_t(width)) dst_data[offset] = 0;
                        else dst_data[offset] = input.values[(size_t(iih) * width + size_t(iiw)) * input.channels + iic];
                    }
                }
            }
        }
    };
    cpu_parallel(patches.rows,extract);
}

PixTensor tensor_conv(const PixModelFile &model, const std::string &name, const PixTensor &input, size_t width, size_t height)
{
    if (input.values.empty()) return {};
    const PixWeight &weight = model.tensor(name + WEIGHT_SUFFIX);
#ifdef PIXELINA_WITH_VULKAN
    if (vulkan_active_state()) return vulkan_conv(model,name,input,width,height);
#endif
    size_t kernel = size_t(weight.shape[0]),inner = input.channels * kernel * kernel,channels = size_t(weight.shape[3]);
    PixWeight flattened = weight;
    flattened.shape = {inner,channels};
    PixTensor output(input.rows,channels);
    for (size_t base = 0; base < input.rows; base += COMPUTE_TILE_ROWS) {
        size_t count = std::min(size_t(COMPUTE_TILE_ROWS),input.rows - base);
        PixTensor patches(count,inner);
        pix_im2col_f32(input,patches,base,width,height,kernel);
        if (!cpu_matmul(model,flattened,patches.values.data(),count,output.values.data() + base * channels)) return {};
    }
    std::vector<float> bias = tensor_vector(model,name + BIAS_SUFFIX);
    for (size_t i = 0; i < output.values.size(); i++) output.values[i] += bias[i % channels];
    return output;
}

PixTensor tensor_conv(const PixModelFile &model, const std::string &name, const PixTensor &input, size_t width, size_t height, const PixConv2D &geometry, bool bias)
{
    if (input.values.empty() || !width || !height || width > SIZE_MAX / height || input.rows != width * height || !geometry.stride_x || !geometry.stride_y) return {};
    auto found = model.weights.find(name + WEIGHT_SUFFIX);
    if (found == model.weights.end()) return {};
    const PixWeight &weight = found->second;
    size_t rank = weight.shape.size();
    if ((rank != 4 && rank != 5) || (rank == 5 && weight.shape[2] != 1) || weight.shape[rank - 2] != input.channels) return {};
    size_t kernel_x = size_t(weight.shape[0]),kernel_y = size_t(weight.shape[1]);
    if (geometry.left > SIZE_MAX - width || geometry.right > SIZE_MAX - width - geometry.left || geometry.top > SIZE_MAX - height || geometry.bottom > SIZE_MAX - height - geometry.top) return {};
    size_t padded_width = width + geometry.left + geometry.right,padded_height = height + geometry.top + geometry.bottom;
    if (padded_width < kernel_x || padded_height < kernel_y || padded_width > INT64_MAX || padded_height > INT64_MAX) return {};
    size_t output_width = (padded_width - kernel_x) / geometry.stride_x + 1,output_height = (padded_height - kernel_y) / geometry.stride_y + 1;
    if (output_width > SIZE_MAX / output_height) return {};
#ifdef PIXELINA_WITH_VULKAN
    if (vulkan_active_state()) return vulkan_conv(model,name,input,width,height,geometry,bias);
#endif
    size_t inner = input.channels * kernel_x * kernel_y,channels = size_t(weight.shape.back());
    PixWeight flattened = weight;
    flattened.shape = {inner,channels};
    PixTensor output(output_width * output_height,channels);
    if (output.values.empty()) return {};
    for (size_t base = 0; base < output.rows; base += COMPUTE_TILE_ROWS) {
        size_t count = std::min(size_t(COMPUTE_TILE_ROWS),output.rows - base);
        PixTensor patches(count,inner);
        cpu_parallel(count,[&](size_t begin, size_t end) {
            for (size_t pixel = begin; pixel < end; pixel++) {
                int64_t x = int64_t((base + pixel) % output_width * geometry.stride_x) - int64_t(geometry.left);
                int64_t y = int64_t((base + pixel) / output_width * geometry.stride_y) - int64_t(geometry.top);
                for (size_t c = 0; c < input.channels; c++) {
                    for (size_t ky = 0; ky < kernel_y; ky++) {
                        for (size_t kx = 0; kx < kernel_x; kx++) {
                            int64_t ix = x + int64_t(kx),iy = y + int64_t(ky);
                            size_t to = pixel * inner + (c * kernel_y + ky) * kernel_x + kx;
                            patches.values[to] = ix < 0 || iy < 0 || ix >= int64_t(width) || iy >= int64_t(height)? 0 : input.values[(size_t(iy) * width + size_t(ix)) * input.channels + c];
                        }
                    }
                }
            }
        });
        if (!cpu_matmul(model,flattened,patches.values.data(),count,output.values.data() + base * channels)) return {};
    }
    if (bias) tensor_bias(output,tensor_vector(model,name + BIAS_SUFFIX));
    return output;
}
