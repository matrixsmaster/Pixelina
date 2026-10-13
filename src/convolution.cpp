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
