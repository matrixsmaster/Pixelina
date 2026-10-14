/* This file is a part of 'Pixelina' Image Generator, Editor, and Viewer.
 *
 * (C) Dmitry 'sciloaf' Solovyev, 2026
 *
 * License: MIT
 */
#include "qwen_vae.h"
#include <algorithm>
#include <cmath>

static bool qwen_vae_weight(const PixModelFile &vae, const std::string &name, const std::vector<uint64_t> &shape, std::string &error)
{
    auto found = vae.weights.find(name);
    if (found != vae.weights.end() && found->second.shape == shape && weight_bytes(shape,found->second.type)) return true;
    error = "Missing or incompatible Qwen Image 2.1 VAE tensor: " + name;
    return false;
}

static bool qwen_vae_conv_valid(const PixModelFile &vae, const std::string &name, size_t input, size_t output, size_t kernel, bool spatial, std::string &error)
{
    std::vector<uint64_t> shape = spatial? std::vector<uint64_t>{kernel,kernel,input,output} : std::vector<uint64_t>{kernel,kernel,1,input,output};
    return qwen_vae_weight(vae,name + WEIGHT_SUFFIX,shape,error) && qwen_vae_weight(vae,name + BIAS_SUFFIX,{output},error);
}

static bool qwen_vae_block_valid(const PixModelFile &vae, const std::string &name, size_t input, size_t output, std::string &error)
{
    if (!qwen_vae_weight(vae,name + QWEN_VAE_NORM1,{1,1,1,input},error) || !qwen_vae_weight(vae,name + QWEN_VAE_NORM2,{1,1,1,output},error)) return false;
    if (!qwen_vae_conv_valid(vae,name + QWEN_VAE_CONV1,input,output,QWEN_VAE_KERNEL,false,error) || !qwen_vae_conv_valid(vae,name + QWEN_VAE_CONV2,output,output,QWEN_VAE_KERNEL,false,error)) return false;
    return input == output || qwen_vae_conv_valid(vae,name + QWEN_VAE_SHORTCUT,input,output,1,false,error);
}

bool qwen_vae_validate(const PixModelFile &vae, std::string &error)
{
    size_t multipliers[] = QWEN_VAE_MULTIPLIERS;
    for (size_t component = 0; component < 2; component++) {
        bool decode = component != 0;
        std::string prefix = decode? QWEN_VAE_DECODER : QWEN_VAE_ENCODER;
        size_t base = decode? QWEN_VAE_DECODER_BASE : QWEN_VAE_ENCODER_BASE;
        size_t input = decode? base * multipliers[QWEN_VAE_LEVELS - 1] : base;
        if (!qwen_vae_conv_valid(vae,prefix + QWEN_VAE_INPUT_CONV,decode? QWEN_VAE_CHANNELS : QWEN_VAE_RGBA,input,QWEN_VAE_KERNEL,false,error)) return false;
        size_t middle = base * multipliers[QWEN_VAE_LEVELS - 1];
        if (!qwen_vae_block_valid(vae,prefix + QWEN_VAE_MIDDLE_FIRST,middle,middle,error) || !qwen_vae_block_valid(vae,prefix + QWEN_VAE_MIDDLE_SECOND,middle,middle,error)) return false;
        if (!qwen_vae_weight(vae,prefix + QWEN_VAE_MIDDLE_NORM,{1,1,middle},error) || !qwen_vae_conv_valid(vae,prefix + QWEN_VAE_MIDDLE_QKV,middle,middle * 3,1,true,error) || !qwen_vae_conv_valid(vae,prefix + QWEN_VAE_MIDDLE_PROJECTION,middle,middle,1,true,error)) return false;
        for (size_t level = 0; level < QWEN_VAE_LEVELS; level++) {
            size_t output = base * multipliers[decode? QWEN_VAE_LEVELS - 1 - level : level];
            std::string level_prefix = prefix + (decode? QWEN_VAE_UP_LEVEL_PREFIX : QWEN_VAE_DOWN_LEVEL_PREFIX) + std::to_string(level) + (decode? QWEN_VAE_UP_BLOCK_PREFIX : QWEN_VAE_DOWN_BLOCK_PREFIX);
            size_t count = decode? QWEN_VAE_UP_BLOCKS : QWEN_VAE_BLOCKS;
            for (size_t block = 0; block < count; block++) {
                if (!qwen_vae_block_valid(vae,level_prefix + std::to_string(block),input,output,error)) return false;
                input = output;
            }
            if (level == QWEN_VAE_LEVELS - 1) continue;
            std::string resample = level_prefix + std::to_string(count);
            if (!qwen_vae_conv_valid(vae,resample + QWEN_VAE_RESAMPLE,output,output,QWEN_VAE_KERNEL,true,error)) return false;
            bool temporal = decode? level < QWEN_VAE_LEVELS - 2 : level != 0;
            if (temporal && !qwen_vae_conv_valid(vae,resample + QWEN_VAE_TIME_CONV,output,decode? output * 2 : output,1,false,error)) return false;
        }
        if (!qwen_vae_weight(vae,prefix + QWEN_VAE_HEAD_NORM,{1,1,1,input},error) || !qwen_vae_conv_valid(vae,prefix + QWEN_VAE_HEAD_CONV,input,decode? QWEN_VAE_RGBA : QWEN_VAE_CHANNELS * 2,QWEN_VAE_KERNEL,false,error)) return false;
    }
    if (!qwen_vae_conv_valid(vae,QWEN_VAE_INPUT_CONV,QWEN_VAE_CHANNELS * 2,QWEN_VAE_CHANNELS * 2,1,false,error) || !qwen_vae_conv_valid(vae,QWEN_VAE_POST_QUANT_CONV,QWEN_VAE_CHANNELS,QWEN_VAE_CHANNELS,1,false,error)) return false;
    for (size_t channel = 0; channel < QWEN_VAE_CHANNELS; channel++) {
        if (!std::isfinite(qwen_vae_mean[channel]) || !std::isfinite(qwen_vae_std[channel]) || qwen_vae_std[channel] <= 0) return false;
    }
    return true;
}

static PixTensor qwen_vae_convolution(const PixModelFile &vae, const std::string &name, const PixTensor &input, size_t width, size_t height)
{
    const PixWeight &weight = vae.tensor(name + WEIGHT_SUFFIX);
    PixConv2D geometry;
    geometry.left = geometry.right = size_t(weight.shape[0] / 2);
    geometry.top = geometry.bottom = size_t(weight.shape[1] / 2);
    return tensor_conv(vae,name,input,width,height,geometry);
}

static PixTensor qwen_vae_block(const PixModelFile &vae, const std::string &name, const PixTensor &input, size_t width, size_t height)
{
    PixTensor hidden = tensor_l2norm(input,tensor_vector(vae,name + QWEN_VAE_NORM1),QWEN_VAE_L2_EPSILON);
    tensor_silu(hidden);
    hidden = qwen_vae_convolution(vae,name + QWEN_VAE_CONV1,hidden,width,height);
    hidden = tensor_l2norm(hidden,tensor_vector(vae,name + QWEN_VAE_NORM2),QWEN_VAE_L2_EPSILON);
    tensor_silu(hidden);
    hidden = qwen_vae_convolution(vae,name + QWEN_VAE_CONV2,hidden,width,height);
    if (input.channels == hidden.channels) tensor_add(hidden,input);
    else tensor_add(hidden,qwen_vae_convolution(vae,name + QWEN_VAE_SHORTCUT,input,width,height));
    return hidden;
}

static PixTensor qwen_vae_middle(const PixModelFile &vae, const std::string &prefix, PixTensor hidden, size_t width, size_t height)
{
    hidden = qwen_vae_block(vae,prefix + QWEN_VAE_MIDDLE_FIRST,hidden,width,height);
    PixTensor normalized = tensor_l2norm(hidden,tensor_vector(vae,prefix + QWEN_VAE_MIDDLE_NORM),QWEN_VAE_L2_EPSILON);
    PixTensor qkv = qwen_vae_convolution(vae,prefix + QWEN_VAE_MIDDLE_QKV,normalized,width,height);
    normalized = PixTensor();
    PixTensor q = tensor_slice(qkv,0,qkv.rows,0,hidden.channels),k = tensor_slice(qkv,0,qkv.rows,hidden.channels,hidden.channels),v = tensor_slice(qkv,0,qkv.rows,hidden.channels * 2,hidden.channels);
    qkv = PixTensor();
    PixTensor attended = tensor_attention(q,k,v,1,1,std::vector<PixAttentionRange>{{0,q.rows,k.rows,false}});
    q = k = v = PixTensor();
    tensor_add(hidden,qwen_vae_convolution(vae,prefix + QWEN_VAE_MIDDLE_PROJECTION,attended,width,height));
    attended = PixTensor();
    return qwen_vae_block(vae,prefix + QWEN_VAE_MIDDLE_SECOND,hidden,width,height);
}

PixTensor qwen_vae_encode(const PixModelFile &vae, const PixImage &image, const InferenceProgress &progress)
{
    if (!image_valid(image) || image.width % QWEN_VAE_FACTOR || image.height % QWEN_VAE_FACTOR) return {};
    size_t width = size_t(image.width),height = size_t(image.height);
    PixTensor hidden(width * height,QWEN_VAE_RGBA);
    for (size_t pixel = 0; pixel < hidden.rows; pixel++) {
        for (size_t c = 0; c < 3; c++) hidden.values[pixel * QWEN_VAE_RGBA + c] = image.rgb[pixel * 3 + c] * QWEN_VAE_PIXEL_SCALE * 2 - 1;
        hidden.values[pixel * QWEN_VAE_RGBA + 3] = image.alpha.empty()? 1 : image.alpha[pixel] * QWEN_VAE_PIXEL_SCALE * 2 - 1;
    }
    hidden = qwen_vae_convolution(vae,QWEN_VAE_ENCODER QWEN_VAE_INPUT_CONV,hidden,width,height);
    for (size_t level = 0; level < QWEN_VAE_LEVELS; level++) {
        std::string prefix = QWEN_VAE_ENCODER QWEN_VAE_DOWN_LEVEL_PREFIX + std::to_string(level) + QWEN_VAE_DOWN_BLOCK_PREFIX;
        bool spatial = level != QWEN_VAE_LEVELS - 1;
        PixTensor shortcut = tensor_resample_shortcut(hidden,width,height,size_t(vae.tensor(prefix + "0" QWEN_VAE_CONV1 WEIGHT_SUFFIX).shape.back()),level && spatial? 2 : 1,false,spatial);
        for (size_t block = 0; block < QWEN_VAE_BLOCKS; block++) hidden = qwen_vae_block(vae,prefix + std::to_string(block),hidden,width,height);
        if (spatial) {
            PixConv2D geometry;
            geometry.stride_x = geometry.stride_y = 2;
            geometry.right = geometry.bottom = 1;
            hidden = tensor_conv(vae,prefix + std::to_string(QWEN_VAE_BLOCKS) + QWEN_VAE_RESAMPLE,hidden,width,height,geometry);
            width /= 2;
            height /= 2;
        }
        tensor_add(hidden,shortcut);
        if (hidden.values.empty()) return {};
        if (progress && !progress(QWEN_VAE_ENCODE_BEGIN + QWEN_VAE_ENCODE_SPAN * float(level + 1) / QWEN_VAE_LEVELS,"VAE encoder level " + std::to_string(level + 1))) return {};
    }
    hidden = qwen_vae_middle(vae,QWEN_VAE_ENCODER,std::move(hidden),width,height);
    hidden = tensor_l2norm(hidden,tensor_vector(vae,QWEN_VAE_ENCODER QWEN_VAE_HEAD_NORM),QWEN_VAE_L2_EPSILON);
    tensor_silu(hidden);
    hidden = qwen_vae_convolution(vae,QWEN_VAE_ENCODER QWEN_VAE_HEAD_CONV,hidden,width,height);
    hidden = qwen_vae_convolution(vae,QWEN_VAE_INPUT_CONV,hidden,width,height);
    hidden = tensor_slice(hidden,0,hidden.rows,0,QWEN_VAE_CHANNELS);
#ifdef PIXELINA_WITH_VULKAN
    if (!vulkan_synchronize()) return {};
#endif
    for (size_t i = 0; i < hidden.values.size(); i++) hidden.values[i] = (hidden.values[i] - qwen_vae_mean[i % QWEN_VAE_CHANNELS]) / qwen_vae_std[i % QWEN_VAE_CHANNELS];
    return hidden;
}

PixImage qwen_vae_decode(const PixModelFile &vae, PixTensor latents, size_t width, size_t height, const InferenceProgress &progress)
{
    if (!width || !height || width > SIZE_MAX / height || latents.rows != width * height || latents.channels != QWEN_VAE_CHANNELS || latents.values.empty() || width > INT32_MAX / QWEN_VAE_FACTOR || height > INT32_MAX / QWEN_VAE_FACTOR || width * height > PIXELINA_MAX_PIXELS / (QWEN_VAE_FACTOR * QWEN_VAE_FACTOR)) return {};
#ifdef PIXELINA_WITH_VULKAN
    if (!vulkan_synchronize()) return {};
#endif
    for (size_t i = 0; i < latents.values.size(); i++) latents.values[i] = latents.values[i] * qwen_vae_std[i % QWEN_VAE_CHANNELS] + qwen_vae_mean[i % QWEN_VAE_CHANNELS];
    PixTensor hidden = qwen_vae_convolution(vae,QWEN_VAE_POST_QUANT_CONV,latents,width,height);
    latents = PixTensor();
    hidden = qwen_vae_convolution(vae,QWEN_VAE_DECODER QWEN_VAE_INPUT_CONV,hidden,width,height);
    hidden = qwen_vae_middle(vae,QWEN_VAE_DECODER,std::move(hidden),width,height);
    for (size_t level = 0; level < QWEN_VAE_LEVELS; level++) {
        std::string prefix = QWEN_VAE_DECODER QWEN_VAE_UP_LEVEL_PREFIX + std::to_string(level) + QWEN_VAE_UP_BLOCK_PREFIX;
        bool spatial = level != QWEN_VAE_LEVELS - 1;
        PixTensor shortcut;
        if (spatial) shortcut = tensor_resample_shortcut(hidden,width,height,size_t(vae.tensor(prefix + "0" QWEN_VAE_CONV1 WEIGHT_SUFFIX).shape.back()),level < QWEN_VAE_LEVELS - 2? 2 : 1,true,true);
        for (size_t block = 0; block < QWEN_VAE_UP_BLOCKS; block++) hidden = qwen_vae_block(vae,prefix + std::to_string(block),hidden,width,height);
        if (spatial) {
            hidden = tensor_upsample(hidden,width,height);
            width *= 2;
            height *= 2;
            hidden = qwen_vae_convolution(vae,prefix + std::to_string(QWEN_VAE_UP_BLOCKS) + QWEN_VAE_RESAMPLE,hidden,width,height);
            tensor_add(hidden,shortcut);
        }
        if (hidden.values.empty()) return {};
        if (progress && !progress(QWEN_VAE_DECODE_BEGIN + QWEN_VAE_DECODE_SPAN * float(level + 1) / QWEN_VAE_LEVELS,"VAE decoder level " + std::to_string(level + 1))) return {};
    }
    hidden = tensor_l2norm(hidden,tensor_vector(vae,QWEN_VAE_DECODER QWEN_VAE_HEAD_NORM),QWEN_VAE_L2_EPSILON);
    tensor_silu(hidden);
    hidden = qwen_vae_convolution(vae,QWEN_VAE_DECODER QWEN_VAE_HEAD_CONV,hidden,width,height);
#ifdef PIXELINA_WITH_VULKAN
    if (!vulkan_synchronize()) return {};
#endif
    if (hidden.values.empty() || width > INT32_MAX || height > INT32_MAX || width * height > PIXELINA_MAX_PIXELS) return {};
    PixImage output;
    output.width = int(width);
    output.height = int(height);
    output.rgb.resize(width * height * 3);
    output.alpha.resize(width * height);
    for (size_t pixel = 0; pixel < hidden.rows; pixel++) {
        for (size_t c = 0; c < QWEN_VAE_RGBA; c++) {
            float value = hidden.values[pixel * QWEN_VAE_RGBA + c];
            if (!std::isfinite(value)) return {};
            unsigned char byte = static_cast<unsigned char>(std::nearbyint((std::max(-1.0f,std::min(1.0f,value)) * 0.5f + 0.5f) * 255));
            if (c == 3) output.alpha[pixel] = byte;
            else output.rgb[pixel * 3 + c] = byte;
        }
    }
    return output;
}
