/* This file is a part of 'Pixelina' Image Generator, Editor, and Viewer.
 *
 * (C) Dmitry 'sciloaf' Solovyev, 2026
 *
 * License: MIT
 */
#include <algorithm>
#include <cmath>
#include <limits>
#include "inference.h"

static bool require_weight(const PixModelFile &model, const std::string &name, const std::vector<uint64_t> &shape)
{
    auto found = model.weights.find(name);
    return found != model.weights.end() && found->second.shape == shape;
}

static bool validate_encoder(const PixModelFile &model)
{
    auto architecture = model.metadata.find("general.architecture");
    if (architecture == model.metadata.end() || architecture->second.scalar != "qwen3") return false;
    if (!require_weight(model,QWEN_VL_EMBEDDING_WEIGHT,{QWEN_HIDDEN,QWEN_VOCABULARY})) return false;
    if (!require_weight(model,QWEN_CHAT_OUTPUT_NORM,{QWEN_HIDDEN})) return false;
    for (unsigned layer = 0; layer < QWEN_LAYERS; layer++) {
        std::string prefix = QWEN_VL_BLOCK_PREFIX + std::to_string(layer) + TENSOR_NAME_SEPARATOR;
        if (!require_weight(model,prefix + QWEN_VL_ATTENTION_NORM,{QWEN_HIDDEN})) return false;
        if (!require_weight(model,prefix + QWEN_VL_FFN_NORM,{QWEN_HIDDEN})) return false;
        if (!require_weight(model,prefix + QWEN_VL_QUERY WEIGHT_SUFFIX,{QWEN_HIDDEN,QWEN_HEADS * QWEN_HEAD_DIM})) return false;
        if (!require_weight(model,prefix + QWEN_VL_KEY WEIGHT_SUFFIX,{QWEN_HIDDEN,QWEN_KV_HEADS * QWEN_HEAD_DIM})) return false;
        if (!require_weight(model,prefix + QWEN_VL_VALUE WEIGHT_SUFFIX,{QWEN_HIDDEN,QWEN_KV_HEADS * QWEN_HEAD_DIM})) return false;
        if (!require_weight(model,prefix + QWEN_VL_ATTENTION_OUTPUT WEIGHT_SUFFIX,{QWEN_HEADS * QWEN_HEAD_DIM,QWEN_HIDDEN})) return false;
        if (!require_weight(model,prefix + QWEN_VL_QUERY_NORM,{QWEN_HEAD_DIM})) return false;
        if (!require_weight(model,prefix + QWEN_VL_KEY_NORM,{QWEN_HEAD_DIM})) return false;
        if (!require_weight(model,prefix + QWEN_VL_FFN_GATE WEIGHT_SUFFIX,{QWEN_HIDDEN,QWEN_INTERMEDIATE})) return false;
        if (!require_weight(model,prefix + QWEN_VL_FFN_UP WEIGHT_SUFFIX,{QWEN_HIDDEN,QWEN_INTERMEDIATE})) return false;
        if (!require_weight(model,prefix + QWEN_VL_FFN_DOWN WEIGHT_SUFFIX,{QWEN_INTERMEDIATE,QWEN_HIDDEN})) return false;
    }
    return true;
}

static bool validate_z_block(const PixModelFile &model, const std::string &prefix, bool modulation)
{
    if (!require_weight(model,prefix + Z_ATTENTION_QKV WEIGHT_SUFFIX,{Z_HIDDEN,Z_HIDDEN * 3})) return false;
    if (!require_weight(model,prefix + Z_ATTENTION_OUTPUT WEIGHT_SUFFIX,{Z_HIDDEN,Z_HIDDEN})) return false;
    if (!require_weight(model,prefix + Z_QUERY_NORM,{Z_HEAD_DIM})) return false;
    if (!require_weight(model,prefix + Z_KEY_NORM,{Z_HEAD_DIM})) return false;
    const char* norms[] = {Z_ATTENTION_NORM1,Z_ATTENTION_NORM2,Z_FFN_NORM1,Z_FFN_NORM2};
    for (const char* norm : norms) if (!require_weight(model,prefix + norm + WEIGHT_SUFFIX,{Z_HIDDEN})) return false;
    if (!require_weight(model,prefix + Z_FFN_GATE WEIGHT_SUFFIX,{Z_HIDDEN,Z_INTERMEDIATE})) return false;
    if (!require_weight(model,prefix + Z_FFN_UP WEIGHT_SUFFIX,{Z_HIDDEN,Z_INTERMEDIATE})) return false;
    if (!require_weight(model,prefix + Z_FFN_DOWN WEIGHT_SUFFIX,{Z_INTERMEDIATE,Z_HIDDEN})) return false;
    if (modulation) {
        if (!require_weight(model,prefix + Z_MODULATION WEIGHT_SUFFIX,{Z_TIME_DIM,Z_HIDDEN * 4})) return false;
        if (!require_weight(model,prefix + Z_MODULATION BIAS_SUFFIX,{Z_HIDDEN * 4})) return false;
    }
    return true;
}

static bool validate_diffuser(const PixModelFile &model)
{
    if (!require_weight(model,Z_CAPTION_NORM,{QWEN_HIDDEN})) return false;
    if (!require_weight(model,Z_CAPTION_PROJECTION WEIGHT_SUFFIX,{QWEN_HIDDEN,Z_HIDDEN})) return false;
    if (!require_weight(model,Z_CAPTION_PROJECTION BIAS_SUFFIX,{Z_HIDDEN})) return false;
    if (!require_weight(model,Z_CAPTION_PAD,{Z_HIDDEN,1})) return false;
    if (!require_weight(model,Z_IMAGE_PAD,{Z_HIDDEN,1})) return false;
    if (!require_weight(model,Z_IMAGE_PROJECTION WEIGHT_SUFFIX,{Z_PATCH_SIZE * Z_PATCH_SIZE * Z_LATENT_CHANNELS,Z_HIDDEN})) return false;
    if (!require_weight(model,Z_IMAGE_PROJECTION BIAS_SUFFIX,{Z_HIDDEN})) return false;
    if (!require_weight(model,Z_TIME_FIRST WEIGHT_SUFFIX,{Z_TIME_DIM,Z_TIME_HIDDEN})) return false;
    if (!require_weight(model,Z_TIME_FIRST BIAS_SUFFIX,{Z_TIME_HIDDEN})) return false;
    if (!require_weight(model,Z_TIME_SECOND WEIGHT_SUFFIX,{Z_TIME_HIDDEN,Z_TIME_DIM})) return false;
    if (!require_weight(model,Z_TIME_SECOND BIAS_SUFFIX,{Z_TIME_DIM})) return false;
    if (!require_weight(model,Z_OUTPUT_PROJECTION WEIGHT_SUFFIX,{Z_HIDDEN,Z_PATCH_SIZE * Z_PATCH_SIZE * Z_LATENT_CHANNELS})) return false;
    if (!require_weight(model,Z_OUTPUT_PROJECTION BIAS_SUFFIX,{Z_PATCH_SIZE * Z_PATCH_SIZE * Z_LATENT_CHANNELS})) return false;
    if (!require_weight(model,Z_OUTPUT_MODULATION WEIGHT_SUFFIX,{Z_TIME_DIM,Z_HIDDEN})) return false;
    if (!require_weight(model,Z_OUTPUT_MODULATION BIAS_SUFFIX,{Z_HIDDEN})) return false;
    for (unsigned layer = 0; layer < Z_REFINERS; layer++) {
        if (!validate_z_block(model,Z_CONTEXT_REFINER_PREFIX + std::to_string(layer) + TENSOR_NAME_SEPARATOR,false)) return false;
        if (!validate_z_block(model,Z_NOISE_REFINER_PREFIX + std::to_string(layer) + TENSOR_NAME_SEPARATOR,true)) return false;
    }
    for (unsigned layer = 0; layer < Z_LAYERS; layer++) if (!validate_z_block(model,Z_BLOCK_PREFIX + std::to_string(layer) + TENSOR_NAME_SEPARATOR,true)) return false;
    return true;
}

static bool validate_conv(const PixModelFile &model, const std::string &name, size_t input, size_t output, size_t kernel)
{
    if (!require_weight(model,name + WEIGHT_SUFFIX,{kernel,kernel,input,output})) return false;
    if (!require_weight(model,name + BIAS_SUFFIX,{output})) return false;
    return true;
}

static bool validate_vae_block(const PixModelFile &model, const std::string &name, size_t input, size_t output)
{
    if (!require_weight(model,name + VAE_NORM1_SUFFIX WEIGHT_SUFFIX,{input})) return false;
    if (!require_weight(model,name + VAE_NORM1_SUFFIX BIAS_SUFFIX,{input})) return false;
    if (!require_weight(model,name + VAE_NORM2_SUFFIX WEIGHT_SUFFIX,{output})) return false;
    if (!require_weight(model,name + VAE_NORM2_SUFFIX BIAS_SUFFIX,{output})) return false;
    if (!validate_conv(model,name + VAE_CONV1_SUFFIX,input,output,3)) return false;
    if (!validate_conv(model,name + VAE_CONV2_SUFFIX,output,output,3)) return false;
    if (input != output && !validate_conv(model,name + VAE_SHORTCUT_SUFFIX,input,output,1)) return false;
    return true;
}

static bool validate_vae(const PixModelFile &model)
{
    static const size_t channels[] = {128,256,512,512};
    size_t current = channels[Z_VAE_LEVELS - 1];
    if (!validate_conv(model,VAE_INPUT_CONV,Z_LATENT_CHANNELS,current,3)) return false;
    if (!validate_vae_block(model,VAE_MID_BLOCK_FIRST,current,current)) return false;
    if (!validate_vae_block(model,VAE_MID_BLOCK_SECOND,current,current)) return false;
    if (!require_weight(model,VAE_ATTENTION_PREFIX VAE_NORM_SUFFIX WEIGHT_SUFFIX,{current})) return false;
    if (!require_weight(model,VAE_ATTENTION_PREFIX VAE_NORM_SUFFIX BIAS_SUFFIX,{current})) return false;
    const char* projections[] = {VAE_QUERY_SUFFIX,VAE_KEY_SUFFIX,VAE_VALUE_SUFFIX,VAE_PROJECTION_SUFFIX};
    for (const char* projection : projections) if (!validate_conv(model,std::string(VAE_ATTENTION_PREFIX) + projection,current,current,1)) return false;
    for (int level = Z_VAE_LEVELS - 1; level >= 0; level--) {
        for (int block = 0; block < Z_VAE_BLOCKS; block++) {
            std::string name = std::string(VAE_UP_PREFIX) + std::to_string(level) + VAE_BLOCK_PREFIX + std::to_string(block);
            if (!validate_vae_block(model,name,current,channels[level])) return false;
            current = channels[level];
        }
        if (level && !validate_conv(model,std::string(VAE_UP_PREFIX) + std::to_string(level) + VAE_UPSAMPLE_SUFFIX,current,current,3)) return false;
    }
    if (!require_weight(model,VAE_OUTPUT_NORM WEIGHT_SUFFIX,{current})) return false;
    if (!require_weight(model,VAE_OUTPUT_NORM BIAS_SUFFIX,{current})) return false;
    if (!validate_conv(model,VAE_OUTPUT_CONV,current,Z_RGB_CHANNELS,3)) return false;
    return true;
}

bool inference_validate(const PixSettings &settings)
{
    PixModelFile file;
    if (file.open(settings.encoder,false)) return false;
    if (!validate_encoder(file)) return false;
    PixTokenz tokenizer;
    if (!tokenizer.load(file)) return false;
    if (file.open(settings.diffuser,false)) return false;
    if (!validate_diffuser(file)) return false;
    if (file.open(settings.vae,true)) return false;
    if (!validate_vae(file)) return false;
    return true;
}

static bool finite_tensor(const PixTensor &tensor)
{
#ifdef PIXELINA_WITH_VULKAN
    if (!vulkan_synchronize()) return false;
#endif
    if (tensor.values.empty()) return false;
    for (float value : tensor.values) {
        if (!std::isfinite(value)) return false;
    }
    return true;
}

static PixTensor head_norm(PixTensor input, const std::vector<float> &weight, size_t dimension, float epsilon)
{
    if (input.values.empty()) return {};
    size_t heads = input.channels / dimension;
    input.rows *= heads;
    input.channels = dimension;
    PixTensor result = tensor_norm(input,weight,epsilon);
    result.rows /= heads;
    result.channels *= heads;
    return result;
}

PixTensor inference_encode(const PixModelFile &encoder, const std::string &prompt, const InferenceProgress &progress)
{
    if (!validate_encoder(encoder)) return {};
    PixTokenz tokenizer;
    if (!tokenizer.load(encoder)) return {};
    std::vector<int> ids = tokenizer.encode(prompt);
    if (ids.empty()) return {};
    const PixWeight &embedding = encoder.tensor(QWEN_VL_EMBEDDING_WEIGHT);
    uint64_t stride = weight_bytes({QWEN_HIDDEN},embedding.type);
    PixTensor hidden(ids.size(),QWEN_HIDDEN);
    for (size_t i = 0; i < ids.size(); i++) {
        if (ids[i] < 0 || ids[i] >= QWEN_VOCABULARY) return {};
#ifdef PIXELINA_WITH_VULKAN
        if (!vulkan_active_state())
#endif
        cpu_decode(encoder.data + embedding.offset + size_t(ids[i]) * stride,embedding.type,QWEN_HIDDEN,hidden.values.data() + i * QWEN_HIDDEN);
    }
#ifdef PIXELINA_WITH_VULKAN
    if (vulkan_active_state()) hidden = vulkan_embedding(encoder,embedding,ids);
#endif
    PixRotary angles;
    angles.cosine.resize(ids.size() * QWEN_HEAD_DIM / 2);
    angles.sine.resize(angles.cosine.size());
    for (size_t row = 0; row < ids.size(); row++) {
        for (size_t pair = 0; pair < QWEN_HEAD_DIM / 2; pair++) {
            float angle = float(double(row) / std::pow(QWEN_ROPE_THETA,double(pair * 2) / QWEN_HEAD_DIM));
            angles.cosine[row * QWEN_HEAD_DIM / 2 + pair] = std::cos(angle);
            angles.sine[row * QWEN_HEAD_DIM / 2 + pair] = std::sin(angle);
        }
    }
    // hidden_states[-2] is the output of block 34, before the last block and final RMSNorm.
    for (unsigned layer = 0; layer < QWEN_LAYERS - 1; layer++) {
        std::string prefix = QWEN_VL_BLOCK_PREFIX + std::to_string(layer) + TENSOR_NAME_SEPARATOR;
        PixTensor normalized = tensor_norm(hidden,tensor_vector(encoder,prefix + QWEN_VL_ATTENTION_NORM),COMPUTE_NORM_EPSILON);
        PixTensor q = tensor_linear(encoder,prefix + QWEN_VL_QUERY,normalized,false);
        PixTensor k = tensor_linear(encoder,prefix + QWEN_VL_KEY,normalized,false);
        PixTensor v = tensor_linear(encoder,prefix + QWEN_VL_VALUE,normalized,false);
        q = head_norm(std::move(q),tensor_vector(encoder,prefix + QWEN_VL_QUERY_NORM),QWEN_HEAD_DIM,COMPUTE_NORM_EPSILON);
        k = head_norm(std::move(k),tensor_vector(encoder,prefix + QWEN_VL_KEY_NORM),QWEN_HEAD_DIM,COMPUTE_NORM_EPSILON);
        tensor_rope(q,QWEN_HEADS,angles.cosine,angles.sine,false);
        tensor_rope(k,QWEN_KV_HEADS,angles.cosine,angles.sine,false);
        PixTensor attended = tensor_attention(q,k,v,QWEN_HEADS,QWEN_KV_HEADS,true);
        tensor_add(hidden,tensor_linear(encoder,prefix + QWEN_VL_ATTENTION_OUTPUT,attended,false));
        q = PixTensor();
        k = PixTensor();
        v = PixTensor();
        attended = PixTensor();
        normalized = tensor_norm(hidden,tensor_vector(encoder,prefix + QWEN_VL_FFN_NORM),COMPUTE_NORM_EPSILON);
        PixTensor gate = tensor_linear(encoder,prefix + QWEN_VL_FFN_GATE,normalized,false);
        PixTensor up = tensor_linear(encoder,prefix + QWEN_VL_FFN_UP,normalized,false);
        tensor_silu(gate);
        tensor_gate(gate,up);
        tensor_add(hidden,tensor_linear(encoder,prefix + QWEN_VL_FFN_DOWN,gate,false));
        if (hidden.values.empty()) return {};
#ifdef PIXELINA_WITH_VULKAN
        if (!vulkan_active_state())
#endif
        if (!finite_tensor(hidden)) return {};
        if (progress && !progress(INFERENCE_ENCODER_FRACTION * float(layer + 1) / (QWEN_LAYERS - 1),"Text encoder " + std::to_string(layer + 1) + "/" + std::to_string(QWEN_LAYERS - 1))) return {};
    }
    return hidden;
}

PixRotary z_rotary(size_t tokens, size_t actual, size_t caption, size_t patch_width, bool image)
{
    static const size_t dimensions[] = {32,48,48};
    PixRotary angles;
    angles.cosine.resize(tokens * Z_HEAD_DIM / 2);
    angles.sine.resize(angles.cosine.size());
    for (size_t row = 0; row < tokens; row++) {
        size_t positions[3] = {};
        if (image) {
            if (row < actual) {
                positions[0] = caption + 1;
                positions[1] = row / patch_width;
                positions[2] = row % patch_width;
            }
        } else {
#ifdef PADDED_TOKEN_ZERO_POS
            positions[0] = row < actual? row + 1 : 0;
#else
            positions[0] = row + 1;
#endif
        }
        size_t pair = 0;
        for (size_t axis = 0; axis < 3; axis++) {
            for (size_t d = 0; d < dimensions[axis] / 2; d++,pair++) {
                // Analytic RoPE preserves reference frequencies without a finite lookup-table limit.
                float angle = float(double(positions[axis]) / std::pow(Z_ROPE_THETA,double(d * 2) / dimensions[axis]));
                angles.cosine[row * Z_HEAD_DIM / 2 + pair] = std::cos(angle);
                angles.sine[row * Z_HEAD_DIM / 2 + pair] = std::sin(angle);
            }
        }
    }
    return angles;
}

static PixTensor z_modulate(const PixTensor &input, const PixTensor &modulation, size_t part)
{
    if (input.values.empty() || modulation.values.empty()) return {};
    PixTensor output(input.rows,input.channels);
#ifdef PIXELINA_WITH_VULKAN
    if (vulkan_active_state()) {
        vulkan_copy_rows(input,output,input.rows);
        vulkan_pointwise(VK_OP_MODULATE,output,&modulation,nullptr,unsigned(part));
        return output;
    }
#endif
    const float* scale = modulation.values.data() + part * input.channels;
    for (size_t i = 0; i < output.values.size(); i++) output.values[i] = input.values[i] * (1.0f + scale[i % input.channels]);
    return output;
}

static void z_residual(PixTensor &input, const PixTensor &output, const PixTensor* modulation, size_t part)
{
    if (output.values.empty() || (modulation && modulation->values.empty())) input = PixTensor();
    if (input.values.empty()) return;
    const float* gate = modulation? modulation->values.data() + part * input.channels : nullptr;
#ifdef PIXELINA_WITH_VULKAN
    if (vulkan_active_state()) {
        vulkan_pointwise(VK_OP_RESIDUAL,input,&output,modulation,unsigned(part));
        return;
    }
#endif
    for (size_t i = 0; i < input.values.size(); i++) input.values[i] += output.values[i] * (gate? std::tanh(gate[i % input.channels]) : 1.0f);
}

static PixTensor z_block(const PixModelFile &model, const std::string &name, PixTensor input, const PixRotary &angles, const PixTensor* timestep)
{
    PixTensor modulation;
    if (timestep) modulation = tensor_linear(model,name + Z_MODULATION,*timestep,true);
    PixTensor normalized = tensor_norm(input,tensor_vector(model,name + Z_ATTENTION_NORM1 WEIGHT_SUFFIX),COMPUTE_Z_NORM_EPSILON);
    if (timestep) normalized = z_modulate(normalized,modulation,0);
    PixTensor qkv = tensor_linear(model,name + Z_ATTENTION_QKV,normalized,false);
    if (qkv.values.empty()) return {};
    PixTensor q(input.rows,Z_HIDDEN),k(input.rows,Z_HIDDEN),v(input.rows,Z_HIDDEN);
#ifdef PIXELINA_WITH_VULKAN
    if (vulkan_active_state()) {
        vulkan_copy_rows(qkv,q,input.rows);
        vulkan_copy_rows(qkv,k,input.rows,0,Z_HIDDEN);
        vulkan_copy_rows(qkv,v,input.rows,0,Z_HIDDEN * 2);
    } else
#endif
    for (size_t row = 0; row < input.rows; row++) {
        std::copy_n(qkv.values.data() + row * qkv.channels,Z_HIDDEN,q.values.data() + row * Z_HIDDEN);
        std::copy_n(qkv.values.data() + row * qkv.channels + Z_HIDDEN,Z_HIDDEN,k.values.data() + row * Z_HIDDEN);
        std::copy_n(qkv.values.data() + row * qkv.channels + Z_HIDDEN * 2,Z_HIDDEN,v.values.data() + row * Z_HIDDEN);
    }
    qkv = PixTensor();
    q = head_norm(std::move(q),tensor_vector(model,name + Z_QUERY_NORM),Z_HEAD_DIM,COMPUTE_Z_NORM_EPSILON);
    k = head_norm(std::move(k),tensor_vector(model,name + Z_KEY_NORM),Z_HEAD_DIM,COMPUTE_Z_NORM_EPSILON);
    tensor_rope(q,Z_HEADS,angles.cosine,angles.sine,true);
    tensor_rope(k,Z_HEADS,angles.cosine,angles.sine,true);
    PixTensor attended = tensor_attention(q,k,v,Z_HEADS,Z_HEADS,false);
    PixTensor projected = tensor_linear(model,name + Z_ATTENTION_OUTPUT,attended,false);
    PixTensor residual = tensor_norm(projected,tensor_vector(model,name + Z_ATTENTION_NORM2 WEIGHT_SUFFIX),COMPUTE_Z_NORM_EPSILON);
    q = PixTensor();
    k = PixTensor();
    v = PixTensor();
    attended = PixTensor();
    projected = PixTensor();
    z_residual(input,residual,timestep? &modulation : nullptr,1);
    residual = PixTensor();
    normalized = tensor_norm(input,tensor_vector(model,name + Z_FFN_NORM1 WEIGHT_SUFFIX),COMPUTE_Z_NORM_EPSILON);
    if (timestep) normalized = z_modulate(normalized,modulation,2);
    PixTensor gate = tensor_linear(model,name + Z_FFN_GATE,normalized,false);
    PixTensor up = tensor_linear(model,name + Z_FFN_UP,normalized,false);
    tensor_silu(gate);
    tensor_gate(gate,up);
    up = PixTensor();
    normalized = PixTensor();
    projected = tensor_linear(model,name + Z_FFN_DOWN,gate,false);
    residual = tensor_norm(projected,tensor_vector(model,name + Z_FFN_NORM2 WEIGHT_SUFFIX),COMPUTE_Z_NORM_EPSILON);
    z_residual(input,residual,timestep? &modulation : nullptr,3);
#ifdef PIXELINA_WITH_VULKAN
    if (!vulkan_active_state())
#endif
    if (!finite_tensor(input)) return {};
    return input;
}

static PixTensor z_pad(const PixModelFile &model, PixTensor input, const std::string &name)
{
    if (input.values.empty()) return {};
    size_t padded = (input.rows + Z_SEQUENCE_MULTIPLE - 1) / Z_SEQUENCE_MULTIPLE * Z_SEQUENCE_MULTIPLE;
    if (padded == input.rows) return input;
    std::vector<float> pad = tensor_vector(model,name);
    PixTensor result(padded,input.channels);
#ifdef PIXELINA_WITH_VULKAN
    if (vulkan_active_state()) vulkan_copy_rows(input,result,input.rows); else
#endif
    std::copy(input.values.begin(),input.values.end(),result.values.begin());
    if (result.values.empty()) return {};
    for (size_t row = input.rows; row < padded; row++) std::copy(pad.begin(),pad.end(),result.values.begin() + row * input.channels);
    return result;
}

PixTensor inference_diffuse(const PixModelFile &diffuser, const PixTensor &latents, const PixTensor &conditioning, size_t width, size_t height, float sigma, const InferenceProgress &progress, float begin, float span)
{
    if (width % Z_PATCH_SIZE || height % Z_PATCH_SIZE || latents.rows != width * height || latents.channels != Z_LATENT_CHANNELS || conditioning.channels != QWEN_HIDDEN) return {};
    PixTensor frequency(1,Z_TIME_DIM);
    for (size_t i = 0; i < Z_TIME_DIM / 2; i++) {
        float angle = float((1.0f - sigma) * Z_TIME_SCALE * std::pow(Z_TIME_PERIOD,-double(i) / (Z_TIME_DIM / 2)));
        frequency.values[i] = std::cos(angle);
        frequency.values[i + Z_TIME_DIM / 2] = std::sin(angle);
    }
    PixTensor timestep = tensor_linear(diffuser,Z_TIME_FIRST,frequency,true);
    tensor_silu(timestep);
    timestep = tensor_linear(diffuser,Z_TIME_SECOND,timestep,true);
    size_t patch_width = width / Z_PATCH_SIZE,patch_height = height / Z_PATCH_SIZE;
    PixTensor patches(patch_width * patch_height,Z_LATENT_CHANNELS * Z_PATCH_SIZE * Z_PATCH_SIZE);
#ifdef PIXELINA_WITH_VULKAN
    if (vulkan_active_state()) vulkan_layout(VK_OP_PATCH,latents,patches,width,Z_PATCH_SIZE); else
#endif
    for (size_t y = 0; y < patch_height; y++) {
        for (size_t x = 0; x < patch_width; x++) {
            for (size_t dy = 0; dy < Z_PATCH_SIZE; dy++) {
                for (size_t dx = 0; dx < Z_PATCH_SIZE; dx++) {
                    size_t from = ((y * Z_PATCH_SIZE + dy) * width + x * Z_PATCH_SIZE + dx) * Z_LATENT_CHANNELS;
                    size_t to = ((y * patch_width + x) * Z_PATCH_SIZE * Z_PATCH_SIZE + dy * Z_PATCH_SIZE + dx) * Z_LATENT_CHANNELS;
                    std::copy_n(latents.values.data() + from,Z_LATENT_CHANNELS,patches.values.data() + to);
                }
            }
        }
    }
    PixTensor visual = z_pad(diffuser,tensor_linear(diffuser,Z_IMAGE_PROJECTION,patches,true),Z_IMAGE_PAD);
    PixTensor caption = tensor_norm(conditioning,tensor_vector(diffuser,Z_CAPTION_NORM),COMPUTE_Z_NORM_EPSILON);
    caption = z_pad(diffuser,tensor_linear(diffuser,Z_CAPTION_PROJECTION,caption,true),Z_CAPTION_PAD);
    if (visual.values.empty() || caption.values.empty()) return {};
    PixRotary caption_angles = z_rotary(caption.rows,conditioning.rows,0,0,false);
    PixRotary image_angles = z_rotary(visual.rows,patches.rows,caption.rows,patch_width,true);
    for (unsigned layer = 0; layer < Z_REFINERS; layer++) {
        visual = z_block(diffuser,Z_NOISE_REFINER_PREFIX + std::to_string(layer) + TENSOR_NAME_SEPARATOR,std::move(visual),image_angles,&timestep);
        caption = z_block(diffuser,Z_CONTEXT_REFINER_PREFIX + std::to_string(layer) + TENSOR_NAME_SEPARATOR,std::move(caption),caption_angles,nullptr);
        if (visual.values.empty() || caption.values.empty()) return {};
    }
    // Attention is permutation-equivariant; use SD.cpp's caption-first joint layout.
    PixTensor joint(caption.rows + visual.rows,Z_HIDDEN);
#ifdef PIXELINA_WITH_VULKAN
    if (vulkan_active_state()) {
        vulkan_copy_rows(caption,joint,caption.rows);
        vulkan_copy_rows(visual,joint,visual.rows,0,0,caption.rows);
    } else
#endif
    {
        std::copy(caption.values.begin(),caption.values.end(),joint.values.begin());
        std::copy(visual.values.begin(),visual.values.end(),joint.values.begin() + caption.values.size());
    }
    if (joint.values.empty()) return {};
    size_t caption_rows = caption.rows;
    caption = PixTensor();
    visual = PixTensor();
    PixRotary angles = std::move(caption_angles);
    angles.cosine.insert(angles.cosine.end(),image_angles.cosine.begin(),image_angles.cosine.end());
    angles.sine.insert(angles.sine.end(),image_angles.sine.begin(),image_angles.sine.end());
    for (unsigned layer = 0; layer < Z_LAYERS; layer++) {
        joint = z_block(diffuser,Z_BLOCK_PREFIX + std::to_string(layer) + TENSOR_NAME_SEPARATOR,std::move(joint),angles,&timestep);
        if (joint.values.empty()) return {};
        if (progress && !progress(begin + span * float(layer + 1) / Z_LAYERS,"Diffuser block " + std::to_string(layer + 1) + "/" + std::to_string(Z_LAYERS))) return {};
    }
    PixTensor final_input(patches.rows,Z_HIDDEN);
#ifdef PIXELINA_WITH_VULKAN
    if (vulkan_active_state()) vulkan_copy_rows(joint,final_input,final_input.rows,caption_rows); else
#endif
    std::copy_n(joint.values.data() + caption_rows * Z_HIDDEN,final_input.values.size(),final_input.values.data());
    joint = PixTensor();
    final_input = tensor_norm(final_input,{},COMPUTE_NORM_EPSILON,true);
    tensor_silu(timestep);
    PixTensor scale = tensor_linear(diffuser,Z_OUTPUT_MODULATION,timestep,true);
#ifdef PIXELINA_WITH_VULKAN
    if (vulkan_active_state()) vulkan_pointwise(VK_OP_MODULATE,final_input,&scale,nullptr); else
#endif
    for (size_t i = 0; i < final_input.values.size(); i++) final_input.values[i] *= 1.0f + scale.values[i % Z_HIDDEN];
    PixTensor prediction = tensor_linear(diffuser,Z_OUTPUT_PROJECTION,final_input,true);
    PixTensor result(width * height,Z_LATENT_CHANNELS);
#ifdef PIXELINA_WITH_VULKAN
    if (vulkan_active_state()) vulkan_layout(VK_OP_UNPATCH,prediction,result,width,Z_PATCH_SIZE); else
#endif
    for (size_t y = 0; y < patch_height; y++) {
        for (size_t x = 0; x < patch_width; x++) {
            for (size_t dy = 0; dy < Z_PATCH_SIZE; dy++) {
                for (size_t dx = 0; dx < Z_PATCH_SIZE; dx++) {
                    size_t from = ((y * patch_width + x) * Z_PATCH_SIZE * Z_PATCH_SIZE + dy * Z_PATCH_SIZE + dx) * Z_LATENT_CHANNELS;
                    size_t to = ((y * Z_PATCH_SIZE + dy) * width + x * Z_PATCH_SIZE + dx) * Z_LATENT_CHANNELS;
                    std::copy_n(prediction.values.data() + from,Z_LATENT_CHANNELS,result.values.data() + to);
                }
            }
        }
    }
    return result;
}

static PixTensor vae_block(const PixModelFile &model, const std::string &name, PixTensor input, size_t width, size_t height)
{
    PixTensor hidden = tensor_groupnorm(model,name + VAE_NORM1_SUFFIX,input);
    tensor_silu(hidden);
    hidden = tensor_conv(model,name + VAE_CONV1_SUFFIX,hidden,width,height);
    hidden = tensor_groupnorm(model,name + VAE_NORM2_SUFFIX,hidden);
    tensor_silu(hidden);
    hidden = tensor_conv(model,name + VAE_CONV2_SUFFIX,hidden,width,height);
    if (hidden.values.empty()) return {};
    if (input.channels != hidden.channels) input = tensor_conv(model,name + VAE_SHORTCUT_SUFFIX,input,width,height);
    tensor_add(hidden,input);
    return hidden;
}

PixImage inference_decode(const PixModelFile &vae, PixTensor latents, size_t width, size_t height, const InferenceProgress &progress)
{
    if (!validate_vae(vae)) return {};
#ifdef PIXELINA_WITH_VULKAN
    if (vulkan_active_state()) vulkan_pointwise(VK_OP_AFFINE,latents,nullptr,nullptr,0,1.0f / Z_VAE_SCALE,Z_VAE_SHIFT); else
#endif
    for (float &value : latents.values) value = value / Z_VAE_SCALE + Z_VAE_SHIFT;
    PixTensor hidden = tensor_conv(vae,VAE_INPUT_CONV,latents,width,height);
    hidden = vae_block(vae,VAE_MID_BLOCK_FIRST,std::move(hidden),width,height);
    PixTensor normalized = tensor_groupnorm(vae,VAE_ATTENTION_PREFIX VAE_NORM_SUFFIX,hidden);
    PixTensor q = tensor_conv(vae,VAE_ATTENTION_PREFIX VAE_QUERY_SUFFIX,normalized,width,height);
    PixTensor k = tensor_conv(vae,VAE_ATTENTION_PREFIX VAE_KEY_SUFFIX,normalized,width,height);
    PixTensor v = tensor_conv(vae,VAE_ATTENTION_PREFIX VAE_VALUE_SUFFIX,normalized,width,height);
    PixTensor attention = tensor_attention(q,k,v,1,1,false);
    tensor_add(hidden,tensor_conv(vae,VAE_ATTENTION_PREFIX VAE_PROJECTION_SUFFIX,attention,width,height));
    q = PixTensor();
    k = PixTensor();
    v = PixTensor();
    normalized = PixTensor();
    attention = PixTensor();
    hidden = vae_block(vae,VAE_MID_BLOCK_SECOND,std::move(hidden),width,height);
    if (hidden.values.empty()) return {};
    unsigned completed = 0;
    for (int level = Z_VAE_LEVELS - 1; level >= 0; level--) {
        for (int block = 0; block < Z_VAE_BLOCKS; block++) {
            std::string name = std::string(VAE_UP_PREFIX) + std::to_string(level) + VAE_BLOCK_PREFIX + std::to_string(block);
            hidden = vae_block(vae,name,std::move(hidden),width,height);
            if (hidden.values.empty()) return {};
            completed++;
            if (progress && !progress(INFERENCE_ENCODER_FRACTION + INFERENCE_DIFFUSION_FRACTION + INFERENCE_VAE_FRACTION * float(completed) / (Z_VAE_LEVELS * Z_VAE_BLOCKS + 1),"VAE decoder " + std::to_string(completed) + "/" + std::to_string(Z_VAE_LEVELS * Z_VAE_BLOCKS))) return {};
        }
        if (level) {
            PixTensor doubled(hidden.rows * 4,hidden.channels);
#ifdef PIXELINA_WITH_VULKAN
            if (vulkan_active_state()) vulkan_layout(VK_OP_UPSAMPLE,hidden,doubled,width); else
#endif
            for (size_t y = 0; y < height * 2; y++) {
                for (size_t x = 0; x < width * 2; x++) std::copy_n(hidden.values.data() + ((y / 2) * width + x / 2) * hidden.channels,hidden.channels,doubled.values.data() + (y * width * 2 + x) * hidden.channels);
            }
            width *= 2;
            height *= 2;
            hidden = tensor_conv(vae,std::string(VAE_UP_PREFIX) + std::to_string(level) + VAE_UPSAMPLE_SUFFIX,doubled,width,height);
        }
    }
    hidden = tensor_groupnorm(vae,VAE_OUTPUT_NORM,hidden);
    tensor_silu(hidden);
    hidden = tensor_conv(vae,VAE_OUTPUT_CONV,hidden,width,height);
    if (!finite_tensor(hidden)) return {};
    PixImage image;
    image.width = int(width);
    image.height = int(height);
    image.rgb.resize(hidden.values.size());
    for (size_t i = 0; i < hidden.values.size(); i++) image.rgb[i] = static_cast<unsigned char>(std::lround(std::clamp(hidden.values[i] * 0.5f + 0.5f,0.0f,1.0f) * 255));
    if (progress && !progress(1,"Complete")) return {};
    return image;
}

std::vector<float> inference_sigmas(int steps, float shift)
{
    if (steps <= 0 || !std::isfinite(shift) || shift <= 0) return {};
    std::vector<float> sigmas(size_t(steps) + 1,0);
    for (int i = 0; i < steps; i++) {
        float raw = 1.0f - float(i) / steps;
        sigmas[i] = shift * raw / (1.0f + (shift - 1.0f) * raw);
    }
    return sigmas;
}

PixTensor inference_noise(size_t rows, uint64_t seed, size_t channels)
{
    PixTensor noise(rows,channels);
    if (noise.values.empty()) return {};
    auto uniform = [&]() {
        seed += NOISE_SPLITMIX_INCREMENT;
        uint64_t bits = seed;
        bits = (bits ^ (bits >> 30)) * NOISE_SPLITMIX_MIX1;
        bits = (bits ^ (bits >> 27)) * NOISE_SPLITMIX_MIX2;
        bits ^= bits >> 31;
        return (double(bits >> 11) + 0.5) * NOISE_UNIFORM_SCALE;
    };
    // Fill channels-first to make the RNG-to-latent layout explicit across backends.
    for (size_t c = 0; c < channels; c++) {
        for (size_t row = 0; row < rows; row += 2) {
            double radius = std::sqrt(-2 * std::log(uniform()));
            double angle = NOISE_TWO_PI * uniform();
            noise.values[row * channels + c] = float(radius * std::cos(angle));
            if (row + 1 < rows) noise.values[(row + 1) * channels + c] = float(radius * std::sin(angle));
        }
    }
    return noise;
}

PixImage inference_generate(const PixSettings &settings, const std::string &prompt, const InferenceProgress &progress, int &error, const std::function<bool()> &decode_enabled, const InferencePreview &preview)
{
    std::string message;
    error = INFERENCE_ERROR_SETTINGS;
    if (!settings_valid(settings,message)) return {};
    auto report = [&](float fraction, const std::string &stage) {
        if (!progress || progress(fraction,stage)) return true;
        error = INFERENCE_ERROR_STOPPED;
        return false;
    };
    if (!report(0,"Starting generation")) return {};
    error = INFERENCE_ERROR_VULKAN;
#ifdef PIXELINA_WITH_VULKAN
    std::unique_ptr<VulkanSession> session;
    if (settings.backend == PIXELINA_VULKAN) {
        session.reset(new VulkanSession());
        if (!session->initialize() || !session->validate_dimensions(settings)) return {};
    }
#else
    if (settings.backend != PIXELINA_CPU) return {};
#endif
    if (!report(0,"Validating model tensors")) return {};
    error = INFERENCE_ERROR_MODEL;
    if (!inference_validate(settings)) return {};
    if (!report(0,"Loading text encoder")) return {};
    PixModelFile model;
    PixModelFile vae;
    error = INFERENCE_ERROR_MODEL;
    if (model.open(settings.encoder,false)) return {};
    error = INFERENCE_ERROR_PROMPT;
    PixTensor conditioning = inference_encode(model,prompt,report);
    if (conditioning.values.empty()) return {};
    model.close();
    size_t width = size_t(settings.width) / Z_VAE_FACTOR,height = size_t(settings.height) / Z_VAE_FACTOR;
    PixTensor latents = inference_noise(width * height,settings.seed);
    std::vector<float> sigmas = inference_sigmas(settings.steps,settings.shift);
    if (!report(INFERENCE_ENCODER_FRACTION,"Loading diffuser")) return {};
    error = INFERENCE_ERROR_MODEL;
    if (model.open(settings.diffuser,false)) return {};
    for (int step = 0; step < settings.steps; step++) {
        float begin = INFERENCE_ENCODER_FRACTION + INFERENCE_DIFFUSION_FRACTION * float(step) / settings.steps;
        float span = INFERENCE_DIFFUSION_FRACTION / settings.steps;
        auto step_progress = [&](float fraction, const std::string &stage) {
            return report(fraction,"Step " + std::to_string(step + 1) + "/" + std::to_string(settings.steps) + ": " + stage);
        };
        if (!step_progress(begin,"Diffusion")) return {};
        error = INFERENCE_ERROR_NUMERICAL;
        PixTensor prediction = inference_diffuse(model,latents,conditioning,width,height,sigmas[step],step_progress,begin,span);
        if (prediction.values.empty()) return {};
        PixTensor clean;
        // The last step already reaches zero noise and uses the final decode below.
        if (step + 1 < settings.steps && preview && decode_enabled && decode_enabled()) {
            clean = PixTensor(latents.rows,latents.channels);
#ifdef PIXELINA_WITH_VULKAN
            if (vulkan_active_state()) {
                vulkan_copy_rows(prediction,clean,prediction.rows);
                vulkan_pointwise(VK_OP_AFFINE,clean,nullptr,nullptr,0,sigmas[step],0);
                tensor_add(clean,latents);
            } else
#endif
            for (size_t i = 0; i < latents.values.size(); i++) clean.values[i] = latents.values[i] + sigmas[step] * prediction.values[i];
        }
        // The checkpoint predicts negative flow; Euler's negative sigma delta cancels that sign.
        float delta = sigmas[step] - sigmas[step + 1];
#ifdef PIXELINA_WITH_VULKAN
        if (vulkan_active_state()) {
            vulkan_pointwise(VK_OP_AFFINE,prediction,nullptr,nullptr,0,delta,0);
            tensor_add(latents,prediction);
        } else
#endif
        for (size_t i = 0; i < latents.values.size(); i++) latents.values[i] += delta * prediction.values[i];
        if (!finite_tensor(latents)) return {};
        prediction = PixTensor();
        if (!clean.values.empty()) {
            if (!step_progress(begin + span,"Decoding preview")) return {};
            error = INFERENCE_ERROR_MODEL;
            if (!vae.data && vae.open(settings.vae,true)) return {};
            auto preview_progress = [&](float, const std::string &stage) {
                return step_progress(begin + span,"Preview | " + stage);
            };
            error = INFERENCE_ERROR_NUMERICAL;
            PixImage decoded = inference_decode(vae,std::move(clean),width,height,preview_progress);
            if (decoded.rgb.empty()) return {};
            preview(std::move(decoded));
        }
    }
    model.close();
    conditioning = PixTensor();
    if (!report(INFERENCE_ENCODER_FRACTION + INFERENCE_DIFFUSION_FRACTION,"Loading VAE")) return {};
    error = INFERENCE_ERROR_MODEL;
    if (!vae.data && vae.open(settings.vae,true)) return {};
    error = INFERENCE_ERROR_NUMERICAL;
    PixImage image = inference_decode(vae,std::move(latents),width,height,report);
    if (!image.rgb.empty()) error = 0;
    return image;
}
