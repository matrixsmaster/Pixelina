/* This file is a part of 'Pixelina' Image Generator, Editor, and Viewer.
 *
 * (C) Dmitry 'sciloaf' Solovyev, 2026
 *
 * License: MIT
 */
#ifndef PIXELINA_INFERENCE_H
#define PIXELINA_INFERENCE_H

#include "tensor.h"
#include "tokenizer.h"

#define INFERENCE_ERROR_SETTINGS 1
#define INFERENCE_ERROR_MODEL 2
#define INFERENCE_ERROR_PROMPT 3
#define INFERENCE_ERROR_NUMERICAL 4
#define INFERENCE_ERROR_VULKAN 5
#define INFERENCE_ERROR_STOPPED 6
#define INFERENCE_ERROR_MEMORY 7
#define QWEN_HIDDEN 2560
#define QWEN_INTERMEDIATE 9728
#define QWEN_LAYERS 36
#define QWEN_HEADS 32
#define QWEN_KV_HEADS 8
#define QWEN_HEAD_DIM 128
#define QWEN_ROPE_THETA 1000000.0
#define QWEN_VOCABULARY 151936
#define Z_HIDDEN 3840
#define Z_INTERMEDIATE 10240
#define Z_LAYERS 30
#define Z_REFINERS 2
#define Z_HEADS 30
#define Z_HEAD_DIM 128
#define Z_LATENT_CHANNELS 16
#define Z_PATCH_SIZE 2
#define Z_SEQUENCE_MULTIPLE 32
#define Z_TIME_DIM 256
#define Z_TIME_HIDDEN 1024
#define Z_TIME_SCALE 1000.0f
#define Z_TIME_PERIOD 10000.0
#define Z_ROPE_THETA 256.0
#define Z_VAE_FACTOR 8
#define Z_VAE_SCALE 0.3611f
#define Z_VAE_SHIFT 0.1159f
#define Z_VAE_LEVELS 4
#define Z_VAE_BLOCKS 3
#define Z_RGB_CHANNELS 3
#define NOISE_TWO_PI 6.283185307179586476925286766559
#define NOISE_SPLITMIX_INCREMENT UINT64_C(0x9e3779b97f4a7c15)
#define NOISE_SPLITMIX_MIX1 UINT64_C(0xbf58476d1ce4e5b9)
#define NOISE_SPLITMIX_MIX2 UINT64_C(0x94d049bb133111eb)
#define NOISE_UNIFORM_SCALE (1.0 / 9007199254740992.0)
#define INFERENCE_ENCODER_FRACTION 0.15f
#define INFERENCE_DIFFUSION_FRACTION 0.65f
#define INFERENCE_VAE_FRACTION 0.20f
#define WEIGHT_SUFFIX ".weight"
#define BIAS_SUFFIX ".bias"

#define VAE_NORM_SUFFIX ".norm"
#define VAE_DECODER_PREFIX "decoder."
#define VAE_MID_PREFIX "decoder.mid."
#define VAE_ATTENTION_PREFIX "decoder.mid.attn_1"
#define VAE_UP_PREFIX "decoder.up."

struct PixRotary {
    std::vector<float> cosine; // one value per token/rotary pair
    std::vector<float> sine; // corresponding sine
};

// Return false to stop generation at a progress boundary.
using InferenceProgress = std::function<bool(float,const std::string &)>;
using InferencePreview = std::function<void(PixImage)>;
PixRotary z_rotary(size_t tokens, size_t actual, size_t caption, size_t patch_width, bool image);
std::vector<float> inference_sigmas(int steps, float shift);
PixTensor inference_noise(size_t rows, uint64_t seed, size_t channels = Z_LATENT_CHANNELS);
bool inference_validate(const PixSettings &settings);
PixTensor inference_encode(const PixModelFile &encoder, const std::string &prompt, const InferenceProgress &progress);
PixTensor inference_diffuse(const PixModelFile &diffuser, const PixTensor &latents, const PixTensor &conditioning, size_t width, size_t height, float sigma, const InferenceProgress &progress, float begin, float span);
PixImage inference_decode(const PixModelFile &vae, PixTensor latents, size_t width, size_t height, const InferenceProgress &progress);
PixImage inference_generate(const PixSettings &settings, const std::string &prompt, const InferenceProgress &progress, int &error, const std::function<bool()> &decode_enabled = {}, const InferencePreview &preview = {});

#endif
