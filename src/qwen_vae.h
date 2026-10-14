/* This file is a part of 'Pixelina' Image Generator, Editor, and Viewer.
 *
 * (C) Dmitry 'sciloaf' Solovyev, 2026
 *
 * License: MIT
 */
#ifndef PIXELINA_QWEN_VAE_H
#define PIXELINA_QWEN_VAE_H

#include "inference.h"

#define QWEN_VAE_INPUT_CONV "conv1"
#define QWEN_VAE_POST_QUANT_CONV "conv2"
#define QWEN_VAE_MIDDLE_FIRST "middle.0"
#define QWEN_VAE_MIDDLE_SECOND "middle.2"
#define QWEN_VAE_MIDDLE_NORM "middle.1.norm.gamma"
#define QWEN_VAE_MIDDLE_QKV "middle.1.to_qkv"
#define QWEN_VAE_MIDDLE_PROJECTION "middle.1.proj"
#define QWEN_VAE_UP_LEVEL_PREFIX "upsamples."
#define QWEN_VAE_DOWN_LEVEL_PREFIX "downsamples."
#define QWEN_VAE_UP_BLOCK_PREFIX ".upsamples."
#define QWEN_VAE_DOWN_BLOCK_PREFIX ".downsamples."
#define QWEN_VAE_HEAD_NORM "head.0.gamma"
#define QWEN_VAE_HEAD_CONV "head.2"

#define QWEN_VAE_CHANNELS 64
#define QWEN_VAE_RGBA 4
#define QWEN_VAE_FACTOR 16
#define QWEN_VAE_ENCODER_BASE 96
#define QWEN_VAE_DECODER_BASE 144
#define QWEN_VAE_LEVELS 5
#define QWEN_VAE_MULTIPLIERS {1,2,4,8,8}
#define QWEN_VAE_MAX_MULTIPLIER 8
#define QWEN_VAE_BLOCKS 2
#define QWEN_VAE_UP_BLOCKS 3
#define QWEN_VAE_KERNEL 3
#define QWEN_VAE_L2_EPSILON 1e-12f
#define QWEN_VAE_PIXEL_SCALE (1.0f / 255.0f)
#define QWEN_VAE_ENCODE_BEGIN 0.32f
#define QWEN_VAE_ENCODE_SPAN 0.10f
#define QWEN_VAE_DECODE_BEGIN 0.90f
#define QWEN_VAE_DECODE_SPAN 0.10f
#define QWEN_VAE_ENCODER "encoder."
#define QWEN_VAE_DECODER "decoder."
#define QWEN_VAE_NORM1 ".residual.0.gamma"
#define QWEN_VAE_CONV1 ".residual.2"
#define QWEN_VAE_NORM2 ".residual.3.gamma"
#define QWEN_VAE_CONV2 ".residual.6"
#define QWEN_VAE_SHORTCUT ".shortcut"
#define QWEN_VAE_RESAMPLE ".resample.1"
#define QWEN_VAE_TIME_CONV ".time_conv"

static const float qwen_vae_mean[QWEN_VAE_CHANNELS] = {
    0.5126f,0.7721f,-0.0631f,1.3506f,-0.7855f,-2.1025f,-0.3458f,1.3722f,1.8873f,-1.7177f,-0.651f,0.2732f,0.7562f,-0.6163f,-1.0277f,3.8363f,2.021f,0.0472f,0.932f,2.0087f,2.4954f,-0.1391f,-1.4249f,1.8464f,-0.5236f,1.2826f,3.7046f,-1.3035f,2.7286f,-1.4518f,-1.9036f,-1.9955f,-0.0342f,-1.0265f,-0.7636f,3.0555f,0.0746f,-3.0751f,-0.1076f,1.7376f,-1.0914f,-1.9435f,-0.2784f,-1.368f,0.4809f,-0.4433f,0.3764f,0.5729f,-2.0595f,1.096f,-1.326f,-2.0211f,-5.0179f,0.5275f,4.0162f,1.8505f,0.3026f,1.9373f,1.4937f,0.2632f,0.5547f,-1.7121f,-0.1562f,0.0304f
};

static const float qwen_vae_std[QWEN_VAE_CHANNELS] = {
    3.2001f,3.2936f,3.4321f,3.0091f,3.1061f,4.0379f,4.0705f,3.791f,3.0785f,3.65f,3.9308f,3.0904f,2.8778f,3.7675f,3.732f,5.0756f,3.2864f,4.0397f,3.1317f,4.0443f,2.9249f,3.9454f,3.0988f,4.2489f,3.4896f,3.8513f,3.9323f,3.4719f,3.7498f,4.283f,3.5694f,4.2467f,3.9037f,3.2947f,5.077f,3.5075f,3.27f,3.4767f,2.8063f,5.1125f,3.5327f,4.7833f,3.1286f,4.1819f,3.8527f,3.8312f,3.5605f,4.3875f,3.9624f,4.0168f,3.5643f,4.055f,5.5614f,4.2963f,4.408f,3.4959f,3.8747f,3.7608f,3.5735f,3.149f,3.7662f,3.6746f,3.4563f,3.8161f
};

bool qwen_vae_validate(const PixModelFile &vae, std::string &error);
PixTensor qwen_vae_encode(const PixModelFile &vae, const PixImage &image, const InferenceProgress &progress);
PixImage qwen_vae_decode(const PixModelFile &vae, PixTensor latents, size_t width, size_t height, const InferenceProgress &progress);

#endif
