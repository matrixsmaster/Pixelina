/* This file is a part of 'Pixelina' Image Generator, Editor, and Viewer.
 *
 * (C) Dmitry 'sciloaf' Solovyev, 2026
 *
 * License: MIT
 */
#ifndef PIXELINA_QWEN_VL_H
#define PIXELINA_QWEN_VL_H

#include <array>
#include "inference.h"

#define QWEN_VL_EMBEDDING_WEIGHT "token_embd.weight"
#define QWEN_VL_BLOCK_PREFIX "blk."
#define QWEN_VL_ATTENTION_NORM "attn_norm.weight"
#define QWEN_VL_FFN_NORM "ffn_norm.weight"
#define QWEN_VL_QUERY_NORM "attn_q_norm.weight"
#define QWEN_VL_KEY_NORM "attn_k_norm.weight"
#define QWEN_VISION_PATCH_WEIGHT "v.patch_embd.weight"
#define QWEN_VISION_PATCH_SECOND_WEIGHT "v.patch_embd.weight.1"
#define QWEN_VISION_PATCH_BIAS "v.patch_embd.bias"
#define QWEN_VISION_POSITION_WEIGHT "v.position_embd.weight"
#define QWEN_VISION_BLOCK_PREFIX "v.blk."
#define QWEN_VISION_ATTENTION_NORM "ln1"
#define QWEN_VISION_FFN_NORM "ln2"
#define QWEN_VISION_QKV "attn_qkv"
#define QWEN_VISION_ATTENTION_OUTPUT "attn_out"
#define QWEN_VL_FFN_UP "ffn_up"
#define QWEN_VL_FFN_DOWN "ffn_down"
#define QWEN_VISION_MERGER_PREFIX "mm."
#define QWEN_VISION_DEEP_PREFIX "v.deepstack."
#define QWEN_VISION_FINAL_NORM "v.post_ln"
#define QWEN_VISION_MERGER_NORM "norm"
#define QWEN_VISION_MERGER_FIRST "fc1"
#define QWEN_VISION_MERGER_SECOND "fc2"

#define QWEN_VL_HIDDEN 4096
#define QWEN_VL_MLP 12288
#define QWEN_VL_LAYERS 36
#define QWEN_VL_HEADS 32
#define QWEN_VL_KV_HEADS 8
#define QWEN_VL_HEAD_DIM 128
#define QWEN_VL_VOCABULARY 151936
#define QWEN_VL_CONTEXT 262144
#define QWEN_VL_THETA 5000000.0
#define QWEN_VL_SECTION_H 20
#define QWEN_VL_SECTION_W 20
#define QWEN_VL_AXIS_PERIOD 3
#define QWEN_VISION_HIDDEN 1152
#define QWEN_VISION_MLP 4304
#define QWEN_VISION_LAYERS 27
#define QWEN_VISION_HEADS 16
#define QWEN_VISION_HEAD_DIM 72
#define QWEN_VISION_PATCH 16
#define QWEN_VISION_TEMPORAL_PATCH 2
#define QWEN_VISION_MERGE 2
#define QWEN_VISION_MERGE_PIXELS 4
#define QWEN_VISION_POSITION_SIDE 48
#define QWEN_VISION_POSITIONS 2304
#define QWEN_VISION_THETA 10000.0
#define QWEN_VISION_DEEPSTACK 3
#define QWEN_VISION_DEEP_LAYERS {8,16,24}
#define QWEN_VL_SECTION_T 24
#define QWEN_VISION_MIN_PIXELS 65536
#define QWEN_VISION_MAX_PIXELS 16777216
#define QWEN_VL_IMAGE_TOKEN 151655
#define QWEN_VL_VISION_START 151652
#define QWEN_VL_VISION_END 151653
#define QWEN_VL_SYSTEM "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
#define QWEN_VL_USER "<|im_start|>user\n"
#define QWEN_VL_IMAGE_PAD "<|image_pad|>"
#define QWEN_VL_REFERENCE_PREFIX "<image"
#define QWEN_VL_VISION_START_TEXT "<|vision_start|>"
#define QWEN_VL_VISION_END_TEXT "<|vision_end|>"
#define QWEN_VL_EPSILON 0.000001f
#define QWEN_VL_VISION_FRACTION 0.12f
#define QWEN_VL_LANGUAGE_FRACTION 0.20f
#define QWEN_VL_PIXELS_SCALE (1.0f / 255.0f)
#define QWEN_VL_NORMALIZE_SCALE 2.0f

struct PixVisionGrid {
    size_t temporal = 1; // still-image frame grid
    size_t height = 0,width = 0; // patch grid dimensions
    size_t reference = 0; // source order
};

struct PixVLInput {
    std::vector<int> ids; // expanded complete raw-template token IDs
    std::vector<unsigned char> image_slots; // one flag per expanded language row
    std::array<std::vector<size_t>,3> positions; // temporal, height and width language positions
    std::vector<PixVisionGrid> grids; // reference patch grids in input order
    std::vector<PixTensor> patches; // channel/temporal/patch-packed normalized pixels
    size_t trim = 0; // token-derived system-prefix count
};

struct PixVisionFeatures {
    PixTensor merged; // final merger features in reference order
    std::array<PixTensor,QWEN_VISION_DEEPSTACK> deepstack; // separately merged selected block features
    std::vector<size_t> offsets; // beginning of each reference's merged rows
};

struct PixQwenCondition {
    PixTensor hidden; // trimmed output of the final language block before RMSNorm
    std::vector<unsigned char> image_slots; // matching reference-image slot flags
};

struct PixVLCacheLayer {
    PixTensor key,value; // incrementally populated normalized/rotated keys and projected values
};

struct PixVLCache {
    std::array<PixVLCacheLayer,QWEN_VL_LAYERS> layers; // capacity-sized per-layer buffers
    size_t rows = 0,capacity = 0; // populated prefix and admitted row capacity
};

bool qwen_vl_validate(const PixModelFile &language, const PixModelFile &vision, std::string &error);
PixVLInput qwen_vl_prepare(const PixTokenz &tokenizer, const std::string &prompt, const std::vector<PixImage> &images, bool pack_pixels = true);
PixVisionFeatures qwen_vl_vision(const PixModelFile &vision, const PixVLInput &input, const InferenceProgress &progress);
PixQwenCondition qwen_vl_language(const PixModelFile &language, const PixVLInput &input, PixVisionFeatures features, const InferenceProgress &progress);
PixTensor qwen_vl_forward(const PixModelFile &language, const PixVLInput &input, PixVisionFeatures features, const InferenceProgress &progress, PixVLCache* cache = nullptr);

#endif
