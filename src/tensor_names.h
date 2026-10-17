/* This file is a part of 'Pixelina' Image Generator, Editor, and Viewer.
 *
 * (C) Dmitry 'sciloaf' Solovyev, 2026
 *
 * License: MIT
 */
#ifndef PIXELINA_TENSOR_NAMES_H
#define PIXELINA_TENSOR_NAMES_H

// Common tensor suffixes and separators.
#define TENSOR_NAME_SEPARATOR "."
#define TENSOR_SLICE_SEPARATOR ":"
#define WEIGHT_SUFFIX TENSOR_NAME_SEPARATOR "weight"
#define BIAS_SUFFIX TENSOR_NAME_SEPARATOR "bias"

// Qwen language and vision tensors.
#define QWEN_VL_EMBEDDING_WEIGHT "token_embd" WEIGHT_SUFFIX
#define QWEN_VL_BLOCK_PREFIX "blk."
#define QWEN_VL_ATTENTION_NORM "attn_norm" WEIGHT_SUFFIX
#define QWEN_VL_FFN_NORM "ffn_norm" WEIGHT_SUFFIX
#define QWEN_VL_QUERY_NORM "attn_q_norm" WEIGHT_SUFFIX
#define QWEN_VL_KEY_NORM "attn_k_norm" WEIGHT_SUFFIX
#define QWEN_VISION_PATCH_NAME "v.patch_embd"
#define QWEN_VISION_PATCH_WEIGHT QWEN_VISION_PATCH_NAME WEIGHT_SUFFIX
#define QWEN_VISION_PATCH_SECOND_WEIGHT QWEN_VISION_PATCH_WEIGHT TENSOR_NAME_SEPARATOR "1"
#define QWEN_VISION_PATCH_BIAS QWEN_VISION_PATCH_NAME BIAS_SUFFIX
#define QWEN_VISION_POSITION_WEIGHT "v.position_embd" WEIGHT_SUFFIX
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
#define QWEN_CHAT_OUTPUT "output"
#define QWEN_CHAT_OUTPUT_NORM "output_norm" WEIGHT_SUFFIX
#define QWEN_VL_QUERY "attn_q"
#define QWEN_VL_KEY "attn_k"
#define QWEN_VL_VALUE "attn_v"
#define QWEN_VL_ATTENTION_OUTPUT "attn_output"
#define QWEN_VL_FFN_GATE "ffn_gate"
#define QWEN_VISION_FINAL_MERGER_FIRST "0"
#define QWEN_VISION_FINAL_MERGER_SECOND "2"

// Z-Image diffuser tensors.
#define Z_ATTENTION_PREFIX "attention."
#define Z_ATTENTION_QKV Z_ATTENTION_PREFIX "qkv"
#define Z_ATTENTION_OUTPUT Z_ATTENTION_PREFIX "out"
#define Z_QUERY_NORM Z_ATTENTION_PREFIX "q_norm" WEIGHT_SUFFIX
#define Z_KEY_NORM Z_ATTENTION_PREFIX "k_norm" WEIGHT_SUFFIX
#define Z_ATTENTION_NORM1 "attention_norm1"
#define Z_ATTENTION_NORM2 "attention_norm2"
#define Z_FFN_NORM1 "ffn_norm1"
#define Z_FFN_NORM2 "ffn_norm2"
#define Z_FFN_GATE "feed_forward.w1"
#define Z_FFN_UP "feed_forward.w3"
#define Z_FFN_DOWN "feed_forward.w2"
#define Z_MODULATION "adaLN_modulation.0"
#define Z_CAPTION_NORM "cap_embedder.0" WEIGHT_SUFFIX
#define Z_CAPTION_PROJECTION "cap_embedder.1"
#define Z_CAPTION_PAD "cap_pad_token"
#define Z_IMAGE_PAD "x_pad_token"
#define Z_IMAGE_PROJECTION "x_embedder"
#define Z_TIME_FIRST "t_embedder.mlp.0"
#define Z_TIME_SECOND "t_embedder.mlp.2"
#define Z_OUTPUT_PROJECTION "final_layer.linear"
#define Z_OUTPUT_MODULATION "final_layer.adaLN_modulation.1"
#define Z_CONTEXT_REFINER_PREFIX "context_refiner."
#define Z_NOISE_REFINER_PREFIX "noise_refiner."
#define Z_BLOCK_PREFIX "layers."

// Z-Image VAE tensors.
#define VAE_NORM_SUFFIX TENSOR_NAME_SEPARATOR "norm"
#define VAE_DECODER_PREFIX "decoder" TENSOR_NAME_SEPARATOR
#define VAE_MID_PREFIX VAE_DECODER_PREFIX "mid" TENSOR_NAME_SEPARATOR
#define VAE_ATTENTION_PREFIX VAE_MID_PREFIX "attn_1"
#define VAE_UP_PREFIX VAE_DECODER_PREFIX "up" TENSOR_NAME_SEPARATOR
#define VAE_INPUT_CONV VAE_DECODER_PREFIX "conv_in"
#define VAE_MID_BLOCK_FIRST VAE_MID_PREFIX "block_1"
#define VAE_MID_BLOCK_SECOND VAE_MID_PREFIX "block_2"
#define VAE_NORM1_SUFFIX VAE_NORM_SUFFIX "1"
#define VAE_NORM2_SUFFIX VAE_NORM_SUFFIX "2"
#define VAE_CONV1_SUFFIX ".conv1"
#define VAE_CONV2_SUFFIX ".conv2"
#define VAE_SHORTCUT_SUFFIX ".nin_shortcut"
#define VAE_QUERY_SUFFIX ".q"
#define VAE_KEY_SUFFIX ".k"
#define VAE_VALUE_SUFFIX ".v"
#define VAE_PROJECTION_SUFFIX ".proj_out"
#define VAE_BLOCK_PREFIX ".block."
#define VAE_UPSAMPLE_SUFFIX ".upsample.conv"
#define VAE_OUTPUT_NORM VAE_DECODER_PREFIX "norm_out"
#define VAE_OUTPUT_CONV VAE_DECODER_PREFIX "conv_out"

// Qwen Image diffuser tensors.
#define QWEN_IMAGE_TEXT_NORM "txt_in.text_norm" WEIGHT_SUFFIX
#define QWEN_IMAGE_BLOCK_PREFIX "transformer_blocks."
#define QWEN_IMAGE_QUERY_NORM "attn.norm_q" WEIGHT_SUFFIX
#define QWEN_IMAGE_KEY_NORM "attn.norm_k" WEIGHT_SUFFIX
#define QWEN_IMAGE_INPUT_PROJECTION "img_in"
#define QWEN_IMAGE_FFN_FUSED "img_mlp.gate_up"
#define QWEN_IMAGE_FFN_FUSED_WEIGHT QWEN_IMAGE_FFN_FUSED WEIGHT_SUFFIX
#define QWEN_IMAGE_OUTPUT_PROJECTION "proj_out"
#define QWEN_IMAGE_TEXT_INPUT "txt_in.in_layer"
#define QWEN_IMAGE_TEXT_OUTPUT "txt_in.out_layer"
#define QWEN_IMAGE_TIME_FIRST "time_text_embed.timestep_embedder.linear_1"
#define QWEN_IMAGE_TIME_SECOND "time_text_embed.timestep_embedder.linear_2"
#define QWEN_IMAGE_MODULATION "modulation.1"
#define QWEN_IMAGE_OUTPUT_NORM "norm_out.linear"
#define QWEN_IMAGE_QUERY "attn.to_q"
#define QWEN_IMAGE_KEY "attn.to_k"
#define QWEN_IMAGE_VALUE "attn.to_v"
#define QWEN_IMAGE_ATTENTION_OUTPUT "attn.to_out.0"
#define QWEN_IMAGE_FFN_OUTPUT "img_mlp.out"

// Qwen Image VAE tensors.
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
#define QWEN_VAE_ENCODER "encoder."
#define QWEN_VAE_DECODER VAE_DECODER_PREFIX
#define QWEN_VAE_NORM1 ".residual.0.gamma"
#define QWEN_VAE_CONV1 ".residual.2"
#define QWEN_VAE_NORM2 ".residual.3.gamma"
#define QWEN_VAE_CONV2 ".residual.6"
#define QWEN_VAE_SHORTCUT ".shortcut"
#define QWEN_VAE_RESAMPLE ".resample.1"
#define QWEN_VAE_TIME_CONV ".time_conv"
#define QWEN_VAE_FIRST_BLOCK "0"

#endif
