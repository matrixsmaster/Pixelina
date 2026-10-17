/* This file is a part of 'Pixelina' Image Generator, Editor, and Viewer.
 *
 * (C) Dmitry 'sciloaf' Solovyev, 2026
 *
 * License: MIT
 */
#ifndef PIXELINA_QWEN_CHAT_H
#define PIXELINA_QWEN_CHAT_H

#include "qwen_vl.h"
#include <random>

#define QWEN_CHAT_END 151645
#define QWEN_CHAT_EOS 151643
#define QWEN_CHAT_TOP_K 20
#define QWEN_CHAT_TOP_P 0.8f
#define QWEN_CHAT_TEMPERATURE 0.7f
#define QWEN_CHAT_PRESENCE 1.5f
#define QWEN_CHAT_THINK_TOP_P 0.95f
#define QWEN_CHAT_THINK_TEMPERATURE 0.6f
#define QWEN_CHAT_CACHE_BLOCK 256
#define QWEN_CHAT_PREFILL_BLOCK 128
#define QWEN_CHAT_LOGITS_BLOCK 4096
#define QWEN_CHAT_ASSISTANT "<|im_start|>assistant\n"
#define QWEN_CHAT_THINK "<think>"
#define QWEN_CHAT_THINK_END "</think>"
#define QWEN_CHAT_MESSAGE_END "<|im_end|>\n"
#define QWEN_CHAT_MEMORY_MAPS 4
#define QWEN_CHAT_VISION_MAPS 8
#define QWEN_CHAT_FFN_MAPS 3
#define QWEN_CHAT_UTF8_CONTINUATION 0x80
#define QWEN_CHAT_UTF8_MASK 0xc0
#define QWEN_CHAT_UTF8_THREE 0xe0
#define QWEN_CHAT_UTF8_FOUR 0xf0

struct PixChatMessage {
    bool assistant = false; // user or assistant role in the ordered conversation
    std::string text; // literal message bytes without template tokens
};

struct PixChatSampling {
    float temperature = QWEN_CHAT_TEMPERATURE; // official Qwen3-VL Instruct temperature
    float top_p = QWEN_CHAT_TOP_P; // cumulative probability threshold after TopK
    float presence = QWEN_CHAT_PRESENCE; // official presence penalty; repetition penalty is unity
};

using PixChatStream = std::function<bool(const std::string &)>;
using PixChatInference = std::function<bool(const PixEditSettings &,const PixImage &,const std::vector<PixChatMessage> &,std::string &,std::string &,const InferenceProgress &,const PixChatStream &)>;

PixVLInput qwen_chat_prepare(const PixTokenz &tokenizer, const std::vector<PixChatMessage> &messages, const PixImage &image, bool pack_pixels = true, bool thinking = false);
int qwen_chat_sample(const PixTensor &logits, const std::vector<unsigned char> &seen, const PixChatSampling &sampling, std::mt19937_64 &random);
size_t qwen_chat_utf8_prefix(const std::string &text);
bool qwen_chat_memory(size_t capacity, size_t rows, uint64_t features, uint64_t device_bytes, uint64_t max_buffer, uint64_t largest_weight, uint64_t &activation, uint64_t &scratch);
bool qwen_chat_reserve(PixVLCache &cache, size_t required);
bool qwen_chat_generate(const PixEditSettings &settings, const PixImage &image, const std::vector<PixChatMessage> &messages, std::string &response, std::string &error, const InferenceProgress &progress, const PixChatStream &stream);

#endif
