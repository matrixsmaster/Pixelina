/* This file is a part of 'Pixelina' Image Generator, Editor, and Viewer.
 *
 * (C) Dmitry 'sciloaf' Solovyev, 2026
 *
 * License: MIT
 */
#ifndef PIXELINA_TOKENIZER_H
#define PIXELINA_TOKENIZER_H

#include "model.h"
#include <unordered_map>

#define TOKENIZER_LETTER 1
#define TOKENIZER_NUMBER 2
#define TOKENIZER_SPECIAL_TYPE 3
#define TOKENIZER_USER_TYPE 4
#define TOKENIZER_CHAT_PREFIX "<|im_start|>user\n"
#define TOKENIZER_CHAT_SUFFIX "<|im_end|>\n<|im_start|>assistant\n"
#define TOKENIZER_ASCII_LIMIT 128
#define TOKENIZER_BYTE_COUNT 256
#define TOKENIZER_KEY_PREFIX 0xc0
#define TOKENIZER_KEY_CONTINUATION 0x80
#define TOKENIZER_KEY_BITS 6
#define TOKENIZER_KEY_MASK 63
#define GPT_BYTE_PRINTABLE_FIRST 33
#define GPT_BYTE_PRINTABLE_LAST 126
#define GPT_BYTE_LATIN_FIRST 161
#define GPT_BYTE_LATIN_SPLIT 172
#define GPT_BYTE_LATIN_RESUME 174
#define GPT_BYTE_LATIN_LAST 255

class PixTokenz
{
    std::unordered_map<std::string,int> vocabulary; // exact embedded token-to-ID map
    std::unordered_map<std::string,unsigned> ranks; // merge pair ranks
    std::vector<std::string> specials; // embedded special-token spellings
    std::vector<std::string> pieces; // token IDs decoded back to original UTF-8 bytes
    std::vector<unsigned char> special_pieces; // IDs whose decoded pieces are template/control tokens
    std::string bytes[TOKENIZER_BYTE_COUNT]; // fixed byte-to-vocabulary strings
    bool bpe(const std::string &piece, std::vector<int> &ids) const;
    bool ordinary(const std::string &text, std::vector<int> &ids) const;
public:
    bool load(const PixModelFile &encoder);
    std::vector<int> encode(const std::string &text, bool chat = true, bool special = true) const;
    std::string decode(int id, bool special = false) const;
};

#endif
