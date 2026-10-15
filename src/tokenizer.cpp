/* This file is a part of 'Pixelina' Image Generator, Editor, and Viewer.
 *
 * (C) Dmitry 'sciloaf' Solovyev, 2026
 *
 * License: MIT
 */
#include "tokenizer.h"
#include <algorithm>

static unsigned ascii_category(unsigned char code)
{
    if ((code >= 'A' && code <= 'Z') || (code >= 'a' && code <= 'z')) return TOKENIZER_LETTER;
    if (code >= '0' && code <= '9') return TOKENIZER_NUMBER;
    return 0;
}

static bool ascii_space(unsigned char code)
{
    return (code >= '\t' && code <= '\r') || code == ' ';
}

bool PixTokenz::load(const PixModelFile &encoder)
{
    vocabulary.clear();
    ranks.clear();
    specials.clear();
    pieces.clear();
    special_pieces.clear();
    auto pre = encoder.metadata.find("tokenizer.ggml.pre");
    auto model = encoder.metadata.find("tokenizer.ggml.model");
    if (pre == encoder.metadata.end() || pre->second.scalar != "qwen2" || model == encoder.metadata.end() || model->second.scalar != "gpt2") return false;
    for (const char* key : {"tokenizer.ggml.tokens","tokenizer.ggml.token_type","tokenizer.ggml.merges"}) {
        auto found = encoder.metadata.find(key);
        if (found == encoder.metadata.end() || found->second.type != PixJson::Array) return false;
    }
    const auto &tokens = encoder.metadata.find("tokenizer.ggml.tokens")->second.array;
    const auto &types = encoder.metadata.find("tokenizer.ggml.token_type")->second.array;
    if (tokens.empty() || types.size() != tokens.size()) return false;
    for (size_t i = 0; i < tokens.size(); i++) {
        if (tokens[i].type != PixJson::String || !vocabulary.emplace(tokens[i].scalar,int(i)).second) return false;
        if (tokens[i].scalar.empty() || types[i].integer() == UINT64_MAX) return false;
        if (types[i].integer() == TOKENIZER_SPECIAL_TYPE || types[i].integer() == TOKENIZER_USER_TYPE) specials.push_back(tokens[i].scalar);
    }
    const auto &merges = encoder.metadata.find("tokenizer.ggml.merges")->second.array;
    for (size_t i = 0; i < merges.size(); i++) {
        if (merges[i].type != PixJson::String || merges[i].scalar.find(' ') == std::string::npos || !ranks.emplace(merges[i].scalar,unsigned(i)).second) return false;
    }
    unsigned extra = TOKENIZER_BYTE_COUNT;
    for (unsigned byte = 0; byte < TOKENIZER_BYTE_COUNT; byte++) {
        bool printable = (byte >= GPT_BYTE_PRINTABLE_FIRST && byte <= GPT_BYTE_PRINTABLE_LAST) || (byte >= GPT_BYTE_LATIN_FIRST && byte <= GPT_BYTE_LATIN_SPLIT) || byte >= GPT_BYTE_LATIN_RESUME;
        // The model's fixed byte alphabet uses one- or two-byte vocabulary keys.
        unsigned value = printable? byte : extra++;
        if (value < TOKENIZER_ASCII_LIMIT) bytes[byte] = char(value);
        else {
            bytes[byte] = char(TOKENIZER_KEY_PREFIX | (value >> TOKENIZER_KEY_BITS));
            bytes[byte] += char(TOKENIZER_KEY_CONTINUATION | (value & TOKENIZER_KEY_MASK));
        }
        if (vocabulary.find(bytes[byte]) == vocabulary.end()) return false;
    }
    std::sort(specials.begin(),specials.end(),[](const std::string &a, const std::string &b) { return a.size() > b.size(); });
    std::unordered_map<std::string,unsigned char> original_bytes;
    for (unsigned byte = 0; byte < TOKENIZER_BYTE_COUNT; byte++) original_bytes.emplace(bytes[byte],byte);
    pieces.resize(tokens.size());
    special_pieces.resize(tokens.size());
    for (size_t i = 0; i < tokens.size(); i++) {
        if (types[i].integer() == TOKENIZER_SPECIAL_TYPE || types[i].integer() == TOKENIZER_USER_TYPE) {
            special_pieces[i] = 1;
            pieces[i] = tokens[i].scalar;
            continue;
        }
        const std::string &token = tokens[i].scalar;
        for (size_t cursor = 0; cursor < token.size();) {
            size_t length = static_cast<unsigned char>(token[cursor]) < TOKENIZER_ASCII_LIMIT? 1 : 2;
            auto found = original_bytes.find(token.substr(cursor,length));
            if (found == original_bytes.end()) return false;
            pieces[i] += char(found->second);
            cursor += length;
        }
    }
    return true;
}

bool PixTokenz::bpe(const std::string &piece, std::vector<int> &ids) const
{
    std::vector<std::string> parts;
    for (unsigned char byte : piece) parts.push_back(bytes[byte]);
    while (parts.size() > 1) {
        unsigned best = UINT32_MAX;
        size_t index = 0;
        for (size_t i = 0; i + 1 < parts.size(); i++) {
            auto found = ranks.find(parts[i] + " " + parts[i + 1]);
            if (found != ranks.end() && found->second < best) {
                best = found->second;
                index = i;
            }
        }
        if (best == UINT32_MAX) break;
        // Merge every occurrence of the selected pair, as in the BPE reference.
        std::string first = parts[index],second = parts[index + 1];
        for (size_t i = 0; i + 1 < parts.size();) {
            if (parts[i] == first && parts[i + 1] == second) {
                parts[i] += parts[i + 1];
                parts.erase(parts.begin() + i + 1);
            } else i++;
        }
    }
    for (const std::string &part : parts) {
        auto found = vocabulary.find(part);
        if (found == vocabulary.end()) return false;
        ids.push_back(found->second);
    }
    return true;
}

bool PixTokenz::ordinary(const std::string &text, std::vector<int> &ids) const
{
    size_t start = 0;
    while (start < text.size()) {
        size_t end = start;
        if (text[start] == '\'') {
            static const char* endings[] = {"s","t","re","ve","m","ll","d"};
            for (const char* ending : endings) {
                size_t i = start + 1,j = 0;
                while (ending[j] && i < text.size() && (text[i] >= 'A' && text[i] <= 'Z'? text[i] - 'A' + 'a' : text[i]) == ending[j]) {
                    i++;
                    j++;
                }
                if (!ending[j]) {
                    end = i;
                    break;
                }
            }
        }
        if (end == start) {
            size_t letters = start;
            unsigned category = ascii_category(text[start]);
            if (text[start] != '\r' && text[start] != '\n' && !category && start + 1 < text.size() && ascii_category(text[start + 1]) == TOKENIZER_LETTER) letters++;
            if (ascii_category(text[letters]) == TOKENIZER_LETTER) {
                end = letters + 1;
                while (end < text.size() && ascii_category(text[end]) == TOKENIZER_LETTER) end++;
            } else if (category == TOKENIZER_NUMBER) end = start + 1;
        }
        if (end == start) {
            size_t punctuation = start + (text[start] == ' '? 1 : 0);
            if (punctuation < text.size() && !ascii_space(text[punctuation]) && !ascii_category(text[punctuation])) {
                end = punctuation + 1;
                while (end < text.size() && !ascii_space(text[end]) && !ascii_category(text[end])) end++;
                while (end < text.size() && (text[end] == '\r' || text[end] == '\n')) end++;
            }
        }
        if (end == start && ascii_space(text[start])) {
            size_t whites = start,last_newline = start;
            while (whites < text.size() && ascii_space(text[whites])) {
                if (text[whites] == '\r' || text[whites] == '\n') last_newline = whites + 1;
                whites++;
            }
            if (last_newline > start) end = last_newline;
            else if (whites < text.size() && whites - start > 1) end = whites - 1;
            else end = whites;
        }
        if (end == start) end++;
        if (!bpe(text.substr(start,end - start),ids)) return false;
        start = end;
    }
    return true;
}

std::vector<int> PixTokenz::encode(const std::string &text, bool chat, bool special) const
{
    std::string input = chat? std::string(TOKENIZER_CHAT_PREFIX) + text + TOKENIZER_CHAT_SUFFIX : text;
    std::vector<int> ids;
    if (!special) {
        if (!ordinary(input,ids) || (chat && ids.size() > MODEL_QWEN_CONTEXT)) return {};
        return ids;
    }
    size_t cursor = 0;
    while (cursor < input.size()) {
        size_t next = input.size();
        const std::string* selected = nullptr;
        for (const std::string &token : specials) {
            size_t found = input.find(token,cursor);
            if (found < next) {
                next = found;
                selected = &token;
            }
        }
        if (!ordinary(input.substr(cursor,next - cursor),ids)) return {};
        if (!selected) break;
        ids.push_back(vocabulary.find(*selected)->second);
        cursor = next + selected->size();
    }
    if (chat && ids.size() > MODEL_QWEN_CONTEXT) return {};
    return ids;
}

std::string PixTokenz::decode(int id, bool special) const
{
    if (id < 0 || size_t(id) >= pieces.size()) return {};
    if (!special && special_pieces[size_t(id)]) return {};
    return pieces[size_t(id)];
}
