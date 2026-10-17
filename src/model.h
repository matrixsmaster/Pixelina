/* This file is a part of 'Pixelina' Image Generator, Editor, and Viewer.
 *
 * (C) Dmitry 'sciloaf' Solovyev, 2026
 *
 * License: MIT
 */
#ifndef PIXELINA_MODEL_H
#define PIXELINA_MODEL_H

#include <map>
#include "json.h"
#include "storage.h"
#include "model_types.h"
#include "tensor_names.h"

#define MODEL_ERROR_OPEN 1
#define MODEL_ERROR_FORMAT 2
#define MODEL_ERROR_SETTINGS 3
#define MODEL_MAX_RANK 5
#define MODEL_MAX_HEADER (64u * 1024u * 1024u)
#define MODEL_MAX_ITEMS 1000000u
#define MODEL_JSON_DEPTH 64
#define MODEL_GGUF_VERSION 3
#define MODEL_GGUF_ALIGNMENT 32
#define MODEL_QWEN_CONTEXT 40960
#define GGUF_TYPE_STRING 8
#define GGUF_TYPE_ARRAY 9
#define GGUF_TYPE_UINT64 10
#define GGUF_TYPE_INT64 11
#define GGUF_TYPE_FLOAT64 12
#define GGUF_MAGIC "GGUF"

struct PixWeight {
    std::string name; // exact container tensor name
    std::vector<uint64_t> shape; // GGML order, first dimension contiguous
    uint32_t type; // GGML dtype identifier
    uint64_t offset; // absolute mapped-file offset
    uint64_t bytes; // bounded tensor storage length
};

class PixModelFile {
    bool read(bool safetensors);
public:
    std::map<std::string,PixJson> metadata; // embedded tokenizer and model metadata
    std::map<std::string,PixWeight> weights; // independently parsed tensor directory
    std::string filename; // source artifact identity
    const unsigned char* data = nullptr; // read-only mapped contents
    size_t size = 0; // total file bytes
    PixModelFile() = default;
    ~PixModelFile();
    PixModelFile(const PixModelFile &) = delete;
    PixModelFile &operator=(const PixModelFile &) = delete;
    int open(const std::string &path, bool safetensors); // zero on success, MODEL_ERROR_* on failure
    void close();
    const PixWeight &tensor(const std::string &name) const;
};

uint64_t weight_bytes(const std::vector<uint64_t> &shape, uint32_t type);
float half_float(uint16_t value);
void cpu_decode(const unsigned char* data, uint32_t type, size_t count, float* output);
bool cpu_matmul(const PixModelFile &file, const PixWeight &weight, const float* input, size_t columns, float* output);
int model_inspect(const PixSettings &settings, std::string &summary);

#endif
