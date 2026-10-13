/* This file is a part of 'Pixelina' Image Generator, Editor, and Viewer.
 *
 * (C) Dmitry 'sciloaf' Solovyev, 2026
 *
 * License: MIT
 */
#include "model.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sstream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>


class PixModelReader {
    const unsigned char* data; // bounded mapped-file bytes
    size_t size; // byte limit
public:
    size_t position = 0; // current offset
    PixModelReader(const unsigned char* input, size_t bytes) : data(input),size(std::min(bytes,size_t(MODEL_MAX_HEADER))) {}

    bool integer(unsigned bytes, uint64_t &value)
    {
        if (position > size || bytes > size - position) return false;
        value = 0;
        for (unsigned i = 0; i < bytes; i++) value |= uint64_t(data[position++]) << (i * 8);
        return true;
    }

    bool string(std::string &result)
    {
        uint64_t length;
        if (!integer(8,length) || length > size - position) return false;
        result.assign(reinterpret_cast<const char*>(data + position),size_t(length));
        position += size_t(length);
        return true;
    }

    PixJson metadata(uint32_t type, unsigned depth)
    {
        if (depth > MODEL_JSON_DEPTH) return {PixJson::Invalid,{},{},{}};
        PixJson result;
        if (type == GGUF_TYPE_STRING) {
            result.type = PixJson::String;
            if (!string(result.scalar)) return {PixJson::Invalid,{},{},{}};
        } else if (type == GGUF_TYPE_ARRAY) {
            result.type = PixJson::Array;
            uint64_t element,count;
            if (!integer(4,element) || !integer(8,count) || count > MODEL_MAX_ITEMS || element == GGUF_TYPE_ARRAY) return {PixJson::Invalid,{},{},{}};
            result.array.reserve(size_t(count));
            for (uint64_t i = 0; i < count; i++) {
                PixJson member = metadata(uint32_t(element),depth + 1);
                if (member.type == PixJson::Invalid) return member;
                result.array.push_back(std::move(member));
            }
        } else {
            static const unsigned widths[] = {1,1,2,2,4,4,4,1,0,0,8,8,8};
            uint64_t raw;
            if (type > GGUF_TYPE_FLOAT64 || !integer(widths[type],raw)) return {PixJson::Invalid,{},{},{}};
            result.type = type == 7? PixJson::Boolean : PixJson::Number;
            if (type == 7) {
                if (raw > 1) return {PixJson::Invalid,{},{},{}};
                result.scalar = raw? "true" : "false";
            } else if (type == 6) {
                uint32_t bits = uint32_t(raw);
                float value;
                std::memcpy(&value,&bits,sizeof(value));
                result.scalar = std::to_string(value);
            } else if (type == GGUF_TYPE_FLOAT64) {
                double value;
                std::memcpy(&value,&raw,sizeof(value));
                result.scalar = std::to_string(value);
            } else if (type == 1) result.scalar = std::to_string(int8_t(raw));
            else if (type == 3) result.scalar = std::to_string(int16_t(raw));
            else if (type == 5) result.scalar = std::to_string(int32_t(raw));
            else if (type == GGUF_TYPE_INT64) result.scalar = std::to_string(int64_t(raw));
            else result.scalar = std::to_string(raw);
        }
        return result;
    }
};

uint64_t weight_bytes(const std::vector<uint64_t> &shape, uint32_t type)
{
    if (shape.empty() || shape.size() > MODEL_MAX_RANK) return 0;
    uint64_t block = 1,bytes;
    switch (type) {
        case MODEL_F32:
            bytes = 4;
            break;
        case MODEL_F16:
        case MODEL_BF16:
            bytes = 2;
            break;
        case MODEL_Q8_0:
            block = MODEL_Q8_BLOCK;
            bytes = MODEL_Q8_BYTES;
            break;
        case MODEL_Q5_K:
            block = MODEL_QUANT_BLOCK;
            bytes = MODEL_Q5_BYTES;
            break;
        case MODEL_Q6_K:
            block = MODEL_QUANT_BLOCK;
            bytes = MODEL_Q6_BYTES;
            break;
        default: return 0;
    }
    if (!shape[0] || shape[0] % block) return 0;
    uint64_t count = shape[0] / block;
    for (size_t i = 1; i < shape.size(); i++) {
        if (!shape[i] || count > UINT64_MAX / shape[i]) return 0;
        count *= shape[i];
    }
    if (count > UINT64_MAX / bytes) return 0;
    return count * bytes;
}

PixModelFile::~PixModelFile()
{
    close();
}

void PixModelFile::close()
{
    if (data) munmap(const_cast<unsigned char*>(data),size);
    data = nullptr;
    size = 0;
    metadata.clear();
    weights.clear();
    filename.clear();
}

int PixModelFile::open(const std::string &path, bool safetensors)
{
    close();
    int fd = ::open(path.c_str(),O_RDONLY);
    if (fd < 0) return MODEL_ERROR_OPEN;
    struct stat info;
    if (fstat(fd,&info) || !S_ISREG(info.st_mode) || info.st_size < 8 || uint64_t(info.st_size) > SIZE_MAX) {
        ::close(fd);
        return MODEL_ERROR_FORMAT;
    }
    size = size_t(info.st_size);
    void* mapped = mmap(nullptr,size,PROT_READ,MAP_PRIVATE,fd,0);
    ::close(fd);
    if (mapped == MAP_FAILED) {
        size = 0;
        return MODEL_ERROR_OPEN;
    }
    data = static_cast<const unsigned char*>(mapped);
    filename = path;
    if (read(safetensors)) return 0;
    close();
    return MODEL_ERROR_FORMAT;
}

bool PixModelFile::read(bool safetensors)
{
    PixModelReader reader(data,size);
    if (safetensors) {
        uint64_t length;
        if (!reader.integer(8,length)) return false;
        if (!length || length > MODEL_MAX_HEADER || length > size - reader.position) return false;
        PixJson root = json_parse(std::string(reinterpret_cast<const char*>(data + reader.position),size_t(length)));
        if (root.type != PixJson::Object) return false;
        uint64_t base = 8 + length;
        for (const auto &entry : root.object) {
            if (entry.first == "__metadata__") {
                metadata = entry.second.object;
                continue;
            }
            PixWeight tensor;
            tensor.name = entry.first;
            const PixJson &dtype = entry.second.at("dtype");
            if (dtype.type != PixJson::String) return false;
            if (dtype.scalar == "F32") tensor.type = MODEL_F32;
            else if (dtype.scalar == "F16") tensor.type = MODEL_F16;
            else if (dtype.scalar == "BF16") tensor.type = MODEL_BF16;
            else return false;
            const PixJson &shape = entry.second.at("shape");
            if (shape.type != PixJson::Array) return false;
            for (auto it = shape.array.rbegin(); it != shape.array.rend(); it++) tensor.shape.push_back(it->integer());
            const PixJson &offsets = entry.second.at("data_offsets");
            if (offsets.type != PixJson::Array || offsets.array.size() != 2) return false;
            uint64_t start = offsets.array[0].integer(),end = offsets.array[1].integer();
            tensor.bytes = weight_bytes(tensor.shape,tensor.type);
            if (!tensor.bytes || start == UINT64_MAX || end == UINT64_MAX || end < start || end > size - base || end - start != tensor.bytes) return false;
            tensor.offset = base + start;
            weights.emplace(tensor.name,std::move(tensor));
        }
    } else {
        if (std::memcmp(data,GGUF_MAGIC,4)) return false;
        reader.position = 4;
        uint64_t version;
        if (!reader.integer(4,version) || version != MODEL_GGUF_VERSION) return false;
        uint64_t count,pairs;
        if (!reader.integer(8,count) || !reader.integer(8,pairs)) return false;
        if (!count || count > MODEL_MAX_ITEMS || pairs > MODEL_MAX_ITEMS) return false;
        for (uint64_t i = 0; i < pairs; i++) {
            std::string key;
            uint64_t type;
            if (!reader.string(key) || !reader.integer(4,type)) return false;
            PixJson member = reader.metadata(uint32_t(type),0);
            if (member.type == PixJson::Invalid || !metadata.emplace(key,std::move(member)).second) return false;
        }
        uint64_t alignment = MODEL_GGUF_ALIGNMENT;
        auto found = metadata.find("general.alignment");
        if (found != metadata.end()) alignment = found->second.integer();
        if (!alignment || (alignment & (alignment - 1)) || alignment > MODEL_MAX_HEADER) return false;
        for (uint64_t i = 0; i < count; i++) {
            PixWeight tensor;
            uint64_t rank,type;
            if (!reader.string(tensor.name) || !reader.integer(4,rank)) return false;
            if (!rank || rank > MODEL_MAX_RANK) return false;
            for (uint64_t d = 0; d < rank; d++) {
                uint64_t dimension;
                if (!reader.integer(8,dimension)) return false;
                tensor.shape.push_back(dimension);
            }
            if (!reader.integer(4,type) || !reader.integer(8,tensor.offset)) return false;
            tensor.type = uint32_t(type);
            tensor.bytes = weight_bytes(tensor.shape,tensor.type);
            if (!tensor.bytes || tensor.offset % alignment) return false;
            if (!weights.emplace(tensor.name,std::move(tensor)).second) return false;
        }
        uint64_t base = (reader.position + alignment - 1) & ~(alignment - 1);
        if (base > size) return false;
        for (auto &entry : weights) {
            PixWeight &tensor = entry.second;
            if (tensor.offset > size - base || tensor.bytes > size - base - tensor.offset) return false;
            tensor.offset += base;
        }
    }
    // Reject aliased or overlapping tensor regions before any kernel reads them.
    std::vector<const PixWeight*> ordered;
    for (const auto &entry : weights) ordered.push_back(&entry.second);
    std::sort(ordered.begin(),ordered.end(),[](const PixWeight* a, const PixWeight* b) { return a->offset < b->offset; });
    uint64_t end = 0;
    for (const PixWeight* tensor : ordered) {
        if (tensor->offset < end) return false;
        end = tensor->offset + tensor->bytes;
    }
    if (weights.empty()) return false;
    return true;
}

const PixWeight &PixModelFile::tensor(const std::string &name) const
{
    auto found = weights.find(name);
    return found->second;
}

int model_inspect(const PixSettings &settings, std::string &summary)
{
    std::string error;
    if (!settings_valid(settings,error)) return MODEL_ERROR_SETTINGS;
    PixModelFile diffuser,encoder,vae;
    int result = diffuser.open(settings.diffuser,false);
    if (!result) result = encoder.open(settings.encoder,false);
    if (!result) result = vae.open(settings.vae,true);
    if (result) return result;
    auto vocabulary = encoder.metadata.find("tokenizer.ggml.tokens");
    auto merges = encoder.metadata.find("tokenizer.ggml.merges");
    if (vocabulary == encoder.metadata.end() || merges == encoder.metadata.end() || vocabulary->second.type != PixJson::Array || merges->second.type != PixJson::Array || vocabulary->second.array.empty() || merges->second.array.empty()) return MODEL_ERROR_FORMAT;
    std::ostringstream out;
    out << "Containers: " << diffuser.weights.size() << " diffuser, " << encoder.weights.size() << " text encoder, " << vae.weights.size() << " VAE tensors. Vocabulary: " << vocabulary->second.array.size() << " tokens.";
    summary = out.str();
    return 0;
}
