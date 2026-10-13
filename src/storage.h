/* This file is a part of 'Pixelina' Image Generator, Editor, and Viewer.
 *
 * (C) Dmitry 'sciloaf' Solovyev, 2026
 *
 * License: MIT
 */
#ifndef PIXELINA_STORAGE_H
#define PIXELINA_STORAGE_H

#include <cstdint>
#include <string>
#include <vector>

#define PADDED_TOKEN_ZERO_POS
#define STBI_MAX_DIMENSIONS 16384
#define IMAGE_TEMP_SUFFIX ".pixelina-export-XXXXXX"
#define PIXELINA_NAME "Pixelina"
#define PIXELINA_CPU "CPU"
#define PIXELINA_VULKAN "Vulkan"
#define PIXELINA_JPG "JPG"
#define PIXELINA_PNG "PNG"
#define PIXELINA_EXT_JPG ".jpg"
#define PIXELINA_EXT_JPEG ".jpeg"
#define PIXELINA_EXT_PNG ".png"
#define PIXELINA_DEFAULT_DIFFUSER "z-image-turbo-Q8_0.gguf"
#define PIXELINA_DEFAULT_ENCODER "Qwen3-4B-Q8_0.gguf"
#define PIXELINA_DEFAULT_VAE "ae.safetensors"
#define PIXELINA_DB "pixelina.txt"
#define PIXELINA_VERSION "$v1;"
#define PIXELINA_JPG_QUALITY 95
#define PIXELINA_IMAGE_MULTIPLE 16
#define PIXELINA_DEFAULT_SEED 1337
#define PIXELINA_DEFAULT_SIZE 1024
#define PIXELINA_DEFAULT_STEPS 8
#define PIXELINA_DEFAULT_SHIFT 3.0f
#define PIXELINA_MAX_PIXELS (128u * 1024u * 1024u)
#define PIXELINA_THUMB_SIZE 112
#define PIXELINA_THUMB_CACHE 32
#define PIXELINA_MAX_IMAGE_NUMBER 99999
#define PIXELINA_RECORD_FIELDS 11
#define PIXELINA_TEMP_PATTERN ".pixelina-XXXXXX"

struct PixSettings {
    uint64_t seed = PIXELINA_DEFAULT_SEED; // actual noise seed, retained for reproduction
    int width = PIXELINA_DEFAULT_SIZE; // output width in pixels
    int height = PIXELINA_DEFAULT_SIZE; // output height in pixels
    int steps = PIXELINA_DEFAULT_STEPS; // diffuser evaluations
    float shift = PIXELINA_DEFAULT_SHIFT; // flow-matching schedule shift
    std::string diffuser = PIXELINA_DEFAULT_DIFFUSER; // selected diffuser file
    std::string encoder = PIXELINA_DEFAULT_ENCODER; // selected text encoder file
    std::string vae = PIXELINA_DEFAULT_VAE; // selected VAE file
    std::string backend = PIXELINA_CPU; // selected execution backend
    std::string format = PIXELINA_JPG; // output format for Save
};

struct PixImage {
    int width = 0; // decoded width
    int height = 0; // decoded height
    std::vector<unsigned char> rgb; // interleaved RGB pixels
};

struct PixRecord {
    PixSettings settings; // UI parameters
    std::string filename; // explicit saved-image association
    std::string prompt; // literal multiline prompt
};

struct PixDirEntry {
    std::string name; // entry name, without parent path
    bool directory; // directory selector navigation flag
};

bool parse_unsigned(const std::string &text, uint64_t &value);
bool parse_float(const std::string &text, float &value);
bool settings_valid(const PixSettings &settings, std::string &error);
bool database_parse(const std::string &text, std::vector<PixRecord> &records, std::string &error);
std::string database_record(const PixRecord &record);
bool database_load(const std::string &directory, std::vector<PixRecord> &records, std::string &error);
bool storage_save(const std::string &directory, PixRecord &record, const PixImage &image, std::string &error);
bool image_decode(const unsigned char* bytes, size_t count, PixImage &image, std::string &error);
bool image_load(const std::string &path, PixImage &image, std::string &error);
bool image_save(const std::string &path, const PixImage &image, std::string &error);
PixImage image_thumbnail(const PixImage &image);
std::string path_join(const std::string &directory, const std::string &name);
bool directory_list(const std::string &path, std::vector<PixDirEntry> &entries, std::string &error);

#endif
