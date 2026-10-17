/* This file is a part of 'Pixelina' Image Generator, Editor, and Viewer.
 *
 * (C) Dmitry 'sciloaf' Solovyev, 2026
 *
 * License: MIT
 */
#ifndef PIXELINA_STORAGE_H
#define PIXELINA_STORAGE_H

#include "os.h"
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#define PIXELINA_NAME "Pixelina"
#define PIXELINA_CPU "CPU"
#define PIXELINA_VULKAN "Vulkan"
#define PIXELINA_JPG "JPG"
#define PIXELINA_PNG "PNG"
#define PIXELINA_EXT_JPG ".jpg"
#define PIXELINA_EXT_JPEG ".jpeg"
#define PIXELINA_EXT_PNG ".png"
#define PIXELINA_DB "pixelina.txt"
#define PIXELINA_USER_DIRECTORY ".pixelina"
#define PIXELINA_BOOKMARKS "bookmarks.txt"
#define PIXELINA_GLOBAL_DB "data.txt"
#define PIXELINA_CONFIG "config.txt"
#define PIXELINA_LOG_GEN_VERSION "$v1;"
#define PIXELINA_LOG_GEN_FIELDS 11
#define PIXELINA_LOG_EDIT_VERSION "$v2;"
#define PIXELINA_LOG_EDIT_FIELDS 13
#define PIXELINA_EDIT_MAX_STEPS (1 << 24)
#define PIXELINA_EDIT_DEF_REFRES 1024
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
#define PIXELINA_TEMP_PATTERN ".pixelina-XXXXXX"
#define QWEN_IMAGE_MULTIPLE 32

struct PixSettings {
    uint64_t seed = PIXELINA_DEFAULT_SEED; // actual noise seed, retained for reproduction
    int width = PIXELINA_DEFAULT_SIZE; // output width in pixels
    int height = PIXELINA_DEFAULT_SIZE; // output height in pixels
    int steps = PIXELINA_DEFAULT_STEPS; // diffuser evaluations
    float shift = PIXELINA_DEFAULT_SHIFT; // flow-matching schedule shift
    std::string diffuser; // selected diffuser file
    std::string encoder; // selected text encoder file
    std::string vae; // selected VAE file
    std::string backend = PIXELINA_CPU; // selected execution backend
    std::string format = PIXELINA_JPG; // output format for Save
};

enum PixModelKind {
    PIX_MODEL_Z_IMAGE,PIX_MODEL_QWEN_IMAGE
};

struct PixEditSettings {
    uint64_t seed = PIXELINA_DEFAULT_SEED; // noise seed
    int width = PIXELINA_DEFAULT_SIZE; // output pixel width
    int height = PIXELINA_DEFAULT_SIZE; // output pixel height
    int steps = PIXELINA_DEFAULT_STEPS; // Euler evaluations
    int reference_resolution = PIXELINA_EDIT_DEF_REFRES; // reference area-side parameter
    std::string diffuser; // selected diffuser GGUF
    std::string encoder; // selected language GGUF
    std::string vision; // matching vision/projector GGUF
    std::string vae; // dedicated RGBA VAE safetensors
    std::string backend = PIXELINA_CPU; // backend captured in the record
    std::string format = PIXELINA_JPG; // output format captured in the record
};

struct PixImage {
    int width = 0; // decoded width
    int height = 0; // decoded height
    std::vector<unsigned char> rgb; // interleaved RGB pixels
    std::vector<unsigned char> alpha; // optional row-major alpha, empty for opaque RGB
};

struct PixRecord {
    PixModelKind model = PIX_MODEL_Z_IMAGE; // versioned model family
    PixSettings settings; // UI parameters
    PixEditSettings edit_settings; // Qwen Image parameters
    std::vector<std::string> references; // persisted reference paths in input order
    std::string filename; // explicit saved-image association
    std::string prompt; // literal multiline prompt
};

enum PixImageFilter {
    PIX_IMAGE_BICUBIC,PIX_IMAGE_LANCZOS
};

enum PixImageAlpha {
    PIX_IMAGE_PREMULTIPLIED,PIX_IMAGE_WHITE,PIX_IMAGE_RGB
};

bool parse_unsigned(const std::string &text, uint64_t &value);
bool parse_float(const std::string &text, float &value);
bool settings_valid(const PixSettings &settings, std::string &error);
bool edit_settings_valid(const PixEditSettings &settings, std::string &error);
bool image_valid(const PixImage &image);
bool database_parse(const std::string &text, std::vector<PixRecord> &records, std::string &error, bool allow_empty_filename = false);
std::string database_record(const PixRecord &record);
bool job_file_load(const std::string &path, std::string &directory, std::vector<PixRecord> &records, std::string &error);
bool job_file_save(const std::string &path, const std::string &directory, const std::vector<PixRecord> &records, std::string &error);
bool config_load(const std::string &directory, std::map<std::string,std::string> &values, std::string &error);
bool config_save(const std::string &directory, const std::map<std::string,std::string> &values, std::string &error);
bool bookmarks_load(const std::string &directory, std::vector<std::string> &paths, std::string &error);
bool bookmarks_save(const std::string &directory, const std::vector<std::string> &paths, std::string &error);
bool database_load(const std::string &directory, std::vector<PixRecord> &records, std::string &error, const std::string &database = {});
bool storage_save(const std::string &directory, PixRecord &record, const PixImage &image, std::string &error, const std::string &database = {});
bool image_decode(const unsigned char* bytes, size_t count, PixImage &image, std::string &error);
bool image_load(const std::string &path, PixImage &image, std::string &error);
bool image_save(const std::string &path, const PixImage &image, std::string &error);
PixImage image_thumbnail(const PixImage &image);
PixImage image_resize(const PixImage &image, int width, int height, PixImageFilter filter, PixImageAlpha alpha);

#endif
