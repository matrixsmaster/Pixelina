/* This file is a part of 'Pixelina' Image Generator, Editor, and Viewer.
 *
 * (C) Dmitry 'sciloaf' Solovyev, 2026
 *
 * License: MIT
 */
#include <algorithm>
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <limits.h>
#include <string.h>
#include <unistd.h>
#include "storage.h"
#include "image.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../third_party/stb_image.h"
#include "../third_party/stb_image_write.h"

#define SIMPLEWEBP_IMPLEMENTATION
#include "../third_party/simplewebp.h"

static bool webp_load(const std::string &path, PixImage &image, std::string &error)
{
    FILE* file = fopen(path.c_str(),"rb");
    if (!file) {
        error = "Cannot read WebP image: " + path;
        return false;
    }
    // SimpleWebP takes ownership of the file on both success and failure.
    simplewebp* decoder = NULL;
    simplewebp_error result = simplewebp_load_from_file(file,NULL,&decoder);
    if (result != SIMPLEWEBP_NO_ERROR) {
        error = "Cannot read WebP image (" + std::string(simplewebp_get_error_text(result)) + "): " + path;
        return false;
    }
    size_t width, height;
    simplewebp_get_dimensions(decoder,&width,&height);
    if (!width || !height || width > INT_MAX / 4 || height > INT_MAX || uint64_t(width) * height > PIXELINA_MAX_PIXELS || uint64_t(width) * height > SIZE_MAX / 4) {
        simplewebp_unload(decoder);
        error = "WebP dimensions exceed the supported memory bound.";
        return false;
    }
    size_t pixels = width * height;
    std::vector<unsigned char> rgba(pixels * 4);
    result = simplewebp_decode(decoder,rgba.data(),NULL);
    simplewebp_unload(decoder);
    if (result != SIMPLEWEBP_NO_ERROR) {
        error = "Cannot decode WebP image (" + std::string(simplewebp_get_error_text(result)) + "): " + path;
        return false;
    }
    bool alpha = false;
    for (size_t pixel = 0; pixel < pixels && !alpha; pixel++) alpha = rgba[pixel * 4 + 3] != 255;
    PixImage loaded;
    loaded.width = (int)width;
    loaded.height = (int)height;
    loaded.rgb.resize(pixels * 3);
    if (alpha) loaded.alpha.resize(pixels);
    for (size_t pixel = 0; pixel < pixels; pixel++) {
        std::copy_n(rgba.data() + pixel * 4,3,loaded.rgb.data() + pixel * 3);
        if (alpha) loaded.alpha[pixel] = rgba[pixel * 4 + 3];
    }
    image = std::move(loaded);
    return true;
}

bool image_decode(const unsigned char* bytes, size_t count, PixImage &image, std::string &error)
{
    int width,height,channels;
    if (count > INT_MAX || !stbi_info_from_memory(bytes,int(count),&width,&height,&channels) || width <= 0 || width > INT32_MAX / 4 || height <= 0 || uint64_t(width) * height > PIXELINA_MAX_PIXELS) {
        error = "Invalid embedded image dimensions.";
        return false;
    }
    unsigned char* rgb = stbi_load_from_memory(bytes,int(count),&width,&height,&channels,4);
    if (!rgb) {
        error = "Cannot decode embedded image.";
        return false;
    }
    PixImage decoded;
    decoded.width = width;
    decoded.height = height;
    size_t pixels = size_t(width) * height;
    decoded.rgb.resize(pixels * 3);
    if (channels == 2 || channels == 4) decoded.alpha.resize(pixels);
    for (size_t pixel = 0; pixel < pixels; pixel++) {
        std::copy_n(rgb + pixel * 4,3,decoded.rgb.data() + pixel * 3);
        if (!decoded.alpha.empty()) decoded.alpha[pixel] = rgb[pixel * 4 + 3];
    }
    stbi_image_free(rgb);
    image = std::move(decoded);
    return true;
}

bool image_load(const std::string &path, PixImage &image, std::string &error)
{
    int width,height,channels;
    unsigned char* rgb = NULL;
    if (stbi_info(path.c_str(),&width,&height,&channels)) {
        if (width <= 0 || height <= 0 || uint64_t(width) * height > PIXELINA_MAX_PIXELS) {
            error = "Image dimensions exceed the supported memory bound.";
            return false;
        }
        rgb = stbi_load(path.c_str(),&width,&height,&channels,4);
    }
    if (!rgb) {
        if (webp_load(path,image,error)) return true;
        error = "Unknown or unsupported image: " + path;
        return false;
    }
    PixImage loaded;
    loaded.width = width;
    loaded.height = height;
    size_t pixels = size_t(width) * height;
    loaded.rgb.resize(pixels * 3);
    if (channels == 2 || channels == 4) loaded.alpha.resize(pixels);
    for (size_t pixel = 0; pixel < pixels; pixel++) {
        std::copy_n(rgb + pixel * 4,3,loaded.rgb.data() + pixel * 3);
        if (!loaded.alpha.empty()) loaded.alpha[pixel] = rgb[pixel * 4 + 3];
    }
    stbi_image_free(rgb);
    image = std::move(loaded);
    return true;
}

bool image_save(const std::string &path, const PixImage &image, std::string &error)
{
    if (!image_valid(image)) {
        error = "No complete image to save.";
        return false;
    }
    size_t dot = path.find_last_of('.');
    std::string suffix = dot == std::string::npos? "" : path.substr(dot);
    std::transform(suffix.begin(),suffix.end(),suffix.begin(),[](unsigned char ch) { return tolower(ch); });
    if (suffix != PIXELINA_EXT_PNG && suffix != PIXELINA_EXT_JPG && suffix != PIXELINA_EXT_JPEG) {
        error = "Save as requires a .png, .jpg or .jpeg extension.";
        return false;
    }
    FILE* file = fopen(path.c_str(),"wb");
    if (!file) {
        error = "Cannot write image: " + path;
        return false;
    }
    bool failed = false;
    auto writer = [](void* context, void* data, int size) {
        auto pair = (std::pair<FILE*,bool*>*)context;
        if (fwrite(data,1,size,pair->first) != size_t(size)) *pair->second = true;
    };
    std::pair<FILE*,bool*> context = {file,&failed};
    int result = 0;
    if (suffix == PIXELINA_EXT_PNG) {
        std::vector<unsigned char> rgba;
        if (!image.alpha.empty()) {
            rgba.resize(image.alpha.size() * 4);
            for (size_t pixel = 0; pixel < image.alpha.size(); pixel++) {
                std::copy_n(image.rgb.data() + pixel * 3,3,rgba.data() + pixel * 4);
                rgba[pixel * 4 + 3] = image.alpha[pixel];
            }
        }
        int channels = rgba.empty()? 3 : 4;
        result = stbi_write_png_to_func(writer,&context,image.width,image.height,channels,rgba.empty()? image.rgb.data() : rgba.data(),image.width * channels);
    }
    else if (suffix == PIXELINA_EXT_JPG || suffix == PIXELINA_EXT_JPEG) result = stbi_write_jpg_to_func(writer,&context,image.width,image.height,3,image.rgb.data(),PIXELINA_JPG_QUALITY);
    if (fflush(file)) failed = true;
    if (fsync(fileno(file))) failed = true;
    if (fclose(file)) failed = true;
    if (!result || failed) {
        if (error.empty()) error = "Image write failed: " + path;
        return false;
    }
    return true;
}

PixImage image_thumbnail(const PixImage &image)
{
    PixImage thumb;
    if (!image_valid(image)) return thumb;
    float scale = std::min(float(PIXELINA_THUMB_SIZE) / image.width,float(PIXELINA_THUMB_SIZE) / image.height);
    scale = std::min(scale,1.0f);
    thumb.width = std::max(1,int(image.width * scale));
    thumb.height = std::max(1,int(image.height * scale));
    thumb.rgb.resize(size_t(thumb.width) * thumb.height * 3);
    if (!image.alpha.empty()) thumb.alpha.resize(size_t(thumb.width) * thumb.height);
    for (int y = 0; y < thumb.height; y++) {
        for (int x = 0; x < thumb.width; x++) {
            size_t from = (size_t(y * image.height / thumb.height) * image.width + x * image.width / thumb.width) * 3;
            size_t to = (size_t(y) * thumb.width + x) * 3;
            std::copy_n(image.rgb.data() + from,3,thumb.rgb.data() + to);
            if (!thumb.alpha.empty()) thumb.alpha[to / 3] = image.alpha[from / 3];
        }
    }
    return thumb;
}

struct PixImageTap {
    int first; // first contributing source pixel
    std::vector<int32_t> coefficients; // normalized fixed-point filter weights
};

static double image_filter(double distance, PixImageFilter filter)
{
    double x = fabs(distance);
    switch (filter) {
    case PIX_IMAGE_BICUBIC:
        if (x < 1) return ((IMAGE_BICUBIC_COEFFICIENT + IMAGE_BICUBIC_INNER_QUADRATIC) * x - (IMAGE_BICUBIC_COEFFICIENT + IMAGE_BICUBIC_INNER_LINEAR)) * x * x + 1;
        if (x < IMAGE_BICUBIC_RADIUS) return (((x - IMAGE_BICUBIC_OUTER_QUADRATIC) * x + IMAGE_BICUBIC_OUTER_LINEAR) * x - IMAGE_BICUBIC_OUTER_CONSTANT) * IMAGE_BICUBIC_COEFFICIENT;
        return 0;
    case PIX_IMAGE_LANCZOS:
        if (!x) return 1;
        if (x >= IMAGE_LANCZOS_RADIUS) return 0;
        return sin(M_PI * x) * sin(M_PI * x / IMAGE_LANCZOS_RADIUS) / (M_PI * M_PI * x * x / IMAGE_LANCZOS_RADIUS);
    }
    return 0;
}

static std::vector<PixImageTap> image_taps(int source, int target, PixImageFilter filter)
{
    std::vector<PixImageTap> taps(size_t(target),PixImageTap{});
    double scale = double(source) / target,stretch = std::max(1.0,scale);
    double radius = (filter == PIX_IMAGE_LANCZOS? IMAGE_LANCZOS_RADIUS : IMAGE_BICUBIC_RADIUS) * stretch;
    for (int pixel = 0; pixel < target; pixel++) {
        double center = (pixel + 0.5) * scale;
        int first = std::max(0,int(center - radius + 0.5)),last = std::min(source,int(center + radius + 0.5));
        PixImageTap &tap = taps[size_t(pixel)];
        tap.first = first;
        std::vector<double> weights;
        double total = 0;
        for (int index = first; index < last; index++) {
            double weight = image_filter((index - center + 0.5) / stretch,filter);
            weights.push_back(weight);
            total += weight;
        }
        if (!total || !isfinite(total)) return {};
        for (double weight : weights) {
            double coefficient = weight / total * IMAGE_RESAMPLE_ONE;
            tap.coefficients.push_back(int32_t(coefficient < 0? coefficient - 0.5 : coefficient + 0.5));
        }
    }
    return taps;
}

PixImage image_resize(const PixImage &image, int width, int height, PixImageFilter filter, PixImageAlpha policy)
{
    if (!image_valid(image) || width <= 0 || height <= 0 || uint64_t(width) * height > PIXELINA_MAX_PIXELS || uint64_t(width) * image.height > PIXELINA_MAX_PIXELS) return {};
    if (filter != PIX_IMAGE_BICUBIC && filter != PIX_IMAGE_LANCZOS) return {};
    if (policy != PIX_IMAGE_PREMULTIPLIED && policy != PIX_IMAGE_WHITE && policy != PIX_IMAGE_RGB) return {};
    if (width == image.width && height == image.height && policy == PIX_IMAGE_PREMULTIPLIED) return image;
    bool alpha = policy == PIX_IMAGE_PREMULTIPLIED && !image.alpha.empty();
    size_t channels = alpha? 4 : 3;
    std::vector<unsigned char> source(size_t(image.width) * image.height * channels);
    for (size_t pixel = 0; pixel < size_t(image.width) * image.height; pixel++) {
        unsigned opacity = image.alpha.empty()? 255 : image.alpha[pixel];
        for (size_t channel = 0; channel < 3; channel++) {
            unsigned value = image.rgb[pixel * 3 + channel];
            if (alpha) value = (value * opacity + 127) / 255;
            if (policy == PIX_IMAGE_WHITE) value = (value * opacity + 255 * (255 - opacity) + 127) / 255;
            source[pixel * channels + channel] = value;
        }
        if (alpha) source[pixel * channels + 3] = opacity;
    }
    if (width != image.width) {
        std::vector<PixImageTap> taps = image_taps(image.width,width,filter);
        if (taps.empty()) return {};
        std::vector<unsigned char> horizontal(size_t(width) * image.height * channels);
        for (int y = 0; y < image.height; y++) {
            for (int x = 0; x < width; x++) {
                const PixImageTap &tap = taps[size_t(x)];
                for (size_t channel = 0; channel < channels; channel++) {
                    int64_t value = IMAGE_RESAMPLE_ONE / 2;
                    for (size_t i = 0; i < tap.coefficients.size(); i++) value += int64_t(source[(size_t(y) * image.width + tap.first + i) * channels + channel]) * tap.coefficients[i];
                    horizontal[(size_t(y) * width + x) * channels + channel] = std::max(int64_t(0),std::min(int64_t(255),value >> IMAGE_RESAMPLE_BITS));
                }
            }
        }
        source.swap(horizontal);
    }
    if (height != image.height) {
        std::vector<PixImageTap> taps = image_taps(image.height,height,filter);
        if (taps.empty()) return {};
        std::vector<unsigned char> vertical(size_t(width) * height * channels);
        for (int y = 0; y < height; y++) {
            const PixImageTap &tap = taps[size_t(y)];
            for (int x = 0; x < width; x++) {
                for (size_t channel = 0; channel < channels; channel++) {
                    int64_t value = IMAGE_RESAMPLE_ONE / 2;
                    for (size_t i = 0; i < tap.coefficients.size(); i++) value += int64_t(source[((size_t(tap.first) + i) * width + x) * channels + channel]) * tap.coefficients[i];
                    vertical[(size_t(y) * width + x) * channels + channel] = std::max(int64_t(0),std::min(int64_t(255),value >> IMAGE_RESAMPLE_BITS));
                }
            }
        }
        source.swap(vertical);
    }
    PixImage resized;
    resized.width = width;
    resized.height = height;
    resized.rgb.resize(size_t(width) * height * 3);
    if (alpha) resized.alpha.resize(size_t(width) * height);
    for (size_t pixel = 0; pixel < size_t(width) * height; pixel++) {
        unsigned opacity = alpha? source[pixel * channels + 3] : 255;
        for (size_t channel = 0; channel < 3; channel++) {
            unsigned value = source[pixel * channels + channel];
            if (alpha) value = opacity? std::min(255u,value * 255 / opacity) : value;
            resized.rgb[pixel * 3 + channel] = value;
        }
        if (alpha) resized.alpha[pixel] = opacity;
    }
    return resized;
}
