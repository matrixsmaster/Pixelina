/* This file is a part of 'Pixelina' Image Generator, Editor, and Viewer.
 *
 * (C) Dmitry 'sciloaf' Solovyev, 2026
 *
 * License: MIT
 */
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <limits>
#include <climits>
#include <cstring>
#include <unistd.h>
#include "storage.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../third_party/stb_image.h"
#include "../third_party/stb_image_write.h"

bool image_decode(const unsigned char* bytes, size_t count, PixImage &image, std::string &error)
{
    int width,height,channels;
    if (count > INT_MAX || !stbi_info_from_memory(bytes,int(count),&width,&height,&channels) || width <= 0 || height <= 0 || uint64_t(width) * height > PIXELINA_MAX_PIXELS) {
        error = "Invalid embedded image dimensions.";
        return false;
    }
    unsigned char* rgb = stbi_load_from_memory(bytes,int(count),&width,&height,&channels,4);
    if (!rgb) {
        error = "Cannot decode embedded image.";
        return false;
    }
    image.width = width;
    image.height = height;
    image.rgb.resize(size_t(width) * height * 3);
    // Composite embedded icon transparency against the About dialog's white background.
    for (size_t pixel = 0; pixel < size_t(width) * height; pixel++) {
        unsigned alpha = rgb[pixel * 4 + 3];
        for (size_t channel = 0; channel < 3; channel++) image.rgb[pixel * 3 + channel] = (unsigned(rgb[pixel * 4 + channel]) * alpha + 255 * (255 - alpha)) / 255;
    }
    stbi_image_free(rgb);
    return true;
}

bool image_load(const std::string &path, PixImage &image, std::string &error)
{
    int width,height,channels;
    if (!stbi_info(path.c_str(),&width,&height,&channels)) {
        error = "Cannot read image: " + path;
        return false;
    }
    if (width <= 0 || height <= 0 || uint64_t(width) * height > PIXELINA_MAX_PIXELS) {
        error = "Image dimensions exceed the supported memory bound.";
        return false;
    }
    unsigned char* rgb = stbi_load(path.c_str(),&width,&height,&channels,3);
    if (!rgb) {
        error = "Cannot decode image: " + path;
        return false;
    }
    PixImage loaded;
    loaded.width = width;
    loaded.height = height;
    loaded.rgb.assign(rgb,rgb + size_t(width) * height * 3);
    stbi_image_free(rgb);
    image = std::move(loaded);
    return true;
}

bool image_save(const std::string &path, const PixImage &image, std::string &error)
{
    if (image.width <= 0 || image.height <= 0 || image.rgb.size() != size_t(image.width) * image.height * 3) {
        error = "No complete image to save.";
        return false;
    }
    size_t dot = path.find_last_of('.');
    std::string suffix = dot == std::string::npos? "" : path.substr(dot);
    std::transform(suffix.begin(),suffix.end(),suffix.begin(),[](unsigned char ch) { return std::tolower(ch); });
    if (suffix != PIXELINA_EXT_PNG && suffix != PIXELINA_EXT_JPG && suffix != PIXELINA_EXT_JPEG) {
        error = "Save as requires a .png, .jpg or .jpeg extension.";
        return false;
    }
    size_t slash = path.find_last_of('/');
    std::string temporary = (slash == std::string::npos? "" : path.substr(0,slash + 1)) + IMAGE_TEMP_SUFFIX;
    std::vector<char> temp_name(temporary.begin(),temporary.end());
    temp_name.push_back(0);
    int fd = mkstemp(temp_name.data());
    FILE* file = fd < 0? nullptr : fdopen(fd,"wb");
    if (!file) {
        if (fd >= 0) {
            close(fd);
            unlink(temp_name.data());
        }
        error = "Cannot write image: " + path;
        return false;
    }
    bool failed = false;
    auto writer = [](void* context, void* data, int size) {
        auto pair = static_cast<std::pair<FILE*,bool*>*>(context);
        if (std::fwrite(data,1,size,pair->first) != size_t(size)) *pair->second = true;
    };
    std::pair<FILE*,bool*> context = {file,&failed};
    int result = 0;
    if (suffix == PIXELINA_EXT_PNG) result = stbi_write_png_to_func(writer,&context,image.width,image.height,3,image.rgb.data(),image.width * 3);
    else if (suffix == PIXELINA_EXT_JPG || suffix == PIXELINA_EXT_JPEG) result = stbi_write_jpg_to_func(writer,&context,image.width,image.height,3,image.rgb.data(),PIXELINA_JPG_QUALITY);
    if (std::fflush(file)) failed = true;
    if (fsync(fileno(file))) failed = true;
    if (std::fclose(file)) failed = true;
    if (result && !failed && rename(temp_name.data(),path.c_str())) failed = true;
    if (!result || failed) {
        unlink(temp_name.data());
        if (error.empty()) error = "Image write failed: " + path;
        return false;
    }
    return true;
}

PixImage image_thumbnail(const PixImage &image)
{
    PixImage thumb;
    if (image.rgb.empty()) return thumb;
    float scale = std::min(float(PIXELINA_THUMB_SIZE) / image.width,float(PIXELINA_THUMB_SIZE) / image.height);
    scale = std::min(scale,1.0f);
    thumb.width = std::max(1,int(image.width * scale));
    thumb.height = std::max(1,int(image.height * scale));
    thumb.rgb.resize(size_t(thumb.width) * thumb.height * 3);
    for (int y = 0; y < thumb.height; y++) {
        for (int x = 0; x < thumb.width; x++) {
            size_t from = (size_t(y * image.height / thumb.height) * image.width + x * image.width / thumb.width) * 3;
            size_t to = (size_t(y) * thumb.width + x) * 3;
            std::copy_n(image.rgb.data() + from,3,thumb.rgb.data() + to);
        }
    }
    return thumb;
}
