/* This file is a part of 'Pixelina' Image Generator, Editor, and Viewer.
 *
 * (C) Dmitry 'sciloaf' Solovyev, 2026
 *
 * License: MIT
 */
#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <iomanip>
#include <limits>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>
#include "storage.h"

bool parse_unsigned(const std::string &text, uint64_t &value)
{
    if (text.empty()) return false;
    value = 0;
    for (char ch : text) {
        if (ch < '0' || ch > '9') return false;
        unsigned digit = ch - '0';
        if (value > (UINT64_MAX - digit) / 10) return false;
        value = value * 10 + digit;
    }
    return true;
}

bool parse_float(const std::string &text, float &value)
{
    char* end;
    errno = 0;
    float parsed = std::strtof(text.c_str(),&end);
    if (end == text.c_str() || end != text.c_str() + text.size() || errno == ERANGE) return false;
    value = parsed;
    return true;
}

bool settings_valid(const PixSettings &settings, std::string &error)
{
    if (settings.width <= 0 || settings.height <= 0 || settings.width % PIXELINA_IMAGE_MULTIPLE || settings.height % PIXELINA_IMAGE_MULTIPLE) error = "Width and height must be positive multiples of 16.";
    else if (uint64_t(settings.width) * settings.height > PIXELINA_MAX_PIXELS) error = "Image dimensions exceed the supported memory bound.";
    else if (settings.steps <= 0) error = "Steps must be positive.";
    else if (!std::isfinite(settings.shift) || settings.shift <= 0) error = "Scheduler shift must be finite and positive.";
    else if (settings.backend != PIXELINA_CPU && settings.backend != PIXELINA_VULKAN) error = "Unknown backend.";
    else if (settings.format != PIXELINA_JPG && settings.format != PIXELINA_PNG) error = "Unknown output format.";
    else if (settings.diffuser.empty() || settings.encoder.empty() || settings.vae.empty()) error = "Select all three model files.";
    else return true;
    return false;
}

bool edit_settings_valid(const PixEditSettings &settings, std::string &error)
{
    if (settings.width <= 0 || settings.height <= 0 || settings.width % QWEN_IMAGE_MULTIPLE || settings.height % QWEN_IMAGE_MULTIPLE) {
        error = "Width and height must be positive multiples of 32.";
        return false;
    }
    if (uint64_t(settings.width) * settings.height > PIXELINA_MAX_PIXELS || uint64_t(settings.width) * settings.height > SIZE_MAX / 4) {
        error = "Image dimensions exceed the supported memory bound.";
        return false;
    }
    if (settings.steps < 2 || settings.steps > QWEN_IMAGE_MAX_STEPS) {
        error = "Edit steps must be between 2 and " + std::to_string(QWEN_IMAGE_MAX_STEPS) + " for the terminal-stretched schedule.";
        return false;
    }
    if (settings.reference_resolution <= 0 || uint64_t(settings.reference_resolution) * settings.reference_resolution > PIXELINA_MAX_PIXELS) {
        error = "Reference resolution exceeds the supported memory bound.";
        return false;
    }
    if (settings.backend != PIXELINA_CPU && settings.backend != PIXELINA_VULKAN) {
        error = "Unknown backend.";
        return false;
    }
    if (settings.format != PIXELINA_JPG && settings.format != PIXELINA_PNG) {
        error = "Unknown output format.";
        return false;
    }
    if (settings.diffuser.empty() || settings.encoder.empty() || settings.vision.empty() || settings.vae.empty()) {
        error = "Select all four Edit model files.";
        return false;
    }
    return true;
}

bool image_valid(const PixImage &image)
{
    if (image.width <= 0 || image.height <= 0) return false;
    uint64_t pixels = uint64_t(image.width) * image.height;
    if (pixels > PIXELINA_MAX_PIXELS || pixels > SIZE_MAX / 4 || image.width > INT_MAX / 4) return false;
    return image.rgb.size() == size_t(pixels) * 3 && (image.alpha.empty() || image.alpha.size() == size_t(pixels));
}

std::string path_join(const std::string &directory, const std::string &name)
{
    if (!name.empty() && name[0] == '/') return name;
    if (directory.empty() || directory.back() == '/') return directory + name;
    return directory + "/" + name;
}

std::string database_record(const PixRecord &record)
{
    std::ostringstream out;
    switch (record.model) {
    case PIX_MODEL_Z_IMAGE: {
        const PixSettings &s = record.settings;
        out << PIXELINA_VERSION << s.seed << ';' << s.width << ';' << s.height << ';' << s.steps << ';' << std::setprecision(std::numeric_limits<float>::max_digits10) << s.shift << ';' << s.diffuser << ';' << s.encoder << ';' << s.vae << ';' << s.backend << ';' << s.format << ';' << record.filename << ';';
        break;
    }
    case PIX_MODEL_QWEN_IMAGE: {
        const PixEditSettings &s = record.edit_settings;
        out << PIXELINA_EDIT_VERSION << s.seed << ';' << s.width << ';' << s.height << ';' << s.steps << ';' << s.reference_resolution << ';' << s.diffuser << ';' << s.encoder << ';' << s.vision << ';' << s.vae << ';' << s.backend << ';' << s.format << ';' << record.filename << ';' << record.references.size() << ';';
        for (const std::string &path : record.references) out << path << ';';
        break;
    }
    default:
        return {};
    }
    out << record.prompt << '\n';
    return out.str();
}

bool database_parse(const std::string &text, std::vector<PixRecord> &records, std::string &error)
{
    std::vector<PixRecord> parsed;
    size_t start = 0;
    while (start < text.size()) {
        PixRecord record;
        size_t count = 0,cursor = start;
        if (!text.compare(start,std::strlen(PIXELINA_VERSION),PIXELINA_VERSION)) {
            count = PIXELINA_RECORD_FIELDS;
            cursor += std::strlen(PIXELINA_VERSION);
        } else if (!text.compare(start,std::strlen(PIXELINA_EDIT_VERSION),PIXELINA_EDIT_VERSION)) {
            record.model = PIX_MODEL_QWEN_IMAGE;
            count = PIXELINA_EDIT_RECORD_FIELDS;
            cursor += std::strlen(PIXELINA_EDIT_VERSION);
        } else {
            error = "Malformed database record or unsupported log version at byte " + std::to_string(start) + ".";
            return false;
        }
        std::vector<std::string> fields;
        for (size_t i = 0; i < count; i++) {
            size_t end = text.find(';',cursor),newline = text.find('\n',cursor);
            if (end == std::string::npos || (newline != std::string::npos && newline < end)) {
                error = "Incomplete database settings prefix.";
                return false;
            }
            fields.push_back(text.substr(cursor,end - cursor));
            cursor = end + 1;
        }
        uint64_t seed,number;
        if (!parse_unsigned(fields[0],seed)) {
            error = "Invalid seed in database.";
            return false;
        }
        int width,height,steps,resolution = 0;
        int* dimensions[] = {&width,&height,&steps,&resolution};
        size_t numeric_count = record.model == PIX_MODEL_QWEN_IMAGE? 4 : 3;
        for (size_t i = 0; i < numeric_count; i++) {
            if (!parse_unsigned(fields[i + 1],number) || number > INT_MAX) {
                error = "Invalid dimensions, steps or reference resolution in database.";
                return false;
            }
            *dimensions[i] = int(number);
        }
        switch (record.model) {
        case PIX_MODEL_Z_IMAGE: {
            PixSettings &s = record.settings;
            s.seed = seed;
            s.width = width;
            s.height = height;
            s.steps = steps;
            if (!parse_float(fields[4],s.shift)) {
                error = "Invalid scheduler shift in database.";
                return false;
            }
            s.diffuser = fields[5];
            s.encoder = fields[6];
            s.vae = fields[7];
            s.backend = fields[8];
            s.format = fields[9];
            record.filename = fields[10];
            if (!settings_valid(s,error)) return false;
            break;
        }
        case PIX_MODEL_QWEN_IMAGE: {
            PixEditSettings &s = record.edit_settings;
            s.seed = seed;
            s.width = width;
            s.height = height;
            s.steps = steps;
            s.reference_resolution = resolution;
            s.diffuser = fields[5];
            s.encoder = fields[6];
            s.vision = fields[7];
            s.vae = fields[8];
            s.backend = fields[9];
            s.format = fields[10];
            record.filename = fields[11];
            uint64_t references;
            if (!parse_unsigned(fields[12],references) || references > text.size() - cursor) {
                error = "Invalid reference count in database.";
                return false;
            }
            // Consume paths without allocating from the untrusted reference count.
            for (uint64_t i = 0; i < references; i++) {
                size_t end = text.find(';',cursor),newline = text.find('\n',cursor);
                if (end == std::string::npos || (newline != std::string::npos && newline < end)) {
                    error = "Incomplete database reference prefix.";
                    return false;
                }
                record.references.push_back(text.substr(cursor,end - cursor));
                cursor = end + 1;
            }
            if (!edit_settings_valid(s,error)) return false;
            break;
        }
        }
        if (record.filename.empty()) {
            error = "Missing result filename in database.";
            return false;
        }
        // A leading dollar on a new line remains a record boundary for both versions.
        size_t next = text.find("\n$",cursor);
        size_t end = next == std::string::npos? text.size() : next + 1;
        size_t prompt_end = end;
        if (prompt_end > cursor && text[prompt_end - 1] == '\n') prompt_end--;
        record.prompt = text.substr(cursor,prompt_end - cursor);
        parsed.push_back(std::move(record));
        start = end;
    }
    records.swap(parsed);
    return true;
}

bool database_load(const std::string &directory, std::vector<PixRecord> &records, std::string &error)
{
    std::string path = path_join(directory,PIXELINA_DB);
    FILE* file = std::fopen(path.c_str(),"rb");
    if (!file) {
        if (errno == ENOENT) {
            records.clear();
            return true;
        }
        error = "Cannot read database: " + std::string(std::strerror(errno));
        return false;
    }
    std::string text;
    char buffer[4096];
    size_t count;
    while ((count = std::fread(buffer,1,sizeof(buffer),file))) text.append(buffer,count);
    bool failed = std::ferror(file);
    std::fclose(file);
    if (failed) {
        error = "Database read failed.";
        return false;
    }
    return database_parse(text,records,error);
}

bool storage_save(const std::string &directory, PixRecord &record, const PixImage &image, std::string &error)
{
    std::vector<PixRecord> existing;
    if (!database_load(directory,existing,error)) return false;
    std::string filename,path;
    bool reserved = false;
    for (unsigned i = 1; i <= PIXELINA_MAX_IMAGE_NUMBER; i++) {
        char stem[32];
        std::snprintf(stem,sizeof(stem),"p%05u",i);
        bool used = false;
        for (const PixRecord &old : existing) {
            if (old.filename == std::string(stem) + PIXELINA_EXT_JPG || old.filename == std::string(stem) + PIXELINA_EXT_PNG) used = true;
        }
        struct stat info;
        if (!stat(path_join(directory,std::string(stem) + PIXELINA_EXT_JPG).c_str(),&info) || !stat(path_join(directory,std::string(stem) + PIXELINA_EXT_PNG).c_str(),&info)) used = true;
        if (used) continue;
        filename = std::string(stem) + ((record.model == PIX_MODEL_QWEN_IMAGE? record.edit_settings.format : record.settings.format) == PIXELINA_PNG? PIXELINA_EXT_PNG : PIXELINA_EXT_JPG);
        path = path_join(directory,filename);
        int fd = open(path.c_str(),O_WRONLY | O_CREAT | O_EXCL,0666);
        if (fd < 0) {
            if (errno == EEXIST) continue;
            error = "Cannot reserve output image: " + std::string(std::strerror(errno));
            return false;
        }
        close(fd);
        reserved = true;
        break;
    }
    if (!reserved) {
        error = "All sequential image numbers are occupied.";
        return false;
    }
    if (!image_save(path,image,error)) {
        unlink(path.c_str());
        return false;
    }
    PixRecord saved = record;
    saved.filename = filename;
    existing.push_back(saved);
    std::string log;
    for (const PixRecord &entry : existing) log += database_record(entry);
    std::string temporary = path_join(directory,PIXELINA_TEMP_PATTERN);
    std::vector<char> writable(temporary.begin(),temporary.end());
    writable.push_back(0);
    int fd = mkstemp(writable.data());
    bool ok = fd >= 0;
    if (ok) {
        size_t offset = 0;
        while (offset < log.size()) {
            ssize_t written = write(fd,log.data() + offset,log.size() - offset);
            if (written < 0 && errno == EINTR) continue;
            if (written <= 0) {
                ok = false;
                break;
            }
            offset += size_t(written);
        }
        if (fsync(fd)) ok = false;
        if (close(fd)) ok = false;
    }
    if (ok && rename(writable.data(),path_join(directory,PIXELINA_DB).c_str())) ok = false;
    if (!ok) {
        if (fd >= 0) unlink(writable.data());
        unlink(path.c_str());
        error = "Image could not be added to the database; existing records are preserved.";
        return false;
    }
    record.filename = filename;
    return true;
}

bool directory_list(const std::string &path, std::vector<PixDirEntry> &entries, std::string &error)
{
    DIR* directory = opendir(path.c_str());
    if (!directory) {
        error = "Cannot open directory: " + std::string(std::strerror(errno));
        return false;
    }
    entries.clear();
    dirent* entry;
    while ((entry = readdir(directory))) {
        if (!std::strcmp(entry->d_name,".")) continue;
        struct stat info;
        if (stat(path_join(path,entry->d_name).c_str(),&info)) continue;
        entries.push_back({entry->d_name,S_ISDIR(info.st_mode)});
    }
    closedir(directory);
    std::sort(entries.begin(),entries.end(),[](const PixDirEntry &a, const PixDirEntry &b) { return a.directory != b.directory? a.directory > b.directory : a.name < b.name; });
    return true;
}
