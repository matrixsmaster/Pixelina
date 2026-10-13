/* This file is a part of 'Pixelina' Image Generator, Editor, and Viewer.
 *
 * (C) Dmitry 'sciloaf' Solovyev, 2026
 *
 * License: MIT
 */
#include <algorithm>
#include <cerrno>
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

std::string path_join(const std::string &directory, const std::string &name)
{
    if (!name.empty() && name[0] == '/') return name;
    if (directory.empty() || directory.back() == '/') return directory + name;
    return directory + "/" + name;
}

std::string database_record(const PixRecord &record)
{
    const PixSettings &s = record.settings;
    std::ostringstream out;
    out << PIXELINA_VERSION << s.seed << ';' << s.width << ';' << s.height << ';' << s.steps << ';' << std::setprecision(std::numeric_limits<float>::max_digits10) << s.shift << ';' << s.diffuser << ';' << s.encoder << ';' << s.vae << ';' << s.backend << ';' << s.format << ';' << record.filename << ';' << record.prompt << '\n';
    return out.str();
}

bool database_parse(const std::string &text, std::vector<PixRecord> &records, std::string &error)
{
    std::vector<PixRecord> parsed;
    size_t start = 0;
    while (start < text.size()) {
        if (text.compare(start, std::strlen(PIXELINA_VERSION), PIXELINA_VERSION)) {
            error = "Malformed database record or unsupported log version at byte " + std::to_string(start) + ".";
            return false;
        }
        PixRecord record;
        std::vector<std::string> fields;
        size_t cursor = start + std::strlen(PIXELINA_VERSION);
        for (unsigned i = 0; i < PIXELINA_RECORD_FIELDS; i++) {
            size_t end = text.find(';',cursor);
            size_t newline = text.find('\n',cursor);
            if (end == std::string::npos || (newline != std::string::npos && newline < end)) {
                error = "Incomplete version 1 settings prefix.";
                return false;
            }
            fields.push_back(text.substr(cursor,end - cursor));
            cursor = end + 1;
        }
        uint64_t number;
        if (!parse_unsigned(fields[0],record.settings.seed)) {
            error = "Invalid seed in database.";
            return false;
        }
        int* dimensions[] = {&record.settings.width,&record.settings.height,&record.settings.steps};
        for (unsigned i = 0; i < 3; i++) {
            if (!parse_unsigned(fields[i + 1],number)) {
                error = "Invalid dimensions or steps in database.";
                return false;
            }
            *dimensions[i] = int(number);
        }
        if (!parse_float(fields[4],record.settings.shift)) {
            error = "Invalid scheduler shift in database.";
            return false;
        }
        record.settings.diffuser = fields[5];
        record.settings.encoder = fields[6];
        record.settings.vae = fields[7];
        record.settings.backend = fields[8];
        record.settings.format = fields[9];
        record.filename = fields[10];
        if (record.filename.empty() || !settings_valid(record.settings,error)) return false;

        // A leading dollar on a new line is a record boundary
        size_t next = text.find("\n$",cursor);
        size_t end = next == std::string::npos? text.size() : next + 1;
        size_t prompt_end = end;
        if (prompt_end > cursor && text[prompt_end - 1] == '\n') prompt_end--;
        record.prompt = text.substr(cursor,prompt_end - cursor);
        parsed.push_back(record);
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
        filename = std::string(stem) + (record.settings.format == PIXELINA_PNG? PIXELINA_EXT_PNG : PIXELINA_EXT_JPG);
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
