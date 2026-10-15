/* This file is a part of 'Pixelina' Image Generator, Editor, and Viewer.
 *
 * (C) Dmitry 'sciloaf' Solovyev, 2026
 *
 * License: MIT
 */
#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <sys/stat.h>
#include "os.h"

std::string path_join(const std::string &directory, const std::string &name)
{
    if (!name.empty() && name[0] == '/') return name;
    if (directory.empty() || directory.back() == '/') return directory + name;
    return directory + "/" + name;
}

std::string path_normalize(const std::string &path)
{
    char* canonical = realpath(path.c_str(),nullptr);
    if (!canonical) return path;
    std::string result = canonical;
    std::free(canonical);
    return result;
}

std::string path_parent(const std::string &path)
{
    size_t slash = path.rfind('/');
    return slash == std::string::npos? "." : path.substr(0,slash + 1);
}

PixPathKind path_kind(const std::string &path)
{
    struct stat info;
    if (stat(path.c_str(),&info)) return PIX_PATH_MISSING;
    if (S_ISDIR(info.st_mode)) return PIX_PATH_DIRECTORY;
    if (S_ISREG(info.st_mode)) return PIX_PATH_FILE;
    return PIX_PATH_OTHER;
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
