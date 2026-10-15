/* This file is a part of 'Pixelina' Image Generator, Editor, and Viewer.
 *
 * (C) Dmitry 'sciloaf' Solovyev, 2026
 *
 * License: MIT
 */
#ifndef PIXELINA_OS_H
#define PIXELINA_OS_H

#include <string>
#include <vector>

struct PixDirEntry {
    std::string name; // entry name, without parent path
    bool directory; // directory selector navigation flag
};

enum PixPathKind {
    PIX_PATH_MISSING,PIX_PATH_DIRECTORY,PIX_PATH_FILE,PIX_PATH_OTHER
};

std::string path_join(const std::string &directory, const std::string &name);
std::string path_normalize(const std::string &path);
std::string path_parent(const std::string &path);
PixPathKind path_kind(const std::string &path);
bool directory_list(const std::string &path, std::vector<PixDirEntry> &entries, std::string &error);

#endif
