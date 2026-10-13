/* This file is a part of 'Pixelina' Image Generator, Editor, and Viewer.
 *
 * (C) Dmitry 'sciloaf' Solovyev, 2026
 *
 * License: MIT
 */
#ifndef PIXELINA_JSON_H
#define PIXELINA_JSON_H

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#define JSON_MAX_DEPTH 64
#define JSON_MAX_ITEMS 1000000u
#define JSON_UNICODE_MAX 0x10ffff
#define JSON_HIGH_SURROGATE 0xd800
#define JSON_LOW_SURROGATE 0xdc00
#define JSON_SURROGATE_END 0xdfff

struct PixJson {
    enum PixType { Invalid,Null,Boolean,Number,String,Array,Object }; // minimal JSON value types
    PixType type = Null; // active value kind
    std::string scalar; // strings and exact textual numbers
    std::vector<PixJson> array; // array elements
    std::map<std::string,PixJson> object; // object members
    uint64_t integer() const; // UINT64_MAX for invalid unsigned metadata
    const PixJson &at(const std::string &key) const;
};

PixJson json_parse(const std::string &text);

#endif
