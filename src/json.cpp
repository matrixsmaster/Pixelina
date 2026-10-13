/* This file is a part of 'Pixelina' Image Generator, Editor, and Viewer.
 *
 * (C) Dmitry 'sciloaf' Solovyev, 2026
 *
 * License: MIT
 */
#include "json.h"
#include <cctype>
#include <cstdlib>
#include <utility>

class PixJsonReader {
    const std::string &text; // bounded input buffer
    size_t position = 0; // next byte to consume

    void whitespace()
    {
        while (position < text.size() && (text[position] == ' ' || text[position] == '\n' || text[position] == '\r' || text[position] == '\t')) position++;
    }

    int hex()
    {
        unsigned value = 0;
        for (unsigned i = 0; i < 4; i++) {
            if (position == text.size()) return -1;
            char ch = text[position++];
            unsigned digit;
            if (ch >= '0' && ch <= '9') digit = ch - '0';
            else if (ch >= 'a' && ch <= 'f') digit = ch - 'a' + 10;
            else if (ch >= 'A' && ch <= 'F') digit = ch - 'A' + 10;
            else return -1;
            value = value * 16 + digit;
        }
        return value;
    }

    bool string(std::string &result)
    {
        if (position == text.size() || text[position++] != '"') return false;
        while (position < text.size()) {
            unsigned char ch = text[position++];
            if (ch == '"') return true;
            if (ch < ' ') return false;
            if (ch != '\\') {
                result += char(ch);
                continue;
            }
            if (position == text.size()) break;
            ch = text[position++];
            if (ch == '"' || ch == '\\' || ch == '/') result += char(ch);
            else if (ch == 'b') result += '\b';
            else if (ch == 'f') result += '\f';
            else if (ch == 'n') result += '\n';
            else if (ch == 'r') result += '\r';
            else if (ch == 't') result += '\t';
            else if (ch == 'u') {
                int code = hex();
                if (code < 0) return false;
                if (code >= JSON_HIGH_SURROGATE && code < JSON_LOW_SURROGATE) {
                    if (text.compare(position,2,"\\u")) return false;
                    position += 2;
                    int low = hex();
                    if (low < JSON_LOW_SURROGATE || low > JSON_SURROGATE_END) return false;
                    code = 0x10000 + (code - JSON_HIGH_SURROGATE) * 1024 + low - JSON_LOW_SURROGATE;
                } else if (code >= JSON_LOW_SURROGATE && code <= JSON_SURROGATE_END) return false;
                if (code < 0x80) result += char(code);
                else if (code < 0x800) {
                    result += char(0xc0 | (code >> 6));
                    result += char(0x80 | (code & 63));
                } else if (code < 0x10000) {
                    result += char(0xe0 | (code >> 12));
                    result += char(0x80 | ((code >> 6) & 63));
                    result += char(0x80 | (code & 63));
                } else {
                    result += char(0xf0 | (code >> 18));
                    result += char(0x80 | ((code >> 12) & 63));
                    result += char(0x80 | ((code >> 6) & 63));
                    result += char(0x80 | (code & 63));
                }
            } else return false;
        }
        return false;
    }

    PixJson value(unsigned depth)
    {
        if (depth > JSON_MAX_DEPTH) return {PixJson::Invalid,{},{},{}};
        whitespace();
        if (position == text.size()) return {PixJson::Invalid,{},{},{}};
        PixJson result;
        char ch = text[position];
        if (ch == '"') {
            result.type = PixJson::String;
            if (!string(result.scalar)) return {PixJson::Invalid,{},{},{}};
        } else if (ch == '[' || ch == '{') {
            bool object = ch == '{';
            result.type = object? PixJson::Object : PixJson::Array;
            position++;
            whitespace();
            char closing = object? '}' : ']';
            if (position < text.size() && text[position] == closing) {
                position++;
                return result;
            }
            while (true) {
                whitespace();
                std::string key;
                if (object) {
                    if (!string(key)) return {PixJson::Invalid,{},{},{}};
                    whitespace();
                    if (position == text.size() || text[position++] != ':') return {PixJson::Invalid,{},{},{}};
                }
                PixJson member = value(depth + 1);
                if (member.type == PixJson::Invalid) return member;
                if (object) {
                    if (!result.object.emplace(key,std::move(member)).second) return {PixJson::Invalid,{},{},{}};
                } else result.array.push_back(std::move(member));
                if (result.object.size() + result.array.size() > JSON_MAX_ITEMS) return {PixJson::Invalid,{},{},{}};
                whitespace();
                if (position == text.size()) return {PixJson::Invalid,{},{},{}};
                ch = text[position++];
                if (ch == closing) break;
                if (ch != ',') return {PixJson::Invalid,{},{},{}};
            }
        } else if (ch == 't' || ch == 'f' || ch == 'n') {
            std::string token = ch == 't'? "true" : (ch == 'f'? "false" : "null");
            if (text.compare(position,token.size(),token)) return {PixJson::Invalid,{},{},{}};
            position += token.size();
            result.type = ch == 'n'? PixJson::Null : PixJson::Boolean;
            result.scalar = token;
        } else {
            size_t start = position;
            if (text[position] == '-') position++;
            if (position == text.size() || !std::isdigit(static_cast<unsigned char>(text[position]))) return {PixJson::Invalid,{},{},{}};
            if (text[position] == '0') position++;
            else while (position < text.size() && std::isdigit(static_cast<unsigned char>(text[position]))) position++;
            if (position < text.size() && text[position] == '.') {
                position++;
                size_t digits = position;
                while (position < text.size() && std::isdigit(static_cast<unsigned char>(text[position]))) position++;
                if (digits == position) return {PixJson::Invalid,{},{},{}};
            }
            if (position < text.size() && (text[position] == 'e' || text[position] == 'E')) {
                position++;
                if (position < text.size() && (text[position] == '+' || text[position] == '-')) position++;
                size_t digits = position;
                while (position < text.size() && std::isdigit(static_cast<unsigned char>(text[position]))) position++;
                if (digits == position) return {PixJson::Invalid,{},{},{}};
            }
            result.type = PixJson::Number;
            result.scalar = text.substr(start,position - start);
        }
        return result;
    }

public:
    explicit PixJsonReader(const std::string &input) : text(input) {}

    PixJson parse()
    {
        PixJson result = value(0);
        whitespace();
        if (position != text.size()) return {PixJson::Invalid,{},{},{}};
        return result;
    }
};

uint64_t PixJson::integer() const
{
    if (type != Number || scalar.empty()) return UINT64_MAX;
    uint64_t result = 0;
    for (char ch : scalar) {
        if (ch < '0' || ch > '9' || result > (UINT64_MAX - (ch - '0')) / 10) return UINT64_MAX;
        result = result * 10 + ch - '0';
    }
    return result;
}

const PixJson &PixJson::at(const std::string &key) const
{
    auto found = object.find(key);
    static const PixJson missing = {Invalid,{},{},{}};
    if (type != Object || found == object.end()) return missing;
    return found->second;
}

PixJson json_parse(const std::string &text)
{
    PixJsonReader reader(text);
    return reader.parse();
}
