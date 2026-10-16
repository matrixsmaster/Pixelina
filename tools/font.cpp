#include "font.h"
#include <ft2build.h>
#include FT_FREETYPE_H
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

static bool font_tool_dimension(const char* text, int limit, int &value)
{
    char* end;
    long parsed = std::strtol(text,&end,10);
    if (!text[0] || *end || parsed < 1 || parsed > limit) return false;
    value = int(parsed);
    return true;
}

static bool font_tool_convert(FT_Face face, const std::string &name, int width, int height)
{
    if (FT_Set_Pixel_Sizes(face,0,FONT_TOOL_RENDER_HEIGHT)) return false;
    int top = 0,bottom = 0;
    for (int ch = FONT_TOOL_FIRST_CHAR; ch <= FONT_TOOL_LAST_CHAR; ch++) {
        if (FT_Load_Char(face,ch,FT_LOAD_RENDER | FT_LOAD_NO_HINTING)) return false;
        top = std::max(top,face->glyph->bitmap_top);
        bottom = std::max(bottom,int(face->glyph->bitmap.rows) - face->glyph->bitmap_top);
    }
    if (top + bottom <= 0) return false;
    std::printf("%s_DEFAULT_H=%d\n%s_DEFAULT_W=%d\n%s_START_CHAR=%d\n%s_END_CHAR=%d\n",name.c_str(),height,name.c_str(),width,name.c_str(),FONT_TOOL_FIRST_CHAR,name.c_str(),FONT_TOOL_LAST_CHAR);
    for (int ch = FONT_TOOL_FIRST_CHAR; ch <= FONT_TOOL_LAST_CHAR; ch++) {
        if (FT_Load_Char(face,ch,FT_LOAD_RENDER | FT_LOAD_NO_HINTING)) return false;
        const FT_Bitmap &bitmap = face->glyph->bitmap;
        if (bitmap.pixel_mode != FT_PIXEL_MODE_GRAY && bitmap.rows) return false;
        double scale_y = double(top + bottom) / height;
        double scale_x = std::max(scale_y,double(bitmap.width) / width);
        double left = (width - bitmap.width / scale_x) / 2;
        std::printf("%s_LINE_%d=",name.c_str(),ch);
        for (int y = 0; y < height; y++) {
            uint32_t bits = 0;
            double y0 = y * scale_y - (top - face->glyph->bitmap_top),y1 = y0 + scale_y;
            for (int x = 0; x < width; x++) {
                double x0 = (x - left) * scale_x,x1 = x0 + scale_x;
                double coverage = 0;
                for (int sy = std::max(0,int(std::floor(y0))); sy < std::min(int(bitmap.rows),int(std::ceil(y1))); sy++) {
                    const unsigned char* row = bitmap.buffer + (bitmap.pitch < 0? int(bitmap.rows) - sy - 1 : sy) * std::abs(bitmap.pitch);
                    double vertical = std::min(y1,double(sy + 1)) - std::max(y0,double(sy));
                    for (int sx = std::max(0,int(std::floor(x0))); sx < std::min(int(bitmap.width),int(std::ceil(x1))); sx++) coverage += row[sx] * vertical * (std::min(x1,double(sx + 1)) - std::max(x0,double(sx)));
                }
                if (coverage >= FONT_TOOL_THRESHOLD * scale_x * scale_y) bits |= uint32_t(1) << x;
            }
            std::printf("%s0x%02X",y? "," : "",unsigned(bits));
        }
        std::putchar('\n');
    }
    return !std::ferror(stdout);
}

int main(int argc, char** argv)
{
    int width = FONT_TOOL_WIDTH,height = FONT_TOOL_HEIGHT;
    if ((argc != 3 && argc != 5) || (argc == 5 && (!font_tool_dimension(argv[3],FONT_TOOL_MAX_WIDTH,width) || !font_tool_dimension(argv[4],FONT_TOOL_MAX_HEIGHT,height)))) {
        std::fprintf(stderr,"Usage: tools/font font.ttf NAME [width height] > font.txt\nAlso accepts OpenType fonts. NAME must contain letters, digits or underscores.\n");
        return 1;
    }
    std::string name = argv[2];
    if (name.empty()) return 1;
    for (char &ch : name) {
        if (!std::isalnum(static_cast<unsigned char>(ch)) && ch != '_') return 1;
        ch = std::toupper(static_cast<unsigned char>(ch));
    }
    FT_Library library = nullptr;
    FT_Face face = nullptr;
    if (FT_Init_FreeType(&library)) return 1;
    bool success = !FT_New_Face(library,argv[1],0,&face);
    if (success) success = font_tool_convert(face,name,width,height);
    if (face) FT_Done_Face(face);
    FT_Done_FreeType(library);
    if (!success) std::fprintf(stderr,"Cannot convert font: %s\n",argv[1]);
    return success? 0 : 1;
}
