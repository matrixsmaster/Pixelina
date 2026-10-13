#include <climits>
#include <cstdio>
#include "icon.h"
#include "../third_party/stb_image.h"
#include "../third_party/stb_image_write.h"

int main(int argc, char** argv)
{
    if (argc != 3) {
        std::fprintf(stderr,"Usage: %s input.jpg output.png\n",argv[0]);
        return 1;
    }
    int width,height,channels;
    unsigned char* pixels = stbi_load(argv[1],&width,&height,&channels,STBI_rgb);
    if (!pixels) {
        std::fprintf(stderr,"Cannot read %s: %s\n",argv[1],stbi_failure_reason());
        return 1;
    }
    bool saved = width <= INT_MAX / STBI_rgb && stbi_write_png(argv[2],width,height,STBI_rgb,pixels,width * STBI_rgb);
    stbi_image_free(pixels);
    if (!saved) std::fprintf(stderr,"Cannot write %s\n",argv[2]);
    return saved? 0 : 1;
}
