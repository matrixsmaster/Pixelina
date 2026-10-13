#include <cstdio>

#include "embed.h"

int main(int argc, char** argv)
{
    if (argc != 2 && argc != 3) return 1;
    FILE* file = std::fopen(argv[1],"rb");
    if (!file) return 1;
    const char* name = argv[2];
    std::printf("alignas(4) static const unsigned char %s[] = {\n",name);
    unsigned column = 0;
    int byte;
    while ((byte = std::fgetc(file)) != EOF) {
        std::printf("0x%02x,",byte);
        if (++column % EMBED_LINE_BYTES == 0) std::putchar('\n');
    }
    bool failed = std::ferror(file);
    std::fclose(file);
    std::puts("0x00\n};");
    return failed? 1 : 0;
}
