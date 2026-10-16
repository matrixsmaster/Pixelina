/* This file is a part of 'Pixelina' Image Generator, Editor, and Viewer.
 *
 * (C) Dmitry 'sciloaf' Solovyev, 2026
 *
 * License: MIT
 */
#ifndef PIXELINA_UI_ABOUT_H
#define PIXELINA_UI_ABOUT_H

#include <SDL.h>
#include "pixuilib.h"
#include "storage.h"

#define PIXELINA_ABOUT { PIXELINA_NAME " " PIXELINA_VERSION, "(C) Dmitry 'sciloaf' Solovyev, 2026", "Z-Image Turbo image generation", "Qwen Image 2.1 image editing" }

#define PIXELINA_ABOUT_ICON_SIZE UI_ABOUT_ICON_SIZE
#define PIXELINA_ABOUT_SECOND 1000
#define PIXELINA_ABOUT_RAINBOW_SECTORS 6
#define PIXELINA_ABOUT_RAINBOW_CHANNEL_MAX 255
#define PIXELINA_ABOUT_RAINBOW_CHANNEL_BITS 8
#define PIXELINA_ABOUT_RAINBOW_LETTER_OFFSET 3
#define PIXELINA_ABOUT_PREFIX "About "
#define PIXELINA_ABOUT_FRAME_COUNT 32
#define PIXELINA_ABOUT_FRAME_COLUMNS 8
#define PIXELINA_ABOUT_FRAME_ROWS (PIXELINA_ABOUT_FRAME_COUNT / PIXELINA_ABOUT_FRAME_COLUMNS)
#define PIXELINA_ABOUT_FPS 10

class PixelinaAbout : public PixUIAboutDialog
{
public:
    explicit PixelinaAbout(PixUICanvas &context, const std::string &caption, int fps, const std::vector<std::string> &lines);

    ~PixelinaAbout() override;

    static bool first_frame(PixImage &image, std::string &error);
    bool prepare_about_animation(SDL_Renderer* renderer, std::string &error);
    bool needs_repaint() override;
    void draw() override;

private:
    std::vector<SDL_Texture*> frames; // owned portrait animation textures
    int fps; // configured animation frames per second
    size_t frame = 0; // current portrait and rainbow phase
    uint64_t started; // animation clock at dialog creation

    static bool animation_frame(const PixImage &sheet, int index, PixImage &image, std::string &error);
};

#endif
