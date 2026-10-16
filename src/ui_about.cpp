/* This file is a part of 'Pixelina' Image Generator, Editor, and Viewer.
 *
 * (C) Dmitry 'sciloaf' Solovyev, 2026
 *
 * License: MIT
 */
#include <algorithm>
#include "ui_about.h"
#include "../assets/pixelina_about_jpg.h"

bool PixelinaAbout::first_frame(PixImage &image, std::string &error)
{
    PixImage sheet;
    if (!image_decode(pixelina_about_jpg,sizeof(pixelina_about_jpg) - 1,sheet,error)) return false;
    return animation_frame(sheet,0,image,error);
}

bool PixelinaAbout::animation_frame(const PixImage &sheet, int index, PixImage &image, std::string &error)
{
    if (sheet.width < PIXELINA_ABOUT_FRAME_COLUMNS || sheet.height < PIXELINA_ABOUT_FRAME_ROWS) {
        error = "Invalid About animation dimensions.";
        return false;
    }
    int column = index % PIXELINA_ABOUT_FRAME_COLUMNS,row = index / PIXELINA_ABOUT_FRAME_COLUMNS;
    int left = column * sheet.width / PIXELINA_ABOUT_FRAME_COLUMNS,top = row * sheet.height / PIXELINA_ABOUT_FRAME_ROWS;
    PixImage frame;
    // Proportional cell edges also accommodate a generated sheet with fractional cell sizes.
    frame.width = (column + 1) * sheet.width / PIXELINA_ABOUT_FRAME_COLUMNS - left;
    frame.height = (row + 1) * sheet.height / PIXELINA_ABOUT_FRAME_ROWS - top;
    frame.rgb.resize(size_t(frame.width) * frame.height * 3);
    for (int y = 0; y < frame.height; y++) std::copy_n(sheet.rgb.data() + (size_t(top + y) * sheet.width + left) * 3,size_t(frame.width) * 3,frame.rgb.data() + size_t(y) * frame.width * 3);
    image = image_resize(frame,PIXELINA_ABOUT_ICON_SIZE,PIXELINA_ABOUT_ICON_SIZE,PIX_IMAGE_LANCZOS,PIX_IMAGE_RGB);
    if (!image_valid(image)) {
        error = "Cannot resize About animation frame.";
        return false;
    }
    return true;
}

bool PixelinaAbout::prepare_about_animation(SDL_Renderer* renderer, std::string &error)
{
    icon->texture = nullptr;
    for (SDL_Texture* texture : frames) SDL_DestroyTexture(texture);
    frames.clear();
    frame = 0;
    PixImage sheet;
    if (!image_decode(pixelina_about_jpg,sizeof(pixelina_about_jpg) - 1,sheet,error)) return false;
    for (int index = 0; index < PIXELINA_ABOUT_FRAME_COUNT; index++) {
        PixImage preview;
        if (!animation_frame(sheet,index,preview,error)) return false;
        SDL_Texture* texture = SDL_CreateTexture(renderer,SDL_PIXELFORMAT_RGB24,SDL_TEXTUREACCESS_STATIC,preview.width,preview.height);
        if (!texture || SDL_UpdateTexture(texture,nullptr,preview.rgb.data(),preview.width * 3)) {
            error = SDL_GetError();
            if (texture) SDL_DestroyTexture(texture);
            return false;
        }
        frames.push_back(texture);
    }
    icon->texture = frames[0];
    started = canvas.ticks();
    return true;
}

PixelinaAbout::PixelinaAbout(PixUICanvas &context, const std::string &caption, int speed, const std::vector<std::string> &text) : PixUIAboutDialog(context,caption,nullptr,PIXELINA_ABOUT_ICON_SIZE,PIXELINA_ABOUT_ICON_SIZE,text),fps(std::max(1,speed)),started(canvas.ticks())
{
}

PixelinaAbout::~PixelinaAbout()
{
    icon->texture = nullptr;
    for (SDL_Texture* texture : frames) SDL_DestroyTexture(texture);
}

bool PixelinaAbout::needs_repaint()
{
    if (!frames.empty()) {
        size_t current = size_t((canvas.ticks() - started) * fps / PIXELINA_ABOUT_SECOND % frames.size());
        if (current != frame) {
            frame = current;
            icon->texture = frames[frame];
            repaint = true;
        }
    }
    return PixUIAboutDialog::needs_repaint();
}

static unsigned pixelina_about_rainbow_color(size_t phase, size_t count)
{
    unsigned hue = unsigned(phase % count * PIXELINA_ABOUT_RAINBOW_SECTORS * PIXELINA_ABOUT_RAINBOW_CHANNEL_MAX / count);
    unsigned rising = hue % PIXELINA_ABOUT_RAINBOW_CHANNEL_MAX,falling = PIXELINA_ABOUT_RAINBOW_CHANNEL_MAX - rising;
    unsigned red = 0,green = 0,blue = 0;
    switch (hue / PIXELINA_ABOUT_RAINBOW_CHANNEL_MAX) {
    case 0:
        red = PIXELINA_ABOUT_RAINBOW_CHANNEL_MAX;
        green = rising;
        break;
    case 1:
        red = falling;
        green = PIXELINA_ABOUT_RAINBOW_CHANNEL_MAX;
        break;
    case 2:
        green = PIXELINA_ABOUT_RAINBOW_CHANNEL_MAX;
        blue = rising;
        break;
    case 3:
        green = falling;
        blue = PIXELINA_ABOUT_RAINBOW_CHANNEL_MAX;
        break;
    case 4:
        red = rising;
        blue = PIXELINA_ABOUT_RAINBOW_CHANNEL_MAX;
        break;
    case 5:
        red = PIXELINA_ABOUT_RAINBOW_CHANNEL_MAX;
        blue = falling;
        break;
    }
    return (red << (PIXELINA_ABOUT_RAINBOW_CHANNEL_BITS * 2)) | (green << PIXELINA_ABOUT_RAINBOW_CHANNEL_BITS) | blue;
}

void PixelinaAbout::draw()
{
    PixUIAboutDialog::draw();
    if (!visible || lines.empty() || frames.empty()) return;
    PixUILabel* name = lines[0];
    PixUIRect previous,effective = body->bounds;
    bool clipped = canvas.clip(previous);
    if (clipped && !pixui_intersect(effective,previous,effective)) return;
    canvas.set_clip(&effective);
    canvas.fill(name->bounds,body->color);
    int x = name->bounds.x;
    size_t letter = 0;
    for (size_t offset = 0; offset < name->caption.size(); offset = pixui_next_character(name->caption,offset)) {
        size_t end = pixui_next_character(name->caption,offset);
        canvas.text(x,name->bounds.y,name->caption.substr(offset,end - offset),pixelina_about_rainbow_color(frame + letter * PIXELINA_ABOUT_RAINBOW_LETTER_OFFSET,frames.size()));
        x += UI_FONT_WIDTH;
        letter++;
    }
    canvas.set_clip(clipped? &previous : nullptr);
}
