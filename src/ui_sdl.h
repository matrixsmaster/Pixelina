/* This file is a part of 'Pixelina' Image Generator, Editor, and Viewer.
 *
 * (C) Dmitry 'sciloaf' Solovyev, 2026
 *
 * License: MIT
 */
#ifndef PIXELINA_UI_SDL_H
#define PIXELINA_UI_SDL_H

#include <SDL.h>
#include "pixuilib.h"

#define UI_START_WIDTH 1024
#define UI_START_HEIGHT 768
#define UI_RESTORE_SIZE_TOLERANCE 3

class PixUISDLCanvas : public PixUICanvas
{
public:
    void fill(const PixUIRect &rect, unsigned color) override;
    void point(int x, int y, unsigned color) override;
    void image(void* texture, const PixUIRect &rect) override;
    bool clip(PixUIRect &rect) const override;
    void set_clip(const PixUIRect* rect) override;
    uint64_t ticks() const override;
    std::string clipboard() const override;
    void clipboard(const std::string &text) override;
    void text_input(const PixUIRect &rect) override;
    PixUIRect window_rect() const override;
    void pointer_position(int &x, int &y) const override;
    void window_command(PixUIWindowCommand command, const PixUIRect &rect) override;
    bool maximized() const override;
    void window_title(const std::string &title) override;
    PixUIEvent translate(const SDL_Event &input);

    SDL_Renderer* renderer = nullptr; // application renderer
    SDL_Window* window = nullptr; // application native window
    PixUIRect restored_window; // normal geometry retained across maximizing and minimizing

private:
    unsigned draw_color = 0; // cached renderer draw color
    bool color_valid = false; // renderer color has been set through this adapter
    bool clipped = false; // clipping enabled through this adapter
    PixUIRect clipping; // cached clip rectangle
    void set_color(unsigned color);
};

#endif
