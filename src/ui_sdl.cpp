/* This file is a part of 'Pixelina' Image Generator, Editor, and Viewer.
 *
 * (C) Dmitry 'sciloaf' Solovyev, 2026
 *
 * License: MIT
 */
#include "ui_sdl.h"
#include <cmath>

PixUISDLCanvas::~PixUISDLCanvas()
{
    if (scaled_texture) SDL_DestroyTexture(scaled_texture);
}

void PixUISDLCanvas::set_main_image(const PixImage* source, SDL_Texture* texture)
{
    if (scaled_texture) SDL_DestroyTexture(scaled_texture);
    scaled_texture = nullptr;
    scaled_width = scaled_height = 0;
    main_image = source;
    main_texture = texture;
}

void PixUISDLCanvas::set_color(unsigned color)
{
    if (color_valid && draw_color == color) return;
    SDL_SetRenderDrawColor(renderer,(color >> 16) & 255,(color >> 8) & 255,color & 255,255);
    draw_color = color;
    color_valid = true;
}

void PixUISDLCanvas::fill(const PixUIRect &rect, unsigned color)
{
    SDL_Rect target = {rect.x,rect.y,rect.w,rect.h};
    set_color(color);
    SDL_RenderFillRect(renderer,&target);
}

void PixUISDLCanvas::point(int x, int y, unsigned color)
{
    set_color(color);
    SDL_RenderDrawPoint(renderer,x,y);
}

void PixUISDLCanvas::image(void* texture, const PixUIRect &rect)
{
    if (rect.w <= 0 || rect.h <= 0) return;
    if (texture == main_texture && main_image && (rect.w != main_image->width || rect.h != main_image->height)) {
        if (rect.w != scaled_width || rect.h != scaled_height) {
            if (scaled_texture) SDL_DestroyTexture(scaled_texture);
            scaled_texture = nullptr;
            scaled_width = rect.w;
            scaled_height = rect.h;
            PixImage resized = image_resize(*main_image,rect.w,rect.h,PIX_IMAGE_LANCZOS,PIX_IMAGE_RGB);
            if (image_valid(resized)) {
                scaled_texture = SDL_CreateTexture(renderer,SDL_PIXELFORMAT_RGB24,SDL_TEXTUREACCESS_STATIC,rect.w,rect.h);
                if (scaled_texture && SDL_UpdateTexture(scaled_texture,nullptr,resized.rgb.data(),rect.w * 3)) {
                    SDL_DestroyTexture(scaled_texture);
                    scaled_texture = nullptr;
                }
            }
        }
        if (scaled_texture) texture = scaled_texture;
    }
    SDL_Rect target = {rect.x,rect.y,rect.w,rect.h};
    SDL_RenderCopy(renderer,static_cast<SDL_Texture*>(texture),nullptr,&target);
}

bool PixUISDLCanvas::clip(PixUIRect &rect) const
{
    rect = clipping;
    return clipped;
}

void PixUISDLCanvas::set_clip(const PixUIRect* rect)
{
    SDL_Rect target = {};
    if (rect) target = {rect->x,rect->y,rect->w,rect->h};
    if (SDL_RenderSetClipRect(renderer,rect? &target : nullptr)) return;
    clipped = rect != nullptr;
    clipping = rect? *rect : PixUIRect{};
}

uint64_t PixUISDLCanvas::ticks() const
{
    return SDL_GetTicks();
}

std::string PixUISDLCanvas::clipboard() const
{
    char* value = SDL_GetClipboardText();
    std::string result = value? value : "";
    SDL_free(value);
    return result;
}

void PixUISDLCanvas::clipboard(const std::string &text)
{
    SDL_SetClipboardText(text.c_str());
}

void PixUISDLCanvas::text_input(const PixUIRect &rect)
{
    SDL_Rect target = {rect.x,rect.y,rect.w,rect.h};
    SDL_SetTextInputRect(&target);
}

PixUIRect PixUISDLCanvas::window_rect() const
{
    PixUIRect result;
    SDL_GetWindowPosition(window,&result.x,&result.y);
    SDL_GetWindowSize(window,&result.w,&result.h);
    return result;
}

void PixUISDLCanvas::pointer_position(int &x, int &y) const
{
    SDL_GetGlobalMouseState(&x,&y);
}

void PixUISDLCanvas::window_command(PixUIWindowCommand command, const PixUIRect &rect)
{
    switch (command) {
    case PIXUI_RESTORE: {
        PixUIRect current = window_rect();
        PixUIRect normal = restored_window;
        bool default_size = maximized() && std::abs(current.w - restored_window.w) <= UI_RESTORE_SIZE_TOLERANCE && std::abs(current.h - restored_window.h) <= UI_RESTORE_SIZE_TOLERANCE;
        if (default_size) {
            normal.w = UI_START_WIDTH;
            normal.h = UI_START_HEIGHT;
        }
        SDL_RestoreWindow(window);
        current = window_rect();
        // Restore the retained normal frame even when the window manager kept the maximized size.
        if (normal.w > 0 && normal.h > 0 && (current.w != normal.w || current.h != normal.h)) SDL_SetWindowSize(window,normal.w,normal.h);
        if (current.x != normal.x || current.y != normal.y) SDL_SetWindowPosition(window,normal.x,normal.y);
        restored_window = normal;
        break;
    }
    case PIXUI_MINIMIZE:
        if (!maximized()) restored_window = window_rect();
        SDL_MinimizeWindow(window);
        break;
    case PIXUI_MAXIMIZE:
        if (!maximized()) restored_window = window_rect();
        SDL_MaximizeWindow(window);
        break;
    case PIXUI_MOVE:
        SDL_SetWindowPosition(window,rect.x,rect.y);
        break;
    case PIXUI_RESIZE:
        SDL_SetWindowSize(window,rect.w,rect.h);
        break;
    default: break;
    }
}

bool PixUISDLCanvas::maximized() const
{
    return SDL_GetWindowFlags(window) & SDL_WINDOW_MAXIMIZED;
}

static int portable_modifiers(SDL_Keymod state)
{
    return (state & KMOD_CTRL? PIXUI_CTRL : 0) | (state & KMOD_SHIFT? PIXUI_SHIFT : 0) | (state & KMOD_ALT? PIXUI_ALT : 0);
}

PixUIEvent PixUISDLCanvas::translate(const SDL_Event &input)
{
    PixUIEvent result;
    result.mod = portable_modifiers(SDL_GetModState());
    switch (input.type) {
    case SDL_KEYDOWN:
        result.type = PIXUI_KEY;
        result.mod = portable_modifiers(static_cast<SDL_Keymod>(input.key.keysym.mod));
        switch (input.key.keysym.sym) {
        case SDLK_RETURN:
            result.key = PIXUI_RETURN;
            break;
        case SDLK_ESCAPE:
            result.key = PIXUI_ESCAPE;
            break;
        case SDLK_TAB:
            result.key = PIXUI_TAB;
            break;
        case SDLK_BACKSPACE:
            result.key = PIXUI_BACKSPACE;
            break;
        case SDLK_DELETE:
            result.key = PIXUI_DELETE;
            break;
        case SDLK_LEFT:
            result.key = PIXUI_LEFT;
            break;
        case SDLK_RIGHT:
            result.key = PIXUI_RIGHT;
            break;
        case SDLK_HOME:
            result.key = PIXUI_HOME;
            break;
        case SDLK_END:
            result.key = PIXUI_END;
            break;
        case SDLK_UP:
            result.key = PIXUI_UP_KEY;
            break;
        case SDLK_DOWN:
            result.key = PIXUI_DOWN_KEY;
            break;
        case SDLK_F2:
            result.key = PIXUI_F2;
            break;
        case SDLK_F3:
            result.key = PIXUI_F3;
            break;
        case SDLK_F4:
            result.key = PIXUI_F4;
            break;
        case SDLK_F5:
            result.key = PIXUI_F5;
            break;
        case SDLK_F11:
            result.key = PIXUI_F11;
            break;
        default:
            result.key = input.key.keysym.sym >= 0 && input.key.keysym.sym < PIXUI_SPECIAL_KEY? input.key.keysym.sym : 0;
            break;
        }
        break;
    case SDL_TEXTINPUT:
        result.type = PIXUI_TEXT;
        result.text = input.text.text;
        break;
    case SDL_MOUSEBUTTONDOWN:
    case SDL_MOUSEBUTTONUP:
        result.type = input.type == SDL_MOUSEBUTTONDOWN? PIXUI_DOWN : PIXUI_UP;
        result.x = input.button.x;
        result.y = input.button.y;
        result.button = input.button.button == SDL_BUTTON_LEFT? PIXUI_LEFT_BUTTON : 0;
        break;
    case SDL_MOUSEMOTION:
        result.type = PIXUI_MOTION;
        result.x = input.motion.x;
        result.y = input.motion.y;
        break;
    case SDL_MOUSEWHEEL:
        result.type = PIXUI_WHEEL;
        result.wheel_x = input.wheel.x;
        result.wheel_y = input.wheel.y;
        SDL_GetMouseState(&result.x,&result.y);
        break;
    default:
        break;
    }

    return result;
}

void PixUISDLCanvas::window_title(const std::string &title)
{
    SDL_SetWindowTitle(window,title.c_str());
}
