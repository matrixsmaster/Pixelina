/* This file is a part of 'Pixelina' Image Generator, Editor, and Viewer.
 *
 * (C) Dmitry 'sciloaf' Solovyev, 2026
 *
 * License: MIT
 */
#include <algorithm>
#include <ctype.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include "pixuilib.h"
#include "os.h"
#include "../third_party/font8x8_basic.h"

PixUITheme pixui_theme;

static const char* theme_color_names[PIXUI_COLOR_COUNT] = {"COLOR_FACE","COLOR_LIGHT","COLOR_SHADOW","COLOR_DARK","COLOR_TITLE","COLOR_DESKTOP","COLOR_SELECTED","COLOR_DISABLED"};
static const char* theme_font_names[PIXUI_FONT_COUNT] = {"LABEL","HEADER","TEXT"};

static std::string theme_trim(const std::string &value)
{
    size_t first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    return value.substr(first,value.find_last_not_of(" \t\r\n") - first + 1);
}

static bool theme_color(const std::string &text, unsigned &color)
{
    unsigned red,green,blue;
    if (std::sscanf(text.c_str(),"%u ,%u ,%u",&red,&green,&blue) != 3) return false;
    color = (uint32_t(uint8_t(red)) << (UI_COLOR_CHANNEL_BITS * 2)) | (uint32_t(uint8_t(green)) << UI_COLOR_CHANNEL_BITS) | uint8_t(blue);
    return true;
}

static bool theme_font(std::map<std::string,std::string> &values, const std::string &name, PixUIBitmapFont &font)
{
    int dimensions[4];
    const char* suffixes[] = {UI_THEME_FONT_WIDTH,UI_THEME_FONT_HEIGHT,UI_THEME_FONT_START,UI_THEME_FONT_END};
    for (size_t i = 0; i < sizeof(suffixes) / sizeof(suffixes[0]); i++) {
        auto entry = values.find(name + suffixes[i]);
        if (entry == values.end()) return false;
        dimensions[i] = std::atoi(entry->second.c_str());
        values.erase(entry);
    }
    if (dimensions[0] < 1 || dimensions[0] > UI_FONT_MAX_WIDTH || dimensions[1] < 1 || dimensions[1] > UI_FONT_MAX_HEIGHT || dimensions[2] < 0 || dimensions[3] >= UI_FONT_GLYPHS || dimensions[2] > dimensions[3]) return false;
    font.width = dimensions[0];
    font.height = dimensions[1];
    font.rows.resize(size_t(UI_FONT_GLYPHS) * font.height);
    for (int ch = dimensions[2]; ch <= dimensions[3]; ch++) {
        auto entry = values.find(name + UI_THEME_FONT_LINE + std::to_string(ch));
        if (entry == values.end()) continue;
        const char* next = entry->second.c_str();
        for (int row = 0; row < font.height; row++) {
            char* end;
            font.rows[size_t(ch) * font.height + row] = std::strtoul(next,&end,0);
            next = *end == ','? end + 1 : end;
        }
        values.erase(entry);
    }
    return true;
}

static bool theme_style(std::map<std::string,std::string> &values, const std::map<std::string,std::shared_ptr<PixUIBitmapFont>> &sources, PixUIFont &font, PixUIFontRole role)
{
    std::string key = std::string(UI_THEME_FONT_PREFIX) + theme_font_names[role];
    auto entry = values.find(key);
    if (entry != values.end()) {
        std::string name = entry->second;
        values.erase(entry);
        if (name != UI_THEME_DEFAULT_FONT) {
            if (name.empty()) return false;
            for (char &ch : name) {
                if (!isalnum((unsigned char)ch) && ch != '_') return false;
                ch = toupper((unsigned char)ch);
            }
            auto source = sources.find(name);
            if (source == sources.end()) return false;
            font.bitmap = source->second;
        }
    }
    int source_width = font.bitmap? font.bitmap->width : UI_DEFAULT_FONT_WIDTH;
    int source_height = font.bitmap? font.bitmap->height : UI_DEFAULT_FONT_HEIGHT;
    font.width = source_width;
    font.height = source_height;
    entry = values.find(key + UI_THEME_SIZE_SUFFIX);
    if (entry != values.end()) {
        font.height = std::atoi(entry->second.c_str());
        if (font.height < 1 || font.height > UI_FONT_MAX_HEIGHT) return false;
        font.width = std::max(1,(source_width * font.height + source_height / 2) / source_height);
        if (font.width > UI_FONT_MAX_SCALED_WIDTH) return false;
        values.erase(entry);
    }
    entry = values.find(key + UI_THEME_COLOR_SUFFIX);
    if (entry != values.end()) {
        if (!theme_color(entry->second,font.color)) return false;
        values.erase(entry);
    }
    if (font.height == source_height) return true;
    // Scale once at load time; native-size fonts render directly from their source rows.
    font.cache.resize(size_t(UI_FONT_GLYPHS) * font.width * font.height);
    for (int ch = 0; ch < UI_FONT_GLYPHS; ch++) {
        for (int y = 0; y < font.height; y++) {
            int source_y = y * source_height / font.height;
            uint32_t bits = font.bitmap? font.bitmap->rows[size_t(ch) * source_height + source_y] : uint8_t(font8x8_basic[ch][source_y]);
            for (int x = 0; x < font.width; x++) font.cache[(size_t(ch) * font.height + y) * font.width + x] = (bits >> (x * source_width / font.width)) & 1;
        }
    }
    return true;
}

bool pixui_load_theme(const std::string &path, std::string &error)
{
    PixUITheme theme;
    if (path.empty()) {
        pixui_theme = std::move(theme);
        return true;
    }
    std::ifstream input(path);
    if (!input) {
        error = "Cannot read theme: " + path;
        return false;
    }
    std::map<std::string,std::string> values;
    std::string line;
    bool valid = true;
    while (std::getline(input,line)) {
        line = theme_trim(line.substr(0,line.find(';')));
        if (line.empty()) continue;
        size_t equal = line.find('=');
        if (equal == std::string::npos || !equal || !values.emplace(theme_trim(line.substr(0,equal)),theme_trim(line.substr(equal + 1))).second) {
            valid = false;
            break;
        }
    }
    for (int i = 0; valid && i < PIXUI_COLOR_COUNT; i++) {
        auto entry = values.find(theme_color_names[i]);
        if (entry == values.end()) continue;
        valid = theme_color(entry->second,theme.colors[i]);
        values.erase(entry);
    }
    std::map<std::string,std::shared_ptr<PixUIBitmapFont>> sources;
    std::vector<std::string> names;
    const std::string suffix = UI_THEME_FONT_HEIGHT;
    for (const auto &entry : values) {
        if (entry.first.size() > suffix.size() && entry.first.compare(entry.first.size() - suffix.size(),suffix.size(),suffix) == 0) names.push_back(entry.first.substr(0,entry.first.size() - suffix.size()));
    }
    for (const auto &name : names) {
        if (!valid) break;
        auto bitmap = std::make_shared<PixUIBitmapFont>();
        valid = theme_font(values,name,*bitmap);
        if (valid) sources.emplace(name,std::move(bitmap));
    }
    for (int i = 0; valid && i < PIXUI_FONT_COUNT; i++) {
        theme.fonts[i].color = theme.colors[i == PIXUI_FONT_HEADER? PIXUI_COLOR_LIGHT : PIXUI_COLOR_DARK];
        valid = theme_style(values,sources,theme.fonts[i],static_cast<PixUIFontRole>(i));
    }
    if (!valid || !values.empty() || input.bad()) {
        error = "Invalid or unreadable theme: " + path;
        return false;
    }
    pixui_theme = std::move(theme);
    return true;
}

static size_t previous_character(const std::string &text, size_t offset)
{
    if (!offset) return 0;
    offset--;
    while (offset && (static_cast<unsigned char>(text[offset]) & UI_UTF8_MASK) == UI_UTF8_CONTINUATION) offset--;
    return offset;
}

static size_t previous_word(const std::string &text, size_t offset)
{
    while (offset) {
        size_t previous = previous_character(text,offset);
        unsigned char ch = (unsigned char)text[previous];
        if (!isspace(ch) && !ispunct(ch)) break;
        offset = previous;
    }
    while (offset) {
        size_t previous = previous_character(text,offset);
        unsigned char ch = (unsigned char)text[previous];
        if (isspace(ch) || ispunct(ch)) break;
        offset = previous;
    }
    return offset;
}

static size_t next_word(const std::string &text, size_t offset)
{
    while (offset < text.size()) {
        unsigned char ch = (unsigned char)text[offset];
        if (isspace(ch) || ispunct(ch)) break;
        offset = pixui_next_character(text,offset);
    }
    while (offset < text.size()) {
        unsigned char ch = (unsigned char)text[offset];
        if (!isspace(ch) && !ispunct(ch)) break;
        offset = pixui_next_character(text,offset);
    }
    return offset;
}

PixUIRect PixUILayout::row(int width, int height)
{
    PixUIRect bounds = {x,y,width,height};
    x += width + spacing;
    return bounds;
}

PixUIRect PixUILayout::column(int width, int height)
{
    PixUIRect bounds = {x,y,width,height};
    y += height + spacing;
    return bounds;
}

void PixUICanvas::bevel(const PixUIRect &rect, bool inset, bool background)
{
    if (background) fill(rect,UI_COLOR_FACE);
    unsigned upper = inset? UI_COLOR_DARK : UI_COLOR_LIGHT;
    unsigned lower = inset? UI_COLOR_SHADOW : UI_COLOR_DARK;
    fill({rect.x,rect.y,rect.w,1},upper);
    fill({rect.x,rect.y,1,rect.h},upper);
    fill({rect.x,rect.y + rect.h - 1,rect.w,1},lower);
    fill({rect.x + rect.w - 1,rect.y,1,rect.h},lower);
}

void PixUICanvas::text(int x, int y, const std::string &value, unsigned color, PixUIFontRole role)
{
    const PixUIFont &font = pixui_theme.fonts[role];
    int origin = x;
    for (size_t i = 0; i < value.size(); i = pixui_next_character(value,i)) {
        unsigned char ch = value[i];
        if (ch == '\n') {
            y += font.height + UI_FONT_LINE_GAP;
            x = origin;
            continue;
        }
        if (ch >= UI_FONT_GLYPHS || ch < ' ') ch = '?';
        for (int row = 0; row < font.height; row++) {
            uint32_t bits = 0;
            if (font.cache.empty()) bits = font.bitmap? font.bitmap->rows[size_t(ch) * font.height + row] : uint8_t(font8x8_basic[ch][row]);
            for (int column = 0; column < font.width; column++) {
                bool pixel = font.cache.empty()? (bits & (uint32_t(1) << column)) != 0 : font.cache[(size_t(ch) * font.height + row) * font.width + column] != 0;
                if (pixel) point(x + column,y + row,color);
            }
        }
        x += font.width;
    }
}

void PixUICanvas::button(const PixUIRect &rect, const std::string &label, bool pressed)
{
    fill(rect,UI_COLOR_DARK);
    PixUIRect inner = {rect.x + 1,rect.y + 1,rect.w - 2,rect.h - 2};
    bevel(inner,pressed);
    if (!pressed) {
        fill({inner.x + 1,inner.y + inner.h - 2,inner.w - 2,1},UI_COLOR_SHADOW);
        fill({inner.x + inner.w - 2,inner.y + 1,1,inner.h - 2},UI_COLOR_SHADOW);
    }
    if (label == UI_CONTROL_UP || label == UI_CONTROL_DOWN || label == UI_CONTROL_LEFT || label == UI_CONTROL_RIGHT) {
        int center = rect.x + rect.w / 2;
        int top = rect.y + (rect.h - UI_ARROW_SIZE) / 2;
        bool horizontal = label == UI_CONTROL_LEFT || label == UI_CONTROL_RIGHT;
        if (horizontal) {
            center = rect.y + rect.h / 2;
            top = rect.x + (rect.w - UI_ARROW_SIZE) / 2;
        }
        for (int row = 0; row < UI_ARROW_SIZE; row++) {
            int radius = label == UI_CONTROL_UP || label == UI_CONTROL_LEFT? row : UI_ARROW_SIZE - row - 1;
            if (horizontal) fill({top + row,center - radius,1,radius * 2 + 1},UI_COLOR_DARK);
            else fill({center - radius,top + row,radius * 2 + 1,1},UI_COLOR_DARK);
        }
    } else if (label == UI_CONTROL_SYSTEM) {
        PixUIRect dash = {rect.x + UI_ARROW_SIZE,rect.y + rect.h / 2 - 1,rect.w - UI_ARROW_SIZE * 2,UI_SYSTEM_BAR_HEIGHT};
        fill(dash,UI_COLOR_DARK);
        fill({dash.x + 1,dash.y + 1,dash.w - 2,1},UI_COLOR_LIGHT);
    } else text(rect.x + std::max(UI_TEXT_PADDING,(rect.w - int(label.size()) * UI_FONT_WIDTH) / 2),rect.y + (rect.h - UI_FONT_HEIGHT) / 2,label,pixui_theme.fonts[PIXUI_FONT_LABEL].color);
}

void PixUICanvas::pointer_position(int &x, int &y) const
{
    PixUIRect rect = window_rect();
    x = rect.x + pointer_x;
    y = rect.y + pointer_y;
}

bool PixUICanvas::event(PixUIForm &form, const PixUIEvent &input)
{
    form.prepare_modal();
    form.layout();
    if (input.type == PIXUI_DOWN || input.type == PIXUI_UP || input.type == PIXUI_MOTION || input.type == PIXUI_WHEEL) {
        pointer_x = input.x;
        pointer_y = input.y;
    }
    if (captured && (input.type == PIXUI_MOTION || input.type == PIXUI_UP)) {
        PixUIObject* control = captured;
        if (!control->interactive()) release_capture();
        else {
            bool handled = control->event(input);
            if (handled) {
                for (PixUIObject* owner = control; owner; owner = owner->parent) owner->repaint = true;
            }
            if (input.type == PIXUI_UP && input.button == PIXUI_LEFT_BUTTON) captured = nullptr;
            return handled;
        }
    }
    bool handled = form.event(input);
    if (handled) form.repaint = true;
    return handled;
}

void PixUICanvas::release_capture()
{
    if (!captured) return;
    for (PixUIObject* owner = captured; owner; owner = owner->parent) owner->repaint = true;
    captured->cancel_capture();
    captured = nullptr;
}

PixUIObject::PixUIObject(PixUICanvas &context) : canvas(context)
{
}

PixUIObject::~PixUIObject()
{
    if (canvas.captured == this) canvas.captured = nullptr;
}

void PixUIObject::layout()
{
}

bool PixUIObject::event(const PixUIEvent &input)
{
    (void)input;
    return false;
}

void PixUIObject::cancel_capture()
{
    if (canvas.captured == this) canvas.captured = nullptr;
}

bool PixUIObject::needs_repaint()
{
    if (painted_visible != visible || painted_enabled != enabled || painted_bounds.x != bounds.x || painted_bounds.y != bounds.y || painted_bounds.w != bounds.w || painted_bounds.h != bounds.h) {
        repaint = true;
        if (parent) parent->repaint = true;
        painted_visible = visible;
        painted_enabled = enabled;
        painted_bounds = bounds;
    }
    return visible && repaint;
}

bool PixUIObject::interactive() const
{
    for (const PixUIObject* object = this; object; object = object->parent) {
        if (!object->visible || !object->layout_visible || !object->enabled) return false;
    }
    return true;
}

PixUIControl::PixUIControl(PixUICanvas &context) : PixUIObject(context)
{
}

bool PixUIControl::event(const PixUIEvent &input)
{
    if (visible && enabled && wheel && input.type == PIXUI_WHEEL) {
        int x = input.x,y = input.y;
        if (pixui_inside(bounds,x,y)) {
            wheel(input);
            return true;
        }
    }
    if (!visible || !enabled || (!activate && !pointer) || input.type != PIXUI_DOWN || input.button != PIXUI_LEFT_BUTTON || !pixui_inside(bounds,input.x,input.y)) return false;
    if (pointer) pointer(input);
    else activate();
    return true;
}

PixUIPanel::PixUIPanel(PixUICanvas &context) : PixUIControl(context)
{
}

void PixUIPanel::draw()
{
    needs_repaint();
    repaint = false;
    if (!visible) return;
    if (background) canvas.fill(bounds,color);
    if (border) canvas.bevel(bounds,inset,false);
}

PixUILabel::PixUILabel(PixUICanvas &context, const std::string &text) : PixUIControl(context),caption(text)
{
}

bool PixUILabel::needs_repaint()
{
    if (caption != painted_caption || color != painted_color) {
        repaint = true;
        painted_caption = caption;
        painted_color = color;
    }
    return PixUIObject::needs_repaint();
}

void PixUILabel::draw()
{
    needs_repaint();
    repaint = false;
    if (!visible) return;
    if (background) canvas.fill(bounds,background_color);
    const PixUIFont &font = pixui_theme.fonts[font_role];
    std::string text = caption;
    if (wrap) {
        int columns = std::max(1,bounds.w / font.width);
        std::vector<size_t> lines = pixui_editor_lines(caption,columns);
        text.clear();
        for (size_t i = 0; i < lines.size(); i++) {
            size_t end = i + 1 < lines.size()? lines[i + 1] : caption.size();
            text += caption.substr(lines[i],end - lines[i]);
            if (i + 1 < lines.size() && (text.empty() || text.back() != '\n')) text += '\n';
        }
    }
    int x = bounds.x,y = bounds.y;
    if (centered) {
        x += std::max(0,(bounds.w - int(text.size()) * font.width) / 2);
        y += std::max(0,(bounds.h - font.height) / 2);
    }
    canvas.text(x,y,text,color,font_role);
    if (underline) canvas.fill({x,y + font.height,font.width,1},color);
}

PixUIButton::PixUIButton(PixUICanvas &context, const std::string &text) : PixUILabel(context,text)
{
}

void PixUIButton::cancel_capture()
{
    pressed = false;
    PixUIObject::cancel_capture();
}

void PixUIButton::draw()
{
    needs_repaint();
    repaint = false;
    if (!visible) return;
    canvas.button(bounds,caption,pressed);
}

bool PixUIButton::event(const PixUIEvent &input)
{
    if (visible && enabled && default_action && input.type == PIXUI_KEY && input.key == PIXUI_RETURN) {
        if (activate) activate();
        return true;
    }
    if (input.type == PIXUI_UP && input.button == PIXUI_LEFT_BUTTON && canvas.captured == this) {
        bool click = visible && enabled && pixui_inside(bounds,input.x,input.y);
        pressed = false;
        canvas.captured = nullptr;
        if (click) {
            if (pointer) pointer(input);
            else if (activate) activate();
        }
        return true;
    }
    if (input.type == PIXUI_MOTION && canvas.captured == this) {
        pressed = pixui_inside(bounds,input.x,input.y);
        return true;
    }
    if (!visible || !enabled || input.type != PIXUI_DOWN || input.button != PIXUI_LEFT_BUTTON || !pixui_inside(bounds,input.x,input.y)) return false;
    pressed = true;
    canvas.captured = this;
    return true;
}

PixUIImage::PixUIImage(PixUICanvas &context) : PixUIControl(context)
{
}

void PixUIImage::draw()
{
    needs_repaint();
    repaint = false;
    if (!visible) return;
    draw_image(bounds);
}

void PixUIImage::draw_image(const PixUIRect &rect)
{
    PixUIRect previous,effective = rect;
    bool clipped = canvas.clip(previous);
    if (clipped && !pixui_intersect(rect,previous,effective)) return;
    canvas.set_clip(&effective);
    if (texture && width > 0 && height > 0) {
        PixUIRect target = {rect.x - offset_x,rect.y - offset_y,width,height};
        if (scaled) {
            float scale = std::min(float(rect.w) / width,float(rect.h) / height);
            target.w = int(width * scale);
            target.h = int(height * scale);
            target.x = rect.x + (rect.w - target.w) / 2;
            target.y = rect.y + (rect.h - target.h) / 2;
        }
        canvas.image(texture,target);
    } else if (!placeholder.empty()) canvas.text(rect.x + UI_LABEL_PADDING,rect.y + UI_LABEL_PADDING,placeholder,UI_COLOR_DISABLED);
    canvas.set_clip(clipped? &previous : nullptr);
}

bool PixUIImage::event(const PixUIEvent &input)
{
    if (visible && enabled && input.type == PIXUI_WHEEL && pixui_inside(bounds,input.x,input.y) && !scaled) {
        if (input.mod & PIXUI_SHIFT) offset_x -= input.wheel_y * PIXUI_WHEEL_SCROLL_Y * wheel_step;
        else offset_y -= input.wheel_y * PIXUI_WHEEL_SCROLL_Y * wheel_step;
        offset_x -= input.wheel_x * PIXUI_WHEEL_SCROLL_X * wheel_step;
        layout();
        return true;
    }
    return PixUIControl::event(input);
}

void PixUIImage::layout()
{
    offset_x = std::max(0,std::min(offset_x,std::max(0,width - bounds.w)));
    offset_y = std::max(0,std::min(offset_y,std::max(0,height - bounds.h)));
}

PixUIThumb::PixUIThumb(PixUICanvas &context) : PixUIImage(context)
{
}

void PixUIThumb::draw()
{
    needs_repaint();
    repaint = false;
    if (!visible) return;
    if (source) {
        PixUIThumbCache &preview = source();
        texture = preview.texture;
        width = preview.width;
        height = preview.height;
    }
    PixUIRect cell = bounds;
    canvas.fill(cell,selected? UI_COLOR_SELECTED : UI_COLOR_LIGHT);
    draw_image({cell.x + UI_TEXT_PADDING,cell.y + UI_CONTENT_INSET,cell.w - UI_TEXT_PADDING * 2,cell.h - UI_LINE_HEIGHT - UI_CONTENT_INSET});
    canvas.text(cell.x + UI_CONTENT_INSET,cell.y + cell.h - UI_LINE_HEIGHT,caption,selected? UI_COLOR_LIGHT : pixui_theme.fonts[PIXUI_FONT_LABEL].color);
}

PixUIScrollbar::PixUIScrollbar(PixUICanvas &context) : PixUIControl(context)
{
}

void PixUIScrollbar::draw()
{
    needs_repaint();
    repaint = false;
    if (!visible || !position) return;
    canvas.bevel(bounds,true);
    int length = vertical? bounds.h : bounds.w;
    int track = std::max(1,length - UI_SCROLLBAR * 2);
    int thumb = std::min(track,std::max(UI_SCROLLBAR,track * page / std::max(1,std::max(page,total))));
    int offset = UI_SCROLLBAR + (track - thumb) * *position / std::max(1,total - page);
    if (vertical) {
        canvas.button({bounds.x,bounds.y,bounds.w,UI_SCROLLBAR},UI_CONTROL_UP,false);
        canvas.button({bounds.x,bounds.y + bounds.h - UI_SCROLLBAR,bounds.w,UI_SCROLLBAR},UI_CONTROL_DOWN,false);
        canvas.bevel({bounds.x,bounds.y + offset,bounds.w,thumb},false);
    } else {
        canvas.button({bounds.x,bounds.y,UI_SCROLLBAR,bounds.h},UI_CONTROL_LEFT,false);
        canvas.button({bounds.x + bounds.w - UI_SCROLLBAR,bounds.y,UI_SCROLLBAR,bounds.h},UI_CONTROL_RIGHT,false);
        canvas.bevel({bounds.x + offset,bounds.y,thumb,bounds.h},false);
    }
}

bool PixUIScrollbar::event(const PixUIEvent &input)
{
    if (!visible || !enabled || !position) return false;
    if (input.type == PIXUI_UP && input.button == PIXUI_LEFT_BUTTON && canvas.captured == this) return true;
    if (input.type == PIXUI_MOTION && canvas.captured == this) {
        *position = pixui_scrollbar_position(vertical? input.y : input.x,vertical? bounds.y : bounds.x,vertical? bounds.h : bounds.w,page,total);
        if (changed) changed();
        return true;
    }
    if (input.type != PIXUI_DOWN || input.button != PIXUI_LEFT_BUTTON || !pixui_inside(bounds,input.x,input.y)) return false;
    int coordinate = vertical? input.y : input.x;
    int origin = vertical? bounds.y : bounds.x;
    int length = vertical? bounds.h : bounds.w;
    if (coordinate < origin + UI_SCROLLBAR) *position -= step;
    else if (coordinate >= origin + length - UI_SCROLLBAR) *position += step;
    else {
        *position = pixui_scrollbar_position(coordinate,origin,length,page,total);
        canvas.captured = this;
        if (capture) capture();
    }
    *position = std::max(0,std::min(std::max(0,total - page),*position));
    if (changed) changed();
    return true;
}

PixUIMemo::PixUIMemo(PixUICanvas &context, PixUIEditor* state) : PixUIControl(context),model(state? *state : editor),scroll(context)
{
    scroll.parent = this;
    scroll.vertical = true;
    scroll.position = &model.scroll;
    scroll.changed = [this]() { model.reveal = false; };
}

PixUIMemo::PixUIMemo(PixUICanvas &context, PixUIEditor &state) : PixUIMemo(context,&state)
{
}

PixUIMemo::~PixUIMemo()
{
    if (focus_owner) focus_owner->unregister_focus(*this);
}

void PixUIMemo::set_text(const std::string &text)
{
    repaint = true;
    model.text = text;
    model.cursor = text.size();
    model.anchor = model.cursor;
    model.scroll = 0;
    model.horizontal = 0;
    model.reveal = true;
}

void PixUIMemo::scroll_to_end()
{
    layout();
    model.cursor = model.text.size();
    model.anchor = model.cursor;
    model.scroll = std::max(0,scroll.total - scroll.page);
    model.horizontal = 0;
    model.reveal = false;
    repaint = true;
}

bool PixUIMemo::needs_repaint()
{
    bool focused = active && interactive() && (!focus_owner || (!focus_owner->modal_blocked && focus_owner->active));
    for (PixUIObject* owner = parent; focused && owner; owner = owner->parent) {
        PixUIForm* form = dynamic_cast<PixUIForm*>(owner);
        if (form && !form->active) focused = false;
    }
    if (focused && caret_phase != canvas.ticks() / UI_CARET_PERIOD) repaint = true;
    return PixUIObject::needs_repaint();
}

void PixUIMemo::draw()
{
    needs_repaint();
    repaint = false;
    if (!visible) return;
    caret_phase = canvas.ticks() / UI_CARET_PERIOD;
    draw_editor();
    layout();
    if (multiline) scroll.draw();
    repaint = false;
}

bool PixUIMemo::event(const PixUIEvent &input)
{
    if (!visible || !enabled) return false;
    layout();
    if (multiline && input.type == PIXUI_WHEEL) {
        int x = input.x,y = input.y;
        if (pixui_inside(bounds,x,y)) {
            model.scroll = std::max(0,std::min(std::max(0,scroll.total - scroll.page),model.scroll - input.wheel_y * PIXUI_WHEEL_SCROLL_Y * UI_WHEEL_LINES));
            model.reveal = false;
            return true;
        }
    }
    if (multiline && scroll.event(input)) return true;
    if (input.type == PIXUI_DOWN && input.button == PIXUI_LEFT_BUTTON && pixui_inside(bounds,input.x,input.y)) {
        if (focus_owner) focus_owner->focus_control(*this);
        if (activate) activate();
        place_cursor(input.x,input.y,input.mod);
        return true;
    }
    if (active && (input.type == PIXUI_KEY || input.type == PIXUI_TEXT)) {
        edit(input);
        return true;
    }
    return false;
}

void PixUIMemo::layout()
{
    scroll.bounds = {bounds.x + bounds.w - UI_CONTENT_INSET - UI_SCROLLBAR,bounds.y + 1,UI_SCROLLBAR,bounds.h - 2};
    scroll.page = std::max(1,(bounds.h - UI_CONTENT_INSET * 2) / UI_TEXT_LINE_HEIGHT);
    scroll.total = int(pixui_editor_lines(model.text,std::max(1,(bounds.w - UI_CONTENT_INSET * 2 - UI_SCROLLBAR) / UI_TEXT_WIDTH)).size());
    scroll.visible = visible && multiline;
    scroll.enabled = enabled;
}

void PixUIMemo::edit(const PixUIEvent &event)
{
    PixUIEditor &editor = model;
    const PixUIRect &area = bounds;
    if (event.type != PIXUI_KEY && event.type != PIXUI_TEXT) return;
    if (read_only) {
        if (event.type == PIXUI_TEXT) return;
        if (event.key == PIXUI_BACKSPACE || event.key == PIXUI_DELETE || event.key == PIXUI_RETURN) return;
        if ((event.mod & PIXUI_CTRL) && (event.key == 'v' || event.key == 'x')) {
            if (event.key == 'x' && editor.cursor != editor.anchor) canvas.clipboard(editor.text.substr(std::min(editor.cursor,editor.anchor),std::max(editor.cursor,editor.anchor) - std::min(editor.cursor,editor.anchor)));
            return;
        }
    }
    editor.reveal = true;
    size_t low = std::min(editor.cursor,editor.anchor),high = std::max(editor.cursor,editor.anchor);
    std::string insertion;
    if (event.type == PIXUI_TEXT) insertion = event.text;
    else {
        int key = event.key;
        bool control = event.mod & PIXUI_CTRL;
        bool shift = event.mod & PIXUI_SHIFT;
        if (control) {
            switch (key) {
            case 'a':
                editor.anchor = 0;
                editor.cursor = editor.text.size();
                return;
            case 'c':
            case 'x':
                if (low != high) canvas.clipboard(editor.text.substr(low,high - low));
                if (key == 'c' || low == high) return;
                editor.text.erase(low,high - low);
                editor.cursor = low;
                editor.anchor = low;
                return;
            case 'v':
                insertion = canvas.clipboard();
                break;
            default:
                break;
            }
        }
        switch (key) {
        case PIXUI_RETURN:
            if (multiline) insertion = "\n";
            break;
        case PIXUI_BACKSPACE:
        case PIXUI_DELETE:
            if (low == high) {
                if (key == PIXUI_BACKSPACE) low = previous_character(editor.text,low);
                else high = pixui_next_character(editor.text,high);
            }
            editor.text.erase(low,high - low);
            editor.cursor = low;
            editor.anchor = low;
            return;
        case PIXUI_LEFT:
        case PIXUI_RIGHT:
        case PIXUI_HOME:
        case PIXUI_END:
        case PIXUI_UP_KEY:
        case PIXUI_DOWN_KEY: {
            size_t target = editor.cursor;
            switch (key) {
            case PIXUI_LEFT:
                if (!shift && low != high) target = low;
                else target = control? previous_word(editor.text,target) : previous_character(editor.text,target);
                break;
            case PIXUI_RIGHT:
                if (!shift && low != high) target = high;
                else target = control? next_word(editor.text,target) : pixui_next_character(editor.text,target);
                break;
            default:
                if (multiline && !control) {
                    int columns = std::max(1,(area.w - UI_CONTENT_INSET * 2 - UI_SCROLLBAR) / UI_TEXT_WIDTH);
                    std::vector<size_t> lines = pixui_editor_lines(editor.text,columns);
                    int line = int(std::upper_bound(lines.begin(),lines.end(),target) - lines.begin()) - 1;
                    size_t end = line + 1 < int(lines.size())? lines[line + 1] : editor.text.size();
                    if (end > lines[line] && editor.text[end - 1] == '\n') end--;
                    if (key == PIXUI_HOME) target = lines[line];
                    else if (key == PIXUI_END) target = end;
                    else {
                        size_t column = 0;
                        for (size_t offset = lines[line]; offset < target; offset = pixui_next_character(editor.text,offset)) column++;
                        int next = line + (key == PIXUI_UP_KEY? -1 : 1);
                        if (next < 0 || next >= int(lines.size())) return;
                        target = lines[next];
                        size_t limit = next + 1 < int(lines.size())? lines[next + 1] : editor.text.size();
                        for (size_t i = 0; i < column && target < limit && editor.text[target] != '\n'; i++) target = pixui_next_character(editor.text,target);
                    }
                } else if (key == PIXUI_HOME || key == PIXUI_END) target = key == PIXUI_HOME? 0 : editor.text.size();
                break;
            }
            editor.cursor = target;
            if (!shift) editor.anchor = target;
            return;
        }
        default:
            break;
        }
    }
    if (!insertion.empty()) {
        if (!multiline) {
            insertion.erase(std::remove(insertion.begin(),insertion.end(),'\n'),insertion.end());
            insertion.erase(std::remove(insertion.begin(),insertion.end(),'\r'),insertion.end());
        }
        editor.text.replace(low,high - low,insertion);
        editor.cursor = low + insertion.size();
        editor.anchor = editor.cursor;
    }
}

void PixUIMemo::draw_editor()
{
    PixUIEditor &editor = model;
    const PixUIRect &rect = bounds;
    bool focused = active && interactive() && (!focus_owner || (!focus_owner->modal_blocked && focus_owner->active));
    for (PixUIObject* owner = parent; focused && owner; owner = owner->parent) {
        PixUIForm* form = dynamic_cast<PixUIForm*>(owner);
        if (form && !form->active) focused = false;
    }
    canvas.bevel(rect,true);
    PixUIRect content = {rect.x + UI_CONTENT_INSET,rect.y + UI_CONTENT_INSET,rect.w - UI_CONTENT_INSET * 2 - (multiline? UI_SCROLLBAR : 0),rect.h - UI_CONTENT_INSET * 2};
    int columns = std::max(1,content.w / UI_TEXT_WIDTH);
    std::vector<size_t> lines = multiline? pixui_editor_lines(editor.text,columns) : std::vector<size_t>{0};
    int caret_line = int(std::upper_bound(lines.begin(),lines.end(),editor.cursor) - lines.begin()) - 1;
    int caret_column = 0;
    for (size_t offset = lines[caret_line]; offset < editor.cursor; offset = pixui_next_character(editor.text,offset)) caret_column++;
    int visible = std::max(1,content.h / UI_TEXT_LINE_HEIGHT);
    editor.scroll = std::max(0,std::min(editor.scroll,std::max(0,int(lines.size()) - visible)));
    if (multiline) editor.horizontal = 0;
    else editor.scroll = 0;
    if (focused && editor.reveal) {
        editor.reveal = false;
        if (caret_line < editor.scroll) editor.scroll = caret_line;
        if (caret_line >= editor.scroll + visible) editor.scroll = caret_line - visible + 1;
        if (!multiline) {
            if (caret_column < editor.horizontal) editor.horizontal = caret_column;
            if (caret_column >= editor.horizontal + columns) editor.horizontal = caret_column - columns + 1;
        }
        PixUIRect caret = {content.x + (caret_column - editor.horizontal) * UI_TEXT_WIDTH,content.y + (caret_line - editor.scroll) * UI_TEXT_LINE_HEIGHT,UI_TEXT_WIDTH,UI_TEXT_LINE_HEIGHT};
        canvas.text_input(caret);
    }
    canvas.fill(content,UI_COLOR_LIGHT);
    PixUIRect previous,effective = content;
    bool clipped = canvas.clip(previous);
    if (clipped) pixui_intersect(content,previous,effective);
    canvas.set_clip(&effective);
    for (int line = editor.scroll; line < int(lines.size()) && line <= editor.scroll + visible; line++) {
        size_t end = line + 1 < int(lines.size())? lines[line + 1] : editor.text.size();
        int column = 0;
        for (size_t offset = lines[line]; offset < end; offset = pixui_next_character(editor.text,offset)) {
            if (editor.text[offset] == '\n') break;
            int x = content.x + (column - editor.horizontal) * UI_TEXT_WIDTH;
            int y = content.y + (line - editor.scroll) * UI_TEXT_LINE_HEIGHT;
            bool selection = offset >= std::min(editor.cursor,editor.anchor) && offset < std::max(editor.cursor,editor.anchor);
            if (selection) canvas.fill({x,y,UI_TEXT_WIDTH,UI_TEXT_LINE_HEIGHT},UI_COLOR_SELECTED);
            canvas.text(x,y + UI_CONTENT_INSET,editor.text.substr(offset,pixui_next_character(editor.text,offset) - offset),selection? UI_COLOR_LIGHT : pixui_theme.fonts[PIXUI_FONT_TEXT].color,PIXUI_FONT_TEXT);
            column++;
        }
    }
    if (focused && (canvas.ticks() / UI_CARET_PERIOD) % 2 == 0) canvas.fill({content.x + (caret_column - editor.horizontal) * UI_TEXT_WIDTH,content.y + (caret_line - editor.scroll) * UI_TEXT_LINE_HEIGHT,1,UI_TEXT_LINE_HEIGHT},pixui_theme.fonts[PIXUI_FONT_TEXT].color);
    canvas.set_clip(clipped? &previous : nullptr);

}

void PixUIMemo::place_cursor(int x, int y, int modifiers)
{
    PixUIEditor &editor = model;
    const PixUIRect &rect = bounds;
    int columns = std::max(1,(rect.w - UI_CONTENT_INSET * 2 - (multiline? UI_SCROLLBAR : 0)) / UI_TEXT_WIDTH);
    std::vector<size_t> lines = multiline? pixui_editor_lines(editor.text,columns) : std::vector<size_t>{0};
    int wanted_line = multiline? std::max(0,(y - rect.y - UI_CONTENT_INSET) / UI_TEXT_LINE_HEIGHT + editor.scroll) : 0;
    wanted_line = std::min(wanted_line,int(lines.size()) - 1);
    int wanted_column = std::max(0,(x - rect.x - UI_CONTENT_INSET + UI_TEXT_WIDTH / 2) / UI_TEXT_WIDTH + editor.horizontal);
    size_t offset = lines[wanted_line];
    size_t end = wanted_line + 1 < int(lines.size())? lines[wanted_line + 1] : editor.text.size();
    for (int column = 0; column < wanted_column && offset < end && editor.text[offset] != '\n'; column++) offset = pixui_next_character(editor.text,offset);
    editor.reveal = true;
    editor.cursor = offset;
    if (!(modifiers & PIXUI_SHIFT)) editor.anchor = offset;
}

PixUIEdit::PixUIEdit(PixUICanvas &context, PixUIEditor* state) : PixUIMemo(context,state)
{
    multiline = false;
}

PixUIEdit::PixUIEdit(PixUICanvas &context, PixUIEditor &state) : PixUIEdit(context,&state)
{
}

bool PixUIEdit::event(const PixUIEvent &input)
{
    if (visible && enabled && active && !read_only && input.type == PIXUI_KEY) {
        if (input.key == PIXUI_UP_KEY && increment) {
            increment();
            return true;
        }
        if (input.key == PIXUI_DOWN_KEY && decrement) {
            decrement();
            return true;
        }
    }
    return PixUIMemo::event(input);
}

PixUIMenuItem::PixUIMenuItem(PixUICanvas &context, const std::string &text) : PixUILabel(context,text)
{
}

PixUIMenuItem::~PixUIMenuItem() = default;

void PixUIMenuItem::activate_item()
{
    if (!enabled) return;
    if (activate) activate();
    update();
}

bool PixUIMenuItem::accelerator(const PixUIEvent &input)
{
    if (!enabled || input.type != PIXUI_KEY) return false;
    if (shortcut_key && input.key == shortcut_key && input.mod == shortcut_mod) {
        activate_item();
        return true;
    }
    return submenu && submenu->accelerator(input);
}

void PixUIMenuItem::bind_shortcut(int key, int modifiers, const std::string &label)
{
    shortcut_key = key;
    shortcut_mod = modifiers;
    shortcut = label;
}

void PixUIMenuItem::update()
{
    if (enabled_state) {
        bool value = enabled_state();
        if (enabled != value) repaint = true;
        enabled = value;
    }
    if (checked_state) {
        bool value = checked_state();
        if (checked != value) repaint = true;
        checked = value;
    }
    if (caption_state) {
        std::string value = caption_state();
        if (caption != value) repaint = true;
        caption = value;
    }
}

void PixUIMenuItem::draw()
{
    needs_repaint();
    repaint = false;
    if (!visible) return;
    bool highlight = enabled && highlighted;
    if (highlight) canvas.fill(bounds,UI_COLOR_SELECTED);
    unsigned ink = !enabled? UI_COLOR_DISABLED : (highlight? UI_COLOR_LIGHT : pixui_theme.fonts[PIXUI_FONT_LABEL].color);
    int x = bounds.x + UI_MENU_PADDING,y = bounds.y + (bounds.h - UI_FONT_HEIGHT) / 2;
    if (checkable || check_column) {
        if (checkable) canvas.text(x,y,checked? UI_CHECK_MARK : UI_UNCHECK_MARK,ink);
        x += UI_MENU_CHECK_WIDTH;
    }
    canvas.text(x,y,caption,ink);
    if (!shortcut.empty()) canvas.text(bounds.x + bounds.w - UI_MENU_PADDING - int(shortcut.size()) * UI_FONT_WIDTH,y,shortcut,ink);
}

PixUIMenu::PixUIMenu(PixUICanvas &context) : PixUIControl(context)
{
}

PixUIMenuItem* PixUIMenu::add(const std::string &caption, const std::function<void()> &action)
{
    auto item = std::make_unique<PixUIMenuItem>(canvas,caption);
    item->parent = this;
    item->activate = action;
    PixUIMenuItem* result = item.get();
    items.push_back(std::move(item));
    return result;
}

PixUIMenu* PixUIMenu::child(PixUIMenuItem &item)
{
    item.submenu = std::make_unique<PixUIMenu>(canvas);
    item.submenu->parent = &item;
    item.submenu->visible = false;
    if (!bar) item.shortcut = UI_CONTROL_RIGHT;
    return item.submenu.get();
}

void PixUIMenu::open(size_t index)
{
    if (index >= items.size() || !items[index]->enabled || !items[index]->submenu) return;
    bool toggle = selected == int(index) && items[index]->submenu->visible;
    dismiss();
    repaint = true;
    if (toggle) return;
    selected = int(index);
    items[index]->highlighted = true;
    items[index]->submenu->visible = true;
    layout();
}

void PixUIMenu::dismiss()
{
    if (selected >= 0) repaint = true;
    for (const auto &item : items) {
        item->highlighted = false;
        if (item->submenu) {
            item->submenu->dismiss();
            item->submenu->visible = false;
        }
    }
    selected = -1;
}

bool PixUIMenu::opened() const
{
    return selected >= 0 && items[size_t(selected)]->submenu && items[size_t(selected)]->submenu->visible;
}

void PixUIMenu::layout()
{
    for (const auto &item : items) {
        item->update();
        if (!bar) item->highlighted = item->enabled && pixui_inside(item->bounds,canvas.pointer_x,canvas.pointer_y);
        if (!item->submenu || !item->submenu->visible) continue;
        int x = bar? item->bounds.x : bounds.x + bounds.w;
        int y = bar? item->bounds.y + item->bounds.h : item->bounds.y - UI_MENU_BORDER;
        int width = bar? bounds.w : client_width;
        item->submenu->layout_popup(x,y,width);
        if (!bar && x + item->submenu->bounds.w > width) item->submenu->layout_popup(std::max(0,bounds.x - item->submenu->bounds.w),y,width);
    }
}

bool PixUIMenu::needs_repaint()
{
    bool changed = PixUIObject::needs_repaint();
    for (const auto &item : items) {
        item->update();
        if (item->needs_repaint()) changed = true;
        if (item->submenu && item->submenu->visible && item->submenu->needs_repaint()) changed = true;
    }
    return changed || repaint;
}

void PixUIMenu::draw()
{
    needs_repaint();
    repaint = false;
    if (!visible) return;
    layout();
    if (!bar) {
        canvas.fill(bounds,UI_COLOR_LIGHT);
        canvas.bevel(bounds,false,false);
    }
    for (const auto &item : items) {
        if (!bar) item->draw();
        else if (item->show_title) {
            item->needs_repaint();
            item->repaint = false;
            canvas.fill(item->bounds,item->highlighted? UI_COLOR_SELECTED : UI_COLOR_LIGHT);
            unsigned ink = item->highlighted? UI_COLOR_LIGHT : pixui_theme.fonts[PIXUI_FONT_LABEL].color;
            int x = item->bounds.x + std::max(0,(item->bounds.w - int(item->caption.size()) * UI_FONT_WIDTH) / 2);
            int y = item->bounds.y + (item->bounds.h - UI_FONT_HEIGHT) / 2;
            canvas.text(x,y,item->caption,ink);
            canvas.fill({x,y + UI_FONT_HEIGHT,UI_FONT_WIDTH,1},ink);
        }
    }
    for (const auto &item : items) {
        item->repaint = false;
        if (item->submenu && item->submenu->visible) item->submenu->draw();
    }
    repaint = false;
}

bool PixUIMenu::event(const PixUIEvent &input)
{
    if (!visible || !enabled) return false;
    if (input.type == PIXUI_DOWN || input.type == PIXUI_UP || input.type == PIXUI_MOTION || input.type == PIXUI_WHEEL) {
        canvas.pointer_x = input.x;
        canvas.pointer_y = input.y;
    }
    if (input.type == PIXUI_KEY && input.key == PIXUI_ESCAPE && opened()) {
        canvas.release_capture();
        dismiss();
        return true;
    }
    if (bar && accelerator(input)) return true;
    if (input.type != PIXUI_DOWN && input.type != PIXUI_MOTION) return false;
    // Descendant popups receive pointer input before their parent rows.
    for (const auto &item : items) {
        if (!item->submenu || !item->submenu->visible || !item->submenu->contains(input.x,input.y)) continue;
        return item->submenu->event(input);
    }
    for (size_t i = 0; i < items.size(); i++) {
        PixUIMenuItem &item = *items[i];
        if ((bar && !item.show_title) || !pixui_inside(item.bounds,input.x,input.y)) {
            if (!bar && int(i) != selected) item.highlighted = false;
            continue;
        }
        if (input.type == PIXUI_MOTION) {
            if (!item.enabled) return !bar;
            if (item.submenu && (!bar || opened()) && (selected != int(i) || !item.submenu->visible)) open(i);
            else if (!bar && !item.submenu && selected != int(i)) {
                dismiss();
                selected = int(i);
            }
            if (!bar && !item.highlighted) repaint = true;
            if (!bar) item.highlighted = true;
            return !bar || opened();
        }
        if (input.button != PIXUI_LEFT_BUTTON) return false;
        if (!item.enabled) return true;
        if (item.submenu) open(i);
        else {
            item.activate_item();
            // Propagate dismissal to the menu-bar owner after invoking an action.
            PixUIMenu* root = this;
            while (root->parent) {
                PixUIMenuItem* owner = dynamic_cast<PixUIMenuItem*>(root->parent);
                if (!owner) break;
                root = static_cast<PixUIMenu*>(owner->parent);
            }
            root->dismiss();
        }
        return true;
    }
    if (!bar) return pixui_inside(bounds,input.x,input.y);
    if (input.type == PIXUI_DOWN && opened()) {
        for (const auto &item : items) {
            if (!item->show_title && pixui_inside(item->bounds,input.x,input.y)) return false;
        }
        dismiss();
        return true;
    }
    return false;
}

bool PixUIMenu::accelerator(const PixUIEvent &input)
{
    for (const auto &item : items) {
        if (!item->accelerator(input)) continue;
        if (!item->submenu || input.key != item->shortcut_key || input.mod != item->shortcut_mod) dismiss();
        return true;
    }
    return false;
}

bool PixUIMenu::contains(int x, int y) const
{
    if (!visible) return false;
    if (!bar && pixui_inside(bounds,x,y)) return true;
    for (const auto &item : items) {
        if (bar && item->show_title && pixui_inside(item->bounds,x,y)) return true;
        if (item->submenu && item->submenu->contains(x,y)) return true;
    }
    return false;
}

void PixUIMenu::layout_popup(int x, int y, int width)
{
    client_width = width;
    int caption_width = 0,shortcut_width = 0;
    bool check_column = std::any_of(items.begin(),items.end(),[](const auto &item) { return item->checkable; });
    for (const auto &item : items) {
        item->update();
        item->check_column = check_column;
        caption_width = std::max(caption_width,int(item->caption.size()) * UI_FONT_WIDTH);
        shortcut_width = std::max(shortcut_width,int(item->shortcut.size()) * UI_FONT_WIDTH);
    }
    int extent = caption_width + (shortcut_width? UI_MENU_SHORTCUT_GAP + shortcut_width : 0) + (check_column? UI_MENU_CHECK_WIDTH : 0);
    bounds = {x,y,extent + UI_MENU_PADDING * 2 + UI_MENU_BORDER * 2,int(items.size()) * UI_MENU_ROW + UI_MENU_BORDER * 2};
    PixUILayout rows = {bounds.x + UI_MENU_BORDER,bounds.y + UI_MENU_BORDER,0};
    for (const auto &item : items) item->bounds = rows.column(bounds.w - UI_MENU_BORDER * 2,UI_MENU_ROW);
    layout();
}

PixUIForm::PixUIForm(PixUICanvas &context) : PixUIControl(context)
{
}

PixUIForm::~PixUIForm()
{
    for (PixUIMemo* memo : tab_order) memo->focus_owner = nullptr;
}

void PixUIForm::decorate(const std::string &caption)
{
    frame = add<PixUIPanel>();
    title = add<PixUIPanel>();
    title->pointer = [this](const PixUIEvent &input) { title_event(input.x,input.y); };
    title_clip = add<PixUIForm>();
    title_label = title_clip->add<PixUILabel>(caption);
    title_label->font_role = PIXUI_FONT_HEADER;
    title_label->centered = true;
    system = add<PixUIButton>(UI_CONTROL_SYSTEM);
    system->pointer = [this](const PixUIEvent &input) { title_event(input.x,input.y,true); };
    minimize = add<PixUIButton>(UI_CONTROL_DOWN);
    minimize->activate = [this]() { command(PIXUI_MINIMIZE); };
    maximize = add<PixUIButton>(UI_CONTROL_UP);
    maximize->activate = [this]() { command(canvas.maximized()? PIXUI_RESTORE : PIXUI_MAXIMIZE); };

}

PixUIMenu* PixUIForm::create_menu()
{
    if (menu_bar) return menu_bar;
    menu_bar = add<PixUIMenu>();
    menu_bar->bar = true;
    menu_bar->bounds = bounds;
    return menu_bar;
}

void PixUIForm::add_system_menu()
{
    PixUIMenu* menu = create_menu();
    size_t index = menu->items.size();
    PixUIMenuItem* heading = menu->add(UI_CONTROL_SYSTEM);
    heading->show_title = false;
    heading->bounds = {UI_FRAME,UI_FRAME,UI_WINDOW_TITLE_HEIGHT,UI_WINDOW_TITLE_HEIGHT};
    system_menu = [menu,index]() { menu->open(index); };
    heading->bind_shortcut(' ',PIXUI_ALT,"");
    heading->activate = system_menu;
    PixUIMenu* popup = menu->child(*heading);
    popup->add("Restore",[this]() { command(PIXUI_RESTORE); });
    popup->add("Minimize",[this]() { command(PIXUI_MINIMIZE); });
    popup->add("Maximize",[this]() { command(PIXUI_MAXIMIZE); });
    popup->add(UI_CLOSE_TEXT,[this]() { command(PIXUI_CLOSE); })->bind_shortcut(PIXUI_F4,PIXUI_ALT,"Alt+F4");
}

void PixUIForm::layout_window(const std::string &caption)
{
    if (menu_bar) {
        menu_bar->bounds = bounds;
        menu_bar->layout();
    }
    if (!frame) return;
    if (caption != window_caption) {
        canvas.window_title(caption);
        window_caption = caption;
    }
    frame->bounds = bounds;
    title->bounds = {UI_FRAME,UI_FRAME,bounds.w - UI_FRAME * 2,UI_WINDOW_TITLE_HEIGHT};
    title->color = modal_blocked || !active? UI_COLOR_LIGHT : UI_COLOR_TITLE;
    title_clip->bounds = {UI_FRAME + UI_WINDOW_TITLE_HEIGHT,UI_FRAME,bounds.w - UI_FRAME * 2 - UI_WINDOW_TITLE_HEIGHT * 3,UI_WINDOW_TITLE_HEIGHT};
    title_label->bounds = title_clip->bounds;
    title_label->caption = caption;
    title_label->color = modal_blocked || !active? UI_COLOR_DARK : pixui_theme.fonts[PIXUI_FONT_HEADER].color;
    system->bounds = {UI_FRAME,UI_FRAME,UI_WINDOW_TITLE_HEIGHT,UI_WINDOW_TITLE_HEIGHT};
    PixUILayout buttons = {bounds.w - UI_FRAME - UI_WINDOW_TITLE_HEIGHT * 2,UI_FRAME,0};
    minimize->bounds = buttons.row(UI_WINDOW_TITLE_HEIGHT,UI_WINDOW_TITLE_HEIGHT);
    maximize->bounds = buttons.row(UI_WINDOW_TITLE_HEIGHT,UI_WINDOW_TITLE_HEIGHT);
}

void PixUIForm::command(PixUIWindowCommand value)
{
    if (value == PIXUI_CLOSE) {
        if (close) close();
        return;
    }
    canvas.window_command(value,canvas.window_rect());
}

void PixUIForm::title_event(int x, int y, bool menu_button)
{
    uint64_t now = canvas.ticks();
    bool twice = click_time && now - click_time <= UI_DOUBLE_CLICK_MS && std::abs(x - click_x) < UI_SCROLLBAR && std::abs(y - click_y) < UI_SCROLLBAR;
    click_time = twice? 0 : now;
    click_x = x;
    click_y = y;
    canvas.pointer_x = x;
    canvas.pointer_y = y;
    if (menu_button) {
        if (twice) command(PIXUI_CLOSE);
        else if (system_menu) system_menu();
    } else if (twice) command(canvas.maximized()? PIXUI_RESTORE : PIXUI_MAXIMIZE);
    else if (!canvas.maximized()) {
        window_drag = PIXUI_MOVE;
        canvas.pointer_position(drag_x,drag_y);
        drag_window = canvas.window_rect();
        canvas.captured = this;
    }
}

PixUIHit PixUIForm::hit_test(int x, int y) const
{
    if (modal_blocked || (menu_bar && menu_bar->opened())) return PIXUI_HIT_NORMAL;
    bool left = x < UI_FRAME,right = x >= bounds.w - UI_FRAME;
    bool top = y < UI_FRAME,bottom = y >= bounds.h - UI_FRAME;
    if (!canvas.maximized()) {
        if (top && left) return PIXUI_HIT_TOPLEFT;
        if (top && right) return PIXUI_HIT_TOPRIGHT;
        if (bottom && left) return PIXUI_HIT_BOTTOMLEFT;
        if (bottom && right) return PIXUI_HIT_BOTTOMRIGHT;
        if (top) return PIXUI_HIT_TOP;
        if (bottom) return PIXUI_HIT_BOTTOM;
        if (left) return PIXUI_HIT_LEFT;
        if (right) return PIXUI_HIT_RIGHT;
        if (x >= bounds.w - UI_SCROLLBAR && y >= bounds.h - UI_SCROLLBAR) return PIXUI_HIT_BOTTOMRIGHT;
    }
    if (y >= UI_FRAME && y < UI_FRAME + UI_WINDOW_TITLE_HEIGHT && x >= UI_FRAME + UI_WINDOW_TITLE_HEIGHT && x < bounds.w - UI_FRAME - UI_WINDOW_TITLE_HEIGHT * 2) return PIXUI_HIT_MOVE;
    return PIXUI_HIT_NORMAL;
}

void PixUIForm::cancel_capture()
{
    window_drag = PIXUI_CLOSE;
    PixUIObject::cancel_capture();
}

void PixUIForm::register_focus(PixUIMemo &memo)
{
    if (memo.focus_owner == this) return;
    if (memo.focus_owner) memo.focus_owner->unregister_focus(memo);
    memo.focus_owner = this;
    memo.active = false;
    tab_order.push_back(&memo);
    if (!focused) focus_control(memo);
}

void PixUIForm::unregister_focus(PixUIMemo &memo)
{
    tab_order.erase(std::remove(tab_order.begin(),tab_order.end(),&memo),tab_order.end());
    memo.focus_owner = nullptr;
    if (saved_focus == &memo) saved_focus = nullptr;
    if (focused != &memo) return;
    focused = nullptr;
    if (!tab_order.empty()) focus_control(*tab_order[0]);
}

void PixUIForm::focus_control(PixUIMemo &memo)
{
    if (modal_blocked) saved_focus = &memo;
    keyboard_focus = nullptr;
    if (focused) {
        focused->active = false;
        focused->repaint = true;
    }
    focused = &memo;
    focused->repaint = true;
    focused->active = true;
    focused->model.reveal = true;
    canvas.text_input(focused->bounds);
}

void PixUIForm::focus_object(PixUIObject &control)
{
    if (keyboard_focus == &control) return;
    if (focused) {
        focused->active = false;
        focused->repaint = true;
    }
    keyboard_focus = &control;
    control.repaint = true;
}

bool PixUIForm::needs_repaint()
{
    prepare_modal();
    if (PixUIObject::needs_repaint()) return true;
    if (modal_dialog && modal_dialog->visible && modal_dialog->needs_repaint()) return true;
    for (const auto &element : elements) {
        bool changed = element->needs_repaint();
        if (changed && element->visible && element->layout_visible) return true;
    }
    return repaint;
}

void PixUIForm::layout()
{
    for (const auto &element : elements) element->layout();
}

void PixUIForm::present(std::unique_ptr<PixUIDialog> dialog)
{
    canvas.release_capture();
    if (menu_bar) menu_bar->dismiss();
    if (!modal()) saved_focus = focused;
    repaint = true;
    pending_dialog = std::move(dialog);
    modal_blocked = true;
}

void PixUIForm::dismiss_dialog()
{
    repaint = true;
    pending_dialog.reset();
    if (modal_dialog) modal_dialog->dismiss();
}

bool PixUIForm::modal() const
{
    return pending_dialog || (modal_dialog && modal_dialog->visible);
}

void PixUIForm::clear()
{
    for (PixUIObject* owner = canvas.captured; owner; owner = owner->parent) {
        if (owner != this) continue;
        canvas.release_capture();
        break;
    }
    pending_dialog.reset();
    modal_dialog.reset();
    saved_focus = nullptr;
    menu_bar = nullptr;
    frame = nullptr;
    title = nullptr;
    title_clip = nullptr;
    title_label = nullptr;
    system = nullptr;
    minimize = nullptr;
    maximize = nullptr;
    default_button = nullptr;
    system_menu = {};
    modal_blocked = false;
    window_caption.clear();
    keyboard_focus = nullptr;
    repaint = true;
    elements.clear();
}

void PixUIForm::remove(PixUIObject &child)
{
    auto found = std::find_if(elements.begin(),elements.end(),[&child](const auto &element) { return element.get() == &child; });
    if (found == elements.end()) return;
    for (PixUIObject* owner = canvas.captured; owner; owner = owner->parent) {
        if (owner != &child) continue;
        canvas.release_capture();
        break;
    }
    for (PixUIObject* owner = keyboard_focus; owner; owner = owner->parent) {
        if (owner != &child) continue;
        keyboard_focus = nullptr;
        break;
    }
    repaint = true;
    if (&child == menu_bar) menu_bar = nullptr;
    if (&child == default_button) default_button = nullptr;
    elements.erase(found);
}

const std::vector<std::unique_ptr<PixUIObject>> &PixUIForm::children() const
{
    return elements;
}

void PixUIForm::draw()
{
    needs_repaint();
    repaint = false;
    if (!visible) return;
    prepare_modal();
    layout();
    PixUIRect area = clip_client? client : bounds;
    PixUIRect previous,effective = area;
    bool clipped = canvas.clip(previous);
    if (clip_children) {
        if (clipped && !pixui_intersect(area,previous,effective)) return;
        canvas.set_clip(&effective);
    }
    for (const auto &element : elements) {
        element->needs_repaint();
        if (element->visible && element->layout_visible && element.get() != menu_bar) element->draw();
    }
    if (menu_bar) menu_bar->draw();
    canvas.set_clip(clipped? &previous : nullptr);
    if (modal_dialog && modal_dialog->visible) modal_dialog->draw();
    repaint = false;
}

bool PixUIForm::event(const PixUIEvent &input)
{
    if (!visible || !enabled) return false;
    prepare_modal();
    if (modal_dialog && modal_dialog->visible) return modal_dialog->event(input);
    if (input.type == PIXUI_KEY && input.key == PIXUI_ESCAPE) canvas.release_capture();
    if (canvas.captured == this && window_drag != PIXUI_CLOSE) {
        if (input.type == PIXUI_UP && input.button == PIXUI_LEFT_BUTTON) {
            cancel_capture();
            return true;
        }
        if (input.type == PIXUI_MOTION) {
            int x,y;
            canvas.pointer_position(x,y);
            PixUIRect rect = drag_window;
            if (window_drag == PIXUI_MOVE) {
                rect.x += x - drag_x;
                rect.y += y - drag_y;
                canvas.window_command(PIXUI_MOVE,rect);
            } else {
                int dx = x - drag_x;
                int dy = y - drag_y;
                bool left = resize_edge == PIXUI_HIT_LEFT || resize_edge == PIXUI_HIT_TOPLEFT || resize_edge == PIXUI_HIT_BOTTOMLEFT;
                bool right = resize_edge == PIXUI_HIT_RIGHT || resize_edge == PIXUI_HIT_TOPRIGHT || resize_edge == PIXUI_HIT_BOTTOMRIGHT;
                bool top = resize_edge == PIXUI_HIT_TOP || resize_edge == PIXUI_HIT_TOPLEFT || resize_edge == PIXUI_HIT_TOPRIGHT;
                bool bottom = resize_edge == PIXUI_HIT_BOTTOM || resize_edge == PIXUI_HIT_BOTTOMLEFT || resize_edge == PIXUI_HIT_BOTTOMRIGHT;
                rect = drag_window;
                if (left || right) rect.w = std::max(UI_MIN_WIDTH,drag_window.w + (left? -dx : dx));
                if (top || bottom) rect.h = std::max(UI_MIN_HEIGHT,drag_window.h + (top? -dy : dy));
                if (left) rect.x += drag_window.w - rect.w;
                if (top) rect.y += drag_window.h - rect.h;
                if (left || top) canvas.window_command(PIXUI_MOVE,rect);
                canvas.window_command(PIXUI_RESIZE,rect);
            }
            return true;
        }
    }
    if (menu_bar && menu_bar->event(input)) return true;
    if (frame && input.type == PIXUI_DOWN && input.button == PIXUI_LEFT_BUTTON) {
        PixUIHit hit = hit_test(input.x,input.y);
        if (hit != PIXUI_HIT_NORMAL && hit != PIXUI_HIT_MOVE) {
            window_drag = PIXUI_RESIZE;
            resize_edge = hit;
            canvas.pointer_position(drag_x,drag_y);
            drag_window = canvas.window_rect();
            canvas.captured = this;
            return true;
        }
    }
    if (keyboard_focus && keyboard_focus != this && (input.type == PIXUI_KEY || input.type == PIXUI_TEXT) && keyboard_focus->interactive() && keyboard_focus->event(input)) return true;
    if (input.type == PIXUI_KEY && input.key == PIXUI_TAB && !tab_order.empty()) {
        auto found = std::find(tab_order.begin(),tab_order.end(),focused);
        int index = found == tab_order.end()? -1 : int(found - tab_order.begin());
        int step = input.mod & PIXUI_SHIFT? -1 : 1;
        for (size_t attempt = 0; attempt < tab_order.size(); attempt++) {
            index = (index + step + int(tab_order.size())) % int(tab_order.size());
            PixUIMemo* candidate = tab_order[size_t(index)];
            bool available = candidate->enabled && candidate->visible;
            for (PixUIObject* owner = candidate->parent; owner; owner = owner->parent) available = available && owner->enabled && owner->visible;
            if (!available) continue;
            focus_control(*tab_order[size_t(index)]);
            if (tab_changed) tab_changed(focused);
            return true;
        }
    }
    if (default_button && !keyboard_focus && input.type == PIXUI_KEY && input.key == PIXUI_RETURN) return default_button->event(input);
    int x = 0,y = 0;
    bool pointer = false;
    if (input.type == PIXUI_DOWN || input.type == PIXUI_UP) {
        x = input.x;
        y = input.y;
        pointer = true;
    } else if (input.type == PIXUI_MOTION) {
        x = input.x;
        y = input.y;
        pointer = true;
    } else if (input.type == PIXUI_WHEEL) {
        x = input.x;
        y = input.y;
        pointer = true;
    }
    if (pointer && clip_children && !pixui_inside(clip_client? client : bounds,x,y)) return false;
    // Reverse paint order gives the topmost control first chance to consume input.
    for (auto element = elements.rbegin(); element != elements.rend(); element++) {
        if (element->get() != menu_bar && (*element)->visible && (*element)->layout_visible && (*element)->enabled && (*element)->event(input)) {
            (*element)->repaint = true;
            return true;
        }
    }
    return PixUIControl::event(input);
}

void PixUIForm::prepare_modal()
{
    if (pending_dialog) modal_dialog = std::move(pending_dialog);
    if (modal_dialog && !modal_dialog->visible) {
        repaint = true;
        modal_dialog.reset();
        if (saved_focus) focus_control(*saved_focus);
        saved_focus = nullptr;
    }
    modal_blocked = modal_dialog != nullptr;
    if (modal_dialog) {
        modal_dialog->active = active;
        auto dialog = dynamic_cast<PixUIDialog*>(this);
        modal_dialog->center(bounds,dialog? dialog->available : bounds);
    }
}

PixUIGroupBox::PixUIGroupBox(PixUICanvas &context, const std::string &text) : PixUIForm(context),caption(text),horizontal(context),vertical(context)
{
    clip_client = true;
    horizontal.parent = vertical.parent = this;
    horizontal.position = &offset_x;
    horizontal.step = UI_FONT_WIDTH;
    horizontal.changed = [this]() { layout(); };
    vertical.position = &scroll;
    vertical.vertical = true;
    vertical.changed = [this]() { layout(); };
}

void PixUIGroupBox::place(PixUIObject &child, const PixUIRect &rect, int flags)
{
    if (child.parent != this) return;
    auto found = std::find_if(placements.begin(),placements.end(),[&child](const PixUIPlacement &placement) { return placement.child == &child; });
    if (found == placements.end()) placements.push_back({&child,rect,flags});
    else *found = {&child,rect,flags};
}

void PixUIGroupBox::reveal(PixUIObject &child)
{
    auto found = std::find_if(placements.begin(),placements.end(),[&child](const PixUIPlacement &placement) { return placement.child == &child; });
    if (found == placements.end()) return;
    scroll = std::max(0,found->rect.y - client.h / 2);
    offset_x = std::max(0,found->rect.x - client.w / 2);
    layout();
}

void PixUIGroupBox::clear()
{
    placements.clear();
    scroll = 0;
    offset_x = 0;
    horizontal.visible = vertical.visible = false;
    PixUIForm::clear();
}

void PixUIGroupBox::remove(PixUIObject &child)
{
    placements.erase(std::remove_if(placements.begin(),placements.end(),[&child](const PixUIPlacement &placement) { return placement.child == &child; }),placements.end());
    PixUIForm::remove(child);
}

void PixUIGroupBox::layout()
{
    client = {bounds.x + inset_x,bounds.y + inset_top,std::max(0,bounds.w - inset_x * 2),std::max(0,bounds.h - inset_top - inset_bottom)};
    int required_width = content_width,required_height = content_height;
    for (const PixUIPlacement &placement : placements) {
        if (!(placement.flags & (PIXUI_STRETCH_X | PIXUI_ALIGN_RIGHT))) required_width = std::max(required_width,placement.rect.x + placement.rect.w);
        if (!(placement.flags & (PIXUI_STRETCH_Y | PIXUI_ALIGN_BOTTOM))) required_height = std::max(required_height,placement.rect.y + placement.rect.h);
    }
    bool show_horizontal = required_width > client.w,show_vertical = required_height > client.h;
    if (show_vertical && required_width > client.w - UI_SCROLLBAR) show_horizontal = true;
    if (show_horizontal && required_height > client.h - UI_SCROLLBAR) show_vertical = true;
    if (show_vertical) client.w = std::max(0,client.w - UI_SCROLLBAR);
    if (show_horizontal) client.h = std::max(0,client.h - UI_SCROLLBAR);
    int layout_width = std::max(client.w,required_width),layout_height = std::max(client.h,required_height);
    scroll = std::max(0,std::min(scroll,std::max(0,required_height - client.h)));
    offset_x = std::max(0,std::min(offset_x,std::max(0,required_width - client.w)));
    horizontal.bounds = {client.x,client.y + client.h,client.w,UI_SCROLLBAR};
    horizontal.page = client.w;
    horizontal.total = layout_width;
    horizontal.visible = show_horizontal;
    vertical.bounds = {client.x + client.w,client.y,UI_SCROLLBAR,client.h};
    vertical.page = client.h;
    vertical.total = layout_height;
    vertical.step = wheel_step;
    vertical.visible = show_vertical;
    for (const PixUIPlacement &placement : placements) {
        PixUIRect rect = placement.rect;
        if (placement.flags & PIXUI_STRETCH_X) rect.w += layout_width;
        if (placement.flags & PIXUI_STRETCH_Y) rect.h += layout_height;
        if (placement.flags & PIXUI_ALIGN_RIGHT) rect.x += layout_width;
        if (placement.flags & PIXUI_ALIGN_BOTTOM) rect.y += layout_height;
        rect.x += client.x - offset_x;
        rect.y += client.y - scroll;
        rect.w = std::max(0,rect.w);
        rect.h = std::max(0,rect.h);
        placement.child->bounds = rect;
        placement.child->layout_visible = !(placement.flags & PIXUI_FULLY_VISIBLE) || (rect.y >= client.y && rect.y + rect.h <= client.y + client.h);
        placement.child->layout();
    }
}

void PixUIGroupBox::draw()
{
    needs_repaint();
    repaint = false;
    if (!visible) return;
    canvas.fill(bounds,color);
    if (border) {
        PixUIRect frame = {bounds.x + UI_TEXT_PADDING,bounds.y + UI_MARGIN + UI_FONT_HEIGHT / 2,std::max(0,bounds.w - UI_TEXT_PADDING * 2),std::max(0,bounds.h - UI_MARGIN - UI_FONT_HEIGHT / 2 - UI_TEXT_PADDING)};
        canvas.fill({frame.x,frame.y,frame.w,1},UI_COLOR_DARK);
        canvas.fill({frame.x,frame.y,1,frame.h},UI_COLOR_DARK);
        canvas.fill({frame.x,frame.y + frame.h - 1,frame.w,1},UI_COLOR_DARK);
        canvas.fill({frame.x + frame.w - 1,frame.y,1,frame.h},UI_COLOR_DARK);
    }
    if (!caption.empty()) {
        canvas.fill({bounds.x + UI_MARGIN - UI_TEXT_PADDING,bounds.y + UI_MARGIN,int(caption.size()) * UI_FONT_WIDTH + UI_TEXT_PADDING * 2,UI_FONT_HEIGHT},color);
        canvas.text(bounds.x + UI_MARGIN,bounds.y + UI_MARGIN,caption,pixui_theme.fonts[PIXUI_FONT_LABEL].color);
    }
    PixUIForm::draw();
    horizontal.draw();
    vertical.draw();
    repaint = false;
}

bool PixUIGroupBox::event(const PixUIEvent &input)
{
    if (!visible || !enabled) return false;
    if (horizontal.event(input) || vertical.event(input)) return true;
    if (PixUIForm::event(input)) return true;
    if ((horizontal.visible || vertical.visible) && input.type == PIXUI_WHEEL && pixui_inside(client,input.x,input.y)) {
        scroll -= input.wheel_y * PIXUI_WHEEL_SCROLL_Y * wheel_step;
        offset_x -= input.wheel_x * PIXUI_WHEEL_SCROLL_X * horizontal.step;
        layout();
        return true;
    }
    return false;
}

PixTabGroup::PixTabGroup(PixUICanvas &context, const std::string &text) : PixUIGroupBox(context),title(text)
{
    border = false;
    inset_x = 0;
    inset_top = 0;
    inset_bottom = 0;
}

PixTab::PixTab(PixUICanvas &context) : PixUIGroupBox(context)
{
    inset_x = inset_bottom = UI_BORDER_WIDTH;
    inset_top = UI_TAB_HEIGHT + UI_BORDER_WIDTH;
}

PixTabGroup* PixTab::add_group(const std::string &title)
{
    PixTabGroup* page = add<PixTabGroup>(title);
    page->visible = false;
    pages.push_back(page);
    if (active_page == SIZE_MAX) select(0);
    return page;
}

size_t PixTab::selected() const
{
    return active_page;
}

bool PixTab::select(size_t index)
{
    if (!selection_enabled || !enabled || index >= pages.size() || !pages[index]->enabled) return false;
    if (active_page == index) return true;
    if (active_page < pages.size()) {
        PixTabGroup* departing = pages[active_page];
        for (PixUIObject* owner = canvas.captured; owner; owner = owner->parent) {
            if (owner != departing) continue;
            canvas.release_capture();
            break;
        }
        departing->visible = false;
    }
    repaint = true;
    active_page = index;
    pages[index]->visible = true;
    layout();
    if (changed) changed(index);
    return true;
}

void PixTab::layout()
{
    PixUIGroupBox::layout();
    headers.clear();
    int x = bounds.x + inset_x;
    int right = bounds.x + std::max(0,bounds.w - inset_x);
    for (PixTabGroup* page : pages) {
        int width = std::min(std::max(0,right - x),int(page->title.size()) * UI_FONT_WIDTH + UI_TAB_PADDING * 2);
        headers.push_back({x,bounds.y + (headers.size() == active_page? 0 : UI_TAB_RAISE),width,UI_TAB_HEIGHT - (headers.size() == active_page? 0 : UI_TAB_RAISE)});
        x += width;
        page->bounds = {client.x,client.y,client.w,std::max(0,client.h - footer_height)};
        page->layout();
    }
}

void PixTab::draw()
{
    needs_repaint();
    repaint = false;
    if (!visible) return;
    canvas.fill(bounds,color);
    PixUIRect body = {bounds.x,bounds.y + UI_TAB_HEIGHT,bounds.w,std::max(0,bounds.h - UI_TAB_HEIGHT)};
    canvas.fill({body.x,body.y,body.w,1},UI_COLOR_DARK);
    canvas.fill({body.x,body.y,1,body.h},UI_COLOR_DARK);
    canvas.fill({body.x,body.y + body.h - 1,body.w,1},UI_COLOR_DARK);
    canvas.fill({body.x + body.w - 1,body.y,1,body.h},UI_COLOR_DARK);
    canvas.bevel({body.x + 1,body.y + 1,body.w - 2,body.h - 2},false,false);
    PixUIForm::draw();
    PixUIRect previous,strip = {bounds.x,bounds.y,bounds.w,inset_top},effective = strip;
    bool clipped = canvas.clip(previous);
    if (clipped && !pixui_intersect(strip,previous,effective)) return;
    canvas.set_clip(&effective);
    for (size_t i = 0; i < pages.size(); i++) {
        PixUIRect rect = headers[i];
        if (rect.w <= 0) continue;
        unsigned face = i == active_page? color : UI_COLOR_FACE;
        int slope = std::min(UI_TAB_SLOPE,rect.w / 2);
        for (int row = 0; row < slope; row++) {
            canvas.fill({rect.x + slope - row,rect.y + row,rect.w - (slope - row) * 2,1},face);
            canvas.point(rect.x + slope - row - 1,rect.y + row,UI_COLOR_DARK);
            if (row) canvas.point(rect.x + slope - row,rect.y + row,UI_COLOR_LIGHT);
            canvas.point(rect.x + rect.w - slope + row,rect.y + row,UI_COLOR_DARK);
        }
        canvas.fill({rect.x,rect.y + slope,rect.w,rect.h - slope},face);
        canvas.fill({rect.x + slope,rect.y,rect.w - slope * 2,1},UI_COLOR_DARK);
        canvas.fill({rect.x + slope,rect.y + 1,rect.w - slope * 2,1},UI_COLOR_LIGHT);
        canvas.fill({rect.x,rect.y + slope,1,rect.h - slope},UI_COLOR_DARK);
        canvas.fill({rect.x + 1,rect.y + slope,1,rect.h - slope},UI_COLOR_LIGHT);
        canvas.fill({rect.x + rect.w - 1,rect.y + slope,1,rect.h - slope},UI_COLOR_DARK);
        if (i == active_page) canvas.fill({rect.x + 1,body.y,rect.w - 2,UI_BORDER_WIDTH},color);
        PixUIRect header_clip;
        if (!pixui_intersect(rect,effective,header_clip)) continue;
        canvas.set_clip(&header_clip);
        int label_x = rect.x + std::max(UI_TAB_PADDING,(rect.w - int(pages[i]->title.size()) * UI_FONT_WIDTH) / 2);
        int label_y = rect.y + (rect.h - UI_FONT_HEIGHT) / 2;
        unsigned ink = selection_enabled && pages[i]->enabled? pixui_theme.fonts[PIXUI_FONT_LABEL].color : UI_COLOR_DISABLED;
        canvas.text(label_x,label_y,pages[i]->title,ink);
        if (i == active_page) canvas.text(label_x + 1,label_y,pages[i]->title,ink);
        canvas.set_clip(&effective);
    }
    canvas.set_clip(clipped? &previous : nullptr);
    repaint = false;
}

bool PixTab::event(const PixUIEvent &input)
{
    if (!visible || !enabled) return false;
    if (input.type == PIXUI_DOWN && input.button == PIXUI_LEFT_BUTTON) {
        for (size_t i = 0; i < headers.size(); i++) {
            if (!pixui_inside(headers[i],input.x,input.y)) continue;
            select(i);
            return true;
        }
    }
    return PixUIGroupBox::event(input);
}

void PixTab::remove(PixUIObject &child)
{
    auto found = std::find(pages.begin(),pages.end(),&child);
    bool notify = false;
    if (found != pages.end()) {
        size_t index = size_t(found - pages.begin());
        notify = index == active_page;
        pages.erase(found);
        if (pages.empty()) active_page = SIZE_MAX;
        else if (notify) active_page = std::min(index,pages.size() - 1);
        else if (index < active_page) active_page--;
        for (size_t i = 0; i < pages.size(); i++) pages[i]->visible = i == active_page;
    }
    PixUIGroupBox::remove(child);
    layout();
    if (notify && changed) changed(active_page);
}

void PixTab::clear()
{
    pages.clear();
    headers.clear();
    active_page = SIZE_MAX;
    PixUIGroupBox::clear();
}

PixUIImageView::PixUIImageView(PixUICanvas &context) : PixUIForm(context)
{
    image = add<PixUIImage>();
    horizontal = add<PixUIScrollbar>();
    horizontal->position = &image->offset_x;
    horizontal->step = UI_FONT_WIDTH;
    vertical = add<PixUIScrollbar>();
    vertical->position = &image->offset_y;
    vertical->step = UI_LINE_HEIGHT;
    vertical->vertical = true;
}

void PixUIImageView::set_image(void* texture, int width, int height)
{
    if (texture != image->texture || width != image->width || height != image->height) repaint = true;
    if (texture != image->texture) {
        image->offset_x = 0;
        image->offset_y = 0;
    }
    image->texture = texture;
    image->width = width;
    image->height = height;
}

void PixUIImageView::layout()
{
    image->bounds = bounds;
    image->scaled = scaled;
    image->placeholder = placeholder;
    if (!scaled) {
        image->bounds.w = std::max(0,bounds.w - UI_SCROLLBAR);
        image->bounds.h = std::max(0,bounds.h - UI_SCROLLBAR);
    }
    image->layout();
    horizontal->bounds = {image->bounds.x,image->bounds.y + image->bounds.h,image->bounds.w,UI_SCROLLBAR};
    horizontal->page = image->bounds.w;
    horizontal->total = image->width;
    horizontal->visible = !scaled;
    vertical->bounds = {image->bounds.x + image->bounds.w,image->bounds.y,UI_SCROLLBAR,image->bounds.h};
    vertical->page = image->bounds.h;
    vertical->total = image->height;
    vertical->visible = !scaled;
}

void PixUIImageView::draw()
{
    if (!visible) return;
    canvas.fill(bounds,UI_COLOR_LIGHT);
    PixUIForm::draw();
}

PixUICarousel::PixUICarousel(PixUICanvas &context) : PixUIForm(context)
{
    previews = add<PixUIForm>();
    scrollbar = add<PixUIScrollbar>();
    scrollbar->position = &offset;
    scrollbar->step = cell_width;
    previews->wheel = [this](const PixUIEvent &input) {
        offset -= (input.wheel_y * PIXUI_WHEEL_SCROLL_Y + input.wheel_x * PIXUI_WHEEL_SCROLL_X) * cell_width;
        layout();
    };
}

PixUICarousel::~PixUICarousel()
{
    for (const auto &entry : cache) {
        if (entry.second.texture && release) release(entry.second.texture);
    }
}

bool PixUICarousel::event(const PixUIEvent &input)
{
    if (!visible || !enabled) return false;
    PixUIForm* root = this;
    for (PixUIObject* owner = parent; owner; owner = owner->parent) {
        if (auto form = dynamic_cast<PixUIForm*>(owner)) root = form;
    }
    if (input.type == PIXUI_DOWN && input.button == PIXUI_LEFT_BUTTON && pixui_inside(bounds,input.x,input.y)) root->focus_object(*this);
    if (input.type == PIXUI_KEY && root->keyboard_focus == this && (input.key == PIXUI_LEFT || input.key == PIXUI_RIGHT)) {
        if (!count) return true;
        int next = selection < 0? 0 : std::max(0,std::min(int(count) - 1,selection + (input.key == PIXUI_LEFT? -1 : 1)));
        if (next != selection && (!selected || selected(size_t(next)))) select(next);
        return true;
    }
    return PixUIForm::event(input);
}

void PixUICarousel::items_changed(size_t total, bool reset)
{
    // Releasing before replacement prevents stale resources after workdir changes.
    for (const auto &entry : cache) {
        if (entry.second.texture && release) release(entry.second.texture);
    }
    cache.clear();
    repaint = true;
    count = total;
    invalid = true;
    if (reset) {
        offset = 0;
        selection = -1;
    } else if (selection >= int(count)) selection = -1;
}

void PixUICarousel::select(int index)
{
    int next = index >= 0 && size_t(index) < count? index : -1;
    if (selection != next) repaint = true;
    selection = next;
    if (selection >= 0) {
        int left = selection * cell_width;
        if (left < offset) offset = left;
        else if (left + cell_width > offset + bounds.w) offset = left + cell_width - bounds.w;
    }
    layout();
}

void PixUICarousel::layout()
{
    trim_cache(std::max(size_t(1),cache_limit));
    offset = std::max(0,std::min(offset,std::max(0,int(count) * cell_width - bounds.w - UI_BORDER_WIDTH * 2)));
    previews->bounds = {bounds.x,bounds.y,bounds.w,std::max(0,bounds.h - UI_SCROLLBAR + UI_BORDER_WIDTH)};
    size_t begin = size_t(offset / cell_width);
    size_t end = std::min(count,begin + size_t(bounds.w / cell_width) + UI_CAROUSEL_EXTRA);
    if (invalid || begin != first || end != last) {
        previews->clear();
        thumbnails.clear();
        empty = nullptr;
        first = begin;
        last = end;
        invalid = false;
        if (!count) {
            empty = previews->add<PixUILabel>("No images");
            empty->color = UI_COLOR_DISABLED;
        }
        for (size_t index = first; index < last; index++) {
            PixUIThumb* preview = previews->add<PixUIThumb>();
            preview->source = [this,index]() -> PixUIThumbCache& { return thumbnail(index); };
            preview->activate = [this,index]() {
                if (!selected || selected(index)) select(int(index));
            };
            preview->placeholder = "Missing image";
            preview->caption = caption? caption(index) : "";
            thumbnails.push_back(preview);
        }
    }
    if (empty) empty->bounds = {bounds.x + UI_MARGIN,bounds.y + UI_MARGIN,bounds.w - UI_MARGIN * 2,UI_FONT_HEIGHT};
    for (size_t index = first; index < last; index++) {
        PixUIThumb* preview = thumbnails[index - first];
        preview->bounds = {bounds.x + int(index) * cell_width - offset + UI_CONTENT_INSET,bounds.y + UI_CONTENT_INSET,cell_width - UI_CONTENT_INSET * 2,previews->bounds.h - UI_CONTENT_INSET * 2};
        preview->selected = int(index) == selection;
        preview->enabled = enabled;
    }
    scrollbar->bounds = {bounds.x,bounds.y + bounds.h - UI_SCROLLBAR + UI_BORDER_WIDTH,bounds.w,UI_SCROLLBAR - UI_BORDER_WIDTH};
    scrollbar->page = bounds.w + UI_BORDER_WIDTH * 2;
    scrollbar->total = int(count) * cell_width;
    scrollbar->step = cell_width;
}

PixUIThumbCache &PixUICarousel::thumbnail(size_t index)
{
    auto found = cache.find(index);
    if (found != cache.end()) {
        found->second.used = ++clock;
        return found->second;
    }
    trim_cache(std::max(size_t(1),cache_limit) - 1);
    PixUIThumbCache value = load? load(index) : PixUIThumbCache{};
    value.missing = !value.texture;
    value.used = ++clock;
    return cache.emplace(index,value).first->second;
}

void PixUICarousel::trim_cache(size_t limit)
{
    while (cache.size() > limit) {
        auto oldest = std::min_element(cache.begin(),cache.end(),[](const auto &a, const auto &b) { return a.second.used < b.second.used; });
        if (oldest->second.texture && release) release(oldest->second.texture);
        cache.erase(oldest);
    }
}

PixUIProgress::PixUIProgress(PixUICanvas &context) : PixUILabel(context,"Idle")
{
}

void PixUIProgress::draw()
{
    needs_repaint();
    repaint = false;
    if (!visible) return;
    canvas.bevel(bounds,true);
    if (busy) {
        float complete = std::max(0.0f,std::min(1.0f,progress));
        canvas.fill({bounds.x + UI_BORDER_WIDTH,bounds.y + UI_BORDER_WIDTH,int((bounds.w - UI_BORDER_WIDTH * 2) * complete),bounds.h - UI_BORDER_WIDTH * 2},UI_COLOR_TITLE);
        canvas.text(bounds.x + UI_LABEL_PADDING,bounds.y + UI_TEXT_PADDING,std::to_string(int(complete * UI_PERCENT_SCALE)) + "%",UI_COLOR_LIGHT);
    } else canvas.text(bounds.x + UI_LABEL_PADDING,bounds.y + UI_TEXT_PADDING,caption,pixui_theme.fonts[PIXUI_FONT_LABEL].color);
}

PixUIStatusBar::PixUIStatusBar(PixUICanvas &context) : PixUIForm(context)
{
    text_clip = add<PixUIForm>();
    label = text_clip->add<PixUILabel>();
    count_clip = add<PixUIForm>();
    count_label = count_clip->add<PixUILabel>();
    timing_clip = add<PixUIForm>();
    timing_label = timing_clip->add<PixUILabel>();
    meter = add<PixUIProgress>();
}

void PixUIStatusBar::set_status(const std::string &message, const std::string &count, const std::string &timing, bool working, float fraction)
{
    if (text != message || count_text != count || timing_text != timing || busy != working || progress != fraction) repaint = true;
    text = message;
    count_text = count;
    timing_text = timing;
    busy = working;
    progress = fraction;
}

void PixUIStatusBar::layout()
{
    int top = bounds.y + UI_BORDER_WIDTH;
    int height = std::max(0,bounds.h - UI_BORDER_WIDTH * 2);
    count_clip->bounds = {bounds.x + UI_BORDER_WIDTH,top,UI_STATUS_COUNT_WIDTH,height};
    count_label->bounds = {count_clip->bounds.x + UI_TEXT_PADDING,top + UI_TEXT_PADDING,UI_STATUS_COUNT_WIDTH - UI_TEXT_PADDING * 2,UI_LINE_HEIGHT};
    count_label->caption = count_text;
    timing_clip->bounds = {count_clip->bounds.x + count_clip->bounds.w + UI_BORDER_WIDTH,top,UI_STATUS_TIME_WIDTH,height};
    timing_label->bounds = {timing_clip->bounds.x + UI_TEXT_PADDING,top + UI_TEXT_PADDING,UI_STATUS_TIME_WIDTH - UI_TEXT_PADDING * 2,UI_LINE_HEIGHT};
    timing_label->caption = timing_text;
    int left = timing_clip->bounds.x + timing_clip->bounds.w + UI_BORDER_WIDTH;
    text_clip->bounds = {left,top,std::max(0,bounds.x + bounds.w - progress_offset - UI_BORDER_WIDTH - left),height};
    label->bounds = {left + UI_TEXT_PADDING,top + UI_TEXT_PADDING,std::max(0,text_clip->bounds.w - UI_TEXT_PADDING * 2),UI_LINE_HEIGHT};
    label->caption = text;
    meter->bounds = {bounds.x + bounds.w - progress_offset,bounds.y + progress_top,progress_width,UI_FONT_HEIGHT + UI_TEXT_PADDING * 2};
    meter->busy = busy;
    meter->progress = progress;
}

void PixUIStatusBar::draw()
{
    needs_repaint();
    repaint = false;
    if (!visible) return;
    canvas.fill(bounds,UI_COLOR_FACE);
    canvas.bevel(bounds,true,false);
    layout();
    canvas.bevel(count_clip->bounds,true,false);
    canvas.bevel(timing_clip->bounds,true,false);
    canvas.bevel(text_clip->bounds,true,false);
    PixUIForm::draw();
}

PixUISplitter::PixUISplitter(PixUICanvas &context) : PixUIPanel(context)
{
    background = false;
    border = false;
}

bool PixUISplitter::event(const PixUIEvent &input)
{
    if (!visible || !enabled) return false;
    if (canvas.captured == this) {
        if (input.type == PIXUI_UP && input.button == PIXUI_LEFT_BUTTON) {
            canvas.captured = nullptr;
            return true;
        }
        if (input.type == PIXUI_MOTION) {
            if (moved) moved(input.x,input.y);
            return true;
        }
    }
    if (input.type != PIXUI_DOWN || input.button != PIXUI_LEFT_BUTTON || !pixui_inside(bounds,input.x,input.y)) return false;
    canvas.captured = this;
    return true;
}

PixUIList::PixUIList(PixUICanvas &context) : PixUIControl(context),scroll(context)
{
    scroll.parent = this;
    scroll.vertical = true;
}

void PixUIList::draw()
{
    needs_repaint();
    repaint = false;
    if (!visible || !position) return;
    canvas.bevel(bounds,true);
    PixUIRect content = {bounds.x + UI_BORDER_WIDTH,bounds.y + UI_BORDER_WIDTH,bounds.w - UI_SCROLLBAR - UI_BORDER_WIDTH * 2,bounds.h - UI_BORDER_WIDTH * 2};
    canvas.fill(content,UI_COLOR_LIGHT);
    PixUIRect previous,effective = content;
    bool clipped = canvas.clip(previous);
    if (!clipped || pixui_intersect(content,previous,effective)) {
        canvas.set_clip(&effective);
        for (int row = 0; row < page && row + *position < int(count); row++) canvas.text(bounds.x + UI_LABEL_PADDING,bounds.y + UI_TEXT_PADDING + row * UI_LINE_HEIGHT,item? item(size_t(row + *position)) : "",pixui_theme.fonts[PIXUI_FONT_LABEL].color);
        canvas.set_clip(clipped? &previous : nullptr);
    }
    layout();
    scroll.draw();
}

bool PixUIList::event(const PixUIEvent &input)
{
    if (!visible || !enabled || !position) return false;
    layout();
    if (input.type == PIXUI_WHEEL) {
        int x = input.x,y = input.y;
        if (pixui_inside(bounds,x,y)) {
            *position = std::max(0,std::min(std::max(0,int(count) - page),*position - input.wheel_y * PIXUI_WHEEL_SCROLL_Y * UI_WHEEL_LINES));
            return true;
        }
    }
    if (scroll.event(input)) return true;
    PixUIRect content = {bounds.x + UI_BORDER_WIDTH,bounds.y + UI_TEXT_PADDING,bounds.w - UI_SCROLLBAR - UI_BORDER_WIDTH * 2,page * UI_LINE_HEIGHT};
    if (input.type != PIXUI_DOWN || input.button != PIXUI_LEFT_BUTTON || !pixui_inside(content,input.x,input.y)) return false;
    size_t index = size_t((input.y - content.y) / UI_LINE_HEIGHT + *position);
    if (index < count && select) select(index);
    return true;
}

void PixUIList::layout()
{
    scroll.bounds = {bounds.x + bounds.w - UI_SCROLLBAR,bounds.y,UI_SCROLLBAR,bounds.h};
    scroll.page = page;
    scroll.total = int(count);
    scroll.position = position;
    scroll.visible = visible;
    scroll.enabled = enabled;
}

PixUIDialog::PixUIDialog(PixUICanvas &context, const std::string &caption, const std::string &accept, const std::string &cancel) : PixUIForm(context),title(caption),accept_caption(accept)
{
    border = add<PixUIPanel>();
    body = add<PixUIPanel>();
    body->border = false;
    body->color = UI_COLOR_LIGHT;
    title_label = add<PixUILabel>(caption);
    title_label->font_role = PIXUI_FONT_HEADER;
    title_label->centered = true;
    title_label->background = true;
    title_label->background_color = UI_COLOR_TITLE;
    title_label->color = pixui_theme.fonts[PIXUI_FONT_HEADER].color;
    close_button = add<PixUIButton>(UI_CONTROL_SYSTEM);
    close_button->activate = [this]() { dismiss(); };
    if (!accept.empty()) {
        accept_button = add<PixUIButton>(accept);
        accept_button->activate = [this]() { this->accept(); };
    }
    cancel_button = add<PixUIButton>(cancel);
    cancel_button->activate = [this]() { dismiss(); };
    default_button = accept_button? accept_button : cancel_button;
    default_button->default_action = true;
    resize_handle = add<PixUILabel>(UI_DIALOG_RESIZE_TEXT);
    resize_handle->centered = true;
    resize_handle->visible = false;
}

void PixUIDialog::center(const PixUIRect &client, const PixUIRect &area)
{
    available = area.w > 0 && area.h > 0? area : client;
    if (resizable) {
        dialog_width = std::min(dialog_width,available.w);
        dialog_height = std::min(dialog_height,available.h);
    }
    if (!positioned) bounds = {client.x + (client.w - dialog_width) / 2,client.y + (client.h - dialog_height) / 2,dialog_width,dialog_height};
    bounds.w = dialog_width;
    bounds.h = dialog_height;
    bounds.x = std::max(available.x,std::min(bounds.x,available.x + available.w - bounds.w));
    bounds.y = std::max(available.y,std::min(bounds.y,available.y + available.h - bounds.h));
    positioned = true;
    layout();
}

void PixUIDialog::layout()
{
    title_label->caption = title;
    if (accept_button) accept_button->caption = accept_caption;
    border->bounds = bounds;
    body->bounds = {bounds.x + UI_FRAME,bounds.y + UI_TITLE_HEIGHT + UI_FRAME,bounds.w - UI_FRAME * 2,bounds.h - UI_TITLE_HEIGHT - UI_FRAME * 2};
    title_label->bounds = {bounds.x + UI_CONTENT_INSET,bounds.y + UI_CONTENT_INSET,bounds.w - UI_CONTENT_INSET * 2,UI_TITLE_HEIGHT};
    close_button->bounds = {bounds.x + UI_CONTENT_INSET,bounds.y + UI_CONTENT_INSET,UI_TITLE_HEIGHT,UI_TITLE_HEIGHT};
    PixUIDialogLayout positions = controls();
    cancel_button->bounds = positions.cancel;
    if (accept_button) accept_button->bounds = positions.ok;
    resize_handle->visible = resizable;
    resize_handle->bounds = {bounds.x + bounds.w - UI_SCROLLBAR,bounds.y + bounds.h - UI_SCROLLBAR,UI_SCROLLBAR,UI_SCROLLBAR};
}

void PixUIDialog::dismiss()
{
    canvas.release_capture();
    visible = false;
    if (parent) parent->repaint = true;
}

void PixUIDialog::accept()
{
    if (accepted) accepted();
}

bool PixUIDialog::event(const PixUIEvent &input)
{
    if (!visible) return false;
    // A nested selector owns Escape and pointer input until it closes.
    if (modal()) {
        PixUIForm::event(input);
        return true;
    }
    if (input.type == PIXUI_KEY && input.key == PIXUI_ESCAPE) {
        dismiss();
        return true;
    }
    if (moving && canvas.captured == this) {
        switch (input.type) {
        case PIXUI_UP:
            if (input.button == PIXUI_LEFT_BUTTON) canvas.release_capture();
            break;
        case PIXUI_MOTION:
            bounds.x = std::max(available.x,std::min(available.x + available.w - bounds.w,bounds.x + input.x - move_x));
            bounds.y = std::max(available.y,std::min(available.y + available.h - bounds.h,bounds.y + input.y - move_y));
            move_x = input.x;
            move_y = input.y;
            layout();
            repaint = true;
            break;
        default:
            break;
        }
        return true;
    }
    if (resizing && canvas.captured == this) {
        switch (input.type) {
        case PIXUI_UP:
            if (input.button == PIXUI_LEFT_BUTTON) canvas.release_capture();
            break;
        case PIXUI_MOTION: {
            int max_width = std::max(0,available.x + available.w - bounds.x);
            int max_height = std::max(0,available.y + available.h - bounds.y);
            dialog_width = std::min(max_width,std::max(minimum_width,resize_width + input.x - resize_x));
            dialog_height = std::min(max_height,std::max(minimum_height,resize_height + input.y - resize_y));
            bounds.w = dialog_width;
            bounds.h = dialog_height;
            repaint = true;
            layout();
            break;
        }
        default:
            break;
        }
        return true;
    }
    if (resizable && input.type == PIXUI_DOWN && input.button == PIXUI_LEFT_BUTTON && pixui_inside(resize_handle->bounds,input.x,input.y)) {
        resizing = true;
        resize_x = input.x;
        resize_y = input.y;
        resize_width = bounds.w;
        resize_height = bounds.h;
        canvas.captured = this;
        return true;
    }
    if (input.type == PIXUI_DOWN && input.button == PIXUI_LEFT_BUTTON && pixui_inside(title_label->bounds,input.x,input.y) && !pixui_inside(close_button->bounds,input.x,input.y)) {
        moving = true;
        move_x = input.x;
        move_y = input.y;
        canvas.captured = this;
        return true;
    }
    PixUIForm::event(input);
    return true;
}

void PixUIDialog::cancel_capture()
{
    resizing = false;
    moving = false;
    PixUIForm::cancel_capture();
}

PixUIDialogLayout PixUIDialog::controls() const
{
    PixUIDialogLayout result = {};
    result.box = bounds;
    PixUILayout content = {bounds.x + UI_MARGIN,bounds.y + UI_TITLE_HEIGHT + UI_MARGIN,UI_MARGIN};
    result.label = content.column(bounds.w - UI_MARGIN * 2,UI_LINE_HEIGHT);
    result.path = content.column(result.label.w,UI_FIELD_HEIGHT);
    PixUILayout browse = {content.x,content.y,UI_MARGIN};
    result.browse = browse.row(UI_DIALOG_BROWSE_WIDTH,UI_FIELD_HEIGHT);
    result.up = browse.row(UI_DIALOG_BROWSE_WIDTH,UI_FIELD_HEIGHT);
    content.column(result.label.w,UI_FIELD_HEIGHT);
    result.list = content.column(result.label.w,UI_DIRECTORY_ROWS * UI_LINE_HEIGHT + UI_MARGIN);
    result.hint = content.column(result.label.w,UI_LINE_HEIGHT);
    int right_margin = UI_MARGIN + (resizable? UI_SCROLLBAR : 0);
    PixUILayout buttons = {bounds.x + bounds.w - right_margin - UI_DIALOG_BUTTON_WIDTH * 2 - UI_MARGIN,bounds.y + bounds.h - UI_MARGIN - UI_FIELD_HEIGHT,UI_MARGIN};
    result.ok = buttons.row(UI_DIALOG_BUTTON_WIDTH,UI_FIELD_HEIGHT);
    result.cancel = buttons.row(UI_DIALOG_BUTTON_WIDTH,UI_FIELD_HEIGHT);
    return result;
}

PixUIMessageDialog::PixUIMessageDialog(PixUICanvas &context, const std::string &text, const std::string &caption, const std::string &accept, const std::string &cancel) : PixUIDialog(context,caption,accept,cancel)
{
    message = add<PixUILabel>(text);
    message->wrap = true;
}

void PixUIMessageDialog::layout()
{
    int longest = 0,column = 0,title_columns = 0;
    for (size_t offset = 0; offset < message->caption.size(); offset = pixui_next_character(message->caption,offset)) {
        if (message->caption[offset] == '\n') column = 0;
        else column++;
        longest = std::max(longest,column);
    }
    for (size_t offset = 0; offset < title.size(); offset = pixui_next_character(title,offset)) title_columns++;
    int cancel_columns = 0,accept_columns = 0;
    for (size_t offset = 0; offset < cancel_button->caption.size(); offset = pixui_next_character(cancel_button->caption,offset)) cancel_columns++;
    for (size_t offset = 0; offset < accept_caption.size(); offset = pixui_next_character(accept_caption,offset)) accept_columns++;
    int cancel_width = std::max(UI_DIALOG_BUTTON_WIDTH,cancel_columns * UI_FONT_WIDTH + UI_TEXT_PADDING * 2);
    int accept_width = std::max(UI_DIALOG_BUTTON_WIDTH,accept_columns * UI_FONT_WIDTH + UI_TEXT_PADDING * 2);
    int button_width = cancel_width + (accept_button? accept_width + UI_MARGIN : 0);
    int title_width = title_columns * pixui_theme.fonts[PIXUI_FONT_HEADER].width + (UI_TITLE_HEIGHT + UI_CONTENT_INSET) * 2;
    dialog_width = std::min(UI_DIALOG_WIDTH,std::max(std::max(longest * UI_FONT_WIDTH + UI_MARGIN * 2,title_width),button_width + UI_MARGIN * 2));
    std::vector<size_t> lines = pixui_editor_lines(message->caption,std::max(1,(dialog_width - UI_MARGIN * 2) / UI_FONT_WIDTH));
    // The editor adds a caret row after an exactly filled line; message labels need only the text rows.
    if (lines.size() > 1 && lines.back() == message->caption.size() && message->caption.back() != '\n') lines.pop_back();
    dialog_height = UI_TITLE_HEIGHT + UI_MARGIN * 3 + int(lines.size()) * UI_LINE_HEIGHT + UI_FIELD_HEIGHT;
    bounds.x += (bounds.w - dialog_width) / 2;
    bounds.y += (bounds.h - dialog_height) / 2;
    bounds.w = dialog_width;
    bounds.h = dialog_height;
    PixUIDialog::layout();
    PixUIDialogLayout positions = controls();
    message->bounds = {positions.label.x,positions.label.y,positions.label.w,positions.cancel.y - positions.label.y - UI_MARGIN};
    cancel_button->bounds.x = bounds.x + bounds.w - UI_MARGIN - cancel_width;
    cancel_button->bounds.w = cancel_width;
    if (accept_button) {
        accept_button->bounds.x = cancel_button->bounds.x - UI_MARGIN - accept_width;
        accept_button->bounds.w = accept_width;
    }
}

PixUIAboutDialog::PixUIAboutDialog(PixUICanvas &context, const std::string &caption, void* texture, int width, int height, const std::vector<std::string> &text) : PixUIDialog(context,caption,"",UI_CLOSE_TEXT)
{
    dialog_width = UI_ABOUT_WIDTH;
    dialog_height = UI_ABOUT_HEIGHT;
    icon = add<PixUIImage>();
    icon->texture = texture;
    icon->width = width;
    icon->height = height;
    for (const auto &line : text) lines.push_back(add<PixUILabel>(line));
    if (!lines.empty()) lines[0]->color = UI_COLOR_TITLE;
}

void PixUIAboutDialog::layout()
{
    PixUIDialog::layout();
    PixUILayout content = {bounds.x + UI_ABOUT_SPACING,bounds.y + UI_TITLE_HEIGHT + UI_ABOUT_SPACING,0};
    icon->bounds = content.row(UI_ABOUT_ICON_SIZE,UI_ABOUT_ICON_SIZE);
    content.x += UI_ABOUT_SPACING;
    int width = bounds.w - (content.x - bounds.x) - UI_ABOUT_SPACING;
    for (size_t i = 0; i < lines.size(); i++) {
        if (i == lines.size() / 2) content.y += UI_MARGIN;
        lines[i]->bounds = content.column(width,UI_LINE_HEIGHT);
    }
}

std::string PixUIFileBrowser::last_directory;

PixUIFileBrowser::PixUIFileBrowser(PixUICanvas &context, const std::string &initial) : PixUIDialog(context,"Open file","Open")
{
    set_path(initial.empty()? last_directory : initial);
    label = add<PixUILabel>();
    edit = add<PixUIEdit>(path);
    register_focus(*edit);
    hint = add<PixUILabel>();
    browse = add<PixUIButton>("Browse");
    browse->activate = [this]() { refresh(); };
    up = add<PixUIButton>("Up");
    up->activate = [this]() {
        set_path(path_join(directory,".."));
        refresh();
    };
    list = add<PixUIList>();
    list->position = &position;
    list->page = UI_DIRECTORY_ROWS;
    list->item = [this](size_t index) { return (entries[index].directory? "[DIR] " : "      ") + entries[index].name; };
    list->select = [this](size_t index) { select(index); };
}

void PixUIFileBrowser::refresh()
{
    directory = path.text;
    if (path_kind(directory) != PIX_PATH_DIRECTORY) directory = path_parent(directory);
    std::string message;
    if (!directory_list(directory,entries,message)) {
        entries.clear();
        if (error) error(message);
    } else last_directory = path_normalize(directory);
    position = 0;
}

void PixUIFileBrowser::accept()
{
    if (path.text.empty()) return;
    PixPathKind kind = path_kind(path.text);
    if (!dir_selector && kind == PIX_PATH_DIRECTORY) {
        refresh();
        return;
    }
    if (dir_selector && kind != PIX_PATH_DIRECTORY) {
        if (error) error(UI_DIRECTORY_ERROR + path.text);
        return;
    }
    if (kind == PIX_PATH_OTHER) return;
    if (selected && selected(path.text)) {
        std::string parent = kind == PIX_PATH_DIRECTORY? path.text : path_parent(path.text);
        if (path_kind(parent) == PIX_PATH_DIRECTORY) last_directory = path_normalize(parent);
        dismiss();
    }
}

void PixUIFileBrowser::set_path(const std::string &value)
{
    path.text = value;
    path.cursor = value.size();
    path.anchor = path.cursor;
    path.horizontal = 0;
    path.reveal = true;
}

void PixUIFileBrowser::layout()
{
    PixUIDialog::layout();
    PixUIDialogLayout positions = controls();
    label->caption = path_label.empty()? (dir_selector? "Directory:" : "File name:") : path_label;
    label->bounds = positions.label;
    edit->bounds = positions.path;
    hint->caption = hint_text.empty()? (dir_selector? "Click a directory to enter it; Open selects it." : "Select a file, then choose Open.") : hint_text;
    browse->visible = show_listing;
    up->visible = show_listing;
    list->visible = show_listing;
    if (!show_listing) {
        hint->bounds = {positions.browse.x,positions.browse.y,positions.browse.w + positions.up.w + UI_MARGIN,UI_LINE_HEIGHT};
        return;
    }
    browse->bounds = positions.browse;
    up->bounds = positions.up;
    list->bounds = positions.list;
    list->count = entries.size();
    list->layout();
    hint->bounds = positions.hint;
}

void PixUIFileBrowser::select(size_t index)
{
    if (index >= entries.size() || (!entries[index].directory && dir_selector)) return;
    bool directory_entry = entries[index].directory;
    std::string value = path_join(directory,entries[index].name);
    value = path_normalize(value);
    set_path(value);
    if (directory_entry) refresh();
}

bool pixui_intersect(const PixUIRect &a, const PixUIRect &b, PixUIRect &result)
{
    result.x = std::max(a.x,b.x);
    result.y = std::max(a.y,b.y);
    result.w = std::max(0,std::min(a.x + a.w,b.x + b.w) - result.x);
    result.h = std::max(0,std::min(a.y + a.h,b.y + b.h) - result.y);
    return result.w > 0 && result.h > 0;
}

bool pixui_inside(const PixUIRect &rect, int x, int y)
{
    return x >= rect.x && y >= rect.y && x < rect.x + rect.w && y < rect.y + rect.h;
}

int pixui_scrollbar_position(int coordinate, int origin, int length, int page, int total)
{
    int track = std::max(1,length - UI_SCROLLBAR * 2);
    int thumb = std::min(track,std::max(UI_SCROLLBAR,track * page / std::max(page,total)));
    int offset = coordinate - origin - UI_SCROLLBAR - thumb / 2;
    return std::max(0,std::min(std::max(0,total - page),std::max(0,total - page) * offset / std::max(1,track - thumb)));
}

size_t pixui_next_character(const std::string &text, size_t offset)
{
    if (offset >= text.size()) return text.size();
    offset++;
    while (offset < text.size() && (static_cast<unsigned char>(text[offset]) & UI_UTF8_MASK) == UI_UTF8_CONTINUATION) offset++;
    return offset;
}

// Store byte offsets so wrapping never changes the prompt or splits UTF-8 characters.
std::vector<size_t> pixui_editor_lines(const std::string &value, int columns)
{
    std::vector<size_t> starts = {0};
    size_t offset = 0,start = 0,space = 0;
    int column = 0;
    while (offset < value.size()) {
        if (value[offset] == '\n') {
            offset++;
            starts.push_back(offset);
            start = offset;
            column = 0;
            space = start;
            continue;
        }
        if (column == columns) {
            offset = space > start? space : offset;
            starts.push_back(offset);
            start = offset;
            column = 0;
            space = start;
            continue;
        }
        if (value[offset] == ' ' || value[offset] == '\t') space = pixui_next_character(value,offset);
        offset = pixui_next_character(value,offset);
        column++;
    }
    if (column == columns && start < value.size()) starts.push_back(value.size());
    return starts;
}
