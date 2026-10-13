/* This file is a part of 'Pixelina' Image Generator, Editor, and Viewer.
 *
 * (C) Dmitry 'sciloaf' Solovyev, 2026
 *
 * License: MIT
 */
#include "pixuilib.h"
#include "../third_party/font8x8_basic.h"
#include <algorithm>

static size_t previous_character(const std::string &text, size_t offset)
{
    if (!offset) return 0;
    offset--;
    while (offset && (static_cast<unsigned char>(text[offset]) & UI_UTF8_MASK) == UI_UTF8_CONTINUATION) offset--;
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
    if (background) fill(rect,COLOR_FACE);
    unsigned upper = inset? COLOR_DARK : COLOR_LIGHT;
    unsigned lower = inset? COLOR_SHADOW : COLOR_DARK;
    fill({rect.x,rect.y,rect.w,1},upper);
    fill({rect.x,rect.y,1,rect.h},upper);
    fill({rect.x,rect.y + rect.h - 1,rect.w,1},lower);
    fill({rect.x + rect.w - 1,rect.y,1,rect.h},lower);
}

void PixUICanvas::text(int x, int y, const std::string &value, unsigned color)
{
    int origin = x;
    for (size_t i = 0; i < value.size(); i = pixui_next_character(value,i)) {
        unsigned char ch = value[i];
        if (ch == '\n') {
            y += UI_LINE_HEIGHT;
            x = origin;
            continue;
        }
        if (ch >= UI_FONT_GLYPHS || ch < ' ') ch = '?';
        for (int row = 0; row < UI_FONT_HEIGHT; row++) {
            unsigned char bits = font8x8_basic[ch][row];
            for (int column = 0; column < UI_FONT_WIDTH; column++) {
                if (bits & (1 << column)) point(x + column,y + row,color);
            }
        }
        x += UI_FONT_WIDTH;
    }
}

void PixUICanvas::button(const PixUIRect &rect, const std::string &label, bool pressed)
{
    fill(rect,COLOR_DARK);
    PixUIRect inner = {rect.x + 1,rect.y + 1,rect.w - 2,rect.h - 2};
    bevel(inner,pressed);
    if (!pressed) {
        fill({inner.x + 1,inner.y + inner.h - 2,inner.w - 2,1},COLOR_SHADOW);
        fill({inner.x + inner.w - 2,inner.y + 1,1,inner.h - 2},COLOR_SHADOW);
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
            if (horizontal) fill({top + row,center - radius,1,radius * 2 + 1},COLOR_DARK);
            else fill({center - radius,top + row,radius * 2 + 1,1},COLOR_DARK);
        }
    } else if (label == UI_CONTROL_SYSTEM) {
        PixUIRect dash = {rect.x + UI_ARROW_SIZE,rect.y + rect.h / 2 - 1,rect.w - UI_ARROW_SIZE * 2,UI_SYSTEM_BAR_HEIGHT};
        fill(dash,COLOR_DARK);
        fill({dash.x + 1,dash.y + 1,dash.w - 2,1},COLOR_LIGHT);
    } else text(rect.x + std::max(UI_TEXT_PADDING,(rect.w - int(label.size()) * UI_FONT_WIDTH) / 2),rect.y + (rect.h - UI_FONT_HEIGHT) / 2,label,COLOR_DARK);
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
            if (input.type == PIXUI_UP && input.button == PIXUI_LEFT_BUTTON) captured = nullptr;
            return handled;
        }
    }
    return form.event(input);
}

void PixUICanvas::release_capture()
{
    if (!captured) return;
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
    if (!visible) return;
    if (background) canvas.fill(bounds,color);
    if (border) canvas.bevel(bounds,inset,false);
}

PixUILabel::PixUILabel(PixUICanvas &context, const std::string &text) : PixUIControl(context),caption(text)
{
}

void PixUILabel::draw()
{
    if (!visible) return;
    if (background) canvas.fill(bounds,background_color);
    std::string text = caption;
    if (wrap) {
        int columns = std::max(1,bounds.w / UI_FONT_WIDTH);
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
        x += std::max(0,(bounds.w - int(text.size()) * UI_FONT_WIDTH) / 2);
        y += std::max(0,(bounds.h - UI_FONT_HEIGHT) / 2);
    }
    canvas.text(x,y,text,color);
    if (underline) canvas.fill({x,y + UI_FONT_HEIGHT,UI_FONT_WIDTH,1},color);
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
    if (!visible) return;
    PixUIRect previous,effective = bounds;
    bool clipped = canvas.clip(previous);
    if (clipped && !pixui_intersect(bounds,previous,effective)) return;
    canvas.set_clip(&effective);
    if (texture && width > 0 && height > 0) {
        PixUIRect target = {bounds.x - offset_x,bounds.y - offset_y,width,height};
        if (scaled) {
            float scale = std::min(float(bounds.w) / width,float(bounds.h) / height);
            target.w = int(width * scale);
            target.h = int(height * scale);
            target.x = bounds.x + (bounds.w - target.w) / 2;
            target.y = bounds.y + (bounds.h - target.h) / 2;
        }
        canvas.image(texture,target);
    } else if (!placeholder.empty()) canvas.text(bounds.x + UI_LABEL_PADDING,bounds.y + UI_LABEL_PADDING,placeholder,COLOR_DISABLED);
    canvas.set_clip(clipped? &previous : nullptr);
}

bool PixUIImage::event(const PixUIEvent &input)
{
    if (visible && enabled && input.type == PIXUI_WHEEL && pixui_inside(bounds,input.x,input.y) && !scaled) {
        if (input.mod & PIXUI_SHIFT) offset_x -= input.wheel_y * wheel_step;
        else offset_y -= input.wheel_y * wheel_step;
        offset_x -= input.wheel_x * wheel_step;
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
    if (!visible) return;
    if (source) {
        PixUIThumbCache &preview = source();
        texture = preview.texture;
        width = preview.width;
        height = preview.height;
    }
    PixUIRect cell = bounds;
    canvas.fill(cell,selected? COLOR_SELECTED : COLOR_LIGHT);
    bounds = {cell.x + UI_TEXT_PADDING,cell.y + UI_CONTENT_INSET,cell.w - UI_TEXT_PADDING * 2,cell.h - UI_LINE_HEIGHT - UI_CONTENT_INSET};
    PixUIImage::draw();
    bounds = cell;
    canvas.text(cell.x + UI_CONTENT_INSET,cell.y + cell.h - UI_LINE_HEIGHT,caption,selected? COLOR_LIGHT : COLOR_DARK);
}

PixUIScrollbar::PixUIScrollbar(PixUICanvas &context) : PixUIControl(context)
{
}

void PixUIScrollbar::draw()
{
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
    model.text = text;
    model.cursor = text.size();
    model.anchor = model.cursor;
    model.scroll = 0;
    model.horizontal = 0;
    model.reveal = true;
}

void PixUIMemo::draw()
{
    if (!visible) return;
    draw_editor();
    layout();
    if (multiline) scroll.draw();
}

bool PixUIMemo::event(const PixUIEvent &input)
{
    if (!visible || !enabled) return false;
    layout();
    if (multiline && input.type == PIXUI_WHEEL) {
        int x = input.x,y = input.y;
        if (pixui_inside(bounds,x,y)) {
            model.scroll = std::max(0,std::min(std::max(0,scroll.total - scroll.page),model.scroll - input.wheel_y * UI_WHEEL_LINES));
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
    scroll.page = std::max(1,(bounds.h - UI_CONTENT_INSET * 2) / UI_LINE_HEIGHT);
    scroll.total = int(pixui_editor_lines(model.text,std::max(1,(bounds.w - UI_CONTENT_INSET * 2 - UI_SCROLLBAR) / UI_FONT_WIDTH)).size());
    scroll.visible = visible && multiline;
    scroll.enabled = enabled;
}

void PixUIMemo::edit(const PixUIEvent &event)
{
    PixUIEditor &editor = model;
    const PixUIRect &area = bounds;
    if (event.type != PIXUI_KEY && event.type != PIXUI_TEXT) return;
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
                target = !shift && low != high? low : previous_character(editor.text,target);
                break;
            case PIXUI_RIGHT:
                target = !shift && low != high? high : pixui_next_character(editor.text,target);
                break;
            default:
                if (multiline && !control) {
                    int columns = std::max(1,(area.w - UI_CONTENT_INSET * 2 - UI_SCROLLBAR) / UI_FONT_WIDTH);
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
    bool focused = active && interactive() && (!focus_owner || !focus_owner->modal_blocked);
    canvas.bevel(rect,true);
    PixUIRect content = {rect.x + UI_CONTENT_INSET,rect.y + UI_CONTENT_INSET,rect.w - UI_CONTENT_INSET * 2 - (multiline? UI_SCROLLBAR : 0),rect.h - UI_CONTENT_INSET * 2};
    int columns = std::max(1,content.w / UI_FONT_WIDTH);
    std::vector<size_t> lines = multiline? pixui_editor_lines(editor.text,columns) : std::vector<size_t>{0};
    int caret_line = int(std::upper_bound(lines.begin(),lines.end(),editor.cursor) - lines.begin()) - 1;
    int caret_column = 0;
    for (size_t offset = lines[caret_line]; offset < editor.cursor; offset = pixui_next_character(editor.text,offset)) caret_column++;
    int visible = std::max(1,content.h / UI_LINE_HEIGHT);
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
        PixUIRect caret = {content.x + (caret_column - editor.horizontal) * UI_FONT_WIDTH,content.y + (caret_line - editor.scroll) * UI_LINE_HEIGHT,UI_FONT_WIDTH,UI_LINE_HEIGHT};
        canvas.text_input(caret);
    }
    canvas.fill(content,COLOR_LIGHT);
    PixUIRect previous,effective = content;
    bool clipped = canvas.clip(previous);
    if (clipped) pixui_intersect(content,previous,effective);
    canvas.set_clip(&effective);
    for (int line = editor.scroll; line < int(lines.size()) && line <= editor.scroll + visible; line++) {
        size_t end = line + 1 < int(lines.size())? lines[line + 1] : editor.text.size();
        int column = 0;
        for (size_t offset = lines[line]; offset < end; offset = pixui_next_character(editor.text,offset)) {
            if (editor.text[offset] == '\n') break;
            int x = content.x + (column - editor.horizontal) * UI_FONT_WIDTH;
            int y = content.y + (line - editor.scroll) * UI_LINE_HEIGHT;
            bool selection = offset >= std::min(editor.cursor,editor.anchor) && offset < std::max(editor.cursor,editor.anchor);
            if (selection) canvas.fill({x,y,UI_FONT_WIDTH,UI_LINE_HEIGHT},COLOR_SELECTED);
            canvas.text(x,y + UI_CONTENT_INSET,editor.text.substr(offset,pixui_next_character(editor.text,offset) - offset),selection? COLOR_LIGHT : COLOR_DARK);
            column++;
        }
    }
    if (focused && (canvas.ticks() / UI_CARET_PERIOD) % 2 == 0) canvas.fill({content.x + (caret_column - editor.horizontal) * UI_FONT_WIDTH,content.y + (caret_line - editor.scroll) * UI_LINE_HEIGHT,1,UI_LINE_HEIGHT},COLOR_DARK);
    canvas.set_clip(clipped? &previous : nullptr);

}

void PixUIMemo::place_cursor(int x, int y, int modifiers)
{
    PixUIEditor &editor = model;
    const PixUIRect &rect = bounds;
    int columns = std::max(1,(rect.w - UI_CONTENT_INSET * 2 - (multiline? UI_SCROLLBAR : 0)) / UI_FONT_WIDTH);
    std::vector<size_t> lines = multiline? pixui_editor_lines(editor.text,columns) : std::vector<size_t>{0};
    int wanted_line = multiline? std::max(0,(y - rect.y - UI_CONTENT_INSET) / UI_LINE_HEIGHT + editor.scroll) : 0;
    wanted_line = std::min(wanted_line,int(lines.size()) - 1);
    int wanted_column = std::max(0,(x - rect.x - UI_CONTENT_INSET + UI_FONT_WIDTH / 2) / UI_FONT_WIDTH + editor.horizontal);
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
    if (checked_state) checked = checked_state();
    if (caption_state) caption = caption_state();
}

void PixUIMenuItem::draw()
{
    if (!visible) return;
    bool highlight = enabled && highlighted;
    if (highlight) canvas.fill(bounds,COLOR_SELECTED);
    unsigned ink = !enabled? COLOR_DISABLED : (highlight? COLOR_LIGHT : COLOR_DARK);
    int x = bounds.x + UI_MENU_PADDING,y = bounds.y + (bounds.h - UI_FONT_HEIGHT) / 2;
    if (checkable) {
        if (checked) canvas.text(x,y,UI_CHECK_MARK,ink);
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
    if (toggle) return;
    selected = int(index);
    items[index]->highlighted = true;
    items[index]->submenu->visible = true;
    layout();
}

void PixUIMenu::dismiss()
{
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

void PixUIMenu::draw()
{
    if (!visible) return;
    layout();
    if (!bar) {
        canvas.fill(bounds,COLOR_LIGHT);
        canvas.bevel(bounds,false,false);
    }
    for (const auto &item : items) {
        if (!bar) item->draw();
        else if (item->show_title) {
            canvas.fill(item->bounds,item->highlighted? COLOR_SELECTED : COLOR_LIGHT);
            unsigned ink = item->highlighted? COLOR_LIGHT : COLOR_DARK;
            int x = item->bounds.x + std::max(0,(item->bounds.w - int(item->caption.size()) * UI_FONT_WIDTH) / 2);
            int y = item->bounds.y + (item->bounds.h - UI_FONT_HEIGHT) / 2;
            canvas.text(x,y,item->caption,ink);
            canvas.fill({x,y + UI_FONT_HEIGHT,UI_FONT_WIDTH,1},ink);
        }
    }
    for (const auto &item : items) {
        if (item->submenu && item->submenu->visible) item->submenu->draw();
    }
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
            if (!bar) item.highlighted = item.enabled;
            return !bar;
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
    int extent = 0;
    for (const auto &item : items) {
        item->update();
        int length = int(item->caption.size()) * UI_FONT_WIDTH;
        if (!item->shortcut.empty()) length += UI_MENU_GAP + int(item->shortcut.size()) * UI_FONT_WIDTH;
        if (item->checkable) length += UI_MENU_CHECK_WIDTH;
        extent = std::max(extent,length);
    }
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
    popup->add("Move",[this]() { command(PIXUI_MOVE); });
    popup->add("Size",[this]() { command(PIXUI_RESIZE); });
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
    title->color = modal_blocked || !active? COLOR_LIGHT : COLOR_TITLE;
    title_clip->bounds = {UI_FRAME + UI_WINDOW_TITLE_HEIGHT,UI_FRAME,bounds.w - UI_FRAME * 2 - UI_WINDOW_TITLE_HEIGHT * 3,UI_WINDOW_TITLE_HEIGHT};
    title_label->bounds = title_clip->bounds;
    title_label->caption = caption;
    title_label->color = modal_blocked || !active? COLOR_DARK : COLOR_LIGHT;
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
    if (value == PIXUI_MOVE || value == PIXUI_RESIZE) {
        window_drag = value;
        drag_x = canvas.pointer_x;
        drag_y = canvas.pointer_y;
        drag_window = canvas.window_rect();
        resize_edge = PIXUI_HIT_BOTTOMRIGHT;
        if (value == PIXUI_RESIZE) {
            drag_x = drag_window.w - UI_FRAME;
            drag_y = drag_window.h - UI_FRAME;
        }
        canvas.captured = this;
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
    else command(PIXUI_MOVE);
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
    if (focused) focused->active = false;
    focused = &memo;
    focused->active = true;
    focused->model.reveal = true;
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
    pending_dialog = std::move(dialog);
    modal_blocked = true;
}

void PixUIForm::dismiss_dialog()
{
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
        if (element->visible && element->layout_visible && element.get() != menu_bar) element->draw();
    }
    if (menu_bar) menu_bar->draw();
    canvas.set_clip(clipped? &previous : nullptr);
    if (modal_dialog && modal_dialog->visible) modal_dialog->draw();
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
            PixUIRect rect = canvas.window_rect();
            if (window_drag == PIXUI_MOVE) {
                rect.x += input.x - drag_x;
                rect.y += input.y - drag_y;
                canvas.window_command(PIXUI_MOVE,rect);
            } else {
                int dx = rect.x + input.x - drag_window.x - drag_x;
                int dy = rect.y + input.y - drag_window.y - drag_y;
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
            command(PIXUI_RESIZE);
            resize_edge = hit;
            drag_x = input.x;
            drag_y = input.y;
            return true;
        }
    }
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
    if (default_button && input.type == PIXUI_KEY && input.key == PIXUI_RETURN) return default_button->event(input);
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
        if (element->get() != menu_bar && (*element)->visible && (*element)->layout_visible && (*element)->enabled && (*element)->event(input)) return true;
    }
    return PixUIControl::event(input);
}

void PixUIForm::prepare_modal()
{
    if (pending_dialog) modal_dialog = std::move(pending_dialog);
    if (modal_dialog && !modal_dialog->visible) {
        modal_dialog.reset();
        if (saved_focus) focus_control(*saved_focus);
        saved_focus = nullptr;
    }
    modal_blocked = modal_dialog != nullptr;
    if (modal_dialog) modal_dialog->center(bounds);
}

PixUIGroupBox::PixUIGroupBox(PixUICanvas &context, const std::string &text) : PixUIForm(context),caption(text)
{
    clip_client = true;
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
    layout();
}

void PixUIGroupBox::clear()
{
    placements.clear();
    scroll = 0;
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
    scroll = std::max(0,std::min(scroll,std::max(0,content_height - client.h)));
    for (const PixUIPlacement &placement : placements) {
        PixUIRect rect = placement.rect;
        if (placement.flags & PIXUI_STRETCH_X) rect.w += client.w;
        if (placement.flags & PIXUI_STRETCH_Y) rect.h += client.h;
        if (placement.flags & PIXUI_ALIGN_RIGHT) rect.x += client.w;
        if (placement.flags & PIXUI_ALIGN_BOTTOM) rect.y += client.h;
        rect.x += client.x;
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
    if (!visible) return;
    canvas.fill(bounds,color);
    if (border) canvas.bevel(bounds,true,false);
    if (!caption.empty()) canvas.text(bounds.x + UI_MARGIN,bounds.y + UI_MARGIN,caption,COLOR_DARK);
    PixUIForm::draw();
}

bool PixUIGroupBox::event(const PixUIEvent &input)
{
    if (!visible || !enabled) return false;
    if (PixUIForm::event(input)) return true;
    if (content_height && input.type == PIXUI_WHEEL && pixui_inside(client,input.x,input.y)) {
        scroll -= input.wheel_y * wheel_step;
        layout();
        return true;
    }
    return false;
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

PixUICarousel::PixUICarousel(PixUICanvas &context) : PixUIForm(context)
{
    previews = add<PixUIForm>();
    scrollbar = add<PixUIScrollbar>();
    scrollbar->position = &offset;
    scrollbar->step = cell_width;
    previews->wheel = [this](const PixUIEvent &input) {
        offset -= (input.wheel_y + input.wheel_x) * cell_width;
        layout();
    };
}

PixUICarousel::~PixUICarousel()
{
    for (const auto &entry : cache) {
        if (entry.second.texture && release) release(entry.second.texture);
    }
}

void PixUICarousel::items_changed(size_t total, bool reset)
{
    // Releasing before replacement prevents stale resources after workdir changes.
    for (const auto &entry : cache) {
        if (entry.second.texture && release) release(entry.second.texture);
    }
    cache.clear();
    count = total;
    invalid = true;
    if (reset) {
        offset = 0;
        selection = -1;
    } else if (selection >= int(count)) selection = -1;
}

void PixUICarousel::select(int index)
{
    selection = index >= 0 && size_t(index) < count? index : -1;
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
            empty = previews->add<PixUILabel>("No saved images");
            empty->color = COLOR_DISABLED;
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
    if (!visible) return;
    canvas.bevel(bounds,true);
    if (busy) {
        float complete = std::max(0.0f,std::min(1.0f,progress));
        canvas.fill({bounds.x + UI_BORDER_WIDTH,bounds.y + UI_BORDER_WIDTH,int((bounds.w - UI_BORDER_WIDTH * 2) * complete),bounds.h - UI_BORDER_WIDTH * 2},COLOR_TITLE);
        canvas.text(bounds.x + UI_LABEL_PADDING,bounds.y + UI_TEXT_PADDING,std::to_string(int(complete * UI_PERCENT_SCALE)) + "%",COLOR_LIGHT);
    } else canvas.text(bounds.x + UI_LABEL_PADDING,bounds.y + UI_TEXT_PADDING,caption,COLOR_DARK);
}

PixUIStatusBar::PixUIStatusBar(PixUICanvas &context) : PixUIForm(context)
{
    text_clip = add<PixUIForm>();
    label = text_clip->add<PixUILabel>();
    meter = add<PixUIProgress>();
}

void PixUIStatusBar::layout()
{
    text_clip->bounds = {bounds.x + margin,bounds.y + margin,bounds.w - text_space,UI_LINE_HEIGHT};
    label->bounds = text_clip->bounds;
    label->caption = text;
    meter->bounds = {bounds.x + bounds.w - progress_offset,bounds.y + progress_top,progress_width,UI_SCROLLBAR};
    meter->busy = busy;
    meter->progress = progress;
}

void PixUIStatusBar::draw()
{
    if (!visible) return;
    canvas.fill(bounds,COLOR_FACE);
    canvas.bevel(bounds,true,false);
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
    if (!visible || !position) return;
    canvas.bevel(bounds,true);
    PixUIRect content = {bounds.x + UI_BORDER_WIDTH,bounds.y + UI_BORDER_WIDTH,bounds.w - UI_SCROLLBAR - UI_BORDER_WIDTH * 2,bounds.h - UI_BORDER_WIDTH * 2};
    canvas.fill(content,COLOR_LIGHT);
    PixUIRect previous,effective = content;
    bool clipped = canvas.clip(previous);
    if (!clipped || pixui_intersect(content,previous,effective)) {
        canvas.set_clip(&effective);
        for (int row = 0; row < page && row + *position < int(count); row++) canvas.text(bounds.x + UI_LABEL_PADDING,bounds.y + UI_TEXT_PADDING + row * UI_LINE_HEIGHT,item? item(size_t(row + *position)) : "",COLOR_DARK);
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
            *position = std::max(0,std::min(std::max(0,int(count) - page),*position - input.wheel_y * UI_WHEEL_LINES));
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
    body->color = COLOR_LIGHT;
    title_label = add<PixUILabel>(caption);
    title_label->centered = true;
    title_label->background = true;
    title_label->background_color = COLOR_TITLE;
    title_label->color = COLOR_LIGHT;
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
}

void PixUIDialog::center(const PixUIRect &client)
{
    bounds = {client.x + (client.w - dialog_width) / 2,client.y + (client.h - dialog_height) / 2,dialog_width,dialog_height};
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
}

void PixUIDialog::dismiss()
{
    canvas.release_capture();
    visible = false;
}

void PixUIDialog::accept()
{
    if (accepted) accepted();
}

bool PixUIDialog::event(const PixUIEvent &input)
{
    if (!visible) return false;
    if (input.type == PIXUI_KEY && input.key == PIXUI_ESCAPE) dismiss();
    else PixUIForm::event(input);
    return true;
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
    PixUILayout buttons = {bounds.x + bounds.w - UI_MARGIN - UI_DIALOG_BUTTON_WIDTH * 2 - UI_MARGIN,bounds.y + bounds.h - UI_MARGIN - UI_FIELD_HEIGHT,UI_MARGIN};
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
    PixUIDialog::layout();
    PixUIDialogLayout positions = controls();
    message->bounds = {positions.label.x,positions.label.y,positions.label.w,positions.cancel.y - positions.label.y - UI_MARGIN};
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
    if (!lines.empty()) lines[0]->color = COLOR_TITLE;
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

PixUIFileBrowser::PixUIFileBrowser(PixUICanvas &context, PixUIFileSystem &provider, const std::string &initial) : PixUIDialog(context,"Open file","Open"),filesystem(provider)
{
    set_path(initial);
    label = add<PixUILabel>();
    edit = add<PixUIEdit>(path);
    register_focus(*edit);
    hint = add<PixUILabel>();
    browse = add<PixUIButton>("Browse");
    browse->activate = [this]() { refresh(); };
    up = add<PixUIButton>("Up");
    up->activate = [this]() {
        set_path(filesystem.join_path(directory,".."));
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
    if (filesystem.path_kind(directory) != PIXUI_PATH_DIRECTORY) directory = filesystem.parent_path(directory);
    std::string message;
    if (!filesystem.read_directory(directory,entries,message)) {
        entries.clear();
        if (error) error(message);
    }
    position = 0;
}

void PixUIFileBrowser::accept()
{
    if (path.text.empty()) return;
    PixUIPathKind kind = filesystem.path_kind(path.text);
    if (!dir_selector && kind == PIXUI_PATH_DIRECTORY) {
        refresh();
        return;
    }
    if (dir_selector && kind != PIXUI_PATH_DIRECTORY) {
        if (error) error(UI_DIRECTORY_ERROR + path.text);
        return;
    }
    if (kind == PIXUI_PATH_OTHER) return;
    if (selected && selected(path.text)) dismiss();
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
    std::string value = filesystem.join_path(directory,entries[index].name);
    value = filesystem.normalize(value);
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
