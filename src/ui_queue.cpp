/* This file is a part of 'Pixelina' Image Generator, Editor, and Viewer.
 *
 * (C) Dmitry 'sciloaf' Solovyev, 2026
 *
 * License: MIT
 */
#include "ui_queue.h"
#include "ui_sdl.h"
#include <algorithm>
#include <math.h>
#include <stdio.h>

PixImage queue_source_thumbnail(const std::vector<PixImage> &references)
{
    if (references.empty()) return {};
    if (references.size() == 1) return image_thumbnail(references.front());
    PixImage result;
    result.width = result.height = PIXELINA_THUMB_SIZE;
    result.rgb.resize(size_t(result.width) * result.height * 3,UINT8_MAX);
    int columns = int(ceil(sqrt(double(references.size()))));
    int rows = (int(references.size()) + columns - 1) / columns;
    int cell_width = result.width / columns,cell_height = result.height / rows;
    for (size_t i = 0; i < references.size(); i++) {
        const PixImage &source = references[i];
        double scale = std::min(double(cell_width) / source.width,double(cell_height) / source.height);
        PixImage tile = image_resize(source,std::max(1,int(source.width * scale)),std::max(1,int(source.height * scale)),PIX_IMAGE_BICUBIC,PIX_IMAGE_WHITE);
        int x = int(i % columns) * cell_width + (cell_width - tile.width) / 2;
        int y = int(i / columns) * cell_height + (cell_height - tile.height) / 2;
        for (int row = 0; row < tile.height; row++) std::copy_n(tile.rgb.data() + size_t(row) * tile.width * 3,size_t(tile.width) * 3,result.rgb.data() + (size_t(y + row) * result.width + x) * 3);
    }
    return result;
}

std::string queue_job_description(const PixJob &job)
{
    bool edit = job.parameters.model == PIX_MODEL_QWEN_IMAGE;
    int width = edit? job.edit_settings.width : job.generate_settings.width;
    int height = edit? job.edit_settings.height : job.generate_settings.height;
    int steps = edit? job.edit_settings.steps : job.generate_settings.steps;
    uint64_t seed = edit? job.edit_settings.seed : job.generate_settings.seed;
    std::string description = edit? "Edit " : "Generate ";
    description += std::to_string(width) + "x" + std::to_string(height) + ", " + std::to_string(steps) + " steps, seed " + std::to_string(seed);
    return description + ": " + job.parameters.prompt;
}

PixelinaQueueRows::PixelinaQueueRows(PixUICanvas &context, const PixQueue &state)
    : PixUIForm(context)
    , queue(state)
    , scrollbar(context)
{
    scrollbar.parent = this;
    scrollbar.vertical = true;
    scrollbar.position = &scroll;
    scrollbar.step = UI_QUEUE_ROW_HEIGHT;
}

PixelinaQueueRows::~PixelinaQueueRows()
{
    for (const auto &entry : textures) {
        if (entry.second.first) SDL_DestroyTexture((SDL_Texture*)entry.second.first);
        if (entry.second.second) SDL_DestroyTexture((SDL_Texture*)entry.second.second);
    }
}

void PixelinaQueueRows::layout()
{
    if (queue.items.empty()) selection = UI_QUEUE_NO_TASK;
    scrollbar.bounds = {bounds.x + bounds.w - UI_SCROLLBAR,bounds.y,UI_SCROLLBAR,bounds.h};
    scrollbar.total = int(queue.items.size()) * UI_QUEUE_ROW_HEIGHT;
    scrollbar.page = bounds.h;
    scrollbar.visible = scrollbar.total > scrollbar.page;
    scroll = std::max(0,std::min(scroll,std::max(0,scrollbar.total - scrollbar.page)));
    size_t first = size_t(scroll / UI_QUEUE_ROW_HEIGHT);
    size_t last = std::min(queue.items.size(),size_t((scroll + bounds.h + UI_QUEUE_ROW_HEIGHT - 1) / UI_QUEUE_ROW_HEIGHT));
    for (auto entry = descriptions.begin(); entry != descriptions.end();) {
        if (entry->first >= first && entry->first < last) {
            ++entry;
            continue;
        }
        remove(*entry->second);
        entry = descriptions.erase(entry);
    }
    int description_x = bounds.x + UI_QUEUE_THUMB_WIDTH;
    int description_width = std::max(0,bounds.w - UI_SCROLLBAR - UI_QUEUE_THUMB_WIDTH * 2 - UI_QUEUE_PROGRESS_WIDTH - UI_MARGIN);
    for (size_t i = first; i < last; i++) {
        PixUIMemo* &memo = descriptions[i];
        if (!memo) {
            memo = add<PixUIMemo>();
            memo->read_only = true;
            register_focus(*memo);
        }
        const PixQueueItem &item = queue.items[i];
        std::string text = item.description;
        if (!item.error.empty()) text += "\n" + item.error;
        if (memo->model.text != text) {
            memo->set_text(text);
            memo->model.cursor = memo->model.anchor = 0;
        }
        memo->bounds = {description_x,bounds.y + int(i) * UI_QUEUE_ROW_HEIGHT - scroll + UI_MARGIN,description_width,UI_QUEUE_ROW_HEIGHT - UI_MARGIN * 2};
        memo->layout();
    }
}

void PixelinaQueueRows::draw()
{
    repaint = false;
    if (!visible) return;
    layout();
    canvas.fill(bounds,UI_COLOR_LIGHT);
    PixUIRect content = {bounds.x,bounds.y,std::max(0,bounds.w - UI_SCROLLBAR),bounds.h};
    PixUIRect previous,effective = content;
    bool clipped = canvas.clip(previous);
    size_t first = size_t(scroll / UI_QUEUE_ROW_HEIGHT);
    size_t last = std::min(queue.items.size(),size_t((scroll + bounds.h + UI_QUEUE_ROW_HEIGHT - 1) / UI_QUEUE_ROW_HEIGHT));
    if (!clipped || pixui_intersect(content,previous,effective)) {
        canvas.set_clip(&effective);
        for (size_t i = first; i < last; i++) draw_row(i,{content.x,content.y + int(i) * UI_QUEUE_ROW_HEIGHT - scroll,content.w,UI_QUEUE_ROW_HEIGHT});
        canvas.set_clip(clipped? &previous : nullptr);
    }
    // Release offscreen textures so long batches do not consume growing GPU memory.
    for (auto entry = textures.begin(); entry != textures.end();) {
        if (entry->first >= first && entry->first < last) {
            ++entry;
            continue;
        }
        if (entry->second.first) SDL_DestroyTexture((SDL_Texture*)entry->second.first);
        if (entry->second.second) SDL_DestroyTexture((SDL_Texture*)entry->second.second);
        entry = textures.erase(entry);
    }
    scrollbar.draw();
}

bool PixelinaQueueRows::event(const PixUIEvent &input)
{
    if (!visible || !enabled) return false;
    layout();
    if (scrollbar.visible && scrollbar.event(input)) {
        repaint = true;
        return true;
    }
    bool selected = false;
    PixUIRect content = {bounds.x,bounds.y,std::max(0,bounds.w - UI_SCROLLBAR),bounds.h};
    if (input.type == PIXUI_DOWN && input.button == PIXUI_LEFT_BUTTON && pixui_inside(content,input.x,input.y)) {
        size_t index = size_t((input.y - bounds.y + scroll) / UI_QUEUE_ROW_HEIGHT);
        selection = index < queue.items.size()? index : UI_QUEUE_NO_TASK;
        repaint = true;
        selected = true;
    }
    if (PixUIForm::event(input)) return true;
    if (input.type == PIXUI_WHEEL && pixui_inside(bounds,input.x,input.y)) {
        scroll -= input.wheel_y * UI_QUEUE_ROW_HEIGHT;
        layout();
        repaint = true;
        return true;
    }
    return selected;
}

void PixelinaQueueRows::items_changed()
{
    // Row indices shift after deletion; discard every cached association before reusing them.
    for (const auto &entry : textures) {
        if (entry.second.first) SDL_DestroyTexture((SDL_Texture*)entry.second.first);
        if (entry.second.second) SDL_DestroyTexture((SDL_Texture*)entry.second.second);
    }
    textures.clear();
    for (const auto &entry : descriptions) remove(*entry.second);
    descriptions.clear();
    selection = UI_QUEUE_NO_TASK;
    repaint = true;
    layout();
}

void PixelinaQueueRows::draw_row(size_t index, const PixUIRect &rect)
{
    const PixQueueItem &item = queue.items[index];
    auto &cached = textures[index];
    PixUISDLCanvas &context = (PixUISDLCanvas&)canvas;
    if (!cached.first && image_valid(item.source)) cached.first = context.create_texture(item.source);
    if (!cached.second && image_valid(item.output)) cached.second = context.create_texture(item.output);
    int output_x = rect.x + rect.w - UI_QUEUE_THUMB_WIDTH;
    int progress_x = output_x - UI_QUEUE_PROGRESS_WIDTH;
    int description_x = rect.x + UI_QUEUE_THUMB_WIDTH;
    if (selection == index) {
        canvas.fill(rect,UI_COLOR_SELECTED);
        canvas.fill({rect.x + UI_BORDER_WIDTH,rect.y + UI_BORDER_WIDTH,std::max(0,rect.w - UI_BORDER_WIDTH * 2),std::max(0,rect.h - UI_BORDER_WIDTH * 2)},UI_COLOR_LIGHT);
    }
    for (bool output : {false,true}) {
        const PixImage &source = output? item.output : item.source;
        PixUIImage thumbnail(canvas);
        thumbnail.bounds = {output? output_x + UI_MARGIN : rect.x + UI_MARGIN,rect.y + UI_MARGIN,PIXELINA_THUMB_SIZE,PIXELINA_THUMB_SIZE};
        if (output && !item.filename.empty()) thumbnail.bounds.h = std::max(1,thumbnail.bounds.h - UI_LINE_HEIGHT);
        thumbnail.texture = output? cached.second : cached.first;
        thumbnail.width = source.width;
        thumbnail.height = source.height;
        thumbnail.placeholder = output? "" : "Text to image";
        thumbnail.draw();
    }
    descriptions[index]->draw();
    PixUIRect previous,effective;
    bool clipped = canvas.clip(previous);
    PixUIProgress meter(canvas);
    meter.bounds = {progress_x + UI_MARGIN,rect.y + (rect.h - UI_FIELD_HEIGHT) / 2,UI_QUEUE_PROGRESS_WIDTH - UI_MARGIN * 2,UI_FIELD_HEIGHT};
    meter.busy = item.state == PIX_QUEUE_RUNNING;
    meter.progress = item.progress;
    switch (item.state) {
    case PIX_QUEUE_PENDING:
        meter.caption = UI_QUEUE_PENDING_TEXT;
        break;
    case PIX_QUEUE_RUNNING:
        break;
    case PIX_QUEUE_COMPLETE: {
        char timing[UI_QUEUE_TIMING_BUFFER];
        snprintf(timing,sizeof(timing),"Time spent: %.2fs",item.seconds);
        meter.caption = timing;
        break;
    }
    case PIX_QUEUE_FAILED:
        meter.caption = UI_QUEUE_FAILED_TEXT;
        break;
    }
    if (item.state == PIX_QUEUE_COMPLETE) {
        PixUILabel elapsed(canvas,meter.caption);
        elapsed.bounds = meter.bounds;
        elapsed.draw();
    } else meter.draw();
    if (!item.filename.empty()) {
        PixUILabel filename(canvas,item.filename.substr(item.filename.find_last_of('/') + 1));
        filename.bounds = {output_x + UI_TEXT_PADDING,rect.y + rect.h - UI_LINE_HEIGHT,UI_QUEUE_THUMB_WIDTH - UI_TEXT_PADDING * 2,UI_LINE_HEIGHT};
        PixUIRect filename_area = filename.bounds;
        if (!clipped || pixui_intersect(filename_area,previous,effective)) {
            canvas.set_clip(clipped? &effective : &filename_area);
            filename.draw();
            canvas.set_clip(clipped? &previous : nullptr);
        }
    }
    canvas.fill({rect.x,rect.y + rect.h - 1,rect.w,1},UI_COLOR_SHADOW);
    for (int x : {description_x - UI_MARGIN,progress_x,output_x}) canvas.fill({x,rect.y,1,rect.h},UI_COLOR_SHADOW);
}

PixelinaQueueDialog::PixelinaQueueDialog(PixUICanvas &context, PixQueue &state)
    : PixUIDialog(context,UI_QUEUE_TITLE,"",UI_CLOSE_TEXT)
    , queue(state)
{
    resizable = true;
    minimum_width = UI_QUEUE_MIN_WIDTH;
    minimum_height = UI_QUEUE_MIN_HEIGHT;
    dialog_width = UI_QUEUE_WIDTH;
    dialog_height = UI_QUEUE_HEIGHT;
    rows = add<PixelinaQueueRows>(queue);
    summary = add<PixUILabel>();
    progress = add<PixUIProgress>();
    clear_button = add<PixUIButton>(UI_QUEUE_CLEAR_TEXT);
    remove_button = add<PixUIButton>(UI_QUEUE_REMOVE_TEXT);
    remove_button->activate = [this]() {
        remove_job();
    };
    start_button = add<PixUIButton>("Start");
    stop_button = add<PixUIButton>(UI_STOP_TEXT);
    pause_button = add<PixUIButton>("Pause");
    import_button = add<PixUIButton>(UI_QUEUE_IMPORT_TEXT);
    export_button = add<PixUIButton>(UI_QUEUE_EXPORT_TEXT);
}

void PixelinaQueueDialog::layout()
{
    PixUIDialog::layout();
    int footer_y = controls().cancel.y - UI_MARGIN - UI_FIELD_HEIGHT;
    int top = bounds.y + UI_TITLE_HEIGHT + UI_MARGIN;
    rows->bounds = {bounds.x + UI_MARGIN,top,bounds.w - UI_MARGIN * 2,std::max(0,footer_y - UI_MARGIN - top)};
    summary->bounds = {rows->bounds.x,footer_y,rows->bounds.w - UI_QUEUE_PROGRESS_WIDTH - UI_MARGIN,UI_FIELD_HEIGHT};
    progress->bounds = {summary->bounds.x + summary->bounds.w + UI_MARGIN,footer_y,UI_QUEUE_PROGRESS_WIDTH,UI_FIELD_HEIGHT};
    size_t completed = 0,task = queue.items.size();
    double fraction = 0,elapsed = queue.elapsed;
    for (size_t i = 0; i < queue.items.size(); i++) {
        const PixQueueItem &item = queue.items[i];
        if (item.state == PIX_QUEUE_COMPLETE) completed++;
        else if (task == queue.items.size()) task = i;
        if (item.state == PIX_QUEUE_RUNNING) {
            fraction += item.progress;
            elapsed += item.seconds;
        }
    }
    fraction = queue.items.empty()? 0 : (completed + fraction) / queue.items.size();
    size_t position = queue.items.empty()? 0 : std::min(task + 1,queue.items.size());
    char timing[UI_QUEUE_TIMING_BUFFER];
    snprintf(timing,sizeof(timing),"%.0fs / ",elapsed);
    std::string timing_text = timing;
    if (fraction > 0) {
        snprintf(timing,sizeof(timing),"%.0fs",elapsed * (1 - fraction) / fraction);
        timing_text += timing;
    } else timing_text += "--";
    summary->caption = "Task " + std::to_string(position) + "/" + std::to_string(queue.items.size()) + " | " + std::to_string(int(fraction * UI_PERCENT_SCALE)) + "% done | " + timing_text + " ETA";
    if (queue.paused || queue.pausing) summary->caption += queue.paused? " | " UI_QUEUE_PAUSED_TEXT : " | " UI_QUEUE_PAUSING_TEXT;
    progress->busy = true;
    progress->progress = float(fraction);
    rows->layout();
    PixUILayout buttons = {rows->bounds.x,controls().cancel.y,UI_MARGIN};
    clear_button->bounds = buttons.row(UI_DIALOG_BUTTON_WIDTH,UI_FIELD_HEIGHT);
    remove_button->bounds = buttons.row(UI_DIALOG_BUTTON_WIDTH + UI_MARGIN * 2,UI_FIELD_HEIGHT);
    for (PixUIButton* button : {start_button,stop_button,pause_button,import_button,export_button}) button->bounds = buttons.row(UI_DIALOG_BUTTON_WIDTH,UI_FIELD_HEIGHT);
}

bool PixelinaQueueDialog::needs_repaint()
{
    if (reset_revision != queue.reset_revision) {
        reset_revision = queue.reset_revision;
        rows->items_changed();
    }
    if (revision != queue.revision) {
        revision = queue.revision;
        repaint = true;
    }
    return PixUIDialog::needs_repaint();
}

void PixelinaQueueDialog::remove_job()
{
    size_t index = rows->selection;
    if (index >= queue.items.size() || index == queue.active || queue.items[index].state == PIX_QUEUE_RUNNING) return;
    queue.items.erase(queue.items.begin() + index);
    if (queue.active != UI_QUEUE_NO_TASK && index < queue.active) queue.active--;
    queue.revision++;
    rows->items_changed();
    layout();
    repaint = true;
}
