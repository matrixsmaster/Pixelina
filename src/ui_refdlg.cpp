/* This file is a part of 'Pixelina' Image Generator, Editor, and Viewer.
 *
 * (C) Dmitry 'sciloaf' Solovyev, 2026
 *
 * License: MIT
 */
#include "ui_refdlg.h"
#include "dlg_bookimg.h"
#include "image.h"
#include "qwen_image.h"
#include <SDL.h>
#include <algorithm>

PixelinaReferenceDialog::PixelinaReferenceDialog(PixUICanvas &context, const std::vector<std::string> &paths, const std::string &text, const std::vector<std::string> &saved) : PixUIDialog(context,UI_REFERENCE_TITLE,UI_OK_TEXT),references(paths),bookmarks(saved)
{
    resizable = true;
    minimum_width = UI_REFERENCE_MIN_WIDTH;
    minimum_height = UI_REFERENCE_MIN_HEIGHT;
    carousel = add<PixUICarousel>();
    carousel->caption = [](size_t index) {
        return QWEN_VL_REFERENCE_PREFIX + std::to_string(index + 1) + ">";
    };
    carousel->load = [this](size_t index) {
        return load? load(references[index]) : PixUIThumbCache{};
    };
    carousel->release = [](void* texture) {
        SDL_DestroyTexture(static_cast<SDL_Texture*>(texture));
    };
    carousel->selected = [this](size_t index) {
        selection = int(index);
        return true;
    };
    carousel->items_changed(references.size(),true);
    selection = references.empty()? -1 : 0;
    carousel->select(selection);
    prompt = add<PixUIMemo>();
    prompt->set_text(text);
    register_focus(*prompt);
    hint = add<PixUILabel>(UI_REFERENCE_HINT);
    add_button = add<PixUIButton>("Add Image");
    add_button->activate = [this]() {
        if (references.size() >= QWEN_IMAGE_MAX_REFERENCES) return;
        auto browser = std::make_unique<PixUIFileBrowser>(canvas);
        browser->title = "Add reference image";
        browser->error = [this](const std::string &error) {
            hint->caption = error;
        };
        browser->selected = [this](const std::string &path) {
            return add_image(path);
        };
        browser->refresh();
        present(std::move(browser));
    };
    bookmark_button = add<PixUIButton>("Bookmarks");
    bookmark_button->activate = [this]() {
        if (references.size() >= QWEN_IMAGE_MAX_REFERENCES) return;
        auto selector = std::make_unique<PixelinaBookmarkDialog>(canvas,bookmarks,load);
        selector->selected = [this](const std::string &path) {
            return add_image(path);
        };
        present(std::move(selector));
    };
    delete_button = add<PixUIButton>("Del Image");
    delete_button->activate = [this]() {
        delete_image();
    };
}

void PixelinaReferenceDialog::layout()
{
    PixUIDialog::layout();
    PixUIDialogLayout positions = controls();
    int button_y = positions.ok.y;
    int top = bounds.y + UI_TITLE_HEIGHT + UI_MARGIN;
    int content_height = std::max(0,button_y - UI_MARGIN - top);
    int carousel_height = int((content_height - UI_LINE_HEIGHT - UI_MARGIN * 2) * UI_REFERENCE_CAROUSEL_FRACTION);
    carousel->bounds = {bounds.x + UI_MARGIN,top,bounds.w - UI_MARGIN * 2,std::max(UI_SCROLLBAR,carousel_height)};
    hint->bounds = {carousel->bounds.x,top + carousel->bounds.h + UI_MARGIN,carousel->bounds.w,UI_LINE_HEIGHT};
    int prompt_top = hint->bounds.y + hint->bounds.h + UI_MARGIN;
    prompt->bounds = {carousel->bounds.x,prompt_top,carousel->bounds.w,std::max(0,button_y - UI_MARGIN - prompt_top)};
    PixUILayout buttons = {carousel->bounds.x,button_y,UI_MARGIN};
    for (PixUIButton* button : {add_button,bookmark_button,delete_button}) button->bounds = buttons.row(UI_REFERENCE_BTN_WIDTH,UI_FIELD_HEIGHT);
    add_button->enabled = references.size() < QWEN_IMAGE_MAX_REFERENCES;
    bookmark_button->enabled = add_button->enabled;
    delete_button->enabled = selection >= 0 && size_t(selection) < references.size();
    carousel->layout();
    prompt->layout();
}

bool PixelinaReferenceDialog::event(const PixUIEvent &input)
{
    // Keep Enter available for multiline prompts instead of invoking the default OK button.
    if (visible && !modal() && prompt->active && input.type == PIXUI_KEY && input.key == PIXUI_RETURN) return prompt->event(input);
    return PixUIDialog::event(input);
}

void PixelinaReferenceDialog::accept()
{
    if (selected) selected(references,prompt->model.text);
    dismiss();
}

bool PixelinaReferenceDialog::add_image(const std::string &path)
{
    if (references.size() >= QWEN_IMAGE_MAX_REFERENCES) return false;
    std::string actual = path_normalize(path);
    PixImage decoded;
    std::string error;
    if (!image_load(actual,decoded,error) || !image_valid(decoded)) {
        hint->caption = error.empty()? UI_REFERENCE_OPEN_ERROR + actual : error;
        return false;
    }
    // Keep only paths and bounded thumbnails; full planes are loaded when freezing a job.
    references.push_back(actual);
    selection = int(references.size()) - 1;
    hint->caption = UI_REFERENCE_HINT;
    carousel->items_changed(references.size());
    carousel->select(selection);
    return true;
}

void PixelinaReferenceDialog::delete_image()
{
    if (selection < 0 || size_t(selection) >= references.size()) return;
    references.erase(references.begin() + selection);
    selection = references.empty()? -1 : std::min(selection,int(references.size()) - 1);
    hint->caption = UI_REFERENCE_HINT;
    carousel->items_changed(references.size());
    carousel->select(selection);
}
