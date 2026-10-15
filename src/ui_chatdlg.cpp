/* This file is a part of 'Pixelina' Image Generator, Editor, and Viewer.
 *
 * (C) Dmitry 'sciloaf' Solovyev, 2026
 *
 * License: MIT
 */
#include "ui_chatdlg.h"
#include <algorithm>

PixelinaChatDialog::PixelinaChatDialog(PixUICanvas &context, const std::string &name, const PixEditSettings &parameters, const PixImage &source, const PixChatInference &run) : PixUIDialog(context,UI_CHAT_TITLE + name,UI_CHAT_SEND,UI_CHAT_CLOSE),settings(parameters),image(source),inference(run)
{
    resizable = true;
    minimum_width = UI_CHAT_MIN_WIDTH;
    minimum_height = UI_CHAT_MIN_HEIGHT;
    dialog_width = UI_CHAT_START_WIDTH;
    dialog_height = UI_CHAT_START_HEIGHT;
    memo = add<PixUIMemo>();
    memo->read_only = true;
    register_focus(*memo);
    edit = add<PixUIEdit>();
    register_focus(*edit);
    focus_control(*edit);
    cancel_button->activate = [this]() { close(); };
    close_button->activate = [this]() { close(); };
}

PixelinaChatDialog::~PixelinaChatDialog()
{
    stop_requested = true;
    if (worker) SDL_WaitThread(worker,nullptr);
}

void PixelinaChatDialog::layout()
{
    accept_caption = worker? (stop_requested? UI_STOPPING_TEXT : UI_STOP_TEXT) : UI_CHAT_SEND;
    PixUIDialog::layout();
    int right = bounds.x + bounds.w - UI_MARGIN - UI_SCROLLBAR;
    int bottom = bounds.y + bounds.h - UI_MARGIN - UI_FIELD_HEIGHT;
    cancel_button->bounds = {bounds.x + UI_MARGIN,bottom,right - bounds.x - UI_MARGIN,UI_FIELD_HEIGHT};
    int input_y = bottom - UI_MARGIN - UI_FIELD_HEIGHT;
    accept_button->bounds = {right - UI_DIALOG_BUTTON_WIDTH,input_y,UI_DIALOG_BUTTON_WIDTH,UI_FIELD_HEIGHT};
    edit->bounds = {bounds.x + UI_MARGIN,input_y,accept_button->bounds.x - bounds.x - UI_MARGIN * 2,UI_FIELD_HEIGHT};
    memo->bounds = {edit->bounds.x,bounds.y + UI_TITLE_HEIGHT + UI_MARGIN,right - edit->bounds.x,std::max(0,input_y - bounds.y - UI_TITLE_HEIGHT - UI_MARGIN * 2)};
    edit->enabled = !worker;
    accept_button->enabled = !worker || !stop_requested;
    memo->layout();
    edit->layout();
}

void PixelinaChatDialog::append(const std::string &text)
{
    if (text.empty()) return;
    memo->model.text += text;
    memo->scroll_to_end();
    repaint = true;
}

void PixelinaChatDialog::accept()
{
    if (worker) {
        stop_requested = true;
        layout();
        return;
    }
    if (edit->model.text.find_first_not_of(" \t\r\n") == std::string::npos) return;
    messages.push_back({false,edit->model.text});
    stop_requested = false;
    worker_done = false;
    pending.clear();
    response.clear();
    error.clear();
    worker = SDL_CreateThread(chat_worker,"pixelina-chat",this);
    if (!worker) {
        messages.pop_back();
        append(UI_CHAT_ERROR + std::string(SDL_GetError()) + '\n');
        return;
    }
    append((memo->model.text.empty()? "" : "\n\n") + std::string(UI_CHAT_USER) + edit->model.text + "\n\n" UI_CHAT_MODEL);
    edit->set_text("");
    layout();
}

int PixelinaChatDialog::chat_worker(void* context)
{
    auto dialog = static_cast<PixelinaChatDialog*>(context);
    auto progress = [dialog](float, const std::string &) {
        return !dialog->stop_requested.load();
    };
    auto stream = [dialog](const std::string &text) {
        if (dialog->stop_requested) return false;
        std::lock_guard<std::mutex> lock(dialog->mutex);
        dialog->pending += text;
        return true;
    };
    std::string response,error;
    bool success = dialog->inference(dialog->settings,dialog->image,dialog->messages,response,error,progress,stream);
    if (!success && error.empty() && !dialog->stop_requested) error = "Qwen-VL chat inference failed.";
    std::lock_guard<std::mutex> lock(dialog->mutex);
    dialog->response = std::move(response);
    dialog->error = std::move(error);
    dialog->worker_done = true;
    return 0;
}

bool PixelinaChatDialog::needs_repaint()
{
    if (worker) {
        bool done;
        std::string text;
        {
            std::lock_guard<std::mutex> lock(mutex);
            text.swap(pending);
            done = worker_done;
        }
        append(text);
        if (done) {
            SDL_WaitThread(worker,nullptr);
            worker = nullptr;
            messages.push_back({true,response});
            if (stop_requested) append(UI_CHAT_STOPPED);
            else if (!error.empty()) append(UI_CHAT_ERROR + error);
            layout();
            focus_control(*edit);
            repaint = true;
        }
    }
    return PixUIDialog::needs_repaint();
}

void PixelinaChatDialog::close()
{
    stop_requested = true;
    if (worker) {
        SDL_WaitThread(worker,nullptr);
        worker = nullptr;
    }
    dismiss();
}

bool PixelinaChatDialog::event(const PixUIEvent &input)
{
    if (!visible) return false;
    if (input.type == PIXUI_KEY && input.key == PIXUI_ESCAPE) {
        close();
        return true;
    }
    if (moving && canvas.captured == this) {
        if (input.type == PIXUI_UP && input.button == PIXUI_LEFT_BUTTON) canvas.release_capture();
        else if (input.type == PIXUI_MOTION) {
            PixUIRect client = parent? parent->bounds : canvas.window_rect();
            bounds.x = std::max(client.x,std::min(client.x + client.w - bounds.w,bounds.x + input.x - move_x));
            bounds.y = std::max(client.y,std::min(client.y + client.h - bounds.h,bounds.y + input.y - move_y));
            move_x = input.x;
            move_y = input.y;
            layout();
            repaint = true;
        }
        return true;
    }
    if (input.type == PIXUI_DOWN && input.button == PIXUI_LEFT_BUTTON && pixui_inside(title_label->bounds,input.x,input.y) && !pixui_inside(close_button->bounds,input.x,input.y)) {
        moving = true;
        move_x = input.x;
        move_y = input.y;
        canvas.captured = this;
        return true;
    }
    return PixUIDialog::event(input);
}

void PixelinaChatDialog::cancel_capture()
{
    moving = false;
    PixUIDialog::cancel_capture();
}
