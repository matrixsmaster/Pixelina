/* This file is a part of 'Pixelina' Image Generator, Editor, and Viewer.
 *
 * (C) Dmitry 'sciloaf' Solovyev, 2026
 *
 * License: MIT
 */
#ifndef PIXELINA_UI_CHATDLG_H
#define PIXELINA_UI_CHATDLG_H

#include <SDL.h>
#include <atomic>
#include <mutex>
#include "pixuilib.h"
#include "qwen_chat.h"

#define UI_STOP_TEXT "Stop"
#define UI_STOPPING_TEXT "Stopping..."
#define UI_CHAT_TITLE "Chat with "
#define UI_CHAT_SEND "Send"
#define UI_CHAT_CLOSE "Close"
#define UI_CHAT_MIN_WIDTH 480
#define UI_CHAT_MIN_HEIGHT 300
#define UI_CHAT_START_WIDTH 640
#define UI_CHAT_START_HEIGHT 480
#define UI_CHAT_USER "You: "
#define UI_CHAT_MODEL "Qwen: "
#define UI_CHAT_STOPPED "\n[Stopped]"
#define UI_CHAT_ERROR "\n[Error] "

class PixelinaChatDialog : public PixUIDialog
{
public:
    PixelinaChatDialog(PixUICanvas &context, const std::string &name, const PixEditSettings &settings, const PixImage &image, const PixChatInference &inference = qwen_chat_generate);
    ~PixelinaChatDialog() override;

    void layout() override;
    bool event(const PixUIEvent &input) override;
    bool needs_repaint() override;
    void accept() override;
    void cancel_capture() override;

private:
    friend struct PixelinaTest;

    PixUIMemo* memo; // owned selectable, read-only conversation log
    PixUIEdit* edit; // owned single-line message input
    PixEditSettings settings; // frozen language, vision, resolution and backend selections
    PixImage image; // frozen current image, including unsaved images
    PixChatInference inference; // worker inference entry, injectable for UI verification
    std::vector<PixChatMessage> messages; // conversation retained only for this dialog lifetime
    SDL_Thread* worker = nullptr; // sole inference worker for the active reply
    std::atomic<bool> stop_requested{false}; // cooperative cancellation read by the worker
    std::mutex mutex; // protects streamed bytes and completion fields
    bool worker_done = false; // completion published under mutex
    std::string pending,response,error; // synchronized streamed bytes and final result
    bool moving = false; // title-bar pointer capture is active
    int move_x = 0,move_y = 0; // last pointer position during title-bar movement

    void close();
    void append(const std::string &text);
    static int chat_worker(void* context);
};

#endif
