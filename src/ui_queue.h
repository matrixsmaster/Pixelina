/* This file is a part of 'Pixelina' Image Generator, Editor, and Viewer.
 *
 * (C) Dmitry 'sciloaf' Solovyev, 2026
 *
 * License: MIT
 */
#ifndef PIXELINA_UI_QUEUE_H
#define PIXELINA_UI_QUEUE_H

#include "pixuilib.h"
#include "storage.h"

#define UI_QUEUE_TITLE "Queue Browser"
#define UI_QUEUE_WIDTH 980
#define UI_QUEUE_HEIGHT 540
#define UI_QUEUE_MIN_WIDTH 760
#define UI_QUEUE_MIN_HEIGHT 300
#define UI_QUEUE_ROW_HEIGHT (PIXELINA_THUMB_SIZE + UI_MARGIN * 2)
#define UI_QUEUE_PROGRESS_COLUMNS 24
#define UI_QUEUE_PROGRESS_WIDTH (UI_FONT_WIDTH * UI_QUEUE_PROGRESS_COLUMNS + UI_MARGIN * 2)
#define UI_QUEUE_TIMING_BUFFER 64
#define UI_QUEUE_THUMB_WIDTH (PIXELINA_THUMB_SIZE + UI_MARGIN * 2)
#define UI_QUEUE_NO_TASK SIZE_MAX
#define UI_QUEUE_PENDING_TEXT "Queued"
#define UI_QUEUE_PAUSED_TEXT "Paused"
#define UI_QUEUE_PAUSING_TEXT "Pausing"
#define UI_QUEUE_FAILED_TEXT "Failed"
#define UI_QUEUE_LOADING_TEXT "Loading models"
#define UI_QUEUE_CLEAR_TEXT "Clear all"
#define UI_QUEUE_IMPORT_TEXT "Import"
#define UI_QUEUE_EXPORT_TEXT "Export"
#define UI_QUEUE_CLEARED_TEXT "Queue cleared"
#define UI_QUEUE_REMOVE_TEXT "Remove job"
#define UI_QUEUE_STOPPING_ERROR "The queue is stopping. Wait for it to finish stopping."
#define UI_QUEUE_NOT_RUNNING_ERROR "The queue is not running."

struct PixJob {
    PixRecord parameters; // unresolved immutable result metadata
    PixSettings generate_settings; // resolved Generate execution settings
    PixEditSettings edit_settings; // resolved Edit execution settings
    std::vector<PixImage> references; // owned complete reference planes
};

enum PixQueueStatus {
    PIX_QUEUE_PENDING,PIX_QUEUE_RUNNING,PIX_QUEUE_COMPLETE,PIX_QUEUE_FAILED
};

struct PixQueueItem {
    PixJob job; // pending job, moved to the worker while running and released on success
    std::string description; // immutable configuration and complete prompt
    PixImage source; // bounded source collage
    PixImage output; // bounded thumbnail of the saved output
    std::string filename; // saved output path
    std::string error; // last failure diagnostic
    PixQueueStatus state = PIX_QUEUE_PENDING; // lifecycle independent of browser visibility
    float progress = 0; // completed inference fraction
    double seconds = 0; // active time for this attempt, excluding pauses
};

struct PixQueue {
    std::vector<PixQueueItem> items; // retained job rows and results until removed
    size_t active = UI_QUEUE_NO_TASK; // current worker row, if any
    bool running = false; // continue to the next pending item
    bool pausing = false,paused = false; // requested and acknowledged boundary suspension
    double elapsed = 0; // active time of finished attempts
    uint64_t revision = 0; // UI-thread invalidation counter
    uint64_t reset_revision = 0; // invalidates cached rows after the entire queue is cleared
};

PixImage queue_source_thumbnail(const std::vector<PixImage> &references);
std::string queue_job_description(const PixJob &job);

class PixelinaQueueRows : public PixUIForm
{
public:
    PixelinaQueueRows(PixUICanvas &context, const PixQueue &state);
    ~PixelinaQueueRows() override;

    void layout() override;
    void draw() override;
    bool event(const PixUIEvent &input) override;
    void items_changed();

    size_t selection = UI_QUEUE_NO_TASK; // selected row eligible for removal when inactive

private:
    const PixQueue &queue; // UI-owned state; the worker never accesses these rows
    PixUIScrollbar scrollbar; // owned vertical row navigation
    int scroll = 0; // content offset in pixels
    std::map<size_t,std::pair<void*,void*>> textures; // textures for visible source and result thumbnails
    std::map<size_t,PixUIMemo*> descriptions; // owned by the form, retained only for visible rows

    void draw_row(size_t index, const PixUIRect &rect);
};

class PixelinaQueueDialog : public PixUIDialog
{
public:
    PixelinaQueueDialog(PixUICanvas &context, PixQueue &state);

    void layout() override;
    bool needs_repaint() override;

    PixUIButton* clear_button; // confirms clearing every queued job
    PixUIButton* start_button; // starts or resumes queued jobs
    PixUIButton* stop_button; // stops the active queued job
    PixUIButton* pause_button; // pauses queued execution
    PixUIButton* import_button; // clears the queue and loads a job file
    PixUIButton* export_button; // saves unfinished jobs to a job file

private:
    PixQueue &queue; // live UI-owned state independent of modal lifetime
    uint64_t revision = UINT64_MAX; // last displayed state version
    uint64_t reset_revision = UINT64_MAX; // last displayed queue reset
    PixelinaQueueRows* rows; // owned scrollable queue body
    PixUILabel* summary; // task count, percentage and elapsed / ETA
    PixUIProgress* progress; // overall completed job fraction
    PixUIButton* remove_button; // deletes the selected inactive job

    void remove_job();
};

#endif
