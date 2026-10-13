/* This file is a part of 'Pixelina' Image Generator, Editor, and Viewer.
 *
 * (C) Dmitry 'sciloaf' Solovyev, 2026
 *
 * License: MIT
 */
#ifndef PIXELINA_UI_H
#define PIXELINA_UI_H

#include <SDL.h>
#include "storage.h"
#include "pixuilib.h"
#include <array>
#include <atomic>

#define PIXELINA_ABOUT { PIXELINA_NAME, "(C) Dmitry 'sciloaf' Solovyev, 2026", "Z-Image Turbo image generation", "Supports inference on CPU and Vulkan" }

#define UI_START_WIDTH 1100
#define UI_CAPTION_SEPARATOR " - "
#define UI_ABOUT_PREFIX "About "
#define UI_STATUS_BUFFER 64
#define UI_LEFT_FRACTION 0.32f
#define UI_PROMPT_FRACTION 0.30f
#define UI_CAROUSEL_FRACTION 0.24f
#define UI_START_HEIGHT 780
#define UI_STATUS_HEIGHT 26
#define UI_SPLITTER_WIDTH 6
#define UI_PANEL_MIN 140
#define UI_WORK_TOP (UI_FRAME + UI_WINDOW_TITLE_HEIGHT + UI_MENU_HEIGHT)
#define UI_TOP_MENU_COUNT 4
#define UI_FIELD_INPUT_Y 18
#define UI_FRAME_DELAY 16
#define UI_SETTINGS_COUNT 8
#define UI_SETTINGS_ROW 48
#define CMD_OPEN 1
#define CMD_SAVE 2
#define CMD_SAVE_AS 3
#define CMD_QUIT 4
#define CMD_GENERATE 5
#define CMD_SCALED 7
#define CMD_ABOUT 8
#define CMD_JPG 9
#define CMD_PNG 10
#define CMD_CPU 11
#define CMD_VULKAN 12
#define CMD_DECODE 13
#define UI_BROWSE_BUTTON_WIDTH 32
#define UI_MODEL_FIRST UI_DIFFUSER_FIELD
#define UI_GENERATE_TEXT "Generate"
#define UI_STOP_TEXT "Stop"
#define UI_STOPPING_TEXT "Stopping..."
#define UI_INVALID_NUMBERS "Seed, dimensions, steps and shift must be valid numbers."
#define UI_STOPPED_TEXT "Generation stopped"

enum PixelinaGroup {
    UI_PROMPT_GROUP,UI_SETTINGS_GROUP,UI_CAROUSEL_GROUP,UI_IMAGE_GROUP,UI_GROUP_COUNT
};

enum PixelinaSplitter {
    UI_SPLIT_LEFT,UI_SPLIT_PROMPT,UI_SPLIT_CAROUSEL,UI_SPLITTER_COUNT
};

enum PixelinaField {
    UI_SEED_FIELD,UI_WIDTH_FIELD,UI_HEIGHT_FIELD,UI_STEPS_FIELD,UI_SHIFT_FIELD,
    UI_DIFFUSER_FIELD,UI_ENCODER_FIELD,UI_VAE_FIELD
};

class PixUISDLCanvas : public PixUICanvas
{
public:
    SDL_Renderer* renderer = nullptr; // application renderer
    SDL_Window* window = nullptr; // application native window
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
    void window_command(PixUIWindowCommand command, const PixUIRect &rect) override;
    bool maximized() const override;
    void window_title(const std::string &title) override;
    PixUIEvent translate(const SDL_Event &input);

private:
    unsigned draw_color = 0; // cached renderer draw color
    bool color_valid = false; // renderer color has been set through this adapter
    bool clipped = false; // clipping enabled through this adapter
    PixUIRect clipping; // cached clip rectangle
    void set_color(unsigned color);
};

class PixUISystemFiles : public PixUIFileSystem
{
public:
    bool read_directory(const std::string &path, std::vector<PixUIFileEntry> &entries, std::string &error) override;
    std::string normalize(const std::string &path) override;
    PixUIPathKind path_kind(const std::string &path) override;
    std::string join_path(const std::string &directory, const std::string &name) override;
    std::string parent_path(const std::string &path) override;
};

class PixelinaForm : public PixUIForm
{
public:
    std::string caption = PIXELINA_NAME; // application window caption
    PixUIMemo* prompt; // prompt editor owned by its group
    std::array<PixUIEdit*,UI_SETTINGS_COUNT> fields; // owned settings editors
    std::array<PixUIButton*,UI_SETTINGS_COUNT> browse = {}; // model selection actions
    PixUILabel* preferences; // backend and format label
    PixUIButton* generate; // generation action
    PixUICarousel* carousel; // complete saved-image widget
    PixUIImageView* image; // complete image widget
    PixUIStatusBar* status; // complete status widget
    explicit PixelinaForm(PixUICanvas &context);
    void layout() override;

private:
    PixUIPanel* menu_background; // full-width menu background
    std::array<PixUIGroupBox*,UI_GROUP_COUNT> groups; // workspace containers
    std::array<PixUISplitter*,UI_SPLITTER_COUNT> splitters; // owned pane dividers
    float left_fraction = UI_LEFT_FRACTION,prompt_fraction = UI_PROMPT_FRACTION,carousel_fraction = UI_CAROUSEL_FRACTION; // owned pane ratios
};

class Pixelina
{
public:
    Pixelina() = default;
    ~Pixelina();
    bool initialize(const std::string &directory, bool png, std::string &error);
    int run();

private:
    PixUISDLCanvas canvas; // shared renderer and control capture
    PixelinaForm main_form{canvas}; // owns persistent workspace controls
    PixUIMenu* menus = nullptr; // borrowed menu root owned by main_form
    PixUISystemFiles files; // filesystem provider for the reusable selector
    SDL_Renderer* renderer = nullptr; // SDL renderer owned by the application
    SDL_Window* window = nullptr; // outer resizable window
    SDL_Texture* app_icon = nullptr; // embedded application icon for the About dialog
    SDL_Texture* current = nullptr; // current full-resolution image texture
    SDL_Thread* worker = nullptr; // image generation worker
    SDL_mutex* mutex = nullptr; // worker-result synchronization
    std::atomic<bool> stop_requested{false}; // cooperative worker cancellation
    std::atomic<bool> decode_on_the_go{false}; // live preview switch read at each diffusion step
    bool worker_done = false; // worker result ready under mutex
    std::string worker_message; // worker error; empty on success
    PixImage worker_image; // completed image moved back to UI under mutex
    PixImage worker_preview; // latest decoded preview moved back to UI under mutex
    float worker_progress = 0; // actual completed inference fraction
    std::string worker_stage; // current inference stage under mutex
    double worker_seconds = 0; // measured complete generation time
    float generation_progress = 0; // UI progress snapshot
    double generation_seconds = 0; // last complete generation duration
    std::string job_prompt; // immutable prompt snapshot
    PixSettings job_parameters; // unresolved model filenames and generation settings for the record
    PixSettings job_settings; // immutable generation snapshot
    PixSettings image_settings; // generation settings for the current image
    std::string image_prompt; // saved prompt for the current image
    PixImage image; // decoded/generated RGB pixels
    bool unsaved = false; // current image requires a new database record
    bool running = true; // event-loop lifetime
    bool busy = false; // model worker active
    std::string workdir; // image/database directory
    PixSettings settings; // generation values
    std::vector<PixRecord> records; // versioned database entries
    std::string dialog_message; // modal message text
    int width = UI_START_WIDTH; // current client width
    int height = UI_START_HEIGHT; // current client height
    std::string status = "Ready"; // status message

    void layout();
    void paint();
    void event(const SDL_Event &event);
    void window_event(const SDL_Event &input);
    void action(int command);
    void message(const std::string &text);
    bool read_settings();
    void write_settings();
    bool open_directory(const std::string &directory);
    bool select_image(size_t index);
    void save_image();
    void export_image(const std::string &destination);
    void generation_image(PixImage completed);
    void build_gui();
    void build_menus();
    PixUIThumbCache load_thumbnail(size_t index);
    static int generation_worker(void* context);
};

#endif
