/* This file is a part of 'Pixelina' Image Generator, Editor, and Viewer.
 *
 * (C) Dmitry 'sciloaf' Solovyev, 2026
 *
 * License: MIT
 */
#ifndef PIXELINA_UI_H
#define PIXELINA_UI_H

#include "ui_sdl.h"
#include "storage.h"
#include "pixuilib.h"
#include "inference.h"
#include <array>
#include <atomic>

#define PIXELINA_VERSION "0.4"

#define UI_CAPTION_SEPARATOR " - "
#define UI_IMAGE_TITLE "Image"
#define UI_STATUS_BUFFER 64
#define UI_IMAGE_EXTENSIONS {".jpg",".jpeg",".jpe",".png",".bmp",".tga",".psd",".gif",".hdr",".pic",".pnm",".ppm",".pgm"}
#define UI_LEFT_FRACTION 0.32f
#define UI_PROMPT_FRACTION 0.30f
#define UI_CAROUSEL_FRACTION 0.24f
#define UI_STATUS_HEIGHT (UI_LINE_HEIGHT + UI_TEXT_PADDING * 2 + UI_BORDER_WIDTH)
#define UI_SPLITTER_WIDTH 6
#define UI_PANEL_MIN 140
#define UI_WORK_TOP (UI_FRAME + UI_WINDOW_TITLE_HEIGHT + UI_MENU_HEIGHT)
#define UI_TOP_MENU_COUNT 4
#define UI_FIELD_INPUT_Y (UI_FONT_HEIGHT + UI_MARGIN + UI_BORDER_WIDTH)
#define UI_FRAME_DELAY 16
#define UI_SETTINGS_COUNT 8
#define UI_EDIT_SETTINGS_COUNT 9
#define UI_SETTINGS_PAGES 2
#define UI_SETTINGS_FOOTER (UI_FIELD_HEIGHT * 2 + UI_MARGIN)
#define UI_CONFIG_THEME "theme"
#define UI_CONFIG_BACKEND "backend"
#define UI_CONFIG_FORMAT "format"
#define UI_CONFIG_AUTO_SEED "auto_seed"
#define UI_CONFIG_DECODE "decode_on_the_go"
#define UI_CONFIG_SCALED "scaled_view"
#define UI_CONFIG_GLOBAL_DB "global_db"
#define UI_CONFIG_PROMPT "prompt"
#define UI_CONFIG_DIRECTORY "directory"
#define UI_CONFIG_BROWSER_DIRECTORY "browser_directory"
#define UI_CONFIG_TAB "settings_tab"
#define UI_CONFIG_GENERATE_PREFIX "generate_"
#define UI_CONFIG_EDIT_PREFIX "edit_"
#define UI_CONFIG_WINDOW_X "window_x"
#define UI_CONFIG_WINDOW_Y "window_y"
#define UI_CONFIG_WINDOW_WIDTH "window_width"
#define UI_CONFIG_WINDOW_HEIGHT "window_height"
#define UI_CONFIG_MAXIMIZED "window_maximized"
#define UI_SETTINGS_MIN_WIDTH 240
#define UI_INVALID_EDIT_NUMBERS "Seed, dimensions, steps and reference resolution must be valid numbers."
#define UI_SETTINGS_ROW (UI_FIELD_INPUT_Y + UI_FIELD_HEIGHT + UI_TEXT_PADDING + UI_BORDER_WIDTH)
#define UI_BROWSE_BUTTON_WIDTH 32
#define UI_MODEL_FIRST UI_DIFFUSER_FIELD
#define UI_GENERATE_TEXT "Generate"
#define UI_INVALID_NUMBERS "Seed, dimensions, steps and shift must be valid numbers."
#define UI_STOPPED_TEXT "Generation stopped"
#define UI_READY_TEXT "Ready"
#define UI_SPIN_BUTTON_WIDTH UI_SCROLLBAR

enum PixelinaCommand {
    CMD_OPEN = 1,CMD_SAVE,CMD_SAVE_AS,CMD_QUIT,CMD_GENERATE,CMD_SCALED,CMD_ABOUT,
    CMD_JPG,CMD_PNG,CMD_CPU,CMD_VULKAN,CMD_DECODE,CMD_OPEN_IMAGE,CMD_CLEAR,CMD_AUTO_SEED,CMD_GLOBAL_DB,CMD_MULTI_REFERENCE,CMD_CHAT,CMD_BOOKMARK,CMD_UNBOOKMARK,CMD_FULLSCREEN
};

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

enum PixelinaPage {
    UI_GENERATE_PAGE,UI_EDIT_PAGE
};

enum PixelinaImagePage {
    UI_IMAGES_PAGE,UI_BOOKMARKS_PAGE
};

enum PixelinaEditField {
    UI_EDIT_SEED,UI_EDIT_WIDTH,UI_EDIT_HEIGHT,UI_EDIT_STEPS,UI_EDIT_RESOLUTION,
    UI_EDIT_DIFFUSER,UI_EDIT_ENCODER,UI_EDIT_VISION,UI_EDIT_VAE
};

struct PixJob {
    PixRecord parameters; // unresolved immutable result metadata
    PixSettings generate_settings; // resolved Generate execution settings
    PixEditSettings edit_settings; // resolved Edit execution settings
    std::vector<PixImage> references; // owned complete reference planes
};

using PixJobInference = std::function<PixImage(const PixJob&,const InferenceProgress&,int&)>;

class PixelinaForm : public PixUIForm
{
public:
    explicit PixelinaForm(PixUICanvas &context);
    void create_controls();
    void layout() override;
    void draw() override;

    std::string caption = PIXELINA_NAME; // application window caption
    bool fullscreen = false; // expand only the image widget to the client bounds
    PixUIMemo* prompt; // prompt editor owned by its group
    std::array<PixUIEdit*,UI_SETTINGS_COUNT> fields; // owned settings editors
    std::array<PixUIButton*,UI_SETTINGS_COUNT> browse = {}; // model selection actions
    PixUILabel* preferences; // backend and format label
    PixUIButton* generate; // generation action
    PixUICarousel* carousel; // complete saved-image widget
    PixUICarousel* bookmarks; // complete global bookmarked-image widget
    PixTab* image_tabs; // Images and Bookmarks pages above the current image
    PixUIImageView* image; // complete image widget
    PixUIStatusBar* status; // complete status widget
    PixTab* tabs; // owned settings tab control
    std::array<PixTabGroup*,UI_SETTINGS_PAGES> pages; // owned settings pages
    std::array<PixUIEdit*,UI_EDIT_SETTINGS_COUNT> edit_fields; // owned Edit editors
    std::array<PixUIButton*,UI_EDIT_SETTINGS_COUNT> edit_browse = {}; // Edit model selection actions
    std::array<PixUIButton*,UI_SETTINGS_PAGES> random_seed = {}; // seed randomization actions
    std::vector<PixUIButton*> dimension_buttons; // dimension increment and decrement actions
    float left_fraction = UI_LEFT_FRACTION,prompt_fraction = UI_PROMPT_FRACTION,carousel_fraction = UI_CAROUSEL_FRACTION; // persistent pane ratios

private:
    friend class Pixelina;

    PixUIPanel* menu_background; // full-width menu background
    std::array<PixUIGroupBox*,UI_GROUP_COUNT> groups; // workspace containers
    std::array<PixUISplitter*,UI_SPLITTER_COUNT> splitters; // owned pane dividers
};

class Pixelina
{
public:
    Pixelina() = default;
    ~Pixelina();
    bool initialize(const std::string &directory, bool png, std::string &error, bool default_directory = false, bool use_saved_directory = false);
    int run();

private:
    PixUISDLCanvas canvas; // shared renderer and control capture
    PixelinaForm main_form{canvas}; // owns persistent workspace controls
    PixUIMenu* menus = nullptr; // borrowed menu root owned by main_form
    SDL_Renderer* renderer = nullptr; // SDL renderer owned by the application
    SDL_Window* window = nullptr; // outer resizable window
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
    uint64_t generation_started = 0; // UI clock at the start of the current job
    PixJob job; // frozen settings, prompt and references read by the worker
    PixRecord image_record; // frozen generation metadata of the displayed image
    bool image_has_record = false; // current picture has generation metadata
    std::string image_source_path; // absolute current picture provenance, empty for temporary results
    std::vector<std::string> reference_paths; // accepted ordered references retained until Clear
    std::string launch_directory; // fixed model-path resolution directory
    PixEditSettings edit_settings; // independent Edit page values
    PixImage image; // complete current RGB and optional alpha planes
    bool unsaved = false; // current image requires a new database record
    bool running = true; // event-loop lifetime
    bool busy = false; // model worker active
    bool auto_seed = true; // advance the active page seed after starting generation
    bool global_db = false; // select the user database instead of the directory database
    bool initialized = false; // save configuration only after successful initialization
    std::string user_directory; // persistent configuration and global database directory
    std::string theme_path; // configured theme filename retained when saving settings
    std::string workdir; // selected output image directory
    PixSettings settings; // generation values
    std::vector<PixRecord> records; // versioned database entries
    std::vector<std::string> images; // supported image files in the current directory
    std::vector<std::string> bookmarks; // global absolute bookmarked image paths
    int bookmark_selection = -1; // explicitly selected bookmark eligible for removal
    std::string dialog_message; // modal message text
    int width = UI_START_WIDTH; // current client width
    int height = UI_START_HEIGHT; // current client height
    std::string status = UI_READY_TEXT; // status message

    void layout();
    void paint();
    void event(const SDL_Event &event);
    void window_event(const SDL_Event &input);
    void action(int command);
    void message(const std::string &text);
    bool read_settings();
    void write_settings();
    bool read_edit_settings();
    void write_edit_settings();
    bool capture_job();
    void randomize_seed(size_t page);
    void reference_resolution();
    void generation_job(const PixJobInference &inference);
    bool open_image(const std::string &path);
    bool open_directory(const std::string &directory);
    bool select_image(size_t index);
    void bookmark_image(bool remove);
    void save_image();
    void export_image(const std::string &destination);
    void generation_image(PixImage completed);
    void build_gui();
    void build_menus();
    void restore_configuration(const std::map<std::string,std::string> &values);
    void save_configuration();
    void switch_database();
    PixUIThumbCache load_thumbnail(const std::string &path);
    static int generation_worker(void* context);
};

#endif
