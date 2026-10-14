/* This file is a part of 'Pixelina' Image Generator, Editor, and Viewer.
 *
 * (C) Dmitry 'sciloaf' Solovyev, 2026
 *
 * License: MIT
 */
#include "ui.h"
#include "inference.h"
#include "qwen_image.h"
#include "backend_vulkan.h"
#include "../assets/pixelina_icon.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>

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

void PixUISDLCanvas::window_command(PixUIWindowCommand command, const PixUIRect &rect)
{
    switch (command) {
    case PIXUI_RESTORE:
        SDL_RestoreWindow(window);
        break;
    case PIXUI_MINIMIZE:
        SDL_MinimizeWindow(window);
        break;
    case PIXUI_MAXIMIZE:
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
        case SDLK_F3:
            result.key = PIXUI_F3;
            break;
        case SDLK_F4:
            result.key = PIXUI_F4;
            break;
        case SDLK_F5:
            result.key = PIXUI_F5;
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

static const char* field_labels[UI_SETTINGS_COUNT] = {"Seed","Width","Height","Steps","Scheduler shift","Diffuser GGUF","Text encoder GGUF","VAE safetensors"};
static const char* edit_field_labels[UI_EDIT_SETTINGS_COUNT] = {"Seed","Width","Height","Steps","Reference resolution","Diffuser GGUF","Language encoder GGUF","Vision projector GGUF","VAE safetensors"};
bool Pixelina::initialize(const std::string &directory, bool png, std::string &error)
{
    char* cwd = getcwd(nullptr,0);
    if (!cwd) {
        error = "Cannot determine the launch directory.";
        return false;
    }
    launch_directory = cwd;
    std::free(cwd);
    files.last_directory = launch_directory;
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER)) {
        error = SDL_GetError();
        return false;
    }
    window = SDL_CreateWindow(PIXELINA_NAME,SDL_WINDOWPOS_CENTERED,SDL_WINDOWPOS_CENTERED,width,height,SDL_WINDOW_RESIZABLE | SDL_WINDOW_BORDERLESS);
    if (!window) {
        error = SDL_GetError();
        return false;
    }
    canvas.window = window;
    main_form.bounds = {0,0,width,height};
    SDL_SetWindowMinimumSize(window,UI_MIN_WIDTH,UI_MIN_HEIGHT);
    renderer = SDL_CreateRenderer(window,-1,SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    if (!renderer) renderer = SDL_CreateRenderer(window,-1,SDL_RENDERER_SOFTWARE);
    mutex = SDL_CreateMutex();
    if (!renderer || !mutex) {
        error = SDL_GetError();
        return false;
    }
    PixImage icon;
    if (!image_decode(pixelina_icon,sizeof(pixelina_icon) - 1,icon,error)) return false;
    SDL_Surface* surface = SDL_CreateRGBSurfaceFrom(icon.rgb.data(),icon.width,icon.height,24,icon.width * 3,SDL_BYTEORDER == SDL_LIL_ENDIAN? 0x0000ff : 0xff0000,0x00ff00,SDL_BYTEORDER == SDL_LIL_ENDIAN? 0xff0000 : 0x0000ff,0);
    if (!surface) {
        error = SDL_GetError();
        return false;
    }
    SDL_SetWindowIcon(window,surface);
    app_icon = SDL_CreateTextureFromSurface(renderer,surface);
    SDL_FreeSurface(surface);
    if (!app_icon) {
        error = SDL_GetError();
        return false;
    }
    canvas.renderer = renderer;
    settings.format = png? PIXELINA_PNG : PIXELINA_JPG;
    build_gui();
    write_settings();
    write_edit_settings();
    bool open_file = files.path_kind(directory) == PIXUI_PATH_FILE;
    if (!open_directory(open_file? launch_directory : directory)) {
        error = dialog_message;
        return false;
    }
    if (open_file && !open_image(directory)) {
        error = dialog_message;
        return false;
    }
    layout();
    SDL_StartTextInput();
    return true;
}

Pixelina::~Pixelina()
{
    stop_requested = true;
    if (worker) SDL_WaitThread(worker,nullptr);
    canvas.release_capture();
    main_form.clear();
    if (current) SDL_DestroyTexture(current);
    if (app_icon) SDL_DestroyTexture(app_icon);
    if (mutex) SDL_DestroyMutex(mutex);
    if (renderer) SDL_DestroyRenderer(renderer);
    if (window) SDL_DestroyWindow(window);
    SDL_Quit();
}

bool Pixelina::open_directory(const std::string &directory)
{
    char* canonical = realpath(directory.c_str(),nullptr);
    if (!canonical) {
        message(UI_DIRECTORY_ERROR + directory);
        return false;
    }
    std::string actual = canonical;
    std::free(canonical);
    std::vector<PixDirEntry> listing;
    std::vector<PixRecord> loaded;
    std::string error;
    if (!directory_list(actual,listing,error) || !database_load(actual,loaded,error)) {
        message(error);
        return false;
    }
    workdir = actual;
    records.swap(loaded);
    main_form.carousel->items_changed(records.size(),true);
    status = "Workdir: " + workdir;
    return true;
}

void Pixelina::layout()
{
    main_form.bounds = {0,0,width,height};
    main_form.caption = PIXELINA_NAME UI_CAPTION_SEPARATOR + workdir;
    main_form.prompt->enabled = !busy;
    for (PixUIEdit* field : main_form.fields) field->enabled = !busy;
    for (PixUIButton* button : main_form.browse) {
        if (button) button->enabled = !busy;
    }
    for (PixUIEdit* field : main_form.edit_fields) field->enabled = !busy;
    for (PixUIButton* button : main_form.edit_browse) {
        if (button) button->enabled = !busy;
    }
    main_form.tabs->selection_enabled = !busy;
    for (PixUIButton* button : main_form.random_seed) button->enabled = !busy;
    main_form.generate->caption = busy? (stop_requested? UI_STOPPING_TEXT : "Stop [F5]") : "Generate [F5]";
    main_form.preferences->caption = settings.backend + " / " + settings.format;
    main_form.carousel->enabled = !busy;
    main_form.image->set_image(current,image.width,image.height);
    char speed[UI_STATUS_BUFFER];
    if (generation_seconds > 0) std::snprintf(speed,sizeof(speed)," | %.2fs (%.3f img/s)",generation_seconds,1.0 / generation_seconds);
    else std::snprintf(speed,sizeof(speed)," | speed: --");
    main_form.status->text = std::to_string(records.size()) + " images" + speed + " | " + status;
    main_form.status->busy = busy;
    main_form.status->progress = generation_progress;
    main_form.layout();
}

void Pixelina::write_settings()
{
    std::ostringstream shift;
    shift << settings.shift;
    std::string values[] = {std::to_string(settings.seed),std::to_string(settings.width),std::to_string(settings.height),std::to_string(settings.steps),shift.str(),settings.diffuser,settings.encoder,settings.vae};
    for (unsigned i = 0; i < UI_SETTINGS_COUNT; i++) main_form.fields[i]->set_text(values[i]);
}

bool Pixelina::read_settings()
{
    PixSettings candidate = settings;
    for (unsigned i = 0; i < UI_SHIFT_FIELD; i++) {
        uint64_t value;
        if (!parse_unsigned(main_form.fields[i]->model.text,value) || (i && value > INT32_MAX)) {
            message(UI_INVALID_NUMBERS);
            return false;
        }
        switch (i) {
        case UI_SEED_FIELD:
            candidate.seed = value;
            break;
        case UI_WIDTH_FIELD:
            candidate.width = int(value);
            break;
        case UI_HEIGHT_FIELD:
            candidate.height = int(value);
            break;
        case UI_STEPS_FIELD:
            candidate.steps = int(value);
            break;
        }
    }
    if (!parse_float(main_form.fields[UI_SHIFT_FIELD]->model.text,candidate.shift)) {
        message(UI_INVALID_NUMBERS);
        return false;
    }
    candidate.diffuser = main_form.fields[UI_DIFFUSER_FIELD]->model.text;
    candidate.encoder = main_form.fields[UI_ENCODER_FIELD]->model.text;
    candidate.vae = main_form.fields[UI_VAE_FIELD]->model.text;
    std::string error;
    if (!settings_valid(candidate,error)) {
        message(error);
        return false;
    }
    settings = candidate;
    return true;
}

void Pixelina::write_edit_settings()
{
    std::string values[] = {std::to_string(edit_settings.seed),std::to_string(edit_settings.width),std::to_string(edit_settings.height),std::to_string(edit_settings.steps),std::to_string(edit_settings.reference_resolution),edit_settings.diffuser,edit_settings.encoder,edit_settings.vision,edit_settings.vae};
    for (size_t i = 0; i < UI_EDIT_SETTINGS_COUNT; i++) main_form.edit_fields[i]->set_text(values[i]);
}

bool Pixelina::read_edit_settings()
{
    PixEditSettings candidate = edit_settings;
    for (size_t i = 0; i < UI_EDIT_DIFFUSER; i++) {
        uint64_t value;
        if (!parse_unsigned(main_form.edit_fields[i]->model.text,value) || (i && value > INT32_MAX)) {
            message(UI_INVALID_EDIT_NUMBERS);
            return false;
        }
        switch (i) {
        case UI_EDIT_SEED:
            candidate.seed = value;
            break;
        case UI_EDIT_WIDTH:
            candidate.width = int(value);
            break;
        case UI_EDIT_HEIGHT:
            candidate.height = int(value);
            break;
        case UI_EDIT_STEPS:
            candidate.steps = int(value);
            break;
        case UI_EDIT_RESOLUTION:
            candidate.reference_resolution = int(value);
            break;
        }
    }
    candidate.diffuser = main_form.edit_fields[UI_EDIT_DIFFUSER]->model.text;
    candidate.encoder = main_form.edit_fields[UI_EDIT_ENCODER]->model.text;
    candidate.vision = main_form.edit_fields[UI_EDIT_VISION]->model.text;
    candidate.vae = main_form.edit_fields[UI_EDIT_VAE]->model.text;
    candidate.backend = settings.backend;
    candidate.format = settings.format;
    std::string error;
    if (!edit_settings_valid(candidate,error)) {
        message(error);
        return false;
    }
    edit_settings = candidate;
    return true;
}

bool Pixelina::capture_job()
{
    PixJob candidate;
    candidate.parameters.prompt = main_form.prompt->model.text;
    if (main_form.tabs->selected() == UI_EDIT_PAGE) {
        if (!read_edit_settings()) return false;
        if (!image_valid(image)) {
            message("Open, select or generate a complete image before editing.");
            return false;
        }
        candidate.parameters.model = PIX_MODEL_QWEN_IMAGE;
        candidate.parameters.edit_settings = edit_settings;
        candidate.edit_settings = edit_settings;
        candidate.references.push_back(image);
        if (!image_source_path.empty()) candidate.parameters.references.push_back(image_source_path);
        candidate.edit_settings.diffuser = path_join(launch_directory,edit_settings.diffuser);
        candidate.edit_settings.encoder = path_join(launch_directory,edit_settings.encoder);
        candidate.edit_settings.vision = path_join(launch_directory,edit_settings.vision);
        candidate.edit_settings.vae = path_join(launch_directory,edit_settings.vae);
    } else {
        if (!read_settings()) return false;
        candidate.parameters.settings = settings;
        candidate.generate_settings = settings;
        candidate.generate_settings.diffuser = path_join(launch_directory,settings.diffuser);
        candidate.generate_settings.encoder = path_join(launch_directory,settings.encoder);
        candidate.generate_settings.vae = path_join(launch_directory,settings.vae);
    }
    job = std::move(candidate);
    return true;
}

void Pixelina::randomize_seed(size_t page)
{
    PixUIEdit* field = page == UI_GENERATE_PAGE? main_form.fields[UI_SEED_FIELD] : main_form.edit_fields[UI_EDIT_SEED];
    uint64_t previous = 0;
    parse_unsigned(field->model.text,previous);
    static std::mt19937_64 random(SDL_GetPerformanceCounter() ^ uint64_t(getpid()));
    uint64_t seed;
    do seed = random();
    while (seed == previous);
    if (page == UI_GENERATE_PAGE) settings.seed = seed;
    else edit_settings.seed = seed;
    field->set_text(std::to_string(seed));
}

void Pixelina::reference_resolution()
{
    double side = std::sqrt(double(image.width) * image.height);
    edit_settings.reference_resolution = std::max(QWEN_IMAGE_MULTIPLE,int(std::round(side / QWEN_IMAGE_MULTIPLE)) * QWEN_IMAGE_MULTIPLE);
    main_form.edit_fields[UI_EDIT_RESOLUTION]->set_text(std::to_string(edit_settings.reference_resolution));
}

bool Pixelina::open_image(const std::string &path)
{
    if (busy) return false;
    std::string actual = files.normalize(path_join(launch_directory,path));
    PixImage loaded;
    std::string error;
    if (files.path_kind(actual) != PIXUI_PATH_FILE || !image_load(actual,loaded,error)) {
        message(error.empty()? "Cannot open image: " + actual : error);
        return false;
    }
    SDL_Texture* texture = SDL_CreateTexture(renderer,SDL_PIXELFORMAT_RGB24,SDL_TEXTUREACCESS_STATIC,loaded.width,loaded.height);
    if (!texture || SDL_UpdateTexture(texture,nullptr,loaded.rgb.data(),loaded.width * 3)) {
        if (texture) SDL_DestroyTexture(texture);
        message(SDL_GetError());
        return false;
    }
    if (current) SDL_DestroyTexture(current);
    current = texture;
    image = std::move(loaded);
    image_record = PixRecord();
    image_has_record = false;
    image_source_path = actual;
    unsaved = false;
    main_form.carousel->select(-1);
    reference_resolution();
    status = actual;
    return true;
}

void Pixelina::message(const std::string &value)
{
    dialog_message = value;
    main_form.present(std::make_unique<PixUIMessageDialog>(canvas,value,PIXELINA_NAME));
}

bool Pixelina::select_image(size_t index)
{
    if (index >= records.size() || busy) return false;
    PixImage loaded;
    std::string error;
    if (!image_load(path_join(workdir,records[index].filename),loaded,error)) {
        message(error);
        return false;
    }
    SDL_Texture* texture = SDL_CreateTexture(renderer,SDL_PIXELFORMAT_RGB24,SDL_TEXTUREACCESS_STATIC,loaded.width,loaded.height);
    if (!texture || SDL_UpdateTexture(texture,nullptr,loaded.rgb.data(),loaded.width * 3)) {
        if (texture) SDL_DestroyTexture(texture);
        message(SDL_GetError());
        return false;
    }
    if (current) SDL_DestroyTexture(current);
    current = texture;
    image = std::move(loaded);
    image_record = records[index];
    image_has_record = true;
    image_source_path = files.normalize(path_join(workdir,image_record.filename));
    main_form.tabs->select(image_record.model == PIX_MODEL_Z_IMAGE? UI_GENERATE_PAGE : UI_EDIT_PAGE);
    if (image_record.model == PIX_MODEL_Z_IMAGE) {
        settings = image_record.settings;
        write_settings();
    } else {
        edit_settings = image_record.edit_settings;
        settings.backend = edit_settings.backend;
        settings.format = edit_settings.format;
        write_edit_settings();
    }
    main_form.prompt->set_text(image_record.prompt);
    reference_resolution();
    unsaved = false;
    status = records[index].filename;
    return true;
}

void Pixelina::save_image()
{
    if (busy) return;
    if (!image_has_record) {
        message(image.rgb.empty()? "Generate an image first." : "This image has no generation record. Use Save as to export it.");
        return;
    }
    if (!unsaved) {
        message("This image is already in the database. Use Save as to export it.");
        return;
    }
    PixRecord record = image_record;
    if (record.model == PIX_MODEL_Z_IMAGE) record.settings.format = settings.format;
    else record.edit_settings.format = settings.format;
    std::string error;
    if (!storage_save(workdir,record,image,error)) {
        message(error);
        return;
    }
    image_record = record;
    image_source_path = files.normalize(path_join(workdir,record.filename));
    records.push_back(record);
    main_form.carousel->items_changed(records.size());
    main_form.carousel->select(int(records.size()) - 1);
    unsaved = false;
    status = "Saved " + record.filename;
}

void Pixelina::export_image(const std::string &destination)
{
    std::string error;
    if (!image_save(destination,image,error)) message(error);
    else {
        main_form.dismiss_dialog();
        status = "Exported " + destination;
    }
}

int Pixelina::generation_worker(void* context)
{
    Pixelina* app = static_cast<Pixelina*>(context);
    app->generation_job({});
    return 0;
}

void Pixelina::generation_job(const PixJobInference &inference)
{
    std::string error;
    PixImage result;
    Uint64 start = SDL_GetPerformanceCounter();
    auto progress = [&](float fraction, const std::string &stage) {
        if (stop_requested) return false;
        SDL_LockMutex(mutex);
        worker_progress = fraction;
        worker_stage = stage;
        SDL_UnlockMutex(mutex);
        return true;
    };
    int code = 0;
    auto decode_enabled = [&]() { return decode_on_the_go.load(); };
    auto preview = [&](PixImage decoded) {
        SDL_LockMutex(mutex);
        worker_preview = std::move(decoded);
        SDL_UnlockMutex(mutex);
    };
    if (inference) result = inference(job,progress,code);
    else if (job.parameters.model == PIX_MODEL_QWEN_IMAGE) result = qwen_image_edit(job.edit_settings,job.references,job.parameters.prompt,progress,code);
    else result = inference_generate(job.generate_settings,job.parameters.prompt,progress,code,decode_enabled,preview);
    if (!code && !image_valid(result)) code = INFERENCE_ERROR_NUMERICAL;
    if (stop_requested) error = UI_STOPPED_TEXT;
    else switch (code) {
    case INFERENCE_ERROR_SETTINGS:
        if (job.parameters.model == PIX_MODEL_Z_IMAGE) settings_valid(job.generate_settings,error);
        else edit_settings_valid(job.edit_settings,error);
        if (error.empty()) error = worker_stage;
        break;
    case INFERENCE_ERROR_MODEL:
        error = "Cannot load models or unsupported model tensors/tokenizer.";
        if (job.parameters.model == PIX_MODEL_QWEN_IMAGE) error += "\n" + worker_stage;
        break;
    case INFERENCE_ERROR_PROMPT:
        error = "Invalid prompt, context exceeded, or text encoding failed.";
        break;
    case INFERENCE_ERROR_NUMERICAL:
        error = "Generation failed during diffusion or VAE decoding.";
        break;
    case INFERENCE_ERROR_VULKAN:
        error = "Vulkan backend unavailable or image exceeds device limits.";
        break;
    case INFERENCE_ERROR_STOPPED:
        error = UI_STOPPED_TEXT;
        break;
    case INFERENCE_ERROR_MEMORY:
        error = worker_stage;
        break;
    default:
        break;
    }
    double seconds = double(SDL_GetPerformanceCounter() - start) / SDL_GetPerformanceFrequency();
    SDL_LockMutex(mutex);
    worker_message = error;
    worker_image = std::move(result);
    worker_seconds = seconds;
    worker_done = true;
    SDL_UnlockMutex(mutex);
}

void Pixelina::action(int command)
{
    if (menus) menus->dismiss();
    if (command == CMD_QUIT) {
        main_form.command(PIXUI_CLOSE);
        return;
    }
    if (command == CMD_ABOUT) {
        int width,height;
        SDL_QueryTexture(app_icon,nullptr,nullptr,&width,&height);
        std::vector<std::string> lines = PIXELINA_ABOUT;
        main_form.present(std::make_unique<PixUIAboutDialog>(canvas,UI_ABOUT_PREFIX PIXELINA_NAME,app_icon,width,height,lines));
        return;
    }
    if (busy && command == CMD_GENERATE) {
        stop_requested = true;
        status = UI_STOPPING_TEXT;
        return;
    }
    if (command == CMD_DECODE) {
        decode_on_the_go = !decode_on_the_go.load();
        return;
    }
    if (command == CMD_AUTO_SEED) {
        auto_seed = !auto_seed;
        return;
    }
    if (busy) return;
    switch (command) {
    case CMD_CLEAR:
        if (current) SDL_DestroyTexture(current);
        current = nullptr;
        image = PixImage();
        image_record = PixRecord();
        job = PixJob();
        worker_image = PixImage();
        worker_preview = PixImage();
        image_has_record = false;
        image_source_path.clear();
        unsaved = false;
        generation_progress = 0;
        generation_seconds = 0;
        main_form.image->set_image(nullptr,0,0);
        main_form.carousel->select(-1);
        status = UI_READY_TEXT;
        break;
    case CMD_OPEN: {
        auto browser = std::make_unique<PixUIFileBrowser>(canvas,files);
        browser->dir_selector = true;
        browser->title = "Open directory";
        browser->error = [this](const std::string &error) { status = error; };
        browser->selected = [this](const std::string &path) { return open_directory(path); };
        browser->refresh();
        main_form.present(std::move(browser));
        break;
    }
    case CMD_OPEN_IMAGE: {
        auto browser = std::make_unique<PixUIFileBrowser>(canvas,files);
        browser->title = "Open image";
        browser->error = [this](const std::string &error) { status = error; };
        browser->selected = [this](const std::string &path) { return open_image(path); };
        browser->refresh();
        main_form.present(std::move(browser));
        break;
    }
    case CMD_SAVE:
        save_image();
        break;
    case CMD_SAVE_AS:
        if (image.rgb.empty()) message("Generate or select an image first.");
        else {
            auto browser = std::make_unique<PixUIFileBrowser>(canvas,files,path_join(files.last_directory,settings.format == PIXELINA_PNG? "image.png" : "image.jpg"));
            browser->title = UI_SAVE_AS_TEXT;
            browser->accept_caption = UI_SAVE_TEXT;
            browser->path_label = "Destination (.jpg or .png):";
            browser->hint_text = "Exports the current image without a database entry.";
            browser->show_listing = false;
            browser->selected = [this](const std::string &path) {
                std::string destination = path_join(workdir,path);
                if (files.path_kind(destination) != PIXUI_PATH_MISSING) {
                    auto confirmation = std::make_unique<PixUIMessageDialog>(canvas,"Overwrite existing file?\n" + destination,PIXELINA_NAME,"Overwrite",UI_CANCEL_TEXT);
                    confirmation->accepted = [this,destination]() { export_image(destination); };
                    main_form.present(std::move(confirmation));
                } else export_image(destination);
                return true;
            };
            main_form.present(std::move(browser));
        }
        break;
    case CMD_SCALED:
        main_form.image->scaled = !main_form.image->scaled;
        break;
    case CMD_JPG:
        settings.format = PIXELINA_JPG;
        break;
    case CMD_PNG:
        settings.format = PIXELINA_PNG;
        break;
    case CMD_CPU:
        settings.backend = PIXELINA_CPU;
        break;
    case CMD_VULKAN:
        settings.backend = PIXELINA_VULKAN;
        break;
    case CMD_GENERATE: {
        if (!capture_job()) return;
        stop_requested = false;
        worker_done = false;
        worker_progress = 0;
        generation_progress = 0;
        worker_stage = "Loading models";
        worker_message.clear();
        worker_preview = PixImage();
        busy = true;
        status = "Generating";
        worker = SDL_CreateThread(generation_worker,PIXELINA_NAME " generation",this);
        if (!worker) {
            busy = false;
            message(SDL_GetError());
        } else if (auto_seed) randomize_seed(main_form.tabs->selected());
        break;
    }
    default:
        break;
    }
}

PixUIThumbCache Pixelina::load_thumbnail(size_t index)
{
    PixUIThumbCache value;
    PixImage full;
    std::string error;
    if (image_load(path_join(workdir,records[index].filename),full,error)) {
        PixImage reduced = image_thumbnail(full);
        value.width = reduced.width;
        value.height = reduced.height;
        value.texture = SDL_CreateTexture(renderer,SDL_PIXELFORMAT_RGB24,SDL_TEXTUREACCESS_STATIC,reduced.width,reduced.height);
        if (value.texture && SDL_UpdateTexture(static_cast<SDL_Texture*>(value.texture),nullptr,reduced.rgb.data(),reduced.width * 3)) {
            SDL_DestroyTexture(static_cast<SDL_Texture*>(value.texture));
            value.texture = nullptr;
        }
    }
    value.missing = !value.texture;
    return value;
}

void Pixelina::build_menus()
{
    menus = main_form.create_menu();
    const char* labels[] = {"File","Edit","View","Help"};
    std::array<PixUIMenu*,UI_TOP_MENU_COUNT> popups;
    for (int i = 0; i < UI_TOP_MENU_COUNT; i++) {
        PixUIMenuItem* heading = menus->add(labels[i]);
        heading->bounds = {i * UI_TOP_MENU_WIDTH,UI_FRAME + UI_WINDOW_TITLE_HEIGHT,UI_TOP_MENU_WIDTH,UI_MENU_HEIGHT};
        popups[i] = menus->child(*heading);
    }
    PixUIMenu* file = popups[0];
    file->add("Open dir",[this]() { action(CMD_OPEN); })->bind_shortcut('o',PIXUI_CTRL,"Ctrl+O");
    file->add("Open image",[this]() { action(CMD_OPEN_IMAGE); })->bind_shortcut('o',PIXUI_CTRL | PIXUI_SHIFT,"Ctrl+Shift+O");
    file->add(UI_SAVE_TEXT,[this]() { action(CMD_SAVE); })->bind_shortcut('s',PIXUI_CTRL,"Ctrl+S");
    file->add(UI_SAVE_AS_TEXT,[this]() { action(CMD_SAVE_AS); })->bind_shortcut('s',PIXUI_CTRL | PIXUI_SHIFT,"Ctrl+Shift+S");
    file->add("Quit",[this]() { action(CMD_QUIT); })->bind_shortcut('q',PIXUI_CTRL,"Ctrl+Q");
    PixUIMenu* edit = popups[1];
    edit->add("Clear",[this]() { action(CMD_CLEAR); });
    PixUIMenuItem* generate = edit->add(UI_GENERATE_TEXT,[this]() { action(CMD_GENERATE); });
    generate->bind_shortcut(PIXUI_F5,0,"F5");
    generate->caption_state = [this]() { return busy? UI_STOP_TEXT : UI_GENERATE_TEXT; };
    PixUIMenu* preferences = edit->child(*edit->add("Preferences"));
    PixUIMenuItem* seed = preferences->add("Auto-seed",[this]() { action(CMD_AUTO_SEED); });
    seed->checkable = true;
    seed->checked_state = [this]() { return auto_seed; };
    PixUIMenu* format = preferences->child(*preferences->add("File format"));
    PixUIMenuItem* jpg = format->add(PIXELINA_JPG,[this]() { action(CMD_JPG); });
    jpg->checkable = true;
    jpg->checked_state = [this]() { return settings.format == PIXELINA_JPG; };
    PixUIMenuItem* png = format->add(PIXELINA_PNG,[this]() { action(CMD_PNG); });
    png->checkable = true;
    png->checked_state = [this]() { return settings.format == PIXELINA_PNG; };
    PixUIMenu* backend = preferences->child(*preferences->add("Backend"));
    PixUIMenuItem* cpu = backend->add(PIXELINA_CPU,[this]() { action(CMD_CPU); });
    cpu->checkable = true;
    cpu->checked_state = [this]() { return settings.backend == PIXELINA_CPU; };
    PixUIMenuItem* vulkan = backend->add(PIXELINA_VULKAN,[this]() { action(CMD_VULKAN); });
    std::string description;
    vulkan->enabled = vulkan_available(description);
    vulkan->checkable = true;
    vulkan->checked_state = [this]() { return settings.backend == PIXELINA_VULKAN; };
    PixUIMenuItem* scaled = popups[2]->add("Scaled view",[this]() { action(CMD_SCALED); });
    scaled->bind_shortcut(PIXUI_F3,0,"F3");
    scaled->checkable = true;
    scaled->checked_state = [this]() { return main_form.image->scaled; };
    PixUIMenuItem* decode = popups[2]->add("Decode on-the-go",[this]() { action(CMD_DECODE); });
    decode->checkable = true;
    decode->checked_state = [this]() { return decode_on_the_go.load(); };
    popups[3]->add(UI_ABOUT_PREFIX PIXELINA_NAME,[this]() { action(CMD_ABOUT); });
    main_form.add_system_menu();
}

void Pixelina::build_gui()
{
    main_form.close = [this]() { running = false; };
    build_menus();
    main_form.generate->activate = [this]() { action(CMD_GENERATE); };
    main_form.carousel->caption = [this](size_t index) { return records[index].filename; };
    main_form.carousel->load = [this](size_t index) { return load_thumbnail(index); };
    main_form.carousel->release = [](void* texture) { SDL_DestroyTexture(static_cast<SDL_Texture*>(texture)); };
    main_form.carousel->cache_limit = PIXELINA_THUMB_CACHE;
    main_form.carousel->selected = [this](size_t index) { return select_image(index); };
    for (size_t page = 0; page < UI_SETTINGS_PAGES; page++) {
        main_form.random_seed[page]->activate = [this,page]() {
            if (!busy) randomize_seed(page);
        };
        size_t count = page == UI_GENERATE_PAGE? UI_SETTINGS_COUNT : UI_EDIT_SETTINGS_COUNT;
        for (size_t i = UI_MODEL_FIRST; i < count; i++) {
            PixUIEdit* field = page == UI_GENERATE_PAGE? main_form.fields[i] : main_form.edit_fields[i];
            PixUIButton* button = page == UI_GENERATE_PAGE? main_form.browse[i] : main_form.edit_browse[i];
            button->activate = [this,field]() {
                if (busy) return;
                auto browser = std::make_unique<PixUIFileBrowser>(canvas,files);
                browser->title = "Open model file";
                browser->error = [this](const std::string &error) { status = error; };
                browser->selected = [this,field](const std::string &path) {
                    if (files.path_kind(path) != PIXUI_PATH_FILE) {
                        status = "Cannot open file: " + path;
                        return false;
                    }
                    field->set_text(path);
                    main_form.focus_control(*field);
                    return true;
                };
                browser->refresh();
                main_form.present(std::move(browser));
            };
        }
    }
}

void Pixelina::paint()
{
    layout();
    main_form.draw();
    SDL_RenderPresent(renderer);
}

void Pixelina::event(const SDL_Event &native)
{
    layout();
    if (native.type == SDL_QUIT) {
        main_form.command(PIXUI_CLOSE);
        return;
    }
    if (native.type == SDL_WINDOWEVENT) {
        window_event(native);
        return;
    }
    PixUIEvent input = canvas.translate(native);
    canvas.event(main_form,input);
    layout();
}

void Pixelina::window_event(const SDL_Event &input)
{
    switch (input.window.event) {
    case SDL_WINDOWEVENT_FOCUS_GAINED:
        main_form.active = true;
        break;
    case SDL_WINDOWEVENT_FOCUS_LOST:
        canvas.release_capture();
        main_form.active = false;
        break;
    case SDL_WINDOWEVENT_SIZE_CHANGED:
        width = input.window.data1;
        height = input.window.data2;
        layout();
        break;
    default:
        break;
    }
}

void Pixelina::generation_image(PixImage completed)
{
    SDL_Texture* texture = SDL_CreateTexture(renderer,SDL_PIXELFORMAT_RGB24,SDL_TEXTUREACCESS_STATIC,completed.width,completed.height);
    bool uploaded = texture && !SDL_UpdateTexture(texture,nullptr,completed.rgb.data(),completed.width * 3);
    image = std::move(completed);
    image_record = job.parameters;
    image_has_record = true;
    image_source_path.clear();
    unsaved = true;
    main_form.carousel->select(-1);
    if (current) SDL_DestroyTexture(current);
    current = uploaded? texture : nullptr;
    if (!uploaded) {
        if (texture) SDL_DestroyTexture(texture);
        message("Image generated and available to Save; display upload failed: " + std::string(SDL_GetError()));
    }
}

int Pixelina::run()
{
    while (running) {
        SDL_Event input;
        while (SDL_PollEvent(&input)) event(input);
        if (busy) {
            SDL_LockMutex(mutex);
            bool done = worker_done;
            generation_progress = worker_progress;
            status = stop_requested? UI_STOPPING_TEXT : worker_stage;
            std::string error = done? worker_message : "";
            PixImage completed;
            PixImage preview = std::move(worker_preview);
            if (done) {
                completed = std::move(worker_image);
                generation_seconds = worker_seconds;
            }
            SDL_UnlockMutex(mutex);
            if (!preview.rgb.empty()) generation_image(std::move(preview));
            if (done) {
                SDL_WaitThread(worker,nullptr);
                worker = nullptr;
                busy = false;
                if (stop_requested || !error.empty()) {
                    status = stop_requested? UI_STOPPED_TEXT : "Generation failed";
                    if (!stop_requested) message(error);
                } else {
                    generation_image(std::move(completed));
                    status = "Generated image (unsaved)";
                }
            }
        }
        layout();
        paint();
        SDL_Delay(UI_FRAME_DELAY);
    }
    return 0;
}

bool PixUISystemFiles::read_directory(const std::string &path, std::vector<PixUIFileEntry> &entries, std::string &error)
{
    std::vector<PixDirEntry> listing;
    bool result = directory_list(path,listing,error);
    entries.clear();
    for (const auto &entry : listing) entries.push_back({entry.name,entry.directory});
    return result;
}

std::string PixUISystemFiles::normalize(const std::string &path)
{
    char* canonical = realpath(path.c_str(),nullptr);
    if (!canonical) return path;
    std::string result = canonical;
    std::free(canonical);
    return result;
}

PixUIPathKind PixUISystemFiles::path_kind(const std::string &path)
{
    struct stat info;
    if (stat(path.c_str(),&info)) return PIXUI_PATH_MISSING;
    if (S_ISDIR(info.st_mode)) return PIXUI_PATH_DIRECTORY;
    if (S_ISREG(info.st_mode)) return PIXUI_PATH_FILE;
    return PIXUI_PATH_OTHER;
}

std::string PixUISystemFiles::join_path(const std::string &directory, const std::string &name)
{
    return path_join(directory,name);
}

std::string PixUISystemFiles::parent_path(const std::string &path)
{
    size_t slash = path.rfind('/');
    return slash == std::string::npos? "." : path.substr(0,slash + 1);
}

PixelinaForm::PixelinaForm(PixUICanvas &context) : PixUIForm(context)
{
    decorate(caption);
    menu_background = add<PixUIPanel>();
    menu_background->color = COLOR_LIGHT;
    menu_background->border = false;
    const char* headings[] = {"Prompt","Settings","Saved images","Image"};
    for (int i = 0; i < UI_GROUP_COUNT; i++) {
        if (i == UI_SETTINGS_GROUP) {
            tabs = add<PixTab>();
            groups[i] = tabs;
        } else groups[i] = add<PixUIGroupBox>(headings[i]);
    }
    PixUIGroupBox* prompt_group = groups[UI_PROMPT_GROUP];
    prompt = prompt_group->add<PixUIMemo>();
    prompt_group->place(*prompt,{0,0,0,0},PIXUI_STRETCH_X | PIXUI_STRETCH_Y);
    register_focus(*prompt);
    tabs->footer_height = UI_SETTINGS_FOOTER;
    for (size_t page = 0; page < UI_SETTINGS_PAGES; page++) {
        PixTabGroup* settings = tabs->add_group(page == UI_GENERATE_PAGE? UI_GENERATE_TEXT : "Edit");
        pages[page] = settings;
        size_t count = page == UI_GENERATE_PAGE? UI_SETTINGS_COUNT : UI_EDIT_SETTINGS_COUNT;
        settings->content_height = int(count) * UI_SETTINGS_ROW + UI_MARGIN;
        settings->content_width = UI_SETTINGS_MIN_WIDTH;
        settings->wheel_step = UI_SETTINGS_ROW;
        for (size_t i = 0; i < count; i++) {
            PixUILabel* label = settings->add<PixUILabel>(page == UI_GENERATE_PAGE? field_labels[i] : edit_field_labels[i]);
            settings->place(*label,{UI_MARGIN - UI_BORDER_WIDTH,int(i) * UI_SETTINGS_ROW + UI_TEXT_PADDING,-(UI_MARGIN - UI_BORDER_WIDTH) * 2,UI_FONT_HEIGHT},PIXUI_STRETCH_X);
            PixUIEdit* field = settings->add<PixUIEdit>();
            if (page == UI_GENERATE_PAGE) fields[i] = field;
            else edit_fields[i] = field;
            register_focus(*field);
            int button_width = i == UI_SEED_FIELD || i >= UI_MODEL_FIRST? UI_BROWSE_BUTTON_WIDTH : 0;
            settings->place(*field,{UI_MARGIN - UI_BORDER_WIDTH,int(i) * UI_SETTINGS_ROW + UI_FIELD_INPUT_Y,-(UI_MARGIN - UI_BORDER_WIDTH) * 2 - button_width,UI_FIELD_HEIGHT},PIXUI_STRETCH_X | PIXUI_FULLY_VISIBLE);
            if (!button_width) continue;
            PixUIButton* button = settings->add<PixUIButton>(i == UI_SEED_FIELD? "R" : "...");
            if (i == UI_SEED_FIELD) random_seed[page] = button;
            else if (page == UI_GENERATE_PAGE) browse[i] = button;
            else edit_browse[i] = button;
            settings->place(*button,{-(UI_MARGIN - UI_BORDER_WIDTH) - button_width,int(i) * UI_SETTINGS_ROW + UI_FIELD_INPUT_Y,button_width,UI_FIELD_HEIGHT},PIXUI_ALIGN_RIGHT | PIXUI_FULLY_VISIBLE);
        }
    }
    preferences = tabs->add<PixUILabel>();
    tabs->place(*preferences,{UI_MARGIN - UI_BORDER_WIDTH,-UI_SETTINGS_FOOTER + UI_TEXT_PADDING,-(UI_MARGIN - UI_BORDER_WIDTH) * 2,UI_FONT_HEIGHT},PIXUI_STRETCH_X | PIXUI_ALIGN_BOTTOM);
    generate = tabs->add<PixUIButton>();
    tabs->place(*generate,{UI_MARGIN - UI_BORDER_WIDTH,-UI_FIELD_HEIGHT - UI_MARGIN,-(UI_MARGIN - UI_BORDER_WIDTH) * 2,UI_FIELD_HEIGHT},PIXUI_STRETCH_X | PIXUI_ALIGN_BOTTOM);
    tab_changed = [this](PixUIMemo* memo) {
        for (PixTabGroup* page : pages) {
            if (memo->parent == page) page->reveal(*memo);
        }
    };
    tabs->changed = [this](size_t page) {
        bool hidden_focus = false;
        if (page == UI_EDIT_PAGE) {
            for (PixUIEdit* field : fields) hidden_focus = hidden_focus || field->active;
        } else {
            for (PixUIEdit* field : edit_fields) hidden_focus = hidden_focus || field->active;
        }
        if (hidden_focus) focus_control(page == UI_GENERATE_PAGE? *fields[0] : *edit_fields[0]);
    };
    carousel = groups[UI_CAROUSEL_GROUP]->add<PixUICarousel>();
    groups[UI_CAROUSEL_GROUP]->place(*carousel,{0,0,0,0},PIXUI_STRETCH_X | PIXUI_STRETCH_Y);
    image = groups[UI_IMAGE_GROUP]->add<PixUIImageView>();
    groups[UI_IMAGE_GROUP]->place(*image,{0,0,0,0},PIXUI_STRETCH_X | PIXUI_STRETCH_Y);
    image->placeholder = "Generate or select an image";
    status = add<PixUIStatusBar>();
    status->progress_width = UI_STATUS_PROGRESS_WIDTH;
    status->progress_offset = UI_STATUS_PROGRESS_OFFSET;
    status->text_space = UI_STATUS_TEXT_SPACE;
    status->progress_top = UI_STATUS_PROGRESS_TOP;
    for (int i = 0; i < UI_SPLITTER_COUNT; i++) {
        splitters[i] = add<PixUISplitter>();
        splitters[i]->moved = [this,i](int x, int y) {
            int workspace = bounds.h - UI_WORK_TOP - UI_STATUS_HEIGHT - UI_FRAME;
            switch (i) {
            case UI_SPLIT_LEFT:
                left_fraction = float(x) / bounds.w;
                break;
            case UI_SPLIT_PROMPT:
                prompt_fraction = float(y - UI_WORK_TOP) / workspace;
                break;
            case UI_SPLIT_CAROUSEL:
                carousel_fraction = float(y - UI_WORK_TOP) / workspace;
                break;
            }
            layout();
        };
    }
}

void PixelinaForm::layout()
{
    layout_window(caption);
    int workspace = bounds.h - UI_WORK_TOP - UI_STATUS_HEIGHT - UI_FRAME;
    int left = std::max(UI_PANEL_MIN,std::min(bounds.w - UI_PANEL_MIN,int(bounds.w * left_fraction)));
    int top_left = std::max(UI_PANEL_MIN,std::min(workspace - UI_PANEL_MIN,int(workspace * prompt_fraction)));
    int top_right = std::max(UI_PANEL_MIN,std::min(workspace - UI_PANEL_MIN,int(workspace * carousel_fraction)));
    groups[UI_PROMPT_GROUP]->bounds = {UI_FRAME,UI_WORK_TOP,left - UI_FRAME,top_left};
    groups[UI_SETTINGS_GROUP]->bounds = {UI_FRAME,UI_WORK_TOP + top_left + UI_SPLITTER_WIDTH,left - UI_FRAME,workspace - top_left - UI_SPLITTER_WIDTH};
    groups[UI_CAROUSEL_GROUP]->bounds = {left + UI_SPLITTER_WIDTH,UI_WORK_TOP,bounds.w - left - UI_SPLITTER_WIDTH - UI_FRAME,top_right};
    groups[UI_IMAGE_GROUP]->bounds = {groups[UI_CAROUSEL_GROUP]->bounds.x,UI_WORK_TOP + top_right + UI_SPLITTER_WIDTH,groups[UI_CAROUSEL_GROUP]->bounds.w,workspace - top_right - UI_SPLITTER_WIDTH};
    menu_background->bounds = {UI_FRAME,UI_FRAME + UI_WINDOW_TITLE_HEIGHT,bounds.w - UI_FRAME * 2,UI_MENU_HEIGHT};
    status->bounds = {0,bounds.h - UI_STATUS_HEIGHT,bounds.w,UI_STATUS_HEIGHT};
    splitters[UI_SPLIT_LEFT]->bounds = {groups[UI_PROMPT_GROUP]->bounds.x + groups[UI_PROMPT_GROUP]->bounds.w,UI_WORK_TOP,UI_SPLITTER_WIDTH,bounds.h - UI_WORK_TOP - UI_STATUS_HEIGHT - UI_FRAME};
    splitters[UI_SPLIT_PROMPT]->bounds = {groups[UI_PROMPT_GROUP]->bounds.x,groups[UI_PROMPT_GROUP]->bounds.y + groups[UI_PROMPT_GROUP]->bounds.h,groups[UI_PROMPT_GROUP]->bounds.w,UI_SPLITTER_WIDTH};
    splitters[UI_SPLIT_CAROUSEL]->bounds = {groups[UI_CAROUSEL_GROUP]->bounds.x,groups[UI_CAROUSEL_GROUP]->bounds.y + groups[UI_CAROUSEL_GROUP]->bounds.h,groups[UI_CAROUSEL_GROUP]->bounds.w,UI_SPLITTER_WIDTH};
    PixUIForm::layout();
}

void PixUISDLCanvas::window_title(const std::string &title)
{
    SDL_SetWindowTitle(window,title.c_str());
}
