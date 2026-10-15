/* This file is a part of 'Pixelina' Image Generator, Editor, and Viewer.
 *
 * (C) Dmitry 'sciloaf' Solovyev, 2026
 *
 * License: MIT
 */
#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>
#include "ui.h"
#include "ui_refdlg.h"
#include "ui_chatdlg.h"
#include "inference.h"
#include "qwen_image.h"
#include "backend_vulkan.h"
#include "../assets/pixelina_icon.h"

static const char* field_labels[UI_SETTINGS_COUNT] = {"Seed","Width","Height","Steps","Scheduler shift","Diffuser GGUF","Text encoder GGUF","VAE safetensors"};
static const char* edit_field_labels[UI_EDIT_SETTINGS_COUNT] = {"Seed","Width","Height","Steps","Reference resolution","Diffuser GGUF","Language encoder GGUF","Vision projector GGUF","VAE safetensors"};
static const char* pane_configuration_keys[UI_SPLITTER_COUNT] = {"left_fraction","prompt_fraction","carousel_fraction"};

static int configuration_integer(const std::map<std::string,std::string> &values, const std::string &key, int fallback)
{
    auto entry = values.find(key);
    if (entry == values.end()) return fallback;
    char* end;
    errno = 0;
    long value = std::strtol(entry->second.c_str(),&end,10);
    if (end == entry->second.c_str() || *end || errno == ERANGE) return fallback;
    return int(value);
}

void Pixelina::restore_configuration(const std::map<std::string,std::string> &values)
{
    auto backend = values.find(UI_CONFIG_BACKEND),format = values.find(UI_CONFIG_FORMAT);
    if (backend != values.end() && (backend->second == PIXELINA_CPU || backend->second == PIXELINA_VULKAN)) settings.backend = backend->second;
    if (format != values.end() && (format->second == PIXELINA_JPG || format->second == PIXELINA_PNG)) settings.format = format->second;
    auto_seed = configuration_integer(values,UI_CONFIG_AUTO_SEED,1) == 1;
    decode_on_the_go = configuration_integer(values,UI_CONFIG_DECODE,0) == 1;
    global_db = configuration_integer(values,UI_CONFIG_GLOBAL_DB,0) == 1;
    main_form.image->scaled = configuration_integer(values,UI_CONFIG_SCALED,1) == 1;
    auto prompt = values.find(UI_CONFIG_PROMPT),browser = values.find(UI_CONFIG_BROWSER_DIRECTORY);
    if (prompt != values.end()) main_form.prompt->set_text(prompt->second);
    if (browser != values.end()) PixUIFileBrowser::last_directory = browser->second;
    width = std::max(UI_MIN_WIDTH,configuration_integer(values,UI_CONFIG_WINDOW_WIDTH,UI_START_WIDTH));
    height = std::max(UI_MIN_HEIGHT,configuration_integer(values,UI_CONFIG_WINDOW_HEIGHT,UI_START_HEIGHT));
    float* fractions[] = {&main_form.left_fraction,&main_form.prompt_fraction,&main_form.carousel_fraction};
    for (size_t i = 0; i < UI_SPLITTER_COUNT; i++) {
        auto entry = values.find(pane_configuration_keys[i]);
        float value;
        if (entry != values.end() && parse_float(entry->second,value) && std::isfinite(value) && value > 0 && value < 1) *fractions[i] = value;
    }
}

void Pixelina::save_configuration()
{
    PixUIRect normal = canvas.restored_window;
    if (!(SDL_GetWindowFlags(window) & (SDL_WINDOW_MAXIMIZED | SDL_WINDOW_MINIMIZED))) normal = canvas.window_rect();
    std::map<std::string,std::string> values = {
        {UI_CONFIG_BACKEND,settings.backend},
        {UI_CONFIG_FORMAT,settings.format},
        {UI_CONFIG_AUTO_SEED,std::to_string(auto_seed)},
        {UI_CONFIG_DECODE,std::to_string(decode_on_the_go.load())},
        {UI_CONFIG_SCALED,std::to_string(main_form.image->scaled)},
        {UI_CONFIG_GLOBAL_DB,std::to_string(global_db)},
        {UI_CONFIG_PROMPT,main_form.prompt->model.text},
        {UI_CONFIG_DIRECTORY,workdir},
        {UI_CONFIG_BROWSER_DIRECTORY,PixUIFileBrowser::last_directory},
        {UI_CONFIG_TAB,std::to_string(main_form.tabs->selected())},
        {UI_CONFIG_WINDOW_X,std::to_string(normal.x)},
        {UI_CONFIG_WINDOW_Y,std::to_string(normal.y)},
        {UI_CONFIG_WINDOW_WIDTH,std::to_string(normal.w)},
        {UI_CONFIG_WINDOW_HEIGHT,std::to_string(normal.h)},
        {UI_CONFIG_MAXIMIZED,std::to_string(canvas.maximized())}
    };
    float fractions[] = {main_form.left_fraction,main_form.prompt_fraction,main_form.carousel_fraction};
    for (size_t i = 0; i < UI_SPLITTER_COUNT; i++) {
        std::ostringstream value;
        value << std::setprecision(std::numeric_limits<float>::max_digits10) << fractions[i];
        values[pane_configuration_keys[i]] = value.str();
    }
    for (size_t page = 0; page < UI_SETTINGS_PAGES; page++) {
        size_t count = page == UI_GENERATE_PAGE? UI_SETTINGS_COUNT : UI_EDIT_SETTINGS_COUNT;
        for (size_t i = 0; i < count; i++) values[std::string(page == UI_GENERATE_PAGE? UI_CONFIG_GENERATE_PREFIX : UI_CONFIG_EDIT_PREFIX) + std::to_string(i)] = (page == UI_GENERATE_PAGE? main_form.fields[i] : main_form.edit_fields[i])->model.text;
    }
    std::string error;
    if (!config_save(user_directory,values,error)) std::fprintf(stderr,PIXELINA_NAME ": %s\n",error.c_str());
}

void Pixelina::switch_database()
{
    std::vector<PixRecord> loaded;
    std::string error;
    if (!database_load(workdir,loaded,error,global_db? std::string{} : path_join(user_directory,PIXELINA_GLOBAL_DB))) {
        message(error);
        return;
    }
    global_db = !global_db;
    records.swap(loaded);
}

bool Pixelina::initialize(const std::string &directory, bool png, std::string &error, bool default_directory, bool use_saved_directory)
{
    char* cwd = getcwd(nullptr,0);
    if (!cwd) {
        error = "Cannot determine the launch directory.";
        return false;
    }
    launch_directory = cwd;
    std::free(cwd);
    PixUIFileBrowser::last_directory = launch_directory;
    const char* home = std::getenv("HOME");
    if (!home || !home[0]) {
        error = "Cannot determine the home directory.";
        return false;
    }
    user_directory = path_join(home,PIXELINA_USER_DIRECTORY);
    if (mkdir(user_directory.c_str(),0700) && errno != EEXIST) {
        error = "Cannot create configuration directory: " + std::string(std::strerror(errno));
        return false;
    }
    std::map<std::string,std::string> configuration;
    if (!config_load(user_directory,configuration,error)) return false;
    restore_configuration(configuration);
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER)) {
        error = SDL_GetError();
        return false;
    }
    window = SDL_CreateWindow(PIXELINA_NAME,configuration_integer(configuration,UI_CONFIG_WINDOW_X,SDL_WINDOWPOS_CENTERED),configuration_integer(configuration,UI_CONFIG_WINDOW_Y,SDL_WINDOWPOS_CENTERED),width,height,SDL_WINDOW_RESIZABLE | SDL_WINDOW_BORDERLESS | SDL_WINDOW_HIDDEN);
    if (!window) {
        error = SDL_GetError();
        return false;
    }
    canvas.window = window;
    main_form.bounds = {0,0,width,height};
    SDL_SetWindowMinimumSize(window,UI_MIN_WIDTH,UI_MIN_HEIGHT);
    canvas.restored_window = canvas.window_rect();
    // Establish normal geometry and maximize before mapping the window to the desktop.
    if (configuration_integer(configuration,UI_CONFIG_MAXIMIZED,0) == 1) canvas.window_command(PIXUI_MAXIMIZE,{});
    SDL_GetWindowSize(window,&width,&height);
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
    if (png) settings.format = PIXELINA_PNG;
    build_gui();
    write_settings();
    write_edit_settings();
    for (size_t page = 0; page < UI_SETTINGS_PAGES; page++) {
        size_t count = page == UI_GENERATE_PAGE? UI_SETTINGS_COUNT : UI_EDIT_SETTINGS_COUNT;
        for (size_t i = 0; i < count; i++) {
            auto value = configuration.find(std::string(page == UI_GENERATE_PAGE? UI_CONFIG_GENERATE_PREFIX : UI_CONFIG_EDIT_PREFIX) + std::to_string(i));
            if (value != configuration.end()) (page == UI_GENERATE_PAGE? main_form.fields[i] : main_form.edit_fields[i])->set_text(value->second);
        }
    }
    main_form.tabs->select(size_t(configuration_integer(configuration,UI_CONFIG_TAB,UI_GENERATE_PAGE)));
    bool open_file = path_kind(directory) == PIX_PATH_FILE;
    std::string selected_directory = open_file? path_parent(path_normalize(directory)) : directory;
    auto saved_directory = configuration.find(UI_CONFIG_DIRECTORY);
    if (default_directory || use_saved_directory) {
        selected_directory = saved_directory != configuration.end() && !saved_directory->second.empty()? saved_directory->second : launch_directory;
    }
    if (!open_directory(selected_directory)) {
        error = dialog_message;
        return false;
    }
    if (open_file && !open_image(directory)) {
        error = dialog_message;
        return false;
    }
    layout();
    SDL_ShowWindow(window);
    SDL_StartTextInput();
    initialized = true;
    return true;
}

Pixelina::~Pixelina()
{
    stop_requested = true;
    if (worker) SDL_WaitThread(worker,nullptr);
    if (initialized) save_configuration();
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
    if (!directory_list(actual,listing,error) || !database_load(actual,loaded,error,global_db? path_join(user_directory,PIXELINA_GLOBAL_DB) : std::string{})) {
        message(error);
        return false;
    }
    workdir = actual;
    records.swap(loaded);
    images.clear();
    const char* extensions[] = UI_IMAGE_EXTENSIONS;
    for (const PixDirEntry &entry : listing) {
        if (entry.directory) continue;
        size_t dot = entry.name.rfind('.');
        if (dot == std::string::npos) continue;
        std::string extension = entry.name.substr(dot);
        for (char &ch : extension) {
            if (ch >= 'A' && ch <= 'Z') ch += 'a' - 'A';
        }
        for (const char* supported : extensions) {
            if (extension != supported) continue;
            images.push_back(entry.name);
            break;
        }
    }
    main_form.carousel->items_changed(images.size(),true);
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
    static_cast<PixUIGroupBox*>(main_form.image->parent)->caption = image.width > 0 && image.height > 0? UI_IMAGE_TITLE " (" + std::to_string(image.width) + "x" + std::to_string(image.height) + ")" : UI_IMAGE_TITLE;
    char timing[UI_STATUS_BUFFER];
    if (busy && generation_progress > 0 && generation_progress < 1) {
        double elapsed = double(canvas.ticks() - generation_started) / 1000;
        double remaining = elapsed * (1 - generation_progress) / generation_progress;
        uint64_t seconds = uint64_t(remaining);
        std::snprintf(timing,sizeof(timing),"ETA %llum %02llus",static_cast<unsigned long long>(seconds / 60),static_cast<unsigned long long>(seconds % 60));
    } else if (busy) std::snprintf(timing,sizeof(timing),generation_progress >= 1? "ETA 0s" : "ETA --");
    else if (generation_seconds > 0) std::snprintf(timing,sizeof(timing),"%.2fs",generation_seconds);
    else timing[0] = 0;
    main_form.status->set_status(status,std::to_string(images.size()) + " images",timing,busy,generation_progress);
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
        candidate.parameters.model = PIX_MODEL_QWEN_IMAGE;
        candidate.parameters.edit_settings = edit_settings;
        candidate.edit_settings = edit_settings;
        if (reference_paths.empty()) {
            if (image_valid(image)) {
                candidate.references.push_back(image);
                if (!image_source_path.empty()) candidate.parameters.references.push_back(image_source_path);
            }
        } else {
            for (const std::string &path : reference_paths) {
                PixImage reference;
                std::string error;
                if (!image_load(path,reference,error) || !image_valid(reference)) {
                    message(error.empty()? UI_REFERENCE_OPEN_ERROR + path : error);
                    return false;
                }
                candidate.references.push_back(std::move(reference));
            }
            candidate.parameters.references = reference_paths;
        }
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
    std::string actual = path_normalize(path_join(launch_directory,path));
    PixImage loaded;
    std::string error;
    if (path_kind(actual) != PIX_PATH_FILE || !image_load(actual,loaded,error)) {
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
    edit_settings.width = (image.width + QWEN_IMAGE_MULTIPLE - 1) / QWEN_IMAGE_MULTIPLE * QWEN_IMAGE_MULTIPLE;
    edit_settings.height = (image.height + QWEN_IMAGE_MULTIPLE - 1) / QWEN_IMAGE_MULTIPLE * QWEN_IMAGE_MULTIPLE;
    main_form.edit_fields[UI_EDIT_WIDTH]->set_text(std::to_string(edit_settings.width));
    main_form.edit_fields[UI_EDIT_HEIGHT]->set_text(std::to_string(edit_settings.height));
    reference_resolution();
    status = actual;
    auto selected = std::find_if(images.begin(),images.end(),[&](const std::string &filename) { return path_normalize(path_join(workdir,filename)) == actual; });
    if (selected == images.end()) return true;
    main_form.carousel->select(int(selected - images.begin()));
    auto record = std::find_if(records.begin(),records.end(),[&](const PixRecord &value) { return path_join(workdir,value.filename) == actual; });
    if (record == records.end()) return true;
    image_record = *record;
    image_has_record = true;
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
    status = *selected;
    return true;
}

void Pixelina::message(const std::string &value)
{
    dialog_message = value;
    main_form.present(std::make_unique<PixUIMessageDialog>(canvas,value,PIXELINA_NAME));
}

bool Pixelina::select_image(size_t index)
{
    if (index >= images.size() || busy) return false;
    return open_image(path_join(workdir,images[index]));
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
    if (!storage_save(workdir,record,image,error,global_db? path_join(user_directory,PIXELINA_GLOBAL_DB) : std::string{})) {
        message(error);
        return;
    }
    image_record = record;
    image_source_path = path_normalize(path_join(workdir,record.filename));
    records.push_back(record);
    std::string filename = image_source_path.substr(image_source_path.rfind('/') + 1);
    images.push_back(filename);
    std::sort(images.begin(),images.end());
    main_form.carousel->items_changed(images.size());
    main_form.carousel->select(int(std::find(images.begin(),images.end(),filename) - images.begin()));
    unsaved = false;
    status = "Saved " + record.filename;
}

void Pixelina::export_image(const std::string &destination)
{
    std::string error;
    if (!image_save(destination,image,error)) message(error);
    else {
        image_source_path = path_normalize(destination);
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
    if (command == CMD_GLOBAL_DB) {
        switch_database();
        return;
    }
    if (busy) return;
    switch (command) {
    case CMD_CHAT: {
        if (!image_valid(image)) {
            message("Open or generate an image before starting a chat.");
            return;
        }
        uint64_t resolution;
        if (!parse_unsigned(main_form.edit_fields[UI_EDIT_RESOLUTION]->model.text,resolution) || !resolution || resolution > INT32_MAX || resolution * resolution > PIXELINA_MAX_PIXELS) {
            message("Enter a valid reference resolution on the Edit page.");
            return;
        }
        PixEditSettings chat;
        chat.encoder = main_form.edit_fields[UI_EDIT_ENCODER]->model.text;
        chat.vision = main_form.edit_fields[UI_EDIT_VISION]->model.text;
        chat.reference_resolution = int(resolution);
        chat.backend = settings.backend;
        if (chat.encoder.empty() || chat.vision.empty()) {
            message("Select the Qwen-VL language model and mmproj on the Edit page.");
            return;
        }
        chat.encoder = path_join(launch_directory,chat.encoder);
        chat.vision = path_join(launch_directory,chat.vision);
        std::string name = image_source_path.empty()? "Unsaved image" : image_source_path.substr(image_source_path.find_last_of('/') + 1);
        main_form.present(std::make_unique<PixelinaChatDialog>(canvas,name,chat,image));
        break;
    }
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
        reference_paths.clear();
        unsaved = false;
        generation_progress = 0;
        generation_seconds = 0;
        main_form.image->set_image(nullptr,0,0);
        main_form.carousel->select(-1);
        status = UI_READY_TEXT;
        break;
    case CMD_MULTI_REFERENCE: {
        std::vector<std::string> paths = reference_paths;
        if (paths.empty() && image_valid(image) && !image_source_path.empty()) paths.push_back(image_source_path);
        auto dialog = std::make_unique<PixelinaReferenceDialog>(canvas,paths,main_form.prompt->model.text);
        dialog->load = [this](const std::string &path) {
            return load_thumbnail(path);
        };
        dialog->selected = [this](const std::vector<std::string> &paths, const std::string &prompt) {
            reference_paths = paths;
            main_form.prompt->set_text(prompt);
        };
        main_form.present(std::move(dialog));
        break;
    }
    case CMD_OPEN: {
        auto browser = std::make_unique<PixUIFileBrowser>(canvas);
        browser->dir_selector = true;
        browser->title = "Open directory";
        browser->error = [this](const std::string &error) { status = error; };
        browser->selected = [this](const std::string &path) { return open_directory(path); };
        browser->refresh();
        main_form.present(std::move(browser));
        break;
    }
    case CMD_OPEN_IMAGE: {
        auto browser = std::make_unique<PixUIFileBrowser>(canvas);
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
            auto browser = std::make_unique<PixUIFileBrowser>(canvas,path_join(PixUIFileBrowser::last_directory,settings.format == PIXELINA_PNG? "image.png" : "image.jpg"));
            browser->title = UI_SAVE_AS_TEXT;
            browser->accept_caption = UI_SAVE_TEXT;
            browser->path_label = "Destination (.jpg or .png):";
            browser->hint_text = "Exports the current image without a database entry.";
            browser->show_listing = false;
            browser->selected = [this](const std::string &path) {
                std::string destination = path_join(workdir,path);
                if (path_kind(destination) != PIX_PATH_MISSING) {
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
        generation_started = canvas.ticks();
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

PixUIThumbCache Pixelina::load_thumbnail(const std::string &path)
{
    PixUIThumbCache value;
    PixImage full;
    std::string error;
    if (image_load(path,full,error)) {
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
    edit->add("Multi-Reference...")->activate = [this]() {
        action(CMD_MULTI_REFERENCE);
    };
    PixUIMenuItem* generate = edit->add(UI_GENERATE_TEXT,[this]() { action(CMD_GENERATE); });
    generate->bind_shortcut(PIXUI_F5,0,"F5");
    generate->caption_state = [this]() { return busy? UI_STOP_TEXT : UI_GENERATE_TEXT; };
    PixUIMenu* preferences = edit->child(*edit->add("Preferences"));
    PixUIMenuItem* seed = preferences->add("Auto-seed",[this]() { action(CMD_AUTO_SEED); });
    seed->checkable = true;
    seed->checked_state = [this]() { return auto_seed; };
    PixUIMenuItem* database = preferences->add("Use global DB",[this]() { action(CMD_GLOBAL_DB); });
    database->checkable = true;
    database->checked_state = [this]() { return global_db; };
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
    PixUIMenuItem* chat = popups[2]->add("Chat",[this]() { action(CMD_CHAT); });
    chat->enabled_state = [this]() { return !busy && image_valid(image); };
    popups[3]->add(UI_ABOUT_PREFIX PIXELINA_NAME,[this]() { action(CMD_ABOUT); });
    main_form.add_system_menu();
}

void Pixelina::build_gui()
{
    main_form.close = [this]() { running = false; };
    build_menus();
    main_form.generate->activate = [this]() { action(CMD_GENERATE); };
    main_form.carousel->caption = [this](size_t index) { return images[index]; };
    main_form.carousel->load = [this](size_t index) {
        return load_thumbnail(path_join(workdir,images[index]));
    };
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
                auto browser = std::make_unique<PixUIFileBrowser>(canvas);
                browser->title = "Open model file";
                browser->error = [this](const std::string &error) { status = error; };
                browser->selected = [this,field](const std::string &path) {
                    if (path_kind(path) != PIX_PATH_FILE) {
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
    SDL_CaptureMouse(canvas.captured? SDL_TRUE : SDL_FALSE);
    layout();
}

void Pixelina::window_event(const SDL_Event &input)
{
    if (!(SDL_GetWindowFlags(window) & (SDL_WINDOW_MAXIMIZED | SDL_WINDOW_MINIMIZED))) {
        switch (input.window.event) {
        case SDL_WINDOWEVENT_MOVED:
        case SDL_WINDOWEVENT_SIZE_CHANGED:
            // Queued events can describe geometry from before a maximize or restore.
            canvas.restored_window = canvas.window_rect();
            break;
        default:
            break;
        }
    }
    switch (input.window.event) {
    case SDL_WINDOWEVENT_FOCUS_GAINED:
        main_form.repaint = true;
        main_form.active = true;
        break;
    case SDL_WINDOWEVENT_FOCUS_LOST:
        canvas.release_capture();
        SDL_CaptureMouse(SDL_FALSE);
        main_form.repaint = true;
        main_form.active = false;
        break;
    case SDL_WINDOWEVENT_SIZE_CHANGED:
        main_form.repaint = true;
        width = input.window.data1;
        height = input.window.data2;
        layout();
        break;
    case SDL_WINDOWEVENT_EXPOSED:
    case SDL_WINDOWEVENT_RESTORED:
    case SDL_WINDOWEVENT_MAXIMIZED:
        main_form.repaint = true;
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
    uint64_t timing_second = 0;
    while (running) {
        SDL_Event input;
        while (SDL_PollEvent(&input)) event(input);
        if (busy) {
            SDL_LockMutex(mutex);
            bool done = worker_done;
            std::string stage = stop_requested? UI_STOPPING_TEXT : worker_stage;
            uint64_t second = canvas.ticks() / 1000;
            bool changed = generation_progress != worker_progress || status != stage || second != timing_second;
            generation_progress = worker_progress;
            status = stage;
            timing_second = second;
            std::string error = done? worker_message : "";
            PixImage completed;
            PixImage preview = std::move(worker_preview);
            if (done) {
                completed = std::move(worker_image);
                generation_seconds = worker_seconds;
            }
            SDL_UnlockMutex(mutex);
            if (!preview.rgb.empty()) {
                generation_image(std::move(preview));
                changed = true;
            }
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
            if (changed || done) layout();
        }
        if (main_form.needs_repaint()) paint();
        SDL_Delay(UI_FRAME_DELAY);
    }
    return 0;
}

PixelinaForm::PixelinaForm(PixUICanvas &context) : PixUIForm(context)
{
    decorate(caption);
    menu_background = add<PixUIPanel>();
    menu_background->color = COLOR_LIGHT;
    menu_background->border = false;
    const char* headings[] = {"Prompt","Settings","Saved images",UI_IMAGE_TITLE};
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
