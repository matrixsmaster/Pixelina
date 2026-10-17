/* This file is a part of 'Pixelina' Image Generator, Editor, and Viewer.
 *
 * (C) Dmitry 'sciloaf' Solovyev, 2026
 *
 * License: MIT
 */
#include <algorithm>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>
#include "ui.h"
#include "ui_about.h"
#include "ui_refdlg.h"
#include "ui_chatdlg.h"
#include "dlg_bookimg.h"
#include "inference.h"
#include "qwen_image.h"
#include "backend_vulkan.h"

static const char* field_labels[UI_SETTINGS_COUNT] = {"Seed","Width","Height","Steps","Scheduler shift","Diffuser GGUF","Text encoder GGUF","VAE safetensors"};
static const char* edit_field_labels[UI_EDIT_SETTINGS_COUNT] = {"Seed","Width","Height","Steps","Reference resolution","Diffuser GGUF","Language encoder GGUF","Vision projector GGUF","VAE safetensors"};
static const char* pane_configuration_keys[UI_SPLITTER_COUNT] = {"left_fraction","prompt_fraction","carousel_fraction"};

static int configuration_integer(const std::map<std::string,std::string> &values, const std::string &key, int fallback)
{
    auto entry = values.find(key);
    if (entry == values.end()) return fallback;
    char* end;
    errno = 0;
    long value = strtol(entry->second.c_str(),&end,10);
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
        if (entry != values.end() && parse_float(entry->second,value) && isfinite(value) && value > 0 && value < 1) *fractions[i] = value;
    }
}

void Pixelina::save_configuration()
{
    PixUIRect normal = canvas.restored_window;
    if (!(SDL_GetWindowFlags(window) & (SDL_WINDOW_MAXIMIZED | SDL_WINDOW_MINIMIZED))) normal = canvas.window_rect();
    std::map<std::string,std::string> values = {
        {UI_CONFIG_THEME,theme_path},
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
    if (!config_save(user_directory,values,error)) fprintf(stderr,PIXELINA_NAME ": %s\n",error.c_str());
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

bool Pixelina::initialize(const std::string &directory, bool png, std::string &error, bool default_directory, bool use_saved_directory, const std::string &job_file)
{
    char* cwd = getcwd(nullptr,0);
    if (!cwd) {
        error = "Cannot determine the launch directory.";
        return false;
    }
    launch_directory = cwd;
    free(cwd);
    PixUIFileBrowser::last_directory = launch_directory;
    const char* home = getenv("HOME");
    if (!home || !home[0]) {
        error = "Cannot determine the home directory.";
        return false;
    }
    user_directory = path_join(home,PIXELINA_USER_DIRECTORY);
    if (mkdir(user_directory.c_str(),0700) && errno != EEXIST) {
        error = "Cannot create configuration directory: " + std::string(strerror(errno));
        return false;
    }
    std::map<std::string,std::string> configuration;
    if (!config_load(user_directory,configuration,error)) return false;
    if (!bookmarks_load(user_directory,bookmarks,error)) return false;
    auto theme = configuration.find(UI_CONFIG_THEME);
    if (theme != configuration.end()) theme_path = theme->second;
    if (!pixui_load_theme(theme_path.empty()? std::string{} : path_join(user_directory,theme_path),error)) return false;
    main_form.create_controls();
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
    pause_condition = SDL_CreateCond();
    if (!renderer || !mutex || !pause_condition) {
        error = SDL_GetError();
        return false;
    }
    PixImage icon;
    if (!PixelinaAbout::first_frame(icon,error)) return false;
    SDL_Surface* surface = SDL_CreateRGBSurfaceFrom(icon.rgb.data(),icon.width,icon.height,24,icon.width * 3,SDL_BYTEORDER == SDL_LIL_ENDIAN? 0x0000ff : 0xff0000,0x00ff00,SDL_BYTEORDER == SDL_LIL_ENDIAN? 0xff0000 : 0x0000ff,0);
    if (!surface) {
        error = SDL_GetError();
        return false;
    }
    SDL_SetWindowIcon(window,surface);
    SDL_FreeSurface(surface);
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
    bool open_file = job_file.empty() && path_kind(directory) == PIX_PATH_FILE;
    std::string selected_directory = open_file? path_parent(path_normalize(directory)) : directory;
    auto saved_directory = configuration.find(UI_CONFIG_DIRECTORY);
    if (default_directory || use_saved_directory) {
        selected_directory = saved_directory != configuration.end() && !saved_directory->second.empty()? saved_directory->second : launch_directory;
    }
    if (!job_file.empty()) {
        if (!queue_import(main_form,job_file)) {
            error = dialog_message;
            return false;
        }
        if (!queue.items.empty()) queue_start(main_form);
    } else if (!open_directory(selected_directory)) {
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
    if (mutex && pause_condition) {
        SDL_LockMutex(mutex);
        pause_requested = false;
        SDL_CondBroadcast(pause_condition);
        SDL_UnlockMutex(mutex);
    }
    if (worker) SDL_WaitThread(worker,nullptr);
    if (initialized) save_configuration();
    canvas.release_capture();
    main_form.clear();
    canvas.set_main_image(nullptr,nullptr);
    if (current) SDL_DestroyTexture(current);
    if (pause_condition) SDL_DestroyCond(pause_condition);
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
    free(canonical);
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
    for (PixUIButton* button : main_form.spin_buttons) button->enabled = !busy;
    main_form.generate->caption = busy? (stop_requested? UI_STOPPING_TEXT : "Stop [F5]") : "Generate [F5]";
    main_form.preferences->caption = settings.backend + " / " + settings.format;
    main_form.carousel->enabled = !busy;
    main_form.bookmarks->enabled = !busy;
    main_form.image->set_image(current,image.width,image.height);
    main_form.groups[UI_IMAGE_GROUP]->caption = image.width > 0 && image.height > 0? UI_IMAGE_TITLE " (" + std::to_string(image.width) + "x" + std::to_string(image.height) + ")" : UI_IMAGE_TITLE;
    char timing[UI_STATUS_BUFFER];
    if (busy) {
        uint64_t seconds = queue.active == UI_QUEUE_NO_TASK? (canvas.ticks() - generation_started) / 1000 : uint64_t(queue.items[queue.active].seconds);
        snprintf(timing,sizeof(timing),"%llum %02llus",(unsigned long long)(seconds / 60),(unsigned long long)(seconds % 60));
    } else if (generation_seconds > 0) snprintf(timing,sizeof(timing),"%.2fs",generation_seconds);
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
            if (image_valid(image) && (!image_has_record || image_record.model != PIX_MODEL_QWEN_IMAGE)) {
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
    double side = sqrt(double(image.width) * image.height);
    edit_settings.reference_resolution = std::max(QWEN_IMAGE_MULTIPLE,int(round(side / QWEN_IMAGE_MULTIPLE)) * QWEN_IMAGE_MULTIPLE);
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
    SDL_Texture* texture = canvas.create_texture(loaded);
    if (!texture) {
        message(SDL_GetError());
        return false;
    }
    if (current) SDL_DestroyTexture(current);
    current = texture;
    image = std::move(loaded);
    canvas.set_main_image(&image,current);
    image_record = PixRecord();
    image_has_record = false;
    image_source_path = actual;
    unsaved = false;
    main_form.carousel->select(-1);
    main_form.bookmarks->select(-1);
    bookmark_selection = -1;
    edit_settings.width = (image.width + QWEN_IMAGE_MULTIPLE - 1) / QWEN_IMAGE_MULTIPLE * QWEN_IMAGE_MULTIPLE;
    edit_settings.height = (image.height + QWEN_IMAGE_MULTIPLE - 1) / QWEN_IMAGE_MULTIPLE * QWEN_IMAGE_MULTIPLE;
    main_form.edit_fields[UI_EDIT_WIDTH]->set_text(std::to_string(edit_settings.width));
    main_form.edit_fields[UI_EDIT_HEIGHT]->set_text(std::to_string(edit_settings.height));
    reference_resolution();
    status = actual;
    auto selected = std::find_if(images.begin(),images.end(),[&](const std::string &filename) { return path_normalize(path_join(workdir,filename)) == actual; });
    if (selected != images.end()) main_form.carousel->select(int(selected - images.begin()));
    auto record = std::find_if(records.begin(),records.end(),[&](const PixRecord &value) { return path_normalize(path_join(workdir,value.filename)) == actual; });
    if (record == records.end()) return true;
    image_record = *record;
    image_has_record = true;
    main_form.tabs->select(image_record.model == PIX_MODEL_Z_IMAGE? UI_GENERATE_PAGE : UI_EDIT_PAGE);
    if (image_record.model == PIX_MODEL_Z_IMAGE) {
        settings = image_record.settings;
        write_settings();
    } else {
        edit_settings = image_record.edit_settings;
        reference_paths = image_record.references;
        settings.backend = edit_settings.backend;
        settings.format = edit_settings.format;
        write_edit_settings();
    }
    main_form.prompt->set_text(image_record.prompt);
    status = selected == images.end()? actual : *selected;
    return true;
}

void Pixelina::message(const std::string &value, PixUIForm* owner)
{
    dialog_message = value;
    (owner? *owner : main_form).present(std::make_unique<PixUIMessageDialog>(canvas,value,PIXELINA_NAME));
}

bool Pixelina::select_image(size_t index)
{
    if (index >= images.size() || busy) return false;
    return open_image(path_join(workdir,images[index]));
}

void Pixelina::bookmark_image(bool remove)
{
    std::vector<std::string> changed = bookmarks;
    if (remove) {
        if (bookmark_selection < 0 || size_t(bookmark_selection) >= bookmarks.size()) {
            message("Select an image in the Bookmarks carousel first.");
            return;
        }
        changed.erase(changed.begin() + bookmark_selection);
    } else {
        if (image_source_path.empty()) {
            message("Save the image before bookmarking it.");
            return;
        }
        if (std::find(bookmarks.begin(),bookmarks.end(),image_source_path) != bookmarks.end()) return;
        changed.push_back(image_source_path);
    }
    std::string error;
    if (!bookmarks_save(user_directory,changed,error)) {
        message(error);
        return;
    }
    bookmarks.swap(changed);
    main_form.bookmarks->items_changed(bookmarks.size());
    if (remove) {
        bookmark_selection = -1;
        main_form.bookmarks->select(-1);
    }
}

bool Pixelina::save_image()
{
    if (busy) return false;
    if (!image_has_record) {
        message(image.rgb.empty()? "Generate an image first." : "This image has no generation record. Use Save as to export it.");
        return false;
    }
    if (!unsaved) {
        message("This image is already in the database. Use Save as to export it.");
        return false;
    }
    PixRecord record = image_record;
    if (queue.active == UI_QUEUE_NO_TASK) {
        if (record.model == PIX_MODEL_Z_IMAGE) record.settings.format = settings.format;
        else record.edit_settings.format = settings.format;
    }
    std::string error;
    if (!storage_save(workdir,record,image,error,global_db? path_join(user_directory,PIXELINA_GLOBAL_DB) : std::string{})) {
        message(error);
        return false;
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
    return true;
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
    std::string last_stage;
    Uint64 start = SDL_GetPerformanceCounter();
    auto progress = [&](float fraction, const std::string &stage) {
        if (stop_requested) return false;
        SDL_LockMutex(mutex);
        worker_progress = fraction;
        worker_stage = stage;
        last_stage = stage;
        if (pause_requested && !stop_requested) {
            worker_paused = true;
            worker_pause_started = SDL_GetPerformanceCounter();
            while (pause_requested && !stop_requested) SDL_CondWait(pause_condition,mutex);
            worker_pause_seconds += double(SDL_GetPerformanceCounter() - worker_pause_started) / SDL_GetPerformanceFrequency();
            worker_paused = false;
            worker_pause_started = 0;
        }
        bool proceed = !stop_requested;
        SDL_UnlockMutex(mutex);
        return proceed;
    };
    int code = 0;
    auto decode_enabled = [&]() { return decode_on_the_go.load(); };
    auto preview = [&](PixImage decoded) {
        SDL_LockMutex(mutex);
        worker_preview = std::move(decoded);
        SDL_UnlockMutex(mutex);
    };
    if (inference) result = inference(job,progress,code);
    else if (job.parameters.model == PIX_MODEL_QWEN_IMAGE) result = qwen_image_edit(job.edit_settings,job.references,job.parameters.prompt,progress,code,decode_enabled,preview);
    else result = inference_generate(job.generate_settings,job.parameters.prompt,progress,code,decode_enabled,preview);
    if (!code && !image_valid(result)) code = INFERENCE_ERROR_NUMERICAL;
    if (stop_requested) error = UI_STOPPED_TEXT;
    else switch (code) {
    case INFERENCE_ERROR_SETTINGS:
        if (job.parameters.model == PIX_MODEL_Z_IMAGE) settings_valid(job.generate_settings,error);
        else edit_settings_valid(job.edit_settings,error);
        if (error.empty()) error = last_stage;
        break;
    case INFERENCE_ERROR_MODEL:
        error = "Cannot load models or unsupported model tensors/tokenizer.";
        break;
    case INFERENCE_ERROR_PROMPT:
        error = "Invalid prompt, context exceeded, or text encoding failed.";
        break;
    case INFERENCE_ERROR_NUMERICAL:
        error = "Inference produced an empty or invalid tensor/image, or non-finite values.";
        break;
    case INFERENCE_ERROR_VULKAN:
        error = "Vulkan backend unavailable or image exceeds device limits.";
        break;
    case INFERENCE_ERROR_STOPPED:
        error = UI_STOPPED_TEXT;
        break;
    case INFERENCE_ERROR_MEMORY:
        error = last_stage;
        break;
    default:
        if (code) error = "Unknown inference failure.";
        break;
    }
    if (code && code != INFERENCE_ERROR_STOPPED && !stop_requested) {
        if (!last_stage.empty() && error != last_stage) error += "\nLast stage: " + last_stage;
        error += "\nInference error code: " + std::to_string(code);
    }
    double seconds = double(SDL_GetPerformanceCounter() - start) / SDL_GetPerformanceFrequency() - worker_pause_seconds;
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
        std::vector<std::string> lines = PIXELINA_ABOUT;
        auto about = std::make_unique<PixelinaAbout>(canvas,PIXELINA_ABOUT_PREFIX PIXELINA_NAME,PIXELINA_ABOUT_FPS,lines);
        std::string error;
        if (!about->prepare_about_animation(renderer,error)) {
            message(error);
            return;
        }
        main_form.present(std::move(about));
        return;
    }
    if (command == CMD_FULLSCREEN) {
        main_form.fullscreen = !main_form.fullscreen;
        if (main_form.fullscreen) main_form.focus_object(main_form.image_tabs->selected() == UI_BOOKMARKS_PAGE? *main_form.bookmarks : *main_form.carousel);
        return;
    }
    if (busy && command == CMD_GENERATE) {
        if (queue.active != UI_QUEUE_NO_TASK) {
            queue_stop(main_form);
            return;
        }
        stop_requested = true;
        status = UI_STOPPING_TEXT;
        return;
    }
    if (command == CMD_DECODE) {
        if (queue.active != UI_QUEUE_NO_TASK) return;
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
    case CMD_BOOKMARK:
        bookmark_image(false);
        break;
    case CMD_UNBOOKMARK:
        bookmark_image(true);
        break;
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
        canvas.set_main_image(nullptr,nullptr);
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
        main_form.bookmarks->select(-1);
        bookmark_selection = -1;
        status = UI_READY_TEXT;
        break;
    case CMD_MULTI_REFERENCE: {
        std::vector<std::string> paths = reference_paths;
        if (paths.empty() && image_valid(image) && !image_source_path.empty() && (!image_has_record || image_record.model != PIX_MODEL_QWEN_IMAGE)) paths.push_back(image_source_path);
        auto dialog = std::make_unique<PixelinaReferenceDialog>(canvas,paths,main_form.prompt->model.text,bookmarks);
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
    case CMD_RELOAD_DIR:
        open_directory(workdir);
        break;
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
                    auto confirmation = std::make_unique<PixUIMessageDialog>(canvas,UI_OVERWRITE_FILE_TEXT + destination,PIXELINA_NAME,UI_OVERWRITE_TEXT,UI_CANCEL_TEXT);
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
        if (start_generation() && auto_seed) randomize_seed(main_form.tabs->selected());
        break;
    }
    default:
        break;
    }
}

bool Pixelina::start_generation()
{
    stop_requested = false;
    pause_requested = false;
    worker_paused = false;
    worker_pause_seconds = 0;
    worker_pause_started = 0;
    worker_done = false;
    worker_progress = 0;
    generation_progress = 0;
    worker_stage = UI_QUEUE_LOADING_TEXT;
    worker_message.clear();
    worker_image = PixImage();
    worker_preview = PixImage();
    busy = true;
    generation_started = canvas.ticks();
    status = "Generating";
    worker = SDL_CreateThread(generation_worker,PIXELINA_NAME " generation",this);
    if (worker) return true;
    busy = false;
    message(SDL_GetError());
    return false;
}

void Pixelina::queue_add()
{
    if (menus) menus->dismiss();
    if (busy || !capture_job()) return;
    PixQueueItem item;
    item.source = queue_source_thumbnail(job.references);
    item.description = queue_job_description(job);
    item.job = std::move(job);
    queue.items.push_back(std::move(item));
    queue.revision++;
    if (auto_seed) randomize_seed(main_form.tabs->selected());
    status = "Added job to queue";
}

void Pixelina::queue_view()
{
    auto browser = std::make_unique<PixelinaQueueDialog>(canvas,queue);
    PixelinaQueueDialog* dialog = browser.get();
    browser->clear_button->activate = [this,dialog]() {
        auto confirmation = std::make_unique<PixUIMessageDialog>(canvas,"Are you sure?",UI_QUEUE_CLEAR_TEXT,"Yes","No");
        PixUIMessageDialog* prompt = confirmation.get();
        confirmation->accepted = [this,prompt]() {
            prompt->dismiss();
            queue_clear();
        };
        dialog->present(std::move(confirmation));
    };
    browser->start_button->activate = [this,dialog]() { queue_start(*dialog); };
    browser->stop_button->activate = [this,dialog]() { queue_stop(*dialog); };
    browser->pause_button->activate = [this,dialog]() { queue_pause(*dialog); };
    browser->import_button->activate = [this,dialog]() { queue_import(*dialog); };
    browser->export_button->activate = [this,dialog]() { queue_export(*dialog); };
    main_form.present(std::move(browser));
}

bool Pixelina::queue_import(PixUIForm &owner, const std::string &path)
{
    if (path.empty()) {
        auto selection = std::make_unique<PixUIFileBrowser>(canvas);
        selection->title = UI_QUEUE_IMPORT_TEXT;
        selection->selected = [this,&owner](const std::string &path) { return queue_import(owner,path); };
        selection->refresh();
        owner.present(std::move(selection));
        return true;
    }
    if (busy && queue.active == UI_QUEUE_NO_TASK) {
        message("Another generation is running. Stop it before importing jobs.",&owner);
        return false;
    }
    queue_clear();
    if (busy) {
        queue_import_path = path;
        return true;
    }
    std::string directory,error;
    std::vector<PixRecord> parameters;
    if (!job_file_load(path,directory,parameters,error)) {
        message(error,&owner);
        return false;
    }
    if (!open_directory(directory)) return false;
    std::vector<PixQueueItem> imported;
    for (PixRecord &record : parameters) {
        PixQueueItem item;
        record.filename.clear();
        item.job.parameters = std::move(record);
        if (item.job.parameters.model == PIX_MODEL_QWEN_IMAGE) {
            item.job.edit_settings = item.job.parameters.edit_settings;
            PixEditSettings &s = item.job.edit_settings;
            s.diffuser = path_join(launch_directory,s.diffuser);
            s.encoder = path_join(launch_directory,s.encoder);
            s.vision = path_join(launch_directory,s.vision);
            s.vae = path_join(launch_directory,s.vae);
            for (const std::string &reference_path : item.job.parameters.references) {
                PixImage reference;
                if (!image_load(reference_path,reference,error)) {
                    message(error,&owner);
                    return false;
                }
                item.job.references.push_back(std::move(reference));
            }
        } else {
            item.job.generate_settings = item.job.parameters.settings;
            PixSettings &s = item.job.generate_settings;
            s.diffuser = path_join(launch_directory,s.diffuser);
            s.encoder = path_join(launch_directory,s.encoder);
            s.vae = path_join(launch_directory,s.vae);
        }
        item.source = queue_source_thumbnail(item.job.references);
        item.description = queue_job_description(item.job);
        imported.push_back(std::move(item));
    }
    queue.items.swap(imported);
    queue.revision++;
    status = "Imported jobs from " + path;
    layout();
    return true;
}

void Pixelina::queue_export(PixUIForm &owner)
{
    auto save = [this,&owner](const std::string &path) {
        std::vector<PixRecord> parameters;
        for (size_t i = 0; i < queue.items.size(); i++) {
            const PixQueueItem &item = queue.items[i];
            if (item.state == PIX_QUEUE_COMPLETE) continue;
            const PixJob &source = i == queue.active? job : item.job;
            if (source.references.size() != source.parameters.references.size()) {
                message("Save the reference image and add the job again before exporting it.",&owner);
                return;
            }
            parameters.push_back(source.parameters);
        }
        std::string error;
        if (!job_file_save(path,workdir,parameters,error)) message(error,&owner);
        else status = "Exported jobs to " + path;
    };
    auto selection = std::make_unique<PixUIFileBrowser>(canvas,path_join(PixUIFileBrowser::last_directory,"jobs.txt"));
    selection->title = UI_QUEUE_EXPORT_TEXT;
    selection->accept_caption = UI_SAVE_TEXT;
    selection->hint_text = "Saves unfinished jobs for later execution.";
    selection->show_listing = false;
    selection->selected = [this,&owner,save](const std::string &path) {
        if (path_kind(path) != PIX_PATH_MISSING) {
            auto confirmation = std::make_unique<PixUIMessageDialog>(canvas,UI_OVERWRITE_FILE_TEXT + path,PIXELINA_NAME,UI_OVERWRITE_TEXT,UI_CANCEL_TEXT);
            confirmation->accepted = [save,path]() { save(path); };
            owner.present(std::move(confirmation));
        } else save(path);
        return true;
    };
    owner.present(std::move(selection));
}

void Pixelina::queue_start(PixUIForm &owner)
{
    if (busy && stop_requested) {
        message(UI_QUEUE_STOPPING_ERROR,&owner);
        return;
    }
    if (busy) {
        if (queue.active == UI_QUEUE_NO_TASK) {
            message("Another generation is running. Stop it before starting the queue.",&owner);
            return;
        }
        if (!queue.pausing && !queue.paused) {
            message("The queue is already running.",&owner);
            return;
        }
        SDL_LockMutex(mutex);
        pause_requested = false;
        SDL_CondBroadcast(pause_condition);
        SDL_UnlockMutex(mutex);
        queue.pausing = queue.paused = false;
        queue.revision++;
        return;
    }
    for (const PixQueueItem &item : queue.items) {
        if (item.state == PIX_QUEUE_COMPLETE) continue;
        queue.pausing = queue.paused = false;
        queue.running = true;
        queue_next();
        return;
    }
    message("There are no unfinished jobs in the queue.",&owner);
}

void Pixelina::queue_pause(PixUIForm &owner)
{
    if (!queue.running) {
        message(UI_QUEUE_NOT_RUNNING_ERROR,&owner);
        return;
    }
    if (stop_requested) {
        message(UI_QUEUE_STOPPING_ERROR,&owner);
        return;
    }
    if (queue.pausing || queue.paused) {
        message("The queue is already paused or pausing.",&owner);
        return;
    }
    if (queue.active == UI_QUEUE_NO_TASK) {
        message(UI_QUEUE_NOT_RUNNING_ERROR,&owner);
        return;
    }
    SDL_LockMutex(mutex);
    pause_requested = true;
    SDL_UnlockMutex(mutex);
    queue.pausing = true;
    queue.revision++;
}

void Pixelina::queue_stop(PixUIForm &owner)
{
    if (!queue.running && queue.active == UI_QUEUE_NO_TASK) {
        message(UI_QUEUE_NOT_RUNNING_ERROR,&owner);
        return;
    }
    if (stop_requested && busy) {
        message(UI_QUEUE_STOPPING_ERROR,&owner);
        return;
    }
    queue.running = false;
    queue.pausing = queue.paused = false;
    if (!busy) status = UI_STOPPED_TEXT;
    if (queue.active != UI_QUEUE_NO_TASK) {
        stop_requested = true;
        SDL_LockMutex(mutex);
        pause_requested = false;
        SDL_CondBroadcast(pause_condition);
        SDL_UnlockMutex(mutex);
        status = UI_STOPPING_TEXT;
    }
    queue.revision++;
}

void Pixelina::queue_clear()
{
    queue_import_path.clear();
    queue.running = false;
    queue.pausing = queue.paused = false;
    if (queue.active != UI_QUEUE_NO_TASK) {
        queue_clear_requested = true;
        stop_requested = true;
        SDL_LockMutex(mutex);
        pause_requested = false;
        SDL_CondBroadcast(pause_condition);
        SDL_UnlockMutex(mutex);
        status = UI_STOPPING_TEXT;
    } else {
        queue.items.clear();
        queue.elapsed = 0;
        queue.reset_revision++;
        status = UI_QUEUE_CLEARED_TEXT;
    }
    queue.revision++;
}

void Pixelina::queue_next()
{
    if (!queue.running || busy) return;
    for (size_t i = 0; i < queue.items.size(); i++) {
        PixQueueItem &item = queue.items[i];
        if (item.state == PIX_QUEUE_COMPLETE) continue;
        if (queue.pausing || queue.paused) return;
        queue.active = i;
        item.state = PIX_QUEUE_RUNNING;
        item.progress = 0;
        item.seconds = 0;
        item.error.clear();
        queue.pausing = queue.paused = false;
        job = std::move(item.job);
        decode_on_the_go = false;
        queue.revision++;
        if (!start_generation()) {
            item.job = std::move(job);
            item.state = PIX_QUEUE_FAILED;
            item.error = dialog_message;
            queue.running = false;
            queue.active = UI_QUEUE_NO_TASK;
        }
        return;
    }
    queue.running = false;
    queue.pausing = queue.paused = false;
    queue.revision++;
    if (!queue.items.empty()) status = "Queue complete";
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
        value.texture = canvas.create_texture(reduced);
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
    file->add("Reload dir",[this]() { action(CMD_RELOAD_DIR); })->bind_shortcut('r',PIXUI_CTRL,"Ctrl+R");
    file->add("Open image",[this]() { action(CMD_OPEN_IMAGE); })->bind_shortcut('o',PIXUI_CTRL | PIXUI_SHIFT,"Ctrl+Shift+O");
    file->add(UI_SAVE_TEXT,[this]() { action(CMD_SAVE); })->bind_shortcut('s',PIXUI_CTRL,"Ctrl+S");
    file->add(UI_SAVE_AS_TEXT,[this]() { action(CMD_SAVE_AS); })->bind_shortcut('s',PIXUI_CTRL | PIXUI_SHIFT,"Ctrl+Shift+S");
    file->add("Quit",[this]() { action(CMD_QUIT); })->bind_shortcut('q',PIXUI_CTRL,"Ctrl+Q");
    PixUIMenu* edit = popups[1];
    edit->add("Clear",[this]() { action(CMD_CLEAR); });
    edit->add("Multi-Reference")->activate = [this]() {
        action(CMD_MULTI_REFERENCE);
    };
    PixUIMenuItem* generate = edit->add(UI_GENERATE_TEXT,[this]() { action(CMD_GENERATE); });
    generate->bind_shortcut(PIXUI_F5,0,"F5");
    generate->caption_state = [this]() { return busy? UI_STOP_TEXT : UI_GENERATE_TEXT; };
    PixUIMenuItem* add_queue = edit->add("Add to queue",[this]() { queue_add(); });
    add_queue->bind_shortcut(PIXUI_F5,PIXUI_SHIFT,"Shift+F5");
    add_queue->enabled_state = [this]() { return !busy; };
    PixUIMenu* preferences = edit->child(*edit->add("Preferences"));
    PixUIMenuItem* seed = preferences->add("Auto-seed",[this]() { action(CMD_AUTO_SEED); });
    seed->checkable = true;
    seed->checked_state = [this]() { return auto_seed; };
    PixUIMenuItem* database = preferences->add("Use global DB",[this]() { action(CMD_GLOBAL_DB); });
    database->checkable = true;
    database->checked_state = [this]() { return global_db; };
    PixUIMenu* themes = preferences->child(*preferences->add("Theme"));
    PixUIMenuItem* none = themes->add("<none>",[this]() { switch_theme({}); });
    none->checkable = true;
    none->checked_state = [this]() { return theme_path.empty(); };
    const std::string extension = UI_THEME_EXTENSION;
    for (const std::string &directory : {std::string(UI_THEME_DIRECTORY),user_directory}) {
        if (path_kind(directory) == PIX_PATH_MISSING) continue;
        std::vector<PixDirEntry> entries;
        std::string error;
        if (!directory_list(directory,entries,error)) {
            message(error);
            continue;
        }
        for (const PixDirEntry &entry : entries) {
            if (entry.directory || entry.name.size() <= extension.size() || entry.name.compare(entry.name.size() - extension.size(),extension.size(),extension)) continue;
            std::string path = path_join(directory,entry.name);
            if (path_kind(path) != PIX_PATH_FILE) continue;
            PixUIMenuItem* item = themes->add(entry.name.substr(0,entry.name.size() - extension.size()),[this,path]() { switch_theme(path); });
            item->checkable = true;
            item->checked_state = [this,path]() { return !theme_path.empty() && path_join(user_directory,theme_path) == path; };
        }
    }
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
    PixUIMenuItem* fullscreen = popups[2]->add("FullScreen",[this]() { action(CMD_FULLSCREEN); });
    fullscreen->bind_shortcut(PIXUI_F11,0,"F11");
    fullscreen->checkable = true;
    fullscreen->checked_state = [this]() { return main_form.fullscreen; };
    popups[2]->add("Bookmark",[this]() { action(CMD_BOOKMARK); })->bind_shortcut(PIXUI_F2,0,"F2");
    popups[2]->add("Unbookmark",[this]() { action(CMD_UNBOOKMARK); })->bind_shortcut(PIXUI_F2,PIXUI_SHIFT,"Shift+F2");
    popups[2]->add("Queue",[this]() { queue_view(); })->bind_shortcut(PIXUI_F6,0,"F6");
    scaled->bind_shortcut(PIXUI_F3,0,"F3");
    scaled->checkable = true;
    scaled->checked_state = [this]() { return main_form.image->scaled; };
    PixUIMenuItem* decode = popups[2]->add("Decode on-the-go",[this]() { action(CMD_DECODE); });
    decode->checkable = true;
    decode->checked_state = [this]() { return decode_on_the_go.load(); };
    decode->enabled_state = [this]() { return queue.active == UI_QUEUE_NO_TASK; };
    PixUIMenuItem* chat = popups[2]->add("Chat",[this]() { action(CMD_CHAT); });
    chat->enabled_state = [this]() { return !busy && image_valid(image); };
    popups[3]->add(PIXELINA_ABOUT_PREFIX PIXELINA_NAME,[this]() { action(CMD_ABOUT); });
    main_form.add_system_menu();
}

static void refresh_theme_controls(PixUIForm &form)
{
    // Only the persistent workspace is refreshed; its modal dialogs block theme selection.
    for (const auto &child : form.children()) {
        if (PixUILabel* label = dynamic_cast<PixUILabel*>(child.get())) label->color = pixui_theme.fonts[label->font_role].color;
        if (PixUIPanel* panel = dynamic_cast<PixUIPanel*>(child.get())) panel->color = UI_COLOR_FACE;
        if (PixUIGroupBox* group = dynamic_cast<PixUIGroupBox*>(child.get())) {
            group->color = UI_COLOR_LIGHT;
            group->wheel_step = UI_LINE_HEIGHT;
            if (dynamic_cast<PixTab*>(group)) group->inset_top = UI_TAB_HEIGHT + UI_BORDER_WIDTH;
            else if (!dynamic_cast<PixTabGroup*>(group)) group->inset_top = UI_FONT_HEIGHT + UI_TITLE_PADDING;
        }
        if (PixUIForm* nested = dynamic_cast<PixUIForm*>(child.get())) refresh_theme_controls(*nested);
    }
    form.repaint = true;
}

void Pixelina::switch_theme(const std::string &path)
{
    menus->dismiss();
    std::string error;
    if (!pixui_load_theme(path,error)) {
        message(error);
        return;
    }
    theme_path = path;
    refresh_theme_controls(main_form);
    main_form.menu_background->color = UI_COLOR_LIGHT;
    main_form.layout_settings();
    for (size_t i = 0; i < menus->items.size(); i++) {
        menus->items[i]->bounds = i < UI_TOP_MENU_COUNT? PixUIRect{int(i) * UI_TOP_MENU_WIDTH,UI_FRAME + UI_WINDOW_TITLE_HEIGHT,UI_TOP_MENU_WIDTH,UI_MENU_HEIGHT} : PixUIRect{UI_FRAME,UI_FRAME,UI_WINDOW_TITLE_HEIGHT,UI_WINDOW_TITLE_HEIGHT};
    }
    layout();
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
    bookmark_carousel_setup(*main_form.bookmarks,bookmarks,[this](const std::string &path) { return load_thumbnail(path); });
    main_form.bookmarks->selected = [this](size_t index) {
        if (busy || index >= bookmarks.size()) return false;
        open_image(bookmarks[index]);
        // Missing files remain selectable so their stale bookmarks can be removed.
        bookmark_selection = int(index);
        return true;
    };
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
        main_form.repaint = true;
        main_form.active = false;
        break;
    case SDL_WINDOWEVENT_LEAVE:
        canvas.release_capture();
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
    SDL_Texture* texture = canvas.create_texture(completed);
    bool uploaded = texture != nullptr;
    image = std::move(completed);
    image_record = job.parameters;
    image_has_record = true;
    image_source_path.clear();
    unsaved = true;
    main_form.carousel->select(-1);
    main_form.bookmarks->select(-1);
    bookmark_selection = -1;
    if (current) SDL_DestroyTexture(current);
    current = uploaded? texture : nullptr;
    canvas.set_main_image(&image,current);
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
        // Wake immediately for drag input without spinning while the pointer is still.
        if (canvas.captured == &main_form && SDL_WaitEventTimeout(&input,UI_FRAME_DELAY)) event(input);
        while (SDL_PollEvent(&input)) event(input);
        canvas.flush_window_commands();
        if (busy) {
            SDL_LockMutex(mutex);
            bool done = worker_done;
            bool pause_after = pause_requested;
            std::string stage = stop_requested? UI_STOPPING_TEXT : (worker_paused? UI_QUEUE_PAUSED_TEXT : (pause_requested? UI_QUEUE_PAUSING_TEXT : worker_stage));
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
            if (queue.active != UI_QUEUE_NO_TASK && (changed || done)) {
                PixQueueItem &item = queue.items[queue.active];
                item.progress = std::min(worker_progress,nextafterf(1,0));
                double paused = worker_pause_seconds;
                if (worker_paused) paused += double(SDL_GetPerformanceCounter() - worker_pause_started) / SDL_GetPerformanceFrequency();
                item.seconds = done? worker_seconds : std::max(0.0,double(canvas.ticks() - generation_started) / 1000 - paused);
                queue.paused = worker_paused;
                queue.pausing = pause_requested && !worker_paused;
                queue.revision++;
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
                bool queued = queue.active != UI_QUEUE_NO_TASK;
                if (stop_requested || !error.empty()) {
                    status = stop_requested? UI_STOPPED_TEXT : "Generation failed";
                    if (!stop_requested) message(error);
                } else {
                    generation_image(std::move(completed));
                    status = "Generated image (unsaved)";
                }
                if (queued) {
                    PixQueueItem &item = queue.items[queue.active];
                    queue.elapsed += generation_seconds;
                    bool saved = !stop_requested && error.empty() && save_image();
                    if (saved) {
                        item.state = PIX_QUEUE_COMPLETE;
                        item.progress = 1;
                        item.output = image_thumbnail(image);
                        item.filename = image_source_path;
                        job = PixJob();
                    } else {
                        item.job = std::move(job);
                        item.state = stop_requested? PIX_QUEUE_PENDING : PIX_QUEUE_FAILED;
                        item.error = stop_requested? std::string{} : (error.empty()? dialog_message : error);
                        if (stop_requested) item.progress = 0;
                        queue.running = false;
                    }
                    queue.active = UI_QUEUE_NO_TASK;
                    queue.pausing = false;
                    queue.paused = queue.running && pause_after;
                    if (queue.paused) status = UI_QUEUE_PAUSED_TEXT;
                    queue.revision++;
                    if (queue_clear_requested) {
                        queue.items.clear();
                        queue.elapsed = 0;
                        queue.reset_revision++;
                        queue_clear_requested = false;
                        status = UI_QUEUE_CLEARED_TEXT;
                    }
                    if (!queue_import_path.empty()) {
                        std::string path = std::move(queue_import_path);
                        queue_import(main_form,path);
                    }
                    queue_next();
                }
            }
            if (changed || done) layout();
        }
        if (main_form.needs_repaint()) paint();
        if (canvas.captured != &main_form) SDL_Delay(UI_FRAME_DELAY);
    }
    return 0;
}

PixelinaForm::PixelinaForm(PixUICanvas &context) : PixUIForm(context)
{
}

void PixelinaForm::create_controls()
{
    decorate(caption);
    menu_background = add<PixUIPanel>();
    menu_background->color = UI_COLOR_LIGHT;
    menu_background->border = false;
    const char* headings[] = {"Prompt","Settings","",UI_IMAGE_TITLE};
    for (int i = 0; i < UI_GROUP_COUNT; i++) {
        if (i == UI_SETTINGS_GROUP) {
            tabs = add<PixTab>();
            groups[i] = tabs;
        } else if (i == UI_CAROUSEL_GROUP) {
            image_tabs = add<PixTab>();
            groups[i] = image_tabs;
        } else groups[i] = add<PixUIGroupBox>(headings[i]);
    }
    PixUIGroupBox* prompt_group = groups[UI_PROMPT_GROUP];
    prompt = prompt_group->add<PixUIMemo>();
    prompt_group->place(*prompt,{0,0,0,0},PIXUI_STRETCH_X | PIXUI_STRETCH_Y);
    register_focus(*prompt);
    for (size_t page = 0; page < UI_SETTINGS_PAGES; page++) {
        PixTabGroup* settings = tabs->add_group(page == UI_GENERATE_PAGE? UI_GENERATE_TEXT : "Edit");
        pages[page] = settings;
        size_t count = page == UI_GENERATE_PAGE? UI_SETTINGS_COUNT : UI_EDIT_SETTINGS_COUNT;
        for (size_t i = 0; i < count; i++) {
            PixUILabel* label = settings->add<PixUILabel>(page == UI_GENERATE_PAGE? field_labels[i] : edit_field_labels[i]);
            settings_labels[page].push_back(label);
            PixUIEdit* field = settings->add<PixUIEdit>();
            if (page == UI_GENERATE_PAGE) fields[i] = field;
            else edit_fields[i] = field;
            register_focus(*field);
            bool dimension = i == UI_WIDTH_FIELD || i == UI_HEIGHT_FIELD || (page == UI_EDIT_PAGE && i == UI_EDIT_RESOLUTION);
            bool spin = dimension || i == UI_STEPS_FIELD;
            int button_width = spin? UI_SPIN_BUTTON_WIDTH : (i == UI_SEED_FIELD || i >= UI_MODEL_FIRST? UI_BROWSE_BUTTON_WIDTH : 0);
            if (!button_width) continue;
            if (spin) {
                for (int direction : {1,-1}) {
                    PixUIButton* button = settings->add<PixUIButton>(direction > 0? UI_CONTROL_UP : UI_CONTROL_DOWN);
                    spin_buttons.push_back(button);
                    int step = dimension? (page == UI_GENERATE_PAGE? PIXELINA_IMAGE_MULTIPLE : QWEN_IMAGE_MULTIPLE) : 1;
                    int minimum = dimension? step : (page == UI_EDIT_PAGE? UI_EDIT_MIN_STEPS : 1);
                    int maximum = !dimension && page == UI_EDIT_PAGE? PIXELINA_EDIT_MAX_STEPS : INT32_MAX;
                    button->activate = [this,field,direction,step,minimum,maximum]() {
                        if (!field->enabled) return;
                        uint64_t value;
                        if (field->model.text.empty() || !parse_unsigned(field->model.text,value) || value < uint64_t(minimum) || value > uint64_t(maximum)) return;
                        if (direction > 0) {
                            if (value > uint64_t(maximum - step)) return;
                            value += step;
                        } else {
                            if (value < uint64_t(minimum + step)) return;
                            value -= step;
                        }
                        field->set_text(std::to_string(value));
                        focus_control(*field);
                    };
                    if (direction > 0) field->increment = button->activate;
                    else field->decrement = button->activate;
                }
                continue;
            }
            PixUIButton* button = settings->add<PixUIButton>(i == UI_SEED_FIELD? "R" : "...");
            if (i == UI_SEED_FIELD) random_seed[page] = button;
            else if (page == UI_GENERATE_PAGE) browse[i] = button;
            else edit_browse[i] = button;
        }
    }
    preferences = tabs->add<PixUILabel>();
    generate = tabs->add<PixUIButton>();
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
    PixTabGroup* images_page = image_tabs->add_group("Images");
    carousel = images_page->add<PixUICarousel>();
    images_page->place(*carousel,{0,0,0,0},PIXUI_STRETCH_X | PIXUI_STRETCH_Y);
    PixTabGroup* bookmarks_page = image_tabs->add_group(UI_BOOKMARKS_TITLE);
    bookmarks = bookmarks_page->add<PixUICarousel>();
    bookmarks_page->place(*bookmarks,{0,0,0,0},PIXUI_STRETCH_X | PIXUI_STRETCH_Y);
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
    image = add<PixUIImageView>();
    image->placeholder = "Generate or select an image";
    layout_settings();
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
    status->bounds = {0,bounds.h - UI_STATUS_HEIGHT,bounds.w - (canvas.maximized()? 0 : UI_WINDOW_RESIZE_SIZE),UI_STATUS_HEIGHT};
    splitters[UI_SPLIT_LEFT]->bounds = {groups[UI_PROMPT_GROUP]->bounds.x + groups[UI_PROMPT_GROUP]->bounds.w,UI_WORK_TOP,UI_SPLITTER_WIDTH,bounds.h - UI_WORK_TOP - UI_STATUS_HEIGHT - UI_FRAME};
    splitters[UI_SPLIT_PROMPT]->bounds = {groups[UI_PROMPT_GROUP]->bounds.x,groups[UI_PROMPT_GROUP]->bounds.y + groups[UI_PROMPT_GROUP]->bounds.h,groups[UI_PROMPT_GROUP]->bounds.w,UI_SPLITTER_WIDTH};
    splitters[UI_SPLIT_CAROUSEL]->bounds = {groups[UI_CAROUSEL_GROUP]->bounds.x,groups[UI_CAROUSEL_GROUP]->bounds.y + groups[UI_CAROUSEL_GROUP]->bounds.h,groups[UI_CAROUSEL_GROUP]->bounds.w,UI_SPLITTER_WIDTH};
    PixUIGroupBox* group = groups[UI_IMAGE_GROUP];
    image->bounds = fullscreen? bounds : PixUIRect{group->bounds.x + group->inset_x,group->bounds.y + group->inset_top,std::max(0,group->bounds.w - group->inset_x * 2),std::max(0,group->bounds.h - group->inset_top - group->inset_bottom)};
    PixUIForm::layout();
}

void PixelinaForm::draw()
{
    if (!visible) return;
    PixUIForm::draw();
    // Menus paint above ordinary children; cover them with the expanded image.
    if (fullscreen && !modal()) image->draw();
}

void PixelinaForm::layout_settings()
{
    tabs->footer_height = UI_SETTINGS_FOOTER;
    size_t spin = 0;
    for (size_t page = 0; page < UI_SETTINGS_PAGES; page++) {
        PixTabGroup* settings = pages[page];
        size_t count = page == UI_GENERATE_PAGE? UI_SETTINGS_COUNT : UI_EDIT_SETTINGS_COUNT;
        settings->content_height = int(count) * UI_SETTINGS_ROW + UI_MARGIN;
        settings->content_width = UI_SETTINGS_MIN_WIDTH;
        settings->wheel_step = UI_SETTINGS_ROW;
        for (size_t i = 0; i < count; i++) {
            PixUIEdit* field = page == UI_GENERATE_PAGE? fields[i] : edit_fields[i];
            bool dimension = i == UI_WIDTH_FIELD || i == UI_HEIGHT_FIELD || (page == UI_EDIT_PAGE && i == UI_EDIT_RESOLUTION);
            bool has_spin = dimension || i == UI_STEPS_FIELD;
            int button_width = has_spin? UI_SPIN_BUTTON_WIDTH : (i == UI_SEED_FIELD || i >= UI_MODEL_FIRST? UI_BROWSE_BUTTON_WIDTH : 0);
            settings->place(*settings_labels[page][i],{UI_MARGIN - UI_BORDER_WIDTH,int(i) * UI_SETTINGS_ROW + UI_TEXT_PADDING,-(UI_MARGIN - UI_BORDER_WIDTH) * 2,UI_FONT_HEIGHT},PIXUI_STRETCH_X);
            settings->place(*field,{UI_MARGIN - UI_BORDER_WIDTH,int(i) * UI_SETTINGS_ROW + UI_FIELD_INPUT_Y,-(UI_MARGIN - UI_BORDER_WIDTH) * 2 - button_width,UI_FIELD_HEIGHT},PIXUI_STRETCH_X | PIXUI_FULLY_VISIBLE);
            if (!button_width) continue;
            if (has_spin) {
                for (int direction : {1,-1}) {
                    int y = int(i) * UI_SETTINGS_ROW + UI_FIELD_INPUT_Y + (direction > 0? 0 : UI_FIELD_HEIGHT / 2);
                    settings->place(*spin_buttons[spin++],{-(UI_MARGIN - UI_BORDER_WIDTH) - button_width,y,button_width,UI_FIELD_HEIGHT / 2},PIXUI_ALIGN_RIGHT | PIXUI_FULLY_VISIBLE);
                }
            } else {
                PixUIButton* button = i == UI_SEED_FIELD? random_seed[page] : (page == UI_GENERATE_PAGE? browse[i] : edit_browse[i]);
                settings->place(*button,{-(UI_MARGIN - UI_BORDER_WIDTH) - button_width,int(i) * UI_SETTINGS_ROW + UI_FIELD_INPUT_Y,button_width,UI_FIELD_HEIGHT},PIXUI_ALIGN_RIGHT | PIXUI_FULLY_VISIBLE);
            }
        }
    }
    tabs->place(*preferences,{UI_MARGIN - UI_BORDER_WIDTH,-UI_SETTINGS_FOOTER + UI_TEXT_PADDING,-(UI_MARGIN - UI_BORDER_WIDTH) * 2,UI_FONT_HEIGHT},PIXUI_STRETCH_X | PIXUI_ALIGN_BOTTOM);
    tabs->place(*generate,{UI_MARGIN - UI_BORDER_WIDTH,-UI_FIELD_HEIGHT - UI_MARGIN,-(UI_MARGIN - UI_BORDER_WIDTH) * 2,UI_FIELD_HEIGHT},PIXUI_STRETCH_X | PIXUI_ALIGN_BOTTOM);
}
