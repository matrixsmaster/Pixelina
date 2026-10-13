/* This file is a part of 'Pixelina' Image Generator, Editor, and Viewer.
 *
 * (C) Dmitry 'sciloaf' Solovyev, 2026
 *
 * License: MIT
 */
#ifndef PIXELINA_PIXUILIB_H
#define PIXELINA_PIXUILIB_H

#include <cstdint>
#include <functional>
#include <memory>
#include <map>
#include <utility>
#include <string>
#include <vector>

#define UI_UTF8_MASK 0xc0
#define UI_UTF8_CONTINUATION 0x80
#define UI_FONT_WIDTH 8
#define UI_FONT_HEIGHT 8
#define UI_LINE_HEIGHT 16
#define UI_MENU_ROW 18
#define UI_MENU_BORDER 2
#define UI_SCROLLBAR 16
#define COLOR_FACE 0xc0c0c0
#define COLOR_LIGHT 0xffffff
#define COLOR_SHADOW 0x808080
#define COLOR_DARK 0x000000
#define COLOR_TITLE 0x0000c0
#define COLOR_DESKTOP 0xffffff
#define COLOR_SELECTED 0x000080
#define COLOR_DISABLED 0x606060
#define UI_CONTENT_INSET 3
#define UI_TEXT_PADDING 4
#define UI_FONT_GLYPHS 128
#define UI_CARET_PERIOD 500
#define UI_ARROW_SIZE 4
#define UI_SYSTEM_BAR_HEIGHT 3
#define UI_CONTROL_UP "^"
#define UI_CONTROL_DOWN "v"
#define UI_CONTROL_SYSTEM "-"
#define UI_CONTROL_LEFT "<"
#define UI_CONTROL_RIGHT ">"
#define UI_BORDER_WIDTH 2
#define UI_LABEL_PADDING 8
#define UI_WHEEL_LINES 3
#define UI_MENU_PADDING 6
#define UI_MENU_CHECK_WIDTH 24
#define UI_CHECK_MARK "[x]"
#define UI_CLOSE_TEXT "Close"
#define UI_CANCEL_TEXT "Cancel"
#define UI_DIRECTORY_ERROR "Cannot open directory: "
#define UI_SAVE_TEXT "Save"
#define UI_SAVE_AS_TEXT "Save as"
#define UI_IMAGE_WHEEL_STEP 48
#define UI_PERCENT_SCALE 100
#define UI_MENU_GAP 8
#define UI_DOUBLE_CLICK_MS 500
#define UI_FIELD_HEIGHT 24
#define UI_TITLE_HEIGHT 24
#define UI_MARGIN 8
#define UI_MENU_HEIGHT 20
#define UI_WINDOW_TITLE_HEIGHT 24
#define UI_FRAME 4
#define UI_DIRECTORY_ROWS 12
#define UI_DIALOG_BROWSE_WIDTH 96
#define UI_DIALOG_BUTTON_WIDTH 80
#define UI_DIALOG_HEIGHT 400
#define UI_DIALOG_WIDTH 580
#define UI_ABOUT_SPACING 16
#define UI_ABOUT_LINE_COUNT 4
#define UI_ABOUT_ICON_SIZE 64
#define UI_ABOUT_HEIGHT 148
#define UI_ABOUT_WIDTH 472
#define UI_TOP_MENU_WIDTH 48
#define UI_MIN_HEIGHT 540
#define UI_MIN_WIDTH 760
#define UI_CAROUSEL_CELL 132
#define UI_CACHE_LIMIT 32
#define UI_CAROUSEL_EXTRA 2
#define UI_STATUS_PROGRESS_WIDTH 162
#define UI_STATUS_PROGRESS_OFFSET 170
#define UI_STATUS_TEXT_SPACE 180
#define UI_STATUS_PROGRESS_TOP 5
#define PIXUI_STRETCH_X 1
#define PIXUI_STRETCH_Y 2
#define PIXUI_ALIGN_RIGHT 4
#define PIXUI_ALIGN_BOTTOM 8
#define PIXUI_FULLY_VISIBLE 16
#define PIXUI_SPECIAL_KEY 256
#define PIXUI_CTRL 1
#define PIXUI_SHIFT 2
#define PIXUI_ALT 4
#define PIXUI_LEFT_BUTTON 1

struct PixUIRect {
    int x = 0,y = 0; // client-space origin
    int w = 0,h = 0; // extent in pixels
};

enum PixUIEventType {
    PIXUI_NONE,PIXUI_KEY,PIXUI_TEXT,PIXUI_DOWN,PIXUI_UP,PIXUI_MOTION,PIXUI_WHEEL
};

enum PixUIKey {
    PIXUI_RETURN = PIXUI_SPECIAL_KEY,PIXUI_ESCAPE,PIXUI_TAB,PIXUI_BACKSPACE,PIXUI_DELETE,
    PIXUI_LEFT,PIXUI_RIGHT,PIXUI_HOME,PIXUI_END,PIXUI_UP_KEY,PIXUI_DOWN_KEY,
    PIXUI_F3,PIXUI_F4,PIXUI_F5
};

struct PixUIEvent {
    PixUIEventType type = PIXUI_NONE; // normalized input category
    int key = 0,mod = 0; // portable key and modifier mask
    int x = 0,y = 0; // client pointer position, including wheel events
    int button = 0; // pointer button
    int wheel_x = 0,wheel_y = 0; // wheel increments
    std::string text; // UTF-8 text input
};

struct PixUIEditor {
    std::string text; // UTF-8 text buffer; the ASCII font renders unsupported characters as ?
    size_t cursor = 0; // caret byte offset
    size_t anchor = 0; // selection byte offset
    int scroll = 0; // first visible text line
    int horizontal = 0; // first visible glyph column
    bool reveal = true; // reveal caret only after editing or focus changes
};

struct PixUIThumbCache {
    void* texture = nullptr; // renderer-owned preview texture
    int width = 0; // preview dimensions
    int height = 0; // preview dimensions
    uint64_t used = 0; // LRU frame marker
    bool missing = false; // missing or undecodable saved image
};

struct PixUILayout {
    int x,y; // accumulated insertion point
    int spacing; // gap after each control
    PixUIRect row(int width, int height);
    PixUIRect column(int width, int height);
};

enum PixUIWindowCommand {
    PIXUI_RESTORE,PIXUI_MINIMIZE,PIXUI_MAXIMIZE,PIXUI_MOVE,PIXUI_RESIZE,PIXUI_CLOSE
};

enum PixUIHit {
    PIXUI_HIT_NORMAL,PIXUI_HIT_MOVE,PIXUI_HIT_TOPLEFT,PIXUI_HIT_TOPRIGHT,
    PIXUI_HIT_BOTTOMLEFT,PIXUI_HIT_BOTTOMRIGHT,PIXUI_HIT_TOP,PIXUI_HIT_BOTTOM,
    PIXUI_HIT_LEFT,PIXUI_HIT_RIGHT
};

struct PixUIFileEntry {
    std::string name; // directory entry caption
    bool directory = false; // entry navigates into a directory
};

enum PixUIPathKind {
    PIXUI_PATH_MISSING,PIXUI_PATH_DIRECTORY,PIXUI_PATH_FILE,PIXUI_PATH_OTHER
};

struct PixUIDialogLayout {
    PixUIRect box,label,path,browse,up,list,hint,ok,cancel; // original modal control positions
};

class PixUIObject;
class PixUIForm;
class PixUIMenu;
class PixUIDialog;

class PixUICanvas
{
public:
    virtual ~PixUICanvas() = default;

    virtual void fill(const PixUIRect &rect, unsigned color) = 0;
    virtual void point(int x, int y, unsigned color) = 0;
    virtual void image(void* texture, const PixUIRect &rect) = 0;
    virtual bool clip(PixUIRect &rect) const = 0;
    virtual void set_clip(const PixUIRect* rect) = 0;
    virtual uint64_t ticks() const = 0;
    virtual std::string clipboard() const = 0;
    virtual void clipboard(const std::string &text) = 0;
    virtual void text_input(const PixUIRect &rect) = 0;
    virtual PixUIRect window_rect() const = 0;
    virtual void window_command(PixUIWindowCommand command, const PixUIRect &rect) = 0;
    virtual bool maximized() const = 0;
    virtual void window_title(const std::string &title) = 0;
    void bevel(const PixUIRect &rect, bool inset, bool fill = true);
    void text(int x, int y, const std::string &value, unsigned color);
    void button(const PixUIRect &rect, const std::string &label, bool pressed);
    bool event(PixUIForm &form, const PixUIEvent &input);
    void release_capture();

    int pointer_x = 0,pointer_y = 0; // latest pointer position for window commands
    PixUIObject* captured = nullptr; // control receiving motion and release until capture ends
};

// Controls borrow the canvas; forms own controls and preserve them between frames.
class PixUIObject
{
public:
    explicit PixUIObject(PixUICanvas &context);
    virtual ~PixUIObject();

    virtual void draw() = 0;
    virtual void layout();
    virtual bool event(const PixUIEvent &input);
    virtual void cancel_capture();
    bool interactive() const;

    PixUIObject* parent = nullptr; // containing form or composite control, without ownership
    PixUICanvas &canvas; // shared rendering and pointer-capture state
    PixUIRect bounds = {}; // client-space geometry used by drawing and hit testing
    bool visible = true; // hidden objects do not draw or receive events
    bool enabled = true; // disabled objects draw but do not receive events

private:
    friend class PixUIForm;
    friend class PixUIGroupBox;
    bool layout_visible = true; // container clipping preserves explicit visibility
};

struct PixUIPlacement {
    PixUIObject* child; // borrowed child owned by the group
    PixUIRect rect; // position and dimensions relative to the client area
    int flags; // stretching, anchoring and visibility constraints
};

class PixUIControl : public PixUIObject
{
public:
    explicit PixUIControl(PixUICanvas &context);

    bool event(const PixUIEvent &input) override;

    std::function<void(const PixUIEvent &)> pointer; // owner callback with the original pointer event
    std::function<void(const PixUIEvent &)> wheel; // owner scrolling callback within the control bounds
    std::function<void()> activate; // owner activation action; Buttons invoke it on release
};

class PixUIPanel : public PixUIControl
{
public:
    explicit PixUIPanel(PixUICanvas &context);

    void draw() override;

    unsigned color = COLOR_FACE; // background color
    bool background = true; // paint a background before the border
    bool border = true; // draw the retro bevel
    bool inset = false; // sunken border when true
};

class PixUILabel : public PixUIControl
{
public:
    explicit PixUILabel(PixUICanvas &context, const std::string &text = "");

    void draw() override;

    std::string caption; // displayed UTF-8 label
    unsigned color = COLOR_DARK; // text color
    unsigned background_color = COLOR_LIGHT; // optional label background
    bool background = false; // fill the label area before drawing
    bool centered = false; // center text horizontally and vertically
    bool wrap = false; // wrap caption to the available width
    bool underline = false; // underline the first glyph for a menu accelerator
};

class PixUIButton : public PixUILabel
{
public:
    explicit PixUIButton(PixUICanvas &context, const std::string &text = "");

    void cancel_capture() override;
    void draw() override;
    bool event(const PixUIEvent &input) override;

    bool default_action = false; // consume Return when selected by the containing form
    bool pressed = false; // mouse capture selects the inset bevel
};

class PixUIImage : public PixUIControl
{
public:
    explicit PixUIImage(PixUICanvas &context);

    void draw() override;
    bool event(const PixUIEvent &input) override;
    void layout();

    void* texture = nullptr; // borrowed texture, owned by the application
    int width = 0,height = 0; // source image dimensions
    bool scaled = true; // fit the image into bounds, otherwise use native dimensions
    int wheel_step = UI_IMAGE_WHEEL_STEP; // native image wheel increment
    int offset_x = 0,offset_y = 0; // native image scroll offsets
    std::string placeholder; // label for an empty image
};

class PixUIThumb : public PixUIImage
{
public:
    explicit PixUIThumb(PixUICanvas &context);

    void draw() override;

    std::string caption; // filename beneath the preview
    bool selected = false; // selection highlight
    std::function<PixUIThumbCache&()> source; // bounded application preview cache
};

class PixUIScrollbar : public PixUIControl
{
public:
    explicit PixUIScrollbar(PixUICanvas &context);

    void draw() override;
    bool event(const PixUIEvent &input) override;

    bool vertical = false; // scrolling axis
    int page = 1,total = 1; // visible and total content extents
    int step = 1; // arrow increment in content units
    int* position = nullptr; // owner scroll offset
    std::function<void()> capture; // owner notification when track dragging starts
    std::function<void()> changed; // owner notification after scrolling
};

class PixUIMemo : public PixUIControl
{
private:
    PixUIEditor editor; // default owned editing state, constructed before the model reference

public:
    explicit PixUIMemo(PixUICanvas &context, PixUIEditor* state = nullptr);
    PixUIMemo(PixUICanvas &context, PixUIEditor &state);
    ~PixUIMemo() override;

    void set_text(const std::string &text);
    void draw() override;
    bool event(const PixUIEvent &input) override;
    void layout();

    PixUIEditor &model; // persistent UTF-8 text, caret and selection state
    bool active = false; // receives text and keyboard input when focused
    PixUIForm* focus_owner = nullptr; // form controlling editor focus and Tab navigation
    bool multiline = true; // word wrapping and vertical scrolling
    PixUIScrollbar scroll; // child scrollbar shares the memo's scroll state

private:
    void edit(const PixUIEvent &event);
    void draw_editor();
    void place_cursor(int x, int y, int modifiers);
};

class PixUIEdit : public PixUIMemo
{
public:
    explicit PixUIEdit(PixUICanvas &context, PixUIEditor* state = nullptr);
    PixUIEdit(PixUICanvas &context, PixUIEditor &state);
};

class PixUIMenuItem : public PixUILabel
{
public:
    explicit PixUIMenuItem(PixUICanvas &context, const std::string &text = "");
    ~PixUIMenuItem() override;

    void activate_item();
    bool accelerator(const PixUIEvent &input);
    void bind_shortcut(int key, int modifiers, const std::string &label);
    void update();
    void draw() override;

    std::string shortcut; // right-aligned key shortcut or submenu arrow
    bool checkable = false; // reserve a checkmark column
    bool checked = false; // display the checkmark
    bool highlighted = false; // selection owned by the containing menu
    int shortcut_key = 0,shortcut_mod = 0; // accelerator owned by this item
    std::function<bool()> checked_state; // application value queried by the item
    std::function<std::string()> caption_state; // dynamic action caption
    std::unique_ptr<PixUIMenu> submenu; // owned submenu tree
    bool show_title = true; // draw a top-level menu-bar label
};

class PixUIMenu : public PixUIControl
{
public:
    explicit PixUIMenu(PixUICanvas &context);

    PixUIMenuItem* add(const std::string &caption, const std::function<void()> &action = {});
    PixUIMenu* child(PixUIMenuItem &item);
    void open(size_t index);
    void dismiss();
    bool opened() const;
    void layout();
    void draw() override;
    bool event(const PixUIEvent &input) override;
    bool accelerator(const PixUIEvent &input);
    bool contains(int x, int y) const;

    bool bar = false; // top-level menu bar rather than a popup
    std::vector<std::unique_ptr<PixUIMenuItem>> items; // persistent owned items

private:
    int selected = -1; // open submenu or highlighted popup row
    int client_width = 0; // right edge used to flip overflowing submenus

    void layout_popup(int x, int y, int width);
};

class PixUIForm : public PixUIControl
{
public:
    explicit PixUIForm(PixUICanvas &context);
    ~PixUIForm() override;

    void decorate(const std::string &caption);
    PixUIMenu* create_menu();
    void add_system_menu();
    void layout_window(const std::string &caption);
    void command(PixUIWindowCommand command);
    void title_event(int x, int y, bool system = false);
    PixUIHit hit_test(int x, int y) const;
    void cancel_capture() override;
    void register_focus(PixUIMemo &memo);
    void unregister_focus(PixUIMemo &memo);
    void focus_control(PixUIMemo &memo);
    template<class T, class... Args> T* add(Args&&... args)
    {
        auto element = std::make_unique<T>(canvas,std::forward<Args>(args)...);
        element->parent = this;
        T* result = element.get();
        elements.push_back(std::move(element));
        return result;
    }
    void layout() override;
    void present(std::unique_ptr<PixUIDialog> dialog);
    void dismiss_dialog();
    bool modal() const;
    virtual void clear();
    virtual void remove(PixUIObject &child);
    const std::vector<std::unique_ptr<PixUIObject>> &children() const;
    void draw() override;
    bool event(const PixUIEvent &input) override;

    PixUIButton* default_button = nullptr; // modal Enter action, borrowed from elements
    bool clip_children = true; // clip nested controls and pointer input to the form
    bool active = true; // platform focus state for the window title
    std::function<void()> close; // application window lifetime callback
    std::function<void()> system_menu; // opens the owned command menu
    std::function<void(PixUIMemo*)> tab_changed; // application scrolls the newly focused field into view

protected:
    std::vector<std::unique_ptr<PixUIObject>> elements; // owned controls in paint order
    PixUIRect client; // optional clipping rectangle for container children
    bool clip_client = false; // use client instead of the outer frame for clipping

private:
    friend class PixUICanvas;
    friend class PixUIMemo;
    bool modal_blocked = false; // owned modal state disables window manipulation
    std::unique_ptr<PixUIDialog> modal_dialog; // owned active modal, destroyed outside input dispatch
    std::unique_ptr<PixUIDialog> pending_dialog; // deferred replacement requested by a callback
    PixUIMemo* saved_focus = nullptr; // editor restored when the modal closes
    PixUIMenu* menu_bar = nullptr; // owned menu root drawn above client controls
    PixUIPanel* frame = nullptr; // optional window frame
    PixUIPanel* title = nullptr; // title-bar background
    PixUIForm* title_clip = nullptr; // title caption clipping container
    PixUILabel* title_label = nullptr; // window caption
    PixUIButton* system = nullptr; // system-menu button
    PixUIButton* minimize = nullptr; // minimize command button
    PixUIButton* maximize = nullptr; // maximize/restore command button
    std::string window_caption; // last title sent to the platform
    std::vector<PixUIMemo*> tab_order; // registered editors in keyboard navigation order
    PixUIMemo* focused = nullptr; // currently selected editor
    PixUIWindowCommand window_drag = PIXUI_CLOSE; // active move/resize, Close denotes no capture
    int drag_x = 0,drag_y = 0; // pointer position at capture
    PixUIRect drag_window; // window geometry at resize start
    PixUIHit resize_edge = PIXUI_HIT_BOTTOMRIGHT; // captured window border
    uint64_t click_time = 0; // previous title click timestamp
    int click_x = 0,click_y = 0; // previous title click coordinates

    void prepare_modal();
};

class PixUIGroupBox : public PixUIForm
{
public:
    explicit PixUIGroupBox(PixUICanvas &context, const std::string &text = "");

    void place(PixUIObject &child, const PixUIRect &rect, int flags = 0);
    void reveal(PixUIObject &child);
    void clear() override;
    void remove(PixUIObject &child) override;
    void layout() override;
    void draw() override;
    bool event(const PixUIEvent &input) override;

    std::string caption; // group heading
    unsigned color = COLOR_LIGHT; // client background
    bool border = true; // draw the inset group frame
    int inset_x = UI_BORDER_WIDTH,inset_top = UI_TITLE_HEIGHT,inset_bottom = UI_BORDER_WIDTH; // client margins
    int content_height = 0; // scrollable content extent; zero disables scrolling
    int wheel_step = UI_LINE_HEIGHT; // vertical scroll increment

private:
    std::vector<PixUIPlacement> placements; // persistent child layout constraints
    int scroll = 0; // owned vertical content offset
};

class PixUIImageView : public PixUIForm
{
public:
    explicit PixUIImageView(PixUICanvas &context);

    void set_image(void* texture, int width, int height);
    void layout() override;

    bool scaled = true; // fit image; native mode owns both scrollbars
    std::string placeholder; // empty image caption

private:
    PixUIImage* image; // owned image and native offsets
    PixUIScrollbar* horizontal; // owned horizontal image scrollbar
    PixUIScrollbar* vertical; // owned vertical image scrollbar
};

class PixUICarousel : public PixUIForm
{
public:
    explicit PixUICarousel(PixUICanvas &context);
    ~PixUICarousel() override;

    void items_changed(size_t count, bool reset = false);
    void select(int index);
    void layout() override;

    int cell_width = UI_CAROUSEL_CELL; // preview slot width
    size_t cache_limit = UI_CACHE_LIMIT; // maximum resident thumbnail resources
    std::function<std::string(size_t)> caption; // backing record caption provider
    std::function<PixUIThumbCache(size_t)> load; // backend creates an opaque thumbnail
    std::function<void(void*)> release; // backend destroys an opaque thumbnail
    std::function<bool(size_t)> selected; // application accepts the activated record

private:
    size_t count = 0,first = 0,last = 0; // record count and visible interval
    int selection = -1,offset = 0; // owned selection and horizontal scroll position
    bool invalid = true; // backing records changed
    uint64_t clock = 0; // thumbnail access sequence for LRU eviction
    std::vector<PixUIThumb*> thumbnails; // borrowed visible children owned by previews
    PixUILabel* empty = nullptr; // borrowed empty-state label owned by previews
    PixUIForm* previews; // clipped visible thumbnails
    PixUIScrollbar* scrollbar; // owned carousel scrollbar
    std::map<size_t,PixUIThumbCache> cache; // bounded owned backend resources

    PixUIThumbCache &thumbnail(size_t index);
    void trim_cache(size_t limit);
};

class PixUIProgress : public PixUILabel
{
public:
    explicit PixUIProgress(PixUICanvas &context);

    void draw() override;

    float progress = 0; // normalized completed fraction
    bool busy = false; // use filled progress and percentage while busy
};

class PixUIStatusBar : public PixUIForm
{
public:
    explicit PixUIStatusBar(PixUICanvas &context);

    void layout() override;
    void draw() override;

    std::string text; // application status text
    bool busy = false; // progress is active
    float progress = 0; // completed fraction
    int margin = UI_MARGIN,progress_width = UI_STATUS_PROGRESS_WIDTH,progress_offset = UI_STATUS_PROGRESS_OFFSET,text_space = UI_STATUS_TEXT_SPACE,progress_top = UI_STATUS_PROGRESS_TOP; // configured original geometry

private:
    PixUIForm* text_clip; // owned clipped status label container
    PixUILabel* label; // owned status text
    PixUIProgress* meter; // owned progress indicator
};

class PixUISplitter : public PixUIPanel
{
public:
    explicit PixUISplitter(PixUICanvas &context);

    bool event(const PixUIEvent &input) override;

    std::function<void(int,int)> moved; // application updates its pane geometry
};

class PixUIList : public PixUIControl
{
public:
    explicit PixUIList(PixUICanvas &context);

    void draw() override;
    bool event(const PixUIEvent &input) override;
    void layout();

    size_t count = 0; // owner item count without copying the backing collection
    std::function<std::string(size_t)> item; // captions fetched only for visible rows
    int* position = nullptr; // first visible row
    int page = 1; // visible row count
    std::function<void(size_t)> select; // owner callback with selected item index
    PixUIScrollbar scroll; // child scrollbar for the item list
};

class PixUIDialog : public PixUIForm
{
public:
    explicit PixUIDialog(PixUICanvas &context, const std::string &caption, const std::string &accept = "", const std::string &cancel = UI_CANCEL_TEXT);

    void center(const PixUIRect &client);
    void layout() override;
    void dismiss();
    virtual void accept();
    bool event(const PixUIEvent &input) override;

    std::string title; // configurable dialog title
    std::string accept_caption; // configurable primary button text
    std::function<void()> accepted; // application action after the primary button

protected:
    PixUIPanel* border; // dialog frame
    PixUIPanel* body; // white client background
    PixUILabel* title_label; // owned modal title caption
    PixUIButton* close_button; // system close control
    PixUIButton* accept_button = nullptr; // optional primary action
    PixUIButton* cancel_button; // dismiss action
    int dialog_width = UI_DIALOG_WIDTH,dialog_height = UI_DIALOG_HEIGHT; // original dialog dimensions

    PixUIDialogLayout controls() const;
};

class PixUIMessageDialog : public PixUIDialog
{
public:
    explicit PixUIMessageDialog(PixUICanvas &context, const std::string &message, const std::string &caption, const std::string &accept = "", const std::string &cancel = "OK");

    void layout() override;

private:
    PixUILabel* message; // wrapped message text
};

class PixUIAboutDialog : public PixUIDialog
{
public:
    explicit PixUIAboutDialog(PixUICanvas &context, const std::string &caption, void* texture, int width, int height, const std::vector<std::string> &lines);

    void layout() override;

private:
    PixUIImage* icon; // borrowed application portrait
    std::vector<PixUILabel*> lines; // owned About captions
};

class PixUIFileSystem
{
public:
    virtual ~PixUIFileSystem() = default;

    virtual bool read_directory(const std::string &path, std::vector<PixUIFileEntry> &entries, std::string &error) = 0;
    virtual std::string normalize(const std::string &path) = 0;
    virtual PixUIPathKind path_kind(const std::string &path) = 0;
    virtual std::string join_path(const std::string &directory, const std::string &name) = 0;
    virtual std::string parent_path(const std::string &path) = 0;
};

class PixUIFileBrowser : public PixUIDialog
{
public:
    explicit PixUIFileBrowser(PixUICanvas &context, PixUIFileSystem &provider, const std::string &initial = "");

    void refresh();
    void accept() override;
    void set_path(const std::string &value);
    void layout() override;

    bool dir_selector = false; // select directories instead of file paths
    bool show_listing = true; // show navigation controls and directory listing
    std::string path_label; // configurable selection caption
    std::string hint_text; // configurable selection instructions
    std::function<bool(const std::string &)> selected; // returns true when the application accepts the path
    std::function<void(const std::string &)> error; // application status reporting

private:
    PixUIFileSystem &filesystem; // borrowed platform filesystem provider
    PixUIEditor path; // owned editable selection
    std::vector<PixUIFileEntry> entries; // owned current directory listing
    std::string directory; // directory represented by the listing
    int position = 0; // list scroll offset
    PixUILabel* label; // path caption
    PixUIEdit* edit; // owned path input control
    PixUILabel* hint; // selection instructions
    PixUIButton* browse = nullptr; // refresh listing button
    PixUIButton* up = nullptr; // parent-directory button
    PixUIList* list = nullptr; // virtualized directory list

    void select(size_t index);
};

bool pixui_intersect(const PixUIRect &a, const PixUIRect &b, PixUIRect &result);
bool pixui_inside(const PixUIRect &rect, int x, int y);
int pixui_scrollbar_position(int coordinate, int origin, int length, int page, int total);
size_t pixui_next_character(const std::string &text, size_t offset);
std::vector<size_t> pixui_editor_lines(const std::string &value, int columns);

#endif
