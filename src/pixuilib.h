/* This file is a part of 'Pixelina' Image Generator, Editor, and Viewer.
 *
 * (C) Dmitry 'sciloaf' Solovyev, 2026
 *
 * License: MIT
 */
#ifndef PIXELINA_PIXUILIB_H
#define PIXELINA_PIXUILIB_H

#include "os.h"
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
#define UI_OK_TEXT "OK"
#define UI_DIRECTORY_ERROR "Cannot open directory: "
#define UI_SAVE_TEXT "Save"
#define UI_SAVE_AS_TEXT "Save as"
#define UI_IMAGE_WHEEL_STEP 48
#define UI_PERCENT_SCALE 100
#define UI_MENU_GAP 8
#define UI_DOUBLE_CLICK_MS 500
#define UI_FIELD_HEIGHT 24
#define UI_TITLE_HEIGHT 24
#define UI_TAB_HEIGHT UI_FIELD_HEIGHT
#define UI_TAB_PADDING UI_MARGIN
#define UI_TAB_SLOPE 4
#define UI_TAB_RAISE 2
#define UI_MARGIN 8
#define UI_MENU_HEIGHT 20
#define UI_WINDOW_TITLE_HEIGHT 24
#define UI_FRAME 4
#define UI_DIRECTORY_ROWS 12
#define UI_DIALOG_BROWSE_WIDTH 96
#define UI_DIALOG_BUTTON_WIDTH 80
#define UI_DIALOG_HEIGHT 400
#define UI_DIALOG_WIDTH 580
#define UI_DIALOG_RESIZE_TEXT "\\"
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
#define UI_STATUS_COUNT_WIDTH 104
#define UI_STATUS_TIME_WIDTH 160
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
    virtual void pointer_position(int &x, int &y) const;
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
    virtual bool needs_repaint();
    bool interactive() const;

    PixUIObject* parent = nullptr; // containing form or composite control, without ownership
    PixUICanvas &canvas; // shared rendering and pointer-capture state
    PixUIRect bounds = {}; // client-space geometry used by drawing and hit testing
    bool repaint = true; // changed control requests a complete frame
    bool visible = true; // hidden objects do not draw or receive events
    bool enabled = true; // disabled objects draw but do not receive events

private:
    friend class PixUIForm;
    friend class PixUIGroupBox;
    bool layout_visible = true; // container clipping preserves explicit visibility
    PixUIRect painted_bounds; // geometry observed by the invalidation scan
    bool painted_visible = true,painted_enabled = true; // visibility and enabled state observed by the scan
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
    bool needs_repaint() override;

    std::string caption; // displayed UTF-8 label
    unsigned color = COLOR_DARK; // text color
    unsigned background_color = COLOR_LIGHT; // optional label background
    bool background = false; // fill the label area before drawing
    bool centered = false; // center text horizontally and vertically
    bool wrap = false; // wrap caption to the available width
    bool underline = false; // underline the first glyph for a menu accelerator

private:
    std::string painted_caption; // last caption observed by the invalidation scan
    unsigned painted_color = COLOR_DARK; // last text color observed by the scan
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

protected:
    void draw_image(const PixUIRect &rect);
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
    void scroll_to_end();
    bool needs_repaint() override;
    void draw() override;
    bool event(const PixUIEvent &input) override;
    void layout();

    PixUIEditor &model; // persistent UTF-8 text, caret and selection state
    bool active = false; // receives text and keyboard input when focused
    PixUIForm* focus_owner = nullptr; // form controlling editor focus and Tab navigation
    bool multiline = true; // word wrapping and vertical scrolling
    bool read_only = false; // permit navigation and copying while preventing text changes
    PixUIScrollbar scroll; // child scrollbar shares the memo's scroll state

private:
    uint64_t caret_phase = 0; // last drawn caret blink interval
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
    bool check_column = false; // containing popup reserves the column for every row
    bool checked = false; // display the checkmark
    bool highlighted = false; // selection owned by the containing menu
    int shortcut_key = 0,shortcut_mod = 0; // accelerator owned by this item
    std::function<bool()> checked_state; // application value queried by the item
    std::function<bool()> enabled_state; // dynamic availability queried before menu input and drawing
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
    bool needs_repaint() override;

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
    void focus_object(PixUIObject &control);
    bool needs_repaint() override;
    template<class T, class... Args> T* add(Args&&... args)
    {
        auto element = std::make_unique<T>(canvas,std::forward<Args>(args)...);
        element->parent = this;
        T* result = element.get();
        elements.push_back(std::move(element));
        repaint = true;
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
    friend class PixUICarousel;
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
    PixUIObject* keyboard_focus = nullptr; // focused non-editor, borrowed from owned descendants
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
    unsigned color = COLOR_LIGHT; // white group and tab-page background
    bool border = true; // draw the outlined group frame
    int inset_x = UI_MARGIN,inset_top = UI_TITLE_HEIGHT,inset_bottom = UI_MARGIN; // client margins
    int content_width = 0,content_height = 0; // minimum content extent, also inferred from fixed placements
    int wheel_step = UI_LINE_HEIGHT; // vertical scroll increment

private:
    std::vector<PixUIPlacement> placements; // persistent child layout constraints
    int scroll = 0; // owned vertical content offset
    int offset_x = 0; // owned horizontal content offset
    PixUIScrollbar horizontal,vertical; // scrollbars outside the clipped client area
};

class PixTabGroup : public PixUIGroupBox
{
public:
    PixTabGroup(PixUICanvas &context, const std::string &text);

    std::string title; // heading owned by the tab strip
};

class PixTab : public PixUIGroupBox
{
public:
    explicit PixTab(PixUICanvas &context);

    PixTabGroup* add_group(const std::string &title);
    bool select(size_t index);
    size_t selected() const;
    void layout() override;
    void draw() override;
    bool event(const PixUIEvent &input) override;
    void remove(PixUIObject &child) override;
    void clear() override;

    int footer_height = 0; // reserved common controls below the page body
    bool selection_enabled = true; // page switching availability, independent of footer actions
    std::function<void(size_t)> changed; // notification after page visibility changes

private:
    std::vector<PixTabGroup*> pages; // ordered borrowed pages owned by inherited elements
    std::vector<PixUIRect> headers; // clipped tab hit rectangles
    size_t active_page = SIZE_MAX; // selected page, or no page
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

    bool event(const PixUIEvent &input) override;
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

    void set_status(const std::string &message, const std::string &count, const std::string &timing, bool working, float fraction);

    std::string text; // application status text
    bool busy = false; // progress is active
    float progress = 0; // completed fraction
    int progress_width = UI_STATUS_PROGRESS_WIDTH,progress_offset = UI_STATUS_PROGRESS_OFFSET,progress_top = UI_STATUS_PROGRESS_TOP; // configured original geometry

private:
    std::string count_text,timing_text; // image count and elapsed time or ETA panels
    PixUIForm* text_clip; // owned clipped status label container
    PixUIForm* count_clip; // owned clipped image count panel
    PixUIForm* timing_clip; // owned clipped timing panel
    PixUILabel* count_label; // owned count label
    PixUILabel* timing_label; // owned timing label
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
    void cancel_capture() override;

    std::string title; // configurable dialog title
    std::string accept_caption; // configurable primary button text
    std::function<void()> accepted; // application action after the primary button
    bool resizable = false; // allow dragging the bottom-right sizing handle
    int minimum_width = UI_DIALOG_WIDTH,minimum_height = UI_DIALOG_HEIGHT; // lower bounds for interactive resizing

protected:
    PixUIPanel* border; // dialog frame
    PixUIPanel* body; // white client background
    PixUILabel* title_label; // owned modal title caption
    PixUIButton* close_button; // system close control
    PixUIButton* accept_button = nullptr; // optional primary action
    PixUIButton* cancel_button; // dismiss action
    int dialog_width = UI_DIALOG_WIDTH,dialog_height = UI_DIALOG_HEIGHT; // original dialog dimensions

    PixUIDialogLayout controls() const;

private:
    PixUILabel* resize_handle; // optional bottom-right sizing grip
    PixUIRect available; // latest containing client rectangle
    bool positioned = false,resizing = false; // preserve resized placement and pointer capture state
    int resize_x = 0,resize_y = 0,resize_width = 0,resize_height = 0; // pointer and dimensions at drag start
};

class PixUIMessageDialog : public PixUIDialog
{
public:
    explicit PixUIMessageDialog(PixUICanvas &context, const std::string &message, const std::string &caption, const std::string &accept = "", const std::string &cancel = UI_OK_TEXT);

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

class PixUIFileBrowser : public PixUIDialog
{
public:
    explicit PixUIFileBrowser(PixUICanvas &context, const std::string &initial = "");

    void refresh();
    void accept() override;
    void set_path(const std::string &value);
    void layout() override;

    static std::string last_directory; // last directory visited by any browser
    bool dir_selector = false; // select directories instead of file paths
    bool show_listing = true; // show navigation controls and directory listing
    std::string path_label; // configurable selection caption
    std::string hint_text; // configurable selection instructions
    std::function<bool(const std::string &)> selected; // returns true when the application accepts the path
    std::function<void(const std::string &)> error; // application status reporting

private:
    PixUIEditor path; // owned editable selection
    std::vector<PixDirEntry> entries; // owned current directory listing
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
