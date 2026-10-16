/* This file is a part of 'Pixelina' Image Generator, Editor, and Viewer.
 *
 * (C) Dmitry 'sciloaf' Solovyev, 2026
 *
 * License: MIT
 */
#include "dlg_bookimg.h"
#include "storage.h"
#include <SDL.h>
#include <algorithm>

void bookmark_carousel_setup(PixUICarousel &carousel, const std::vector<std::string> &paths, const std::function<PixUIThumbCache(const std::string &)> &load)
{
    carousel.caption = [&paths](size_t index) {
        return paths[index].substr(paths[index].find_last_of('/') + 1);
    };
    carousel.load = [&paths,load](size_t index) {
        return load? load(paths[index]) : PixUIThumbCache{};
    };
    carousel.release = [](void* texture) {
        SDL_DestroyTexture(static_cast<SDL_Texture*>(texture));
    };
    carousel.cache_limit = PIXELINA_THUMB_CACHE;
    carousel.items_changed(paths.size(),true);
}

PixelinaBookmarkDialog::PixelinaBookmarkDialog(PixUICanvas &context, const std::vector<std::string> &paths, const std::function<PixUIThumbCache(const std::string &)> &load) : PixUIDialog(context,UI_BOOKMARK_SELECT_TITLE,UI_BOOKMARK_SELECT_TEXT),bookmarks(paths)
{
    resizable = true;
    minimum_width = UI_BOOKMARK_MIN_WIDTH;
    minimum_height = UI_BOOKMARK_MIN_HEIGHT;
    carousel = add<PixUICarousel>();
    bookmark_carousel_setup(*carousel,bookmarks,load);
    carousel->selected = [this](size_t index) {
        selection = int(index);
        layout();
        return true;
    };
}

void PixelinaBookmarkDialog::layout()
{
    PixUIDialog::layout();
    PixUIDialogLayout positions = controls();
    int top = bounds.y + UI_TITLE_HEIGHT + UI_MARGIN;
    carousel->bounds = {bounds.x + UI_MARGIN,top,bounds.w - UI_MARGIN * 2,std::max(0,positions.ok.y - UI_MARGIN - top)};
    accept_button->enabled = selection >= 0 && size_t(selection) < bookmarks.size();
    carousel->layout();
}

void PixelinaBookmarkDialog::accept()
{
    if (selection < 0 || size_t(selection) >= bookmarks.size()) return;
    if (!selected || selected(bookmarks[size_t(selection)])) dismiss();
}
