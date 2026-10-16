/* This file is a part of 'Pixelina' Image Generator, Editor, and Viewer.
 *
 * (C) Dmitry 'sciloaf' Solovyev, 2026
 *
 * License: MIT
 */
#ifndef PIXELINA_DLG_BOOKIMG_H
#define PIXELINA_DLG_BOOKIMG_H

#include "pixuilib.h"

#define UI_BOOKMARKS_TITLE "Bookmarks"
#define UI_BOOKMARK_SELECT_TITLE "Select bookmarked image"
#define UI_BOOKMARK_SELECT_TEXT "Select"
#define UI_BOOKMARK_MIN_WIDTH 320
#define UI_BOOKMARK_MIN_HEIGHT 200

void bookmark_carousel_setup(PixUICarousel &carousel, const std::vector<std::string> &paths, const std::function<PixUIThumbCache(const std::string &)> &load);

class PixelinaBookmarkDialog : public PixUIDialog
{
public:
    PixelinaBookmarkDialog(PixUICanvas &context, const std::vector<std::string> &paths, const std::function<PixUIThumbCache(const std::string &)> &load);

    void layout() override;
    void accept() override;

    std::function<bool(const std::string &)> selected; // caller accepts a path before the selector closes

private:
    std::vector<std::string> bookmarks; // owned path snapshot for this selection
    PixUICarousel* carousel; // owned bounded previews matching the main bookmarks carousel
    int selection = -1; // selected path, or no selection
};

#endif
