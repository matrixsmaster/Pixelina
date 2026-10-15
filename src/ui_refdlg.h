/* This file is a part of 'Pixelina' Image Generator, Editor, and Viewer.
 *
 * (C) Dmitry 'sciloaf' Solovyev, 2026
 *
 * License: MIT
 */
#ifndef PIXELINA_UI_REFDLG_H
#define PIXELINA_UI_REFDLG_H

#include "pixuilib.h"

#define UI_MULTI_REFERENCE_TITLE "Multi-Reference Selection"
#define UI_MULTI_REFERENCE_MIN_WIDTH 480
#define UI_MULTI_REFERENCE_MIN_HEIGHT 300
#define UI_REFERENCE_BUTTON_WIDTH 120
#define UI_REFERENCE_CAROUSEL_FRACTION 0.5f
#define UI_REFERENCE_HINT "Use <image1>, <image2>, etc. in the prompt."
#define UI_REFERENCE_OPEN_ERROR "Cannot open reference image: "

class PixelinaReferenceDialog : public PixUIDialog
{
public:
    PixelinaReferenceDialog(PixUICanvas &context, const std::vector<std::string> &paths, const std::string &text);

    void layout() override;
    bool event(const PixUIEvent &input) override;
    void accept() override;
    bool add_image(const std::string &path);

    std::function<PixUIThumbCache(const std::string &)> load; // application thumbnail loader without retained full images
    std::function<void(const std::vector<std::string> &,const std::string &)> selected; // commit the draft paths and prompt together

private:
    friend struct PixelinaTest;

    std::vector<std::string> references; // draft absolute reference paths in numbered order
    int selection = -1; // selected draft reference for deletion
    PixUICarousel* carousel; // owned numbered reference previews
    PixUIMemo* prompt; // owned multiline draft prompt
    PixUILabel* hint; // prompt numbering instructions or file errors
    PixUIButton* add_button; // opens a nested image selector
    PixUIButton* delete_button; // removes the selected draft reference

    void delete_image();
};

#endif
