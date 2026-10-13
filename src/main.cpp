/* This file is a part of 'Pixelina' Image Generator, Editor, and Viewer.
 *
 * (C) Dmitry 'sciloaf' Solovyev, 2026
 *
 * License: MIT
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <signal.h>
#include <string>
#include "ui.h"
#include "tensor.h"

int main(int argc, char** argv)
{
    std::string directory = ".";
    bool png = false;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i],"-p")) png = true;
        else if (!strcmp(argv[i],"-t") && i + 1 < argc) cpu_set_threads(atoi(argv[++i]));
        else if (argv[i][0] != '-' && i == argc - 1) directory = argv[i];
        else {
            fprintf(stderr,"Usage: pixelina [-p] [-t threads] [path]\nUnknown or incomplete argument: %s\n",argv[i]);
            return 1;
        }
    }

    Pixelina app;
    std::string error;
    if (!app.initialize(directory,png,error)) {
        fprintf(stderr,PIXELINA_NAME ": %s\n",error.c_str());
        return 1;
    }

    // override SDL's signal overrides
    signal(SIGINT,SIG_DFL);
    signal(SIGTERM,SIG_DFL);

    return app.run();
}
