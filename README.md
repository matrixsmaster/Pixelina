# Pixelina

Pixelina is a self-contained image generation, editing and viewer application with a distinctive Windows 3.11-styled retro UI. It runs Z-Image Turbo locally with a Qwen3 text encoder and a VAE decoder, offering a fast, offline creative workflow with as little external dependency as possible.

Pixelina brings modern image generation capabilities to a nostalgic interface, proving that you don't need flashy UI frameworks to create beautiful art.

## Use cases

- **Local text-to-image prototyping**: type a prompt, adjust seed, steps and shift, and generate arbitrarily sized images entirely offline. Reproducibility is guaranteed by seed and settings snapshot.
- **Live Preview**: Watch generation progress in real-time as diffusion steps complete.
- **Batch exploration**: keep a working directory with `pixelina.txt` database and generated JPG/PNG files. The carousel provides instant visual history; click to reload any previous generation with its exact prompt and parameters.
- **Model swapping**: point the diffuser, encoder and VAE fields to alternative files in the same format. Pixelina validates settings before generation and persists the last used models.
- **CPU-first deployment**: runs on commodity hardware with only a CPU backend. Optional Vulkan backend accelerates matmul and attention on supported GPUs.
- **Minimal-dependency distribution**: the binary bundles its own UI widgets, image I/O via stb, and JSON parsing. No Python, no pip packages, no large runtimes.

## Supported models

Pixelina targets the Z-Image Turbo pipeline:

- **Diffuser**: `z-image-turbo-Q8_0.gguf` by default. GGUF quantized diffuser weights in the Z-Image Turbo family.
- **Text encoder**: `Qwen3-4B-Q8_0.gguf` by default. Qwen3 text encoder in GGUF format with `general.architecture = qwen3`.
- **VAE**: `ae.safetensors` by default. Safetensors VAE decoder matching Z-Image Turbo.

The application validates model metadata and tensor names at load time. Only the above three model roles are required; Pixelina does not support arbitrary diffusion families.

Tested with both Q5_K and Q8_0 quants.

## UI and experience

Pixelina is built around a faithful Windows 3.11 desktop metaphor with system menu, title bar, min/max/close and resizable borders. Controls use 8×8 bitmap font rendering, beveled buttons, faceplate colors and the characteristic 3D bevels of Windows 3.11.

The UI library `pixuilib` is written from scratch with no external GUI toolkit. Rendering is done via SDL2 as a thin canvas adapter, keeping the widget set minimal and portable.

## Build requirements

- C++17 compiler
- SDL2
- GNU Make
- Optional Vulkan: `glslc` for shader compilation

Yes, that's it! Minimal dependencies are intentional. No external ML framework needed.

The application is platform-independent and compiles cleanly on any modern Unix-like system with SDL2 support. Vulkan shaders require Vulkan 1.0+ with optional 1.1/1.2 features for subgroup and cooperative matmul.

Build:
```
make
```

With optional Vulkan:
```
make VULKAN=1
```

Run:
```
pixelina [-p] [-t threads] [path]
```

`-p` forces PNG output for Save, `-t threads` sets max CPU thread count for CPU-only inference, `path` selects working directory.

## Minimal dependency philosophy

Pixelina is designed to have as little dependency on other libs as possible. Core inference, kernels, tokenization and UI are all implemented in-house. Only SDL2 is required for windowing and input.

## License

MIT. See LICENSE.

**Pixelina** - (C) Dmitry 'sciloaf' Solovyev, 2026
