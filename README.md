# TermRender3D

A high-performance 3D software rasterizer written in Python that renders standard OBJ models directly to the terminal using 24-bit TrueColor Braille characters. 

The engine implements a dual-backend architecture, utilizing Tile-Based Rendering (TBR) for multi-core CPUs and a Visibility Buffer approach for CUDA-enabled GPUs, pushing the limits of terminal-based graphics.

<p align="center">
  <img src="output/agera.gif" width="1000" alt="TermRender GPU Backend Demo">
  <img src="output/skull.gif" width="1000" alt="TermRender GPU Backend Demo">
</p>

## Architecture & Features

This project does not rely on OpenGL, DirectX, or Vulkan. The entire graphics pipeline (MVP transformations, clipping, barycentric interpolation, and rasterization) is implemented from scratch.

### 1. GPU Backend (Taichi CUDA)
- **Visibility Buffer**: Packs 32-bit physical depth and 32-bit triangle indices into a single `uint64`. Solves Z-buffer race conditions natively using atomic operations (`ti.atomic_min`).
- **Deferred Shading**: Lighting and barycentric coordinates are only evaluated for the strictly visible pixels, achieving zero overdraw.
- **JIT Compilation**: Powered by `taichi`, compiling Python functions directly into optimized CUDA machine code at runtime.

### 2. CPU Backend (Software TBR)
- **Tile-Based Rendering**: Divides the screen into independent tiles for lock-free parallel rasterization across multiple CPU cores.
- **Zero-Copy IPC**: Utilizes `multiprocessing.shared_memory` to pass geometric indices between processes, eliminating serialization overhead.
- **Precision**: Uses `float64` for edge function evaluations to prevent clipping tearing and coordinate overflow at extreme zoom levels.

### 3. Terminal Output Layer
- **High-Density Braille**: Maps $2 \times 4$ sub-pixel grids to Unicode Braille characters (`U+2800` to `U+28FF`).
- **ANSI TrueColor**: Calculates the average RGB value of active sub-pixels for precise 24-bit color output.
- **Tear-Free Rendering**: Uses low-level `os.write` and ANSI cursor positioning (`\033[H`) to bypass standard output buffering issues.

## Project Structure

The engine is encapsulated in three core files:

* `mvp_re_engine.py`: The main entry point. Handles OBJ parsing, MVP matrix transformations, near-plane Sutherland-Hodgman clipping, and CPU multiprocessing scheduling.
* `cuda_engine.py`: The Taichi-powered GPU backend. Handles VRAM memory allocation and massively parallel rasterization kernels.
* `braille_art.py`: The display driver. Handles the conversion of 2D pixel arrays into Unicode Braille patterns and ANSI color sequences.
* `bayer_pattern.py`: A utility for converting images to ASCII art using Bayer dithering.

## Installation

Ensure you have Python 3.8+ installed. 

## OBJ File For Test

You can find compatible OBJ models on [Free3D](https://free3d.com/zh/3d-models/obj).

### Installation

```bash
# Basic dependencies
pip install numpy

# Required for GPU backend (Highly Recommended)
pip install taichi
```

### Usage Example

1. 3D Model Rendering
```bash
# GPU Accelerated Mode (Recommended)
python mvp_re_engine.py models/agera.obj --backend gpu --color --interactive

# CPU Multi-core Mode
python mvp_re_engine.py models/agera.obj --backend cpu --workers 44 --color
```

2. Video & Webcam Streaming
```bash
# Video to Braille
python braille_art.py --file video.mp4 --fps 24 --color

# Live Webcam to Braille
python braille_art.py --camera 0 --fps 30 --color --mirror
```

3. Static Image Conversion
```bash
# Image to Braille
python braille_art.py --image photo.jpg --dither bayer8 --color

# Image to ASCII (with auto-resize)
python bayer_pattern.py photo.jpg --watch
```

### Technical Specifications

| Feature | Specification |
|---------|---------------|
| Matrix Operations | Model, View (Look-At), Perspective Projection |
| Clipping Algorithm | Sutherland-Hodgman (Near-Plane) |
| Lighting Models | Flat, Lambert, Phong |
| Interpolation | Perspective-Correct (1/W) for attributes, Linear for Depth |
| Input Formats | OBJ, JPG, PNG, MP4, Webcam (V4L2) |
| Output Protocol | ANSI X3.64 / ECMA-48 (24-bit TrueColor) |


Author: Li-Wei Jiang

License: MIT