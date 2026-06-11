"""
將圖片轉成終端 ASCII Art 的工具，支援多核心平行處理和終端尺寸自動調整。

Usage:
    python bayer_pattern.py path/to/image.jpg [--width MAX_WIDTH] [--workers NUM_WORKERS] [--watch]

Options:
    --width MAX_WIDTH       ASCII 最大輸出寬度，預設會依照終端大小自動限制
    --workers NUM_WORKERS   平行處理核心數，預設使用可用 CPU 數量
    --watch                 監聽終端視窗尺寸變化並自動重繪

原理說明:
1. 圖像讀取與預處理：使用 Pillow 讀取圖片並轉換成 RGB 格式的 byte array。
2. 核心運算單元：每個 CPU 核心負責處理圖像的一個橫切面 (Strip)，直接從共享記憶體讀取 RGB 資料，計算亮度並映射到 ASCII 字元。
3. 主排程器：負責計算輸出格子大小、分配任務給各核心、收集結果並輸出到終端。當啟用監聽模式時，會在終端尺寸變化時自動重繪 ASCII Art。

優化點:
- 使用共享記憶體避免大型 RGB 資料被多次拷貝，減少記憶體使用和提升效能。
- 動態計算輸出格子大小以適應不同終端尺寸，確保 ASCII Art 的比例和清晰度。
- 支援多核心平行處理，提升大圖像的處理速度。   

Author: Li-Wei Jiang
Date: 2026-05-05
"""

import argparse
import math
import os
import multiprocessing as mp
import signal
import shutil
import threading
from multiprocessing import shared_memory
import sys

from PIL import Image, ImageOps

CHARSET = " .:-=+*#%@"
CHAR_HEIGHT_RATIO = 2.0
TERMINAL_ROW_PADDING = 1


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("數值必須大於 0")
    return parsed


def load_image_rgb_bytes(image_path):
    with Image.open(image_path) as img:
        rgb_image = ImageOps.exif_transpose(img).convert("RGB")
        width, height = rgb_image.size
        return rgb_image.tobytes(), width, height


def get_terminal_size():
    return shutil.get_terminal_size(fallback=(120, 40))


def calculate_output_grid(image_width, image_height, terminal_size, max_width=None):
    max_columns = terminal_size.columns
    if max_width is not None:
        max_columns = min(max_columns, max_width)
    max_columns = max(1, max_columns)

    max_rows = max(1, terminal_size.lines - TERMINAL_ROW_PADDING)
    rows_for_max_columns = max(
        1,
        math.ceil(max_columns * image_height / (image_width * CHAR_HEIGHT_RATIO)),
    )
    if rows_for_max_columns <= max_rows:
        return max_columns, rows_for_max_columns

    fitted_columns = max(
        1,
        math.floor(max_rows * image_width * CHAR_HEIGHT_RATIO / image_height),
    )
    return min(max_columns, fitted_columns), max_rows


def rgb_bytes_to_ascii(rgb_bytes, width, height, output_columns, output_rows):
    scale_w = max(1, math.ceil(width / output_columns))
    scale_h = max(1, math.ceil(height / output_rows))
    num_chars = len(CHARSET)
    rgb_view = memoryview(rgb_bytes)
    output_lines = []

    for y in range(0, height, scale_h):
        row_chars = []
        row_offset = y * width * 3
        for x in range(0, width, scale_w):
            idx = row_offset + (x * 3)
            r = rgb_view[idx]
            g = rgb_view[idx + 1]
            b = rgb_view[idx + 2]

            luminance = (0.299 * r) + (0.587 * g) + (0.114 * b)
            char_idx = int((luminance / 255) * (num_chars - 1))
            row_chars.append(CHARSET[char_idx])
        output_lines.append("".join(row_chars))

    return "\n".join(output_lines)


def frame_to_rgb_bytes(frame):
    if isinstance(frame, Image.Image):
        rgb_image = ImageOps.exif_transpose(frame).convert("RGB")
        width, height = rgb_image.size
        return rgb_image.tobytes(), width, height

    if hasattr(frame, "ndim") and hasattr(frame, "shape"):
        if frame.ndim == 2:
            rgb_image = Image.fromarray(frame).convert("RGB")
            width, height = rgb_image.size
            return rgb_image.tobytes(), width, height

        if frame.ndim == 3 and frame.shape[2] in (3, 4):
            height, width = frame.shape[:2]
            rgb_frame = frame[:, :, 2::-1]
            return rgb_frame.tobytes(), width, height

    raise TypeError("frame must be a PIL image or an OpenCV-style uint8 frame")


def parse_args():
    parser = argparse.ArgumentParser(description="將圖片轉成終端 ASCII Art")
    parser.add_argument("image", help="要轉成 ASCII Art 的圖片路徑")
    parser.add_argument(
        "--width",
        type=positive_int,
        default=None,
        help="ASCII 最大輸出寬度，預設會依照終端大小自動限制",
    )
    parser.add_argument(
        "--workers",
        type=positive_int,
        default=None,
        help="平行處理核心數，預設使用可用 CPU 數量",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="監聽終端視窗尺寸變化並自動重繪",
    )
    return parser.parse_args()

# ==========================================
# 2. 核心運算單元 (每個 CPU 核心執行的任務)
# ==========================================
def process_strip(args):
    """
    處理圖像的其中一個橫切面 (Strip)。
    直接透過 shared memory 讀取，避免大型 RGB 資料被重複拷貝。
    """
    shm_name, width, start_y, end_y, scale_w, scale_h = args
    num_chars = len(CHARSET)
    output_lines = []

    shm = shared_memory.SharedMemory(name=shm_name)
    try:
        mv = shm.buf

        for y in range(start_y, end_y, scale_h):
            line = ""
            for x in range(0, width, scale_w):
                # RGB 格式，每個像素佔 3 bytes
                idx = (y * width + x) * 3

                try:
                    r = mv[idx]
                    g = mv[idx + 1]
                    b = mv[idx + 2]

                    # 亮度公式 (色彩學標準)
                    luminance = (0.299 * r) + (0.587 * g) + (0.114 * b)

                    # 映射到 ASCII 字元
                    char_idx = int((luminance / 255) * (num_chars - 1))
                    line += CHARSET[char_idx]
                except IndexError:
                    break  # 超出共享記憶體範圍
            output_lines.append(line)

    finally:
        shm.close()

    return "\n".join(output_lines)

# ==========================================
# 3. 主排程器 (交通指揮官)
# ==========================================
def render_rgb_to_ascii_text(rgb_bytes, width, height, output_columns, output_rows, num_workers=None):
    if num_workers is None:
        num_workers = mp.cpu_count()
    num_workers = max(1, int(num_workers))

    scale_w = max(1, math.ceil(width / output_columns))
    scale_h = max(1, math.ceil(height / output_rows))

    if num_workers == 1:
        return rgb_bytes_to_ascii(
            rgb_bytes,
            width,
            height,
            output_columns=output_columns,
            output_rows=output_rows,
        )

    max_workers_by_rows = max(1, math.ceil(height / scale_h))
    num_workers = min(num_workers, max_workers_by_rows)

    # 計算每個核心要處理的 Y 軸範圍
    # 確保切割邊界對齊 scale_h，才不會導致畫面撕裂
    strip_height = (height // num_workers)
    strip_height = strip_height - (strip_height % scale_h)
    if strip_height == 0:
        strip_height = scale_h

    strip_ranges = []
    for i in range(num_workers):
        start_y = i * strip_height
        end_y = (i + 1) * strip_height if i != (num_workers - 1) else height
        if start_y >= height:
            break
        strip_ranges.append((start_y, end_y))

    shm = shared_memory.SharedMemory(create=True, size=len(rgb_bytes))
    try:
        shm.buf[: len(rgb_bytes)] = rgb_bytes
        tasks = [
            (shm.name, width, start_y, end_y, scale_w, scale_h)
            for start_y, end_y in strip_ranges
        ]

        # 啟動多進程池
        with mp.Pool(processes=num_workers) as pool:
            results = pool.map(process_strip, tasks)

        # 將所有核心的結果按順序拼接
        final_output = "\n".join(results)
        return final_output
    finally:
        shm.close()
        shm.unlink()


def render_rgb_to_ascii(rgb_bytes, width, height, output_columns, output_rows, num_workers=None):
    final_output = render_rgb_to_ascii_text(
        rgb_bytes,
        width,
        height,
        output_columns=output_columns,
        output_rows=output_rows,
        num_workers=num_workers,
    )
    print("\033[H\033[J", end="")
    print(final_output)
    return final_output


def render_ascii_frame(frame, max_width=None, workers=None) -> str:
    rgb_bytes, width, height = frame_to_rgb_bytes(frame)
    terminal_size = get_terminal_size()
    output_columns, output_rows = calculate_output_grid(
        width,
        height,
        terminal_size,
        max_width=max_width,
    )
    return render_rgb_to_ascii_text(
        rgb_bytes,
        width,
        height,
        output_columns=output_columns,
        output_rows=output_rows,
        num_workers=workers,
    )


def render_image_to_terminal(rgb_bytes, width, height, max_width=None, num_workers=None):
    terminal_size = get_terminal_size()
    output_columns, output_rows = calculate_output_grid(
        width,
        height,
        terminal_size,
        max_width=max_width,
    )
    render_rgb_to_ascii(
        rgb_bytes,
        width,
        height,
        output_columns=output_columns,
        output_rows=output_rows,
        num_workers=num_workers,
    )


def watch_terminal_resize(rgb_bytes, width, height, max_width=None, num_workers=None):
    redraw_event = threading.Event()
    redraw_event.set()

    def handle_resize(signum, frame):
        del signum, frame
        redraw_event.set()

    previous_handler = signal.getsignal(signal.SIGWINCH)
    signal.signal(signal.SIGWINCH, handle_resize)
    try:
        while True:
            redraw_event.wait()
            redraw_event.clear()
            render_image_to_terminal(
                rgb_bytes,
                width,
                height,
                max_width=max_width,
                num_workers=num_workers,
            )
    except KeyboardInterrupt:
        print()
    finally:
        signal.signal(signal.SIGWINCH, previous_handler)


def main():
    args = parse_args()
    image_path = os.path.abspath(os.path.expanduser(args.image))

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"找不到圖片檔: {image_path}")

    rgb_bytes, width, height = load_image_rgb_bytes(image_path)

    if args.watch:
        watch_terminal_resize(
            rgb_bytes,
            width,
            height,
            max_width=args.width,
            num_workers=args.workers,
        )
        return

    render_image_to_terminal(
        rgb_bytes,
        width,
        height,
        max_width=args.width,
        num_workers=args.workers,
    )

# ==========================================
# 執行區
# ==========================================
if __name__ == "__main__":
    main()
