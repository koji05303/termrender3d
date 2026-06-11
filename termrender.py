"""
Unified terminal rendering CLI.

Usage:
    python termrender.py image path/to/image.jpg --mode ascii
    python termrender.py image path/to/image.jpg --mode braille
    python termrender.py video path/to/video.mp4 --mode braille
    python termrender.py camera 0 --mode braille
"""

import argparse
import os
import sys
import time

MODE_CHOICES = ("ascii", "braille")
DITHER_CHOICES = ("bayer4", "bayer8", "threshold", "floyd-steinberg")
HELP_EPILOG = """
commands:
  image PATH          render a still image
  video PATH          stream a video file
  camera INDEX        stream a webcam

render options:
  --mode {ascii,braille}
                     rendering backend, default: braille
  --width WIDTH      maximum terminal output width
  --workers WORKERS  ASCII worker process count
  --threshold VALUE  Braille threshold from 0 to 255, default: 127
  --dither MODE      Braille dithering mode: bayer4, bayer8, threshold, floyd-steinberg
  --color            enable Braille 24-bit TrueColor output
  --mirror           mirror the frame horizontally

stream options:
  --fps FPS          target FPS for video/camera, default: 30

examples:
  python termrender.py image xxx.jpg --mode ascii
  python termrender.py image xxx.jpg --mode braille --color
  python termrender.py video xxx.mp4 --mode braille --fps 24 --color
  python termrender.py camera 0 --mode braille --mirror
"""


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("數值必須大於 0")
    return parsed


def byte_value(value):
    parsed = int(value)
    if not 0 <= parsed <= 255:
        raise argparse.ArgumentTypeError("數值必須介於 0 到 255")
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render images, videos, or camera frames in the terminal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=HELP_EPILOG,
    )
    subparsers = parser.add_subparsers(dest="source", required=True)

    image_parser = subparsers.add_parser("image", help="render a still image")
    image_parser.add_argument("path", help="image path")
    add_render_options(image_parser)

    video_parser = subparsers.add_parser("video", help="stream a video file")
    video_parser.add_argument("path", help="video path")
    add_render_options(video_parser)
    add_stream_options(video_parser)

    camera_parser = subparsers.add_parser("camera", help="stream a webcam")
    camera_parser.add_argument("index", type=int, help="camera index")
    add_render_options(camera_parser)
    add_stream_options(camera_parser)

    return parser.parse_args()


def add_render_options(parser):
    parser.add_argument(
        "--mode",
        choices=MODE_CHOICES,
        default="braille",
        help="rendering backend",
    )
    parser.add_argument(
        "--width",
        type=positive_int,
        default=None,
        help="maximum terminal output width",
    )
    parser.add_argument(
        "--workers",
        type=positive_int,
        default=None,
        help="ASCII worker process count",
    )
    parser.add_argument(
        "--threshold",
        type=byte_value,
        default=127,
        help="Braille threshold from 0 to 255",
    )
    parser.add_argument(
        "--dither",
        choices=DITHER_CHOICES,
        default="threshold",
        help="Braille dithering mode",
    )
    parser.add_argument(
        "--color",
        action="store_true",
        help="enable Braille 24-bit TrueColor output",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="mirror the frame horizontally",
    )


def add_stream_options(parser):
    parser.add_argument(
        "--fps",
        type=positive_int,
        default=30,
        help="target stream FPS",
    )


def resolve_path(path):
    return os.path.abspath(os.path.expanduser(path))


def get_ascii_backend():
    try:
        import bayer_pattern
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"缺少相依套件 {exc.name!r}，ASCII 模式需要 Pillow") from exc

    return bayer_pattern


def get_braille_backend():
    try:
        import braille_art
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"缺少相依套件 {exc.name!r}，Braille、video、camera 模式需要 OpenCV/Numpy/Pillow"
        ) from exc

    braille_art.configure_opencv_logging()
    return braille_art


def write_stdout_text(text):
    sys.stdout.write(text)
    sys.stdout.flush()


def text_frame_size(text):
    lines = text.splitlines()
    if not lines:
        return 0, 0
    return max(len(line) for line in lines), len(lines)


def render_frame(frame, args, braille_backend=None, ascii_backend=None):
    if args.mode == "ascii":
        if args.mirror:
            frame = frame[:, ::-1]
        if ascii_backend is None:
            ascii_backend = get_ascii_backend()
        text = ascii_backend.render_ascii_frame(
            frame,
            max_width=args.width,
            workers=args.workers,
        )
        return text, text_frame_size(text)

    text, output_columns, output_rows = braille_backend.frame_to_braille(
        frame,
        max_width=args.width,
        threshold=args.threshold,
        dither=args.dither,
        color=args.color,
        mirror=args.mirror,
    )
    return text, (output_columns, output_rows)


def print_terminal_frame(text, frame_size, previous_size):
    if previous_size != frame_size:
        prefix = "\033[H\033[J"
    else:
        prefix = "\033[H"
    write_stdout_text(f"{prefix}{text}\n")


def render_image(args):
    path = resolve_path(args.path)
    if not os.path.isfile(path):
        raise RuntimeError(f"找不到圖片檔: {path}")

    if args.mode == "ascii":
        ascii_backend = get_ascii_backend()
        from PIL import Image, ImageOps

        with Image.open(path) as image:
            frame = ImageOps.mirror(image) if args.mirror else image
            text = ascii_backend.render_ascii_frame(
                frame,
                max_width=args.width,
                workers=args.workers,
            )
        frame_size = text_frame_size(text)
    else:
        braille = get_braille_backend()
        frame = braille.load_image_frame(path)
        text, frame_size = render_frame(frame, args, braille_backend=braille)

    print_terminal_frame(text, frame_size, previous_size=None)


def stream_capture(
    capture,
    fps,
    args,
    empty_stream_message,
    braille_backend=None,
    ascii_backend=None,
):
    frame_interval = 1.0 / fps
    frames_rendered = 0
    previous_size = None

    try:
        write_stdout_text("\033[?25l")

        while True:
            frame_start = time.monotonic()
            success, frame = capture.read()
            if not success:
                if frames_rendered == 0:
                    raise RuntimeError(empty_stream_message)
                break

            text, frame_size = render_frame(
                frame,
                args,
                braille_backend=braille_backend,
                ascii_backend=ascii_backend,
            )
            print_terminal_frame(text, frame_size, previous_size)
            previous_size = frame_size
            frames_rendered += 1

            remaining = frame_interval - (time.monotonic() - frame_start)
            if remaining > 0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        write_stdout_text("\n")
    finally:
        capture.release()
        write_stdout_text("\033[?25h")


def stream_video(args):
    path = resolve_path(args.path)
    if not os.path.isfile(path):
        raise RuntimeError(f"找不到影片檔: {path}")

    braille = get_braille_backend()
    ascii_backend = get_ascii_backend() if args.mode == "ascii" else None
    capture = braille.open_video(path)
    fps = braille.resolve_playback_fps(capture, args.fps)
    stream_capture(
        capture,
        fps=fps,
        args=args,
        empty_stream_message="影片讀取失敗或內容為空",
        braille_backend=braille,
        ascii_backend=ascii_backend,
    )


def stream_camera(args):
    braille = get_braille_backend()
    ascii_backend = get_ascii_backend() if args.mode == "ascii" else None
    capture = braille.open_camera(args.index)
    stream_capture(
        capture,
        fps=args.fps,
        args=args,
        empty_stream_message="webcam 畫面讀取失敗",
        braille_backend=braille,
        ascii_backend=ascii_backend,
    )


def main():
    args = parse_args()

    try:
        if args.source == "image":
            render_image(args)
        elif args.source == "video":
            stream_video(args)
        else:
            stream_camera(args)
    except RuntimeError as exc:
        print(f"錯誤: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
