"""
Unified terminal rendering CLI.

Usage:
    python termrender.py render image path/to/image.jpg --mode ascii
    python termrender.py render video path/to/video.mp4 --mode braille
    python termrender.py render camera 0 --mode braille
    python termrender.py render fluid --size 160 --color
    python termrender.py image path/to/image.jpg --mode ascii
    python termrender.py image path/to/image.jpg --mode braille
    python termrender.py video path/to/video.mp4 --mode braille
    python termrender.py camera 0 --mode braille
"""

import argparse
from contextlib import contextmanager
import os
import sys
import time

MODE_CHOICES = ("ascii", "braille")
FLUID_MODE_CHOICES = ("ascii", "braille", "none")
DITHER_CHOICES = ("bayer4", "bayer8", "threshold", "floyd-steinberg")
FLUID_OBSTACLE_CHOICES = ("none", "circle", "box")
HELP_EPILOG = """
commands:
  render image PATH   render a still image
  render video PATH   stream a video file
  render camera INDEX stream a webcam
  render fluid [PATH] run the fixed-grid fluid simulator

legacy commands:
  image PATH          same as: render image PATH
  video PATH          same as: render video PATH
  camera INDEX        same as: render camera INDEX

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
  python termrender.py render fluid --size 160 --color
  python termrender.py render fluid photo.jpg --mode ascii --workers 8
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


def non_negative_int(value):
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("數值必須大於等於 0")
    return parsed


def positive_float(value):
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("數值必須大於 0")
    return parsed


def unit_float(value):
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("數值必須介於 0 到 1")
    return parsed


def byte_value(value):
    parsed = int(value)
    if not 0 <= parsed <= 255:
        raise argparse.ArgumentTypeError("數值必須介於 0 到 255")
    return parsed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Render images, videos, or camera frames in the terminal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=HELP_EPILOG,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    render_parser = subparsers.add_parser("render", help="render media or fluid simulation")
    render_parser.set_defaults(command="render")
    render_subparsers = render_parser.add_subparsers(dest="source", required=True)
    add_render_source_parsers(render_subparsers)

    add_render_source_parsers(subparsers, legacy=True)

    return parser.parse_args(argv)


def add_render_source_parsers(subparsers, legacy=False):
    image_parser = subparsers.add_parser("image", help="render a still image")
    image_parser.add_argument("path", help="image path")
    add_render_options(image_parser)
    image_parser.set_defaults(command="render", source="image")

    video_parser = subparsers.add_parser("video", help="stream a video file")
    video_parser.add_argument("path", help="video path")
    add_render_options(video_parser)
    add_stream_options(video_parser)
    video_parser.set_defaults(command="render", source="video")

    camera_parser = subparsers.add_parser("camera", help="stream a webcam")
    camera_parser.add_argument("index", type=int, help="camera index")
    add_render_options(camera_parser)
    add_stream_options(camera_parser)
    camera_parser.set_defaults(command="render", source="camera")

    if not legacy:
        fluid_parser = subparsers.add_parser("fluid", help="run the fluid simulator")
        fluid_parser.add_argument(
            "image",
            nargs="?",
            help="input image path; omit it to use a generated RGB grid",
        )
        add_fluid_options(fluid_parser)
        fluid_parser.set_defaults(command="render", source="fluid")


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


def add_fluid_options(parser):
    parser.add_argument(
        "--size",
        type=positive_int,
        default=160,
        help="square simulation grid size",
    )
    parser.add_argument(
        "--steps",
        type=non_negative_int,
        default=90,
        help="simulation time steps",
    )
    parser.add_argument(
        "--dt",
        type=positive_float,
        default=0.35,
        help="fluid time step size",
    )
    parser.add_argument(
        "--pressure-iters",
        type=positive_int,
        default=45,
        help="Jacobi pressure iterations per step",
    )
    parser.add_argument(
        "--force-strength",
        type=float,
        default=16.0,
        help="procedural vortex force strength",
    )
    parser.add_argument(
        "--initial-swirl",
        type=float,
        default=2.0,
        help="initial velocity swirl strength",
    )
    parser.add_argument(
        "--velocity-damping",
        type=unit_float,
        default=0.995,
        help="velocity damping after external forces",
    )
    parser.add_argument(
        "--dye-decay",
        type=unit_float,
        default=0.999,
        help="RGB dye multiplier after advection",
    )
    parser.add_argument(
        "--obstacle",
        choices=FLUID_OBSTACLE_CHOICES,
        default="none",
        help="static solid mask shape",
    )
    parser.add_argument(
        "--obstacle-size",
        type=unit_float,
        default=0.16,
        help="solid obstacle radius/half-width as a grid fraction",
    )
    parser.add_argument(
        "--mode",
        "--render",
        dest="render",
        choices=FLUID_MODE_CHOICES,
        default="braille",
        help="fluid terminal renderer",
    )
    parser.add_argument(
        "--render-every",
        type=positive_int,
        default=1,
        help="render/save every N steps",
    )
    parser.add_argument(
        "--fps",
        type=positive_int,
        default=24,
        help="terminal animation FPS cap",
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
        default=96,
        help="Braille threshold from 0 to 255",
    )
    parser.add_argument(
        "--dither",
        choices=DITHER_CHOICES,
        default="bayer4",
        help="Braille dithering mode",
    )
    parser.add_argument(
        "--color",
        action="store_true",
        help="enable Braille 24-bit TrueColor output",
    )
    parser.add_argument(
        "--save-final",
        help="write the final dye image to this path",
    )
    parser.add_argument(
        "--save-frames",
        help="write rendered dye frames as PNGs into this directory",
    )
    parser.add_argument(
        "--no-interactive",
        dest="interactive",
        action="store_false",
        default=True,
        help="disable non-blocking WASD/arrow-key force injection",
    )
    parser.add_argument(
        "--keyboard-force",
        type=float,
        default=28.0,
        help="interactive keyboard force strength",
    )
    parser.add_argument(
        "--keyboard-decay",
        type=unit_float,
        default=0.78,
        help="interactive keyboard force decay per emitted frame",
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


@contextmanager
def ascii_worker_pool(args, ascii_backend):
    pool = None
    if ascii_backend is not None and args.mode == "ascii" and args.workers != 1:
        pool = ascii_backend.create_ascii_pool(args.workers)
    try:
        yield pool
    except BaseException:
        if pool is not None:
            pool.terminate()
        raise
    else:
        if pool is not None:
            pool.close()
    finally:
        if pool is not None:
            pool.join()


def write_stdout_text(text):
    sys.stdout.write(text)
    sys.stdout.flush()


def text_frame_size(text):
    lines = text.splitlines()
    if not lines:
        return 0, 0
    return max(len(line) for line in lines), len(lines)


def render_frame(frame, args, braille_backend=None, ascii_backend=None, ascii_pool=None):
    if args.mode == "ascii":
        if args.mirror:
            frame = frame[:, ::-1]
        if ascii_backend is None:
            ascii_backend = get_ascii_backend()
        text = ascii_backend.render_ascii_frame(
            frame,
            max_width=args.width,
            workers=args.workers,
            pool=ascii_pool,
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
    ascii_pool=None,
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
                ascii_pool=ascii_pool,
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
    with ascii_worker_pool(args, ascii_backend) as ascii_pool:
        stream_capture(
            capture,
            fps=fps,
            args=args,
            empty_stream_message="影片讀取失敗或內容為空",
            braille_backend=braille,
            ascii_backend=ascii_backend,
            ascii_pool=ascii_pool,
        )


def stream_camera(args):
    braille = get_braille_backend()
    ascii_backend = get_ascii_backend() if args.mode == "ascii" else None
    capture = braille.open_camera(args.index)
    with ascii_worker_pool(args, ascii_backend) as ascii_pool:
        stream_capture(
            capture,
            fps=args.fps,
            args=args,
            empty_stream_message="webcam 畫面讀取失敗",
            braille_backend=braille,
            ascii_backend=ascii_backend,
            ascii_pool=ascii_pool,
        )


def render_fluid(args):
    import fluid

    saved_path = fluid.run_simulation(args)
    if saved_path:
        print(f"saved final dye: {saved_path}")


def main(argv=None):
    args = parse_args(argv)

    try:
        if args.source == "image":
            render_image(args)
        elif args.source == "video":
            stream_video(args)
        elif args.source == "camera":
            stream_camera(args)
        elif args.source == "fluid":
            render_fluid(args)
    except RuntimeError as exc:
        print(f"錯誤: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
