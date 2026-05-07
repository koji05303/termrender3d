"""
將圖片、影片或 webcam 畫面轉成 Unicode Braille Art。

Braille 字元會把一個終端字元框拆成 2x4 的點陣，因此在同樣的終端大小下，
可以比傳統 ASCII 字元提供更高的細節密度。

Usage:
	python braille_art.py --image IMAGE_PATH [--width MAX_WIDTH] [--threshold VALUE] [--dither MODE] [--color] [--mirror]
	python braille_art.py --file VIDEO_PATH [--width MAX_WIDTH] [--fps FPS] [--threshold VALUE] [--dither MODE] [--color] [--mirror]
	python braille_art.py --camera CAMERA_INDEX [--width MAX_WIDTH] [--fps FPS] [--threshold VALUE] [--dither MODE] [--color] [--mirror]

Options:
	--image IMAGE_PATH    圖片檔案路徑
	--file VIDEO_PATH     影片檔案路徑
	--camera INDEX        webcam 裝置索引
	--width MAX_WIDTH     Braille 最大輸出寬度，預設依照終端大小自動限制
	--fps FPS             串流或影片目標更新頻率，預設為 30
	--threshold VALUE     二值化門檻，範圍 0 到 255，Bayer 模式下作為亮度偏移，預設為 127
	--dither MODE         二值化模式，可選 bayer4、bayer8、threshold 或 floyd-steinberg，預設 threshold
	--color               啟用 24-bit TrueColor，使用每個 2x4 Braille cell 中亮點的平均色當前景色
	--mirror              水平鏡像畫面，符合常見預覽習慣
"""

import argparse
import math
import os
import select
import sys
import time

import cv2
import numpy as np

from bayer_pattern import get_terminal_size, positive_int

BRAILLE_UNICODE_BASE = 0x2800
BRAILLE_CELL_WIDTH = 2
BRAILLE_CELL_HEIGHT = 4
TERMINAL_ROW_PADDING = 1
CV2_LOG_LEVEL_SILENT = 0
DITHER_CHOICES = ("bayer4", "bayer8", "threshold", "floyd-steinberg")
ANSI_RESET = "\033[0m"
BRAILLE_DOT_MASKS = np.array(
	[
		[0x01, 0x08],
		[0x02, 0x10],
		[0x04, 0x20],
		[0x40, 0x80],
	],
	dtype=np.uint16,
)
BAYER_MATRIX_4X4 = np.array(
	[
		[0, 8, 2, 10],
		[12, 4, 14, 6],
		[3, 11, 1, 9],
		[15, 7, 13, 5],
	],
	dtype=np.float32,
)
BAYER_MATRIX_8X8 = np.array(
	[
		[0, 48, 12, 60, 3, 51, 15, 63],
		[32, 16, 44, 28, 35, 19, 47, 31],
		[8, 56, 4, 52, 11, 59, 7, 55],
		[40, 24, 36, 20, 43, 27, 39, 23],
		[2, 50, 14, 62, 1, 49, 13, 61],
		[34, 18, 46, 30, 33, 17, 45, 29],
		[10, 58, 6, 54, 9, 57, 5, 53],
		[42, 26, 38, 22, 41, 25, 37, 21],
	],
	dtype=np.float32,
)


def byte_value(value):
	parsed = int(value)
	if not 0 <= parsed <= 255:
		raise argparse.ArgumentTypeError("數值必須介於 0 到 255")
	return parsed


def parse_args():
	parser = argparse.ArgumentParser(description="將畫面轉成終端 Braille Art")
	source_group = parser.add_mutually_exclusive_group(required=True)
	source_group.add_argument("--image", help="要轉成 Braille Art 的圖片檔案路徑")
	source_group.add_argument("--file", help="要轉成 Braille Art 的影片檔案路徑")
	source_group.add_argument(
		"--camera",
		type=int,
		help="webcam 裝置索引",
	)
	parser.add_argument(
		"--width",
		type=positive_int,
		default=None,
		help="Braille 最大輸出寬度，預設依照終端大小自動限制",
	)
	parser.add_argument(
		"--fps",
		type=positive_int,
		default=30,
		help="串流或影片目標更新頻率，預設為 30",
	)
	parser.add_argument(
		"--threshold",
		type=byte_value,
		default=127,
		help="二值化門檻，範圍 0 到 255；Bayer 模式下會作為亮度偏移，預設為 127",
	)
	parser.add_argument(
		"--dither",
		choices=DITHER_CHOICES,
		default="threshold",
		help="二值化模式，預設為 threshold",
	)
	parser.add_argument(
		"--color",
		action="store_true",
		help="啟用 24-bit TrueColor，使用每個 2x4 Braille cell 中亮點的平均色",
	)
	parser.add_argument(
		"--mirror",
		action="store_true",
		help="水平鏡像畫面，符合常見預覽習慣",
	)
	return parser.parse_args()


def configure_opencv_logging():
	if hasattr(cv2, "setLogLevel"):
		cv2.setLogLevel(CV2_LOG_LEVEL_SILENT)


def calculate_braille_grid(image_width, image_height, terminal_size, max_width=None):
	max_columns = terminal_size.columns
	if max_width is not None:
		max_columns = min(max_columns, max_width)
	max_columns = max(1, max_columns)

	max_rows = max(1, terminal_size.lines - TERMINAL_ROW_PADDING)
	rows_for_max_columns = max(
		1,
		math.ceil(max_columns * image_height / (image_width * 2)),
	)
	if rows_for_max_columns <= max_rows:
		return max_columns, rows_for_max_columns

	fitted_columns = max(
		1,
		math.floor(max_rows * image_width * 2 / image_height),
	)
	return min(max_columns, fitted_columns), max_rows


def resize_to_braille_grid(gray_frame, output_columns, output_rows):
	target_width = max(1, output_columns * BRAILLE_CELL_WIDTH)
	target_height = max(1, output_rows * BRAILLE_CELL_HEIGHT)
	return cv2.resize(gray_frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def floyd_steinberg_dither(gray_frame, threshold):
	working = gray_frame.astype(np.float32)
	height, width = working.shape
	binary = np.zeros((height, width), dtype=bool)

	for y in range(height):
		for x in range(width):
			old_value = working[y, x]
			new_value = 255.0 if old_value >= threshold else 0.0
			binary[y, x] = new_value > 0
			quant_error = old_value - new_value

			if x + 1 < width:
				working[y, x + 1] += quant_error * (7 / 16)
			if y + 1 < height:
				if x > 0:
					working[y + 1, x - 1] += quant_error * (3 / 16)
				working[y + 1, x] += quant_error * (5 / 16)
				if x + 1 < width:
					working[y + 1, x + 1] += quant_error * (1 / 16)

	return binary


def ordered_dither(gray_frame, threshold, matrix):
	height, width = gray_frame.shape
	row_pattern = np.arange(height) % matrix.shape[0]
	col_pattern = np.arange(width) % matrix.shape[1]
	bayer_pattern = matrix[row_pattern[:, None], col_pattern[None, :]]
	bias = float(threshold) - 127.0
	threshold_map = ((bayer_pattern + 0.5) * (255.0 / matrix.size)) + bias
	threshold_map = np.clip(threshold_map, 0.0, 255.0)
	return gray_frame.astype(np.float32) >= threshold_map


def binarize_gray_frame(gray_frame, threshold, dither):
	if dither == "bayer4":
		return ordered_dither(gray_frame, threshold, BAYER_MATRIX_4X4)
	if dither == "bayer8":
		return ordered_dither(gray_frame, threshold, BAYER_MATRIX_8X8)
	if dither == "floyd-steinberg":
		return floyd_steinberg_dither(gray_frame, threshold)
	return gray_frame >= threshold


def pack_braille_codes(binary_frame):
	output_rows = binary_frame.shape[0] // BRAILLE_CELL_HEIGHT
	output_columns = binary_frame.shape[1] // BRAILLE_CELL_WIDTH
	tiled = binary_frame.reshape(
		output_rows,
		BRAILLE_CELL_HEIGHT,
		output_columns,
		BRAILLE_CELL_WIDTH,
	).transpose(0, 2, 1, 3)
	return BRAILLE_UNICODE_BASE + (
		tiled.astype(np.uint16) * BRAILLE_DOT_MASKS[None, None, :, :]
	).sum(axis=(2, 3))


def braille_codes_to_string(braille_codes):
	return "\n".join("".join(chr(code) for code in row) for row in braille_codes)


def binary_to_braille(binary_frame):
	return braille_codes_to_string(pack_braille_codes(binary_frame))


def compute_braille_cell_colors(resized_rgb_frame, binary_frame):
	output_rows = binary_frame.shape[0] // BRAILLE_CELL_HEIGHT
	output_columns = binary_frame.shape[1] // BRAILLE_CELL_WIDTH
	tiled_rgb = resized_rgb_frame.reshape(
		output_rows,
		BRAILLE_CELL_HEIGHT,
		output_columns,
		BRAILLE_CELL_WIDTH,
		3,
	).transpose(0, 2, 1, 3, 4).astype(np.float32)
	tiled_binary = binary_frame.reshape(
		output_rows,
		BRAILLE_CELL_HEIGHT,
		output_columns,
		BRAILLE_CELL_WIDTH,
	).transpose(0, 2, 1, 3)
	active_mask = tiled_binary[..., None].astype(np.float32)
	active_counts = active_mask.sum(axis=(2, 3))
	color_sums = (tiled_rgb * active_mask).sum(axis=(2, 3))
	mean_colors = tiled_rgb.mean(axis=(2, 3))
	safe_counts = np.maximum(active_counts, 1.0)
	active_mean_colors = color_sums / safe_counts
	cell_colors = np.where(active_counts > 0, active_mean_colors, mean_colors)
	return cell_colors.astype(np.uint8)


def colorize_braille_codes(braille_codes, cell_colors):
	lines = []
	for code_row, color_row in zip(braille_codes, cell_colors):
		parts = []
		previous_color = None
		for code, color in zip(code_row, color_row):
			if code == BRAILLE_UNICODE_BASE:
				if previous_color is not None:
					parts.append(ANSI_RESET)
					previous_color = None
				parts.append(chr(code))
				continue

			current_color = (int(color[0]), int(color[1]), int(color[2]))
			if current_color != previous_color:
				parts.append(f"\033[38;2;{current_color[0]};{current_color[1]};{current_color[2]}m")
				previous_color = current_color
			parts.append(chr(code))

		if previous_color is not None:
			parts.append(ANSI_RESET)
		lines.append("".join(parts))

	return "\n".join(lines)


def frame_to_braille(frame, max_width=None, threshold=127, dither="bayer4", color=False, mirror=False):
	if mirror:
		frame = cv2.flip(frame, 1)

	if frame.ndim == 2:
		gray_frame = frame
		rgb_frame = np.repeat(gray_frame[:, :, None], 3, axis=2) if color else None
		frame_height, frame_width = gray_frame.shape
	else:
		gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if color else None
		frame_height, frame_width = gray_frame.shape[:2]

	terminal_size = get_terminal_size()
	output_columns, output_rows = calculate_braille_grid(
		frame_width,
		frame_height,
		terminal_size,
		max_width=max_width,
	)
	resized_gray = resize_to_braille_grid(gray_frame, output_columns, output_rows)
	binary_frame = binarize_gray_frame(resized_gray, threshold=threshold, dither=dither)
	braille_codes = pack_braille_codes(binary_frame)
	if not color:
		return braille_codes_to_string(braille_codes), output_columns, output_rows

	resized_rgb = resize_to_braille_grid(rgb_frame, output_columns, output_rows)
	cell_colors = compute_braille_cell_colors(resized_rgb, binary_frame)
	return colorize_braille_codes(braille_codes, cell_colors), output_columns, output_rows


def write_stdout_text(text):
	if not text:
		return

	stream = sys.stdout
	try:
		fd = stream.fileno()
	except (AttributeError, OSError, ValueError):
		stream.write(text)
		stream.flush()
		return

	encoding = stream.encoding or "utf-8"
	errors = stream.errors or "replace"
	encoded = text.encode(encoding, errors=errors)
	view = memoryview(encoded)
	total_written = 0
	while total_written < len(encoded):
		try:
			written = os.write(fd, view[total_written:])
		except BlockingIOError:
			select.select((), (fd,), ())
			continue
		except InterruptedError:
			continue
		if written <= 0:
			raise BrokenPipeError("stdout write returned no bytes")
		total_written += written


def print_braille_frame(braille_text, frame_size, previous_size):
	if previous_size != frame_size:
		prefix = "\033[H\033[J"
	else:
		prefix = "\033[H"
	write_stdout_text(f"{prefix}{braille_text}\n")


def load_image_frame(image_path):
	if not os.path.isfile(image_path):
		raise RuntimeError(f"找不到圖片檔: {image_path}")

	frame = cv2.imread(image_path, cv2.IMREAD_COLOR)
	if frame is None:
		raise RuntimeError(f"無法讀取圖片檔: {image_path}")
	return frame


def open_video(file_path):
	if not os.path.isfile(file_path):
		raise RuntimeError(f"找不到影片檔: {file_path}")

	capture = cv2.VideoCapture(file_path)
	if not capture.isOpened():
		capture.release()
		raise RuntimeError(f"無法開啟影片檔: {file_path}")
	return capture


def open_camera(camera_index):
	capture = cv2.VideoCapture(camera_index)
	capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
	if not capture.isOpened():
		capture.release()
		raise RuntimeError(f"無法開啟 webcam 裝置: {camera_index}")
	return capture


def resolve_playback_fps(capture, target_fps):
	source_fps = capture.get(cv2.CAP_PROP_FPS)
	if source_fps and source_fps > 0:
		return min(source_fps, target_fps)
	return target_fps


def render_image_to_terminal(image_path, max_width=None, threshold=127, dither="bayer4", color=False, mirror=False):
	frame = load_image_frame(image_path)
	braille_text, output_columns, output_rows = frame_to_braille(
		frame,
		max_width=max_width,
		threshold=threshold,
		dither=dither,
		color=color,
		mirror=mirror,
	)
	print_braille_frame(braille_text, (output_columns, output_rows), previous_size=None)


def stream_capture_to_braille(
	capture,
	fps,
	max_width=None,
	threshold=127,
	dither="bayer4",
	color=False,
	mirror=False,
	empty_stream_message="串流讀取失敗",
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

			braille_text, output_columns, output_rows = frame_to_braille(
				frame,
				max_width=max_width,
				threshold=threshold,
				dither=dither,
				color=color,
				mirror=mirror,
			)
			frame_size = (output_columns, output_rows)
			print_braille_frame(braille_text, frame_size, previous_size)
			previous_size = frame_size
			frames_rendered += 1

			elapsed = time.monotonic() - frame_start
			remaining = frame_interval - elapsed
			if remaining > 0:
				time.sleep(remaining)
	except KeyboardInterrupt:
		write_stdout_text("\n")
	finally:
		capture.release()
		write_stdout_text("\033[?25h")


def stream_video_to_braille(file_path, max_width=None, fps=30, threshold=127, dither="bayer4", color=False, mirror=False):
	capture = open_video(file_path)
	effective_fps = resolve_playback_fps(capture, fps)
	stream_capture_to_braille(
		capture,
		fps=effective_fps,
		max_width=max_width,
		threshold=threshold,
		dither=dither,
		color=color,
		mirror=mirror,
		empty_stream_message="影片讀取失敗或內容為空",
	)


def stream_camera_to_braille(camera_index=0, max_width=None, fps=30, threshold=127, dither="bayer4", color=False, mirror=False):
	capture = open_camera(camera_index)
	stream_capture_to_braille(
		capture,
		fps=fps,
		max_width=max_width,
		threshold=threshold,
		dither=dither,
		color=color,
		mirror=mirror,
		empty_stream_message="webcam 畫面讀取失敗",
	)


def main():
	args = parse_args()
	configure_opencv_logging()

	try:
		if args.image:
			image_path = os.path.abspath(os.path.expanduser(args.image))
			render_image_to_terminal(
				image_path,
				max_width=args.width,
				threshold=args.threshold,
				dither=args.dither,
				color=args.color,
				mirror=args.mirror,
			)
		elif args.file:
			file_path = os.path.abspath(os.path.expanduser(args.file))
			stream_video_to_braille(
				file_path,
				max_width=args.width,
				fps=args.fps,
				threshold=args.threshold,
				dither=args.dither,
				color=args.color,
				mirror=args.mirror,
			)
		else:
			stream_camera_to_braille(
				camera_index=args.camera,
				max_width=args.width,
				fps=args.fps,
				threshold=args.threshold,
				dither=args.dither,
				color=args.color,
				mirror=args.mirror,
			)
	except RuntimeError as exc:
		print(f"錯誤: {exc}", file=sys.stderr)
		return 1
	return 0


if __name__ == "__main__":
	raise SystemExit(main())