"""
預設純 CPU 的終端機 3D 引擎，可選擇把 MVP 與光柵化卸載到 GPU 後端。

技術重點:
1. 手寫模型、觀察、投影矩陣，走完整的 MVP pipeline。
2. 以三角形為單位進行光柵化，並使用 Z-buffer 解決遮擋。
3. 支援 Lambert 與簡化 Phong 光照，最後轉成 Unicode Braille。

Usage:
    python mvp_re_engine.py koenigsegg-agera/agera.obj --color
    python mvp_re_engine.py koenigsegg-agera/agera.obj --frames 1 --rotate 15 35 0
	python mvp_re_engine.py koenigsegg-agera/agera.obj --backend gpu --color
"""

import argparse
import atexit
import colorsys
import math
import multiprocessing as mp
import os
import select
import sys
import time
import zlib
from dataclasses import dataclass
from multiprocessing import shared_memory

import numpy as np

from cuda_engine import build_taichi_renderer

try:
	import fcntl
	import termios
	import tty
except ImportError:
	fcntl = None
	termios = None
	tty = None

from bayer_pattern import get_terminal_size, positive_int
from braille_art import (
	BRAILLE_CELL_HEIGHT,
	BRAILLE_CELL_WIDTH,
	DITHER_CHOICES,
	braille_codes_to_string,
	byte_value,
	binarize_gray_frame,
	colorize_braille_codes,
	compute_braille_cell_colors,
	pack_braille_codes,
	print_braille_frame,
	write_stdout_text,
)

EPSILON = 1e-6
MIN_CLIP_W = 1e-5
DEFAULT_LIGHT_DIR = np.array((0.35, 0.8, 1.0), dtype=np.float32)
DEFAULT_CAMERA_POS = np.array((0.0, 0.2, 4.2), dtype=np.float32)
DEFAULT_LOOK_AT = np.array((0.0, 0.0, 0.0), dtype=np.float32)
DEFAULT_BASE_COLOR = np.array((214.0, 220.0, 228.0), dtype=np.float32)
DEFAULT_TILE_SIZE = 32
DEFAULT_GPU_DEPTH_BIAS = 1e-5
FRAME_CONFIG_DTYPE = np.dtype(np.int32)
FRAME_CONFIG_SHAPE = (3,)
FRAME_CONFIG_WIDTH_INDEX = 0
FRAME_CONFIG_HEIGHT_INDEX = 1
FRAME_CONFIG_TILE_SIZE_INDEX = 2
PROJECTED_TRIANGLE_FIELD_SPECS = (
	("screen_triangles", np.float32, (3, 2)),
	("depth_triangles", np.float32, (3,)),
	("inverse_w_triangles", np.float32, (3,)),
	("intensity_triangles", np.float32, (3,)),
	("triangle_colors", np.float32, (3,)),
	("bounds", np.int32, (4,)),
)
WORKER_PROJECTED_BUFFER = None
WORKER_FRAME_CONFIG_MEMORY = None
WORKER_FRAME_CONFIG = None
INTERACTIVE_CONTROLS_TEXT = (
	"互動控制: WASD/方向鍵 旋轉, Q/E 滾轉, IJKL/UO 平移, Z/X 縮放, "
	"P 暫停自轉, R 重置, M 切換著色, B 切換背面剔除, Esc 離開\n"
	"近距離觀察建議優先用 IJKL/UO 平移或 --camera-pos，而不是只靠 Z/X 放大 scale。"
)


@dataclass(frozen=True)
class Mesh:
	vertices: np.ndarray
	vertices_h: np.ndarray
	triangles: np.ndarray
	normals: np.ndarray | None
	normal_indices: np.ndarray | None
	triangle_colors: np.ndarray


@dataclass(frozen=True)
class RenderTarget:
	pixel_width: int
	pixel_height: int
	braille_columns: int
	braille_rows: int


@dataclass(frozen=True)
class SharedProjectedTriangleBuffer:
	capacity: int
	shm: shared_memory.SharedMemory
	screen_triangles: np.ndarray
	depth_triangles: np.ndarray
	inverse_w_triangles: np.ndarray
	intensity_triangles: np.ndarray
	triangle_colors: np.ndarray
	bounds: np.ndarray

	@classmethod
	def create(cls, capacity):
		shm = shared_memory.SharedMemory(
			create=True,
			size=get_projected_triangle_buffer_nbytes(capacity),
		)
		return cls.attach_from_shared_memory(shm, capacity)

	@classmethod
	def attach(cls, name, capacity):
		shm = shared_memory.SharedMemory(name=name)
		return cls.attach_from_shared_memory(shm, capacity)

	@classmethod
	def attach_from_shared_memory(cls, shm, capacity):
		views = build_projected_triangle_views(shm.buf, capacity)
		return cls(capacity=capacity, shm=shm, **views)

	def close(self):
		self.shm.close()

	def unlink(self):
		self.shm.unlink()


@dataclass
class SceneState:
	scale: float
	translation: np.ndarray
	rotation: np.ndarray
	auto_rotation: bool
	initial_scale: float
	initial_translation: np.ndarray
	initial_rotation: np.ndarray

	@classmethod
	def from_args(cls, args):
		return cls(
			scale=float(args.scale),
			translation=args.translate.astype(np.float32).copy(),
			rotation=args.rotate.astype(np.float32).copy(),
			auto_rotation=bool(float(np.linalg.norm(args.rotation_speed)) > EPSILON),
			initial_scale=float(args.scale),
			initial_translation=args.translate.astype(np.float32).copy(),
			initial_rotation=args.rotate.astype(np.float32).copy(),
		)

	def reset(self):
		self.scale = self.initial_scale
		self.translation = self.initial_translation.copy()
		self.rotation = self.initial_rotation.copy()


def get_projected_triangle_buffer_nbytes(capacity):
	total_bytes = 0
	for _, dtype, component_shape in PROJECTED_TRIANGLE_FIELD_SPECS:
		total_bytes += int(np.dtype(dtype).itemsize * capacity * math.prod(component_shape))
	return total_bytes


def build_projected_triangle_views(buffer, capacity):
	views = {}
	offset = 0
	for name, dtype, component_shape in PROJECTED_TRIANGLE_FIELD_SPECS:
		dtype_obj = np.dtype(dtype)
		shape = (capacity, *component_shape)
		nbytes = int(dtype_obj.itemsize * np.prod(shape))
		views[name] = np.ndarray(shape=shape, dtype=dtype_obj, buffer=buffer, offset=offset)
		offset += nbytes
	return views


def create_shared_frame_config():
	shm = shared_memory.SharedMemory(
		create=True,
		size=int(FRAME_CONFIG_DTYPE.itemsize * math.prod(FRAME_CONFIG_SHAPE)),
	)
	frame_config = np.ndarray(FRAME_CONFIG_SHAPE, dtype=FRAME_CONFIG_DTYPE, buffer=shm.buf)
	frame_config.fill(0)
	return shm, frame_config


def close_worker_shared_state():
	global WORKER_PROJECTED_BUFFER, WORKER_FRAME_CONFIG_MEMORY, WORKER_FRAME_CONFIG

	if WORKER_PROJECTED_BUFFER is not None:
		WORKER_PROJECTED_BUFFER.close()
		WORKER_PROJECTED_BUFFER = None
	if WORKER_FRAME_CONFIG_MEMORY is not None:
		WORKER_FRAME_CONFIG_MEMORY.close()
		WORKER_FRAME_CONFIG_MEMORY = None
	WORKER_FRAME_CONFIG = None


def initialize_raster_worker(projected_buffer_name, projected_capacity, frame_config_name):
	global WORKER_PROJECTED_BUFFER, WORKER_FRAME_CONFIG_MEMORY, WORKER_FRAME_CONFIG

	close_worker_shared_state()
	WORKER_PROJECTED_BUFFER = SharedProjectedTriangleBuffer.attach(
		projected_buffer_name,
		projected_capacity,
	)
	WORKER_FRAME_CONFIG_MEMORY = shared_memory.SharedMemory(name=frame_config_name)
	WORKER_FRAME_CONFIG = np.ndarray(
		FRAME_CONFIG_SHAPE,
		dtype=FRAME_CONFIG_DTYPE,
		buffer=WORKER_FRAME_CONFIG_MEMORY.buf,
	)
	atexit.register(close_worker_shared_state)


def decode_input_bytes(data):
	decoded_keys = []
	index = 0
	while index < len(data):
		current = data[index]
		if current == 27:
			if index + 2 < len(data) and data[index + 1] == 91:
				arrow_map = {65: "UP", 66: "DOWN", 67: "RIGHT", 68: "LEFT"}
				decoded_keys.append(arrow_map.get(data[index + 2], "ESC"))
				index += 3
				continue
			decoded_keys.append("ESC")
			index += 1
			continue
		if current == 3:
			decoded_keys.append("CTRL_C")
			index += 1
			continue
		try:
			decoded_keys.append(bytes((current,)).decode("utf-8"))
		except UnicodeDecodeError:
			pass
		index += 1
	return decoded_keys


class RawTerminalInput:
	def __init__(self, enabled):
		self.requested = enabled
		self.enabled = False
		self.fd = None
		self.original_attributes = None
		self.original_flags = None

	def __enter__(self):
		if not self.requested:
			return self
		if fcntl is None or termios is None or tty is None or not sys.stdin.isatty():
			print("警告: 互動模式需要 POSIX TTY，已退回非互動模式。", file=sys.stderr)
			return self

		self.fd = sys.stdin.fileno()
		self.original_attributes = termios.tcgetattr(self.fd)
		self.original_flags = fcntl.fcntl(self.fd, fcntl.F_GETFL)
		tty.setcbreak(self.fd)
		fcntl.fcntl(self.fd, fcntl.F_SETFL, self.original_flags | os.O_NONBLOCK)
		self.enabled = True
		return self

	def __exit__(self, exc_type, exc_value, traceback):
		if not self.enabled:
			return False
		termios.tcsetattr(self.fd, termios.TCSADRAIN, self.original_attributes)
		fcntl.fcntl(self.fd, fcntl.F_SETFL, self.original_flags)
		return False

	def poll_keys(self):
		if not self.enabled:
			return []

		keys = []
		while True:
			ready, _, _ = select.select((self.fd,), (), (), 0.0)
			if not ready:
				break
			try:
				chunk = os.read(self.fd, 64)
			except BlockingIOError:
				break
			if not chunk:
				break
			keys.extend(decode_input_bytes(chunk))
		return keys


def positive_float(value):
	parsed = float(value)
	if parsed <= 0:
		raise argparse.ArgumentTypeError("數值必須大於 0")
	return parsed


def non_negative_float(value):
	parsed = float(value)
	if parsed < 0:
		raise argparse.ArgumentTypeError("數值不可小於 0")
	return parsed


def parse_args():
	parser = argparse.ArgumentParser(description="在終端機中用 Braille Art 跑 3D 引擎，預設 CPU，可選 Taichi GPU backend")
	parser.add_argument("model", help="OBJ 模型路徑")
	parser.add_argument(
		"--width",
		type=positive_int,
		default=None,
		help="Braille 最大輸出寬度，預設跟隨終端機寬度",
	)
	parser.add_argument(
		"--fps",
		type=positive_int,
		default=18,
		help="目標更新頻率，預設 18 FPS",
	)
	parser.add_argument(
		"--frames",
		type=positive_int,
		default=None,
		help="要渲染的總幀數，預設持續播放直到 Ctrl+C",
	)
	parser.add_argument(
		"--scale",
		type=positive_float,
		default=2.0,
		help="模型整體縮放，預設 2.0；近距離觀察建議改用平移或 camera-pos，避免極端 scale 放大",
	)
	parser.add_argument(
		"--translate",
		type=float,
		nargs=3,
		metavar=("X", "Y", "Z"),
		default=(0.0, 0.0, 0.0),
		help="模型平移量，預設 0 0 0",
	)
	parser.add_argument(
		"--rotate",
		type=float,
		nargs=3,
		metavar=("X", "Y", "Z"),
		default=(0.0, 0.0, 0.0),
		help="模型初始旋轉角度（度），預設 0 0 0",
	)
	parser.add_argument(
		"--rotation-speed",
		type=float,
		nargs=3,
		metavar=("X", "Y", "Z"),
		default=(0.0, 18.0, 0.0),
		help="每秒旋轉角速度（度），預設 0 18 0",
	)
	parser.add_argument(
		"--camera-pos",
		type=float,
		nargs=3,
		metavar=("X", "Y", "Z"),
		default=tuple(DEFAULT_CAMERA_POS.tolist()),
		help="攝影機位置，預設 0 0.2 4.2",
	)
	parser.add_argument(
		"--look-at",
		type=float,
		nargs=3,
		metavar=("X", "Y", "Z"),
		default=tuple(DEFAULT_LOOK_AT.tolist()),
		help="攝影機注視點，預設 0 0 0",
	)
	parser.add_argument(
		"--fov",
		type=positive_float,
		default=52.0,
		help="透視投影視角（度），預設 52",
	)
	parser.add_argument(
		"--near",
		type=positive_float,
		default=0.1,
		help="近裁剪面距離，預設 0.1；GPU 近拍若有破圖或閃爍，建議提高到 1.0 以上",
	)
	parser.add_argument(
		"--far",
		type=positive_float,
		default=24.0,
		help="遠裁剪面距離，預設 24.0",
	)
	parser.add_argument(
		"--light-dir",
		type=float,
		nargs=3,
		metavar=("X", "Y", "Z"),
		default=tuple(DEFAULT_LIGHT_DIR.tolist()),
		help="由表面指向光源的方向向量，預設 0.35 0.8 1.0",
	)
	parser.add_argument(
		"--ambient",
		type=float,
		default=0.12,
		help="環境光強度，預設 0.12",
	)
	parser.add_argument(
		"--diffuse",
		type=float,
		default=0.88,
		help="漫反射權重，預設 0.88",
	)
	parser.add_argument(
		"--specular",
		type=float,
		default=0.25,
		help="Phong 鏡面高光權重，預設 0.25",
	)
	parser.add_argument(
		"--shininess",
		type=positive_float,
		default=24.0,
		help="Phong 高光指數，預設 24",
	)
	parser.add_argument(
		"--shading",
		choices=("lambert", "phong"),
		default="phong",
		help="著色模式，預設 phong",
	)
	parser.add_argument(
		"--triangle-stride",
		type=positive_int,
		default=1,
		help="每 N 個三角形取 1 個進行光柵化，可用來換取 FPS",
	)
	parser.add_argument(
		"--supersample",
		type=positive_int,
		default=1,
		help="內部光柵化倍率，預設 1；可設 2 提升邊緣品質",
	)
	parser.add_argument(
		"--tile-size",
		type=positive_int,
		default=DEFAULT_TILE_SIZE,
		help="Tile-based rasterizer 的 tile 邊長（像素），預設 32",
	)
	parser.add_argument(
		"--workers",
		type=positive_int,
		default=max(1, os.cpu_count() or 1),
		help="平行 tile rasterizer 的 worker 數量，預設使用全部 CPU 核心",
	)
	parser.add_argument(
		"--backend",
		choices=("cpu", "gpu"),
		default="cpu",
		help="渲染 backend，預設 cpu；gpu 會使用 Taichi CUDA",
	)
	parser.add_argument(
		"--gpu-depth-bias",
		type=non_negative_float,
		default=None,
		help="GPU visibility buffer 的深度偏移；預設為自動（gpu: 1e-5, cpu: 0）",
	)
	parser.add_argument(
		"--interactive",
		action="store_true",
		help="啟用即時鍵盤控制，直接在終端機調整旋轉、平移與縮放",
	)
	parser.add_argument(
		"--rotate-step",
		type=positive_float,
		default=6.0,
		help="互動模式每次旋轉角度，預設 6 度",
	)
	parser.add_argument(
		"--translate-step",
		type=positive_float,
		default=0.08,
		help="互動模式每次平移距離，預設 0.08",
	)
	parser.add_argument(
		"--scale-step",
		type=positive_float,
		default=0.08,
		help="互動模式每次縮放比例，預設 0.08",
	)
	parser.add_argument(
		"--threshold",
		type=byte_value,
		default=127,
		help="Braille 二值化門檻，預設 127",
	)
	parser.add_argument(
		"--dither",
		choices=DITHER_CHOICES,
		default="bayer8",
		help="Braille 抖動模式，預設 bayer8",
	)
	parser.add_argument(
		"--color",
		action="store_true",
		help="輸出 24-bit 彩色 Braille",
	)
	parser.add_argument(
		"--base-color",
		type=byte_value,
		nargs=3,
		metavar=("R", "G", "B"),
		default=tuple(int(channel) for channel in DEFAULT_BASE_COLOR),
		help="物體基礎顏色，預設 214 220 228",
	)
	parser.add_argument(
		"--no-backface-culling",
		action="store_true",
		help="停用背面剔除",
	)
	return parser.parse_args()


def normalize_vector(vector):
	length = float(np.linalg.norm(vector))
	if length <= EPSILON:
		return np.zeros_like(vector, dtype=np.float32)
	return (vector / length).astype(np.float32)


def normalize_rows(vectors):
	lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
	safe_lengths = np.where(lengths > EPSILON, lengths, 1.0)
	return (vectors / safe_lengths).astype(np.float32)


def resolve_obj_index(raw_index, collection_length):
	index = int(raw_index)
	if index > 0:
		return index - 1
	return collection_length + index


def normalize_mesh_vertices(vertices):
	minimum = vertices.min(axis=0)
	maximum = vertices.max(axis=0)
	center = (minimum + maximum) * 0.5
	extent = float(np.max(maximum - minimum))
	if extent <= EPSILON:
		raise RuntimeError("OBJ 模型包圍盒無效，無法正規化")
	return ((vertices - center) / (extent * 0.5)).astype(np.float32)


def parse_material_color(payload):
	if len(payload) < 3:
		return None
	color = np.clip(np.asarray(payload[:3], dtype=np.float32), 0.0, 1.0) * 255.0
	return color.astype(np.float32)


def load_material_library_colors(model_path, library_names):
	material_colors = {}
	model_dir = os.path.dirname(model_path)

	for library_name in library_names:
		library_path = os.path.join(model_dir, library_name)
		if not os.path.isfile(library_path):
			continue

		current_material = None
		with open(library_path, "r", encoding="utf-8", errors="ignore") as handle:
			for line in handle:
				stripped = line.strip()
				if not stripped or stripped.startswith("#"):
					continue

				parts = stripped.split()
				prefix = parts[0]
				payload = parts[1:]

				if prefix == "newmtl" and payload:
					current_material = payload[0]
					material_colors.setdefault(current_material, None)
				elif prefix in {"Kd", "Ka"} and current_material is not None:
					color = parse_material_color(payload)
					if color is None:
						continue
					if prefix == "Kd" or material_colors[current_material] is None:
						material_colors[current_material] = color

	return {
		name: color
		for name, color in material_colors.items()
		if color is not None
	}


def build_fallback_color(label):
	if not label:
		return DEFAULT_BASE_COLOR.copy()

	seed = zlib.crc32(label.encode("utf-8")) & 0xFFFFFFFF
	hue = (seed % 360) / 360.0
	saturation = 0.45 + ((((seed >> 8) & 0xFF) / 255.0) * 0.30)
	value = 0.70 + ((((seed >> 16) & 0xFF) / 255.0) * 0.22)
	red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
	return np.array((red * 255.0, green * 255.0, blue * 255.0), dtype=np.float32)


def resolve_triangle_colors(material_names, object_names, material_palette):
	resolved_colors = []
	for material_name, object_name in zip(material_names, object_names):
		if material_name and material_name in material_palette:
			resolved_colors.append(material_palette[material_name])
		elif material_name:
			resolved_colors.append(build_fallback_color(material_name))
		elif object_name:
			resolved_colors.append(build_fallback_color(object_name))
		else:
			resolved_colors.append(DEFAULT_BASE_COLOR.copy())
	return np.asarray(resolved_colors, dtype=np.float32)


def load_obj_mesh(model_path):
	if not os.path.isfile(model_path):
		raise RuntimeError(f"找不到 OBJ 模型檔: {model_path}")

	vertices = []
	normals = []
	triangles = []
	normal_triangles = []
	triangle_material_names = []
	triangle_object_names = []
	material_library_names = []
	current_material = None
	current_object = "default"

	with open(model_path, "r", encoding="utf-8", errors="ignore") as handle:
		for line in handle:
			stripped = line.strip()
			if not stripped or stripped.startswith("#"):
				continue

			parts = stripped.split()
			prefix = parts[0]
			payload = parts[1:]

			if prefix == "mtllib" and payload:
				material_library_names.extend(payload)
			elif prefix in {"o", "g"} and payload:
				current_object = payload[0]
			elif prefix == "usemtl":
				current_material = payload[0] if payload else None
			elif prefix == "v" and len(payload) >= 3:
				vertices.append((float(payload[0]), float(payload[1]), float(payload[2])))
			elif prefix == "vn" and len(payload) >= 3:
				normals.append((float(payload[0]), float(payload[1]), float(payload[2])))
			elif prefix == "f" and len(payload) >= 3:
				face_vertices = []
				face_normals = []
				for corner in payload:
					indices = corner.split("/")
					if not indices[0]:
						continue
					face_vertices.append(resolve_obj_index(indices[0], len(vertices)))
					normal_index = None
					if len(indices) >= 3 and indices[2]:
						normal_index = resolve_obj_index(indices[2], len(normals))
					face_normals.append(normal_index)

				for offset in range(1, len(face_vertices) - 1):
					triangles.append(
						(face_vertices[0], face_vertices[offset], face_vertices[offset + 1])
					)
					triangle_material_names.append(current_material)
					triangle_object_names.append(current_object)
					if all(
						index is not None
						for index in (
							face_normals[0],
							face_normals[offset],
							face_normals[offset + 1],
						)
					):
						normal_triangles.append(
							(
								face_normals[0],
								face_normals[offset],
								face_normals[offset + 1],
							)
						)
					else:
						normal_triangles.append((-1, -1, -1))

	if not vertices:
		raise RuntimeError(f"OBJ 模型沒有任何頂點資料: {model_path}")
	if not triangles:
		raise RuntimeError(f"OBJ 模型沒有任何面資料: {model_path}")

	vertex_array = normalize_mesh_vertices(np.asarray(vertices, dtype=np.float32))
	vertex_h = np.concatenate(
		(vertex_array, np.ones((vertex_array.shape[0], 1), dtype=np.float32)),
		axis=1,
	)
	triangle_array = np.asarray(triangles, dtype=np.int32)
	triangle_colors = resolve_triangle_colors(
		triangle_material_names,
		triangle_object_names,
		load_material_library_colors(model_path, material_library_names),
	)

	normal_array = None
	normal_index_array = None
	if normals:
		normal_array = normalize_rows(np.asarray(normals, dtype=np.float32))
		normal_index_array = np.asarray(normal_triangles, dtype=np.int32)

	return Mesh(
		vertices=vertex_array,
		vertices_h=vertex_h,
		triangles=triangle_array,
		normals=normal_array,
		normal_indices=normal_index_array,
		triangle_colors=triangle_colors,
	)


def make_scale_matrix(scale):
	return np.array(
		[
			[scale, 0.0, 0.0, 0.0],
			[0.0, scale, 0.0, 0.0],
			[0.0, 0.0, scale, 0.0],
			[0.0, 0.0, 0.0, 1.0],
		],
		dtype=np.float32,
	)


def make_translation_matrix(translation):
	tx, ty, tz = translation
	return np.array(
		[
			[1.0, 0.0, 0.0, tx],
			[0.0, 1.0, 0.0, ty],
			[0.0, 0.0, 1.0, tz],
			[0.0, 0.0, 0.0, 1.0],
		],
		dtype=np.float32,
	)


def make_rotation_x(angle_radians):
	cosine = math.cos(angle_radians)
	sine = math.sin(angle_radians)
	return np.array(
		[
			[1.0, 0.0, 0.0, 0.0],
			[0.0, cosine, -sine, 0.0],
			[0.0, sine, cosine, 0.0],
			[0.0, 0.0, 0.0, 1.0],
		],
		dtype=np.float32,
	)


def make_rotation_y(angle_radians):
	cosine = math.cos(angle_radians)
	sine = math.sin(angle_radians)
	return np.array(
		[
			[cosine, 0.0, sine, 0.0],
			[0.0, 1.0, 0.0, 0.0],
			[-sine, 0.0, cosine, 0.0],
			[0.0, 0.0, 0.0, 1.0],
		],
		dtype=np.float32,
	)


def make_rotation_z(angle_radians):
	cosine = math.cos(angle_radians)
	sine = math.sin(angle_radians)
	return np.array(
		[
			[cosine, -sine, 0.0, 0.0],
			[sine, cosine, 0.0, 0.0],
			[0.0, 0.0, 1.0, 0.0],
			[0.0, 0.0, 0.0, 1.0],
		],
		dtype=np.float32,
	)


def make_model_matrix(scale, rotation_degrees, translation):
	rotation_x, rotation_y, rotation_z = [math.radians(value) for value in rotation_degrees]
	rotation_matrix = (
		make_rotation_z(rotation_z)
		@ make_rotation_y(rotation_y)
		@ make_rotation_x(rotation_x)
	)
	return make_translation_matrix(translation) @ rotation_matrix @ make_scale_matrix(scale)


def make_look_at_matrix(eye, target, up=(0.0, 1.0, 0.0)):
	eye_vector = np.asarray(eye, dtype=np.float32)
	target_vector = np.asarray(target, dtype=np.float32)
	up_vector = normalize_vector(np.asarray(up, dtype=np.float32))
	forward = normalize_vector(target_vector - eye_vector)
	right = np.cross(forward, up_vector)
	if float(np.linalg.norm(right)) <= EPSILON:
		up_vector = np.array((0.0, 0.0, 1.0), dtype=np.float32)
		right = np.cross(forward, up_vector)
	right = normalize_vector(right)
	true_up = normalize_vector(np.cross(right, forward))

	return np.array(
		[
			[right[0], right[1], right[2], -float(np.dot(right, eye_vector))],
			[
				true_up[0],
				true_up[1],
				true_up[2],
				-float(np.dot(true_up, eye_vector)),
			],
			[
				-forward[0],
				-forward[1],
				-forward[2],
				float(np.dot(forward, eye_vector)),
			],
			[0.0, 0.0, 0.0, 1.0],
		],
		dtype=np.float32,
	)


def make_perspective_matrix(fov_degrees, aspect_ratio, near_plane, far_plane):
	f = 1.0 / math.tan(math.radians(fov_degrees) * 0.5)
	depth = near_plane - far_plane
	return np.array(
		[
			[f / aspect_ratio, 0.0, 0.0, 0.0],
			[0.0, f, 0.0, 0.0],
			[0.0, 0.0, (far_plane + near_plane) / depth, (2.0 * far_plane * near_plane) / depth],
			[0.0, 0.0, -1.0, 0.0],
		],
		dtype=np.float32,
	)


def transform_homogeneous(points_h, matrix):
	return (points_h @ matrix.T).astype(np.float32)


def edge_function(start, end, x_values, y_values):
	return ((x_values - start[0]) * (end[1] - start[1])) - ((y_values - start[1]) * (end[0] - start[0]))


def build_render_target(max_width=None, supersample=1):
	terminal_size = get_terminal_size()
	braille_columns = terminal_size.columns
	if max_width is not None:
		braille_columns = min(braille_columns, max_width)
	braille_columns = max(1, braille_columns)
	braille_rows = max(1, terminal_size.lines - 1)
	return RenderTarget(
		pixel_width=braille_columns * BRAILLE_CELL_WIDTH * supersample,
		pixel_height=braille_rows * BRAILLE_CELL_HEIGHT * supersample,
		braille_columns=braille_columns,
		braille_rows=braille_rows,
	)


def downsample_rgb_frame(rgb_frame, factor):
	if factor <= 1:
		return rgb_frame

	height, width, _ = rgb_frame.shape
	reduced_height = height // factor
	reduced_width = width // factor
	trimmed = rgb_frame[: reduced_height * factor, : reduced_width * factor]
	reshaped = trimmed.reshape(reduced_height, factor, reduced_width, factor, 3)
	return reshaped.mean(axis=(1, 3)).astype(np.uint8)


def rgb_frame_to_braille(rgb_frame, threshold, dither, color):
	gray_frame = np.clip(
		(rgb_frame[:, :, 0].astype(np.float32) * 0.299)
		+ (rgb_frame[:, :, 1].astype(np.float32) * 0.587)
		+ (rgb_frame[:, :, 2].astype(np.float32) * 0.114),
		0.0,
		255.0,
	).astype(np.uint8)
	binary_frame = binarize_gray_frame(gray_frame, threshold=threshold, dither=dither)
	braille_codes = pack_braille_codes(binary_frame)
	if not color:
		return braille_codes_to_string(braille_codes)

	cell_colors = compute_braille_cell_colors(rgb_frame, binary_frame)
	return colorize_braille_codes(braille_codes, cell_colors)


def compute_lighting(
	normals,
	positions,
	light_dir,
	camera_pos,
	shading,
	ambient,
	diffuse_weight,
	specular_weight,
	shininess,
):
	normalized_normals = normalize_rows(normals.astype(np.float32))
	light_vector = normalize_vector(np.asarray(light_dir, dtype=np.float32))
	diffuse = np.maximum((normalized_normals * light_vector[None, :]).sum(axis=1), 0.0)
	intensity = ambient + (diffuse_weight * diffuse)

	if shading == "phong":
		camera_vector = np.asarray(camera_pos, dtype=np.float32)[None, :]
		view_dirs = normalize_rows(camera_vector - positions.astype(np.float32))
		half_vectors = normalize_rows(view_dirs + light_vector[None, :])
		specular = np.power(
			np.maximum((normalized_normals * half_vectors).sum(axis=1), 0.0),
			shininess,
		)
		intensity += specular_weight * specular * (diffuse > 0.0)

	return np.clip(intensity, 0.0, 1.0).astype(np.float32)


def compute_triangle_intensity(mesh, triangle_index, world_vertices, world_normals, args, light_dir, camera_pos):
	vertex_indices = mesh.triangles[triangle_index]
	triangle_positions = world_vertices[vertex_indices]

	if world_normals is not None and mesh.normal_indices is not None:
		normal_indices = mesh.normal_indices[triangle_index]
		if np.all(normal_indices >= 0):
			triangle_normals = world_normals[normal_indices]
			return compute_lighting(
				triangle_normals,
				triangle_positions,
				light_dir=light_dir,
				camera_pos=camera_pos,
				shading=args.shading,
				ambient=args.ambient,
				diffuse_weight=args.diffuse,
				specular_weight=args.specular,
				shininess=args.shininess,
			)

	face_normal = normalize_vector(
		np.cross(
			triangle_positions[1] - triangle_positions[0],
			triangle_positions[2] - triangle_positions[0],
		)
	)
	triangle_normals = np.repeat(face_normal[None, :], 3, axis=0)
	triangle_intensity = compute_lighting(
		triangle_normals,
		triangle_positions,
		light_dir=light_dir,
		camera_pos=camera_pos,
		shading=args.shading,
		ambient=args.ambient,
		diffuse_weight=args.diffuse,
		specular_weight=args.specular,
		shininess=args.shininess,
	)
	return np.repeat(np.mean(triangle_intensity, dtype=np.float32), 3).astype(np.float32)


def clip_triangle_against_near_plane(view_triangle, intensities, near_plane):
	clipped_positions = []
	clipped_intensities = []
	plane_z = -float(near_plane)

	for index in range(view_triangle.shape[0]):
		previous_index = (index - 1) % view_triangle.shape[0]
		current_position = view_triangle[index]
		previous_position = view_triangle[previous_index]
		current_intensity = float(intensities[index])
		previous_intensity = float(intensities[previous_index])

		current_inside = current_position[2] <= plane_z
		previous_inside = previous_position[2] <= plane_z

		if current_inside != previous_inside:
			denominator = current_position[2] - previous_position[2]
			if abs(float(denominator)) > EPSILON:
				factor = (plane_z - previous_position[2]) / denominator
				clipped_positions.append(
					previous_position + (factor * (current_position - previous_position))
				)
				clipped_intensities.append(
					previous_intensity + (factor * (current_intensity - previous_intensity))
				)

		if current_inside:
			clipped_positions.append(current_position.copy())
			clipped_intensities.append(current_intensity)

	if len(clipped_positions) < 3:
		return []

	clipped_positions = np.asarray(clipped_positions, dtype=np.float32)
	clipped_intensities = np.asarray(clipped_intensities, dtype=np.float32)
	triangles = []
	for index in range(1, len(clipped_positions) - 1):
		triangles.append(
			(
				np.stack(
					(
						clipped_positions[0],
						clipped_positions[index],
						clipped_positions[index + 1],
					),
					axis=0,
				).astype(np.float32),
				np.asarray(
					(
						clipped_intensities[0],
						clipped_intensities[index],
						clipped_intensities[index + 1],
					),
					dtype=np.float32,
				),
			)
		)
	return triangles


def is_back_facing(view_triangle):
	face_normal = np.cross(
		view_triangle[1] - view_triangle[0],
		view_triangle[2] - view_triangle[0],
	)
	face_to_camera = -view_triangle.mean(axis=0)
	return float(np.dot(face_normal, face_to_camera)) <= 0.0


def project_view_triangle(view_triangle, projection_matrix, target):
	triangle_h = np.concatenate(
		(view_triangle, np.ones((view_triangle.shape[0], 1), dtype=np.float32)),
		axis=1,
	)
	clip_triangle = transform_homogeneous(triangle_h, projection_matrix)
	clip_w = clip_triangle[:, 3]
	if np.any(clip_w <= MIN_CLIP_W):
		return None

	for axis in range(3):
		axis_values = clip_triangle[:, axis]
		if np.all(axis_values < -clip_w) or np.all(axis_values > clip_w):
			return None

	ndc_triangle = np.divide(
		clip_triangle[:, :3],
		clip_w[:, None],
		out=np.zeros_like(clip_triangle[:, :3]),
		where=np.abs(clip_w[:, None]) > MIN_CLIP_W,
	)
	screen_triangle = np.empty((view_triangle.shape[0], 2), dtype=np.float32)
	screen_triangle[:, 0] = (ndc_triangle[:, 0] * 0.5 + 0.5) * (target.pixel_width - 1)
	screen_triangle[:, 1] = (1.0 - (ndc_triangle[:, 1] * 0.5 + 0.5)) * (target.pixel_height - 1)
	depth_triangle = ((ndc_triangle[:, 2] + 1.0) * 0.5).astype(np.float32)
	inverse_w_triangle = np.divide(
		1.0,
		clip_w,
		out=np.zeros_like(clip_w, dtype=np.float32),
		where=np.abs(clip_w) > MIN_CLIP_W,
	)
	return screen_triangle, depth_triangle, inverse_w_triangle


def compute_screen_bounds(triangle_screen, target):
	min_x = max(int(math.floor(float(triangle_screen[:, 0].min()))), 0)
	max_x = min(int(math.ceil(float(triangle_screen[:, 0].max()))), target.pixel_width - 1)
	min_y = max(int(math.floor(float(triangle_screen[:, 1].min()))), 0)
	max_y = min(int(math.ceil(float(triangle_screen[:, 1].max()))), target.pixel_height - 1)
	if min_x > max_x or min_y > max_y:
		return None
	return np.array((min_x, max_x, min_y, max_y), dtype=np.int32)


def prepare_projected_triangles(mesh, args, world_vertices, world_normals, view_vertices, projection_matrix, target, fallback_color, projected_buffer):
	triangle_colors = mesh.triangle_colors if args.color else None
	projected_triangle_count = 0

	for triangle_index in range(0, mesh.triangles.shape[0], args.triangle_stride):
		vertex_indices = mesh.triangles[triangle_index]
		triangle_view = view_vertices[vertex_indices]
		if not args.no_backface_culling and is_back_facing(triangle_view):
			continue

		triangle_intensity = compute_triangle_intensity(
			mesh,
			triangle_index,
			world_vertices,
			world_normals,
			args,
			light_dir=args.light_dir,
			camera_pos=args.camera_pos,
		)
		clipped_triangles = clip_triangle_against_near_plane(
			triangle_view,
			triangle_intensity,
			args.near,
		)
		if not clipped_triangles:
			continue

		triangle_color = (
			triangle_colors[triangle_index]
			if triangle_colors is not None
			else fallback_color
		)

		for clipped_view_triangle, clipped_intensity in clipped_triangles:
			projected_triangle = project_view_triangle(
				clipped_view_triangle,
				projection_matrix,
				target,
			)
			if projected_triangle is None:
				continue

			triangle_screen, depth_triangle, inverse_w_triangle = projected_triangle
			triangle_bounds = compute_screen_bounds(triangle_screen, target)
			if triangle_bounds is None:
				continue

			if projected_triangle_count >= projected_buffer.capacity:
				raise RuntimeError("Projected triangle shared buffer 容量不足")

			projected_buffer.screen_triangles[projected_triangle_count] = triangle_screen
			projected_buffer.depth_triangles[projected_triangle_count] = depth_triangle
			projected_buffer.inverse_w_triangles[projected_triangle_count] = inverse_w_triangle
			projected_buffer.intensity_triangles[projected_triangle_count] = clipped_intensity
			projected_buffer.triangle_colors[projected_triangle_count] = triangle_color
			projected_buffer.bounds[projected_triangle_count] = triangle_bounds
			projected_triangle_count += 1

	return projected_triangle_count


def build_tile_tasks(projected_buffer, projected_triangle_count, target, tile_size):
	num_tiles_x = max(1, math.ceil(target.pixel_width / tile_size))
	num_tiles_y = max(1, math.ceil(target.pixel_height / tile_size))
	tile_bins = [[] for _ in range(num_tiles_x * num_tiles_y)]
	bounds_view = projected_buffer.bounds[:projected_triangle_count]

	for triangle_index, triangle_bounds in enumerate(bounds_view):
		start_tile_x = int(triangle_bounds[0]) // tile_size
		end_tile_x = int(triangle_bounds[1]) // tile_size
		start_tile_y = int(triangle_bounds[2]) // tile_size
		end_tile_y = int(triangle_bounds[3]) // tile_size

		for tile_y in range(start_tile_y, end_tile_y + 1):
			row_offset = tile_y * num_tiles_x
			for tile_x in range(start_tile_x, end_tile_x + 1):
				tile_bins[row_offset + tile_x].append(triangle_index)

	tasks = []
	for tile_index, triangle_indices in enumerate(tile_bins):
		if not triangle_indices:
			continue

		tile_x = tile_index % num_tiles_x
		tile_y = tile_index // num_tiles_x
		tile_min_x = tile_x * tile_size
		tile_min_y = tile_y * tile_size
		tile_max_x = min(tile_min_x + tile_size, target.pixel_width)
		tile_max_y = min(tile_min_y + tile_size, target.pixel_height)
		if tile_max_x <= tile_min_x or tile_max_y <= tile_min_y:
			continue
		triangle_indices_array = np.asarray(triangle_indices, dtype=np.int32)

		tasks.append(
			(
				tile_min_x,
				tile_min_y,
				triangle_indices_array,
			)
		)

	return tasks


def rasterize_tile_core(tile_min_x, tile_min_y, triangle_indices, projected_buffer, target_width, target_height, tile_size):
	tile_max_x = min(tile_min_x + tile_size, target_width)
	tile_max_y = min(tile_min_y + tile_size, target_height)

	tile_width = tile_max_x - tile_min_x
	tile_height = tile_max_y - tile_min_y
	tile_frame = np.zeros((tile_height, tile_width, 3), dtype=np.uint8)
	tile_depth = np.full((tile_height, tile_width), np.inf, dtype=np.float64)

	for triangle_index in triangle_indices:
		triangle_screen = projected_buffer.screen_triangles[triangle_index]
		triangle_screen64 = triangle_screen.astype(np.float64, copy=False)
		depth_triangle = projected_buffer.depth_triangles[triangle_index]
		depth_triangle64 = depth_triangle.astype(np.float64, copy=False)
		inverse_w_triangle = projected_buffer.inverse_w_triangles[triangle_index]
		inverse_w_triangle64 = inverse_w_triangle.astype(np.float64, copy=False)
		triangle_intensity = projected_buffer.intensity_triangles[triangle_index]
		triangle_color = projected_buffer.triangle_colors[triangle_index]
		triangle_bounds = projected_buffer.bounds[triangle_index]
		min_x = max(int(triangle_bounds[0]), tile_min_x)
		max_x = min(int(triangle_bounds[1]), tile_max_x - 1)
		min_y = max(int(triangle_bounds[2]), tile_min_y)
		max_y = min(int(triangle_bounds[3]), tile_max_y - 1)
		if min_x > max_x or min_y > max_y:
			continue

		point_a, point_b, point_c = triangle_screen64
		area = edge_function(point_a, point_b, point_c[0], point_c[1])
		if abs(float(area)) <= EPSILON:
			continue

		x_values, y_values = np.meshgrid(
			np.arange(min_x, max_x + 1, dtype=np.float64) + 0.5,
			np.arange(min_y, max_y + 1, dtype=np.float64) + 0.5,
		)
		w0 = edge_function(point_b, point_c, x_values, y_values)
		w1 = edge_function(point_c, point_a, x_values, y_values)
		w2 = edge_function(point_a, point_b, x_values, y_values)

		if area > 0.0:
			inside = (w0 >= 0.0) & (w1 >= 0.0) & (w2 >= 0.0)
		else:
			inside = (w0 <= 0.0) & (w1 <= 0.0) & (w2 <= 0.0)
		if not np.any(inside):
			continue

		barycentric = np.stack((w0, w1, w2), axis=-1) / area
		corrected = barycentric * inverse_w_triangle64[None, None, :]
		weight_sum = corrected.sum(axis=-1, keepdims=True)
		corrected = np.divide(
			corrected,
			weight_sum,
			out=np.zeros_like(corrected),
			where=np.abs(weight_sum) > EPSILON,
		)

		pixel_ndc_depth = np.sum(barycentric * depth_triangle64[None, None, :], axis=-1)
		pixel_inv_w = weight_sum[:, :, 0]
		pixel_z = np.divide(
			1.0,
			pixel_inv_w,
			out=np.full_like(pixel_inv_w, np.inf, dtype=np.float64),
			where=pixel_inv_w > EPSILON,
		)
		region_y0 = min_y - tile_min_y
		region_y1 = max_y - tile_min_y + 1
		region_x0 = min_x - tile_min_x
		region_x1 = max_x - tile_min_x + 1
		depth_region = tile_depth[region_y0:region_y1, region_x0:region_x1]
		closer = (
			inside
			& (pixel_inv_w > EPSILON)
			& (pixel_ndc_depth >= 0.0)
			& (pixel_ndc_depth <= 1.0)
			& (pixel_z < depth_region)
		)
		if not np.any(closer):
			continue

		pixel_intensity = np.sum(corrected * triangle_intensity[None, None, :], axis=-1)
		shade = np.clip(
			triangle_color[None, None, :] * pixel_intensity[:, :, None],
			0.0,
			255.0,
		).astype(np.uint8)
		frame_region = tile_frame[region_y0:region_y1, region_x0:region_x1]
		depth_region[closer] = pixel_z[closer]
		frame_region[closer] = shade[closer]

	return tile_min_x, tile_min_y, tile_frame


def rasterize_tile(task):
	if WORKER_PROJECTED_BUFFER is None or WORKER_FRAME_CONFIG is None:
		raise RuntimeError("Tile worker 尚未初始化 shared memory")

	tile_min_x, tile_min_y, triangle_indices = task
	return rasterize_tile_core(
		tile_min_x,
		tile_min_y,
		triangle_indices,
		WORKER_PROJECTED_BUFFER,
		int(WORKER_FRAME_CONFIG[FRAME_CONFIG_WIDTH_INDEX]),
		int(WORKER_FRAME_CONFIG[FRAME_CONFIG_HEIGHT_INDEX]),
		int(WORKER_FRAME_CONFIG[FRAME_CONFIG_TILE_SIZE_INDEX]),
	)


def composite_tiles_into_frame(tile_results, target):
	frame = np.zeros((target.pixel_height, target.pixel_width, 3), dtype=np.uint8)
	for tile_min_x, tile_min_y, tile_frame in tile_results:
		tile_height, tile_width = tile_frame.shape[:2]
		frame[
			tile_min_y : tile_min_y + tile_height,
			tile_min_x : tile_min_x + tile_width,
		] = tile_frame
	return frame


def render_mesh_to_rgb(
	mesh,
	args,
	scene_state,
	target,
	fallback_color,
	projected_buffer,
	frame_config,
	worker_pool=None,
):

	model_matrix = make_model_matrix(scene_state.scale, scene_state.rotation, scene_state.translation)
	view_matrix = make_look_at_matrix(args.camera_pos, args.look_at)
	projection_matrix = make_perspective_matrix(
		args.fov,
		target.pixel_width / max(target.pixel_height, 1),
		args.near,
		args.far,
	)

	world_vertices_h = transform_homogeneous(mesh.vertices_h, model_matrix)
	world_vertices = world_vertices_h[:, :3]
	view_vertices_h = transform_homogeneous(world_vertices_h, view_matrix)
	view_vertices = view_vertices_h[:, :3]

	world_normals = None
	if mesh.normals is not None:
		normal_matrix = np.linalg.inv(model_matrix[:3, :3]).T.astype(np.float32)
		world_normals = normalize_rows(mesh.normals @ normal_matrix.T)

	projected_triangle_count = prepare_projected_triangles(
		mesh,
		args,
		world_vertices,
		world_normals,
		view_vertices,
		projection_matrix,
		target,
		fallback_color,
		projected_buffer,
	)
	if projected_triangle_count == 0:
		return np.zeros((target.pixel_height, target.pixel_width, 3), dtype=np.uint8)

	frame_config[FRAME_CONFIG_WIDTH_INDEX] = target.pixel_width
	frame_config[FRAME_CONFIG_HEIGHT_INDEX] = target.pixel_height
	frame_config[FRAME_CONFIG_TILE_SIZE_INDEX] = args.tile_size

	tile_tasks = build_tile_tasks(projected_buffer, projected_triangle_count, target, args.tile_size)
	if not tile_tasks:
		return np.zeros((target.pixel_height, target.pixel_width, 3), dtype=np.uint8)

	if worker_pool is not None and len(tile_tasks) > 1:
		tile_results = worker_pool.map(rasterize_tile, tile_tasks)
	else:
		tile_results = [
			rasterize_tile_core(
				tile_min_x,
				tile_min_y,
				triangle_indices,
				projected_buffer,
				target.pixel_width,
				target.pixel_height,
				args.tile_size,
			)
			for tile_min_x, tile_min_y, triangle_indices in tile_tasks
		]

	return composite_tiles_into_frame(tile_results, target)


def render_mesh_to_rgb_gpu(args, scene_state, target, fallback_color, gpu_renderer):
	model_matrix = make_model_matrix(scene_state.scale, scene_state.rotation, scene_state.translation)
	view_matrix = make_look_at_matrix(args.camera_pos, args.look_at)
	projection_matrix = make_perspective_matrix(
		args.fov,
		target.pixel_width / max(target.pixel_height, 1),
		args.near,
		args.far,
	)
	normal_matrix = np.linalg.inv(model_matrix[:3, :3]).T.astype(np.float32)
	shading_mode = 1 if args.shading == "phong" else 0

	return gpu_renderer.render(
		model_matrix=model_matrix,
		view_matrix=view_matrix,
		projection_matrix=projection_matrix,
		normal_matrix=normal_matrix,
		light_dir=args.light_dir,
		camera_pos=args.camera_pos,
		fallback_color=fallback_color,
		near_plane=args.near,
		far_plane=args.far,
		depth_bias=args.gpu_depth_bias,
		triangle_stride=args.triangle_stride,
		use_mesh_color=args.color,
		disable_backface_culling=args.no_backface_culling,
		shading_mode=shading_mode,
		ambient=args.ambient,
		diffuse_weight=args.diffuse,
		specular_weight=args.specular,
		shininess=args.shininess,
	)


def apply_interactive_key(key, scene_state, args):
	if key in {"CTRL_C", "ESC"}:
		return False

	normalized_key = key.lower() if len(key) == 1 else key
	rotate_step = float(args.rotate_step)
	translate_step = float(args.translate_step)
	scale_step = float(args.scale_step)

	if normalized_key in {"w", "UP"}:
		scene_state.rotation[0] -= rotate_step
	elif normalized_key in {"s", "DOWN"}:
		scene_state.rotation[0] += rotate_step
	elif normalized_key in {"a", "LEFT"}:
		scene_state.rotation[1] -= rotate_step
	elif normalized_key in {"d", "RIGHT"}:
		scene_state.rotation[1] += rotate_step
	elif normalized_key == "q":
		scene_state.rotation[2] -= rotate_step
	elif normalized_key == "e":
		scene_state.rotation[2] += rotate_step
	elif normalized_key == "j":
		scene_state.translation[0] -= translate_step
	elif normalized_key == "l":
		scene_state.translation[0] += translate_step
	elif normalized_key == "i":
		scene_state.translation[1] += translate_step
	elif normalized_key == "k":
		scene_state.translation[1] -= translate_step
	elif normalized_key == "u":
		scene_state.translation[2] += translate_step
	elif normalized_key == "o":
		scene_state.translation[2] -= translate_step
	elif normalized_key == "z":
		scene_state.scale = max(0.05, scene_state.scale * (1.0 - scale_step))
	elif normalized_key == "x":
		scene_state.scale *= 1.0 + scale_step
	elif normalized_key == "p":
		scene_state.auto_rotation = not scene_state.auto_rotation
	elif normalized_key == "r":
		scene_state.reset()
	elif normalized_key == "m":
		args.shading = "lambert" if args.shading == "phong" else "phong"
	elif normalized_key == "b":
		args.no_backface_culling = not args.no_backface_culling

	return True


def render_stream(mesh, args):
	previous_size = None
	frame_count = 0
	base_color = np.asarray(args.base_color, dtype=np.float32)
	if not args.color:
		base_color = np.array((255.0, 255.0, 255.0), dtype=np.float32)
	scene_state = SceneState.from_args(args)
	last_frame_time = None
	projected_buffer = None
	frame_config_memory = None
	frame_config = None
	worker_pool = None
	gpu_renderer = None
	if args.backend == "cpu":
		projected_triangle_capacity = max(1, math.ceil(mesh.triangles.shape[0] / args.triangle_stride) * 2)
		projected_buffer = SharedProjectedTriangleBuffer.create(projected_triangle_capacity)
		frame_config_memory, frame_config = create_shared_frame_config()
		if args.workers > 1:
			worker_pool = mp.Pool(
				processes=args.workers,
				initializer=initialize_raster_worker,
				initargs=(
					projected_buffer.shm.name,
					projected_triangle_capacity,
					frame_config_memory.name,
				),
			)

	try:
		write_stdout_text("\033[?25l")

		with RawTerminalInput(enabled=args.interactive) as keyboard:
			if args.interactive and keyboard.enabled:
				print(INTERACTIVE_CONTROLS_TEXT, file=sys.stderr)

			running = True
			while running and (args.frames is None or frame_count < args.frames):
				frame_start = time.monotonic()
				if last_frame_time is None:
					delta_time = 0.0
				else:
					delta_time = frame_start - last_frame_time
				last_frame_time = frame_start

				for key in keyboard.poll_keys():
					if not apply_interactive_key(key, scene_state, args):
						running = False
						break
				if not running:
					break

				if scene_state.auto_rotation:
					scene_state.rotation += args.rotation_speed * delta_time

				target = build_render_target(max_width=args.width, supersample=args.supersample)
				if args.backend == "gpu":
					if gpu_renderer is None or not gpu_renderer.matches_target(target.pixel_width, target.pixel_height):
						gpu_renderer = build_taichi_renderer(mesh, target.pixel_width, target.pixel_height)
					rgb_frame = render_mesh_to_rgb_gpu(
						args,
						scene_state,
						target,
						base_color,
						gpu_renderer,
					)
				else:
					rgb_frame = render_mesh_to_rgb(
						mesh,
						args,
						scene_state,
						target,
						base_color,
						projected_buffer,
						frame_config,
						worker_pool=worker_pool,
					)
				final_rgb = downsample_rgb_frame(rgb_frame, args.supersample)
				braille_text = rgb_frame_to_braille(
					final_rgb,
					threshold=args.threshold,
					dither=args.dither,
					color=args.color,
				)
				frame_size = (target.braille_columns, target.braille_rows)
				print_braille_frame(braille_text, frame_size, previous_size)
				previous_size = frame_size
				frame_count += 1

				remaining = (1.0 / args.fps) - (time.monotonic() - frame_start)
				if remaining > 0.0:
					time.sleep(remaining)
	except KeyboardInterrupt:
		write_stdout_text("\n")
	finally:
		if worker_pool is not None:
			worker_pool.close()
			worker_pool.join()
		if projected_buffer is not None:
			projected_buffer.close()
			projected_buffer.unlink()
		if frame_config_memory is not None:
			frame_config_memory.close()
			frame_config_memory.unlink()
		write_stdout_text("\033[?25h")


def main():
	args = parse_args()
	model_path = os.path.abspath(os.path.expanduser(args.model))
	args.translate = np.asarray(args.translate, dtype=np.float32)
	args.rotate = np.asarray(args.rotate, dtype=np.float32)
	args.rotation_speed = np.asarray(args.rotation_speed, dtype=np.float32)
	args.camera_pos = np.asarray(args.camera_pos, dtype=np.float32)
	args.look_at = np.asarray(args.look_at, dtype=np.float32)
	args.light_dir = normalize_vector(np.asarray(args.light_dir, dtype=np.float32))
	if args.gpu_depth_bias is None:
		args.gpu_depth_bias = DEFAULT_GPU_DEPTH_BIAS if args.backend == "gpu" else 0.0

	try:
		mesh = load_obj_mesh(model_path)
		render_stream(mesh, args)
	except RuntimeError as exc:
		print(f"錯誤: {exc}", file=sys.stderr)
		return 1
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
