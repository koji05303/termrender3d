"""供 mvp_re_engine 使用的 Taichi CUDA rasterizer。"""

# pyright: reportInvalidTypeForm=false

import math

import numpy as np

try:
	import taichi as ti
except ImportError as exc:  # pragma: no cover - optional dependency
	ti = None
	_TAICHI_IMPORT_ERROR = exc
else:
	_TAICHI_IMPORT_ERROR = None

EPSILON = 1e-6
MIN_CLIP_W = 1e-5
DEPTH_QUANTIZATION_SCALE = float((1 << 32) - 2)
MAX_DEPTH_TRIANGLE_KEY = np.uint64((1 << 64) - 1)

_TAICHI_RUNTIME_READY = False



def taichi_available():
	#[DEBUG] 
    #print (f"Taichi import error: {_TAICHI_IMPORT_ERROR}")
	return ti is not None


def _require_taichi():
	if ti is None:
		raise RuntimeError(
			"GPU backend 需要 Taichi。請先安裝 taichi，或改用 --backend cpu。"
		) from _TAICHI_IMPORT_ERROR


def ensure_taichi_cuda_ready():
	global _TAICHI_RUNTIME_READY

	_require_taichi()
	if _TAICHI_RUNTIME_READY:
		return

	try:
		ti.init(arch=ti.cuda, offline_cache=True)
	except Exception as exc:  # pragma: no cover - depends on local GPU runtime
		raise RuntimeError(
			"無法初始化 Taichi CUDA backend。請確認 CUDA 驅動可用，或改用 --backend cpu。"
		) from exc

	_TAICHI_RUNTIME_READY = True


if ti is not None:

	@ti.data_oriented
	class TaichiCudaRasterizer:
		def __init__(self, mesh, width, height):
			ensure_taichi_cuda_ready()

			self.width = int(width)
			self.height = int(height)
			self.vertex_count = int(mesh.vertices_h.shape[0])
			self.triangle_count = int(mesh.triangles.shape[0])
			self.normal_count = int(mesh.normals.shape[0]) if mesh.normals is not None else 0
			self.has_normals = self.normal_count > 0 and mesh.normal_indices is not None
			self.projected_capacity = max(1, self.triangle_count * 2)

			self.vertices_h = ti.Vector.field(4, dtype=ti.f32, shape=self.vertex_count)
			self.triangles = ti.Vector.field(3, dtype=ti.i32, shape=self.triangle_count)
			self.triangle_colors = ti.Vector.field(3, dtype=ti.f32, shape=self.triangle_count)
			self.world_vertices = ti.Vector.field(3, dtype=ti.f32, shape=self.vertex_count)
			self.view_vertices = ti.Vector.field(3, dtype=ti.f32, shape=self.vertex_count)

			normal_field_size = max(1, self.normal_count)
			self.normals = ti.Vector.field(3, dtype=ti.f32, shape=normal_field_size)
			self.world_normals = ti.Vector.field(3, dtype=ti.f32, shape=normal_field_size)
			self.normal_indices = ti.Vector.field(3, dtype=ti.i32, shape=self.triangle_count)

			self.projected_screen = ti.Vector.field(
				2,
				dtype=ti.f32,
				shape=(self.projected_capacity, 3),
			)
			self.projected_depth = ti.Vector.field(3, dtype=ti.f32, shape=self.projected_capacity)
			self.projected_inverse_w = ti.Vector.field(3, dtype=ti.f32, shape=self.projected_capacity)
			self.projected_intensity = ti.Vector.field(3, dtype=ti.f32, shape=self.projected_capacity)
			self.projected_color = ti.Vector.field(3, dtype=ti.f32, shape=self.projected_capacity)
			self.projected_bounds = ti.Vector.field(4, dtype=ti.i32, shape=self.projected_capacity)
			self.projected_count = ti.field(dtype=ti.i32, shape=())

			self.pixel_triangle_key = ti.field(dtype=ti.u64, shape=(self.height, self.width))
			self.frame = ti.Vector.field(3, dtype=ti.u8, shape=(self.height, self.width))

			self.vertices_h.from_numpy(np.asarray(mesh.vertices_h, dtype=np.float32))
			self.triangles.from_numpy(np.asarray(mesh.triangles, dtype=np.int32))
			self.triangle_colors.from_numpy(np.asarray(mesh.triangle_colors, dtype=np.float32))

			if self.has_normals:
				self.normals.from_numpy(np.asarray(mesh.normals, dtype=np.float32))
				self.normal_indices.from_numpy(np.asarray(mesh.normal_indices, dtype=np.int32))
			else:
				self.normals.from_numpy(np.zeros((normal_field_size, 3), dtype=np.float32))
				self.normal_indices.from_numpy(
					np.full((self.triangle_count, 3), -1, dtype=np.int32)
				)

		def matches_target(self, width, height):
			return self.width == int(width) and self.height == int(height)

		@staticmethod
		def _matrix4(matrix):
			return ti.Matrix(np.asarray(matrix, dtype=np.float32).tolist(), dt=ti.f32)

		@staticmethod
		def _matrix3(matrix):
			return ti.Matrix(np.asarray(matrix, dtype=np.float32).tolist(), dt=ti.f32)

		@staticmethod
		def _vector3(vector):
			return ti.Vector(np.asarray(vector, dtype=np.float32).tolist(), dt=ti.f32)

		@ti.func
		def _safe_normalize(self, vector):
			length = ti.sqrt(vector.dot(vector))
			result = ti.Vector([0.0, 0.0, 0.0])
			if length > EPSILON:
				result = vector / length
			return result

		@ti.func
		def _edge_function(self, start, end, sample_x, sample_y):
			sx = ti.cast(sample_x, ti.f64)
			sy = ti.cast(sample_y, ti.f64)
			x0 = ti.cast(start[0], ti.f64)
			y0 = ti.cast(start[1], ti.f64)
			x1 = ti.cast(end[0], ti.f64)
			y1 = ti.cast(end[1], ti.f64)
			return ((sx - x0) * (y1 - y0)) - ((sy - y0) * (x1 - x0))

		@ti.func
		def _triangle_row(self, triangle_matrix, row_index):
			return ti.Vector(
				[
					triangle_matrix[row_index, 0],
					triangle_matrix[row_index, 1],
					triangle_matrix[row_index, 2],
				]
			)

		@ti.func
		def _is_back_facing(self, view_triangle):
			point_a = self._triangle_row(view_triangle, 0)
			point_b = self._triangle_row(view_triangle, 1)
			point_c = self._triangle_row(view_triangle, 2)
			face_normal = (point_b - point_a).cross(point_c - point_a)
			face_to_camera = -(point_a + point_b + point_c) / 3.0
			return face_normal.dot(face_to_camera) <= 0.0

		@ti.func
		def _compute_light_intensity(
			self,
			normal,
			position,
			light_dir,
			camera_pos,
			shading_mode,
			ambient,
			diffuse_weight,
			specular_weight,
			shininess,
		):
			normalized_normal = self._safe_normalize(normal)
			diffuse = ti.max(normalized_normal.dot(light_dir), 0.0)
			intensity = ambient + (diffuse_weight * diffuse)

			if shading_mode == 1:
				view_dir = self._safe_normalize(camera_pos - position)
				half_vector = self._safe_normalize(view_dir + light_dir)
				specular = ti.pow(ti.max(normalized_normal.dot(half_vector), 0.0), shininess)
				if diffuse > 0.0:
					intensity += specular_weight * specular

			return ti.min(1.0, ti.max(0.0, intensity))

		@ti.func
		def _compute_face_triangle_intensity(
			self,
			world_triangle,
			light_dir,
			camera_pos,
			shading_mode,
			ambient,
			diffuse_weight,
			specular_weight,
			shininess,
		):
			point_a = self._triangle_row(world_triangle, 0)
			point_b = self._triangle_row(world_triangle, 1)
			point_c = self._triangle_row(world_triangle, 2)
			face_normal = self._safe_normalize((point_b - point_a).cross(point_c - point_a))

			total_intensity = 0.0
			for corner in ti.static(range(3)):
				total_intensity += self._compute_light_intensity(
					face_normal,
					self._triangle_row(world_triangle, corner),
					light_dir,
					camera_pos,
					shading_mode,
					ambient,
					diffuse_weight,
					specular_weight,
					shininess,
				)

			average_intensity = total_intensity / 3.0
			return ti.Vector([average_intensity, average_intensity, average_intensity])

		@ti.func
		def _clip_triangle_against_near_plane(self, view_triangle, intensities, near_plane):
			clipped_positions = ti.Matrix.zero(ti.f32, 4, 3)
			clipped_intensities = ti.Vector.zero(ti.f32, 4)
			clipped_count = 0
			plane_z = -near_plane

			for index in ti.static(range(3)):
				previous_index = 2 if index == 0 else index - 1
				current_position = self._triangle_row(view_triangle, index)
				previous_position = self._triangle_row(view_triangle, previous_index)
				current_intensity = intensities[index]
				previous_intensity = intensities[previous_index]

				current_inside = current_position[2] <= plane_z
				previous_inside = previous_position[2] <= plane_z

				if current_inside != previous_inside:
					denominator = current_position[2] - previous_position[2]
					if ti.abs(denominator) > EPSILON:
						factor = (plane_z - previous_position[2]) / denominator
						clipped_position = previous_position + (
							factor * (current_position - previous_position)
						)
						clipped_positions[clipped_count, 0] = clipped_position[0]
						clipped_positions[clipped_count, 1] = clipped_position[1]
						clipped_positions[clipped_count, 2] = clipped_position[2]
						clipped_intensities[clipped_count] = previous_intensity + (
							factor * (current_intensity - previous_intensity)
						)
						clipped_count += 1

				if current_inside:
					clipped_positions[clipped_count, 0] = current_position[0]
					clipped_positions[clipped_count, 1] = current_position[1]
					clipped_positions[clipped_count, 2] = current_position[2]
					clipped_intensities[clipped_count] = current_intensity
					clipped_count += 1

			return clipped_positions, clipped_intensities, clipped_count

		@ti.func
		def _emit_projected_triangle(
			self,
			output_index,
			view_triangle,
			intensities,
			triangle_color,
			projection_matrix,
		):
			clip_positions = ti.Matrix.zero(ti.f32, 3, 4)
			clip_w = ti.Vector.zero(ti.f32, 3)
			accepted = 1

			for corner in ti.static(range(3)):
				point = self._triangle_row(view_triangle, corner)
				clip_point = projection_matrix @ ti.Vector([point[0], point[1], point[2], 1.0])
				for component in ti.static(range(4)):
					clip_positions[corner, component] = clip_point[component]
				clip_w[corner] = clip_point[3]
				if clip_point[3] <= MIN_CLIP_W:
					accepted = 0

			for axis in ti.static(range(3)):
				all_less = 1
				all_greater = 1
				for corner in ti.static(range(3)):
					axis_value = clip_positions[corner, axis]
					if axis_value >= -clip_w[corner]:
						all_less = 0
					if axis_value <= clip_w[corner]:
						all_greater = 0
				if all_less == 1 or all_greater == 1:
					accepted = 0

			ndc_triangle = ti.Matrix.zero(ti.f32, 3, 3)
			depth_triangle = ti.Vector.zero(ti.f32, 3)
			inverse_w_triangle = ti.Vector.zero(ti.f32, 3)
			screen_triangle = ti.Matrix.zero(ti.f32, 3, 2)
			min_x = 0
			max_x = 0
			min_y = 0
			max_y = 0


			if accepted == 1:
				for corner in ti.static(range(3)):
					point = self._triangle_row(view_triangle, corner)
					inverse_w = 0.0
					if ti.abs(clip_w[corner]) > MIN_CLIP_W:
						inverse_w = 1.0 / clip_w[corner]
					inverse_w_triangle[corner] = inverse_w

					ndc_x = clip_positions[corner, 0] * inverse_w
					ndc_y = clip_positions[corner, 1] * inverse_w
					ndc_triangle[corner, 0] = ndc_x
					ndc_triangle[corner, 1] = ndc_y
					ndc_triangle[corner, 2] = clip_positions[corner, 2] * inverse_w
					depth_triangle[corner] = -point[2]
					screen_triangle[corner, 0] = (ndc_x * 0.5 + 0.5) * (self.width - 1)
					screen_triangle[corner, 1] = (1.0 - (ndc_y * 0.5 + 0.5)) * (self.height - 1)

				min_x = ti.max(
					ti.cast(ti.floor(ti.min(screen_triangle[0, 0], screen_triangle[1, 0], screen_triangle[2, 0])), ti.i32),
					0,
				)
				max_x = ti.min(
					ti.cast(ti.ceil(ti.max(screen_triangle[0, 0], screen_triangle[1, 0], screen_triangle[2, 0])), ti.i32),
					self.width - 1,
				)
				min_y = ti.max(
					ti.cast(ti.floor(ti.min(screen_triangle[0, 1], screen_triangle[1, 1], screen_triangle[2, 1])), ti.i32),
					0,
				)
				max_y = ti.min(
					ti.cast(ti.ceil(ti.max(screen_triangle[0, 1], screen_triangle[1, 1], screen_triangle[2, 1])), ti.i32),
					self.height - 1,
				)

				if min_x > max_x or min_y > max_y:
					accepted = 0

			if accepted == 1:
				for corner in ti.static(range(3)):
					self.projected_screen[output_index, corner] = ti.Vector(
						[screen_triangle[corner, 0], screen_triangle[corner, 1]]
					)
				self.projected_depth[output_index] = depth_triangle
				self.projected_inverse_w[output_index] = inverse_w_triangle
				self.projected_intensity[output_index] = intensities
				self.projected_color[output_index] = triangle_color
				self.projected_bounds[output_index] = ti.Vector([min_x, max_x, min_y, max_y])
			return accepted

		@ti.kernel
		def _clear_framebuffers(self):
			max_key = ti.u64(0xFFFFFFFFFFFFFFFF)
			for pixel_y, pixel_x in self.pixel_triangle_key:
				self.pixel_triangle_key[pixel_y, pixel_x] = max_key
				self.frame[pixel_y, pixel_x] = ti.Vector([ti.u8(0), ti.u8(0), ti.u8(0)])

		@ti.kernel
		def _transform_vertices(
			self,
			model_matrix: ti.types.matrix(4, 4, ti.f32),
			view_matrix: ti.types.matrix(4, 4, ti.f32),
		):
			for vertex_index in range(self.vertex_count):
				world_position_h = model_matrix @ self.vertices_h[vertex_index]
				self.world_vertices[vertex_index] = ti.Vector(
					[world_position_h[0], world_position_h[1], world_position_h[2]]
				)
				view_position_h = view_matrix @ world_position_h
				self.view_vertices[vertex_index] = ti.Vector(
					[view_position_h[0], view_position_h[1], view_position_h[2]]
				)

		@ti.kernel
		def _transform_normals(self, normal_matrix: ti.types.matrix(3, 3, ti.f32)):
			for normal_index in range(self.normal_count):
				self.world_normals[normal_index] = self._safe_normalize(
					normal_matrix @ self.normals[normal_index]
				)

		@ti.kernel
		def _project_triangles(
			self,
			projection_matrix: ti.types.matrix(4, 4, ti.f32),
			light_dir: ti.types.vector(3, ti.f32),
			camera_pos: ti.types.vector(3, ti.f32),
			fallback_color: ti.types.vector(3, ti.f32),
			near_plane: ti.f32,
			triangle_stride: ti.i32,
			use_mesh_color: ti.i32,
			disable_backface_culling: ti.i32,
			shading_mode: ti.i32,
			ambient: ti.f32,
			diffuse_weight: ti.f32,
			specular_weight: ti.f32,
			shininess: ti.f32,
		):
			for triangle_index in range(self.triangle_count):
				if triangle_index % triangle_stride != 0:
					continue

				vertex_indices = self.triangles[triangle_index]
				world_triangle = ti.Matrix.zero(ti.f32, 3, 3)
				view_triangle = ti.Matrix.zero(ti.f32, 3, 3)

				for corner in ti.static(range(3)):
					world_triangle[corner, 0] = self.world_vertices[vertex_indices[corner]][0]
					world_triangle[corner, 1] = self.world_vertices[vertex_indices[corner]][1]
					world_triangle[corner, 2] = self.world_vertices[vertex_indices[corner]][2]
					view_triangle[corner, 0] = self.view_vertices[vertex_indices[corner]][0]
					view_triangle[corner, 1] = self.view_vertices[vertex_indices[corner]][1]
					view_triangle[corner, 2] = self.view_vertices[vertex_indices[corner]][2]

				if disable_backface_culling == 0 and self._is_back_facing(view_triangle):
					continue

				triangle_intensity = ti.Vector.zero(ti.f32, 3)
				if ti.static(self.has_normals):
					normal_indices = self.normal_indices[triangle_index]
					if (
						normal_indices[0] >= 0
						and normal_indices[1] >= 0
						and normal_indices[2] >= 0
					):
						for corner in ti.static(range(3)):
							triangle_intensity[corner] = self._compute_light_intensity(
								self.world_normals[normal_indices[corner]],
								self.world_vertices[vertex_indices[corner]],
								light_dir,
								camera_pos,
								shading_mode,
								ambient,
								diffuse_weight,
								specular_weight,
								shininess,
							)
					else:
						triangle_intensity = self._compute_face_triangle_intensity(
							world_triangle,
							light_dir,
							camera_pos,
							shading_mode,
							ambient,
							diffuse_weight,
							specular_weight,
							shininess,
						)
				else:
					triangle_intensity = self._compute_face_triangle_intensity(
						world_triangle,
						light_dir,
						camera_pos,
						shading_mode,
						ambient,
						diffuse_weight,
						specular_weight,
						shininess,
					)

				triangle_color = fallback_color
				if use_mesh_color == 1:
					triangle_color = self.triangle_colors[triangle_index]

				clipped_positions, clipped_intensities, clipped_count = self._clip_triangle_against_near_plane(
					view_triangle,
					triangle_intensity,
					near_plane,
				)
				if clipped_count < 3:
					continue

				for fan_index in ti.static(range(2)):
					if fan_index + 2 < clipped_count:
						clipped_triangle = ti.Matrix.zero(ti.f32, 3, 3)
						emitted_intensities = ti.Vector.zero(ti.f32, 3)
						clipped_triangle[0, 0] = clipped_positions[0, 0]
						clipped_triangle[0, 1] = clipped_positions[0, 1]
						clipped_triangle[0, 2] = clipped_positions[0, 2]
						emitted_intensities[0] = clipped_intensities[0]
						clipped_triangle[1, 0] = clipped_positions[fan_index + 1, 0]
						clipped_triangle[1, 1] = clipped_positions[fan_index + 1, 1]
						clipped_triangle[1, 2] = clipped_positions[fan_index + 1, 2]
						emitted_intensities[1] = clipped_intensities[fan_index + 1]
						clipped_triangle[2, 0] = clipped_positions[fan_index + 2, 0]
						clipped_triangle[2, 1] = clipped_positions[fan_index + 2, 1]
						clipped_triangle[2, 2] = clipped_positions[fan_index + 2, 2]
						emitted_intensities[2] = clipped_intensities[fan_index + 2]

						output_index = ti.atomic_add(self.projected_count[None], 1)
						if output_index < self.projected_capacity:
							accepted = self._emit_projected_triangle(
								output_index,
								clipped_triangle,
								emitted_intensities,
								triangle_color,
								projection_matrix,
							)
							if accepted == 0:
								ti.atomic_sub(self.projected_count[None], 1)
						else:
							ti.atomic_sub(self.projected_count[None], 1)

		@ti.kernel
		def _rasterize_triangles(
			self,
			projected_triangle_count: ti.i32,
			near_plane: ti.f32,
			far_plane: ti.f32,
			depth_bias: ti.f32,
		):
			for triangle_index in range(projected_triangle_count):
				bounds = self.projected_bounds[triangle_index]
				min_x = ti.max(bounds[0], 0)
				max_x = ti.min(bounds[1], self.width - 1)
				min_y = ti.max(bounds[2], 0)
				max_y = ti.min(bounds[3], self.height - 1)
				if min_x > max_x or min_y > max_y:
					continue

				point_a = self.projected_screen[triangle_index, 0]
				point_b = self.projected_screen[triangle_index, 1]
				point_c = self.projected_screen[triangle_index, 2]
				area_f64 = self._edge_function(point_a, point_b, point_c[0], point_c[1])
				if ti.abs(area_f64) <= ti.cast(EPSILON, ti.f64):
					continue

				inverse_w_triangle = self.projected_inverse_w[triangle_index]

				for pixel_y, pixel_x in ti.ndrange((min_y, max_y + 1), (min_x, max_x + 1)):
					sample_x = ti.cast(pixel_x, ti.f32) + 0.5
					sample_y = ti.cast(pixel_y, ti.f32) + 0.5
					w0_f64 = self._edge_function(point_b, point_c, sample_x, sample_y)
					w1_f64 = self._edge_function(point_c, point_a, sample_x, sample_y)
					w2_f64 = self._edge_function(point_a, point_b, sample_x, sample_y)

					inside = 0
					if area_f64 > 0.0:
						inside = 1 if w0_f64 >= 0.0 and w1_f64 >= 0.0 and w2_f64 >= 0.0 else 0
					else:
						inside = 1 if w0_f64 <= 0.0 and w1_f64 <= 0.0 and w2_f64 <= 0.0 else 0
					if inside == 0:
						continue

					barycentric = ti.Vector(
						[
							ti.cast(w0_f64 / area_f64, ti.f32),
							ti.cast(w1_f64 / area_f64, ti.f32),
							ti.cast(w2_f64 / area_f64, ti.f32),
						]
					)
					corrected = ti.Vector(
						[
							barycentric[0] * inverse_w_triangle[0],
							barycentric[1] * inverse_w_triangle[1],
							barycentric[2] * inverse_w_triangle[2],
						]
					)
					weight_sum = corrected[0] + corrected[1] + corrected[2]
					if ti.abs(weight_sum) <= EPSILON:
						continue
					corrected = corrected / weight_sum

					pixel_inv_w = barycentric.dot(inverse_w_triangle)
					if pixel_inv_w <= EPSILON:
						continue

					pixel_z = ti.cast((1.0 / pixel_inv_w) + depth_bias, ti.f32)
					if pixel_z < near_plane or pixel_z > far_plane:
						continue

					depth_u32 = ti.bit_cast(pixel_z, ti.u32)
					packed_key = (ti.cast(depth_u32, ti.u64) << 32) | ti.cast(triangle_index + 1, ti.u64)
					ti.atomic_min(self.pixel_triangle_key[pixel_y, pixel_x], packed_key)

		@ti.kernel
		def _shade_pixels(self):
			max_key = ti.u64(0xFFFFFFFFFFFFFFFF)
			triangle_mask = ti.u64(0xFFFFFFFF)

			for pixel_y, pixel_x in self.pixel_triangle_key:
				packed_key = self.pixel_triangle_key[pixel_y, pixel_x]
				if packed_key == max_key:
					continue

				triangle_index = ti.cast((packed_key & triangle_mask) - 1, ti.i32)
				point_a = self.projected_screen[triangle_index, 0]
				point_b = self.projected_screen[triangle_index, 1]
				point_c = self.projected_screen[triangle_index, 2]
				area_f64 = self._edge_function(point_a, point_b, point_c[0], point_c[1])
				if ti.abs(area_f64) <= ti.cast(EPSILON, ti.f64):
					continue

				sample_x = ti.cast(pixel_x, ti.f32) + 0.5
				sample_y = ti.cast(pixel_y, ti.f32) + 0.5
				w0_f64 = self._edge_function(point_b, point_c, sample_x, sample_y)
				w1_f64 = self._edge_function(point_c, point_a, sample_x, sample_y)
				w2_f64 = self._edge_function(point_a, point_b, sample_x, sample_y)
				barycentric = ti.Vector(
					[
						ti.cast(w0_f64 / area_f64, ti.f32),
						ti.cast(w1_f64 / area_f64, ti.f32),
						ti.cast(w2_f64 / area_f64, ti.f32),
					]
				)
				corrected = ti.Vector(
					[
						barycentric[0] * self.projected_inverse_w[triangle_index][0],
						barycentric[1] * self.projected_inverse_w[triangle_index][1],
						barycentric[2] * self.projected_inverse_w[triangle_index][2],
					]
				)
				weight_sum = corrected[0] + corrected[1] + corrected[2]
				if ti.abs(weight_sum) <= EPSILON:
					continue
				corrected = corrected / weight_sum

				pixel_intensity = corrected.dot(self.projected_intensity[triangle_index])
				shaded_color = ti.min(
					ti.max(
						self.projected_color[triangle_index] * pixel_intensity,
						0.0,
					),
					255.0,
				)
				self.frame[pixel_y, pixel_x] = ti.cast(shaded_color, ti.u8)

		def render(
			self,
			model_matrix,
			view_matrix,
			projection_matrix,
			normal_matrix,
			light_dir,
			camera_pos,
			fallback_color,
			near_plane,
			far_plane,
			depth_bias,
			triangle_stride,
			use_mesh_color,
			disable_backface_culling,
			shading_mode,
			ambient,
			diffuse_weight,
			specular_weight,
			shininess,
		):
			self.projected_count[None] = 0
			self._clear_framebuffers()
			self._transform_vertices(self._matrix4(model_matrix), self._matrix4(view_matrix))
			if self.has_normals:
				self._transform_normals(self._matrix3(normal_matrix))
			self._project_triangles(
				self._matrix4(projection_matrix),
				self._vector3(light_dir),
				self._vector3(camera_pos),
				self._vector3(fallback_color),
				float(near_plane),
				int(triangle_stride),
				int(bool(use_mesh_color)),
				int(bool(disable_backface_culling)),
				int(shading_mode),
				float(ambient),
				float(diffuse_weight),
				float(specular_weight),
				float(shininess),
			)

			projected_triangle_count = int(self.projected_count[None])
			if projected_triangle_count <= 0:
				return np.zeros((self.height, self.width, 3), dtype=np.uint8)

			self._rasterize_triangles(
				projected_triangle_count,
				float(near_plane),
				float(far_plane),
				float(depth_bias),
			)
			self._shade_pixels()
			return self.frame.to_numpy()


else:

	class TaichiCudaRasterizer:  # pragma: no cover - fallback path only
		def __init__(self, *_args, **_kwargs):
			_require_taichi()


def build_taichi_renderer(mesh, width, height):
	ensure_taichi_cuda_ready()
	return TaichiCudaRasterizer(mesh, width, height)