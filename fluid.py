"""
TermRender fluid demo.

This module implements the TODO at the top of the original draft:

Input image -> fixed 2D Eulerian grid -> RGB dye advection ->
incompressible velocity projection -> TermRender terminal output.

It intentionally stays small: no particles, free surfaces, GPU backend,
MacCormack advection, diffusion, curl noise, or cut-cell SDF handling.
"""

from __future__ import annotations

import select
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace


np = None
Image = None
ImageOps = None


@dataclass
class FluidConfig:
    size: int
    steps: int
    dt: float
    pressure_iters: int
    force_strength: float
    initial_swirl: float
    velocity_damping: float
    dye_decay: float
    obstacle: str
    obstacle_size: float


def require_runtime_deps():
    global np, Image, ImageOps

    if np is not None and Image is not None and ImageOps is not None:
        return

    try:
        import numpy as numpy_module
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "missing dependency 'numpy'. Install the README dependencies or run in the yolo conda env."
        ) from exc

    try:
        from PIL import Image as pil_image
        from PIL import ImageOps as pil_image_ops
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "missing dependency 'Pillow'. Install the README dependencies or run in the yolo conda env."
        ) from exc

    np = numpy_module
    Image = pil_image
    ImageOps = pil_image_ops


def resize_filter():
    if hasattr(Image, "Resampling"):
        return Image.Resampling.LANCZOS
    return Image.LANCZOS


def make_demo_dye(size):
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    nx = x / max(1, size - 1)
    ny = y / max(1, size - 1)

    dye = np.zeros((size, size, 3), dtype=np.float32)
    dye[..., 0] = np.clip(1.25 - 2.4 * np.hypot(nx - 0.35, ny - 0.45), 0.0, 1.0)
    dye[..., 1] = np.clip(1.15 - 2.1 * np.hypot(nx - 0.67, ny - 0.52), 0.0, 1.0)
    dye[..., 2] = np.clip(1.05 - 2.0 * np.hypot(nx - 0.50, ny - 0.35), 0.0, 1.0)

    spacing = max(8, size // 12)
    dye[::spacing, :, 0] = 1.0
    dye[:, ::spacing, 1] = 1.0
    dye[size // 3 : 2 * size // 3, size // 3 : 2 * size // 3, 2] = np.maximum(
        dye[size // 3 : 2 * size // 3, size // 3 : 2 * size // 3, 2],
        0.75,
    )
    return dye


def load_dye(image_path, size):
    if not image_path:
        return make_demo_dye(size)

    path = Path(image_path).expanduser()
    if not path.is_file():
        raise RuntimeError(f"input image not found: {path}")

    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = image.resize((size, size), resize_filter())
        return np.asarray(image, dtype=np.float32) / 255.0


def build_solid_mask(size, obstacle, obstacle_size):
    solid = np.zeros((size, size), dtype=bool)
    solid[0, :] = True
    solid[-1, :] = True
    solid[:, 0] = True
    solid[:, -1] = True

    if obstacle == "circle":
        y, x = np.mgrid[0:size, 0:size]
        center = (size - 1) * 0.5
        radius = max(1.0, size * obstacle_size)
        solid |= ((x - center) ** 2 + (y - center) ** 2) <= radius**2
    elif obstacle == "box":
        half_width = max(1, int(size * obstacle_size))
        center = size // 2
        solid[
            center - half_width : center + half_width + 1,
            center - half_width : center + half_width + 1,
        ] = True

    return solid


def normalized_grid(size):
    axis = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    x, y = np.meshgrid(axis, axis)
    return x, y


def vortex_field(size, strength, t=0.0, center_x=0.0, center_y=0.0):
    x, y = normalized_grid(size)
    dx = x - center_x
    dy = y - center_y
    r2 = dx * dx + dy * dy + 1e-6
    falloff = np.exp(-3.5 * r2).astype(np.float32)
    pulse = 1.0 + 0.25 * np.sin(t * 1.7)

    u = -dy * falloff * strength * pulse
    v = dx * falloff * strength * pulse

    inward = -0.10 * strength * falloff
    u += dx * inward
    v += dy * inward
    return u.astype(np.float32), v.astype(np.float32)


def external_forces(size, strength, t):
    main_u, main_v = vortex_field(size, strength, t=t)
    satellite_x = 0.42 * np.sin(t * 0.7)
    satellite_y = 0.35 * np.cos(t * 0.9)
    sat_u, sat_v = vortex_field(
        size,
        -0.35 * strength,
        t=t + 1.7,
        center_x=satellite_x,
        center_y=satellite_y,
    )
    return main_u + sat_u, main_v + sat_v


def directional_force_field(size, force_x, force_y):
    x, y = normalized_grid(size)
    center_weight = 0.68 + 0.32 * np.exp(-1.8 * (x * x + y * y))
    return (
        (center_weight * force_x).astype(np.float32),
        (center_weight * force_y).astype(np.float32),
    )


class KeyboardForceController:
    KEY_VECTORS = {
        "w": (0.0, -1.0),
        "a": (-1.0, 0.0),
        "s": (0.0, 1.0),
        "d": (1.0, 0.0),
        "\x1b[A": (0.0, -1.0),
        "\x1b[D": (-1.0, 0.0),
        "\x1b[B": (0.0, 1.0),
        "\x1b[C": (1.0, 0.0),
    }

    def __init__(self, enabled=True, strength=28.0, decay=0.78):
        self.enabled = enabled and sys.stdin.isatty()
        self.strength = float(strength)
        self.decay = float(decay)
        self.vector = np.zeros(2, dtype=np.float32)
        self.buffer = ""
        self.fd = None
        self.termios = None
        self.old_attrs = None

    def __enter__(self):
        if not self.enabled:
            return self

        import termios
        import tty

        self.termios = termios
        self.fd = sys.stdin.fileno()
        self.old_attrs = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.enabled and self.old_attrs is not None:
            self.termios.tcsetattr(self.fd, self.termios.TCSADRAIN, self.old_attrs)

    def poll_force(self):
        if not self.enabled:
            return None

        self._read_available()
        impulse = self._consume_impulse()
        if impulse is not None:
            self.vector = impulse
        else:
            self.vector *= self.decay

        magnitude = float(np.linalg.norm(self.vector))
        if magnitude < 0.05:
            self.vector[:] = 0.0
            return None

        direction = self.vector / magnitude
        return float(direction[0] * self.strength), float(direction[1] * self.strength)

    def _read_available(self):
        while select.select([sys.stdin], [], [], 0.0)[0]:
            chunk = sys.stdin.read(1)
            if not chunk:
                break
            self.buffer += chunk

    def _consume_impulse(self):
        impulse = np.zeros(2, dtype=np.float32)
        saw_key = False

        while self.buffer:
            key = None
            if self.buffer.startswith("\x1b["):
                if len(self.buffer) < 3:
                    break
                key = self.buffer[:3]
                self.buffer = self.buffer[3:]
            else:
                key = self.buffer[0].lower()
                self.buffer = self.buffer[1:]

            vector = self.KEY_VECTORS.get(key)
            if vector is None:
                continue

            impulse += np.array(vector, dtype=np.float32)
            saw_key = True

        if not saw_key:
            return None

        magnitude = float(np.linalg.norm(impulse))
        if magnitude > 0.0:
            impulse /= magnitude
        return impulse


def bilinear_sample(field, src_x, src_y):
    height, width = field.shape[:2]
    x0 = np.floor(src_x).astype(np.int32)
    y0 = np.floor(src_y).astype(np.int32)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)

    wx = src_x - x0
    wy = src_y - y0

    if field.ndim == 2:
        top = (1.0 - wx) * field[y0, x0] + wx * field[y0, x1]
        bottom = (1.0 - wx) * field[y1, x0] + wx * field[y1, x1]
        return ((1.0 - wy) * top + wy * bottom).astype(field.dtype)

    wx = wx[..., None]
    wy = wy[..., None]
    top = (1.0 - wx) * field[y0, x0, :] + wx * field[y0, x1, :]
    bottom = (1.0 - wx) * field[y1, x0, :] + wx * field[y1, x1, :]
    return ((1.0 - wy) * top + wy * bottom).astype(field.dtype)


def advect_field(field, u, v, dt, solid):
    height, width = u.shape
    y, x = np.mgrid[0:height, 0:width]
    src_x = np.clip(x.astype(np.float32) - dt * u, 0.0, width - 1.0)
    src_y = np.clip(y.astype(np.float32) - dt * v, 0.0, height - 1.0)
    advected = bilinear_sample(field, src_x, src_y)

    if advected.ndim == 2:
        advected[solid] = 0.0
    else:
        advected[solid, :] = 0.0
    return advected


def apply_velocity_boundary(u, v, solid):
    u = u.copy()
    v = v.copy()

    left_solid = np.roll(solid, 1, axis=1)
    right_solid = np.roll(solid, -1, axis=1)
    up_solid = np.roll(solid, 1, axis=0)
    down_solid = np.roll(solid, -1, axis=0)

    u[solid] = 0.0
    v[solid] = 0.0
    u[(right_solid & (u > 0.0)) | (left_solid & (u < 0.0))] = 0.0
    v[(down_solid & (v > 0.0)) | (up_solid & (v < 0.0))] = 0.0

    u[:, 0] = 0.0
    u[:, -1] = 0.0
    u[0, :] = 0.0
    u[-1, :] = 0.0
    v[:, 0] = 0.0
    v[:, -1] = 0.0
    v[0, :] = 0.0
    v[-1, :] = 0.0
    return u, v


def compute_divergence(u, v, solid):
    div = 0.5 * (
        np.roll(u, -1, axis=1)
        - np.roll(u, 1, axis=1)
        + np.roll(v, -1, axis=0)
        - np.roll(v, 1, axis=0)
    )
    div[solid] = 0.0
    return div.astype(np.float32)


def pressure_neighbor(pressure, solid, shift, axis):
    neighbor_pressure = np.roll(pressure, shift, axis=axis)
    neighbor_solid = np.roll(solid, shift, axis=axis)
    return np.where(neighbor_solid, pressure, neighbor_pressure)


def solve_pressure(divergence, solid, iterations):
    pressure = np.zeros_like(divergence, dtype=np.float32)
    fluid = ~solid

    for _ in range(iterations):
        left = pressure_neighbor(pressure, solid, 1, axis=1)
        right = pressure_neighbor(pressure, solid, -1, axis=1)
        up = pressure_neighbor(pressure, solid, 1, axis=0)
        down = pressure_neighbor(pressure, solid, -1, axis=0)
        next_pressure = (left + right + up + down - divergence) * 0.25
        pressure = np.where(fluid, next_pressure, 0.0).astype(np.float32)

    return pressure


def subtract_pressure_gradient(u, v, pressure, solid):
    left = pressure_neighbor(pressure, solid, 1, axis=1)
    right = pressure_neighbor(pressure, solid, -1, axis=1)
    up = pressure_neighbor(pressure, solid, 1, axis=0)
    down = pressure_neighbor(pressure, solid, -1, axis=0)

    u = u - 0.5 * (right - left)
    v = v - 0.5 * (down - up)
    return apply_velocity_boundary(u.astype(np.float32), v.astype(np.float32), solid)


def simulation_step(dye, u, v, solid, config, step_index, keyboard_force=None):
    t = step_index * config.dt
    if keyboard_force is None:
        force_u, force_v = external_forces(config.size, config.force_strength, t)
    else:
        force_u, force_v = directional_force_field(config.size, *keyboard_force)

    u = (u + config.dt * force_u) * config.velocity_damping
    v = (v + config.dt * force_v) * config.velocity_damping

    old_u = u.copy()
    old_v = v.copy()
    u = advect_field(old_u, old_u, old_v, config.dt, solid)
    v = advect_field(old_v, old_u, old_v, config.dt, solid)
    u, v = apply_velocity_boundary(u, v, solid)

    divergence = compute_divergence(u, v, solid)
    pressure = solve_pressure(divergence, solid, config.pressure_iters)
    u, v = subtract_pressure_gradient(u, v, pressure, solid)

    dye = advect_field(dye, u, v, config.dt, solid)
    dye = np.clip(dye * config.dye_decay, 0.0, 1.0).astype(np.float32)
    dye[solid, :] = 0.0
    return dye, u, v


def dye_to_rgb(dye):
    return np.clip(dye * 255.0, 0.0, 255.0).astype(np.uint8)


def dye_to_bgr(dye):
    return dye_to_rgb(dye)[..., ::-1].copy()


def save_dye_png(path, dye):
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(dye_to_rgb(dye)).save(output_path)
    return output_path


class TerminalFluidRenderer:
    def __init__(self, args):
        import termrender

        self.termrender = termrender
        self.args = SimpleNamespace(
            mode=args.render,
            width=args.width,
            workers=args.workers,
            threshold=args.threshold,
            dither=args.dither,
            color=args.color,
            mirror=False,
        )
        self.previous_size = None
        self.braille_backend = None
        self.ascii_backend = None
        self.ascii_pool_context = None
        self.ascii_pool = None

        if args.render == "braille":
            self.braille_backend = termrender.get_braille_backend()
        elif args.render == "ascii":
            self.ascii_backend = termrender.get_ascii_backend()
            self.ascii_pool_context = termrender.ascii_worker_pool(
                self.args,
                self.ascii_backend,
            )
            self.ascii_pool = self.ascii_pool_context.__enter__()

    def hide_cursor(self):
        self.termrender.write_stdout_text("\033[?25l")

    def show_cursor(self):
        self.termrender.write_stdout_text("\033[?25h")

    def close(self):
        if self.ascii_pool_context is not None:
            self.ascii_pool_context.__exit__(None, None, None)
            self.ascii_pool_context = None
            self.ascii_pool = None

    def render(self, frame_bgr):
        text, frame_size = self.termrender.render_frame(
            frame_bgr,
            self.args,
            braille_backend=self.braille_backend,
            ascii_backend=self.ascii_backend,
            ascii_pool=self.ascii_pool,
        )
        self.termrender.print_terminal_frame(text, frame_size, self.previous_size)
        self.previous_size = frame_size


def should_emit_frame(step_index, config, render_every):
    return step_index == 0 or step_index == config.steps or step_index % render_every == 0


def run_simulation(args):
    require_runtime_deps()
    args.render = getattr(args, "render", getattr(args, "mode", "braille"))

    config = FluidConfig(
        size=args.size,
        steps=args.steps,
        dt=args.dt,
        pressure_iters=args.pressure_iters,
        force_strength=args.force_strength,
        initial_swirl=args.initial_swirl,
        velocity_damping=args.velocity_damping,
        dye_decay=args.dye_decay,
        obstacle=args.obstacle,
        obstacle_size=args.obstacle_size,
    )

    dye = load_dye(args.image, config.size)
    solid = build_solid_mask(config.size, config.obstacle, config.obstacle_size)
    u, v = vortex_field(config.size, config.initial_swirl)
    u, v = apply_velocity_boundary(u, v, solid)
    dye[solid, :] = 0.0

    renderer = None
    if args.render != "none":
        renderer = TerminalFluidRenderer(args)
        renderer.hide_cursor()

    frame_interval = 1.0 / args.fps
    save_frames_dir = Path(args.save_frames).expanduser() if args.save_frames else None
    if save_frames_dir:
        save_frames_dir.mkdir(parents=True, exist_ok=True)

    keyboard_enabled = getattr(args, "interactive", True) and args.render != "none"
    keyboard_strength = getattr(args, "keyboard_force", 28.0)
    keyboard_decay = getattr(args, "keyboard_decay", 0.78)

    try:
        with KeyboardForceController(
            enabled=keyboard_enabled,
            strength=keyboard_strength,
            decay=keyboard_decay,
        ) as keyboard:
            for step_index in range(config.steps + 1):
                frame_start = time.monotonic()

                if step_index > 0:
                    keyboard_force = keyboard.poll_force()
                    dye, u, v = simulation_step(
                        dye,
                        u,
                        v,
                        solid,
                        config,
                        step_index,
                        keyboard_force=keyboard_force,
                    )

                if should_emit_frame(step_index, config, args.render_every):
                    if save_frames_dir:
                        save_dye_png(save_frames_dir / f"fluid_{step_index:04d}.png", dye)

                    if renderer is not None:
                        renderer.render(dye_to_bgr(dye))
                        remaining = frame_interval - (time.monotonic() - frame_start)
                        if remaining > 0.0:
                            time.sleep(remaining)
    except KeyboardInterrupt:
        if renderer is not None:
            renderer.termrender.write_stdout_text("\n")
    finally:
        if renderer is not None:
            renderer.show_cursor()
            renderer.close()

    if args.save_final:
        return save_dye_png(args.save_final, dye)
    return None


def main(argv=None):
    import termrender

    if argv is None:
        argv = sys.argv[1:]
    sys.modules.setdefault("fluid", sys.modules[__name__])
    return termrender.main(["render", "fluid", *argv])


if __name__ == "__main__":
    raise SystemExit(main())
