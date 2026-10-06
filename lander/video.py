"""Offscreen rendering to images and MP4 (works without a window; on Linux/Colab set MUJOCO_GL=egl).

One frame per control step, so the video plays in real time at SIM_CONTROL_HZ frames per second.
The HUD is drawn onto each frame with Pillow.
"""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .hud import hud_lines
from .visuals import decorate, named_camera


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                      # very old Pillow
        return ImageFont.load_default()


def overlay_hud(img: np.ndarray, rows: list[tuple[str, str]], font_size: int | None = None) -> np.ndarray:
    """Draw (label, value) rows in a compact top-left panel; the last row ('Status') as a bottom banner."""
    im = Image.fromarray(img).convert("RGBA")
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    size = font_size or max(11, im.size[1] // 45)
    font, line_h = _font(size), int(size * 1.25)
    *panel, (status_label, status) = rows
    label_w = max(draw.textlength(label, font=font) for label, _ in panel) + size
    value_w = max(draw.textlength(value, font=font) for _, value in panel)
    draw.rectangle([6, 6, 18 + label_w + value_w, 12 + line_h * len(panel)], fill=(0, 0, 0, 110))
    for i, (label, value) in enumerate(panel):
        y = 9 + i * line_h
        draw.text((12, y), label, font=font, fill=(255, 215, 150, 255))
        draw.text((12 + label_w, y), value, font=font, fill=(255, 255, 255, 255))
    big = _font(int(size * 1.2))
    text = f"{status_label}: {status}"
    w = draw.textlength(text, font=big)
    x0, y0 = (im.size[0] - w) / 2, im.size[1] - 2.2 * size - 6
    draw.rectangle([x0 - 10, y0 - 6, x0 + w + 10, y0 + 1.4 * size + 6], fill=(0, 0, 0, 130))
    draw.text((x0, y0), text, font=big, fill=(255, 255, 255, 255))
    return np.asarray(Image.alpha_composite(im, layer).convert("RGB"))


class FrameRenderer:
    """Renders the current simulator state (plus plumes, arrows and HUD) to an RGB array."""

    def __init__(self, sim, width: int = 1280, height: int = 720, camera: str = "chase", hud: bool = True):
        self.sim, self.hud = sim, hud
        self.renderer = mujoco.Renderer(sim.model, height=height, width=width)
        self.camera = named_camera(sim, camera)

    def render(self, res=None) -> np.ndarray:
        self.renderer.update_scene(self.sim.data, camera=self.camera)
        decorate(self.renderer.scene, self.sim)
        img = self.renderer.render()
        if self.hud and res is not None:
            img = overlay_hud(img, hud_lines(self.sim, res))
        return img

    def close(self) -> None:
        self.renderer.close()


class VideoWriter:
    """MP4 writer: call `add(res)` after every `sim.step()`, then `close()`."""

    def __init__(self, sim, path: str | Path, width: int = 1280, height: int = 720, camera: str = "chase",
                 hud: bool = True, fps: float | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.frames = FrameRenderer(sim, width, height, camera, hud)
        self.writer = imageio.get_writer(str(self.path), fps=fps or sim.cfg.sim.control_hz, codec="libx264",
                                         quality=7, macro_block_size=8, ffmpeg_log_level="error")
        self.count = 0

    def add(self, res) -> None:
        self.writer.append_data(self.frames.render(res))
        self.count += 1

    def close(self) -> None:
        self.writer.close()
        self.frames.close()
