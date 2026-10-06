"""Offscreen video works headless; README explains every file/function; the vehicle layer has no MuJoCo."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
BACKEND_AGNOSTIC = ["config", "geometry", "vehicle", "aero", "sensors", "touchdown"]


def test_offscreen_render_and_mp4(make_sim, tmp_path):
    from lander.scenarios import make_scenario
    from lander.video import FrameRenderer, VideoWriter
    sim = make_sim()
    sc = make_scenario("drop", sim)
    res = sim.reset(initial_state=sc.initial_state)
    try:
        frames = FrameRenderer(sim, 320, 240)
    except Exception as exc:                      # no OpenGL context on this machine
        pytest.skip(f"offscreen rendering unavailable: {exc}")
    img = frames.render(res)
    frames.close()
    assert img.shape == (240, 320, 3) and img.std() > 5
    out = tmp_path / "clip.mp4"
    video = VideoWriter(sim, out, 320, 240)
    for _ in range(5):
        video.add(sim.step())
    video.close()
    assert out.exists() and out.stat().st_size > 1000 and video.count == 5


def _public_names(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not node.name.startswith("_"):
            names.append(node.name)
            if isinstance(node, ast.ClassDef):
                names += [f"{node.name}.{n.name}" for n in node.body
                          if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")]
    return names


def test_readme_explains_every_file_and_function():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    missing = []
    for folder in ("lander", "scripts"):
        for path in sorted((ROOT / folder).glob("*.py")):
            if path.name == "__init__.py":
                continue
            if f"{folder}/{path.name}" not in readme:
                missing.append(f"{folder}/{path.name}")
            for name in _public_names(path):
                short = name.split(".")[-1]
                if f"`{name}" not in readme and f"`{short}" not in readme:
                    missing.append(f"{folder}/{path.name}: {name}")
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        if f"tests/{path.name}" not in readme:
            missing.append(f"tests/{path.name}")
    assert not missing, "README.md does not mention: " + ", ".join(missing)


def test_vehicle_layer_does_not_import_mujoco():
    for mod in BACKEND_AGNOSTIC:
        tree = ast.parse((ROOT / "lander" / f"{mod}.py").read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not any(a.name.startswith("mujoco") for a in node.names), mod
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("mujoco"), mod
