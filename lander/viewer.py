"""Live 3-D viewer (MuJoCo passive viewer, a normal window on Windows) for WATCHING the lander.

There is no manual flight: commands come from a policy function (a scripted scenario now, the RL
agent from M4). Keys: Space = pause / resume, Enter = restart the episode. Everything else is
MuJoCo's own viewer controls (mouse to orbit/zoom, [ and ] to cycle cameras, Esc for a free camera).
"""

from __future__ import annotations

import time
from typing import Callable

import mujoco
import mujoco.viewer

from .hud import hud_columns
from .visuals import decorate, tracking_camera

KEY_SPACE, KEY_ENTER = 32, 257     # GLFW key codes


def watch(sim, policy: Callable, reset_fn: Callable, speed: float = 1.0, max_seconds: float | None = None,
          hud: bool = True, hold_after_done_s: float = 2.0) -> None:
    """Run `policy(sim) -> Command` in real time (scaled by `speed`) in the passive viewer.

    `reset_fn()` must reset the simulator and return the first StepResult; it is called at start and
    whenever Enter is pressed. After a landing is decided the sim keeps running `hold_after_done_s`
    seconds and then pauses on the result.
    """
    flags = {"paused": False, "restart": False}

    def on_key(key: int) -> None:            # runs on the viewer thread: only set flags here
        if key == KEY_SPACE:
            flags["paused"] = not flags["paused"]
        elif key == KEY_ENTER:
            flags["restart"] = True

    res = reset_fn()
    wall0 = time.perf_counter()
    with mujoco.viewer.launch_passive(sim.model, sim.data, key_callback=on_key,
                                      show_left_ui=False, show_right_ui=False) as v:
        cam = tracking_camera(sim)
        with v.lock():
            v.cam.type, v.cam.trackbodyid = cam.type, cam.trackbodyid
            v.cam.distance, v.cam.azimuth, v.cam.elevation = cam.distance, cam.azimuth, cam.elevation
        anchor_wall, anchor_sim = time.perf_counter(), sim.t
        done_at = None
        while v.is_running():
            if max_seconds is not None and time.perf_counter() - wall0 > max_seconds:
                break
            if flags["restart"]:
                flags["restart"], done_at = False, None
                res = reset_fn()
                anchor_wall, anchor_sim = time.perf_counter(), sim.t
            hold = done_at is not None and sim.t >= done_at + hold_after_done_s
            if not flags["paused"] and not hold:
                res = sim.step(policy(sim))
                if res.report.done and done_at is None:
                    done_at = sim.t
            else:
                anchor_wall, anchor_sim = time.perf_counter(), sim.t
            with v.lock():
                v.user_scn.ngeom = 0
                decorate(v.user_scn, sim)
            if hud:
                labels, values = hud_columns(sim, res)
                v.set_texts([(None, mujoco.mjtGridPos.mjGRID_TOPLEFT, labels, values)])
            v.sync()
            lag = (sim.t - anchor_sim) / max(speed, 1e-6) - (time.perf_counter() - anchor_wall)
            time.sleep(max(0.0, lag) if not flags["paused"] and not hold else 0.02)
