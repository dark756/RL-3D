"""Extra scene geometry drawn over the MuJoCo model, shared by the live viewer and the video export:
engine plumes sized by each engine's ACTUAL thrust, and velocity / wind arrows.

Hydrazine exhaust is nearly invisible in reality; the plumes are a visual aid only.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

PLUME_MAX_LEN_M = 3.5
ARROW_SCALE_S = 0.6          # arrow length = speed x this (m per m/s), clipped
ARROW_MAX_M = 8.0


def _frame_with_z(z: np.ndarray) -> np.ndarray:
    """3x3 rotation (columns x, y, z) whose z column is the unit vector z."""
    z = z / (np.linalg.norm(z) or 1.0)
    helper = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(helper, z)
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def add_geom(scene: mujoco.MjvScene, gtype, size, pos, mat, rgba) -> bool:
    """Append one decorative geom to a scene; returns False if the scene is full."""
    if scene.ngeom >= scene.maxgeom:
        return False
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, int(gtype), np.asarray(size, dtype=float), np.asarray(pos, dtype=float),
                        np.asarray(mat, dtype=float).reshape(9), np.asarray(rgba, dtype=np.float32))
    g.category = mujoco.mjtCatBit.mjCAT_DECOR
    scene.ngeom += 1
    return True


def add_plumes(scene: mujoco.MjvScene, sim) -> None:
    """Three nested translucent ellipsoids per engine; length and width grow with actual thrust."""
    tmax = sim.cfg.engine.thrust_max_n
    for k, sid in enumerate(sim.idx.engine_sites):
        frac = float(sim._last_thrust[k]) / tmax
        if frac < 0.01:
            continue
        pos = sim.data.site_xpos[sid]
        mat = sim.data.site_xmat[sid].reshape(3, 3)
        exhaust = mat[:, 2]
        length = PLUME_MAX_LEN_M * frac ** 0.7
        for part, radius, rgba in ((0.45, 0.07, (1.0, 0.97, 0.88, 0.60)),
                                   (0.75, 0.13, (1.0, 0.68, 0.28, 0.32)),
                                   (1.00, 0.20, (1.0, 0.48, 0.18, 0.16))):
            half = 0.5 * length * part
            w = radius * (0.6 + frac)
            add_geom(scene, mujoco.mjtGeom.mjGEOM_ELLIPSOID, [w, w, half], pos + exhaust * half, mat, rgba)


def add_arrow(scene: mujoco.MjvScene, start: np.ndarray, vec: np.ndarray, rgba, width: float = 0.06) -> None:
    length = float(np.linalg.norm(vec))
    if length < 1e-3:
        return
    shown = min(ARROW_MAX_M, length * ARROW_SCALE_S)
    add_geom(scene, mujoco.mjtGeom.mjGEOM_ARROW, [width, width, shown], start, _frame_with_z(vec / length), rgba)


def decorate(scene: mujoco.MjvScene, sim, plumes: bool = True, arrows: bool = True) -> None:
    """Add every overlay for the current simulator state."""
    if plumes:
        add_plumes(scene, sim)
    if arrows:
        tl = sim.telemetry()
        top = tl.cg + np.array([0.0, 0.0, 1.6])
        add_arrow(scene, tl.cg, tl.vel, (0.2, 0.95, 0.35, 0.9))          # velocity (green)
        add_arrow(scene, top, tl.wind, (0.35, 0.65, 1.0, 0.9), 0.04)     # wind (blue)


def tracking_camera(sim, distance: float = 10.0, azimuth: float = 135.0, elevation: float = -12.0) -> mujoco.MjvCamera:
    """A camera that follows the lander (world-aligned, so wind/drift directions read correctly)."""
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = sim.idx.lander_body
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    return cam


def named_camera(sim, name: str) -> mujoco.MjvCamera:
    """'chase' (tracking), or a fixed MJCF camera: 'ground', 'padcam'."""
    if name == "chase":
        return tracking_camera(sim)
    cam_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
    if cam_id < 0:
        raise ValueError(f"unknown camera '{name}' (chase, ground, padcam)")
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    cam.fixedcamid = cam_id
    return cam


def wind_heading_deg(wind: np.ndarray) -> float:
    """Direction the wind blows toward, degrees from +X toward +Y."""
    return math.degrees(math.atan2(wind[1], wind[0])) % 360.0
