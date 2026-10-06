"""Lander geometry derived from the config (backend-agnostic, numpy only).

Frame (TN D-7959, S4 Fig 3): origin on the centre line at the footpad contact plane, +Z up the
thrust axis, leg 1 on +Y, azimuths measured from +X toward +Y.

The vehicle is simplified (brief D15): the legs are rigid, so every position is fixed. Three
footpads; three engines on the body underside, thrusting along +Z; the CG on the centre line; the
IMU at the CG; both radar antennas at the centre of the underside.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import Config

BODY_BOTTOM_Z_M = 0.5265   # S5 Fig 8: body underside above the footpad plane (20.73 in); engines and antennas sit here
DOPPLER_BEAMS = 4          # S3 p111: four-beam Doppler landing radar (any 3 beams give the velocity)


def unit_azimuth(az_rad: float) -> np.ndarray:
    """Horizontal unit vector at azimuth az (from +X toward +Y)."""
    return np.array([math.cos(az_rad), math.sin(az_rad), 0.0])


def cross3(a, b) -> np.ndarray:
    """Cross product of two 3-vectors (much faster than np.cross for single vectors)."""
    a0, a1, a2 = float(a[0]), float(a[1]), float(a[2])
    b0, b1, b2 = float(b[0]), float(b[1]), float(b[2])
    return np.array([a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0])


@dataclass(frozen=True)
class LanderGeometry:
    """Everything positional about the lander, in the lander frame."""

    footpads: np.ndarray          # (3, 3) footpad contact-sphere centres
    footpad_radius: float         # contact sphere radius = footpad junction height
    leg_azimuth_rad: tuple        # (3,) leg directions
    cg: np.ndarray                # (3,) centre of gravity, on the centre line (D15)
    engine_pos: np.ndarray        # (3, 3) thrust application points (body underside)
    radar_pos: np.ndarray         # (3,) altimeter and Doppler antennas: centre of the underside
    doppler_beams: np.ndarray     # (4, 3) unit vectors, body frame, pointing at the ground


def hexagon_vertices(long_side: float, short_side: float, leg_azimuths_rad: list[float]) -> np.ndarray:
    """Body outline: an equilateral triangle of side L+2S with its corners cut, short sides facing legs."""
    tri = long_side + 2.0 * short_side
    apothem_short = tri / math.sqrt(3.0) - short_side * math.sqrt(3.0) / 2.0
    verts = []
    for az in sorted(leg_azimuths_rad):
        r, t = unit_azimuth(az), unit_azimuth(az + math.pi / 2)
        centre = apothem_short * r
        verts.append((centre - 0.5 * short_side * t)[:2])
        verts.append((centre + 0.5 * short_side * t)[:2])
    return np.array(verts)


def build_geometry(cfg: Config) -> LanderGeometry:
    lc, ec = cfg.leg, cfg.engine
    az = tuple(math.radians(a) for a in lc.azimuth_deg)
    footpads = np.array([lc.footpad_radius_m * unit_azimuth(a) + np.array([0.0, 0.0, lc.footpad_z_m]) for a in az])
    engine_pos = np.array([[x, y, BODY_BOTTOM_Z_M] for x, y in zip(ec.x_m, ec.y_m)])

    rc = cfg.radar
    sp, ck = math.radians(rc.doppler_splay_deg), math.radians(rc.doppler_clock_deg)
    beams = []
    for k in range(DOPPLER_BEAMS):
        phi = ck + k * 2.0 * math.pi / DOPPLER_BEAMS
        beams.append(np.array([math.sin(sp) * math.cos(phi), math.sin(sp) * math.sin(phi), -math.cos(sp)]))

    return LanderGeometry(
        footpads=footpads, footpad_radius=lc.footpad_z_m, leg_azimuth_rad=az,
        cg=np.array([0.0, 0.0, cfg.mass.cg_z_m]), engine_pos=engine_pos,
        radar_pos=np.array([0.0, 0.0, BODY_BOTTOM_Z_M]), doppler_beams=np.array(beams))
