"""Mars air and wind (numpy only, D1): constant density, one steady wind per episode, quadratic drag.

MuJoCo's own fluid model is switched off; these forces are applied by `sim.py` each physics step.
Density is the surface value at every height (it changes < 5 % over the 500 m an episode covers).
Calm episodes have no wind: 2-10 m/s moves the lander < 0.08 m/s in 30 s. Storms (17-30 m/s, S11)
push with 5-15 N. Gusts are left out: they add < 0.1 m/s of drift even in a storm.
"""

from __future__ import annotations

import math

import numpy as np

from .config import Config


def drag_force_body(v_rel_body: np.ndarray, rho: float, cda: np.ndarray) -> np.ndarray:
    """Quadratic drag in the body frame: F_k = -1/2 rho |v| (Cd A)_k v_k (axes x, y lateral; z axial)."""
    speed = math.sqrt(float(v_rel_body @ v_rel_body))
    return -0.5 * rho * speed * cda * v_rel_body


def drag_areas(cfg: Config) -> np.ndarray:
    """(Cd*A) per body axis: lateral, lateral, axial (m^2)."""
    a = cfg.aero
    return np.array([a.cda_lateral_m2, a.cda_lateral_m2, a.cda_axial_m2])


def wind_vector(speed_mps: float, direction_deg: float) -> np.ndarray:
    """Steady horizontal wind (world frame, m/s) blowing TOWARD `direction_deg` (from +X toward +Y)."""
    a = math.radians(direction_deg)
    return float(speed_mps) * np.array([math.cos(a), math.sin(a), 0.0])
