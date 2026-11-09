"""Mars air and wind (numpy only, D1): constant density, one steady wind per episode, quadratic drag.

MuJoCo's own fluid model is switched off; these forces are applied by `sim.py` each physics step.
Density is the surface value at every height (it changes < 5 % over the 500 m an episode covers).
Calm episodes have no wind: 2-10 m/s moves the lander < 0.08 m/s in 30 s. Storms (17-30 m/s, S11)
push with 5-15 N. Gusts are left out: they add < 0.1 m/s of drift even in a storm.
One Cd*A for every direction (VIKING_AERO_CDA_M2, calibrated on sideways flow): since 2026-10-10 the
descent drag uses it too, which adds ~1.5 N at 30 m/s.
"""

from __future__ import annotations

import math

import numpy as np


def drag_force(v_rel, rho: float, cda: float) -> tuple[float, float, float]:
    """Quadratic drag F = -1/2 rho CdA |v| v, in the frame of `v_rel` (air-relative velocity, any 3 numbers)."""
    x, y, z = v_rel
    c = -0.5 * rho * cda * math.sqrt(x * x + y * y + z * z)
    return (c * x, c * y, c * z)


def wind_vector(speed_mps: float, direction_deg: float) -> np.ndarray:
    """Steady horizontal wind (world frame, m/s) blowing TOWARD `direction_deg` (from +X toward +Y)."""
    a = math.radians(direction_deg)
    return float(speed_mps) * np.array([math.cos(a), math.sin(a), 0.0])
