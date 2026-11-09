"""Onboard navigation (numpy only, D1): what the lander knows about its own height, velocity and position.

The agent never sees the truth. It sees this estimate, built the way Viking's guidance used its sensors
(S3): the radars are the authority and the IMU bridges the gaps between their readings.

- Velocity: every new Doppler reading (each 0.2 s, when at least 3 beams lock) sets it. Between readings,
  and while the Doppler has no lock (closing speed under 1 m/s, e.g. near hover), the IMU carries it forward.
- Height of the feet above the ground: every new altimeter reading sets it (the range minus the antenna's
  0.53 m above the footpad plane). The range is the first return, from the nearest ground point, so on a
  slope it is the perpendicular distance: cos(slope) times the vertical height, up to 5 % short at 19 deg.
  Below 41 m there are no readings, so the estimated vertical velocity carries it forward as if the ground
  were flat.
- Horizontal position relative to the target: known at the start (handed over by the earlier phases of
  the landing), then carried forward with the estimated velocity.

The IMU is ideal (D15), so carrying forward adds no error of its own: it moves the estimate by exactly the
lander's true change in velocity and position. What remains is the error of the last radar reading, the
velocity error integrated since then, and, on a slope, the ground rising or falling under a lander that
moves sideways, which flat-ground carrying-forward misses. A real filter would also average readings; this
one trusts each new reading, since Viking's published accuracies are per reading.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .geometry import LanderGeometry, cross3


@dataclass
class NavEstimate:
    """The onboard estimate at one control step."""

    t: float
    position: np.ndarray       # CG relative to the target (the world origin), world frame (m)
    velocity: np.ndarray       # CG velocity, world frame (m/s)
    height: float              # feet above the ground below them (m), altimeter-based
    altimeter_ok: bool         # the latest altimeter reading was valid
    doppler_ok: bool           # the latest Doppler reading gave a velocity


class Navigator:
    """Turns sensor readings into a `NavEstimate`; call `reset` once per episode, then `update` each control step."""

    def __init__(self, geom: LanderGeometry):
        self.arm_b = geom.radar_pos - geom.cg            # CG to the radar antennas, body frame
        self.antenna_height = float(geom.radar_pos[2])   # antennas above the footpad plane

    def reset(self, t: float, cg: np.ndarray, vel: np.ndarray, height: float, sensors, rot: np.ndarray,
              omega_b: np.ndarray) -> NavEstimate:
        """Start from the handed-over state (the truth at t), then apply the radar readings taken at t."""
        self.position, self.velocity, self.height = cg.copy(), vel.copy(), float(height)
        self._t, self._cg, self._vel = t, cg.copy(), vel.copy()
        self._fix_t = -math.inf
        return self._apply_readings(t, sensors, rot, omega_b)

    def update(self, t: float, cg: np.ndarray, vel: np.ndarray, sensors, rot: np.ndarray,
               omega_b: np.ndarray) -> NavEstimate:
        """Carry the estimate forward to t with the ideal IMU, then apply any new radar readings.

        `cg` and `vel` are the truth: the ideal IMU measures exactly how much they changed since the last
        update, so the estimate moves by that change and keeps its own error.
        """
        dt = t - self._t
        err_v = self.velocity - self._vel                # velocity error, constant between Doppler readings
        dp = cg - self._cg
        self.position = self.position + dp + err_v * dt
        self.height += float(dp[2] + err_v[2] * dt)      # flat-ground assumption
        self.velocity = vel + err_v
        self._t, self._cg, self._vel = t, cg.copy(), vel.copy()
        return self._apply_readings(t, sensors, rot, omega_b)

    def _apply_readings(self, t: float, sn, rot: np.ndarray, omega_b: np.ndarray) -> NavEstimate:
        doppler_ok = bool(np.isfinite(sn.doppler_vel).all())
        if sn.radar_t > self._fix_t:                     # a new radar sample (they are held for 0.2 s)
            self._fix_t = sn.radar_t
            if sn.altitude_valid:
                self.height = float(sn.altitude) - self.antenna_height
            if doppler_ok:                               # antenna velocity -> CG velocity
                self.velocity = rot @ sn.doppler_vel - cross3(rot @ omega_b, rot @ self.arm_b)
        return NavEstimate(t, self.position.copy(), self.velocity.copy(), self.height,
                           bool(sn.altitude_valid), doppler_ok)
