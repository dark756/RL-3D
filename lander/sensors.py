"""Sensor models (numpy only, D1): ideal IMU at the CG, radar altimeter, four-beam Doppler radar.

The simulator supplies the truth (specific force, body rates, pose, velocity) and a `terrain`
object with `nearest(point) -> (distance, unit vector toward the surface)` and
`raycast(origin, direction) -> distance or inf`.

Both radar antennas sit at the centre of the underside and are sample-and-hold at
VIKING_RADAR_PERIOD_S (S3 p93). The altimeter measures the range to the nearest ground point
(first return) and is ignored below 41 m (S3 p103). Each Doppler beam needs >= 1 m/s closing speed to
lock; any 3 locked beams give the velocity (least squares). Radar errors are Viking's published
accuracies, treated as 3 sigma, times SIM_SENSOR_NOISE_SCALE (1 = on, 0 = perfect readings).
Invalid readings are NaN. The IMU is ideal: Viking's IMU errors (S3 Table 3) were 10^4-10^6 times
smaller than the signals over a landing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import Config
from .geometry import LanderGeometry, cross3


@dataclass
class SensorReadings:
    t: float
    accel: np.ndarray          # specific force at the CG, body frame (m/s^2)
    gyro: np.ndarray           # body angular rates (rad/s)
    altitude: float            # radar altimeter: range to the nearest ground point (m); NaN = invalid
    doppler_beam: np.ndarray   # (4,) closing speed along each beam (m/s); NaN = no lock
    doppler_vel: np.ndarray    # (3,) body-frame velocity from the locked beams; NaN if < 3 beams

    @property
    def altitude_valid(self) -> bool:
        return math.isfinite(self.altitude)

    @property
    def doppler_valid(self) -> np.ndarray:
        return np.isfinite(self.doppler_beam)


class SensorSuite:
    def __init__(self, cfg: Config, geom: LanderGeometry):
        self.cfg, self.geom = cfg, geom
        self.beams = geom.doppler_beams
        self.noise_scale = cfg.sim.sensor_noise_scale
        self.reset(np.random.default_rng(0))

    def reset(self, rng: np.random.Generator, noise_scale: float | None = None) -> None:
        self.rng = rng
        self.noise_scale = self.cfg.sim.sensor_noise_scale if noise_scale is None else float(noise_scale)
        self._next = -math.inf
        self._altitude = math.nan
        self._beam = np.full(len(self.beams), np.nan)
        self._vel = np.full(3, np.nan)

    def sample(self, t: float, f_body: np.ndarray, omega_body: np.ndarray, origin: np.ndarray,
               rot: np.ndarray, v_origin: np.ndarray, omega_world: np.ndarray, terrain) -> SensorReadings:
        """Read all sensors at time t. `origin`, `rot`, `v_origin` describe the lander frame in the world."""
        if t >= self._next - 1e-9:
            self._next = t + self.cfg.radar.period_s
            p_ant = origin + rot @ self.geom.radar_pos
            v_ant = v_origin + cross3(omega_world, rot @ self.geom.radar_pos)
            self._altitude = self._read_altimeter(p_ant, rot, terrain)
            self._beam, self._vel = self._read_doppler(p_ant, v_ant, rot, terrain)
        return SensorReadings(t, f_body.copy(), omega_body.copy(), self._altitude, self._beam.copy(), self._vel.copy())

    def _read_altimeter(self, p_ant: np.ndarray, rot: np.ndarray, terrain) -> float:
        rc = self.cfg.radar
        dist, toward = terrain.nearest(p_ant)
        # first return; the antenna faces down, so the ground must be below the underside
        if dist < rc.alt_min_range_m or float(toward @ (-rot[:, 2])) <= 0.0:
            return math.nan
        sigma = self.noise_scale * (rc.alt_err_frac * dist + rc.alt_err_m) / 3.0
        return dist + (float(self.rng.normal(0.0, sigma)) if sigma > 0 else 0.0)

    def _read_doppler(self, p_ant: np.ndarray, v_ant: np.ndarray, rot: np.ndarray, terrain):
        rc = self.cfg.radar
        beams = np.full(len(self.beams), np.nan)
        for k, d_b in enumerate(self.beams):
            d_w = rot @ d_b
            closing = float(v_ant @ d_w)
            if not math.isfinite(terrain.raycast(p_ant, d_w)) or closing < rc.doppler_min_speed_mps:
                continue
            sigma = self.noise_scale * max(rc.doppler_err_frac * abs(closing), rc.doppler_err_mps) / 3.0
            beams[k] = closing + (float(self.rng.normal(0.0, sigma)) if sigma > 0 else 0.0)
        valid = np.isfinite(beams)
        vel = np.full(3, np.nan)
        if valid.sum() >= 3:
            vel = np.linalg.lstsq(self.beams[valid], beams[valid], rcond=None)[0]
        return beams, vel
