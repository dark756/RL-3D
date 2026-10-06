"""Drag and the storm push, and the sensor models (ideal and with Viking's radar errors)."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import CLEAN
from lander.aero import drag_areas, drag_force_body, wind_vector
from lander.vehicle import Command


def test_drag_calm_versus_storm(cfg):
    cda, rho = drag_areas(cfg), cfg.mars.air_density_kgpm3
    # calm air, Viking's 2.44 m/s final descent: a tenth of a newton
    assert np.abs(drag_force_body(np.array([0.0, 0.0, 2.44]), rho, cda)).max() < 0.2
    # dust storm 17-30 m/s (S11) on a hovering lander: 5-15 N sideways, under 1 % of its weight
    low, high = cfg.wind.storm_mps
    f_low = drag_force_body(np.array([-low, 0.0, 0.0]), rho, cda)[0]
    f_high = drag_force_body(np.array([-high, 0.0, 0.0]), rho, cda)[0]
    assert 4.0 < f_low < f_high < 15.0
    assert drag_force_body(np.array([10.0, 0.0, 0.0]), 0.0, cda) == pytest.approx(np.zeros(3))
    assert wind_vector(20.0, 90.0) == pytest.approx([0.0, 20.0, 0.0], abs=1e-12)


def test_storm_drift_matches_the_review(make_sim):
    """30 s of open-loop hover: a 25 m/s storm adds ~0.48 m/s of sideways speed (0.22-0.69 m/s for 17-30)."""
    def drift(wind):
        sim = make_sim({"VIKING_ENGINE_MISALIGN_DEG": 0.0, "SIM_SENSOR_NOISE_SCALE": 0.0})
        res = sim.reset(initial_state=sim.state_above_ground(300.0), options={"wind_speed_mps": wind})
        for _ in range(750):
            res = sim.step(Command(sim.hover_trim()))
        return res.telemetry.vel[:2]
    extra = np.linalg.norm(drift(25.0) - drift(0.0))
    assert 0.4 < extra < 0.6


def test_ideal_radar_altimeter_and_cutoff(make_sim):
    sim = make_sim(CLEAN)
    res = sim.reset(initial_state=sim.state_above_ground(500.0))
    ant = sim.data.qpos[0:3] + sim.geom.radar_pos            # upright, so body = world orientation
    assert res.sensors.altitude == pytest.approx(ant[2], abs=1e-9)
    res = sim.reset(initial_state=sim.state_above_ground(20.0))
    assert not res.sensors.altitude_valid                    # below 41 m: ignored (S3 p103)


def test_ideal_doppler_radar(make_sim):
    sim = make_sim(CLEAN)
    v = np.array([3.0, -2.0, -12.0])
    res = sim.reset(initial_state=sim.state_above_ground(300.0, velocity=tuple(v), tilt_deg=8.0,
                                                         tilt_azimuth_deg=40.0, heading_deg=25.0))
    rot = sim.data.xmat[sim.idx.lander_body].reshape(3, 3)
    assert res.sensors.doppler_valid.all()
    assert res.sensors.doppler_vel == pytest.approx(rot.T @ res.telemetry.vel, abs=1e-9)   # no rotation
    # hovering: closing speed < 1 m/s on every beam -> no lock (S3 p112)
    res = sim.reset(initial_state=sim.state_above_ground(300.0))
    assert not res.sensors.doppler_valid.any() and np.isnan(res.sensors.doppler_vel).all()


def test_ideal_imu(make_sim):
    sim = make_sim(CLEAN)
    st = sim.state_above_ground(300.0, omega=(0.02, -0.01, 0.03))
    sim.reset(initial_state=st, options={"engines_enabled": False})
    res = sim.step(Command())
    assert res.sensors.gyro == pytest.approx(sim.data.qvel[3:6], abs=1e-12)
    assert np.linalg.norm(res.sensors.accel) < 0.01          # free fall: no specific force


def test_radar_noise_uses_published_levels(make_sim, cfg):
    sim = make_sim()                                         # .env default: noise on (scale 1)
    assert cfg.sim.sensor_noise_scale == 1.0
    rc = cfg.radar
    alt_err, dop_err = [], []
    v = np.array([0.0, 0.0, -20.0])
    for seed in range(40):
        res = sim.reset(seed=seed, initial_state=sim.state_above_ground(1000.0, velocity=tuple(v)))
        ant = sim.data.qpos[0:3] + sim.geom.radar_pos
        alt_err.append(res.sensors.altitude - ant[2])
        dop_err.append(res.sensors.doppler_beam - sim.geom.doppler_beams @ v)
    sigma_alt = (rc.alt_err_frac * ant[2] + rc.alt_err_m) / 3.0
    closing = float(sim.geom.doppler_beams[0] @ v)
    sigma_dop = max(rc.doppler_err_frac * closing, rc.doppler_err_mps) / 3.0
    alt_err, dop_err = np.array(alt_err), np.array(dop_err).ravel()
    assert 0.6 * sigma_alt < alt_err.std() < 1.4 * sigma_alt
    assert 0.7 * sigma_dop < dop_err.std() < 1.3 * sigma_dop
