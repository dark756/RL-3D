"""Onboard navigation: radar readings set the estimate; the ideal IMU carries it between readings and below 41 m."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import CLEAN
from lander.vehicle import Command


def test_perfect_sensors_give_the_truth(make_sim):
    sim = make_sim(CLEAN)
    res = sim.reset(initial_state=sim.state_above_ground(300.0, velocity=(2.0, -1.0, -15.0), heading_deg=30.0))
    for _ in range(100):
        res = sim.step(Command(sim.hover_trim()))
        nav, tl = res.nav, res.telemetry
        assert nav.altimeter_ok and nav.doppler_ok
        assert nav.position == pytest.approx(tl.cg, abs=1e-6)
        assert nav.velocity == pytest.approx(tl.vel, abs=1e-6)
        assert nav.height == pytest.approx(tl.pad_altitude, abs=1e-6)     # level over flat ground


def test_below_41_m_the_imu_carries_the_height(make_sim):
    sim = make_sim(CLEAN)
    res = sim.reset(initial_state=sim.state_above_ground(45.0, velocity=(0.0, 0.0, -3.0)))
    assert res.nav.altimeter_ok
    while res.telemetry.pad_altitude > 5.0:
        res = sim.step(Command(sim.hover_trim()))
    assert not res.nav.altimeter_ok                                         # no readings below 41 m (S3 p103)
    assert res.nav.height == pytest.approx(res.telemetry.pad_altitude, abs=1e-6)   # exact on flat ground


def test_a_slope_misleads_the_carried_height(make_sim):
    """Moving downhill below 41 m, the ground drops away unseen: the lander thinks it is lower than it is."""
    sim = make_sim(CLEAN)
    slope = {"slope_deg": 10.0, "slope_direction_deg": 0.0}
    res = sim.reset(initial_state=sim.state_above_ground(50.0, velocity=(3.0, 0.0, -3.0), **slope), options=slope)
    x_blind = None
    while res.telemetry.pad_altitude > 5.0:
        res = sim.step(Command(sim.hover_trim()))
        if x_blind is None and not res.nav.altimeter_ok:
            x_blind = res.telemetry.cg[0]
    error = res.nav.height - res.telemetry.pad_altitude
    drop = (res.telemetry.cg[0] - x_blind) * np.tan(np.radians(10.0))      # ground fall since the last reading
    assert error < 0 and abs(error) == pytest.approx(drop, rel=0.15)


def test_radar_errors_reach_the_estimate(make_sim, cfg):
    """With Viking's errors on (the default), each reading's error goes straight into the estimate."""
    sim = make_sim()
    rc = cfg.radar
    err_h, err_v = [], []
    for seed in range(40):
        res = sim.reset(seed=seed, initial_state=sim.state_above_ground(300.0, velocity=(0.0, 0.0, -20.0)))
        err_h.append(res.nav.height - res.telemetry.pad_altitude)
        err_v.append(res.nav.velocity - res.telemetry.vel)
    err_v = np.array(err_v)
    sigma_h = (rc.alt_err_frac * (300.0 + 0.5265) + rc.alt_err_m) / 3.0
    # four beams 21.5 deg from vertical, each 0.1 m/s (1 sigma) at 18.6 m/s closing: least squares gives
    # 0.1 / (2 cos 21.5) = 0.054 m/s vertically and 0.1 / (sqrt 2 sin 21.5) = 0.19 m/s per horizontal axis
    beam = max(rc.doppler_err_frac * 20.0 * np.cos(np.radians(21.5)), rc.doppler_err_mps) / 3.0
    sigma_vz = beam / (2.0 * np.cos(np.radians(rc.doppler_splay_deg)))
    sigma_vh = beam / (np.sqrt(2.0) * np.sin(np.radians(rc.doppler_splay_deg)))
    assert 0.6 * sigma_h < np.std(err_h) < 1.4 * sigma_h
    assert 0.6 * sigma_vz < np.std(err_v[:, 2]) < 1.4 * sigma_vz
    assert 0.6 * sigma_vh < np.std(err_v[:, :2]) < 1.4 * sigma_vh
    print(f"1-sigma errors: height {np.std(err_h):.2f} m (expected {sigma_h:.2f}), vertical speed "
          f"{np.std(err_v[:, 2]):.3f} m/s ({sigma_vz:.3f}), horizontal speed {np.std(err_v[:, :2]):.3f} m/s ({sigma_vh:.3f})")
