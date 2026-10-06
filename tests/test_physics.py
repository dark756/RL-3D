"""Brief section 11: free fall, hover, mass flow, determinism, the engines-off command, misalignment,
plus rest stability on the ground (dt check)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from conftest import CLEAN
from lander.config import G0_MPS2
from lander.vehicle import Command


def test_free_fall_matches_closed_form(make_sim, cfg):
    sim = make_sim(CLEAN)
    v0 = -5.0
    st = sim.state_above_ground(200.0, velocity=(0.0, 0.0, v0))
    res = sim.reset(initial_state=st, options={"engines_enabled": False})
    z0, g, dt = res.telemetry.cg[2], cfg.mars.gravity_mps2, cfg.sim.physics_dt_s
    for _ in range(75):                                   # 3 s
        res = sim.step(Command())
    t = res.telemetry.t
    assert t == pytest.approx(3.0)
    assert res.telemetry.vel[2] == pytest.approx(v0 - g * t, abs=1e-9)
    # semi-implicit Euler is exact in velocity and first order in position: error = g t dt / 2
    assert res.telemetry.cg[2] == pytest.approx(z0 + v0 * t - 0.5 * g * t * t, abs=0.5 * g * t * dt + 1e-6)
    assert np.abs(res.telemetry.cg[:2] - st.position[:2]).max() < 1e-9


def test_hover_holds_altitude_and_stays_level(make_sim, cfg):
    sim = make_sim(CLEAN)
    res = sim.reset(initial_state=sim.state_above_ground(50.0))
    z0 = res.telemetry.cg.copy()
    for _ in range(250):                                  # 10 s, re-trimmed as the mass drops
        res = sim.step(Command(sim.hover_trim()))
    assert abs(res.telemetry.cg[2] - z0[2]) < 0.05       # sum of thrust = m g holds altitude
    # open loop: with the CG fixed on the centre line and no misalignment nothing tilts the lander
    assert res.telemetry.tilt_deg < 1e-3
    assert np.linalg.norm(res.telemetry.cg[:2] - z0[:2]) < 1e-3
    assert res.telemetry.propellant_kg < cfg.mass.start_propellant_kg - 10.0   # it really burned propellant


def test_misalignment_tilts_an_uncontrolled_lander(make_sim):
    sim = make_sim({"SIM_SENSOR_NOISE_SCALE": 0.0})
    res = sim.reset(seed=0, initial_state=sim.state_above_ground(300.0))
    for _ in range(250):                                  # 10 s with no attitude control
        res = sim.step(Command(sim.hover_trim()))
    assert res.telemetry.tilt_deg > 0.5                   # 0.044 deg misalignment, uncorrected: ~1-3 deg
    assert abs(res.telemetry.spin_dps) > 0.1              # and nothing stops the spin (no roll jets)


def test_mass_flow_equals_thrust_integral(make_sim):
    sim = make_sim()
    sim.reset(initial_state=sim.state_above_ground(300.0))
    isp_g0 = sim.vehicle.engines.isp_g0
    used = {"kg": 0.0}

    def tally(s, k):   # after each physics step: the actual thrust of that step
        used["kg"] += s._last_thrust.sum() / isp_g0 * s.dt

    m0 = sim.vehicle.propellant
    for i in range(125):                                  # 5 s of varying throttle
        u = 0.3 + 0.25 * math.sin(0.2 * i)
        sim.step(Command((u, u + 0.1, u - 0.1)), substep_callback=tally)
    assert m0 - sim.vehicle.propellant == pytest.approx(used["kg"], rel=1e-12)
    assert used["kg"] > 1.0
    assert isp_g0 == pytest.approx(195 * G0_MPS2)


def test_engines_off_command_is_permanent(make_sim, cfg):
    sim = make_sim(CLEAN)
    sim.reset(initial_state=sim.state_above_ground(100.0))
    log = []
    cb = lambda s, k: log.append((s.t, float(s._last_thrust.sum())))
    for _ in range(25):
        sim.step(Command(sim.hover_trim()), substep_callback=cb)
    t_cut, lit = sim.t, log[-1][1]
    res = sim.step(Command(sim.hover_trim(), engines_off=True), substep_callback=cb)
    assert "engines_cut" in res.events and res.telemetry.engines_off
    for _ in range(10):                                   # full throttle afterwards: no relight
        res = sim.step(Command((1.0, 1.0, 1.0)), substep_callback=cb)
    after = [(t - t_cut, th / lit) for t, th in log if t > t_cut]
    first_10pct = next(dt for dt, f in after if f <= 0.1)
    assert first_10pct == pytest.approx(cfg.engine.t90_s, abs=2 * cfg.sim.physics_dt_s)   # first-order lag
    assert all(b[1] <= a[1] for a, b in zip(after, after[1:]))                           # only ever decays
    assert res.telemetry.thrust_n.sum() < 0.01 and res.telemetry.engines_off       # exponential tail-off


def _run(sim, seed, steps=60):
    rng = np.random.default_rng(123)
    sim.reset(seed=seed, initial_state=sim.state_above_ground(60.0, velocity=(1.0, -0.5, -3.0)))
    states = []
    for _ in range(steps):
        res = sim.step(Command(tuple(rng.uniform(0.1, 0.5, 3))))
        states.append(np.concatenate([sim.data.qpos, sim.data.qvel, res.sensors.accel, [res.sensors.altitude]]))
    return np.array(states)


def test_determinism_same_seed_bit_identical(make_sim):
    a, b = _run(make_sim(), seed=7), _run(make_sim(), seed=7)
    assert np.array_equal(a, b, equal_nan=True)
    c = _run(make_sim(), seed=8)
    assert not np.array_equal(a, c, equal_nan=True)       # different seed -> different misalignment and noise


def test_rest_on_flat_ground_is_stable(make_sim):
    sim = make_sim(CLEAN)
    sim.reset(initial_state=sim.state_above_ground(0.002, propellant_kg=2.0), options={"engines_enabled": False})
    positions = []
    for _ in range(100):                                  # 4 s
        res = sim.step(Command())
        positions.append(sim.data.qpos[0:3].copy())
    tl = res.telemetry
    assert np.linalg.norm(tl.vel) < 1e-3 and np.linalg.norm(tl.omega_b) < 1e-3
    assert not tl.body_contact and all(tl.pad_contact)
    assert np.abs(positions[-1] - positions[-25]).max() < 1e-4      # no creep over the last second
    pads_z = sim.data.geom_xpos[list(sim.idx.pad_geoms), 2] - sim.geom.footpad_radius
    assert pads_z.min() > -0.005                                    # contact penetration < 5 mm
