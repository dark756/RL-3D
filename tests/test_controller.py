"""The classical controller (M3): thrust mixing, the descent plan, the engine cut, and landings it must make."""

from __future__ import annotations

import math

import numpy as np
import pytest

from controllers.classical import ClassicalController
from envs.mars_lander_env import OBS_NAMES, MarsLanderEnv


@pytest.fixture
def ctl(cfg):
    return ClassicalController(cfg)


def _fly(env, ctl, **reset_kwargs):
    """One episode with the controller; returns (outcome, touchdown report, telemetry at the end)."""
    obs, _ = env.reset(**reset_kwargs)
    ctl.reset()
    for _ in range(10_000):
        obs, _, terminated, truncated, info = env.step(ctl(obs))
        if terminated or truncated:
            return info["outcome"], env.last_result.report, env.last_result.telemetry
    raise AssertionError("episode did not end")


def test_mixer_gives_the_thrust_and_torques_asked_for(ctl, make_sim):
    sim = make_sim()
    arm = sim.geom.engine_pos - sim.geom.cg
    weight = sim.vehicle.mass.total(sim.vehicle.propellant) * sim.cfg.mars.gravity_mps2
    level = ctl.mix(weight, 0.0, 0.0)
    assert np.allclose(level, sim.vehicle.engines.valve_to_thrust(sim.hover_trim()), rtol=1e-9)   # = the sim's trim
    t = ctl.mix(weight, 150.0, -80.0)                       # the vehicle's torque model gives back what was asked
    assert t.sum() == pytest.approx(weight) and t @ arm[:, 1] == pytest.approx(150.0) and -(t @ arm[:, 0]) == pytest.approx(-80.0)
    t = ctl.mix(9000.0, 400.0, 0.0)                         # more than the engines can give: the total gives way
    assert ctl.t_min - 1e-9 <= t.min() and t.max() <= ctl.t_max + 1e-9
    assert t @ arm[:, 1] == pytest.approx(400.0) and t.sum() < 9000.0


def test_descent_plan(ctl, cfg):
    sq, c = cfg.seq, cfg.ctrl
    assert ctl.contour(0.0) == ctl.contour(sq.terminal_alt_m) == pytest.approx(sq.terminal_speed_mps)   # the gate
    assert ctl.contour(100.0) == pytest.approx(math.sqrt(sq.terminal_speed_mps ** 2 + 2 * c.brake_mps2 * (100 - sq.terminal_alt_m)))
    assert ctl.side_time((0.0, 0.0), (0.0, 0.0)) == pytest.approx(c.side_settle_s)          # nothing left to do
    assert ctl.side_time((50.0, 0.0), (0.0, 0.0)) > ctl.side_time((10.0, 0.0), (0.0, 0.0))
    assert ctl.side_time((50.0, 0.0), (10.0, 0.0)) > ctl.side_time((50.0, 0.0), (-10.0, 0.0))  # moving away costs time
    full = cfg.mass.dry_kg + cfg.mass.start_propellant_kg
    assert 20.0 < ctl.fuel_to_land(full, 500.0, 30.0) < cfg.mass.start_propellant_kg       # the hardest start fits
    assert ctl.fuel_to_land(full, 200.0, 10.0) < ctl.fuel_to_land(full, 500.0, 30.0)


def test_engines_off_once_a_leg_switch_trips(ctl):
    obs = np.zeros(len(OBS_NAMES), dtype=np.float32)
    obs[OBS_NAMES.index("height")], obs[OBS_NAMES.index("propellant_kg")] = 1.0, 30.0
    assert ctl(obs)[3] < 0.0
    obs[OBS_NAMES.index("leg_2")] = 1.0
    assert ctl(obs)[3] > 0.0 and ctl.state.phase == "down"


def test_lands_calm_and_in_a_storm(cfg):
    env, ctl = MarsLanderEnv(config=cfg), ClassicalController(cfg)
    for seed, storm in ((1_000_000, False), (2_000_000, True)):          # the first episode of each test set
        o, r, _ = _fly(env, ctl, seed=seed, options={"storm": storm})
        assert o["success"] and o["distance_m"] < cfg.episode.zones_m[0]
        assert r.vz_at_contact < 3.0 and r.vh_at_contact < 0.5 and r.tilt_at_contact_deg < 3.0 and r.rate_at_contact_dps < 5.0


def test_hardest_start_lands_with_propellant_left(cfg):
    """500 m up, 30 m/s down, 50 m off and moving 10 m/s away, in a 30 m/s storm blowing away: 60 kg is enough."""
    env = MarsLanderEnv(config=cfg)
    st = env.sim.state_above_ground(500.0, velocity=(10.0, 0.0, -30.0), xy=(50.0, 0.0))
    start = {"position": list(st.position), "velocity": list(st.velocity), "quat": list(st.quat), "omega": [0.0] * 3,
             "wind_speed_mps": 30.0, "wind_direction_deg": 0.0, "slope_deg": 0.0, "friction": 0.6}
    o, _, tl = _fly(env, ClassicalController(cfg), seed=0, options=start)
    assert o["success"] and tl.propellant_kg > 2.0
