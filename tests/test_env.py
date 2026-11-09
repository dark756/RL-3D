"""The Gymnasium environment (M2): API, start distribution, options, actions, how episodes end, the reward."""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from conftest import CLEAN
from envs.mars_lander_env import (FAILURES, IN_FLIGHT, OBS_NAMES, MarsLanderEnv, action_to_command, failure_reward,
                                  landing_reward)
from scripts.run_episodes import GENTLE, descend


@pytest.fixture
def make_env(cfg):
    """make_env(overrides, **kwargs) -> MarsLanderEnv on the .env config (LANDER_* overrides ignored)."""
    def _make(overrides=None, **kwargs):
        return MarsLanderEnv(config=cfg, overrides=overrides, **kwargs)
    return _make


def _run(env, policy, **reset_kwargs):
    """Reset, then step `policy(env, obs)` until the episode ends; returns (last reward, last info, steps)."""
    obs, _ = env.reset(**reset_kwargs)
    for steps in range(1, 10_000):
        obs, reward, terminated, truncated, info = env.step(policy(env, obs))
        if terminated or truncated:
            return reward, info, steps
    raise AssertionError("episode did not end")


def _hover(env, obs):
    sim = env.sim
    valves = np.array(sim.hover_trim())
    return np.concatenate([2.0 * valves - 1.0, [-1.0]])


@pytest.mark.filterwarnings("ignore:.*Box observation space m")    # the bounds are infinite on purpose
def test_gymnasium_checker_and_registration(make_env):
    check_env(make_env(), skip_render_check=True)
    import envs  # noqa: F401  (registers the id)
    env = gym.make("MarsLander-v0")
    assert env.observation_space.shape == (len(OBS_NAMES),) == (22,)
    assert env.action_space.shape == (4,) and env.action_space.low.min() == -1 and env.action_space.high.max() == 1
    env.close()


def test_same_seed_same_episode_and_replay_from_info(make_env):
    a, b = make_env(), make_env()
    oa, ia = a.reset(seed=7)
    ob, ib = b.reset(seed=7)
    assert np.array_equal(oa, ob) and ia["episode"] == ib["episode"]
    actions = np.random.default_rng(0).uniform(-1.0, 1.0, (40, 4))
    actions[:, 3] = -1.0                                  # keep the engines on

    def run(env, **kw):
        obs, _ = env.reset(**kw)
        out = [obs]
        for act in actions:
            obs, _, terminated, _, _ = env.step(act)
            out.append(obs)
            if terminated:
                break
        return np.array(out)

    first = run(a, seed=7)
    assert np.array_equal(first, run(b, seed=7))                       # bit-identical (D10)
    assert np.array_equal(first, run(b, options=ia["episode"]))        # the resolved start replays it
    _, ic = a.reset(seed=8)
    assert ic["episode"]["sim_seed"] != ia["episode"]["sim_seed"]


def test_start_distribution(make_env, cfg):
    env, ep = make_env(), cfg.episode
    storms, slopes, n = 0, [], 200
    for seed in range(n):
        _, info = env.reset(seed=seed)
        e, tl = info["episode"], env.last_result.telemetry
        assert ep.start_height_m[0] - 1e-6 <= tl.pad_altitude <= ep.start_height_m[1] + 1e-6
        assert math.hypot(*e["position"][:2]) <= ep.start_offset_m[1] + 0.5          # CG vs footprint centre
        assert ep.start_descent_mps[0] <= -e["velocity"][2] <= ep.start_descent_mps[1]
        assert math.hypot(*e["velocity"][:2]) <= ep.start_sideways_mps[1] + 1e-9
        assert tl.tilt_deg <= ep.start_tilt_deg[1] + 1e-6
        assert max(abs(w) for w in e["omega"]) <= math.radians(ep.start_rate_dps) + 1e-12
        assert cfg.terrain.friction_range[0] <= e["friction"] <= cfg.terrain.friction_range[1]
        assert e["friction"] >= cfg.terrain.hold_ratio * math.tan(math.radians(e["slope_deg"])) - 1e-12   # holds (D26)
        assert 0.0 <= e["slope_deg"] <= cfg.terrain.slope_max_deg
        assert e["propellant_kg"] == cfg.mass.start_propellant_kg and e["sensor_noise_scale"] == 1.0
        slopes.append(e["slope_deg"])
        if e["storm"]:
            storms += 1
            assert cfg.wind.storm_mps[0] <= e["wind_speed_mps"] <= cfg.wind.storm_mps[1]
        else:
            assert e["wind_speed_mps"] == 0.0
    assert 0.2 < storms / n < 0.4                                       # 30 % storms (D17)
    # exponential, 90 % below 11.7 deg (S4), cut at 19 deg: 0.90 / 0.976 = 92 % below 11.7 deg
    assert 0.85 < np.mean(np.array(slopes) < cfg.terrain.slope_p90_deg) < 0.98


def test_options_set_any_part_of_the_episode(make_env):
    env = make_env()
    _, base = env.reset(seed=3)
    _, info = env.reset(seed=3, options={"wind_speed_mps": 25.0, "wind_direction_deg": 90.0})
    e = info["episode"]
    assert e["storm"] and e["wind_speed_mps"] == 25.0 and np.allclose(env.sim.wind, [0.0, 25.0, 0.0])
    assert e["position"] == base["episode"]["position"]             # the other draws do not move
    _, info = env.reset(seed=3, options={"slope_deg": 15.0})         # a given slope: friction still holds on it
    assert info["episode"]["friction"] >= 1.25 * math.tan(math.radians(15.0))
    state = {"position": [0.0, 0.0, 50.0], "velocity": [0.0, 0.0, -2.0], "quat": [1.0, 0.0, 0.0, 0.0],
             "omega": [0.0, 0.0, 0.0], "propellant_kg": 10.0}
    env.reset(seed=3, options=state)
    tl = env.last_result.telemetry
    assert np.allclose(tl.cg, [0.0, 0.0, 50.0]) and np.allclose(tl.vel, [0.0, 0.0, -2.0]) and tl.propellant_kg == 10.0
    with pytest.raises(ValueError, match="unknown episode options"):
        env.reset(options={"windspeed": 3.0})
    env.reset(seed=1, options={"overrides": {"MARS_GRAVITY_MPS2": 3.0}})    # this episode only: rebuilt
    assert env.sim.cfg.mars.gravity_mps2 == 3.0 and env.sim.model.opt.gravity[2] == pytest.approx(-3.0)
    env.reset(seed=1)
    assert env.sim.cfg.mars.gravity_mps2 == 3.71


def test_action_mapping_and_permanent_cut(make_env):
    c = action_to_command([-1.0, 0.0, 1.0, -0.5])
    assert c.throttle == (0.0, 0.5, 1.0) and not c.engines_off
    assert action_to_command([2.0, -3.0, 0.0, 0.01]).throttle == (1.0, 0.0, 0.5)     # clipped to [-1, 1]
    assert action_to_command([0.0, 0.0, 0.0, 0.01]).engines_off
    with pytest.raises(ValueError):
        action_to_command([np.nan, 0.0, 0.0, 0.0])
    env = make_env(CLEAN)
    env.reset(seed=0)
    obs, *_ = env.step([0.0, 0.0, 0.0, 1.0])                        # engines off ...
    for _ in range(10):
        obs, *_ = env.step([1.0, 1.0, 1.0, -1.0])                   # ... and full throttle asked for later
    assert obs[OBS_NAMES.index("engines_off")] == 1.0 and env.last_result.telemetry.thrust_n.sum() < 1.0


def test_gentle_landing_on_and_off_the_pad(make_env, cfg):
    env = make_env(GENTLE)
    reward, info, _ = _run(env, descend, seed=0)
    o = info["outcome"]
    assert o["result"] == "landed" and o["success"] and info["is_success"] and o["distance_m"] < 5.0
    assert reward == pytest.approx(100.0 - cfg.reward.time_per_s * o["time_s"])   # green zone, no tilt cost
    _, start = env.reset(seed=0)
    pos = [30.0, 0.0, start["episode"]["position"][2]]
    reward, info, _ = _run(env, descend, seed=0, options={"position": pos})
    o = info["outcome"]
    assert o["result"] == "landed_off_pad" and o["safe"] and not o["success"]
    zone = 80.0 - 50.0 * (o["distance_m"] - 20.0) / 180.0
    assert reward == pytest.approx(zone - cfg.reward.time_per_s * o["time_s"])


def test_in_flight_failures(make_env):
    tumble = lambda env, obs: np.array([1.0, -1.0, -1.0, -1.0])
    reward, info, _ = _run(make_env(), tumble, seed=0)
    assert info["outcome"]["reason"] == "lost_attitude" and reward == -100.0
    reward, info, steps = _run(make_env({"EPISODE_TIME_LIMIT_S": 2.0}), _hover, seed=0)
    assert info["outcome"]["reason"] == "time_limit" and reward == -100.0 and steps == 50
    reward, info, _ = _run(make_env(), _hover, seed=0, options={"propellant_kg": 0.3})
    assert info["outcome"]["reason"] == "propellant_exhausted"
    edge = {"position": [199.0, 0.0, 300.0], "velocity": [10.0, 0.0, -10.0], "quat": [1.0, 0.0, 0.0, 0.0]}
    _, info, _ = _run(make_env(), _hover, seed=0, options=edge)
    assert info["outcome"]["reason"] == "left_arena"
    low_ceiling = {"EPISODE_START_HEIGHT_M": "100,110", "EPISODE_CEILING_M": 120}
    climb = lambda env, obs: np.concatenate([2.0 * np.array(env.sim.hover_trim(total_thrust=7700.0)) - 1.0, [-1.0]])
    _, info, _ = _run(make_env(low_ceiling), climb, seed=0)
    assert info["outcome"]["reason"] == "above_ceiling"


def test_touchdown_failures_are_graded_by_impact_speed(make_env, cfg):
    never_cut = lambda env, obs: np.concatenate([descend(env, obs)[:3], [-1.0]])
    reward, info, _ = _run(make_env(GENTLE), never_cut, seed=0)
    o = info["outcome"]
    assert o["reason"] == "engines_on" and reward == pytest.approx(-50.0)      # soft, but the engines stayed on
    cut_now = lambda env, obs: np.array([-1.0, -1.0, -1.0, 1.0])               # free fall from 5-10 m
    reward, info, _ = _run(make_env(GENTLE), cut_now, seed=0)
    o = info["outcome"]
    q = o["parts"]["impact_ratio"]
    assert o["reason"] == "leg_broken" and 1.0 < q < cfg.reward.crash_ratio
    assert reward == pytest.approx(-50.0 - 50.0 * (q - 1.0) / (cfg.reward.crash_ratio - 1.0))


def test_reward_functions(cfg):
    assert landing_reward(cfg, 0.0, 0.0, 0.0)[0] == 100.0
    assert landing_reward(cfg, 5.0, 0.0, 0.0)[0] == pytest.approx(100.0)
    assert landing_reward(cfg, 12.5, 0.0, 0.0)[0] == pytest.approx(90.0)      # halfway across the yellow zone
    assert landing_reward(cfg, 20.0, 0.0, 0.0)[0] == pytest.approx(80.0)
    assert landing_reward(cfg, 200.0, 0.0, 0.0)[0] == pytest.approx(30.0)
    assert landing_reward(cfg, 0.0, 30.0, 0.0)[0] == pytest.approx(94.0)      # 0.2 per second
    assert landing_reward(cfg, 0.0, 0.0, 30.0)[0] == pytest.approx(96.0)      # 0.4 per degree above 20
    ep = cfg.episode
    worst_landing = landing_reward(cfg, ep.zones_m[2], ep.time_limit_s + ep.settle_window_s, ep.tilt_limit_deg)[0]
    assert worst_landing > cfg.reward.failure[1]                               # any safe landing beats any failure
    assert failure_reward(cfg, "lost_attitude") == -100.0
    assert failure_reward(cfg, "leg_broken", 0.8) == -50.0
    assert failure_reward(cfg, "leg_broken", 3.5) == pytest.approx(-75.0)
    assert failure_reward(cfg, "leg_broken", 9.0) == -100.0
    assert set(IN_FLIGHT) <= set(FAILURES)


def test_observation_is_the_onboard_view(make_env):
    env = make_env(CLEAN)                           # perfect radar readings: the estimate equals the truth
    obs, info = env.reset(seed=4)
    o = dict(zip(OBS_NAMES, obs))
    tl = env.last_result.telemetry
    assert math.hypot(o["pos_x"], o["pos_y"]) == pytest.approx(math.hypot(tl.cg[0], tl.cg[1]), rel=1e-5)
    assert math.hypot(o["vel_x"], o["vel_y"]) == pytest.approx(tl.vh, rel=1e-5)
    assert o["vel_z"] == pytest.approx(tl.vz, rel=1e-5)
    assert math.degrees(math.asin(math.hypot(o["tilt_x"], o["tilt_y"]))) == pytest.approx(tl.tilt_deg, abs=1e-3)
    assert o["propellant_kg"] == 60.0 and o["altimeter_ok"] == 1.0 and o["doppler_ok"] == 1.0
    assert "wind" not in OBS_NAMES                  # the agent is not told the weather (D17)


def test_render_rgb_array(make_env):
    env = make_env(render_mode="rgb_array", render_size=(160, 120))
    env.reset(seed=0)
    try:
        img = env.render()
    except Exception as exc:                        # no OpenGL context on this machine
        pytest.skip(f"offscreen rendering unavailable: {exc}")
    assert img.shape == (120, 160, 3) and img.std() > 5
    env.close()
