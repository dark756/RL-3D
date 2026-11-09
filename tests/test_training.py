"""M4 training: the curriculum levels, the untrained agent's start, the training loop, resume and checkpoints."""

from __future__ import annotations

import json
import math
from statistics import NormalDist
from types import SimpleNamespace

import numpy as np
import pytest

from controllers.ppo import FixedMeanStd, PpoPolicy
from envs.mars_lander_env import MarsLanderEnv, sample_episode, start_ranges
from lander.config import ConfigError, load_config
from scripts.evaluate import run_episode
from scripts.train import RunSettings, SeedState, _callbacks, config_changes, parse_seeds, start_run, train_seed
from scripts.training_report import seed_status

TINY = {"PPO_STEPS_PER_UPDATE": "512", "PPO_MINIBATCH": "256", "PPO_EPOCHS": "2"}


def test_curriculum_ranges(cfg):
    ep, cu = cfg.episode, cfg.curriculum
    linear = cfg.with_overrides({"CURRICULUM_GEOMETRIC": "0"})
    full, easy, mid = start_ranges(cfg, 1.0), start_ranges(cfg, 0.0), start_ranges(linear, 0.5)
    assert full["height"] == ep.start_height_m and full["tilt"] == ep.start_tilt_deg       # exactly D19 at level 1
    assert full["descent"] == ep.start_descent_mps and full["rate"] == ep.start_rate_dps
    assert easy["height"] == pytest.approx(cu.start_height_m) and easy["offset"] == pytest.approx(cu.start_offset_m)
    assert mid["height"] == pytest.approx(tuple((a + b) / 2 for a, b in zip(ep.start_height_m, cu.start_height_m)))
    with pytest.raises(ValueError, match="between 0 and 1"):
        start_ranges(cfg, 1.5)


def test_geometric_levels_and_mixing(cfg):
    geo = cfg.with_overrides({"CURRICULUM_GEOMETRIC": "1", "CURRICULUM_START_HEIGHT_M": "5,15"})
    ep = geo.episode
    assert start_ranges(geo, 1.0)["height"] == ep.start_height_m and start_ranges(geo, 1.0)["rate"] == ep.start_rate_dps
    steps = [start_ranges(geo, k / 10)["height"][1] for k in range(11)]
    ratios = np.array(steps[1:]) / np.array(steps[:-1])
    assert np.allclose(ratios, (ep.start_height_m[1] / 15.0) ** 0.1)            # every level: the same factor
    assert start_ranges(geo, 0.5)["offset"][0] == 0.0                           # an end at 0 moves linearly
    mixed = MarsLanderEnv(config=cfg.with_overrides({"CURRICULUM_MIX": "0.5"}))
    _, info = mixed.reset(seed=3)
    assert info["episode"]["level"] == 1.0                                      # no mixing unless training set a level
    mixed.set_curriculum_level(0.6)
    levels = np.array([mixed.reset(seed=k)[1]["episode"]["level"] for k in range(200)])
    assert 0.35 < np.mean(levels == 0.6) < 0.65 and levels.min() < 0.3 and levels.max() == 0.6
    assert mixed.reset(seed=1, options={"level": 0.6})[1]["episode"]["level"] == 0.6   # an explicit level wins


def test_level_one_is_the_episode_without_curriculum(cfg):
    env = MarsLanderEnv(config=cfg)
    _, info = env.reset(seed=1_000_000, options={"storm": False})
    plain = info["episode"]
    rng = np.random.default_rng(np.random.SeedSequence(1_000_000))
    assert plain["level"] == 1.0
    env.set_curriculum_level(0.0)                       # an explicit level in the options wins over the env's
    _, info = env.reset(seed=1_000_000, options={"storm": False, "level": 1.0})
    assert info["episode"] == plain
    assert sample_episode(env.sim, rng, {"storm": False}, 1.0).keys() == plain.keys()


def test_low_level_starts_and_the_level_in_the_last_info(cfg):
    env = MarsLanderEnv(config=cfg)
    env.set_curriculum_level(0.0)
    cu = cfg.curriculum
    for k in range(20):
        _, info = env.reset(seed=50_000 + k)
        e, tl = info["episode"], env.last_result.telemetry
        assert e["level"] == 0.0
        assert cu.start_height_m[0] - 1e-6 <= tl.pad_altitude <= cu.start_height_m[1] + 1e-6
        assert cu.start_descent_mps[0] - 1e-9 <= -e["velocity"][2] <= cu.start_descent_mps[1] + 1e-9
        assert math.hypot(*e["position"][:2]) <= cu.start_offset_m[1] + 1e-9
        assert tl.tilt_deg <= cu.start_tilt_deg[1] + 1e-6
    done = False
    while not done:
        _, _, done, _, info = env.step(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))   # engines off: falls
    assert info["level"] == 0.0 and info["storm"] == e["storm"]
    with pytest.raises(ValueError):
        env.set_curriculum_level(-0.1)


def test_training_settings_are_checked():
    with pytest.raises(ConfigError, match="CURRICULUM_START_HEIGHT_M"):
        load_config(overrides={"CURRICULUM_START_HEIGHT_M": "50,20"}, environ={})
    with pytest.raises(ConfigError, match="PPO_GAMMA"):
        load_config(overrides={"PPO_GAMMA": "1"}, environ={})
    with pytest.raises(ConfigError, match="_MINIBATCH"):
        load_config(overrides={"PPO_MINIBATCH": "9000"}, environ={})              # more than one update's steps
    with pytest.raises(ConfigError, match="PPO_INIT_CUT_PROB"):
        load_config(overrides={"PPO_INIT_CUT_PROB": "0.6"}, environ={})
    with pytest.raises(ConfigError, match="_TARGET_KL"):
        load_config(overrides={"PPO_TARGET_KL": "0"}, environ={})
    assert parse_seeds("0-4") == [0, 1, 2, 3, 4] and parse_seeds("3,7") == [3, 7]
    with pytest.raises(ValueError):
        parse_seeds("2,2")


def test_curriculum_raises_the_level_after_a_successful_window(cfg, tmp_path):
    curriculum, _, _ = _callbacks()
    state = SeedState()
    cb = curriculum(cfg, state, tmp_path, 0, lambda: 0.0)
    calls = []

    class Venv:
        def env_method(self, name, value):
            calls.append((name, value))

    cb.model, cb.num_timesteps = SimpleNamespace(get_env=Venv), 0
    window = int(cfg.curriculum.window_episodes)
    done = lambda success, level: {"outcome": {"success": success, "reward": 90.0 if success else -100.0},
                                   "level": level, "episode": {"l": 300}}
    good = int(math.ceil(cfg.curriculum.promote_success * window))
    cb.locals = {"infos": [done(False, 0.0)] * (window - good) + [done(True, 0.5)] * 50}   # other levels don't count
    cb._on_step()
    assert state.level == 0.0 and not calls
    cb.locals = {"infos": [done(True, 0.0)] * (good - 1)}
    cb._on_step()
    assert state.level == 0.0                                  # one short of the window
    cb.locals = {"infos": [done(True, 0.0)]}
    cb._on_step()
    assert state.level == pytest.approx(cfg.curriculum.step) and calls == [("set_curriculum_level", state.level)]
    assert state.episodes_at_level == 0 and len(cb.window) == 0


def test_untrained_agent_keeps_the_engines_on_at_hover_trim(cfg, tmp_path):
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecNormalize

    from lander.sim import LanderSim
    from scripts.train import make_model

    venv = VecNormalize(make_vec_env(MarsLanderEnv, n_envs=1, env_kwargs={"config": cfg}), gamma=cfg.ppo.gamma)
    model = make_model(cfg, venv, 0, tmp_path)
    dist = model.policy.get_distribution(model.policy.obs_to_tensor(np.zeros((1, 22), dtype=np.float32))[0])
    mean = dist.distribution.mean.detach().numpy()[0]
    std = dist.distribution.stddev.detach().numpy()[0]
    hover = 2.0 * np.array(LanderSim(cfg).hover_trim(cfg.mass.start_propellant_kg)) - 1.0
    assert mean[:3] == pytest.approx(hover, abs=0.05) and std == pytest.approx(cfg.ppo.init_std, rel=1e-6)
    p_cut = 1.0 - NormalDist(float(mean[3]), float(std[3])).cdf(0.0)
    assert p_cut == pytest.approx(cfg.ppo.init_cut_prob, rel=0.5)
    venv.close()


def test_train_resume_checkpoints_and_evaluate(tmp_path, cfg):
    run = tmp_path / "run"
    settings = RunSettings(seeds=[0], envs_per_seed=1, subproc=False, checkpoint_every=512, keep_every=1024,
                           validate_every=512, validate_episodes=2, calibration_episodes=1, overrides=TINY)
    start_run(run, settings)
    scaling = json.loads((run / "obs_scaling.json").read_text())
    assert len(scaling["mean"]) == 22 and min(scaling["var"]) > 0
    first = train_seed(run, 0, steps=1024)
    assert first.timesteps == 1024 and first.sessions == 1 and first.episodes > 0
    second = train_seed(run, 0, steps=512)                       # resumes from checkpoints/latest
    assert second.timesteps == 1536 and second.sessions == 2
    seed = run / "seed0"
    for path in ("checkpoints/latest/model.zip", "checkpoints/latest/vecnormalize.pkl", "checkpoints/step_1024",
                 "best/model.zip", "curriculum.csv", "validation.csv"):
        assert (seed / path).exists(), path
    assert seed_status(seed)["steps"] >= 1024
    resumed = PpoPolicy(seed / "checkpoints" / "latest")
    assert resumed.model.target_kl == cfg.ppo.target_kl            # every update is capped, also after a resume
    policy = PpoPolicy(seed / "best")
    assert isinstance(policy.norm.obs_rms, FixedMeanStd)
    before = policy.norm.obs_rms.mean.copy()
    policy.norm.obs_rms.update(np.ones((3, 22)))
    assert np.array_equal(policy.norm.obs_rms.mean, before)        # the scaling never moves
    row = run_episode(MarsLanderEnv(config=cfg), f"ppo:{seed / 'best'}", 1_000_000, False)
    assert row["result"] in ("landed", "landed_off_pad", "failed") and row["fuel_reserve"] is None
    assert config_changes(json.loads((run / "run.json").read_text()), load_config(overrides=TINY)) == []
