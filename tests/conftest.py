"""Shared pytest fixtures. Tests ignore LANDER_* environment overrides so results are reproducible."""

from __future__ import annotations

import pytest

from lander.config import load_config
from lander.sim import LanderSim


@pytest.fixture(scope="session")
def cfg():
    return load_config(environ={})


@pytest.fixture
def make_sim(cfg):
    """make_sim(overrides={...}, seed=0) -> LanderSim with optional .env overrides."""
    def _make(overrides=None, seed=0):
        c = cfg.with_overrides(overrides, origin="test") if overrides else cfg
        return LanderSim(c, seed=seed)
    return _make


# no air, no misalignment, perfect radar readings: for tests that compare against closed-form physics
# (wind is calm unless a reset option sets it)
CLEAN = {"MARS_AIR_DENSITY_KGPM3": 0.0, "VIKING_ENGINE_MISALIGN_DEG": 0.0, "SIM_SENSOR_NOISE_SCALE": 0.0}
