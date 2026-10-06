"""The fact file: strict parsing, overrides, parked facts, and the derived numbers quoted in the brief."""

from __future__ import annotations

import pytest

from lander.config import (DEFAULT_ENV_PATH, STATUSES, ConfigError, derived, load_config,
                           parse_env_file, parse_set_args)


def _env_copy(tmp_path, edit=None):
    text = DEFAULT_ENV_PATH.read_text(encoding="utf-8")
    if edit:
        text = edit(text)
    p = tmp_path / ".env"
    p.write_text(text, encoding="utf-8")
    return p


def test_every_fact_has_status_and_source(cfg):
    assert len(cfg.params) >= 30
    for p in cfg.params.values():
        assert p.status in STATUSES, p.key
        assert p.source, p.key
        if p.status != "assumed":
            assert p.source != "-", f"{p.key} is {p.status} but has no source"


def test_parked_facts_are_comments_only():
    text = DEFAULT_ENV_PATH.read_text(encoding="utf-8")
    assert "# VIKING_SEQ_TERMINAL_SPEED_MPS=2.44" in text           # kept for M3, with its source
    assert not any(k.startswith("VIKING_SEQ_") for k in parse_env_file(DEFAULT_ENV_PATH))


def test_missing_metadata_is_rejected(tmp_path):
    path = _env_copy(tmp_path, lambda t: t.replace("VIKING_ENGINE_ISP_S=195 ", "VIKING_ENGINE_ISP_S=195\n#", 1))
    with pytest.raises(ConfigError, match="unit \\| status"):
        parse_env_file(path)


def test_bad_status_is_rejected(tmp_path):
    path = _env_copy(tmp_path, lambda t: t.replace("| derived | S7,S8", "| guessed | S7,S8", 1))
    with pytest.raises(ConfigError, match="status 'guessed'"):
        parse_env_file(path)


def test_published_value_needs_a_source(tmp_path):
    path = _env_copy(tmp_path, lambda t: t.replace("| published | S2,S9 | per engine at valve setting 0",
                                                   "| published | - | per engine at valve setting 0", 1))
    with pytest.raises(ConfigError, match="no source"):
        parse_env_file(path)


def test_unknown_and_missing_keys_are_rejected(tmp_path):
    extra = _env_copy(tmp_path, lambda t: t + "\nVIKING_ENGINE_TYPO_N=1  # N | assumed | - |\n")
    with pytest.raises(ConfigError, match="unknown keys"):
        load_config(extra, environ={})
    (tmp_path / "b").mkdir()
    missing = _env_copy(tmp_path / "b", lambda t: "\n".join(
        line for line in t.splitlines() if not line.startswith("VIKING_MASS_DRY_KG")))
    with pytest.raises(ConfigError, match="missing keys"):
        load_config(missing, environ={})


def test_env_var_override_is_recorded():
    cfg = load_config(environ={"LANDER_VIKING_ENGINE_T90_S": "0.1"})
    assert cfg.engine.t90_s == 0.1
    p = cfg.param("VIKING_ENGINE_T90_S")
    assert p.status == "override" and p.origin == "env:LANDER_VIKING_ENGINE_T90_S" and "was 0.053" in p.note
    assert "VIKING_ENGINE_T90_S" in cfg.overridden()
    assert cfg.to_log_dict()["VIKING_ENGINE_T90_S"]["status"] == "override"


def test_cli_override_and_errors():
    cfg = load_config(overrides=parse_set_args(["VIKING_ENGINE_X_M=1,2,3", "MARS_GRAVITY_MPS2=3.7"]), environ={})
    assert cfg.engine.x_m == (1.0, 2.0, 3.0) and cfg.mars.gravity_mps2 == 3.7
    with pytest.raises(ConfigError):
        load_config(overrides={"VIKING_ENGINE_X_M": "1,2"}, environ={})          # wrong list length
    with pytest.raises(ConfigError):
        load_config(environ={"LANDER_NOT_A_KEY": "1"})                          # typo in env override
    with pytest.raises(ConfigError):
        load_config(overrides={"VIKING_ENGINE_THRUST_MIN_N": "3000"}, environ={})   # min above max


def test_derived_numbers_match_the_brief(cfg):
    d = {k: v[0] for k, v in derived(cfg).items()}
    assert d["mdot_per_engine_max"] == pytest.approx(1.39, abs=0.01)     # brief section 4
    assert d["mdot_per_engine_min"] == pytest.approx(0.14, abs=0.01)
    assert d["start_mass"] == pytest.approx(573.1 + 60.0)
    assert d["thrust_to_weight_start"] == pytest.approx(3.41, abs=0.01)
    assert d["min_thrust_to_weight_start"] == pytest.approx(0.35, abs=0.01)
    assert d["hover_mdot_start"] == pytest.approx(1.23, abs=0.01)
    assert 14.0 < d["storm_push_max"] < 15.0                             # a 30 m/s storm pushes ~14.5 N
    assert d["cg_height_ratio"] == pytest.approx(0.566, abs=0.001)      # S4 p5
