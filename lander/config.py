"""Load every physical fact and simulation setting from `.env` into typed, read-only dataclasses.

Each `.env` line is ``KEY=value  # unit | status | source | note``. The parser is strict: a line
without a valid status (published / derived / assumed) or source, an unknown key, or a missing key
stops the program, so nothing silently falls back to a hidden default. Commented-out lines (the
"parked" facts at the end of `.env`) are ignored.

Values can be overridden without editing the file, by an environment variable ``LANDER_<KEY>`` or
by `load_config(overrides=...)` (the scripts' ``--set KEY=VALUE``). An overridden value gets status
``override`` and its origin, so episode logs show exactly what differed from `.env`.
"""

from __future__ import annotations

import math
import os
import re
import typing
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping

STATUSES = ("published", "derived", "assumed")
OVERRIDE_STATUS = "override"
ENV_PREFIX = "LANDER_"
DEFAULT_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
G0_MPS2 = 9.80665   # standard gravity (CGPM 1901, exact by definition): Isp and g-loads

_LINE_RE = re.compile(r"^([A-Z][A-Z0-9_]*)\s*=\s*([^#]*?)\s*(?:#(.*))?$")
_INT_RE = re.compile(r"[+-]?\d+")


class ConfigError(ValueError):
    """Raised for any malformed, missing, unknown or inconsistent configuration value."""


@dataclass(frozen=True)
class Param:
    """One fact: its value plus the metadata needed to verify it."""

    key: str
    value: Any
    unit: str
    status: str
    source: str
    note: str
    origin: str = ".env"


# ----------------------------------------------------------------------------------------------
# Parameter groups. Field name = .env key without the group prefix, lower case.
# docs/env_variables.md explains every field and why it is kept.
# ----------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class EngineCfg:
    thrust_min_n: float
    thrust_max_n: float
    isp_s: float
    x_m: tuple[float, ...]
    y_m: tuple[float, ...]
    t90_s: float
    misalign_deg: float


@dataclass(frozen=True)
class MassCfg:
    dry_kg: float
    start_propellant_kg: float
    ixx_kgm2: float
    iyy_kgm2: float
    izz_kgm2: float
    cg_z_m: float


@dataclass(frozen=True)
class LegCfg:
    azimuth_deg: tuple[float, ...]
    footpad_radius_m: float
    footpad_z_m: float


@dataclass(frozen=True)
class AeroCfg:
    cda_m2: float


@dataclass(frozen=True)
class RadarCfg:
    period_s: float
    alt_min_range_m: float
    alt_err_frac: float
    alt_err_m: float
    doppler_splay_deg: float
    doppler_min_speed_mps: float
    doppler_err_frac: float
    doppler_err_mps: float


@dataclass(frozen=True)
class RequirementsCfg:
    vertical_speed_mps: float
    horizontal_speed_mps: float
    settle_ke_frac: float
    tilt_deg: float
    rate_pitchyaw_dps: float


@dataclass(frozen=True)
class SequenceCfg:
    terminal_speed_mps: float
    terminal_alt_m: float
    axial_throttle_limit: float


@dataclass(frozen=True)
class MarsCfg:
    gravity_mps2: float
    air_density_kgpm3: float


@dataclass(frozen=True)
class WindCfg:
    storm_mps: tuple[float, ...]


@dataclass(frozen=True)
class TerrainCfg:
    slope_deg: float
    slope_direction_deg: float
    friction: float
    slope_p90_deg: float
    slope_max_deg: float
    friction_range: tuple[float, ...]
    hold_ratio: float


@dataclass(frozen=True)
class SimCfg:
    physics_dt_s: float
    control_hz: float
    contact_timeconst_s: float
    sensor_noise_scale: float

    @property
    def control_dt_s(self) -> float:
        return 1.0 / self.control_hz

    @property
    def substeps(self) -> int:
        """Physics steps per control step (validated to be a whole number)."""
        return int(round(self.control_dt_s / self.physics_dt_s))


@dataclass(frozen=True)
class EpisodeCfg:
    start_height_m: tuple[float, ...]
    start_offset_m: tuple[float, ...]
    start_descent_mps: tuple[float, ...]
    start_sideways_mps: tuple[float, ...]
    start_tilt_deg: tuple[float, ...]
    start_rate_dps: float
    storm_fraction: float
    zones_m: tuple[float, ...]
    ceiling_m: float
    time_limit_s: float
    tilt_limit_deg: float
    settle_window_s: float

    @property
    def pad_radius_m(self) -> float:
        """A successful landing settles within this distance of the target (the middle zone)."""
        return self.zones_m[1]


@dataclass(frozen=True)
class RewardCfg:
    landing: tuple[float, ...]
    time_per_s: float
    tilt_free_deg: float
    tilt_per_deg: float
    failure: tuple[float, ...]
    crash_ratio: float


@dataclass(frozen=True)
class ControllerCfg:
    brake_mps2: float
    descent_gain_per_s: float
    position_gain_per_s: float
    side_speed_mps: float
    side_gain_per_s: float
    side_accel_mps2: float
    side_settle_s: float
    tilt_max_deg: tuple[float, ...]
    tilt_rate_dps: tuple[float, ...]
    attitude_radps: float
    attitude_damping: float
    fuel_margin: float


@dataclass(frozen=True)
class CurriculumCfg:
    start_height_m: tuple[float, ...]
    start_offset_m: tuple[float, ...]
    start_descent_mps: tuple[float, ...]
    start_sideways_mps: tuple[float, ...]
    start_tilt_deg: tuple[float, ...]
    start_rate_dps: float
    step: float
    promote_success: float
    window_episodes: float
    geometric: float
    mix: float


@dataclass(frozen=True)
class PpoCfg:
    gamma: float
    gae_lambda: float
    learning_rate: float
    steps_per_update: float
    minibatch: float
    epochs: float
    clip: float
    target_kl: float
    value_coef: float
    max_grad_norm: float
    net_arch: tuple[float, ...]
    init_std: float
    init_cut_prob: float


# Order matters: a key is assigned to the FIRST prefix it starts with (MARS_WIND_ before MARS_).
GROUPS: tuple[tuple[str, str, type], ...] = (
    ("VIKING_ENGINE_", "engine", EngineCfg),
    ("VIKING_MASS_", "mass", MassCfg),
    ("VIKING_LEG_", "leg", LegCfg),
    ("VIKING_AERO_", "aero", AeroCfg),
    ("VIKING_RADAR_", "radar", RadarCfg),
    ("VIKING_REQ_", "req", RequirementsCfg),
    ("VIKING_SEQ_", "seq", SequenceCfg),
    ("MARS_WIND_", "wind", WindCfg),
    ("MARS_TERRAIN_", "terrain", TerrainCfg),
    ("MARS_", "mars", MarsCfg),
    ("SIM_", "sim", SimCfg),
    ("EPISODE_", "episode", EpisodeCfg),
    ("REWARD_", "reward", RewardCfg),
    ("CTRL_", "ctrl", ControllerCfg),
    ("CURRICULUM_", "curriculum", CurriculumCfg),
    ("PPO_", "ppo", PpoCfg),
)


@dataclass(frozen=True, eq=False)
class Config:
    """All parameter groups plus the raw `Param` records (value + unit/status/source/origin)."""

    engine: EngineCfg
    mass: MassCfg
    leg: LegCfg
    aero: AeroCfg
    radar: RadarCfg
    req: RequirementsCfg
    seq: SequenceCfg
    wind: WindCfg
    terrain: TerrainCfg
    mars: MarsCfg
    sim: SimCfg
    episode: EpisodeCfg
    reward: RewardCfg
    ctrl: ControllerCfg
    curriculum: CurriculumCfg
    ppo: PpoCfg
    params: Mapping[str, Param]
    env_path: str

    def param(self, key: str) -> Param:
        """The full record (value, unit, status, source, note, origin) for one key."""
        if key not in self.params:
            raise ConfigError(f"unknown key {key}")
        return self.params[key]

    def by_status(self, status: str) -> list[Param]:
        """All params with the given status, e.g. 'assumed' (the ones to verify first)."""
        return [p for p in self.params.values() if p.status == status]

    def overridden(self) -> dict[str, Param]:
        """Params whose value came from an override instead of `.env`."""
        return {k: p for k, p in self.params.items() if p.status == OVERRIDE_STATUS}

    def to_log_dict(self) -> dict[str, dict[str, Any]]:
        """JSON-ready dump of every param, for episode logs."""
        return {
            k: {"value": list(p.value) if isinstance(p.value, tuple) else p.value,
                "unit": p.unit, "status": p.status, "source": p.source,
                "note": p.note, "origin": p.origin}
            for k, p in self.params.items()
        }

    def with_overrides(self, overrides: Mapping[str, Any], origin: str = "code") -> "Config":
        """A new Config with some values replaced (marked as overrides)."""
        params = dict(self.params)
        for key, raw in overrides.items():
            if key not in params:
                raise ConfigError(f"override of unknown key {key}")
            params[key] = _override(params[key], raw, origin)
        return build_config(params, self.env_path)


# ----------------------------------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------------------------------

def parse_value(text: str, key: str) -> Any:
    """Parse '3', '0.5' or '1,2,3' into an int, a float or a tuple of floats."""
    parts = [p.strip() for p in str(text).split(",")]
    if any(p == "" for p in parts):
        raise ConfigError(f"{key}: empty value in '{text}'")
    values: list[float | int] = []
    for p in parts:
        try:
            values.append(int(p) if _INT_RE.fullmatch(p) else float(p))
        except ValueError:
            raise ConfigError(f"{key}: '{p}' is not a number") from None
    if len(values) == 1:
        return values[0]
    return tuple(float(v) for v in values)


def parse_env_file(path: str | Path) -> dict[str, Param]:
    """Read a `.env` fact file into {key: Param}, enforcing the metadata format."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"fact file not found: {path}")
    params: dict[str, Param] = {}
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        where = f"{path.name}:{lineno}"
        m = _LINE_RE.match(line)
        if not m:
            raise ConfigError(f"{where}: cannot parse line: {raw!r}")
        key, value_text, meta = m.groups()
        if key in params:
            raise ConfigError(f"{where}: duplicate key {key}")
        if meta is None:
            raise ConfigError(f"{where}: {key} needs '# unit | status | source | note'")
        parts = [p.strip() for p in meta.split("|", 3)]
        if len(parts) < 3:
            raise ConfigError(f"{where}: {key} needs '# unit | status | source | note'")
        unit, status, source = parts[0], parts[1], parts[2]
        note = parts[3] if len(parts) > 3 else ""
        if status not in STATUSES:
            raise ConfigError(f"{where}: {key} status '{status}' is not one of {STATUSES}")
        if not source:
            raise ConfigError(f"{where}: {key} has no source (use '-' only for assumed values)")
        if source == "-" and status != "assumed":
            raise ConfigError(f"{where}: {key} is '{status}' but has no source")
        params[key] = Param(key, parse_value(value_text, key), unit, status, source, note)
    return params


def _override(old: Param, raw: Any, origin: str) -> Param:
    value = parse_value(raw, old.key) if isinstance(raw, str) else raw
    if isinstance(value, (list, tuple)):
        value = tuple(float(v) for v in value)
    if isinstance(old.value, tuple):
        if not isinstance(value, tuple):
            value = (float(value),)
        if len(value) != len(old.value):
            raise ConfigError(f"{old.key}: override needs {len(old.value)} values, got {len(value)}")
    elif isinstance(value, tuple):
        raise ConfigError(f"{old.key}: override must be a single number")
    was = f"was {old.value} ({old.status}; {old.source})"
    return Param(old.key, value, old.unit, OVERRIDE_STATUS, origin, was, origin)


def _group_of(key: str) -> tuple[str, str, type] | None:
    for prefix, name, cls in GROUPS:
        if key.startswith(prefix):
            return prefix, name, cls
    return None


def _coerce(key: str, value: Any, typ: Any) -> Any:
    if typ is int:
        if isinstance(value, tuple) or float(value) != int(float(value)):
            raise ConfigError(f"{key}: expected a whole number, got {value}")
        return int(value)
    if typ is float:
        if isinstance(value, tuple):
            raise ConfigError(f"{key}: expected one number, got a list {value}")
        return float(value)
    # tuple[float, ...]
    if isinstance(value, tuple):
        return tuple(float(v) for v in value)
    return (float(value),)


def build_config(params: Mapping[str, Param], env_path: str = "") -> Config:
    """Map {key: Param} onto the group dataclasses, checking that nothing is missing or extra."""
    buckets: dict[str, dict[str, Any]] = {name: {} for _, name, _ in GROUPS}
    for key, p in params.items():
        grp = _group_of(key)
        if grp is None:
            raise ConfigError(f"{key}: unknown prefix (expected one of {[g[0] for g in GROUPS]})")
        prefix, name, _ = grp
        buckets[name][key[len(prefix):].lower()] = (key, p.value)

    groups: dict[str, Any] = {}
    for prefix, name, cls in GROUPS:
        hints = typing.get_type_hints(cls)
        wanted = {f.name for f in fields(cls)}
        have = set(buckets[name])
        missing = sorted(prefix + f.upper() for f in wanted - have)
        unknown = sorted(prefix + f.upper() for f in have - wanted)
        if missing:
            raise ConfigError(f"missing keys: {missing}")
        if unknown:
            raise ConfigError(f"unknown keys (typo?): {unknown}")
        kwargs = {f: _coerce(buckets[name][f][0], buckets[name][f][1], hints[f]) for f in wanted}
        groups[name] = cls(**kwargs)

    cfg = Config(**groups, params=dict(params), env_path=str(env_path))
    validate(cfg)
    return cfg


def load_config(env_path: str | Path | None = None,
                overrides: Mapping[str, Any] | None = None,
                environ: Mapping[str, str] | None = None,
                origin: str = "cli") -> Config:
    """Read `.env`, apply LANDER_<KEY> environment overrides, then explicit overrides."""
    path = Path(env_path) if env_path else DEFAULT_ENV_PATH
    params = parse_env_file(path)
    env = os.environ if environ is None else environ
    for name, raw in env.items():
        if not name.startswith(ENV_PREFIX):
            continue
        key = name[len(ENV_PREFIX):]
        if key not in params:
            raise ConfigError(f"environment variable {name} does not match any key in {path.name}")
        params[key] = _override(params[key], raw, f"env:{name}")
    for key, raw in (overrides or {}).items():
        if key not in params:
            raise ConfigError(f"--set {key}: unknown key")
        params[key] = _override(params[key], raw, origin)
    return build_config(params, str(path))


def parse_set_args(items: list[str] | None) -> dict[str, str]:
    """Turn CLI ['KEY=VALUE', ...] into a dict for load_config(overrides=...)."""
    out: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise ConfigError(f"--set expects KEY=VALUE, got '{item}'")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


# ----------------------------------------------------------------------------------------------
# Consistency checks and derived numbers
# ----------------------------------------------------------------------------------------------

def validate(cfg: Config) -> None:
    """Cross-field sanity checks; raises ConfigError with a readable message."""
    e, m, leg, sim = cfg.engine, cfg.mass, cfg.leg, cfg.sim
    problems: list[str] = []
    if not (len(e.x_m) == len(e.y_m) == 3):
        problems.append("VIKING_ENGINE_X_M / _Y_M need one value for each of the 3 engines (locked decision D4)")
    if not 0 < e.thrust_min_n < e.thrust_max_n:
        problems.append("need 0 < VIKING_ENGINE_THRUST_MIN_N < VIKING_ENGINE_THRUST_MAX_N")
    if e.t90_s <= 0 or e.isp_s <= 0:
        problems.append("VIKING_ENGINE_T90_S and VIKING_ENGINE_ISP_S must be positive")
    if m.dry_kg <= 0 or m.start_propellant_kg < 0:
        problems.append("need VIKING_MASS_DRY_KG > 0 and VIKING_MASS_START_PROPELLANT_KG >= 0")
    i = (m.ixx_kgm2, m.iyy_kgm2, m.izz_kgm2)
    if min(i) <= 0 or i[0] + i[1] < i[2] or i[0] + i[2] < i[1] or i[1] + i[2] < i[0]:
        problems.append("principal inertias must be positive and satisfy the triangle inequality")
    if len(leg.azimuth_deg) != 3 or leg.footpad_radius_m <= 0:
        problems.append("need 3 leg azimuths and a positive VIKING_LEG_FOOTPAD_RADIUS_M")
    if len(cfg.wind.storm_mps) != 2 or not 0 <= cfg.wind.storm_mps[0] <= cfg.wind.storm_mps[1]:
        problems.append("MARS_WIND_STORM_MPS needs a range 'low,high' with 0 <= low <= high")
    if cfg.radar.period_s <= 0:
        problems.append("VIKING_RADAR_PERIOD_S must be positive")
    if sim.physics_dt_s <= 0 or sim.control_hz <= 0:
        problems.append("SIM_PHYSICS_DT_S and SIM_CONTROL_HZ must be positive")
    elif abs(sim.substeps * sim.physics_dt_s - sim.control_dt_s) > 1e-9:
        problems.append("control period must be a whole number of physics steps")
    if sim.contact_timeconst_s < 2 * sim.physics_dt_s:
        problems.append("SIM_CONTACT_TIMECONST_S should be >= 2 x SIM_PHYSICS_DT_S (MuJoCo stability)")
    if sim.sensor_noise_scale < 0:
        problems.append("SIM_SENSOR_NOISE_SCALE must be >= 0")
    ep, ter, rw = cfg.episode, cfg.terrain, cfg.reward
    for key, rng in (("EPISODE_START_HEIGHT_M", ep.start_height_m), ("EPISODE_START_OFFSET_M", ep.start_offset_m),
                     ("EPISODE_START_DESCENT_MPS", ep.start_descent_mps),
                     ("EPISODE_START_SIDEWAYS_MPS", ep.start_sideways_mps), ("EPISODE_START_TILT_DEG", ep.start_tilt_deg),
                     ("MARS_TERRAIN_FRICTION_RANGE", ter.friction_range)):
        if len(rng) != 2 or not 0 <= rng[0] <= rng[1]:
            problems.append(f"{key} needs a range 'low,high' with 0 <= low <= high")
    if len(ep.zones_m) != 3 or not 0 < ep.zones_m[0] < ep.zones_m[1] < ep.zones_m[2]:
        problems.append("EPISODE_ZONES_M needs three radii 'green,pad,arena', each larger than the last")
    elif ep.start_offset_m[-1] >= ep.zones_m[2]:
        problems.append("EPISODE_START_OFFSET_M reaches outside the arena (EPISODE_ZONES_M)")
    if ep.ceiling_m <= ep.start_height_m[-1]:
        problems.append("EPISODE_CEILING_M must be above the highest start (EPISODE_START_HEIGHT_M)")
    if ep.start_tilt_deg[-1] >= ep.tilt_limit_deg:
        problems.append("EPISODE_TILT_LIMIT_DEG must be above the largest start tilt (EPISODE_START_TILT_DEG)")
    if not 0 <= ep.storm_fraction <= 1:
        problems.append("EPISODE_STORM_FRACTION must be between 0 and 1")
    if ep.start_rate_dps < 0 or ep.time_limit_s <= 0 or ep.settle_window_s <= 0:
        problems.append("need EPISODE_START_RATE_DPS >= 0, EPISODE_TIME_LIMIT_S > 0 and EPISODE_SETTLE_WINDOW_S > 0")
    if ter.slope_p90_deg <= 0 or ter.slope_max_deg < 0:
        problems.append("need MARS_TERRAIN_SLOPE_P90_DEG > 0 and MARS_TERRAIN_SLOPE_MAX_DEG >= 0")
    elif ter.hold_ratio < 0 or (len(ter.friction_range) == 2 and 0 <= ter.slope_max_deg < 90 and
                                ter.hold_ratio * math.tan(math.radians(ter.slope_max_deg)) > ter.friction_range[1]):
        problems.append("MARS_TERRAIN_HOLD_RATIO must be >= 0 and leave friction to draw at the steepest slope "
                        "(HOLD_RATIO x tan(SLOPE_MAX_DEG) <= the top of FRICTION_RANGE)")
    if len(rw.landing) != 3 or len(rw.failure) != 2 or not rw.failure[0] <= rw.failure[1]:
        problems.append("REWARD_LANDING needs 3 values and REWARD_FAILURE 2 ('hard,soft', hard <= soft)")
    if rw.crash_ratio <= 1:
        problems.append("REWARD_CRASH_RATIO must be > 1 (a multiple of the leg speed limits)")
    sq, c = cfg.seq, cfg.ctrl
    if sq.terminal_speed_mps <= 0 or sq.terminal_alt_m <= 0 or not 0 < sq.axial_throttle_limit <= 1:
        problems.append("need VIKING_SEQ_TERMINAL_SPEED_MPS > 0, _TERMINAL_ALT_M > 0 and 0 < _AXIAL_THROTTLE_LIMIT <= 1")
    if min(c.brake_mps2, c.descent_gain_per_s, c.position_gain_per_s, c.side_speed_mps, c.side_gain_per_s,
           c.side_accel_mps2, c.attitude_radps, c.attitude_damping) <= 0 or c.side_settle_s < 0 or c.fuel_margin < 0:
        problems.append("CTRL_* gains, speeds and accelerations must be positive (SIDE_SETTLE_S, FUEL_MARGIN >= 0)")
    for key, pair in (("CTRL_TILT_MAX_DEG", c.tilt_max_deg), ("CTRL_TILT_RATE_DPS", c.tilt_rate_dps)):
        if len(pair) != 2 or not 0 < pair[1] <= pair[0]:
            problems.append(f"{key} needs 'above the gate, near the ground' with 0 < near <= above")
    if len(c.tilt_max_deg) == 2 and c.tilt_max_deg[0] >= ep.tilt_limit_deg:
        problems.append("CTRL_TILT_MAX_DEG must stay below EPISODE_TILT_LIMIT_DEG")
    axial = 3 * (e.thrust_min_n + sq.axial_throttle_limit * (e.thrust_max_n - e.thrust_min_n))
    if c.brake_mps2 >= axial / (m.dry_kg + m.start_propellant_kg) - cfg.mars.gravity_mps2:
        problems.append("CTRL_BRAKE_MPS2 must be below the braking the axial throttle limit allows at the start mass")
    cu, p = cfg.curriculum, cfg.ppo
    for key, rng in (("CURRICULUM_START_HEIGHT_M", cu.start_height_m), ("CURRICULUM_START_OFFSET_M", cu.start_offset_m),
                     ("CURRICULUM_START_DESCENT_MPS", cu.start_descent_mps),
                     ("CURRICULUM_START_SIDEWAYS_MPS", cu.start_sideways_mps),
                     ("CURRICULUM_START_TILT_DEG", cu.start_tilt_deg)):
        if len(rng) != 2 or not 0 <= rng[0] <= rng[1]:
            problems.append(f"{key} needs a range 'low,high' with 0 <= low <= high")
    if len(cu.start_height_m) == 2 and cu.start_height_m[0] <= 0:
        problems.append("CURRICULUM_START_HEIGHT_M must start above the ground")
    if cu.start_rate_dps < 0 or not 0 < cu.step <= 1 or not 0 < cu.promote_success <= 1 or cu.window_episodes < 10:
        problems.append("need CURRICULUM_START_RATE_DPS >= 0, 0 < _STEP <= 1, 0 < _PROMOTE_SUCCESS <= 1 and "
                        "_WINDOW_EPISODES >= 10")
    if cu.geometric not in (0.0, 1.0) or not 0 <= cu.mix < 1:
        problems.append("need CURRICULUM_GEOMETRIC 0 or 1 and 0 <= CURRICULUM_MIX < 1")
    if not 0 < p.gamma < 1 or not 0 < p.gae_lambda <= 1 or min(p.learning_rate, p.clip, p.target_kl, p.value_coef,
                                                                 p.max_grad_norm, p.init_std) <= 0:
        problems.append("need 0 < PPO_GAMMA < 1, 0 < PPO_GAE_LAMBDA <= 1 and positive PPO_LEARNING_RATE, _CLIP, "
                        "_TARGET_KL, _VALUE_COEF, _MAX_GRAD_NORM and _INIT_STD")
    whole = (p.steps_per_update, p.minibatch, p.epochs) + tuple(p.net_arch)
    if not p.net_arch or any(v < 1 or v != int(v) for v in whole) or p.minibatch > p.steps_per_update:
        problems.append("PPO_STEPS_PER_UPDATE, _MINIBATCH (<= steps per update), _EPOCHS and _NET_ARCH must be "
                        "whole numbers >= 1")
    if not 0 < p.init_cut_prob < 0.5:
        problems.append("PPO_INIT_CUT_PROB must be between 0 and 0.5 (D23: the untrained agent keeps the engines on)")
    if problems:
        raise ConfigError("inconsistent configuration:\n  - " + "\n  - ".join(problems))


def derived(cfg: Config) -> dict[str, tuple[float, str, str]]:
    """Numbers computed from the facts: {name: (value, unit, formula)}. Shown by check_config."""
    e, m, g = cfg.engine, cfg.mass, cfg.mars.gravity_mps2
    n = len(e.x_m)
    start = m.dry_kg + m.start_propellant_kg
    w_start = start * g
    hover_mdot = w_start / (e.isp_s * G0_MPS2)
    storm_hi = cfg.wind.storm_mps[1]
    out: dict[str, tuple[float, str, str]] = {
        "start_mass": (start, "kg", "dry + start propellant"),
        "weight_start": (w_start, "N", "start mass * g_Mars"),
        "mdot_per_engine_max": (e.thrust_max_n / (e.isp_s * G0_MPS2), "kg/s", "T_max / (Isp g0)"),
        "mdot_per_engine_min": (e.thrust_min_n / (e.isp_s * G0_MPS2), "kg/s", "T_min / (Isp g0)"),
        "thrust_to_weight_start": (n * e.thrust_max_n / w_start, "-", "3 T_max / weight_start"),
        "min_thrust_to_weight_start": (n * e.thrust_min_n / w_start, "-", "3 T_min / weight_start (the floor while lit)"),
        "hover_valve_setting_start": ((w_start / n - e.thrust_min_n) / (e.thrust_max_n - e.thrust_min_n),
                                      "-", "(weight/3 - T_min) / (T_max - T_min)"),
        "hover_mdot_start": (hover_mdot, "kg/s", "weight_start / (Isp g0)"),
        "hover_time_start_tank": (m.start_propellant_kg / hover_mdot, "s", "start propellant / hover mdot"),
        "engine_time_constant": (e.t90_s / 2.302585, "s", "T90 / ln 10 (first-order lag)"),
        "storm_push_max": (0.5 * cfg.mars.air_density_kgpm3 * storm_hi ** 2 * cfg.aero.cda_m2, "N",
                           "1/2 rho v^2 CdA at the top storm wind"),
        "axial_braking_start": (n * (e.thrust_min_n + cfg.seq.axial_throttle_limit * (e.thrust_max_n - e.thrust_min_n))
                                / start - g, "m/s2", "3 T(axial limit) / start mass - g (vs CTRL_BRAKE_MPS2)"),
        "cg_height_ratio": (m.cg_z_m / cfg.leg.footpad_radius_m, "-", "CG height / footpad radius (S4: 0.566)"),
        "control_substeps": (float(cfg.sim.substeps), "-", "control period / physics step"),
    }
    return out
