"""Gymnasium environment for the Viking lander (M2): `MarsLanderEnv`, registered as "MarsLander-v0".

Brief section 9 and decisions D16-D22:
- `reset(seed, options)` samples a start (D16, D19), or takes any part of it from `options`. The returned
  info["episode"] holds every resolved value, so `reset(options=info["episode"])` repeats the episode.
- Action: 4 numbers in [-1, 1] (D18). Three valve settings, u = (a + 1) / 2, and engines-off when the
  fourth is > 0 (permanent, D7).
- Observation: 22 numbers from the onboard navigation (radars + ideal IMU), never the truth (`OBS_NAMES`).
  Horizontal vectors are in the heading frame (world axes turned with the lander about the vertical), so
  the agent sees them relative to its own engines; spin is uncontrolled (D4).
- Reward: 0 every step, then one payment when the episode ends (D21): `landing_reward` or `failure_reward`.
- The episode ends on a landing (settled on all three feet with the engines off) or a failure (`FAILURES`).
- Curriculum (M4, D23, D29): `set_curriculum_level` or the reset option "level" picks the start ranges
  (`start_ranges`): 0 = the gentle start (CURRICULUM_START_*), 1 = D19's full range, the default. Level 1
  gives exactly the episodes the env gave before the curriculum existed (the fixed test sets use it).
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field, fields
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from lander.config import Config, load_config
from lander.sim import InitialState, LanderSim, ResetOptions, StepResult, quat_to_mat
from lander.vehicle import Command

OBS_NAMES = (
    "pos_x", "pos_y",                    # lander relative to the target, heading frame (m)
    "height",                            # feet above the ground, altimeter-based (m)
    "vel_x", "vel_y", "vel_z",           # velocity: horizontal in the heading frame, vertical up (m/s)
    "tilt_x", "tilt_y",                  # thrust axis leaning along heading x / y (sin of the lean)
    "rate_x", "rate_y", "rate_z",        # body rates from the IMU (rad/s); z is the spin
    "valve_1", "valve_2", "valve_3",     # valve settings commanded last step (0..1)
    "leg_1", "leg_2", "leg_3",           # leg has touched the ground (latched leg switch, 0/1)
    "engines_off",                       # engines shut down (0/1)
    "propellant_kg",                     # propellant left
    "altimeter_ok", "doppler_ok",        # latest radar readings valid (0/1)
    "time_s",                            # time since the start
)

FAILURES = {
    # in flight: REWARD_FAILURE[0]
    "lost_attitude": "tilted beyond EPISODE_TILT_LIMIT_DEG in flight",
    "left_arena": "flew out of the arena (EPISODE_ZONES_M)",
    "above_ceiling": "climbed above EPISODE_CEILING_M",
    "time_limit": "no touchdown within EPISODE_TIME_LIMIT_S",
    "propellant_exhausted": "the tank ran dry before touchdown",
    # at touchdown: graded by the impact speed (REWARD_FAILURE, REWARD_CRASH_RATIO)
    "leg_broken": "first contact faster than VIKING_REQ_VERTICAL_SPEED_MPS down or _HORIZONTAL_SPEED_MPS sideways",
    "touchdown_tilt": "tilt at first contact above VIKING_REQ_TILT_DEG",
    "touchdown_rate": "pitch/yaw rate at first contact above VIKING_REQ_RATE_PITCHYAW_DPS",
    "tipped_over": "the CG crossed the vertical plane through two footpads",
    "body_strike": "the body touched the ground",
    "engines_on": "the engines were still on when the settle window closed",
    "not_settled": "not resting on all three feet when the settle window closed",
}
IN_FLIGHT = ("lost_attitude", "left_arena", "above_ceiling", "time_limit", "propellant_exhausted")

_STATE_KEYS = ("position", "velocity", "quat", "omega", "propellant_kg")
_CONDITION_KEYS = tuple(f.name for f in fields(ResetOptions))
EPISODE_KEYS = ("sim_seed", "storm") + _STATE_KEYS + _CONDITION_KEYS


# ------------------------------------------------------------------------------------------ actions
def action_to_command(action) -> Command:
    """Agent action (4 numbers in [-1, 1]) -> simulator command: valves (a + 1) / 2, engines off if a[3] > 0."""
    a = np.asarray(action, dtype=np.float64).reshape(4)
    if not np.isfinite(a).all():
        raise ValueError(f"action must be finite, got {a}")
    a = np.clip(a, -1.0, 1.0)
    return Command(tuple(float(v) for v in (a[:3] + 1.0) / 2.0), engines_off=bool(a[3] > 0.0))


# ------------------------------------------------------------------------------------------ observations
def heading_axes(rot: np.ndarray) -> tuple[float, float]:
    """cos and sin of the lander's heading: its body X axis projected on the ground."""
    psi = math.atan2(float(rot[1, 0]), float(rot[0, 0]))
    return math.cos(psi), math.sin(psi)


def to_heading(c: float, s: float, v) -> tuple[float, float]:
    """The horizontal part of world vector v, in the heading frame."""
    return c * float(v[0]) + s * float(v[1]), -s * float(v[0]) + c * float(v[1])


def observe(res: StepResult) -> np.ndarray:
    """The agent's observation (`OBS_NAMES`) from a StepResult: navigation estimate, IMU, own state."""
    tl, nav = res.telemetry, res.nav
    rot = quat_to_mat(tl.quat)
    c, s = heading_axes(rot)
    px, py = to_heading(c, s, nav.position)
    vx, vy = to_heading(c, s, nav.velocity)
    zx, zy = to_heading(c, s, rot[:, 2])
    w = res.sensors.gyro
    return np.array([px, py, nav.height, vx, vy, nav.velocity[2], zx, zy, w[0], w[1], w[2],
                     *tl.throttle_cmd, *(float(b) for b in tl.pads_touched), float(tl.engines_off),
                     tl.propellant_kg, float(nav.altimeter_ok), float(nav.doppler_ok), tl.t], dtype=np.float32)


def truth(res: StepResult) -> dict[str, Any]:
    """Ground truth for info dicts (debugging and analysis only; the agent never sees it)."""
    tl = res.telemetry
    return {"position": tl.cg.copy(), "velocity": tl.vel.copy(), "tilt_deg": tl.tilt_deg,
            "height": tl.pad_altitude, "propellant_kg": tl.propellant_kg, "wind": tl.wind.copy()}


# ------------------------------------------------------------------------------------------ episode start
def _as_list(x) -> list:
    return [float(v) for v in np.asarray(x, dtype=float).ravel()]


def _holding_friction(ter, slope_deg: float, u: float) -> float:
    """Friction from a uniform draw u in [0, 1): uniform in MARS_TERRAIN_FRICTION_RANGE, but at least
    MARS_TERRAIN_HOLD_RATIO x tan(slope), so a lander at rest holds on the slope (D26)."""
    lo, hi = ter.friction_range
    lo = min(hi, max(lo, ter.hold_ratio * math.tan(math.radians(min(abs(slope_deg), 89.0)))))
    return lo + (hi - lo) * u


def start_ranges(cfg: Config, level: float = 1.0) -> dict[str, Any]:
    """The start ranges at a curriculum level (D29): 0 = CURRICULUM_START_*, 1 = EPISODE_START_* (D19).

    Both ends of every range move with the level: by equal factors (CURRICULUM_GEOMETRIC = 1; an end that is 0
    at either level moves linearly) or by equal amounts (0). Level 1 returns EPISODE_START_* exactly.
    """
    if not 0.0 <= level <= 1.0:
        raise ValueError(f"curriculum level must be between 0 and 1, got {level}")
    ep, cu = cfg.episode, cfg.curriculum
    geometric = cu.geometric >= 0.5

    def mix(full: float, easy: float) -> float:
        if geometric and full > 0.0 and easy > 0.0:
            return full * (easy / full) ** (1.0 - level)
        return full + (1.0 - level) * (easy - full)

    pairs = {"height": (ep.start_height_m, cu.start_height_m), "offset": (ep.start_offset_m, cu.start_offset_m),
             "descent": (ep.start_descent_mps, cu.start_descent_mps),
             "sideways": (ep.start_sideways_mps, cu.start_sideways_mps), "tilt": (ep.start_tilt_deg, cu.start_tilt_deg)}
    out: dict[str, Any] = {k: (mix(full[0], easy[0]), mix(full[1], easy[1])) for k, (full, easy) in pairs.items()}
    out["rate"] = mix(ep.start_rate_dps, cu.start_rate_dps)
    return out


def sample_episode(sim: LanderSim, rng: np.random.Generator, given: dict | None = None,
                   level: float = 1.0) -> dict[str, Any]:
    """Every random choice of an episode (D16, D19, D26); values in `given` (keys: `EPISODE_KEYS`) take precedence.

    All numbers are drawn every time in a fixed order, so giving one value never changes the others. The
    friction draw is placed in its range once the slope is known (given or drawn). The start ranges are those
    of the curriculum `level` (`start_ranges`; 1 = D19).
    Returns a JSON-ready dict with every value resolved.
    """
    given = dict(given or {})
    unknown = set(given) - set(EPISODE_KEYS)
    if unknown:
        raise ValueError(f"unknown episode options {sorted(unknown)}; valid: {sorted(EPISODE_KEYS)}, 'level' "
                         "and 'overrides'")
    cfg = sim.cfg
    ep, ter = cfg.episode, cfg.terrain
    ranges = start_ranges(cfg, level)
    # ---- draws, in a fixed order
    sim_seed = int(rng.integers(0, 2 ** 31 - 1))
    lam = math.log(10.0) / ter.slope_p90_deg                 # exponential slope with this 90th percentile (S4),
    top = 1.0 - math.exp(-lam * ter.slope_max_deg)           # truncated at the cap (inverse CDF, one draw)
    slope = abs(math.log(1.0 - float(rng.random()) * top) / lam)
    downhill = float(rng.uniform(0.0, 360.0))
    friction_u = float(rng.random())
    storm = bool(rng.random() < ep.storm_fraction)
    storm_speed = float(rng.uniform(*cfg.wind.storm_mps))
    wind_dir = float(rng.uniform(0.0, 360.0))
    height = float(rng.uniform(*ranges["height"]))
    offset, offset_az = float(rng.uniform(*ranges["offset"])), float(rng.uniform(0.0, 2.0 * math.pi))
    descent = float(rng.uniform(*ranges["descent"]))
    side, side_az = float(rng.uniform(*ranges["sideways"])), float(rng.uniform(0.0, 2.0 * math.pi))
    tilt, tilt_az = float(rng.uniform(*ranges["tilt"])), float(rng.uniform(0.0, 360.0))
    heading = float(rng.uniform(0.0, 360.0))
    omega = rng.uniform(-1.0, 1.0, 3) * math.radians(ranges["rate"])
    # ---- given values win
    if "storm" in given:
        storm = bool(given["storm"])
    elif "wind_speed_mps" in given:
        storm = float(given["wind_speed_mps"]) > 0.0
    e = {
        "sim_seed": int(given.get("sim_seed", sim_seed)),
        "storm": storm,
        "wind_speed_mps": float(given.get("wind_speed_mps", storm_speed if storm else 0.0)),
        "wind_direction_deg": float(given.get("wind_direction_deg", wind_dir)),
        "slope_deg": float(given.get("slope_deg", slope)),
        "slope_direction_deg": float(given.get("slope_direction_deg", downhill)),
    }
    e["friction"] = float(given["friction"]) if "friction" in given else _holding_friction(ter, e["slope_deg"], friction_u)
    st = sim.state_above_ground(
        height, velocity=(side * math.cos(side_az), side * math.sin(side_az), -descent), tilt_deg=tilt,
        tilt_azimuth_deg=tilt_az, heading_deg=heading, omega=tuple(omega),
        xy=(offset * math.cos(offset_az), offset * math.sin(offset_az)),
        slope_deg=e["slope_deg"], slope_direction_deg=e["slope_direction_deg"])
    for key in _STATE_KEYS:
        value = given.get(key, getattr(st, key))
        e[key] = float(value) if key == "propellant_kg" else _as_list(value)
    defaults = ResetOptions()
    for key in _CONDITION_KEYS:
        if key not in e:
            value = given.get(key, getattr(defaults, key))
            if key == "sensor_noise_scale" and value is None:
                value = cfg.sim.sensor_noise_scale
            e[key] = bool(value) if key == "engines_enabled" else (
                _as_list(value) if isinstance(value, (tuple, list, np.ndarray)) else float(value))
    e["level"] = float(level)
    return e


# ------------------------------------------------------------------------------------------ outcome and reward
@dataclass
class Outcome:
    """How an episode ended."""

    result: str                  # "landed" (inside the pad radius), "landed_off_pad" or "failed"
    reason: str                  # a key of FAILURES, or "" for a landing
    reward: float
    parts: dict[str, float] = field(default_factory=dict)   # how the reward was made up
    distance_m: float = math.nan                             # final horizontal distance from the target
    time_s: float = math.nan

    @property
    def success(self) -> bool:
        """Brief section 7: a safe landing inside the pad radius."""
        return self.result == "landed"

    @property
    def safe(self) -> bool:
        """Landed safely, on or off the pad."""
        return self.result != "failed"

    def as_dict(self) -> dict[str, Any]:
        return {"result": self.result, "reason": self.reason, "success": self.success, "safe": self.safe,
                "reward": self.reward, "parts": dict(self.parts), "distance_m": self.distance_m, "time_s": self.time_s}


def landing_reward(cfg: Config, distance_m: float, time_s: float, peak_tilt_deg: float) -> tuple[float, dict]:
    """A safe landing's reward (D21): zone score by distance, minus the time and peak-tilt costs.

    Zone score: REWARD_LANDING[0] inside the green zone, then linear to REWARD_LANDING[1] at the pad edge
    and to REWARD_LANDING[2] at the arena edge (EPISODE_ZONES_M).
    """
    rw, zones = cfg.reward, cfg.episode.zones_m
    zone = float(np.interp(distance_m, zones, rw.landing))
    time_cost = rw.time_per_s * time_s
    tilt_cost = rw.tilt_per_deg * max(0.0, peak_tilt_deg - rw.tilt_free_deg)
    return zone - time_cost - tilt_cost, {"zone": zone, "time": -time_cost, "tilt": -tilt_cost}


def impact_ratio(cfg: Config, report) -> float:
    """First-contact speed as a multiple of the leg limits: max(down / 3.34, sideways / 1.22); <= 1 is safe."""
    q = cfg.req
    return max(max(report.vz_at_contact, 0.0) / q.vertical_speed_mps, report.vh_at_contact / q.horizontal_speed_mps)


def failure_reward(cfg: Config, reason: str, ratio: float | None = None) -> float:
    """A failure's reward (D21): REWARD_FAILURE[0] in flight; at touchdown from REWARD_FAILURE[1] within the
    leg limits (ratio <= 1), falling linearly to REWARD_FAILURE[0] at REWARD_CRASH_RATIO x the limits."""
    hard, soft = cfg.reward.failure
    if reason in IN_FLIGHT or ratio is None:
        return hard
    frac = min(1.0, max(0.0, (ratio - 1.0) / (cfg.reward.crash_ratio - 1.0)))
    return soft + (hard - soft) * frac


def judge(sim: LanderSim, res: StepResult, peak_tilt_deg: float) -> Outcome | None:
    """The episode's outcome after a step, or None while it goes on (in-flight limits, then the touchdown)."""
    cfg, tl, r = sim.cfg, res.telemetry, res.report
    ep = cfg.episode
    dist = math.hypot(float(tl.cg[0]), float(tl.cg[1]))

    def failed(reason: str, ratio: float | None = None) -> Outcome:
        rew = failure_reward(cfg, reason, ratio)
        parts = {"failure": rew} if ratio is None else {"failure": rew, "impact_ratio": ratio}
        return Outcome("failed", reason, rew, parts, dist, tl.t)

    if not r.touched_down:
        if r.propellant_exhausted_airborne or sim.vehicle.empty:
            return failed("propellant_exhausted")
        if tl.tilt_deg > ep.tilt_limit_deg:
            return failed("lost_attitude")
        if dist > ep.zones_m[2]:
            return failed("left_arena")
        if tl.pad_altitude > ep.ceiling_m:              # height above the ground, like the start height
            return failed("above_ceiling")
        if tl.t >= ep.time_limit_s - 1e-9:
            return failed("time_limit")
        return None
    ratio = impact_ratio(cfg, r)
    crit = r.criteria
    for reason, bad in (("leg_broken", r.leg_broken), ("touchdown_tilt", not crit.get("tilt_within_limit", True)),
                        ("touchdown_rate", not crit.get("rate_within_limit", True)), ("tipped_over", r.tipped_over),
                        ("body_strike", r.body_strike)):
        if bad:
            return failed(reason, ratio)
    if r.settled and r.engines_off and all(tl.pad_contact) and float(tl.thrust_n.sum()) < 1.0:
        rew, parts = landing_reward(cfg, dist, tl.t, peak_tilt_deg)
        return Outcome("landed" if dist <= ep.pad_radius_m else "landed_off_pad", "", rew, parts, dist, tl.t)
    if tl.t - r.first_contact_time >= ep.settle_window_s - 1e-9:
        return failed("not_settled" if r.engines_off else "engines_on", ratio)
    return None


# ------------------------------------------------------------------------------------------ the env
class MarsLanderEnv(gym.Env):
    """The Viking lander on Mars as a Gymnasium environment (see the module docstring).

    `config` defaults to `.env`; `overrides` ({KEY: value}) apply to every episode. A reset can also carry
    "overrides" in its options for that episode alone (the simulator is rebuilt when the config changes).
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}

    def __init__(self, config: Config | None = None, overrides: dict | None = None, render_mode: str | None = None,
                 render_size: tuple[int, int] = (640, 480), camera: str = "chase"):
        base = config if config is not None else load_config()
        self.base_cfg = base.with_overrides(overrides, origin="env") if overrides else base
        self.sim = LanderSim(self.base_cfg)
        self.metadata = {**self.metadata, "render_fps": self.base_cfg.sim.control_hz}
        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(len(OBS_NAMES),), dtype=np.float32)
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"render_mode must be one of {self.metadata['render_modes']} or None")
        self.render_mode, self.render_size, self.camera = render_mode, render_size, camera
        self._frames = None
        self._res: StepResult | None = None
        self._peak_tilt = 0.0
        self.episode: dict[str, Any] = {}
        self.outcome: Outcome | None = None
        self.curriculum_level = 1.0
        self._mix = 0.0                 # share of episodes started below the level (training only, CURRICULUM_MIX)

    @property
    def last_result(self) -> StepResult | None:
        """The simulator's StepResult from the last reset or step: truth, sensors, navigation, touchdown report."""
        return self._res

    def set_curriculum_level(self, level: float) -> None:
        """Start the following episodes at this curriculum level (0 = gentle, 1 = D19's full range; M4 training).

        From then on a share CURRICULUM_MIX of the episodes starts at a random level below it instead.
        """
        start_ranges(self.base_cfg, level)          # checks the range
        self.curriculum_level = float(level)
        self._mix = self.base_cfg.curriculum.mix

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        given = dict(options or {})
        overrides = given.pop("overrides", None)
        if "level" in given:
            level = float(given.pop("level"))
        else:
            level = self.curriculum_level
            if self._mix > 0.0 and level > 0.0 and self.np_random.random() < self._mix:
                level = float(self.np_random.uniform(0.0, level))
        cfg = self.base_cfg.with_overrides(overrides, origin="episode") if overrides else self.base_cfg
        if cfg.params != self.sim.cfg.params:
            self.close()
            self.sim = LanderSim(cfg)
        self.episode = sample_episode(self.sim, self.np_random, given, level)
        e = self.episode
        st = InitialState(position=tuple(e["position"]), velocity=tuple(e["velocity"]), quat=tuple(e["quat"]),
                          omega=tuple(e["omega"]), propellant_kg=e["propellant_kg"])
        opts = ResetOptions(**{k: tuple(e[k]) if isinstance(e[k], list) else e[k] for k in _CONDITION_KEYS})
        self._res = self.sim.reset(seed=e["sim_seed"], initial_state=st, options=opts)
        self._peak_tilt = self._res.telemetry.tilt_deg
        self.outcome = None
        return observe(self._res), {"episode": copy.deepcopy(e), "truth": truth(self._res)}

    def step(self, action):
        if self.outcome is not None:
            raise RuntimeError("the episode has ended: call reset()")
        res = self.sim.step(action_to_command(action))
        self._res = res
        if not res.report.touched_down:
            self._peak_tilt = max(self._peak_tilt, res.telemetry.tilt_deg)
        self.outcome = judge(self.sim, res, self._peak_tilt)
        info: dict[str, Any] = {"truth": truth(res), "events": list(res.events)}
        reward = 0.0
        if self.outcome is not None:
            reward = self.outcome.reward
            info["outcome"] = self.outcome.as_dict()
            info["is_success"] = self.outcome.success
            info["level"], info["storm"] = self.episode["level"], self.episode["storm"]
        return observe(res), float(reward), self.outcome is not None, False, info

    def render(self):
        """An RGB frame with plumes and the HUD (render_mode "rgb_array"), or None."""
        if self.render_mode != "rgb_array" or self._res is None:
            return None
        if self._frames is None:
            from lander.video import FrameRenderer
            self._frames = FrameRenderer(self.sim, *self.render_size, camera=self.camera)
        return self._frames.render(self._res)

    def close(self):
        if self._frames is not None:
            self._frames.close()
            self._frames = None
