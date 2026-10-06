"""Vehicle layer: the three throttled terminal engines, propellant and mass (numpy only, D1).

Engine chain per physics step (brief section 6): command -> first-order lag (VIKING_ENGINE_T90_S, the
same up and down) -> clamp to [T_min, T_max] while lit -> engine fault scale -> misalignment -> force
at the engine site. Propellant flow uses the ACTUAL thrust (D6).

Shutdown (brief D7): the agent's `engines_off` command, or an empty tank, shuts all three engines
down for good. Thrust then decays with the same lag and nothing relights them (Viking's shutdown
valves could not reopen).

Throttle convention (Viking, S3 p96): a valve setting u in [0, 1] maps linearly to thrust
T_min + u (T_max - T_min); u = 0 is the minimum thrust (276 N), not zero.

Mass (brief D15): mass = dry + propellant left. The CG stays on the centre line and the inertias stay
at NASA's touchdown values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .config import G0_MPS2, Config
from .geometry import LanderGeometry


@dataclass
class Command:
    """Actuator command held for one control period (zero-order hold)."""

    throttle: tuple[float, ...] = (0.0, 0.0, 0.0)   # valve settings 0..1 per engine
    engines_off: bool = False                        # True shuts every engine down for good


class MassModel:
    """Mass = dry + propellant; fixed CG on the centre line and constant principal inertia (D15)."""

    def __init__(self, cfg: Config, geom: LanderGeometry):
        m = cfg.mass
        self.dry = m.dry_kg
        self.cg = geom.cg.copy()
        self.inertia = np.array([m.ixx_kgm2, m.iyy_kgm2, m.izz_kgm2])

    def total(self, propellant_kg: float) -> float:
        """Whole-vehicle mass (kg) with this much propellant left."""
        return self.dry + max(propellant_kg, 0.0)


class EngineBank:
    """The three throttled terminal engines."""

    def __init__(self, cfg: Config, geom: LanderGeometry):
        e = cfg.engine
        self.n = len(e.x_m)
        self.t_min, self.t_max = e.thrust_min_n, e.thrust_max_n
        self.isp_g0 = e.isp_s * G0_MPS2
        self.tau = e.t90_s / math.log(10.0)
        self.misalign = math.radians(e.misalign_deg)
        self.pos = geom.engine_pos
        self.dir = np.tile([0.0, 0.0, 1.0], (self.n, 1))
        self.thrust = np.zeros(self.n)
        self.cmd = np.zeros(self.n)
        self.fault = np.ones(self.n)
        self.lit = True

    def valve_to_thrust(self, u) -> np.ndarray:
        return self.t_min + np.clip(np.asarray(u, dtype=float), 0.0, 1.0) * (self.t_max - self.t_min)

    def thrust_to_valve(self, t) -> np.ndarray:
        return (np.asarray(t, dtype=float) - self.t_min) / (self.t_max - self.t_min)

    def reset(self, rng: np.random.Generator, throttle0, enabled: bool = True, fault_scale=None) -> None:
        self.lit = enabled
        self.fault = np.ones(self.n) if fault_scale is None else np.asarray(fault_scale, dtype=float)
        # misalignment: each thrust axis leans by the fixed angle toward a random azimuth
        beta = rng.uniform(0.0, 2.0 * math.pi, self.n)
        s, c = math.sin(self.misalign), math.cos(self.misalign)
        self.dir = np.column_stack([s * np.cos(beta), s * np.sin(beta), np.full(self.n, c)])
        start = self.valve_to_thrust(throttle0)
        self.cmd = start.copy()
        self.thrust = start.copy() if enabled else np.zeros(self.n)

    def command(self, throttle, engines_off: bool = False) -> None:
        self.cmd = self.valve_to_thrust(throttle)
        if engines_off:
            self.lit = False

    def step(self, dt: float, cut: bool = False) -> np.ndarray:
        """Advance one physics step; returns ACTUAL thrust per engine (N)."""
        if cut:
            self.lit = False
        target = self.cmd if self.lit else 0.0
        new = self.thrust + (target - self.thrust) * (1.0 - math.exp(-dt / self.tau))
        self.thrust = np.clip(new, self.t_min, self.t_max) if self.lit else np.clip(new, 0.0, self.t_max)
        return self.thrust * self.fault


@dataclass
class VehicleStep:
    thrust: np.ndarray            # actual thrust per engine (N)
    force_b: np.ndarray           # total thrust force, body frame (N)
    torque_b: np.ndarray          # thrust torque about the CG, body frame (N*m)
    mdot: float                   # propellant flow (kg/s)
    empty: bool                   # tank ran dry this step or earlier


@dataclass
class Vehicle:
    """Engines + propellant + mass model, stepped once per physics step."""

    cfg: Config
    geom: LanderGeometry
    engines: EngineBank = field(init=False)
    mass: MassModel = field(init=False)
    propellant: float = field(init=False, default=0.0)
    empty: bool = field(init=False, default=False)

    def __post_init__(self):
        self.engines = EngineBank(self.cfg, self.geom)
        self.mass = MassModel(self.cfg, self.geom)
        self._arm = self.geom.engine_pos - self.geom.cg   # lever arms; fixed because the CG does not move

    def reset(self, rng: np.random.Generator, propellant_kg: float, throttle0,
              engines_enabled: bool = True, fault_scale=None) -> None:
        self.propellant = float(propellant_kg)
        self.empty = self.propellant <= 0.0
        self.engines.reset(rng, throttle0, engines_enabled and not self.empty, fault_scale)

    def command(self, cmd: Command) -> None:
        self.engines.command(cmd.throttle, cmd.engines_off)

    def step(self, dt: float) -> VehicleStep:
        thrust = self.engines.step(dt, self.empty)
        mdot = float(thrust.sum()) / self.engines.isp_g0
        self.propellant -= mdot * dt
        if self.propellant <= 0.0:
            self.propellant, self.empty = 0.0, True
        forces = thrust[:, None] * self.engines.dir
        r = self._arm
        torque_b = np.array([r[:, 1] @ forces[:, 2] - r[:, 2] @ forces[:, 1],
                             r[:, 2] @ forces[:, 0] - r[:, 0] @ forces[:, 2],
                             r[:, 0] @ forces[:, 1] - r[:, 1] @ forces[:, 0]])
        return VehicleStep(thrust, forces.sum(axis=0), torque_b, mdot, self.empty)

    def trim_thrust(self, total_thrust: float) -> np.ndarray:
        """Per-engine thrust that gives `total_thrust` with zero pitch/yaw torque about the CG.

        Solves sum(T) = F, sum(T y) = 0, sum(T x) = 0 (engine 3 sits a little further out than
        engines 1 and 2 in S4 Fig 6a, so it runs ~2.5 % lower).
        """
        p, c = self.engines.pos, self.geom.cg
        a = np.vstack([np.ones(3), p[:, 1] - c[1], p[:, 0] - c[0]])
        return np.linalg.solve(a, np.array([total_thrust, 0.0, 0.0]))
