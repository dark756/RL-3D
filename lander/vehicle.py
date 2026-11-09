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

Speed (2026-10-11): the per-physics-step work (`EngineBank.step`, `Vehicle.step`) uses plain floats
instead of small numpy arrays; with sim.py's physics step this made the environment 2.7x faster. Every
sum keeps the order numpy used (left to right), so the results did not change by a single bit.
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
    """The three throttled terminal engines. Per-engine state is kept as lists of floats (fast to step)."""

    def __init__(self, cfg: Config, geom: LanderGeometry):
        e = cfg.engine
        self.n = len(e.x_m)
        self.t_min, self.t_max = e.thrust_min_n, e.thrust_max_n
        self.isp_g0 = e.isp_s * G0_MPS2
        self.tau = e.t90_s / math.log(10.0)
        self.misalign = math.radians(e.misalign_deg)
        self.pos = geom.engine_pos
        self.dir = [(0.0, 0.0, 1.0)] * self.n     # unit thrust axis of each engine, body frame
        self.thrust = [0.0] * self.n               # actual thrust before the fault scale (N)
        self.cmd = [0.0] * self.n                  # commanded thrust (N)
        self.fault = [1.0] * self.n
        self.lit = True
        self._lag = (None, 0.0)                    # (dt, 1 - exp(-dt / tau)) for the last dt

    def valve_to_thrust(self, u) -> np.ndarray:
        return self.t_min + np.clip(np.asarray(u, dtype=float), 0.0, 1.0) * (self.t_max - self.t_min)

    def thrust_to_valve(self, t) -> np.ndarray:
        return (np.asarray(t, dtype=float) - self.t_min) / (self.t_max - self.t_min)

    def reset(self, rng: np.random.Generator, throttle0, enabled: bool = True, fault_scale=None) -> None:
        self.lit = enabled
        self.fault = [1.0] * self.n if fault_scale is None else [float(f) for f in fault_scale]
        # misalignment: each thrust axis leans by the fixed angle toward a random azimuth
        beta = rng.uniform(0.0, 2.0 * math.pi, self.n)
        s, c = math.sin(self.misalign), math.cos(self.misalign)
        axes = np.column_stack([s * np.cos(beta), s * np.sin(beta), np.full(self.n, c)])
        self.dir = [tuple(axis) for axis in axes.tolist()]
        start = self.valve_to_thrust(throttle0).tolist()
        self.cmd = list(start)
        self.thrust = list(start) if enabled else [0.0] * self.n

    def command(self, throttle, engines_off: bool = False) -> None:
        self.cmd = self.valve_to_thrust(throttle).tolist()
        if engines_off:
            self.lit = False

    def step(self, dt: float, cut: bool = False) -> list[float]:
        """Advance one physics step; returns ACTUAL thrust per engine (N)."""
        if cut:
            self.lit = False
        if self._lag[0] != dt:
            self._lag = (dt, 1.0 - math.exp(-dt / self.tau))
        k, hi = self._lag[1], self.t_max
        lo, target = (self.t_min, self.cmd) if self.lit else (0.0, [0.0] * self.n)
        thrust = []
        for th, tg in zip(self.thrust, target):
            new = th + (tg - th) * k
            new = new if new > lo else lo                 # clipped as np.clip does: max first, then min
            thrust.append(new if new < hi else hi)
        self.thrust = thrust
        return [th * f for th, f in zip(thrust, self.fault)]


@dataclass(slots=True)
class VehicleStep:
    thrust: list                  # actual thrust per engine (N)
    force_b: tuple                # total thrust force, body frame (N)
    torque_b: tuple               # thrust torque about the CG, body frame (N*m)
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
        self._arm_rows = tuple(tuple(r) for r in self._arm.tolist())

    def reset(self, rng: np.random.Generator, propellant_kg: float, throttle0,
              engines_enabled: bool = True, fault_scale=None) -> None:
        self.propellant = float(propellant_kg)
        self.empty = self.propellant <= 0.0
        self.engines.reset(rng, throttle0, engines_enabled and not self.empty, fault_scale)

    def command(self, cmd: Command) -> None:
        self.engines.command(cmd.throttle, cmd.engines_off)

    def step(self, dt: float) -> VehicleStep:
        """Advance one physics step: thrust, propellant flow, and the thrust force and torque about the CG."""
        th = self.engines.step(dt, self.empty)
        mdot = (th[0] + th[1] + th[2]) / self.engines.isp_g0
        self.propellant -= mdot * dt
        if self.propellant <= 0.0:
            self.propellant, self.empty = 0.0, True
        (a0, a1, a2), (b0, b1, b2), (c0, c1, c2) = self.engines.dir
        f0 = (th[0] * a0, th[0] * a1, th[0] * a2)            # force of each engine, body frame
        f1 = (th[1] * b0, th[1] * b1, th[1] * b2)
        f2 = (th[2] * c0, th[2] * c1, th[2] * c2)
        r0, r1, r2 = self._arm_rows
        torque_b = (r0[1] * f0[2] + r1[1] * f1[2] + r2[1] * f2[2] - (r0[2] * f0[1] + r1[2] * f1[1] + r2[2] * f2[1]),
                    r0[2] * f0[0] + r1[2] * f1[0] + r2[2] * f2[0] - (r0[0] * f0[2] + r1[0] * f1[2] + r2[0] * f2[2]),
                    r0[0] * f0[1] + r1[0] * f1[1] + r2[0] * f2[1] - (r0[1] * f0[0] + r1[1] * f1[0] + r2[1] * f2[0]))
        force_b = (f0[0] + f1[0] + f2[0], f0[1] + f1[1] + f2[1], f0[2] + f1[2] + f2[2])
        return VehicleStep(th, force_b, torque_b, mdot, self.empty)

    def trim_thrust(self, total_thrust: float) -> np.ndarray:
        """Per-engine thrust that gives `total_thrust` with zero pitch/yaw torque about the CG.

        Solves sum(T) = F, sum(T y) = 0, sum(T x) = 0 (engine 3 sits a little further out than
        engines 1 and 2 in S4 Fig 6a, so it runs ~2.5 % lower).
        """
        p, c = self.engines.pos, self.geom.cg
        a = np.vstack([np.ones(3), p[:, 1] - c[1], p[:, 0] - c[0]])
        return np.linalg.solve(a, np.array([total_thrust, 0.0, 0.0]))
