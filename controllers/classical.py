"""The classical controller (M3): Viking's terminal-descent guidance plus steering to the target.

It flies the agent's 22 observations (brief D24): the onboard navigation estimate with Viking's radar errors and
blind zones, never the truth. Besides them it knows only fixed vehicle facts from the config (dry mass, engine
positions and thrust limits, Isp, inertias, Mars gravity); the fuel gauge gives the mass. It keeps a little state
of its own (the tilt command and the fuel-reserve latch); everything else is worked out afresh each step.

Viking's scheme (S3 p96-104), which this follows:
- Vertical: the engines idle at minimum thrust (Viking held 10 %, S3 p103) and the lander falls until it meets a
  speed-height contour; a speed loop then brakes along the contour to 2.44 m/s at 16.8 m, and that speed is held
  down to touchdown (`VIKING_SEQ_*`). The speed loop may command at most an average valve setting of 0.84; the
  travel above it is kept for attitude control (S3 p99).
- Attitude: pitch and yaw by differential throttle with rate damping (S3 p96-98); spin is free (no roll jets, D4).
- Touchdown: any leg switch shuts the engines down (TESS, S3 p104), here with the engines-off action.

Ours, because Viking had no target (D20), and because a 0-50 m offset with 0-10 m/s sideways needs time:
- Sideways, above the gate: distance -> speed command (capped) -> acceleration -> tilt command (capped and
  rate limited). Below the gate only the sideways speed is nulled, with a small and slowly changing tilt, so the
  first contact comes level and calm (tilt <= 5 deg, pitch/yaw rate <= 7 deg/s, S3 Table 8).
- The descent waits for the sideways move: above the gate the descent speed is at most the height left divided
  by the time the move needs (`side_time`), so the lander reaches the gate over the target.
- Fuel reserve: once the propellant left falls under (1 + CTRL_FUEL_MARGIN) x what a straight-down landing from
  there needs (`fuel_to_land`), the target is given up for the rest of the episode and the lander comes straight
  down, still nulling its sideways speed.

Gains and limits are the `CTRL_*` facts (assumed, tuned on M3 runs; docs/env_variables.md explains each).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from envs.mars_lander_env import OBS_NAMES
from lander.config import G0_MPS2, Config
from lander.geometry import build_geometry

_I = {name: k for k, name in enumerate(OBS_NAMES)}
_LEGS = (_I["leg_1"], _I["leg_2"], _I["leg_3"])
_CUT = np.array([-1.0, -1.0, -1.0, 1.0], dtype=np.float32)


@dataclass
class GuidanceState:
    """What the controller decided at its last step (for plots, tests and debugging)."""

    phase: str = "fall"                  # "fall" (minimum thrust), "brake", "terminal" (below the gate) or "down"
    descent_ref_mps: float = math.nan    # descent speed it was steering to
    waiting: bool = False                # the descent was slowed so the sideways move can finish above the gate
    reserve: bool = False                # the target was given up to keep enough propellant (latched)
    tilt_ref: tuple = (0.0, 0.0)         # tilt command, heading frame (sin of the lean)
    thrust_n: np.ndarray = field(default_factory=lambda: np.zeros(3))   # commanded thrust per engine


class ClassicalController:
    """Callable policy: `reset()` at each episode start, then `controller(obs)` -> action (4 numbers in [-1, 1])."""

    def __init__(self, cfg: Config):
        e, m, sq, c = cfg.engine, cfg.mass, cfg.seq, cfg.ctrl
        self.cfg = cfg
        geom = build_geometry(cfg)
        arm = geom.engine_pos - geom.cg
        # [total thrust, torque about body x, torque about body y] = A @ engine thrusts (thrust along body +z)
        self._a_inv = np.linalg.inv(np.array([np.ones(3), arm[:, 1], -arm[:, 0]]))
        self._unit = self._a_inv @ np.array([1.0, 0.0, 0.0])     # torque-free thrust split per newton
        self.t_min, self.t_max = e.thrust_min_n, e.thrust_max_n
        self.isp_g0 = e.isp_s * G0_MPS2
        self.dry, self.g = m.dry_kg, cfg.mars.gravity_mps2
        self.ixx, self.iyy = m.ixx_kgm2, m.iyy_kgm2
        self.dt = cfg.sim.control_dt_s
        self.v_gate, self.h_gate = sq.terminal_speed_mps, sq.terminal_alt_m
        self.f_axial = 3.0 * (e.thrust_min_n + sq.axial_throttle_limit * (e.thrust_max_n - e.thrust_min_n))
        self.c = c
        self.reset()

    def reset(self) -> None:
        """Forget the last episode (tilt command, reserve latch)."""
        self.state = GuidanceState()
        self._tilt_ref: np.ndarray | None = None

    # ------------------------------------------------------------------------------------- building blocks
    def contour(self, height: float) -> float:
        """Descent speed (m/s) of the speed-height contour: braking at CTRL_BRAKE_MPS2 to the gate, then constant."""
        above = max(0.0, height - self.h_gate)
        return math.sqrt(self.v_gate ** 2 + 2.0 * self.c.brake_mps2 * above)

    def side_time(self, pos, vel) -> float:
        """Seconds the sideways move needs: stop the current sideways motion, fly to the target (accelerate,
        cruise at CTRL_SIDE_SPEED_MPS, brake) and settle, all at CTRL_SIDE_ACCEL_MPS2."""
        c = self.c
        a, cruise = c.side_accel_mps2, c.side_speed_mps
        speed = math.hypot(vel[0], vel[1])
        stop = (pos[0] + vel[0] * speed / (2.0 * a), pos[1] + vel[1] * speed / (2.0 * a))   # where it would stop
        d = math.hypot(*stop)
        fly = 2.0 * math.sqrt(d / a) if d < cruise * cruise / a else d / cruise + cruise / a
        return speed / a + fly + c.side_settle_s

    def fuel_to_land(self, mass: float, height: float, v_down: float) -> float:
        """Propellant (kg) a straight-down landing from here needs: fall at minimum thrust until the contour,
        brake along it to the gate, then hold the gate speed to the ground."""
        a_fall = self.g - 3.0 * self.t_min / mass              # downward acceleration at minimum thrust
        a_brake = self.c.brake_mps2
        v = max(v_down, 0.0)
        if v >= self.contour(height):
            v_meet, t_fall = v, 0.0
        else:   # where the fall meets the contour: v^2 + 2 a_fall (h - x) = v_gate^2 + 2 a_brake (x - h_gate)
            x = (v * v + 2.0 * a_fall * height - self.v_gate ** 2 + 2.0 * a_brake * self.h_gate) / (2.0 * (a_fall + a_brake))
            v_meet = self.contour(min(max(x, self.h_gate), height))
            t_fall = max(0.0, (v_meet - v) / a_fall) if a_fall > 0 else 0.0
        impulse = 3.0 * self.t_min * t_fall
        impulse += mass * (self.g + a_brake) * max(0.0, v_meet - self.v_gate) / a_brake
        impulse += mass * self.g * min(max(height, 0.0), self.h_gate) / self.v_gate
        return impulse / self.isp_g0

    def mix(self, total: float, torque_x: float, torque_y: float) -> np.ndarray:
        """Engine thrusts (N) for a total thrust and pitch/yaw torques, each within its limits.

        Attitude comes first: if no split gives both, the total thrust gives way, and only if even that cannot
        make room are the torques scaled down.
        """
        unit, diff = self._unit, self._a_inv @ np.array([0.0, torque_x, torque_y])
        scale = 1.0
        for i in range(3):          # largest torque scale that leaves some total thrust feasible
            for j in range(3):
                slope = diff[j] / unit[j] - diff[i] / unit[i]
                if slope > 1e-12:
                    scale = min(scale, (self.t_max / unit[j] - self.t_min / unit[i]) / slope)
        scale = max(0.0, scale)
        low = max((self.t_min - scale * diff[i]) / unit[i] for i in range(3))
        high = min((self.t_max - scale * diff[i]) / unit[i] for i in range(3))
        return min(max(total, low), high) * unit + scale * diff

    # ------------------------------------------------------------------------------------- one step
    def __call__(self, obs) -> np.ndarray:
        o = np.asarray(obs, dtype=np.float64)
        st, c = self.state, self.c
        if o[_I["engines_off"]] > 0.5 or any(o[k] > 0.5 for k in _LEGS):
            st.phase = "down"
            return _CUT.copy()
        mass = self.dry + max(0.0, o[_I["propellant_kg"]])
        h, v_down = o[_I["height"]], -o[_I["vel_z"]]
        pos, vel = (o[_I["pos_x"]], o[_I["pos_y"]]), (o[_I["vel_x"]], o[_I["vel_y"]])
        sx, sy = o[_I["tilt_x"]], o[_I["tilt_y"]]
        above = max(0.0, h - self.h_gate)

        # ---- vertical: contour, slowed while the sideways move needs time
        if not st.reserve and o[_I["propellant_kg"]] < (1.0 + c.fuel_margin) * self.fuel_to_land(mass, h, v_down):
            st.reserve = True
        v_ref = self.contour(h)
        a_ff = c.brake_mps2 if above > 0.0 else 0.0      # the contour's own deceleration, fed forward
        st.waiting = False
        if above > 0.0 and not st.reserve:
            t_side = self.side_time(pos, vel)
            v_wait = max(self.v_gate, above / t_side)
            if v_wait < v_ref:
                v_ref, a_ff, st.waiting = v_wait, (v_wait / t_side if v_wait > self.v_gate else 0.0), True
        a_up = a_ff + c.descent_gain_per_s * (v_down - v_ref)
        cos_tilt = math.sqrt(max(0.0, 1.0 - sx * sx - sy * sy))
        total = mass * (self.g + a_up) / max(cos_tilt, math.cos(math.radians(c.tilt_max_deg[0])))
        total = min(max(total, 3.0 * self.t_min), self.f_axial)

        # ---- sideways: distance -> speed -> acceleration -> tilt command (heading frame)
        cap = 0.0 if (st.reserve or above <= 0.0) else c.side_speed_mps
        want = np.array([-c.position_gain_per_s * pos[0], -c.position_gain_per_s * pos[1]])
        n = float(np.hypot(*want))
        if n > cap:
            want *= cap / n
        accel = c.side_gain_per_s * (want - np.array(vel))
        tilt = accel * mass / total
        frac = min(1.0, max(0.0, h / self.h_gate))
        lim = math.sin(math.radians(c.tilt_max_deg[1] + (c.tilt_max_deg[0] - c.tilt_max_deg[1]) * frac))
        n = float(np.hypot(*tilt))
        if n > lim:
            tilt *= lim / n
        spin = o[_I["rate_z"]]
        if self._tilt_ref is None:
            self._tilt_ref = np.array([sx, sy])
        else:   # the heading frame turned with the spin since the last step
            ca, sa = math.cos(spin * self.dt), math.sin(spin * self.dt)
            r = self._tilt_ref
            self._tilt_ref = np.array([ca * r[0] + sa * r[1], -sa * r[0] + ca * r[1]])
        step = math.radians(c.tilt_rate_dps[0] if above > 0.0 else c.tilt_rate_dps[1]) * self.dt
        delta = tilt - self._tilt_ref
        n = float(np.hypot(*delta))
        if n > step:
            delta *= step / n
        self._tilt_ref = self._tilt_ref + delta

        # ---- attitude: PD on the tilt -> body angular accelerations -> torques -> engine thrusts
        wn, zeta = c.attitude_radps, c.attitude_damping
        wx, wy = o[_I["rate_x"]], o[_I["rate_y"]]
        dsx, dsy = wy + spin * sy, -wx - spin * sx           # how fast the tilt moves in the heading frame
        ddx = wn * wn * (self._tilt_ref[0] - sx) - 2.0 * zeta * wn * dsx
        ddy = wn * wn * (self._tilt_ref[1] - sy) - 2.0 * zeta * wn * dsy
        thrust = self.mix(total, -self.ixx * ddy, self.iyy * ddx)

        st.phase = "terminal" if above <= 0.0 else ("fall" if total <= 3.0 * self.t_min + 1e-6 else "brake")
        st.descent_ref_mps, st.tilt_ref, st.thrust_n = v_ref, (float(self._tilt_ref[0]), float(self._tilt_ref[1])), thrust
        valve = np.clip((thrust - self.t_min) / (self.t_max - self.t_min), 0.0, 1.0)
        return np.concatenate([2.0 * valve - 1.0, [-1.0]]).astype(np.float32)
