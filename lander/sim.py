"""LanderSim: ties the backend-agnostic vehicle layer to MuJoCo (D1).

Every physics step (SIM_PHYSICS_DT_S): read the state -> engines and propellant -> drag in the
episode's steady wind -> apply forces (`xfrc_applied`) -> update the mass -> `mj_step` -> contacts
and the touchdown monitor. Every control step (SIM_CONTROL_HZ) the command is latched (zero-order
hold) and the sensors, the onboard navigation estimate, telemetry and events are produced.

Simplified vehicle (brief D15): one rigid body (rigid legs), CG fixed on the centre line, constant
inertia, mass = dry + propellant. The agent shuts the engines down itself (`Command.engines_off`,
permanent, brief D7).

Determinism (D10): all randomness comes from numpy generators spawned from one seed in a fixed
order, and MuJoCo is deterministic, so the same seed + config + commands on the same machine and
MuJoCo version give a bit-identical episode.

Speed (2026-10-11): the physics step works in plain floats (`rotation_rows`, `mat_vec`), not small numpy
arrays, with every operation rounded as before, so the speed-up changed no result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from typing import Any, Callable

import mujoco
import numpy as np

from .aero import drag_force, wind_vector
from .config import Config, load_config
from .geometry import build_geometry, cross3, mat_vec
from .mjcf import build_model, set_terrain, terrain_normal
from .navigation import NavEstimate, Navigator
from .sensors import SensorReadings, SensorSuite
from .touchdown import TouchdownMonitor, TouchdownReport, stability_distance
from .vehicle import Command, Vehicle

_TERRAIN_GROUP = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)


# ------------------------------------------------------------------------------------------ math
def rotation_rows(w, x, y, z) -> tuple:
    """Rows of the rotation matrix (body -> world) of the unit quaternion (w, x, y, z), as plain numbers."""
    return ((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
            (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
            (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)))


def quat_to_mat(q) -> np.ndarray:
    """Rotation matrix (body -> world) from a unit quaternion (w, x, y, z)."""
    return np.array(rotation_rows(*q))


def quat_mul(a, b) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw])


def quat_from_axis_angle(axis, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    return np.concatenate([[math.cos(angle / 2)], math.sin(angle / 2) * axis])


def quat_from_tilt(tilt_deg: float, tilt_azimuth_deg: float = 0.0, heading_deg: float = 0.0) -> np.ndarray:
    """Attitude: rotate by `heading` about +Z, then lean the thrust axis `tilt` toward `tilt_azimuth`."""
    a = math.radians(tilt_azimuth_deg)
    q_head = quat_from_axis_angle([0.0, 0.0, 1.0], math.radians(heading_deg))
    q_tilt = quat_from_axis_angle([-math.sin(a), math.cos(a), 0.0], math.radians(tilt_deg))
    return quat_mul(q_tilt, q_head)


def tilt_of(rot: np.ndarray) -> float:
    """Angle (deg) between the body +Z (thrust) axis and world +Z."""
    return math.degrees(math.acos(max(-1.0, min(1.0, float(rot[2, 2])))))


def plane_height(n: np.ndarray, x: float, y: float) -> float:
    """Height of the ground plane through the origin with unit normal n, at (x, y)."""
    return -(n[0] * x + n[1] * y) / n[2]


# ------------------------------------------------------------------------------------------ data
@dataclass
class InitialState:
    """Full initial state (brief section 9). Position/velocity are of the vehicle CG, world frame."""

    position: tuple = (0.0, 0.0, 30.8)
    velocity: tuple = (0.0, 0.0, 0.0)
    quat: tuple = (1.0, 0.0, 0.0, 0.0)          # body -> world (w, x, y, z)
    omega: tuple = (0.0, 0.0, 0.0)              # body-frame angular velocity (rad/s)
    propellant_kg: float | None = None          # None = VIKING_MASS_START_PROPELLANT_KG
    throttle: tuple | None = None               # starting valve settings; None = hover trim


@dataclass
class ResetOptions:
    """Per-episode environment conditions and faults (None = use the .env value)."""

    wind_speed_mps: float = 0.0                      # steady wind; 0 = calm (storms: MARS_WIND_STORM_MPS)
    wind_direction_deg: float = 0.0                  # direction the wind blows TOWARD, from +X toward +Y
    slope_deg: float | None = None
    slope_direction_deg: float | None = None
    friction: float | None = None
    disturbance_force_n: tuple = (0.0, 0.0, 0.0)     # constant, world frame
    disturbance_torque_nm: tuple = (0.0, 0.0, 0.0)   # constant, body frame
    engine_thrust_scale: tuple = (1.0, 1.0, 1.0)     # 1 = nominal, 0 = engine failed
    engines_enabled: bool = True                     # False = start with the engines already shut down
    sensor_noise_scale: float | None = None          # None = SIM_SENSOR_NOISE_SCALE

    @classmethod
    def from_any(cls, options: Any) -> "ResetOptions":
        if options is None:
            return cls()
        if isinstance(options, cls):
            return options
        valid = {f.name for f in fields(cls)}
        unknown = set(options) - valid
        if unknown:
            raise ValueError(f"unknown reset options {sorted(unknown)}; valid: {sorted(valid)}")
        return cls(**options)


@dataclass
class Telemetry:
    """Ground-truth snapshot at the end of a control step (for HUD, logs and debugging)."""

    t: float
    cg: np.ndarray
    vel: np.ndarray
    quat: np.ndarray
    omega_b: np.ndarray
    tilt_deg: float
    height_cg: float               # CG height above the ground directly below it
    pad_altitude: float            # lowest footpad above the ground (0 at contact)
    propellant_kg: float
    mass_kg: float
    thrust_n: np.ndarray
    throttle_cmd: np.ndarray
    engines_off: bool              # engines shut down (by command or an empty tank)
    wind: np.ndarray
    pad_contact: tuple             # each footpad touching the ground now
    body_contact: bool
    pads_touched: tuple = (False, False, False)   # each footpad has touched the ground (latched leg switch)

    @property
    def vz(self) -> float:
        return float(self.vel[2])

    @property
    def vh(self) -> float:
        return float(math.hypot(self.vel[0], self.vel[1]))

    @property
    def spin_dps(self) -> float:
        """Rotation rate about the lander's own vertical (thrust) axis, deg/s (uncontrolled, D15)."""
        return math.degrees(float(self.omega_b[2]))


@dataclass
class StepResult:
    telemetry: Telemetry
    sensors: SensorReadings
    report: TouchdownReport
    events: list[str] = field(default_factory=list)
    nav: NavEstimate | None = None   # what the lander knows (the agent's view); telemetry is the truth


class PlaneTerrain:
    """The ground: a plane through the origin with unit normal `n` (queries for the sensor models)."""

    def __init__(self, sim: "LanderSim"):
        self._sim = sim
        self.n = np.array([0.0, 0.0, 1.0])
        self._geomid = np.zeros((1, 1), dtype=np.int32)

    def height(self, x: float, y: float) -> float:
        return plane_height(self.n, x, y)

    def height_above(self, p: np.ndarray) -> float:
        """Vertical height of point p above the ground."""
        return float(p[2] - self.height(p[0], p[1]))

    def nearest(self, p: np.ndarray) -> tuple[float, np.ndarray]:
        """Distance to the closest ground point and the unit vector toward it."""
        return float(self.n @ p), -self.n

    def raycast(self, origin: np.ndarray, direction: np.ndarray) -> float:
        s = self._sim
        dist = mujoco.mj_ray(s.model, s.data, np.asarray(origin, dtype=float),
                             np.asarray(direction, dtype=float), _TERRAIN_GROUP, 1, -1, self._geomid)
        return math.inf if dist < 0 else float(dist)


# ------------------------------------------------------------------------------------------ sim
class LanderSim:
    """The Viking lander on Mars. `reset()` then `step(Command)` at SIM_CONTROL_HZ."""

    def __init__(self, cfg: Config | None = None, seed: int = 0):
        self.cfg = cfg if cfg is not None else load_config()
        self.geom = build_geometry(self.cfg)
        self.model, self.idx = build_model(self.cfg, self.geom)
        self.data = mujoco.MjData(self.model)
        self.vehicle = Vehicle(self.cfg, self.geom)
        self.sensors = SensorSuite(self.cfg, self.geom)
        self.navigator = Navigator(self.geom)
        self.monitor = TouchdownMonitor(self.cfg)
        self.terrain = PlaneTerrain(self)
        self.dt = self.cfg.sim.physics_dt_s
        self.substeps = self.cfg.sim.substeps
        self.control_dt = self.cfg.sim.control_dt_s
        self._cda = self.cfg.aero.cda_m2
        self._rho = self.cfg.mars.air_density_kgpm3
        self._cg_b = self.geom.cg
        self._cg_t = tuple(self.geom.cg.tolist())
        self._inertia = self.vehicle.mass.inertia
        self._pads = set(self.idx.pad_geoms)
        self._body_geoms = set(self.idx.body_geoms)
        # MuJoCo arrays the physics step touches (views into the model and data; they never move)
        self._qpos, self._qvel, self._xfrc = self.data.qpos, self.data.qvel, self.data.xfrc_applied
        self._body_mass, self._subtree_mass = self.model.body_mass, self.model.body_subtreemass
        self.seed = seed
        self.reset(seed=seed)

    # ---------------------------------------------------------------- helpers
    @property
    def t(self) -> float:
        return float(self.data.time)

    @property
    def engines_off(self) -> bool:
        """True once the engines are shut down (engines-off command or an empty tank); permanent."""
        return not self.vehicle.engines.lit

    @property
    def last_thrust(self) -> np.ndarray:
        """Actual thrust of each engine (N) in the last physics step."""
        return np.array(self._thrust_now)

    def _apply_mass(self, propellant: float) -> None:
        """Mass = dry + propellant; the CG and inertia never change (D15)."""
        total = self.vehicle.mass.total(propellant)
        b = self.idx.lander_body
        self._body_mass[b] = total
        self._subtree_mass[b] = total
        self._subtree_mass[0] = total
        self._mass = total

    def hover_trim(self, propellant: float | None = None, total_thrust: float | None = None) -> tuple:
        """Valve settings that hold `total_thrust` (default: the current weight) with zero pitch/yaw torque."""
        prop = self.vehicle.propellant if propellant is None else propellant
        f = self.vehicle.mass.total(prop) * self.cfg.mars.gravity_mps2 if total_thrust is None else total_thrust
        thrust = self.vehicle.trim_thrust(f)
        return tuple(float(v) for v in self.vehicle.engines.thrust_to_valve(thrust))

    def state_above_ground(self, pad_altitude: float, velocity=(0.0, 0.0, 0.0), tilt_deg: float = 0.0,
                           tilt_azimuth_deg: float = 0.0, heading_deg: float = 0.0, omega=(0.0, 0.0, 0.0),
                           xy=(0.0, 0.0), propellant_kg: float | None = None, throttle: tuple | None = None,
                           slope_deg: float | None = None, slope_direction_deg: float | None = None) -> InitialState:
        """InitialState with the lowest footpad `pad_altitude` m (vertically) above the ground at `xy`.

        The ground is the .env slope unless `slope_deg` / `slope_direction_deg` are given (pass the same
        values as the reset options of that episode).
        """
        prop = self.cfg.mass.start_propellant_kg if propellant_kg is None else propellant_kg
        n = terrain_normal(self.cfg.terrain.slope_deg if slope_deg is None else slope_deg,
                           self.cfg.terrain.slope_direction_deg if slope_direction_deg is None else slope_direction_deg)
        q = quat_from_tilt(tilt_deg, tilt_azimuth_deg, heading_deg)
        rot = quat_to_mat(q)
        r = self.geom.footpad_radius
        pads = (rot @ self.geom.footpads.T).T
        lowest = min(float(p[2]) - r - plane_height(n, xy[0] + p[0], xy[1] + p[1]) for p in pads)
        origin = np.array([xy[0], xy[1], pad_altitude - lowest])
        cg = origin + rot @ self._cg_b
        return InitialState(position=tuple(cg), velocity=tuple(velocity), quat=tuple(q), omega=tuple(omega),
                            propellant_kg=prop, throttle=throttle)

    # ---------------------------------------------------------------- reset / step
    def reset(self, seed: int | None = None, initial_state: InitialState | None = None,
              options: Any = None) -> StepResult:
        """Start a new episode. `options` = ResetOptions or a dict of its fields."""
        cfg, m, d, idx = self.cfg, self.model, self.data, self.idx
        if seed is not None:
            self.seed = int(seed)
        opts = ResetOptions.from_any(options)
        self.options = opts
        rng_eng, rng_sens = (np.random.default_rng(s) for s in np.random.SeedSequence(self.seed).spawn(2))

        slope = cfg.terrain.slope_deg if opts.slope_deg is None else opts.slope_deg
        downhill = cfg.terrain.slope_direction_deg if opts.slope_direction_deg is None else opts.slope_direction_deg
        friction = cfg.terrain.friction if opts.friction is None else opts.friction
        mujoco.mj_resetData(m, d)
        self.terrain.n = set_terrain(m, d, idx, slope, downhill, friction)
        self.slope_deg, self.slope_direction_deg, self.friction = slope, downhill, friction
        st = initial_state if initial_state is not None else self.state_above_ground(30.0)
        prop = cfg.mass.start_propellant_kg if st.propellant_kg is None else float(st.propellant_kg)
        self._apply_mass(prop)

        q = np.asarray(st.quat, dtype=float)
        q = q / np.linalg.norm(q)
        rot = quat_to_mat(q)
        omega_b = np.asarray(st.omega, dtype=float)
        rc = rot @ self._cg_b
        d.qpos[0:3] = np.asarray(st.position, dtype=float) - rc
        d.qpos[3:7] = q
        d.qvel[0:3] = np.asarray(st.velocity, dtype=float) - cross3(rot @ omega_b, rc)
        d.qvel[3:6] = omega_b

        throttle0 = st.throttle if st.throttle is not None else self.hover_trim(prop)
        self.vehicle.reset(rng_eng, prop, throttle0, opts.engines_enabled, opts.engine_thrust_scale)
        self.command = Command(tuple(throttle0))
        self.wind = wind_vector(opts.wind_speed_mps, opts.wind_direction_deg)
        self.sensors.reset(rng_sens, opts.sensor_noise_scale)
        self.monitor.reset()
        self.monitor.note_engines(self.engines_off)
        self._wind_t = tuple(self.wind.tolist())
        self._dist_force = tuple(float(f) for f in opts.disturbance_force_n)
        self._dist_torque = tuple(float(f) for f in opts.disturbance_torque_nm)
        self._thrust_now = tuple(self.vehicle.engines.thrust)
        self._pad_contact = (False, False, False)
        self._pads_touched = (False, False, False)
        self._body_contact = False
        self._flags = {"cut": self.engines_off, "empty": self.vehicle.empty}
        d.xfrc_applied[:] = 0.0
        mujoco.mj_forward(m, d)
        return self._result([], reset=True)

    def step(self, command: Command | None = None,
             substep_callback: Callable[["LanderSim", int], None] | None = None) -> StepResult:
        """Advance one control period with `command` held (zero-order hold)."""
        if command is not None:
            self.command = command
        self.vehicle.command(self.command)
        events: list[str] = []
        for k in range(self.substeps):
            self._physics_step(events)
            if substep_callback is not None:
                substep_callback(self, k)
        mujoco.mj_forward(self.model, self.data)
        return self._result(events)

    def _physics_step(self, events: list[str]) -> None:
        """One physics step in plain floats (see the module docstring); the touchdown checks use arrays."""
        d, idx = self.data, self.idx
        t, dt = float(d.time), self.dt
        px, py, pz, qw, qx, qy, qz = self._qpos.tolist()
        vx, vy, vz, wx, wy, wz = self._qvel.tolist()
        rot = rotation_rows(qw, qx, qy, qz)
        rc = mat_vec(rot, self._cg_t)                                # CG offset from the body origin, world frame
        w = mat_vec(rot, (wx, wy, wz))                               # body rates, world frame
        v_cg = (vx + (w[1] * rc[2] - w[2] * rc[1]), vy + (w[2] * rc[0] - w[0] * rc[2]),
                vz + (w[0] * rc[1] - w[1] * rc[0]))

        vs = self.vehicle.step(dt)                                   # engines and propellant
        wind = self._wind_t                                          # drag: world frame, acts at the CG
        f_aero = drag_force((v_cg[0] - wind[0], v_cg[1] - wind[1], v_cg[2] - wind[2]), self._rho, self._cda)

        f, df = mat_vec(rot, vs.force_b), self._dist_force
        tb, dq = vs.torque_b, self._dist_torque
        torque = mat_vec(rot, (tb[0] + dq[0], tb[1] + dq[1], tb[2] + dq[2]))
        self._xfrc[idx.lander_body] = (f[0] + f_aero[0] + df[0], f[1] + f_aero[1] + df[1], f[2] + f_aero[2] + df[2],
                                       torque[0], torque[1], torque[2])
        self._apply_mass(self.vehicle.propellant)
        mujoco.mj_step(self.model, d)

        self._thrust_now = vs.thrust
        self._contacts()
        self._flag_events(events)
        self.monitor.note_engines(self.engines_off)
        if self.monitor.report.done:
            return
        near = self._body_contact or any(self._pad_contact) or self.monitor.report.touched_down
        if not near and not self.vehicle.empty:
            return
        cg = np.array((px + rc[0], py + rc[1], pz + rc[2]))
        v_cg, omega_b, rot = np.array(v_cg), np.array((wx, wy, wz)), np.array(rot)
        ke = 0.5 * self._mass * float(v_cg @ v_cg) + 0.5 * float(self._inertia @ (omega_b * omega_b))
        stab = stability_distance(cg, d.geom_xpos[list(idx.pad_geoms)])
        before = (self.monitor.report.touched_down, self.monitor.report.settled, self.monitor.report.tipped_over)
        self.monitor.update(t, self._pad_contact, self._body_contact, v_cg, ke, stab, tilt_of(rot),
                            self.vehicle.empty, cg, omega_b)
        r = self.monitor.report
        if r.touched_down and not before[0]:
            events.append("touchdown")
            if r.leg_broken:
                events.append("leg_broken")
        if r.settled and not before[1]:
            events.append("settled")
        if r.tipped_over and not before[2]:
            events.append("tipped_over")

    def _contacts(self) -> None:
        d, terrain = self.data, self.idx.terrain_geom
        if d.ncon == 0:                       # in flight
            self._pad_contact, self._body_contact = (False, False, False), False
            return
        pads = [False, False, False]
        body = False
        for g1, g2 in d.contact.geom[: d.ncon]:
            other = g2 if g1 == terrain else g1 if g2 == terrain else -1
            if other in self._pads:
                pads[self.idx.pad_geoms.index(other)] = True
            elif other in self._body_geoms:
                body = True
        self._pad_contact, self._body_contact = tuple(pads), body
        self._pads_touched = tuple(a or b for a, b in zip(self._pads_touched, pads))

    def _flag_events(self, events: list[str]) -> None:
        f = self._flags
        if not f["cut"] and self.engines_off:
            f["cut"] = True
            events.append("engines_cut")
        if not f["empty"] and self.vehicle.empty:
            f["empty"] = True
            events.append("propellant_exhausted")
        if self._body_contact and "body_strike" not in f:
            f["body_strike"] = True
            events.append("body_strike")

    # ---------------------------------------------------------------- outputs
    def telemetry(self) -> Telemetry:
        d, idx = self.data, self.idx
        q = d.qpos[3:7].copy()
        rot = quat_to_mat(q)
        omega_b = d.qvel[3:6].copy()
        rc = rot @ self._cg_b
        cg = d.qpos[0:3] + rc
        vel = d.qvel[0:3] + cross3(rot @ omega_b, rc)
        r = self.geom.footpad_radius
        pad_alt = min(self.terrain.height_above(p) - r for p in d.geom_xpos[list(idx.pad_geoms)])
        return Telemetry(
            t=self.t, cg=cg, vel=vel, quat=q, omega_b=omega_b, tilt_deg=tilt_of(rot),
            height_cg=self.terrain.height_above(cg), pad_altitude=float(pad_alt),
            propellant_kg=self.vehicle.propellant, mass_kg=self._mass, thrust_n=self.last_thrust,
            throttle_cmd=np.clip(np.asarray(self.command.throttle, dtype=float), 0.0, 1.0),
            engines_off=self.engines_off, wind=self.wind.copy(), pad_contact=self._pad_contact,
            body_contact=self._body_contact, pads_touched=self._pads_touched)

    def read_sensors(self) -> SensorReadings:
        d, idx = self.data, self.idx
        rot = quat_to_mat(d.qpos[3:7])
        omega_b = d.qvel[3:6].copy()
        return self.sensors.sample(
            self.t, d.sensordata[idx.imu_acc_adr:idx.imu_acc_adr + 3].copy(),
            d.sensordata[idx.imu_gyro_adr:idx.imu_gyro_adr + 3].copy(),
            d.qpos[0:3].copy(), rot, d.qvel[0:3].copy(), rot @ omega_b, self.terrain)

    def _result(self, events: list[str], reset: bool = False) -> StepResult:
        tl, sn = self.telemetry(), self.read_sensors()
        rot = quat_to_mat(tl.quat)
        if reset:      # navigation starts from the handed-over state: the antennas' true height, minus their mount
            antenna = tl.cg + rot @ (self.geom.radar_pos - self._cg_b)
            height = self.terrain.nearest(antenna)[0] - self.navigator.antenna_height
            nav = self.navigator.reset(tl.t, tl.cg, tl.vel, height, sn, rot, tl.omega_b)
        else:
            nav = self.navigator.update(tl.t, tl.cg, tl.vel, sn, rot, tl.omega_b)
        return StepResult(tl, sn, self.monitor.report, events, nav)
