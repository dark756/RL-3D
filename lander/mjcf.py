"""Build the MuJoCo model (MJCF XML) from the config, so editing a geometry fact in .env changes the model.

MuJoCo only integrates rigid-body motion, resolves contacts and renders (D1): its fluid model is off
and every engine and aero force is applied from Python by `sim.py`. The lander is ONE free body
(brief D15): rigid legs end in three footpad contact spheres, the CG sits on the centre line, and
only the mass changes at run time as propellant burns.

The sizes below only shape the drawing (and the hull that touches the ground after a tip-over);
they do not change the flight physics, so they are constants here rather than facts in .env.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from .config import Config
from .geometry import BODY_BOTTOM_Z_M, LanderGeometry, build_geometry, hexagon_vertices, unit_azimuth
from .vehicle import MassModel

GROUP_TERRAIN, GROUP_LANDER, GROUP_DECOR, GROUP_COLLISION = 0, 1, 2, 3
TARGET_RING_M = 10.0          # radius of the decorative marker ring around the target (visual only)

BODY_LONG_SIDE_M = 1.09       # S2: hexagonal body; engines on the long sides
BODY_SHORT_SIDE_M = 0.56      # S2: legs attach on the short sides
BODY_HEIGHT_M = 0.46          # assumed: commonly quoted NSSDCA value
RADAR_HOUSING_M = (0.40, 0.0889)        # Doppler radar housing: width (assumed), drop below the underside (S5 Fig 8)
TANKS = ((30.0, 150.0), 0.62, 1.10, 0.579)   # azimuths (S3 p36), radius and height (assumed), diameter (S3 p140)
STRUT_TOP = (0.9398, 1.2438)            # primary-strut upper attachment: radius, height (S4 Fig 3)
BIPOD = (0.8382, 0.4247, 0.2992)        # bipod attachments: radius, height, half span (S4 Fig 3)
FOOTPAD_DIAMETER_M = 0.451              # S5 Fig 8: conical footpad


@dataclass(frozen=True)
class ModelIndex:
    """MuJoCo ids/addresses the simulator needs every step."""

    lander_body: int
    pad_geoms: tuple[int, ...]
    body_geoms: tuple[int, ...]
    terrain_geom: int
    ground_mocap: int
    engine_sites: tuple[int, ...]
    imu_acc_adr: int
    imu_gyro_adr: int


def _v(*vals) -> str:
    flat = []
    for v in vals:
        flat.extend(np.asarray(v, dtype=float).ravel().tolist())
    return " ".join(f"{x:.6g}" for x in flat)


def _zquat(az: float) -> str:
    return _v(math.cos(az / 2), 0.0, 0.0, math.sin(az / 2))


def terrain_normal(slope_deg: float, downhill_deg: float) -> np.ndarray:
    """Unit normal of a plane through the origin that falls toward the azimuth `downhill_deg`."""
    s, a = math.radians(slope_deg), math.radians(downhill_deg)
    return np.array([math.sin(s) * math.cos(a), math.sin(s) * math.sin(a), math.cos(s)])


def quat_z_to(n: np.ndarray) -> np.ndarray:
    """Quaternion (w, x, y, z) rotating +Z onto unit vector n."""
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z, n)
    s = np.linalg.norm(axis)
    if s < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    ang = math.atan2(s, float(z @ n))
    axis /= s
    return np.concatenate([[math.cos(ang / 2)], math.sin(ang / 2) * axis])


def build_mjcf(cfg: Config, geom: LanderGeometry | None = None) -> str:
    """Return the MJCF XML string for the lander + flat (later re-tilted) terrain."""
    geom = geom or build_geometry(cfg)
    mass = MassModel(cfg, geom)
    m_start = mass.total(cfg.mass.start_propellant_kg)
    tc = cfg.sim.contact_timeconst_s
    bottom, top = BODY_BOTTOM_Z_M, BODY_BOTTOM_Z_M + BODY_HEIGHT_M
    hexagon = hexagon_vertices(BODY_LONG_SIDE_M, BODY_SHORT_SIDE_M, list(geom.leg_azimuth_rad))
    hex_verts = [(x, y, z) for z in (bottom, top) for x, y in hexagon]
    housing_w, housing_drop = RADAR_HOUSING_M
    housing_c = np.array([0.0, 0.0, bottom - 0.5 * housing_drop])
    housing_h = np.array([0.5 * housing_w, 0.5 * housing_w, 0.5 * housing_drop])
    r_pad = geom.footpad_radius
    ring = "\n".join(
        f'      <geom class="decor" type="box" size="0.6 0.12 0.005" '
        f'pos="{_v(TARGET_RING_M * math.cos(a), TARGET_RING_M * math.sin(a), 0.01)}" '
        f'quat="{_zquat(a + math.pi / 2)}" rgba="0.96 0.84 0.30 0.9"/>'
        for a in np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False))

    x = []
    x.append(f"""<mujoco model="viking_lander">
  <compiler angle="radian" autolimits="true" inertiafromgeom="false"/>
  <option timestep="{cfg.sim.physics_dt_s}" gravity="0 0 {-cfg.mars.gravity_mps2}" integrator="implicitfast"
          cone="elliptic" density="0" viscosity="0" wind="0 0 0"/>
  <statistic extent="10" center="0 0 1"/>
  <visual>
    <global offwidth="1920" offheight="1080" fovy="40" azimuth="135" elevation="-12"/>
    <map znear="0.004" zfar="600" haze="0.6" shadowclip="3" shadowscale="0.5"/>
    <quality shadowsize="4096" offsamples="4"/>
    <headlight ambient="0.34 0.30 0.27" diffuse="0.45 0.41 0.37" specular="0.05 0.05 0.05"/>
    <rgba haze="0.86 0.66 0.50 1" fog="0.86 0.66 0.50 1"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.93 0.76 0.58" rgb2="0.38 0.26 0.20" width="512" height="512"/>
    <texture name="ground" type="2d" builtin="checker" rgb1="0.66 0.40 0.25" rgb2="0.62 0.37 0.23"
             mark="edge" markrgb="0.55 0.33 0.20" width="512" height="512"/>
    <material name="ground" texture="ground" texrepeat="0.1 0.1" texuniform="true" reflectance="0"/>
    <material name="hull" rgba="0.80 0.80 0.78 1" specular="0.3" shininess="0.3"/>
    <material name="gold" rgba="0.85 0.68 0.30 1" specular="0.6" shininess="0.6"/>
    <material name="tank" rgba="0.62 0.64 0.66 1" specular="0.7" shininess="0.7"/>
    <material name="dark" rgba="0.18 0.18 0.19 1"/>
    <material name="strut" rgba="0.72 0.72 0.74 1" specular="0.5"/>
    <material name="pad" rgba="0.55 0.55 0.57 1"/>
    <material name="white" rgba="0.95 0.95 0.95 1"/>
    <mesh name="hexbody" vertex="{_v(hex_verts)}"/>
  </asset>
  <default>
    <geom solref="{tc} 1" solimp="0.95 0.99 0.001" friction="0.01 0.005 0.0001" condim="3"/>
    <default class="vis"><geom contype="0" conaffinity="0" group="{GROUP_LANDER}"/></default>
    <default class="col"><geom contype="2" conaffinity="1" group="{GROUP_LANDER}"/></default>
    <default class="pad"><geom contype="2" conaffinity="1" group="{GROUP_COLLISION}" rgba="0.3 0.3 0.3 1"/></default>
    <default class="decor"><geom contype="0" conaffinity="0" group="{GROUP_DECOR}"/></default>
    <site group="4" rgba="1 0 0 0.5" size="0.02"/>
  </default>
  <worldbody>
    <body name="ground" mocap="true" pos="0 0 0">
      <geom name="terrain" type="plane" size="0 0 1" material="ground" contype="1" conaffinity="1"
            group="{GROUP_TERRAIN}" friction="{cfg.terrain.friction} 0.005 0.0001"/>
      <geom name="target" class="decor" type="cylinder" size="1 0.005" pos="0 0 0.01" rgba="0.98 0.98 0.95 0.9"/>
{ring}
    </body>
    <camera name="ground" mode="targetbody" target="lander" pos="22 -22 2.5"/>
    <body name="lander" pos="0 0 0">
      <freejoint name="root"/>
      <inertial pos="{_v(geom.cg)}" mass="{m_start:.6g}" diaginertia="{_v(mass.inertia)}"/>
      <light name="sun" mode="trackcom" directional="true" pos="0 0 30" dir="-0.45 0.35 -1"
             diffuse="0.95 0.85 0.75" specular="0.25 0.25 0.25" castshadow="true"/>
      <camera name="chase" mode="trackcom" pos="-11 -11 6" xyaxes="0.707 -0.707 0 0.28 0.28 0.92"/>
      <camera name="padcam" pos="0 0 {bottom - housing_drop - 0.03:.4f}" xyaxes="1 0 0 0 1 0" fovy="75"/>
      <geom name="hull" class="col" type="mesh" mesh="hexbody" material="hull"/>
      <geom name="radar_housing" class="col" type="box" pos="{_v(housing_c)}" size="{_v(housing_h)}" material="dark"/>
      <site name="imu" pos="{_v(geom.cg)}"/>
      <site name="radar" pos="{_v(geom.radar_pos)}"/>""")

    # engines (collision + visual) and their sites; the exhaust points down (-Z)
    for k, p in enumerate(geom.engine_pos):
        x.append(f'      <geom name="engine{k}" class="col" type="cylinder" size="0.10 0.13" '
                 f'pos="{_v(p + np.array([0.0, 0.0, 0.13]))}" material="gold"/>')
        x.append(f'      <geom class="vis" type="cylinder" size="0.115 0.012" pos="{_v(p + np.array([0.0, 0.0, 0.012]))}" '
                 f'material="dark"/>')
        x.append(f'      <site name="engine{k}" pos="{_v(p)}" zaxis="0 0 -1"/>')
    # propellant tanks (drawn; the propellant mass is lumped at the CG), RTG wind covers above them (S2)
    tank_az, tank_r, tank_z, tank_d = TANKS
    for k, az_deg in enumerate(tank_az):
        az = math.radians(az_deg)
        p = tank_r * unit_azimuth(az) + np.array([0.0, 0.0, tank_z])
        x.append(f'      <geom name="tank{k}" class="col" type="sphere" size="{0.5 * tank_d:.4f}" pos="{_v(p)}" material="tank"/>')
        rtg = 0.45 * unit_azimuth(az) + np.array([0.0, 0.0, top + 0.17])
        x.append(f'      <geom class="vis" type="box" size="0.17 0.30 0.17" pos="{_v(rtg)}" quat="{_zquat(az)}" material="white"/>')
    # high-gain antenna (side between legs 1 and 2, S3 p36), meteorology boom (leg 2), camera turrets
    hga_base = 0.30 * unit_azimuth(math.radians(30)) + np.array([0.0, 0.0, top])
    hga_top = hga_base + np.array([0.0, 0.0, 0.62])
    x.append(f'      <geom class="vis" type="capsule" size="0.03" fromto="{_v(hga_base, hga_top)}" material="strut"/>')
    x.append(f'      <geom class="vis" type="cylinder" size="0.38 0.012" pos="{_v(hga_top)}" '
             f'zaxis="{_v(0.5 * unit_azimuth(math.radians(200)) + np.array([0, 0, 0.85]))}" material="white"/>')
    met0 = 0.70 * unit_azimuth(math.radians(-30)) + np.array([0.0, 0.0, top])
    met1 = 0.95 * unit_azimuth(math.radians(-30)) + np.array([0.0, 0.0, top + 0.55])
    x.append(f'      <geom class="vis" type="capsule" size="0.015" fromto="{_v(met0, met1)}" material="strut"/>')
    for az in (250.0, 290.0):
        p = 0.55 * unit_azimuth(math.radians(az)) + np.array([0.0, 0.0, top + 0.15])
        x.append(f'      <geom class="vis" type="cylinder" size="0.08 0.15" pos="{_v(p)}" material="dark"/>')

    # rigid legs: footpad contact spheres; strut and bipod drawn from the S4 Fig 3 attachment points
    top_r, top_z = STRUT_TOP
    bip_r, bip_z, half = BIPOD
    for k, (pad, az) in enumerate(zip(geom.footpads, geom.leg_azimuth_rad)):
        radial, tangential = unit_azimuth(az), unit_azimuth(az + math.pi / 2)
        strut_top = top_r * radial + np.array([0.0, 0.0, top_z])
        hinge = bip_r * radial + np.array([0.0, 0.0, bip_z])
        x.append(f'      <geom name="pad{k}" class="pad" type="sphere" size="{r_pad:.6f}" pos="{_v(pad)}"/>')
        x.append(f'      <geom class="vis" type="cylinder" size="{0.5 * FOOTPAD_DIAMETER_M:.4f} 0.014" '
                 f'pos="{_v(pad + np.array([0.0, 0.0, 0.014 - r_pad]))}" material="pad"/>')
        x.append(f'      <geom class="vis" type="capsule" size="0.045" fromto="{_v(strut_top, pad)}" material="strut"/>')
        for side in (-1.0, 1.0):
            x.append(f'      <geom class="vis" type="capsule" size="0.022" '
                     f'fromto="{_v(hinge + side * half * tangential, pad)}" material="strut"/>')
    x.append("""    </body>
  </worldbody>
  <sensor>
    <accelerometer name="imu_acc" site="imu"/>
    <gyro name="imu_gyro" site="imu"/>
  </sensor>
</mujoco>
""")
    return "\n".join(x)


def build_model(cfg: Config, geom: LanderGeometry | None = None) -> tuple[mujoco.MjModel, ModelIndex]:
    """Compile the MJCF and collect the ids the simulator uses."""
    geom = geom or build_geometry(cfg)
    model = mujoco.MjModel.from_xml_string(build_mjcf(cfg, geom))

    def gid(name: str) -> int:
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)

    def sensor_adr(name: str) -> int:
        return int(model.sensor_adr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)])

    n_eng = len(geom.engine_pos)
    body_geom_names = ["hull", "radar_housing"] + [f"engine{k}" for k in range(n_eng)] + [f"tank{k}" for k in range(len(TANKS[0]))]
    idx = ModelIndex(
        lander_body=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "lander"),
        pad_geoms=tuple(gid(f"pad{k}") for k in range(len(geom.footpads))),
        body_geoms=tuple(gid(n) for n in body_geom_names),
        terrain_geom=gid("terrain"),
        ground_mocap=int(model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ground")]),
        engine_sites=tuple(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"engine{k}") for k in range(n_eng)),
        imu_acc_adr=sensor_adr("imu_acc"),
        imu_gyro_adr=sensor_adr("imu_gyro"),
    )
    return model, idx


def set_terrain(model: mujoco.MjModel, data: mujoco.MjData, idx: ModelIndex, slope_deg: float,
                downhill_deg: float, friction: float) -> np.ndarray:
    """Tilt the ground (a mocap body through the target at the origin) and set footpad friction.

    Call after `mj_resetData` (which resets mocap poses). Returns the ground's unit normal.
    """
    n = terrain_normal(slope_deg, downhill_deg)
    data.mocap_pos[idx.ground_mocap] = 0.0
    data.mocap_quat[idx.ground_mocap] = quat_z_to(n)
    model.geom_friction[idx.terrain_geom, 0] = friction
    return n
