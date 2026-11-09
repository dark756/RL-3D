"""Call-tree specification for the code guide: every node with its summary and interface.

F(...)  a real function/method/property/closure in the source (its home: explained here)
R(...)  a reference to a node explained elsewhere (another call site of the same function)
X(...)  an external library call (MuJoCo, imageio, ...)
P(...)  a pseudo node (program entry, test suite, test file, notebook)
Children are listed in the order the parent calls them.
"""

from __future__ import annotations


class Node:
    def __init__(self, kind, key=None, label=None, summary="", ins="", outs="", children=(), state=None,
                 verify=True, call_name=None, note=None):
        self.kind = kind            # function | method | property | closure | external | ref | entry | ...
        self.key = key              # "lander/sim.py::LanderSim.reset" for source nodes
        self.label = label
        self.summary = summary
        self.ins = ins
        self.outs = outs
        self.children = list(children)
        self.state = state
        self.verify = verify
        self.call_name = call_name  # externals: the attribute that must appear in the parent's source
        self.note = note
        self.parent = None
        self.num = None
        self.target = None          # refs: the home node


def F(key, summary, ins, outs, *children, state=None, verify=True, note=None, label=None):
    return Node("source", key, label, summary, ins, outs, children, state, verify, note=note)


def R(key, ins=None, outs=None, verify=True, note=None):
    return Node("ref", key, None, "", ins, outs, (), None, verify, note=note)


def X(label, call_name, summary, ins, outs):
    return Node("external", None, label, summary, ins, outs, (), None, True, call_name=call_name)


def P(kind, label, summary, ins, outs, *children, note=None, key=None):
    return Node(kind, key, label, summary, ins, outs, children, None, False, note=note)


S, C, V, G, A = "lander/sim.py", "lander/config.py", "lander/vehicle.py", "lander/geometry.py", "lander/aero.py"
SEN, T, M, SC, VI, H = "lander/sensors.py", "lander/touchdown.py", "lander/mjcf.py", "lander/scenarios.py", "lander/visuals.py", "lander/hud.py"
VD, VW = "lander/video.py", "lander/viewer.py"
CM, CK, EV, WT, BM, FS = ("scripts/common.py", "scripts/check_config.py", "scripts/export_video.py", "scripts/watch.py",
                          "scripts/benchmark.py", "scripts/fetch_sources.py")


def k(path, qual):
    return f"{path}::{qual}"


# ======================================================================================= config subtree
CONFIG_FROM_ARGS = F(
    k(CM, "config_from_args"),
    "Turns the parsed command line into a validated `Config`: the `--set` items become an overrides dict "
    "(origin `cli`) and the fact file is loaded with them.",
    "args: argparse.Namespace — `args.set` (list of 'KEY=VALUE'), `args.env_file` (path or None)",
    "Config",
    F(k(C, "parse_set_args"),
      "Splits each 'KEY=VALUE' item at the first '='. An item without '=' raises `ConfigError`.",
      "items: list[str], e.g. ['VIKING_ENGINE_T90_S=0.1']",
      "dict[str, str] — {key: raw value text}"),
    F(k(C, "load_config"),
      "The one way to obtain a `Config`. Reads the fact file, applies environment overrides `LANDER_<KEY>`, "
      "then explicit overrides, and builds the typed configuration. An unknown `LANDER_` variable or `--set` "
      "key is an error, so a typo can never be silently ignored.",
      "env_path (default ./.env); overrides: dict key → raw value; environ (default os.environ); "
      "origin label for the explicit overrides",
      "Config — every value plus its `Param` record (unit, status, source, note, origin)",
      F(k(C, "parse_env_file"),
        "Reads `.env` line by line. A fact line must be `KEY=value  # unit | status | source | note`. "
        "Rejects unparsable lines, duplicate keys, a status other than published / derived / assumed, an "
        "empty source, and '-' (no source) unless the status is assumed. Comment lines, including the parked "
        "facts at the end of the file, are skipped.",
        "path: Path of the fact file",
        "dict[str, Param] — origin '.env'",
        F(k(C, "parse_value"),
          "Converts the text of one value: an integer literal stays `int`, any other number becomes "
          "`float`, and comma-separated numbers become a tuple of floats.",
          "text: str ('3', '0.5', '0.835,-0.835,0'); key (only for error messages)",
          "int | float | tuple[float, ...]")),
      F(k(C, "_override"),
        "Replaces one fact's value. The new `Param` keeps the unit, gets status `override`, records its origin "
        "and notes the old value, status and source. A list must keep its length.",
        "old: Param; raw: str or number; origin: 'env:LANDER_<KEY>' | 'cli' | 'code' | 'test'",
        "Param with status 'override'",
        R(k(C, "parse_value"), "raw text of the override", "the parsed number(s)")),
      F(k(C, "build_config"),
        "Sorts the facts into the 11 group dataclasses by key prefix (the first matching prefix wins, so "
        "`MARS_WIND_` beats `MARS_`), checks that each group has exactly the keys its dataclass declares, "
        "converts every value to the declared type, then validates the whole configuration.",
        "params: dict[str, Param]; env_path: str",
        "Config (frozen)",
        F(k(C, "_group_of"),
          "Finds the parameter group of a key from the ordered prefix table `GROUPS`.",
          "key: str",
          "(prefix, group name, dataclass) or None"),
        F(k(C, "_coerce"),
          "Converts a parsed value to the field's declared type: `int` must be whole, `float` must be a single "
          "number, tuple fields accept one or more numbers.",
          "key; value: int | float | tuple; typ: int | float | tuple[float, ...]",
          "the value in the declared type (`ConfigError` otherwise)"),
        F(k(C, "validate"),
          "Cross-field sanity checks, all reported at once: one x and y for each of the 3 engines (D4); "
          "0 < T_min < T_max; positive t90 and Isp; positive dry mass and non-negative start propellant; "
          "positive inertias obeying the triangle inequality; 3 leg azimuths and a positive footpad radius; a "
          "storm range 'low, high'; a positive radar period; a whole number of physics steps per control step; "
          "contact time constant ≥ 2 physics steps; a non-negative noise scale.",
          "cfg: Config",
          "None — raises `ConfigError` listing every problem",
          F(k(C, "SimCfg.substeps"),
            "Physics steps per control step: round(control period / physics step) = 0.04 / 0.002 = 20.",
            "(self) `physics_dt_s`, `control_hz`",
            "int",
            F(k(C, "SimCfg.control_dt_s"),
              "Control period = 1 / `SIM_CONTROL_HZ` = 0.04 s: how long each command is held.",
              "(self) `control_hz`",
              "float (s)")),
          R(k(C, "SimCfg.control_dt_s"), "—", "0.04 s"))),
    ),
)

CHECK_CONFIG = F(
    k(CK, "main"),
    "Prints every fact from `.env` grouped by section with unit, status and source, then the derived "
    "sanity numbers and a count per status. With `--dump-mjcf` it also writes the generated MuJoCo XML.",
    "argv → --status, --grep, --dump-mjcf PATH, --set KEY=VALUE…, --env-file",
    "None — printed table; optional XML file",
    F(k(CM, "add_config_args"),
      "Adds the two options every script shares: `--set KEY=VALUE` (repeatable) to override a fact for one "
      "run, and `--env-file` to load a different fact file.",
      "parser: argparse.ArgumentParser",
      "None — the parser gains the two options"),
    CONFIG_FROM_ARGS,
    F(k(CK, "print_table"),
      "Prints the facts section by section ([engine] VIKING_ENGINE_*, …): key, value, unit, status, source "
      "and note. Keys of a more specific prefix are kept out of the general section (MARS_WIND_* is not "
      "repeated under MARS_*).",
      "cfg; status filter or None; grep substring or None",
      "None (stdout)",
      F(k(CK, "format_value"),
        "Compact text for a number or a list.",
        "value: int | float | tuple",
        "str, e.g. '0.835,-0.835,0' or '1.8'")),
    F(k(C, "derived"),
      "Computes the sanity numbers from the facts: start mass and weight, propellant flow at maximum and "
      "minimum thrust, thrust-to-weight (maximum and the floor while lit), hover valve setting, hover flow and "
      "hover time on the start tank, the engine time constant, the push of the strongest storm, the CG height "
      "ratio and the physics substeps.",
      "cfg: Config",
      "dict name → (value, unit, formula)",
      R(k(C, "SimCfg.substeps"), "—", "20")),
    R(k(G, "build_geometry"), "cfg (only with --dump-mjcf)", "LanderGeometry"),
    R(k(M, "build_mjcf"), "cfg, geom", "XML text, written to PATH"),
)

# ======================================================================================= simulator construction
BUILD_GEOMETRY = F(
    k(G, "build_geometry"),
    "Turns the geometry facts into vectors in the lander frame (TN D-7959 axes: origin on the centre line at "
    "the footpad plane, +Z up, leg 1 on +Y). The legs are rigid (D12), so everything is fixed: the three "
    "footpad contact-sphere centres, the CG on the centre line (D15), the engines on the body underside "
    "thrusting along +Z, both radar antennas at the centre of the underside, and the 4 Doppler beam "
    "directions (21.5° splay, 45° clock angle).",
    "cfg: Config (groups leg, engine, mass.cg_z_m, radar)",
    "LanderGeometry (footpads, footpad_radius, leg azimuths, cg, engine_pos, radar_pos, doppler_beams)",
    F(k(G, "unit_azimuth"),
      "Horizontal unit vector (cos az, sin az, 0).",
      "az_rad: float",
      "ndarray (3,)"),
)

BUILD_MODEL = F(
    k(M, "build_model"),
    "Compiles the generated XML with MuJoCo and looks up by name every id and address the simulator touches "
    "each step: lander body, footpad and body geoms, the terrain geom, the ground mocap index, engine sites, "
    "IMU sensor-data addresses.",
    "cfg, geom",
    "(mujoco.MjModel, ModelIndex)",
    F(k(M, "build_mjcf"),
      "Writes the MuJoCo model as text, so geometry always comes from `.env`: options (2 ms step, implicitfast "
      "integrator, Mars gravity, MuJoCo's fluid model off); visuals; assets (sky, ground texture, hexagonal "
      "body mesh); a mocap 'ground' body holding the terrain plane and the target marker; ONE free 'lander' "
      "body with the start mass, the CG and NASA's inertias, hull/radar-housing/engine/tank collision geoms, "
      "three footpad contact spheres, legs drawn as struts, sites and cameras; two sensors (IMU accelerometer "
      "and gyro at the CG). Sizes that only shape the drawing are constants in mjcf.py.",
      "cfg, geom",
      "str — MJCF XML",
      F(k(V, "MassModel.__init__"),
        "Mass model of the simplified vehicle (D15): dry mass 573.1 kg, the CG fixed on the centre line at "
        "0.79 m, the principal inertias fixed at NASA's touchdown values (S4 Fig 3).",
        "cfg.mass, geom.cg",
        "MassModel (dry, cg, inertia)"),
      F(k(V, "MassModel.total"),
        "Whole-vehicle mass with this much propellant left: dry + propellant (never below dry).",
        "propellant_kg",
        "float (kg)"),
      F(k(G, "hexagon_vertices"),
        "Body outline from the S2 side lengths: an equilateral triangle of side L + 2S with its corners cut, "
        "short sides facing the legs. Returns the 6 corners counter-clockwise.",
        "long_side, short_side (m); leg azimuths (rad)",
        "ndarray (6, 2) — x, y in m",
        R(k(G, "unit_azimuth"), "a leg azimuth and azimuth + 90°", "radial and tangential unit vectors")),
      F(k(M, "_v"),
        "Formats numbers and arrays as one space-separated string (6 significant digits) for XML attributes.",
        "*vals: numbers or arrays",
        "str"),
      F(k(M, "_zquat"),
        "Quaternion text for a rotation by az about +Z (target-ring markers, RTG covers).",
        "az (rad)",
        "str 'w x y z'"),
      R(k(G, "unit_azimuth"), "azimuths of the tanks, RTGs, antenna, boom, cameras and legs", "unit vectors")),
    X("mujoco.MjModel.from_xml_string(xml)", "from_xml_string",
      "MuJoCo compiles the XML into an MjModel: every constant of the simulation (bodies, geoms, sites, "
      "sensors, options).",
      "xml: str", "MjModel"),
    F(k(M, "build_model.gid"),
      "Helper inside build_model: id of a named geom.",
      "name: str",
      "int geom id"),
    F(k(M, "build_model.sensor_adr"),
      "Helper inside build_model: start address of a named sensor's values in `data.sensordata`.",
      "name: str",
      "int address"),
    X("mujoco.mj_name2id(model, type, name)", "mj_name2id",
      "Integer id of a named body or site.",
      "model, object type, name", "int"),
)

VEHICLE_INIT = F(
    k(V, "Vehicle.__post_init__"),
    "Creates the vehicle layer's two parts after the dataclass fields are set (engine bank and mass model) "
    "and caches the engines' lever arms about the CG, which never change because the CG is fixed.",
    "cfg, geom (dataclass fields)",
    "Vehicle (engines, mass, propellant = 0, empty = False)",
    F(k(V, "EngineBank.__init__"),
      "Reads the engine facts once: thrust limits 276 / 2,667 N, Isp·g_{0}, the first-order time constant "
      "τ = t90 / ln 10 = 23.0 ms (the same up and down), the misalignment angle and the engine positions.",
      "cfg.engine, geom.engine_pos",
      "EngineBank (thrust = cmd = 0, lit, thrust directions +Z)"),
    R(k(V, "MassModel.__init__"), "cfg, geom", "MassModel"),
)

LANDERSIM_INIT = F(
    k(S, "LanderSim.__init__"),
    "Builds a simulator once: geometry from the facts, the MuJoCo model compiled from generated XML, the "
    "MuJoCo state (`MjData`), and the backend-agnostic parts (vehicle, sensors, touchdown monitor, ground "
    "queries). Caches step sizes, drag areas, air density, the CG and inertia, then resets once so the "
    "object is ready to use.",
    "cfg: Config (None → `load_config()`, as the benchmark does); seed: int",
    "LanderSim (cfg, geom, model, idx, data, vehicle, sensors, monitor, terrain, dt, substeps, control_dt)",
    R(k(C, "load_config"), "— (only when cfg is None)", "Config"),
    BUILD_GEOMETRY,
    BUILD_MODEL,
    X("mujoco.MjData(model)", "MjData",
      "Allocates the simulation state: qpos (7 = frame origin xyz and attitude quaternion), qvel (6), time, "
      "applied forces, contacts, sensor data.",
      "model", "MjData"),
    VEHICLE_INIT,
    F(k(SEN, "SensorSuite.__init__"),
      "Stores the Doppler beam directions and the noise scale, then resets.",
      "cfg, geom",
      "SensorSuite",
      F(k(SEN, "SensorSuite.reset"),
        "Takes this episode's noise generator and noise scale (Viking's radar errors × scale; 1 by default) "
        "and clears the radars' sample-and-hold so the next sample is taken at once.",
        "rng; noise_scale or None (the .env value)",
        "None")),
    F(k(T, "TouchdownMonitor.__init__"),
      "Keeps the section 7 limits (`cfg.req`) and an empty `TouchdownReport`.",
      "cfg",
      "TouchdownMonitor"),
    F(k(S, "PlaneTerrain.__init__"),
      "Ground-query object for the sensor models: normal n = +Z until reset tilts it, and a back-reference to "
      "the simulator for MuJoCo ray casts.",
      "sim",
      "PlaneTerrain"),
    F(k(A, "drag_areas"),
      "C_{d}·A for each body axis: lateral x, y = 2.01 m² (calibrated to S4's touchdown aero forces), axial "
      "z = 1.80 m².",
      "cfg.aero",
      "ndarray (3,) m²"),
    R(k(S, "LanderSim.reset"), "seed", "first StepResult (discarded)"),
)

# ======================================================================================= scenario + video setup
MAKE_SCENARIO = F(
    k(SC, "make_scenario"),
    "Builds one of the five scripted cases: drop (6 m up, 2.44 m/s down, 6 kg propellant), hard_drop (1 m, "
    "4 m/s down + 0.6 m/s sideways), slope45 (2.44 m/s onto a 45° slope), hover (30 m up, start tank, never "
    "cuts) and freefall (100 m, engines off). All but hover cut the engines at first contact.",
    "name; sim; seed",
    "Scenario (initial_state, options dict, duration, policy, seed)",
    F(k(S, "LanderSim.state_above_ground"),
      "Turns 'pad_altitude metres above the ground, moving at velocity, tilted like this' into the CG state the "
      "simulator needs. It builds the attitude, rotates the three footpads, and places the lander frame so the "
      "lowest footpad sphere is exactly pad_altitude above the (possibly sloped) ground below it.",
      "pad_altitude (m); velocity (3, m/s world); tilt_deg; tilt_azimuth_deg; heading_deg; omega (3, rad/s "
      "body); xy; propellant_kg; throttle; slope_deg; slope_direction_deg",
      "InitialState (CG position and velocity in the world, quaternion, body rates, propellant, throttle)",
      F(k(M, "terrain_normal"),
        "Unit normal of a plane through the origin that falls toward `downhill_deg`: (sin s cos a, sin s sin a, "
        "cos s).",
        "slope_deg, downhill_deg",
        "ndarray (3,)"),
      F(k(S, "quat_from_tilt"),
        "Attitude quaternion: rotate `heading` about +Z, then lean the thrust axis by `tilt` toward "
        "`tilt_azimuth`.",
        "tilt_deg, tilt_azimuth_deg, heading_deg",
        "quaternion (w, x, y, z)",
        F(k(S, "quat_from_axis_angle"),
          "Quaternion of a rotation by an angle about an axis.",
          "axis (3,), angle (rad)",
          "quaternion"),
        F(k(S, "quat_mul"),
          "Quaternion (Hamilton) product a·b: apply b, then a.",
          "a, b: quaternions",
          "quaternion")),
      F(k(S, "quat_to_mat"),
        "Rotation matrix body → world from a unit quaternion (w, x, y, z).",
        "q: 4 numbers",
        "ndarray (3, 3)"),
      F(k(S, "plane_height"),
        "Ground height z at (x, y) for a plane through the origin with normal n.",
        "n (3,), x, y",
        "float (m)")),
)

VIDEO_INIT = F(
    k(VD, "VideoWriter.__init__"),
    "Opens an MP4 writer (libx264 through imageio-ffmpeg) at `SIM_CONTROL_HZ` frames per second, so one frame "
    "per control step plays back in real time, and creates the frame renderer.",
    "sim, path, width, height, camera name, hud flag, fps",
    "VideoWriter (frames, writer, count = 0)",
    F(k(VD, "FrameRenderer.__init__"),
      "Creates MuJoCo's offscreen renderer (an OpenGL context: hidden GLFW window on Windows, EGL on headless "
      "Linux) and the camera.",
      "sim, width, height, camera, hud",
      "FrameRenderer",
      X("mujoco.Renderer(model, height, width)", "Renderer",
        "Offscreen OpenGL renderer with its own MjvScene.", "model, size", "Renderer"),
      F(k(VI, "named_camera"),
        "'chase' → a camera that follows the lander; 'ground' or 'padcam' → the fixed cameras defined in the "
        "MJCF.",
        "sim, name",
        "MjvCamera",
        F(k(VI, "tracking_camera"),
          "Camera that follows the lander's centre of mass at a fixed, world-aligned distance, azimuth and "
          "elevation (10 m, 135°, −12°), so wind and drift directions read correctly.",
          "sim, distance, azimuth, elevation",
          "MjvCamera"))),
    X("imageio.get_writer(path, fps, codec='libx264')", "get_writer",
      "ffmpeg-backed MP4 writer from imageio-ffmpeg.", "path, fps, codec, quality", "writer"),
)

# ======================================================================================= one control step
TELEMETRY = F(
    k(S, "LanderSim.telemetry"),
    "Ground-truth snapshot from the MuJoCo state: CG position and velocity (frame origin plus the rotated CG "
    "offset), attitude, body rates, tilt, CG height and lowest-footpad height above the ground, propellant, "
    "mass, actual thrusts, last throttle command, whether the engines are off, the wind and the contacts.",
    "— reads data.qpos, data.qvel, data.geom_xpos and the components' state",
    "Telemetry",
    R(k(S, "quat_to_mat"), "data.qpos[3:7]", "rotation R"),
    F(k(G, "cross3"),
      "Cross product of two 3-vectors written out with floats; np.cross was a bottleneck of the physics loop.",
      "a, b: (3,)",
      "ndarray (3,)"),
    F(k(S, "PlaneTerrain.height_above"),
      "Vertical height of a point above the ground directly below it.",
      "p: (3,) world",
      "float (m)",
      F(k(S, "PlaneTerrain.height"),
        "Ground height at (x, y).",
        "x, y",
        "z (m)",
        R(k(S, "plane_height"), "n, x, y", "z"))),
    F(k(S, "LanderSim.t"),
      "Simulation time = `data.time`.",
      "(self)",
      "float (s)"),
    F(k(S, "tilt_of"),
      "Angle between the body +Z (thrust) axis and world +Z: acos(R[2, 2]).",
      "rot: (3, 3)",
      "float (deg)"),
    R(k(S, "LanderSim.engines_off"), "—", "bool"),
)

READ_SENSORS = F(
    k(S, "LanderSim.read_sensors"),
    "Feeds the sensor models with the truth: specific force and body rates from MuJoCo's accelerometer and "
    "gyro at the IMU site (the CG), the frame pose and velocities, and the ground-query object.",
    "— reads data.sensordata, data.qpos, data.qvel",
    "SensorReadings",
    R(k(S, "quat_to_mat"), "data.qpos[3:7]", "rotation R"),
    F(k(SEN, "SensorSuite.sample"),
      "The IMU is ideal: it passes the true specific force and rates through. The radars are sampled only "
      "every 200 ms and hold their values in between (S3 p93): it computes the antenna position and velocity "
      "at the centre of the underside, then reads the altimeter and the Doppler radar.",
      "t, f_body (3), omega_body (3), origin (3), rot (3×3), v_origin (3), omega_world (3), terrain",
      "SensorReadings (accel, gyro, altitude, doppler_beam × 4, doppler_vel)",
      R(k(G, "cross3"), "ω × r (antenna offset from the frame origin)", "rotation part of the antenna velocity"),
      F(k(SEN, "SensorSuite._read_altimeter"),
        "First-return radar altimeter (S3 p109): range to the nearest ground point from the antenna, discarded "
        "below 41 m (false echoes off the lander) or if the ground is not below the underside. Noise: "
        "(4.5 % + 1.5 m) / 3 × the noise scale (1σ).",
        "p_ant (3), rot, terrain",
        "float (m), NaN = invalid",
        F(k(S, "PlaneTerrain.nearest"),
          "For the plane: perpendicular distance n·p and the direction −n toward the surface.",
          "p: (3,) world",
          "(distance m, unit vector)")),
      F(k(SEN, "SensorSuite._read_doppler"),
        "Four-beam Doppler radar (S3 p111–112): each beam (21.5° from −Z, 45° clock) must hit the ground and "
        "see a closing speed of at least 1 m/s, so it has no lock while hovering. Noise: max(1.5 % × speed, "
        "0.3 m/s) / 3 × the noise scale. With ≥ 3 locked beams the body-frame ground velocity is solved by "
        "least squares.",
        "p_ant (3), v_ant (3), rot, terrain",
        "(beam speeds (4,), velocity (3,)) — NaN = no lock",
        F(k(S, "PlaneTerrain.raycast"),
          "Distance along a ray to the ground, using MuJoCo's ray cast restricted to geom group 0 (terrain "
          "only).",
          "origin (3,), direction (3,)",
          "float (m) or inf",
          X("mujoco.mj_ray(model, data, pnt, vec, geomgroup, …)", "mj_ray",
            "Intersects a ray with the geoms of the selected groups.", "point, direction, group mask",
            "distance or −1")))),
)

RESET = F(
    k(S, "LanderSim.reset"),
    "Starts an episode. Spawns two random streams (engine misalignment, radar noise) from one seed in a fixed "
    "order (D10); resets MuJoCo; tilts the ground and sets friction; sets the mass for the starting "
    "propellant; writes qpos and qvel from the CG state; picks the starting throttle (hover trim if none "
    "given); resets the vehicle, the episode's steady wind, the sensors and the touchdown monitor; zeroes the "
    "applied forces; recomputes MuJoCo; returns the first StepResult.",
    "seed (int or None = keep the last); initial_state: InitialState (None = 30 m hover); options: "
    "ResetOptions, dict or None",
    "StepResult (telemetry at t = 0, sensor readings, empty report, events [])",
    F(k(S, "ResetOptions.from_any"),
      "Accepts None (all defaults), a `ResetOptions` or a dict of its fields; unknown keys raise an error "
      "listing the valid ones.",
      "options",
      "ResetOptions"),
    X("mujoco.mj_resetData(model, data)", "mj_resetData",
      "Back to the model's defaults: time 0, default qpos, zero velocities, default mocap poses.",
      "model, data", "None"),
    F(k(M, "set_terrain"),
      "Tilts the ground for this episode by setting the 'ground' mocap body's quaternion (the plane passes "
      "through the target at the origin) and sets the terrain friction (contact friction = the larger of the "
      "two geoms'; footpads use 0.01).",
      "model, data, idx, slope_deg, downhill_deg, friction",
      "ground normal (3,)",
      R(k(M, "terrain_normal"), "slope_deg, downhill_deg", "normal n"),
      F(k(M, "quat_z_to"),
        "Quaternion that rotates +Z onto a unit vector.",
        "n: (3,)",
        "quaternion")),
    R(k(S, "LanderSim.state_above_ground"), "30 m (only when no initial_state is given)", "InitialState"),
    F(k(S, "LanderSim._apply_mass"),
      "Writes the lander's mass (dry + propellant) into the MuJoCo model; the CG and inertia never change "
      "(D15), so nothing else is touched.",
      "propellant (kg)",
      "None",
      R(k(V, "MassModel.total"), "propellant", "mass (kg)"),
      state="writes model.body_mass and body_subtreemass; sets _mass"),
    R(k(S, "quat_to_mat"), "initial quaternion", "rotation R"),
    R(k(G, "cross3"), "ω_world × R·cg", "velocity offset between CG and frame origin"),
    F(k(S, "LanderSim.hover_trim"),
      "Valve settings that give a total thrust (default: the current weight on Mars) with zero pitch and yaw "
      "torque about the CG.",
      "propellant (default: current); total_thrust (default: weight)",
      "tuple of 3 valve settings",
      R(k(V, "MassModel.total"), "propellant", "mass for the weight"),
      F(k(V, "Vehicle.trim_thrust"),
        "Solves ΣT = F, ΣT·y = 0, ΣT·x = 0 for the three engine thrusts. Engine 3 sits a little further out "
        "(S4 Fig 6a), so it runs ~2.5 % lower than engines 1 and 2.",
        "total_thrust (N)",
        "ndarray (3,) N"),
      F(k(V, "EngineBank.thrust_to_valve"),
        "Inverse of the valve law: u = (T − T_min) / (T_max − T_min), not clipped.",
        "thrust (N)",
        "valve setting(s)")),
    F(k(V, "Vehicle.reset"),
      "Sets the propellant, marks the tank empty if it is, and resets the engines (off when they are disabled "
      "or the tank is empty).",
      "rng, propellant_kg, throttle0 (3), engines_enabled, fault_scale (3)",
      "None",
      F(k(V, "EngineBank.reset"),
        "Per episode: lit flag, per-engine fault scale, and each thrust direction leaned by the misalignment "
        "angle (0.044°) toward a random azimuth. The engines start at the commanded throttle with no "
        "transient.",
        "rng, throttle0 (3), enabled, fault_scale (3)",
        "None",
        F(k(V, "EngineBank.valve_to_thrust"),
          "Viking valve law (S3 p96): T = T_min + clip(u, 0, 1)·(T_max − T_min). u = 0 gives 276 N, not off.",
          "u: one or three valve settings",
          "thrust (N)"))),
    F(k(A, "wind_vector"),
      "The episode's steady wind: speed × (cos dir, sin dir, 0), blowing toward `dir`. Calm episodes pass 0; "
      "dust storms are 17–30 m/s.",
      "speed_mps, direction_deg (reset options)",
      "ndarray (3,) m/s, world frame"),
    R(k(SEN, "SensorSuite.reset"), "rng, sensor_noise_scale option", "None"),
    F(k(T, "TouchdownMonitor.reset"),
      "Starts a fresh `TouchdownReport` and clears the settle timer.",
      "—",
      "None"),
    F(k(S, "LanderSim.engines_off"),
      "True once the engines are shut down, by the engines-off command or an empty tank. Permanent: nothing "
      "relights them.",
      "(self) — reads vehicle.engines.lit",
      "bool"),
    F(k(T, "TouchdownMonitor.note_engines"),
      "Records every physics step whether the engines are shut down, even after the landing is decided, "
      "and re-judges the criteria when that changes after touchdown: a landing only counts once the engines "
      "are off.",
      "engines_off: bool",
      "None",
      F(k(T, "TouchdownMonitor._judge"),
        "Fills report.criteria per brief §7: vertical speed ≤ 3.34 m/s and horizontal ≤ 1.22 m/s at first "
        "contact (otherwise a leg broke), stable (once settled or tipped), no body strike, engines off, "
        "propellant left at touchdown. inside_pad_radius and tilt_within_limit stay None until O3 (M2).",
        "—",
        "None")),
    X("mujoco.mj_forward(model, data)", "mj_forward",
      "Recomputes everything derived from the state (positions, contacts, sensor data) without advancing "
      "time.", "model, data", "None"),
    F(k(S, "LanderSim._result"),
      "Packs a control step's outputs.",
      "events: list[str]",
      "StepResult(telemetry, sensors, report, events)",
      TELEMETRY,
      READ_SENSORS),
    state="writes data.qpos, data.qvel, the ground mocap pose, the model mass; stores options, wind, slope, "
          "friction",
)

PHYSICS_STEP = F(
    k(S, "LanderSim._physics_step"),
    "One 2 ms physics step, in this order: read the state (rotation, CG, CG velocity) → engines and "
    "propellant → drag in the steady wind → write the applied force and torque into MuJoCo → update the mass "
    "→ mj_step → contacts and events → note the engine state → touchdown monitor (only near the ground, after "
    "touchdown, or once the tank is empty). Figure 2 shows the data flow.",
    "events: list (appended in place); reads data.qpos and data.qvel",
    "None",
    R(k(S, "quat_to_mat"), "data.qpos[3:7]", "rotation R"),
    R(k(G, "cross3"), "ω_world × R·cg", "CG velocity offset"),
    F(k(V, "Vehicle.step"),
      "Advances the engines (cut once the tank is empty), integrates propellant use from the actual thrust "
      "(D6), and sums the thrust force and its torque about the CG in body axes.",
      "dt",
      "VehicleStep (thrust (3), force_b, torque_b, mdot, empty)",
      F(k(V, "EngineBank.step"),
        "Per engine: first-order lag toward the target with τ = 23 ms (the same up and down), clamp to "
        "[T_min, T_max] while lit or [0, T_max] once shut down, then the fault scale. A cut latches the engines "
        "off for good.",
        "dt, cut",
        "actual thrust (3) N"),
      state="decreases propellant; sets empty"),
    F(k(A, "drag_force_body"),
      "Quadratic drag per body axis: F_{k} = −½ ρ |v| (C_{d}A)_{k} v_{k}, with v the CG velocity relative to "
      "the steady wind in body axes and ρ = 0.016 kg/m³ at every height. It acts at the CG (no torque).",
      "v_rel_body (3), rho, cda (3)",
      "ndarray (3,) N, body frame"),
    R(k(S, "LanderSim._apply_mass"), "propellant left", "None — mass = dry + propellant"),
    X("mujoco.mj_step(model, data)", "mj_step",
      "MuJoCo advances 2 ms: forward dynamics with the applied force and torque, gravity and soft contacts "
      "(implicitfast), then integrates qpos and qvel and fills sensordata.", "model, data",
      "None (new qpos, qvel, contacts, sensordata)"),
    F(k(S, "LanderSim._contacts"),
      "Scans MuJoCo's contact list for pairs that involve the terrain: footpad spheres → pad_contact (3 "
      "flags); hull, radar housing, engines or tanks → body_contact.",
      "— reads data.contact",
      "None",
      state="sets _pad_contact, _body_contact"),
    F(k(S, "LanderSim._flag_events"),
      "Adds each event once per episode: engines_cut, propellant_exhausted, body_strike.",
      "events list",
      "None (events appended)",
      R(k(S, "LanderSim.engines_off"), "—", "bool")),
    R(k(S, "LanderSim.engines_off"), "—", "bool for note_engines"),
    R(k(T, "TouchdownMonitor.note_engines"), "engines_off", "None"),
    F(k(T, "TouchdownReport.done"),
      "The landing is decided: settled, tipped over, a broken leg, or the body hit the ground after "
      "touchdown.",
      "(self)",
      "bool"),
    F(k(T, "stability_distance"),
      "TN D-7959 stability measure: the smallest signed horizontal distance from the CG to the vertical planes "
      "through each pair of footpads. Positive inside the footpad triangle, negative = tipping over.",
      "cg (3), pads (3, 3) world",
      "float (m)"),
    R(k(S, "tilt_of"), "rotation R", "tilt (deg)"),
    F(k(T, "TouchdownMonitor.update"),
      "Before contact it only watches for an empty tank or a body strike. At first contact it records the "
      "speeds and tilt, decides whether a leg broke (faster than 3.34 m/s down or 1.22 m/s sideways, S5) and "
      "keeps the impact kinetic energy (floor 100 J). After that it tracks the minimum stability distance, "
      "flags a tip-over (stability < 0) and declares the lander settled when it rests on all three feet with "
      "kinetic energy below 0.5 % of the impact energy for 0.5 s. Then it re-judges the criteria.",
      "t, pad_contact (3), body_contact, v_cg (3), ke (J), stability (m), tilt_deg, empty, xy",
      "None (report updated)",
      R(k(T, "TouchdownMonitor._judge"), "—", "None"),
      R(k(T, "TouchdownReport.done"), "—", "bool: is the landing decided?")),
    state="writes data.xfrc_applied (force + torque on the lander body, world frame) and the model mass; "
          "caches _last_thrust",
)

STEP = F(
    k(S, "LanderSim.step"),
    "Advances one control period (40 ms): latches the command (zero-order hold), runs 20 physics steps, "
    "recomputes MuJoCo's derived quantities for the final state and returns the outputs.",
    "command: Command (None = repeat the last); substep_callback(sim, k) or None",
    "StepResult",
    F(k(V, "Vehicle.command"),
      "Hands the command to the engines at the start of the control period.",
      "cmd: Command (throttle (3), engines_off)",
      "None",
      F(k(V, "EngineBank.command"),
        "New thrust targets from the valve settings; `engines_off` shuts the engines down for good.",
        "throttle (3), engines_off",
        "None (cmd updated; lit may become False)",
        R(k(V, "EngineBank.valve_to_thrust"), "valve settings", "target thrusts (N)"))),
    PHYSICS_STEP,
    X("mujoco.mj_forward(model, data)", "mj_forward",
      "Brings positions and sensor data up to date for the final state of the control period.",
      "model, data", "None"),
    R(k(S, "LanderSim._result"), "events of this control step", "StepResult"),
)

# ======================================================================================= rendering
RENDER = F(
    k(VD, "FrameRenderer.render"),
    "Updates MuJoCo's scene from the state and camera, adds plumes and arrows, renders to pixels, then draws "
    "the HUD.",
    "res: StepResult or None",
    "ndarray (H, W, 3) uint8",
    X("Renderer.update_scene(data, camera)", "update_scene",
      "Builds the list of geoms to draw (MjvScene) for this state and camera.", "data, camera", "None"),
    F(k(VI, "decorate"),
      "Adds every overlay for the current state: engine plumes, a velocity arrow at the CG (green) and a wind "
      "arrow above it (blue).",
      "scene (MjvScene), sim, plumes flag, arrows flag",
      "None (scene.ngeom grows)",
      F(k(VI, "add_plumes"),
        "For each engine, three nested translucent ellipsoids along the exhaust direction, length 3.5 m × "
        "(T / T_max)^{0.7}, width growing with thrust; engines below 1 % thrust draw nothing.",
        "scene, sim — reads sim._last_thrust and the engine site poses",
        "None",
        F(k(VI, "add_geom"),
          "Appends one decorative geom to a scene if there is room and marks it as decor.",
          "scene, geom type, size (3), position (3), rotation (3×3), rgba (4)",
          "bool (False when the scene is full)",
          X("mujoco.mjv_initGeom(geom, type, size, pos, mat, rgba)", "mjv_initGeom",
            "Fills a scene geom's type, size, pose and colour.", "geom fields", "None"))),
      R(k(S, "LanderSim.telemetry"), "—", "Telemetry (CG, velocity, wind)"),
      F(k(VI, "add_arrow"),
        "An arrow from `start` along `vec`, length = speed × 0.6 s (at most 8 m).",
        "scene, start (3), vec (3), rgba, width",
        "None",
        F(k(VI, "_frame_with_z"),
          "Rotation matrix whose third column is the given direction (arrows point along their local +Z).",
          "z (3,)",
          "ndarray (3, 3)"),
        R(k(VI, "add_geom"), "arrow geom", "bool"))),
    X("Renderer.render()", "render",
      "Draws the scene offscreen and reads the pixels back.", "—", "ndarray (H, W, 3)"),
    F(k(H, "hud_lines"),
      "The HUD rows (brief §8): time, radar altitude ('--' when the altimeter has no valid reading, e.g. below "
      "41 m), true footpad altitude, vertical and horizontal speed, tilt, spin, propellant, each engine's "
      "thrust as % of maximum, engines ON / CUT, wind ('calm' or speed and heading) and a status line.",
      "sim, res: StepResult",
      "list of (label, value) strings",
      F(k(SEN, "SensorReadings.altitude_valid"),
        "True when the altimeter reading is a number (not NaN).",
        "(self)",
        "bool"),
      F(k(VI, "wind_heading_deg"),
        "Direction the wind blows toward, 0–360° from +X toward +Y.",
        "wind (3,)",
        "float (deg)"),
      F(k(S, "Telemetry.vz"),
        "Vertical CG speed (+ up).",
        "(self)",
        "float (m/s)"),
      F(k(S, "Telemetry.vh"),
        "Horizontal CG speed.",
        "(self)",
        "float (m/s)"),
      F(k(S, "Telemetry.spin_dps"),
        "Rotation rate about the lander's own vertical (thrust) axis. Nothing controls it (no roll jets, D4).",
        "(self)",
        "float (deg/s)"),
      F(k(H, "status_text"),
        "One status line: 'LANDED - all section 7 checks pass', 'DOWN - engines still firing (must be cut)', "
        "'FAILED: leg broken at …', 'FAILED: <criteria>', 'TOUCHDOWN <speeds>', 'ENGINES OFF' or "
        "'POWERED DESCENT'.",
        "res: StepResult",
        "str",
        R(k(T, "TouchdownReport.done"), "—", "bool"),
        F(k(T, "TouchdownReport.success"),
          "Decided, and every judged criterion passed (criteria still None, the O3 limits, are skipped).",
          "(self)",
          "bool",
          R(k(T, "TouchdownReport.done"), "—", "bool")),
        F(k(T, "TouchdownReport.failures"),
          "Names of the criteria that are False.",
          "—",
          "list[str]"))),
    F(k(VD, "overlay_hud"),
      "Draws the rows in a translucent top-left panel (labels amber, values white) and the last row, the "
      "status, as a banner at the bottom centre. The font size scales with the image height.",
      "img (H, W, 3), rows, font_size or None",
      "ndarray (H, W, 3)",
      F(k(VD, "_font"),
        "Pillow's built-in scalable font at a size (bitmap font on very old Pillow).",
        "size (px)",
        "ImageFont")),
)

RUN_SCENARIO = F(
    k(SC, "run_scenario"),
    "The M1 episode loop. Resets the simulator with the scenario's seed, initial state and options, then "
    "repeatedly asks the policy for a Command and steps, until the scenario's duration or 1 s after the "
    "landing is decided. The optional `on_step` callback sees every StepResult (export_video renders a frame "
    "there).",
    "sim, scenario, on_step(sim, StepResult) or None, settle_extra_s",
    "list[StepResult] — one per control step, including the reset",
    RESET,
    F(k(VD, "VideoWriter.add"),
      "Renders the current state and appends the frame to the MP4. `run_scenario` reaches it through the "
      "`on_step` callback that export_video.main passes in.",
      "res: StepResult",
      "None (count + 1)",
      RENDER,
      X("writer.append_data(frame)", "append_data",
        "Encodes one frame into the MP4.", "frame (H, W, 3)", "None"),
      verify=False, note="called through the on_step lambda"),
    F(k(SC, "hover_then_cut"),
      "The policy of drop, hard_drop, slope45 and freefall: thrust = weight with zero pitch and yaw torque "
      "until the first footpad contact, then engines off for good. It uses the latched touchdown flag because "
      "the rigid legs rebound slightly (~13 % of the impact speed). Nothing corrects attitude or drift, so the "
      "scenarios show the raw physics.",
      "sim",
      "Command (hover trim, engines_off = touched down)",
      R(k(S, "LanderSim.hover_trim"), "— (current propellant)", "3 valve settings"),
      verify=False, note="called as scenario.policy(sim)"),
    F(k(SC, "hover"),
      "The hover scenario's policy: thrust = weight, engines never cut. With no attitude control the thrust "
      "misalignment slowly tilts and spins the lander.",
      "sim",
      "Command (hover trim)",
      R(k(S, "LanderSim.hover_trim"), "— (current propellant)", "3 valve settings"),
      verify=False, note="called as scenario.policy(sim) for the hover scenario"),
    STEP,
    R(k(T, "TouchdownReport.done"), "—", "bool: stop 1 s after this turns True"),
)

SUMMARIZE = F(
    k(CM, "summarize"),
    "Readable multi-line outcome: time, propellant left and engine state; touchdown speeds and tilt (and "
    "'LEG BROKEN'); minimum stability distance, settled, tipped or body strike; each criterion (n/a for the "
    "M2 ones); RESULT: SUCCESS, FAILED: <criteria> or undecided.",
    "report: TouchdownReport, telemetry: Telemetry",
    "str",
    R(k(T, "TouchdownReport.success"), "—", "bool"),
    R(k(T, "TouchdownReport.done"), "—", "bool"),
    R(k(T, "TouchdownReport.failures"), "—", "list[str]"),
)

EXPORT_VIDEO = F(
    k(EV, "main"),
    "Records a scenario to MP4 without opening a window: builds the simulator, makes the scenario, adds the "
    "`--wind` option, runs it while rendering one frame per control step (plumes + HUD), closes the file and "
    "prints the landing summary.",
    "argv → --scenario, --seed, --out, --width, --height, --camera chase|ground|padcam, --no-hud, --wind, "
    "--set, --env-file",
    "Path of the MP4 (default runs/videos/<scenario>_seed<N>.mp4)",
    F(k(CM, "add_wind_arg"),
      "Adds `--wind SPEED[,DIRECTION]`: a steady wind for the episode (calm by default; Mars dust storms are "
      "17–30 m/s).",
      "parser",
      "None — the parser gains the option"),
    R(k(CM, "add_config_args"), "the script's parser", "None"),
    R(k(CM, "config_from_args"), "parsed args", "Config"),
    LANDERSIM_INIT,
    MAKE_SCENARIO,
    F(k(CM, "wind_options"),
      "Turns `--wind 25,90` into reset options {wind_speed_mps: 25, wind_direction_deg: 90}; empty when the flag "
      "is not given.",
      "args",
      "dict of ResetOptions fields"),
    VIDEO_INIT,
    RUN_SCENARIO,
    F(k(VD, "VideoWriter.close"),
      "Finishes the MP4 file and releases the renderer.",
      "—",
      "None",
      F(k(VD, "FrameRenderer.close"),
        "Releases the OpenGL context.",
        "—",
        "None")),
    SUMMARIZE,
)

# ======================================================================================= viewer, benchmark, fetch
WATCH = F(
    k(WT, "main"),
    "Watches a scenario live in MuJoCo's viewer window; the scenario's policy flies it (the RL agent will plug "
    "in here at M4/M5). Prints the landing summary when the window closes.",
    "argv → --scenario, --seed, --speed, --max-seconds, --no-hud, --wind, --set, --env-file",
    "None — a window, then a printed summary",
    R(k(CM, "add_wind_arg"), "the script's parser", "None"),
    R(k(CM, "add_config_args"), "the script's parser", "None"),
    R(k(CM, "config_from_args"), "parsed args", "Config"),
    R(k(S, "LanderSim.__init__"), "cfg, seed", "LanderSim"),
    R(k(SC, "make_scenario"), "scenario name, sim, seed", "Scenario"),
    R(k(CM, "wind_options"), "parsed args", "reset options added to the scenario's"),
    F(k(VW, "watch"),
      "The live viewer loop. Resets via reset_fn, opens MuJoCo's passive viewer with a key callback (Space "
      "pause, Enter restart), points the camera at the lander, then each iteration: restart if asked; step "
      "the policy unless paused or holding after the landing is decided (2 s); redraw the overlays into the "
      "viewer's user scene; show the HUD with set_texts; sync; sleep so that simulated time runs at speed × "
      "real time.",
      "sim, policy(sim) → Command, reset_fn() → StepResult, speed, max_seconds, hud, hold_after_done_s",
      "None — returns when the window closes or max_seconds have passed",
      F(k(WT, "main.reset_fn"),
        "Defined in watch.main: resets the simulator with the scenario's seed, initial state and options, and "
        "remembers the result.",
        "—",
        "StepResult",
        R(k(S, "LanderSim.reset"), "scenario seed, initial_state, options", "StepResult"),
        label="reset_fn (in watch.main)"),
      X("mujoco.viewer.launch_passive(model, data, key_callback, …)", "launch_passive",
        "Opens the viewer window on its own thread and returns a handle (lock, user_scn, cam, set_texts, sync, "
        "is_running).", "model, data, key_callback", "viewer handle"),
      F(k(VW, "watch.on_key"),
        "Runs on the viewer's thread: Space toggles pause, Enter asks for a restart. It only sets flags; the "
        "main loop acts on them.",
        "key: GLFW key code",
        "None",
        label="on_key (in watch)"),
      R(k(VI, "tracking_camera"), "sim", "camera settings copied into the viewer"),
      F(k(WT, "main.policy"),
        "Defined in watch.main: forwards to the scenario's policy.",
        "sim",
        "Command",
        R(k(SC, "hover_then_cut"), "sim", "Command", verify=False,
          note="sc.policy(s); hover for the hover scenario"),
        label="policy (in watch.main)"),
      R(k(S, "LanderSim.step"), "the policy's Command", "StepResult"),
      R(k(VI, "decorate"), "viewer.user_scn, sim", "None"),
      F(k(H, "hud_columns"),
        "The HUD rows as two newline-joined columns (labels, values) for MuJoCo's text overlay.",
        "sim, res",
        "(str, str)",
        R(k(H, "hud_lines"), "sim, res", "rows")),
      X("handle.set_texts(…) / handle.sync()", "set_texts",
        "Overlays the HUD text and copies the latest state to the window.", "texts", "None")),
    R(k(S, "LanderSim.telemetry"), "—", "Telemetry for the summary"),
    R(k(CM, "summarize"), "final report, telemetry", "str"),
)

BENCHMARK = F(
    k(BM, "main"),
    "Measures simulator speed with 1, 2, 4, 6, 8 worker processes and prints control steps per second, the "
    "real-time factor and the time 10 M environment steps would take.",
    "argv → --procs, --sim-seconds",
    "None — printed table",
    F(k(BM, "run"),
      "Starts a pool of N spawned processes running _worker and aggregates: total control steps ÷ the slowest "
      "worker's wall time.",
      "procs, sim_seconds",
      "(control steps per second, real-time factor)",
      F(k(BM, "_worker"),
        "In its own process: builds a LanderSim, flies the hover scenario again and again (a new episode every "
        "20 simulated seconds) until sim_seconds have been simulated, and times it.",
        "sim_seconds, seed",
        "(control steps, wall seconds)",
        R(k(S, "LanderSim.__init__"), "seed (cfg loaded inside)", "LanderSim"),
        R(k(SC, "make_scenario"), "'hover', sim, seed", "Scenario"),
        R(k(S, "LanderSim.reset"), "seed + steps, hover state", "StepResult"),
        R(k(SC, "hover"), "sim", "Command", verify=False, note="sc.policy(sim)"),
        R(k(S, "LanderSim.step"), "Command", "StepResult"))),
)

FETCH = F(
    k(FS, "main"),
    "Downloads the NASA source PDFs S3 (RP-1027), S4 (TN D-7959) and S5 (Pohlen et al.) into docs/sources/ "
    "with urllib, skipping files that already exist unless --force. The .env citations point to these.",
    "argv → --dest (default docs/sources), --force",
    "None — three PDF files and their sizes printed",
)

# ======================================================================================= tests, notebook, root
TESTS = P(
    "tests", "tests  (python -m pytest)",
    "The test suite: 37 headless tests in 7 files (≈ 25 s). Tests call the library directly rather than the "
    "scripts. The children are the test files; library functions that only the tests use are explained under "
    "them.",
    "pytest command line", "pass / fail report",
    P("testfile", "tests/conftest.py",
      "Fixtures shared by all tests: `cfg` = load_config(environ={}) so local LANDER_* overrides cannot change "
      "results; `make_sim(overrides, seed)` = LanderSim(cfg.with_overrides(...)); `CLEAN` = no air, no "
      "misalignment and perfect radar readings, for comparisons with closed-form physics.",
      "—", "fixtures",
      F(k(C, "Config.with_overrides"),
        "A new Config with some values replaced: each becomes an `override` Param with the given origin, then "
        "everything is rebuilt and re-validated.",
        "overrides: dict key → value, origin",
        "Config",
        R(k(C, "_override"), "old Param, value, origin", "override Param"),
        R(k(C, "build_config"), "updated params", "Config")),
      key="tests/conftest.py"),
    P("testfile", "tests/test_config.py",
      "The fact file: strict parsing, metadata rules, parked facts ignored, overrides, and the derived numbers.",
      "—", "9 tests",
      F(k(C, "Config.param"),
        "The full record of one key (`ConfigError` if unknown).",
        "key",
        "Param"),
      F(k(C, "Config.overridden"),
        "The params whose value came from an override.",
        "—",
        "dict key → Param"),
      F(k(C, "Config.to_log_dict"),
        "JSON-ready dump of every param (value, unit, status, source, note, origin), for M2's episode logs.",
        "—",
        "dict"),
      key="tests/test_config.py"),
    P("testfile", "tests/test_model.py",
      "The generated MuJoCo model is one free body; mass = dry + propellant at every step while the CG and "
      "inertia stay fixed; the footprint matches the drawings; editing .env changes the compiled model.",
      "—", "4 tests", key="tests/test_model.py"),
    P("testfile", "tests/test_physics.py",
      "Brief §11: free fall vs closed form, hover holds altitude and stays level, misalignment tilts an "
      "uncontrolled lander, mass flow = ∫T / (Isp·g_{0}) dt, the engines-off command is permanent, "
      "bit-identical determinism, rest stability on flat ground (the dt check).",
      "—", "7 tests", key="tests/test_physics.py"),
    P("testfile", "tests/test_legs.py",
      "Rigid legs: a leg breaks above 3.34 m/s down or 1.22 m/s sideways and not below; slope drops (45° tips "
      "over, 10° never; NASA's 0.85 printed for comparison).",
      "—", "2 tests", key="tests/test_legs.py"),
    P("testfile", "tests/test_aero_sensors.py",
      "Drag in calm descent and in a dust storm, storm drift after 30 s, ideal radar altimeter (41 m cut-off), "
      "ideal Doppler (no lock in hover), ideal IMU, radar noise at the published levels.",
      "—", "6 tests",
      F(k(SEN, "SensorReadings.doppler_valid"),
        "Mask of Doppler beams with a lock (finite readings).",
        "(self)",
        "bool array (4,)"),
      key="tests/test_aero_sensors.py"),
    P("testfile", "tests/test_touchdown.py",
      "Stability-distance geometry; the nominal drop succeeds with events in order; the hard drop breaks a "
      "leg; free fall crashes; a landing only counts once the engines are cut; running out of propellant in "
      "the air is flagged.",
      "—", "6 tests", key="tests/test_touchdown.py"),
    P("testfile", "tests/test_video_docs.py",
      "Offscreen frame and MP4 work; README names every module, class and function; the vehicle layer never "
      "imports MuJoCo.",
      "—", "3 tests", key="tests/test_video_docs.py"),
)

NOTEBOOK = P(
    "notebook", "notebooks/m1_physics_tour.ipynb",
    "Explores and plots the physics checks; all logic lives in the package. It uses `load_config` and "
    "`derived`, `EngineBank` (valve step and the engines-off command), `LanderSim` (misalignment drift, storm "
    "drift, free fall, mass flow), `drag_areas` and `drag_force_body` (the storm push), `run_scenario` (the "
    "drop) and `export_video.main` (embedded video).",
    "a Jupyter kernel running the .venv", "6 figures and an embedded MP4",
)

ROOT = P(
    "entry", "main",
    "The program entry. Every command is `python -m scripts.<name> [options]`, and Python runs that script's "
    "`main()`. This root stands for that dispatch, so the whole project forms one tree. Its children go from "
    "the simplest command to the most complex, followed by the two other ways the library is driven: the "
    "test suite and the notebook.",
    "command line: which script, plus its options (argv)",
    "exit status; the command's effects (printed report, files in runs/ or docs/, a viewer window)",
    FETCH, CHECK_CONFIG, EXPORT_VIDEO, WATCH, BENCHMARK, TESTS, NOTEBOOK,
)
