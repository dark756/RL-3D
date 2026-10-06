# Mars Lander RL — Viking lander on Mars (MuJoCo)

A 3-D simulator of NASA's **Viking lander (1976)** in the final few hundred metres of a Mars landing. An RL
agent will learn to land it against the environment (dust storms, slopes, faults). There is **no manual
flight**: only the agent's commands fly it. The project brief is [claude.md](claude.md).

**Status: milestone M1 done, then simplified (2026-10-05).**
- Physics, Viking model, Mars air and dust-storm wind, rigid legs, radar sensors with Viking's errors.
- Live viewer with plumes and a HUD, plus headless MP4 export.
- Scripted test scenarios and 37 headless tests.
- The simplification trimmed `.env` from 155 to 41 facts. [docs/env_variables.md](docs/env_variables.md) explains
  every variable and why the rest went.
- Next is **M2**: Gymnasium environment, logging and replay, and the rest of Open decisions O2–O4.

| | |
|---|---|
| Python | **3.13.5** (the project `.venv`) |
| MuJoCo | **3.14.0** (all pins in [requirements.txt](requirements.txt)) |
| OS tested | Windows 11, i5-12450H, RTX 3050 6 GB; training runs on CPU (D9) |

---

## 1. Quick start (Windows PowerShell)

```powershell
cd "C:\Users\gurka\Desktop\RL-main\3d env"
.venv\Scripts\Activate.ps1                     # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install -r requirements.txt      # pinned versions (torch is the CPU build)
python -m pytest                               # headless test suite, ~1 minute
python -m scripts.check_config                 # every fact with unit, status and source
python -m scripts.watch --scenario drop        # opens the 3-D viewer
python -m scripts.export_video --scenario drop # writes runs/videos/drop_seed0.mp4 (no window)
```

Run every command from the project folder, as `python -m scripts.<name>`, so the `lander` package is found.

## 2. Commands

| Command | What it does |
|---|---|
| `python -m scripts.check_config` | Table of all 41 facts (value, unit, status, source, note) plus derived numbers (start mass, ṁ, T/W, hover throttle, storm push, …). |
| `… check_config --status assumed` | Only the **assumed** values: the ones to verify or improve first. |
| `… check_config --grep RADAR` | Only keys containing `RADAR`. |
| `… check_config --dump-mjcf runs/viking.xml` | Write the generated MuJoCo XML for inspection. |
| `python -m scripts.watch --scenario drop` | Live viewer. Scenarios: `drop`, `hard_drop`, `slope45`, `hover`, `freefall`. Options: `--wind 25,90` (a 25 m/s storm blowing toward 90°), `--speed 0.25` (slow motion), `--seed N`, `--max-seconds S`, `--no-hud`. |
| `python -m scripts.export_video --scenario slope45 --camera ground` | Headless MP4 to `runs/videos/`. Takes `--wind` too. Cameras: `chase` (follows the lander), `ground` (fixed near the target), `padcam` (looks down from the lander). |
| `python -m scripts.benchmark` | Simulator speed with 1, 2, 4, 6, 8 processes (`--procs 1,4`, `--sim-seconds 60`). |
| `python -m scripts.fetch_sources` | Download the NASA PDFs S3, S4, S5 into `docs/sources/` to check citations. |
| `python -m pytest` | All tests, headless. `-k slope` runs a subset; `-s` shows printed numbers. |

**Any script** also takes `--set KEY=VALUE` (repeatable) to override a fact for that run, and `--env-file PATH`.
For example: `python -m scripts.watch --scenario drop --set VIKING_ENGINE_T90_S=0.15` (slower engines).

**Viewer keys:**
- Space: pause / resume. Enter: restart the episode.
- Mouse: orbit (left), zoom (right / scroll), pan (Shift+left).
- `[` / `]`: cycle cameras. Esc: free camera. Tab: show/hide panels.

There are no flight keys, by design.

**Notebook** `notebooks/m1_physics_tour.ipynb` plots each physics check (engine lag and cut, misalignment drift,
storm push, free fall, propellant, a touchdown) and embeds a video.
- **VS Code:** open the notebook and pick the kernel "Python 3.13 (.venv mars-lander)" or the `.venv` interpreter.
- **Headless:** run `python -m nbconvert --to notebook --execute --inplace notebooks/m1_physics_tour.ipynb`.
  Use `python -m nbconvert`, not `jupyter nbconvert`: on this PC the latter starts the system-wide Jupyter,
  which has no MuJoCo.

**Linux / Colab (headless):** `export MUJOCO_GL=egl` before tests and `export_video`. The live viewer needs a desktop.

## 3. The fact file `.env`

Every physical fact and simulation setting the code uses is one line:

```
VIKING_ENGINE_THRUST_MAX_N=2667   # N | published | S2,S9 | per engine at valve setting 1 ...
KEY=value                         # unit | status | source | note
```

- **status**:
  - `published`: the number is printed in the source; the page, figure or table is in `source`.
  - `derived`: computed from published numbers; `note` says how.
  - `assumed`: our estimate; `note` says what to check.
- **what is in it**: 41 facts, each explained in [docs/env_variables.md](docs/env_variables.md) with the reason it is
  kept. Eight facts for later milestones are **parked** as commented-out lines at the end (Viking's final-phase
  guidance for M3, O3's attitude limits, the slope and friction distributions, the M5 stability target). The
  parser ignores them until you uncomment one and add its field to `lander/config.py`.
- **sources**: listed with URLs at the top of `.env` (the brief's S1–S11). Local copies of the NASA PDFs go in
  `docs/sources/`. Page numbers like `S3 p96` are PDF pages of those copies.
- **strict**: `lander/config.py` refuses to start if a line lacks a valid status or source, a key is unknown (typo),
  or a key is missing. Nothing falls back to a hidden default.
- **override without editing**: an environment variable `LANDER_<KEY>` (PowerShell:
  `$env:LANDER_VIKING_ENGINE_T90_S="0.1"`) or `--set KEY=VALUE`. An overridden value gets status `override` plus its
  origin, and M2's episode logs will record it (D5).
- **geometry frame** (as in NASA TN D-7959 Fig 3):
  - origin on the lander's centre line at the footpad plane;
  - +Z up the thrust axis, leg 1 on +Y;
  - azimuths measured from +X toward +Y.
- `.env` holds no secrets and is meant to be committed. VS Code also loads it into the environment; that's harmless,
  because only `LANDER_`-prefixed variables count as overrides.

**Where the model departs from the sources:**

| Fact | Used | Source | Note |
|---|---|---|---|
| Descent propellant | 83.9 kg (gives the dry mass 573.1 kg) | S3 p141 | S2 rounds to 85 kg |
| Footpad radius | 1.397 m | S4 Fig 3, S5 Fig 8 | S2's "2.21 m triangle" implies 1.276 m |
| CG | on the centre line (your decision) | — | S4 Fig 3 puts it 4.06 cm toward leg 1 (S3 Fig 74: 4.95 cm) |
| Engine thrust | 276–2,667 N, constant (your choice) | S2, S9 | S3 p96: 400–2,700 N at ignition, falling in blowdown |
| Constant-velocity phase (parked, M3) | from 16.8 m | S3 p104 | ~12 m (brief, S10) |

## 4. How the simulation works

- **D1 split.** `config`, `geometry`, `vehicle`, `aero`, `sensors` and `touchdown` are plain numpy
  (backend-agnostic; a test enforces it). `sim.py` applies their forces to MuJoCo. MuJoCo integrates the rigid
  body, resolves contacts and renders. Its own fluid model is off.
- **Rates.**
  - Physics: 2 ms, MuJoCo `implicitfast` integrator.
  - Control: 25 Hz, so every command is held 40 ms. That equals Viking's navigator period (S3 p105).
  - Radars: sample-and-hold every 200 ms (S3 p93).
- **Each physics step**, in order:
  1. Engines (one lag, clamp, misalignment); propellant flow uses actual thrust (D6).
  2. Drag in the episode's steady wind.
  3. Forces go to MuJoCo; the mass is updated (CG and inertia never change).
  4. `mj_step`, then contacts and the touchdown monitor.
- **One rigid body** (brief D15). The legs are rigid and end in three footpad contact spheres; the CG sits on the
  centre line at 0.79 m; the inertias are NASA's touchdown values; mass = 573.1 kg dry + propellant left.
- **Engines.** The command is Viking's **valve setting** u in [0, 1] (S3 p96). Thrust = 276 N + u × (2,667 − 276) N,
  so u = 0 is minimum thrust, not off. Thrust follows the command with one first-order lag (t90 = 53 ms, Viking's
  shutdown trace, S4 Fig 6b). A torque-free hover needs engine 3 about 2.5 % lower (it sits a little further out).
- **Engines off** (brief D7). The agent's `engines_off` command, or an empty tank, shuts all three engines down for
  good. Thrust then decays with the same lag and nothing relights them, like Viking's shutdown valves.
- **Misalignment** (0.044°, random direction each episode) is the only steady disturbance. Uncorrected, it tilts a
  hovering lander by tens of degrees and spins it at 2–9°/s within 30 s. There are no roll jets, so spin is never
  corrected; it is harmless with rigid legs.
- **Air** (S11). Constant density 0.016 kg/m³; quadratic drag in body axes. Wind is one steady vector per episode:
  calm episodes have none, and dust storms are 17–30 m/s, which pushes with 5–15 N. The lateral C_d·A is calibrated
  to S4's touchdown aero forces.
- **Sensors** (S3). Both radar antennas sit at the centre of the underside.
  - IMU at the CG, ideal: Viking's IMU errors were 10⁴–10⁶× smaller than the signals.
  - Radar altimeter: first return = nearest surface point; readings below 41 m are discarded.
  - 4-beam Doppler radar: 21.5° splay; a beam needs ≥ 1 m/s closing speed, so it has **no lock while hovering**.
  - Viking's published radar errors are on by default (`SIM_SENSOR_NOISE_SCALE=1`).
- **Touchdown and success** (brief §7):
  - A leg breaks if first contact is faster than 3.34 m/s down or 1.22 m/s sideways (S5); that decides the landing
    at once.
  - Stability distance: CG to the vertical planes through footpad pairs (S4). Negative means it tips over.
  - Settled: resting on all three feet with kinetic energy below 0.5 % of impact KE for 0.5 s. The rigid legs
    rebound a little (~13 % of the impact speed) first.
  - The engines must be off by the end of the episode.
  - The pad-radius and tilt limits are Open decision O3 (M2).
- **Determinism** (D10): one seed → numpy streams spawned in a fixed order; same seed + config + commands is
  bit-identical on the same machine and MuJoCo version (tested).

## 5. Validation

| Check | Simulator | Reference |
|---|---|---|
| CG height / footpad radius | 0.5658 | 0.566 (S4 p5) |
| Engine shutdown: thrust down to 10 % | 54 ms after the command | t90 53 ms (S4 Fig 6b) |
| Level hover, no misalignment, 10 s open loop | tilt 0.000°, drift 0.000 m, height ±2 cm | stays level (fixed CG) |
| Nominal 2.44 m/s touchdown, engines cut on contact | lands, settles | — |
| Tip-over on a 45° slope (24 drops) | 1.00 (rigid legs: feet slide, μ 0.6 < tan 45°) | 0.85 with Viking's crushable legs (S4 p15) |
| Tip-over on 10°, 20°, 30° slopes | 0.00 | — |
| Free fall vs closed form | error ≤ g·t·dt/2 (integrator order) | — |
| Propellant used vs ∫T/(Isp g₀) dt | equal to 1e-12 | brief D6 |
| Storm push, 30 m/s | 14.5 N | 57.6 N at 51 m/s in dense air calibrates C_d·A (S4 Table II) |
| Storm drift after 30 s, uncorrected | 0.22 / 0.48 / 0.69 m/s at 17 / 25 / 30 m/s | — |
| Radar noise | matches the published 1σ | S3 p109, p112 |

**Speed on this laptop** (measured on the previous, heavier M1 model; re-run `python -m scripts.benchmark` on an idle
machine to update):

| Processes | Control steps/s | Real time | Speed-up |
|---|---|---|---|
| 1 | 546 | 22× | 1× |
| 4 | 1,600 | — | 2.9× |
| 8 | 2,331 | — | 4.3× |

At 8 processes, 10M environment steps took about 1.2 h of simulation.

## 6. Where to run: locally

- **Local (recommended):**
  - The live viewer needs a local window.
  - PPO with small networks is limited by CPU-side simulation (D9). This laptop's 8 cores beat free Colab's 2 vCPUs,
    and the GPU barely helps there. The GPU is still used for rendering.
  - Keep the laptop on mains power in "performance" mode, with games and other heavy programs closed, during
    training and benchmarks.
- **Colab:** optional extra capacity for headless seeds or fuzzing batches (`MUJOCO_GL=egl`).
  - Pins may need adjusting for Colab's Python version.
  - A replay is bit-identical only on the machine and MuJoCo version that recorded it.

## 7. Decisions and simplifications

**Your decisions:**
- No human or manual flight. The brief is updated: §8, the M1 row and §14.
- No `git init`. Episode logs (M2) will record `"unversioned"`.
- Fixed thrust envelope; blowdown not modelled.
- Notebooks used alongside the package, not instead of it.
- Run locally.
- **The 2026-10-05 simplification** (details and reasons in [docs/env_variables.md](docs/env_variables.md)):
  - rigid legs that break above 3.34 m/s down or 1.22 m/s sideways;
  - the agent cuts the engines (permanent, must be off by the end);
  - episodes start at 200–500 m;
  - one agent learns calm and storm landings;
  - CG fixed on the centre line, mass from fuel burned, constant inertia;
  - misalignment kept, roll jets removed;
  - Viking's radar errors on;
  - one engine lag.

**Choices made while building:**
- MJCF is generated from `.env` at run time (`lander/mjcf.py`); there is no hand-written XML.
- Control rate is 25 Hz.
- Episode logging and `replay` move to M2, as the M2 done-criterion says.
- The ground is a mocap body, so slope and direction can change per episode without recompiling.
- The scripted scenarios cut the engines on the latched first contact, because the rigid legs rebound slightly.

**Simplifications** (also in `.env` notes and `docs/env_variables.md`):
- Footpad contact is a 3.2 cm sphere at the junction (the real pad is a 45 cm cone). The disc is visual only.
- No aero torque; no gusts; constant air density.
- One radar altimeter (Viking had two).
- The hydrazine plume is a visual aid.

## 8. Code map: every file and function

### `lander/` — the package

**`lander/config.py`** — loads `.env` into typed, read-only dataclasses and enforces the metadata.
- `ConfigError`: raised for any malformed, missing, unknown or inconsistent value.
- `Param`: one fact (`key`, `value`, `unit`, `status`, `source`, `note`, `origin`).
- `G0_MPS2`: standard gravity (a definition, so a constant rather than a fact).
- `EngineCfg`, `MassCfg`, `LegCfg`, `AeroCfg`, `RadarCfg`, `RequirementsCfg`, `ReferenceCfg`, `MarsCfg`, `WindCfg`,
  `TerrainCfg`, `SimCfg`: one dataclass per key prefix. Field name = key minus prefix, lower case.
- `SimCfg.control_dt_s`: control period (1 / `SIM_CONTROL_HZ`).
- `SimCfg.substeps`: physics steps per control step.
- `Config`: all groups plus the raw `params`.
  - `Config.param`: full record for one key.
  - `Config.by_status`: e.g. all `assumed` facts.
  - `Config.overridden`: the overridden facts.
  - `Config.to_log_dict`: JSON-ready dump for logs.
  - `Config.with_overrides`: a copy with some values replaced (marked `override`).
- `parse_value`: `"3"`, `"0.5"` or `"1,2,3"` → int, float or tuple.
- `parse_env_file`: reads `.env` into `{key: Param}`, enforces `# unit | status | source | note`, skips comments.
- `build_config`: maps params onto the dataclasses and rejects missing or unknown keys.
- `load_config`: `.env`, then `LANDER_<KEY>` environment overrides, then explicit overrides.
- `parse_set_args`: CLI `["KEY=VALUE", …]` → dict.
- `validate`: cross-field checks (3 engines, 3 legs, thrust order, inertia triangle inequality, whole substeps, …).
- `derived`: computed sanity numbers `{name: (value, unit, formula)}`, shown by `check_config`.

**`lander/geometry.py`** — fixed positions from the config (numpy only).
- `unit_azimuth`: horizontal unit vector at an azimuth.
- `cross3`: fast 3-vector cross product (np.cross was a bottleneck).
- `LanderGeometry`: footpads, CG, engine positions, radar antenna position, Doppler beam directions.
- `hexagon_vertices`: body outline from the S2 side lengths (an equilateral triangle with the corners cut).
- `build_geometry`: builds `LanderGeometry` from a `Config`.

**`lander/vehicle.py`** — engines, propellant and mass (numpy only).
- `Command`: actuator command held for one control period. `throttle` = 3 valve settings 0..1; `engines_off` = shut
  the engines down for good.
- `MassModel`: dry mass + propellant, fixed CG, constant inertia.
  - `MassModel.total`: whole-vehicle mass for the propellant left.
- `EngineBank`: the three engines.
  - `EngineBank.valve_to_thrust` / `EngineBank.thrust_to_valve`: Viking valve convention.
  - `EngineBank.reset`: draws each engine's misalignment direction.
  - `EngineBank.command`: latches a new command; `engines_off` shuts the engines for good.
  - `EngineBank.step`: lag, clamp and fault scale; returns actual thrust.
- `VehicleStep`: one physics step's thrust, body-frame force and torque about the CG, flow, empty flag.
- `Vehicle`: engines + propellant + mass model.
  - `Vehicle.reset`: new episode.
  - `Vehicle.command`: forwards a `Command`.
  - `Vehicle.step`: advances one physics step; returns `VehicleStep`.
  - `Vehicle.trim_thrust`: per-engine thrust for a total thrust with zero pitch/yaw torque about the CG.

**`lander/aero.py`** — Mars air and wind (numpy only).
- `drag_force_body`: quadratic drag per body axis.
- `drag_areas`: (C_d·A) per body axis from the config.
- `wind_vector`: the episode's steady wind from a speed and the direction it blows toward.

**`lander/sensors.py`** — IMU, radar altimeter, Doppler landing radar (numpy only; terrain queries injected).
- `SensorReadings`: one sample (`accel`, `gyro`, `altitude`, `doppler_beam`, `doppler_vel`; NaN = invalid).
  - `SensorReadings.altitude_valid` / `SensorReadings.doppler_valid`: validity.
- `SensorSuite`: all sensors.
  - `SensorSuite.reset`: new noise stream, clears the radar holds.
  - `SensorSuite.sample`: reads every sensor from the truth state and a terrain object.

**`lander/touchdown.py`** — the brief §7 outcome for the simplified lander (numpy only).
- `stability_distance`: CG distance to the nearest vertical plane through two footpads (negative = tipping).
- `TouchdownReport`: everything measured, plus `criteria` (True / False / None = decided at M2).
  - `TouchdownReport.done`: settled, tipped over, broken leg, or body strike after touchdown.
  - `TouchdownReport.success`: done and every decided criterion passed (including engines off).
  - `TouchdownReport.failures`: names of failed criteria.
- `TouchdownMonitor`: fills the report from the physics steps.
  - `TouchdownMonitor.reset`: clears it for a new episode.
  - `TouchdownMonitor.note_engines`: records the engine state every step, even after landing.
  - `TouchdownMonitor.update`: called every physics step near the ground.

**`lander/mjcf.py`** — builds the MuJoCo model from the config (no hand-written XML).
- `ModelIndex`: MuJoCo ids and addresses the simulator uses every step.
- `terrain_normal`: unit normal of a slope falling toward an azimuth.
- `quat_z_to`: quaternion rotating +Z onto a vector.
- `build_mjcf`: the XML string.
  - One free lander body with three footpad contact spheres; legs drawn as struts.
  - Engines, tanks, RTG covers, antenna; sites, IMU sensors, cameras.
  - A mocap ground with the target marker.
- `build_model`: compiles the XML and collects a `ModelIndex`.
- `set_terrain`: tilts the ground (mocap body) and sets footpad friction for an episode.

**`lander/sim.py`** — `LanderSim`, the MuJoCo glue.
- `quat_to_mat`, `quat_mul`, `quat_from_axis_angle`: quaternion helpers.
- `quat_from_tilt`: attitude from heading + tilt + tilt direction.
- `tilt_of`: angle of the thrust axis from vertical.
- `plane_height`: ground height of a plane at (x, y).
- `InitialState`: full initial state (CG position and velocity, attitude, body rates, propellant, starting throttles).
- `ResetOptions`: per-episode conditions and faults: wind speed and direction, slope, friction, disturbance
  force/torque, engine fault scale, engines off at the start, sensor noise.
  - `ResetOptions.from_any`: builds it from a dict (rejects unknown keys).
- `Telemetry`: ground-truth snapshot each control step.
  - `Telemetry.vz`: vertical speed.
  - `Telemetry.vh`: horizontal speed.
  - `Telemetry.spin_dps`: spin about the lander's own vertical axis.
- `StepResult`: telemetry + sensor readings + touchdown report + events.
- `PlaneTerrain`: ground queries for the sensors.
  - `PlaneTerrain.height`: ground height at (x, y).
  - `PlaneTerrain.height_above`: vertical height of a point.
  - `PlaneTerrain.nearest`: closest ground point.
  - `PlaneTerrain.raycast`: MuJoCo ray against the ground.
- `LanderSim`: the simulator.
  - `LanderSim.t`: simulation time.
  - `LanderSim.engines_off`: true once the engines are shut down.
  - `LanderSim.hover_trim`: torque-free valve settings for a total thrust (default: the current weight).
  - `LanderSim.state_above_ground`: `InitialState` with the lowest footpad a given height above the ground.
  - `LanderSim.reset`: new episode from a seed, initial state and options.
  - `LanderSim.step`: one control period.
  - `LanderSim.telemetry`: current truth snapshot.
  - `LanderSim.read_sensors`: current sensor readings.

**`lander/scenarios.py`** — scripted open-loop cases (no controller, no RL).
- `hover`: policy: thrust = weight, engines never cut.
- `hover_then_cut`: policy: thrust = weight, engines off at the first footpad contact.
- `Scenario`: name, description, duration, initial state, options, policy, seed.
- `make_scenario`: builds `drop`, `hard_drop`, `slope45`, `hover` or `freefall`.
- `run_scenario`: runs one until its duration or a decided landing.

**`lander/visuals.py`** — scene overlays shared by the viewer and the video.
- `add_geom`: append one decorative geom to a scene.
- `add_plumes`: three nested translucent plumes per engine, sized by actual thrust.
- `add_arrow`: an arrow (velocity is green, wind is blue).
- `decorate`: all overlays for the current state.
- `tracking_camera`: camera that follows the lander.
- `named_camera`: `chase`, or a fixed MJCF camera (`ground`, `padcam`).
- `wind_heading_deg`: direction the wind blows toward.

**`lander/hud.py`** — HUD text (brief §8).
- `status_text`: one-line flight / landing status (including "leg broken" and "engines still firing").
- `hud_lines`: (label, value) rows: altitude radar/true, speeds, tilt, spin, propellant, thrusts, engines, wind,
  status.
- `hud_columns`: the same as two text columns for MuJoCo's overlay.

**`lander/video.py`** — offscreen rendering and MP4 export.
- `overlay_hud`: draws the HUD panel and status banner on a frame (Pillow).
- `FrameRenderer`: renders state + overlays + HUD to an RGB array.
  - `FrameRenderer.render`: one frame.
  - `FrameRenderer.close`: release the renderer.
- `VideoWriter`: MP4 writer, one frame per control step (real time at 25 fps).
  - `VideoWriter.add`: append a frame.
  - `VideoWriter.close`: finish the file.

**`lander/viewer.py`** — live MuJoCo viewer for watching (no manual flight).
- `watch`: runs a policy in real time (or slow motion) with plumes and HUD; Space pauses, Enter restarts.

### `scripts/` — command-line tools (`python -m scripts.<name>`)

- **`scripts/common.py`**
  - `add_config_args`: adds `--set` and `--env-file` to a parser.
  - `config_from_args`: loads the config with those overrides.
  - `add_wind_arg`: adds `--wind SPEED[,DIRECTION]`.
  - `wind_options`: turns `--wind` into reset options.
  - `summarize`: readable outcome of an episode.
- **`scripts/check_config.py`**
  - `format_value`: prints numbers and lists compactly.
  - `print_table`: the facts table by section.
  - `main`: CLI.
- **`scripts/watch.py`**
  - `main`: live viewer for a scenario.
- **`scripts/export_video.py`**
  - `main`: headless MP4 of a scenario; returns the output path.
- **`scripts/benchmark.py`**
  - `run`: aggregate control steps/s with N worker processes.
  - `main`: CLI.
- **`scripts/fetch_sources.py`**
  - `main`: downloads S3/S4/S5 into `docs/sources/`.

### Other files

| File | Purpose |
|---|---|
| `.env` | All facts and settings (see section 3). |
| `requirements.txt` | Pinned dependencies (your "req.txt"). |
| `pytest.ini` | Test settings (`pythonpath`, `testpaths`). |
| `.gitignore` | Ignores `.venv/`, `runs/`, caches, downloaded PDFs. **Not** `.env`. |
| `notebooks/m1_physics_tour.ipynb` | Plots of the physics checks plus a video. |
| `docs/env_variables.md` | Every `.env` variable explained, the error-data review, and why the rest were removed. |
| `docs/sources/` | Local copies of the NASA PDFs (`scripts/fetch_sources.py`). |
| `docs/Mars_Lander_RL_code_guide_v2.pdf` | Code guide v2: every `.env` variable with why it was kept and the functions that read it (Section 2, including where each key of the old 155-key file went), then the call tree from `main` with every node explained in in-order (LVR) and the data passed between nodes. |
| `docs/Mars_Lander_RL_code_guide.pdf` | Code guide v1: the same call tree without the variables section. |
| `runs/` | Outputs: videos now; logs, checkpoints and TensorBoard later. |
| `claude.md` | The project brief. |

## 9. Tests (`python -m pytest`)

- **`tests/conftest.py`**: fixtures. `cfg` ignores `LANDER_*` overrides; `make_sim(overrides, seed)`; `CLEAN` = no air,
  no misalignment, perfect radar readings.
- **`tests/test_config.py`**:
  - Every fact has a status and source; parked facts are comments only.
  - Missing metadata, a bad status or a sourceless "published" value are rejected.
  - Unknown and missing keys are rejected.
  - Env-var and CLI overrides are recorded.
  - Derived numbers match the brief (ṁ 1.39/0.14 kg/s, start mass 633.1 kg, T/W 3.41, storm push ~14.5 N,
    CG ratio 0.566).
- **`tests/test_model.py`**:
  - One free body (no joints), three footpads, three engines.
  - Mass = dry + propellant at every step; the CG and inertia never change.
  - Footprint matches S4/S5.
  - Editing `.env` changes the compiled model.
- **`tests/test_physics.py`** (brief §11):
  - Free fall vs closed form.
  - Hover holds altitude and stays level with no misalignment (10 s open loop).
  - Misalignment tilts and spins an uncontrolled lander.
  - Mass flow = ∫T/(Isp g₀).
  - The engines-off command: thrust reaches 10 % after t90 and never relights.
  - Determinism: bit-identical runs; a different seed changes misalignment and noise.
  - Rest on flat ground is stable (the dt check).
- **`tests/test_legs.py`**:
  - A leg breaks above 3.34 m/s down or 1.22 m/s sideways, not below.
  - Slopes: 45° tips (≥ 0.6; NASA's 0.85 printed for comparison), 10° never tips.
- **`tests/test_aero_sensors.py`**:
  - Drag: a tenth of a newton in calm descent, 5–15 N in a dust storm.
  - Storm drift after 30 s matches the review.
  - Ideal radar altimeter, including the 41 m cut-off.
  - Ideal Doppler velocity, and no lock in hover.
  - Ideal IMU.
  - Radar noise matches Viking's published levels.
- **`tests/test_touchdown.py`**:
  - Stability-distance geometry.
  - Nominal drop succeeds and events come in order.
  - The hard drop breaks a leg.
  - Free fall crashes.
  - A landing only counts once the engines are cut.
  - Running out of propellant airborne is flagged.
- **`tests/test_video_docs.py`**:
  - Offscreen frame and MP4 work.
  - This README names every module, class and function.
  - The vehicle layer never imports MuJoCo.

## 10. Milestones

| M | Scope | State |
|---|---|---|
| M1 | Physics, Viking model, Mars air + wind, legs, viewer (plumes + HUD), video, scripted scenarios; simplified 2026-10-05 | **done** |
| M2 | Gymnasium env, `reset(seed, options)`, logging + replay; resolve O2–O4 first (O1 decided: start at 200–500 m) | next |
| M3 | Classical controller (guidance → 2.44 m/s → cut engines on contact; attitude by differential throttle) | |
| M4 | PPO (SAC if needed), `VecNormalize`, checkpoints, TensorBoard, CPU, ≥ 5 seeds | |
| M5 | Evaluation with 95 % Wilson CIs on calm and storm test sets, comparison with M3, videos | |
