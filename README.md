# Mars Lander RL — Viking lander on Mars (MuJoCo)

> **One sentence:** a MuJoCo 3-D simulator of NASA's 1976 Viking lander in Mars terminal descent, with a Gymnasium
> environment, a classical controller that lands it, and a PPO (Stable-Baselines3) agent being trained to do the same.

## At a glance

**Architecture**

```
.env facts (value, unit, status, source)
   -> lander/ vehicle layer (engines, lag, fuel, drag, wind, radar errors)
   -> MuJoCo (rigid body, contacts, rendering)
   -> envs/ Gymnasium MarsLander-v0 (onboard navigation -> 22 observations, 4 actions, reward)
   -> controllers/classical.py   (Viking's descent scheme + steering)   [baseline, M3]
   -> controllers/ppo.py + scripts/train.py (SB3 PPO, VecNormalize, 5 seeds, curriculum)   [M4]
   -> scripts/evaluate.py (fixed calm and storm test sets, Wilson 95 % intervals)
```

**Hardware constraints.** Built and run on one laptop: Intel i5-12450H (8 cores) and an RTX 3050 with 6 GB VRAM,
Windows 11, Python 3.13. Training is CPU-only by design (small MLPs, simulator-bound), so the 6 GB GPU is not
needed and no cloud compute is used. The whole pipeline, five training seeds included, fits in that one machine:
the simulator was sped up 2.7x with bit-identical results, and five seeds with two environment workers each train
at 10.0 M steps an hour in total.

**Results so far** (all numbers measured in this repository, see sections 6-8)
- Classical controller: 1,000 of 1,000 fixed test episodes landed (500 calm, 500 dust storm; 95 % Wilson interval
  [99.2, 100] % per set), all within 2.3 m of the target; median propellant 40 kg of 60 kg.
- PPO agent: first night, 5 seeds x about 18 M steps (8 h), **did not converge** (no full-range test flight landed;
  the curriculum stalled at levels 0.0-0.5). The cause was unstable PPO updates plus a hovering trap, and a fix
  (target-KL cap of 0.02) is built and tested but its run is still to come. No trained-agent success rate is
  claimed.
- Verification: 75 headless tests pass; episode logs replay bit-identically.

**Run it (3 steps, Windows PowerShell)**

```powershell
py -3.13 -m venv .venv; .venv\Scripts\python -m pip install -r requirements.txt      # 1. install
.venv\Scripts\python -m pytest                                                        # 2. check (about 30 s)
.venv\Scripts\python -m scripts.evaluate --episodes 100                               # 3. classical controller lands 100 calm + 100 storm episodes
```

Add `python -m scripts.watch --scenario drop` for the live 3-D viewer, or `python -m scripts.train` to train the
agent (8 h, five seeds).

A 3-D simulator of NASA's **Viking lander (1976)** in the final few hundred metres of a Mars landing. An RL
agent will learn to land it against the environment (dust storms, slopes, faults). There is **no manual
flight**: only the agent's commands fly it.

**Status: milestone M4 built (2026-10-11): the PPO training is ready to run at night; M5 (evaluation) after it.**
- **M1:** physics, Viking model, Mars air and dust-storm wind, rigid legs, radar sensors with Viking's errors; live
  viewer with plumes and a HUD, headless MP4 export, scripted scenarios.
- **M2:** the Gymnasium environment `MarsLander-v0` (section 5), with onboard navigation, a start distribution,
  episode limits, a reward, episode logs and bit-identical replay.
- **M3:** the classical controller (section 6): Viking's terminal-descent scheme plus steering to the target, flying
  the agent's observations; an evaluation script with fixed calm and storm test sets.
- **M4:** PPO training (section 7): five seeds in parallel, a curriculum from low starts to the full range,
  validation, checkpoints, resume, learning curves. The simulator steps 2.7× faster than at M3, with bit-identical
  results.
- 75 headless tests.
- `.env` has 101 facts: 44 for the lander, Mars and the simulation, 18 for the training episode and reward, 15
  for the classical controller and 24 for training the agent. [docs/env_variables.md](docs/env_variables.md) explains every variable and why the
  rest went.

| | |
|---|---|
| Python | **3.13.5** (the project `.venv`) |
| MuJoCo | **3.14.0** (all pins in [requirements.txt](requirements.txt)) |
| OS tested | Windows 11, i5-12450H, RTX 3050 6 GB; training runs on CPU (D9) |

---

## 1. Quick start (Windows PowerShell)

```powershell
cd "C:\Users\gurka\Desktop\RL-3D"              # the project folder (wherever you cloned it)
py -3.13 -m venv .venv                         # first time on a computer only: .venv is not in git
.venv\Scripts\Activate.ps1                     # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install -r requirements.txt      # pinned versions (torch is the CPU build)
python -m pytest                               # headless test suite, ~30 s on this laptop
python -m scripts.check_config                 # every fact with unit, status and source
python -m scripts.watch --scenario drop        # opens the 3-D viewer
python -m scripts.export_video --scenario drop # writes runs/videos/drop_seed0.mp4 (no window)
python -m scripts.run_episodes --episodes 3 --log runs/episodes --trajectory   # the controller flies the env
python -m scripts.replay runs/episodes/<a file it wrote>.json --watch              # watch one again
python -m scripts.evaluate --episodes 100                                          # success rate, calm and storm
python -m scripts.train                                   # PPO, 5 seeds, 8 h (run it at night) -> runs/train/<run>
python -m scripts.training_report runs/train/<run>        # where each seed stands + learning_curves.png
```

Run every command from the project folder, as `python -m scripts.<name>`, so the packages are found.

## 2. Commands

| Command | What it does |
|---|---|
| `python -m scripts.check_config` | Table of all 101 facts (value, unit, status, source, note) plus derived numbers (start mass, ṁ, T/W, hover throttle, storm push, …). |
| `… check_config --status assumed` | Only the **assumed** values: the ones to verify or improve first. |
| `… check_config --grep RADAR` | Only keys containing `RADAR`. |
| `… check_config --dump-mjcf runs/viking.xml` | Write the generated MuJoCo XML for inspection. |
| `python -m scripts.watch --scenario drop` | Live viewer. Scenarios: `drop`, `hard_drop`, `slope45`, `hover`, `freefall`. Options: `--wind 25,90` (a 25 m/s storm blowing toward 90°), `--speed 0.25` (slow motion), `--seed N`, `--max-seconds S`, `--no-hud`. |
| `python -m scripts.export_video --scenario slope45 --camera ground` | Headless MP4 to `runs/videos/`. Takes `--wind` too. Cameras: `chase` (follows the lander), `ground` (fixed near the target), `padcam` (looks down from the lander). |
| `python -m scripts.run_episodes --episodes 3` | Runs the training environment with a policy and prints each outcome. The default `--policy classical` is the M3 controller. `--policy random` sends random actions; `--policy descend` falls slowly and cuts the engines on contact, and lands only from gentle low starts (add `--gentle`: 5–10 m up, 1–2 m/s down, calm, flat). `--log DIR` writes every episode, and `--trajectory` adds the true path for replay checks. `--policy ppo:FOLDER` flies a trained checkpoint. |
| `python -m scripts.evaluate` | Flies a policy (default `classical`) on the fixed test sets, 500 calm and 500 storm episodes, in parallel (`--procs`). Prints success and safe-landing rates with 95 % Wilson intervals, failure causes, touchdown speeds and attitude, propellant and time. `--episodes 100` for a quick run; `--out DIR` writes `summary.json` and `episodes.csv`; `--log-failures` also logs every failed episode for replay. `--policy ppo:runs/train/<run>/seed0/best` evaluates a trained agent on the same episodes. |
| `python -m scripts.train` | Trains the PPO agent (section 7): seeds 0–4 in parallel for 8 h, then saves and stops (Ctrl+C saves too). `--resume runs/train/<run>` carries on from the last checkpoints; `--hours`, `--steps`, `--seeds 0,1`, `--out DIR`, `--set KEY=VALUE`. Keeps the laptop awake and runs at below-normal priority. |
| `python -m scripts.training_report runs/train/<run>` | One line per seed (steps, hours, curriculum level, success there, validation calm/storm, best) and `learning_curves.png`. Live curves: `tensorboard --logdir runs/train/<run>`. |
| `python -m scripts.replay runs/episodes/<file>.json` | Re-runs a logged episode and checks it matches the log bit for bit. `--video out.mp4` exports it; `--watch` shows it in the viewer. |
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

**Code guides:** the PDFs in `docs/` are built by `tools/code_guide/` (see its README). They describe the code as of
2026-10-06 (commit `006e351`) and only build from that version.

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
  - `assumed`: our estimate or design choice; `note` says what to check.
- **what is in it**: 101 facts (31 published, 5 derived, 65 assumed), each explained in
  [docs/env_variables.md](docs/env_variables.md) with the reason it is kept.
  - 44 describe the lander, Mars and the simulation.
  - 18 (`EPISODE_*`, `REWARD_*`) define the training episode and its reward. They are design choices, so all are
    assumed.
  - 15 belong to the classical controller: Viking's final-phase guidance (`VIKING_SEQ_*`, published) and the
    controller's gains (`CTRL_*`, assumed). Only `controllers/classical.py` reads them.
  - 24 set how the agent is trained: the curriculum (`CURRICULUM_*`) and the PPO settings (`PPO_*`),
    all assumed. Only the training reads them; the test sets never depend on them.
  - One fact for M5, the gear stability target, is **parked** as a commented-out line at the end. The parser
    ignores it until you uncomment it and add its field to `lander/config.py`.
- **sources**: listed with URLs at the top of `.env` (the brief's S1–S11). Local copies of the NASA PDFs go in
  `docs/sources/`. Page numbers like `S3 p96` are PDF pages of those copies.
- **strict**: `lander/config.py` refuses to start if a line lacks a valid status or source, a key is unknown (typo),
  a key is missing, or values contradict each other (e.g. a start above the ceiling). Nothing falls back to a hidden
  default.
- **override without editing**: an environment variable `LANDER_<KEY>` (PowerShell:
  `$env:LANDER_VIKING_ENGINE_T90_S="0.1"`) or `--set KEY=VALUE`. An overridden value gets status `override` plus its
  origin, and every episode log records it (D5).
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
| Drag C_d·A | 2.01 m² in every direction (your cut, 2026-10-10) | S4 Table II | Calibrated on sideways flow; along the thrust axis it adds ~1.5 N at 30 m/s |
| Doppler beams | in the body X-Z and Y-Z planes (your cut, 2026-10-10) | S3 Fig 60 | Viking's were turned 45° about Z |
| Constant-velocity phase (the controller's gate) | from 16.8 m | S3 p104 | ~12 m (brief, S10) |

## 4. How the simulation works

- **D1 split.** `config`, `geometry`, `vehicle`, `aero`, `sensors`, `navigation` and `touchdown` are plain numpy
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
- **Air** (S11). Constant density 0.016 kg/m³; quadratic drag with one C_d·A for every direction, calibrated to S4's
  touchdown aero forces. Wind is one steady vector per episode: calm episodes have none, and dust storms are
  17–30 m/s, which pushes with 5–15 N.
- **Sensors** (S3). Both radar antennas sit at the centre of the underside.
  - IMU at the CG, ideal: Viking's IMU errors were 10⁴–10⁶× smaller than the signals.
  - Radar altimeter: first return = nearest surface point; readings below 41 m are discarded.
  - 4-beam Doppler radar: 21.5° splay; a beam needs ≥ 1 m/s closing speed, so it has **no lock while hovering**.
  - Viking's published radar errors are on by default (`SIM_SENSOR_NOISE_SCALE=1`).
- **Onboard navigation** (`navigation.py`): what the lander knows, and all the agent sees (section 5).
  - Each new radar reading sets the height of the feet or the velocity.
  - Between readings, and in the blind zones, the ideal IMU carries the estimate forward. The blind zones are below
    41 m for the altimeter and under 1 m/s closing speed for the Doppler.
  - The horizontal position relative to the target is handed over at the start and then integrated.
- **Touchdown and success** (brief §7):
  - A leg breaks if first contact is faster than 3.34 m/s down or 1.22 m/s sideways (S5); that decides the landing
    at once.
  - First contact must also come within 5° of tilt and 7°/s of pitch/yaw rate (S3 Table 8; spin is free).
  - Stability distance: CG to the vertical planes through footpad pairs (S4). Negative means it tips over.
  - Settled: resting on all three feet with kinetic energy below 0.5 % of impact KE for 0.5 s. The rigid legs
    rebound a little (~12 % of the impact speed) first. The ground's contact time constant is 10 ms (D27): at 5 ms a
    lander at rest could micro-bounce and never count as settled.
  - The engines must be off by the end of the episode, and a success settles within 20 m of the target.
- **Determinism** (D10): one seed → numpy streams spawned in a fixed order; same seed + config + commands is
  bit-identical on the same machine and MuJoCo version (tested, and checked by every replay).

## 5. The training environment (M2)

```python
from envs.mars_lander_env import MarsLanderEnv      # or: import envs; gymnasium.make("MarsLander-v0")
env = MarsLanderEnv()                               # MarsLanderEnv(overrides={...}) changes facts for every episode
obs, info = env.reset(seed=0)                       # info["episode"] holds every value of this start
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

**Action** (D18): four numbers in [−1, 1].
- The first three are the engines' valve settings, u = (a + 1) / 2: −1 gives 276 N, +1 gives 2,667 N.
- The fourth shuts all engines down for good when it is > 0 (D7).

**Observation** (D21): 22 numbers, all things the lander itself can know (`OBS_NAMES`).

| Names | What | From |
|---|---|---|
| `pos_x`, `pos_y` | Where the lander is relative to the target | Navigation: handed over at the start, then integrated |
| `height` | Feet above the ground | Altimeter; carried forward below 41 m |
| `vel_x`, `vel_y`, `vel_z` | Velocity | Doppler radar; carried forward by the IMU without lock |
| `tilt_x`, `tilt_y` | Which way and how far the thrust axis leans | IMU |
| `rate_x`, `rate_y`, `rate_z` | Body rates (z is the uncontrolled spin) | IMU |
| `valve_1`–`valve_3` | Valve settings commanded last step | Own command |
| `leg_1`–`leg_3` | The leg has touched the ground (latched) | Leg switches |
| `engines_off` | Engines shut down | Own state |
| `propellant_kg` | Propellant left | Tank gauge |
| `altimeter_ok`, `doppler_ok` | The latest radar readings were valid | Radars |
| `time_s` | Time since the start | Clock |

- **Heading frame.** Horizontal vectors use the world axes turned about the vertical to the lander's own heading.
  `x` is where the body X axis faces, so the agent sees everything relative to its engines.
- **No weather.** The agent is not told the wind (D17).
- **Measured errors** at 300 m (1σ): height 4.4 m, vertical speed 0.05 m/s, horizontal speed 0.21 m/s.
- **On slopes** the altimeter reads the distance to the nearest ground point, which is cos(slope) × the vertical
  height: up to 5 % short at 19°. That is how a first-return radar altimeter works (`sensors.py` models it so).
- **Near the ground** the height is off by the last altimeter reading's error (~1.1 m at 41 m), and more on slopes.
  Moving 36 m downhill on a 10° slope below 41 m leaves it about 6 m too low.

**Start** (D16, D19), drawn from the `EPISODE_*` facts:
- **Position:** 200–500 m up, 0–50 m from the target.
- **Motion:** falling at 10–30 m/s, moving sideways at 0–10 m/s.
- **Attitude:** tilted 0–15°, any heading, body rates up to ±2°/s.
- **Weather:** a dust storm (17–30 m/s) in 30 % of episodes, calm otherwise.
- **Ground:** slope exponential with 90 % under 11.7°, capped at 19°; friction 0.2–1.0, but at least 1.25 × tan(slope)
  so a lander at rest holds on the slope (D26; this narrows the range only on slopes above 9.1°).
- **Lander:** 60 kg of propellant, engines lit at hover trim, Viking's radar errors on.

**Options:** `reset(options=...)` sets any part of the start; missing values are sampled. Giving one never changes
the others, and `reset(options=info["episode"])` repeats an episode exactly.
- **Initial state:** `position`, `velocity`, `quat`, `omega`, `propellant_kg`.
- **Conditions:** `storm`, `wind_speed_mps`, `wind_direction_deg`, `slope_deg`, `slope_direction_deg`, `friction`.
- **Disturbances and faults:** `disturbance_force_n`, `disturbance_torque_nm`, `engine_thrust_scale`,
  `engines_enabled`, `sensor_noise_scale`.
- **Randomness:** `sim_seed`.
- **Facts for that episode alone:** `overrides`, e.g. `{"VIKING_ENGINE_T90_S": 0.1, "MARS_GRAVITY_MPS2": 3.6}`.

**How an episode ends** (D20) **and what it pays** (D21). Every other step pays 0.

| Ends with | When | Reward |
|---|---|---|
| **Landed** (success) | Settled on all three feet, engines off, within 20 m of the target | 100 inside 5 m, linear to 80 at 20 m; then minus 0.2 per second and 0.4 per degree of peak tilt above 20° |
| **Landed off the pad** | The same, but beyond 20 m | Linear from 80 at 20 m to 30 at 200 m, minus the same costs |
| **Failure at touchdown** | At first contact: a broken leg (> 3.34 m/s down or > 1.22 m/s sideways), tilt > 5° or pitch/yaw rate > 7°/s. Later: a tip-over or body strike, or engines still on or not settled 5 s after first contact | −50 within the leg speed limits, falling linearly to −100 at 6× them (20 m/s down) |
| **Failure in flight** | Tilt > 45°, more than 200 m from the target, more than 600 m up, no touchdown within 100 s, or the tank runs dry before touchdown | −100 |

A safe landing always scores more than any failure.

**`info` contents:**
- **Last step:** `outcome` (result, reason, success, reward parts, distance, time) and `is_success`.
- **Every step:** `truth` (ground truth, for debugging only) and `events`.

**Logging and replay** (D10, D22):
- `EpisodeLogger(env, "runs/episodes", trajectory=True)` writes each episode as two files:
  - **JSON:** the code version (git commit plus an uncommitted-changes flag), package versions, every fact with
    status and origin, the seed, the resolved start and the outcome.
  - **NPZ:** actions, rewards and the true trajectory.
- `python -m scripts.replay <file>.json` re-runs an episode and checks it is bit-identical.

**Rendering:** `MarsLanderEnv(render_mode="rgb_array")` makes `env.render()` return a frame with plumes and the HUD.
The scene marks the target with a green ring at 5 m and a yellow ring at 20 m.

## 6. The classical controller (M3)

```python
from controllers.classical import ClassicalController
ctl = ClassicalController(env.unwrapped.base_cfg)   # the nominal vehicle; per-episode faults stay unknown to it
obs, info = env.reset(seed=0)
ctl.reset()
obs, reward, terminated, truncated, info = env.step(ctl(obs))   # ctl.state says what it decided
```

**What it sees** (D24): the agent's 22 observations, with Viking's radar errors and blind zones, never the truth.
It also knows fixed vehicle facts: dry mass (with the fuel gauge, the mass), engine positions and thrust limits, Isp,
inertias, gravity. It keeps a little state: the tilt command and the fuel-reserve latch.

**How it flies** (D28). Steps 1, 2, 6 and 7 are Viking's scheme (S3 p96–104); steps 3–5 are ours, because Viking had
no target.
1. **Fall, then brake along a contour.** The descent speed allowed at height h is √(2.44² + 2 × 4.5 × (h − 16.8)) m/s.
   While the lander is slower, the engines idle at minimum thrust and it falls (Viking idled at 10 %). At the contour a
   speed loop brakes along it. The loop may command an average valve setting of at most 0.84; the travel above that is
   kept for attitude control (S3 p99).
2. **The gate.** From 16.8 m the lander holds 2.44 m/s straight down to contact (S3 p104). At that speed the Doppler
   radar keeps its lock, while the altimeter has been blind since 41 m.
3. **Steer to the target.** Above the gate: distance → sideways speed command (≤ 8 m/s) → acceleration → tilt
   command (≤ 20°, changing ≤ 20°/s). Below it the controller only nulls the sideways speed; the tilt cap shrinks
   from 20° at the gate to 3° at the ground, and the tilt command changes ≤ 3°/s. So first contact comes level and
   calm, even though the Doppler readings jump by a few tenths of a m/s of noise every 0.2 s.
4. **The descent waits for the sideways move.** Above the gate the descent speed is also limited to the height left
   divided by the time the move needs (`side_time`). So the lander reaches the gate over the target.
5. **Fuel reserve.** If the propellant left falls under 1.2 × what a straight-down landing from there needs
   (`fuel_to_land`), the controller gives up the target and comes straight down.
6. **Attitude.** A PD loop on the tilt (4 rad/s, damping 0.8) gives body torques. The mixer turns the total thrust
   and the torques into three engine thrusts; attitude comes first when an engine saturates. Spin is left alone (D4).
7. **Touchdown.** Engines off as soon as any leg switch trips (Viking's TESS).

The gains are the 12 `CTRL_*` facts, explained with the measurements behind them in
[docs/env_variables.md](docs/env_variables.md).

**Evaluation** (`python -m scripts.evaluate`). Two fixed test sets: episode k of the calm set uses seed
1,000,000 + k and of the storm set 2,000,000 + k. Only the weather is fixed; everything else is drawn as in training.
Training must use seeds below 1,000,000. Results are reported with 95 % Wilson intervals and no fixed pass mark (D25).

**Results** (2026-10-10, `runs/eval/m3_classical`; 362 s in 6 processes):

| | Calm (500) | Dust storm (500) |
|---|---|---|
| Success, 95 % Wilson interval | 500, 100 % [99.2, 100] | 500, 100 % [99.2, 100] |
| Distance from the target: median / max | 0.7 / 2.1 m | 0.7 / 2.3 m |
| First contact, worst: down / sideways speed | 2.51 / 0.21 m/s | 2.50 / 0.20 m/s |
| First contact, worst: tilt / pitch-yaw rate | 1.9° / 2.9°/s | 1.8° / 2.7°/s |
| Propellant used: median / 95th percentile / max | 40.1 / 46.8 / 50.1 kg | 39.7 / 46.3 / 53.1 kg |
| Time: median / max | 29.3 / 40.0 s | 29.2 / 42.4 s |
| Mean reward | 94.1 | 94.1 |

- Every landing is in the green zone (within 5 m). The limits at first contact are 3.34 m/s down, 1.22 m/s sideways,
  5° and 7°/s.
- 1,000 of 1,000 is not proof of perfection: the 95 % interval over both sets starts at 99.6 %.
- The fuel reserve was never needed. The hardest start (500 m up, 30 m/s down, 50 m off and moving 10 m/s away, in a
  30 m/s storm blowing away) lands 0.5 m from the target with 6.1 kg left. With an extra 15° of tilt it lands 0.2 m
  off with 2.5 kg left, after the reserve took over. So the 60 kg tank (D16) is enough, with a thin margin at the
  corner.

**Found and fixed while building it** (with your approval):
- **Slides (D26).** About 1.4 % of episodes paired a slope with friction below ~1.25 × tan(slope). After a normal
  2.44 m/s touchdown the lander then slid down the slope for longer than the 5 s settle window (forever below 1.0 ×).
  The training env now draws friction from at least 1.25 × tan(slope).
- **Micro-bounce (D27).** With the old 5 ms contact time constant (2.5 physics steps) a lander at rest could bounce by
  18 µm every 44 ms, so the "all three feet down for 0.5 s" check never passed. Now 10 ms.
- Before these fixes the controller landed 491 of 500 calm and 491 of 500 storm episodes. All 18 failures were these
  two effects, and all 18 land with both fixes.

## 7. Training the agent (M4)

```powershell
python -m scripts.train                                 # seeds 0-4 in parallel for 8 h -> runs/train/<date-time>
python -m scripts.train --resume runs/train/<run>       # another 8 h from each seed's last checkpoint
python -m scripts.training_report runs/train/<run>      # status table + learning_curves.png
tensorboard --logdir runs/train/<run>                   # live curves in the browser
python -m scripts.evaluate --policy ppo:runs/train/<run>/seed0/best    # the M3 test sets
```

**A night of training** (your choice, D30):
- Five seeds train in parallel, one process each, and each steps two copies of the environment in worker processes.
  This was the fastest layout measured on this laptop: 10.0 M steps an hour in total, 2.0 M per seed, so about
  16 M per seed in an 8-hour night (7.0 M an hour before the speed-up, D31).
- The processes run at below-normal priority, so anything else you run comes first, and Windows is asked not to
  sleep (the screen may still turn off). Keep the laptop plugged in with the lid open.
- After `--hours` (default 8) every seed saves and stops; Ctrl+C does the same. `--resume` carries on the next night
  from the last checkpoint, with the curriculum where it was. If `.env` changed in between, it says which facts.

**The agent** (D9, D23, D32; every setting is a `PPO_*` fact, explained in
[docs/env_variables.md](docs/env_variables.md)):
- Stable-Baselines3 PPO. Policy and value function are separate networks with two hidden layers of 256 (tanh).
  γ = 0.999 per 40 ms step (a horizon of ~40 s, one landing from 500 m); 8,192 steps per update.
- It sees the 22 observations (D21). Each is scaled by a mean and spread measured once per run on 40
  classical-controller landings over the full start range, and then kept fixed (VecNormalize with frozen
  statistics). A running average would follow the curriculum's low starts, and later heights of 200–500 m would end
  up 10+ spreads out and be clipped. Rewards are scaled by VecNormalize as usual.
- It acts with the raw four numbers (D18). Untrained, its mean action holds the valves at hover trim and keeps the
  engines lit: the engines-off output starts at −0.33 with exploration noise 0.1, so it cuts the engines with a
  0.05 % chance per step (D23).
- Each update is capped (`PPO_TARGET_KL` 0.02, since 2026-10-12). Its passes over the batch stop once the policy has
  moved about as far as the healthy updates of the first hour did. Without the cap, the first night's updates grew
  3–20× bigger as the exploration noise shrank, and the seeds forgot how to land (below). A run keeps the PPO
  settings it started with: they are saved in each checkpoint, and `--resume` says if `.env` has changed them.

**Curriculum** (your choice, D29). Level 0 starts 5–15 m up, falling 1–3 m/s, at most 5 m off target and 1 m/s
sideways, tilted at most 3°, rates within ±0.5°/s. Level 1 is D19's full range. From one level to the next every
range end grows by the same factor (the top start height by 1.42: 15 m, 21 m, 30 m … 500 m). Once 70 % of the
last 200 episodes started at the current level succeed, the level rises by 0.1. Half the training episodes start at
a random easier level, so successful landings keep coming. Weather, slope and friction are drawn as usual at every
level. Level 1 gives exactly the episodes the environment gave before the curriculum existed, so the test sets did
not change. The pilots below set these numbers; the 20–50 m first proposed taught the agent to hover.

**What gets written**, per seed in `runs/train/<run>/seed<k>/`:
- `curriculum.csv`: after every update, the level, the success rate there, and recent reward and episode length.
- `validation.csv`: every 1 M steps, the policy's mean action on 100 fixed episodes at level 1 (50 calm, 50 storm).
- `tb/`: TensorBoard logs.
- `checkpoints/latest` every 0.5 M steps and at the end, `checkpoints/step_<N>` every 5 M, and `best/`, the highest
  validation success so far.

Seeds never overlap: training environments start from 10,000 × (seed + 1), calibration uses 800,000+, validation
900,000+ and the test sets 1,000,000+.

**What an untrained agent does** (measured). Valves at hover trim plus random noise tumble the lander past 45° within
about 2 s, at any noise level from 1.0 to 0.2, at the lowest curriculum level too (40 of 40 episodes each). Random
differential thrust integrates twice into tilt. So the first thing the agent must learn is to hold its attitude.
It learns this from surviving longer, because a later failure is discounted slightly less, and from reaching the
ground: a failed touchdown scores −50 to −100, a failure in flight −100.

**The pilots** (2026-10-11): ten single-seed runs of up to 45 minutes, side by side at below-normal priority, before
the first night. Details and reasons are in [docs/env_variables.md](docs/env_variables.md).

| Pilot | Easiest start (level 0) | Steps between levels | Mixing | Noise | γ | Ran | What happened |
|---|---|---|---|---|---|---|---|
| A | 20–50 m, 2–5 m/s | equal | – | 0.3 | 0.999 | 0.25 M steps | Never left level 0 (1 % success). Learned to hover: episodes grew to 41 s, ending with an empty tank (reward −98). |
| B | 20–50 m, 2–5 m/s | equal | – | 0.1 | 0.999 | 1.5 M | Level 0.1 at 0.40 M steps, then stuck there at 47 % success; episodes kept getting longer. |
| C | 5–15 m, 1–3 m/s | equal | – | 0.3 | 0.999 | 0.67 M | Fastest at level 0 (68 %), level 0.1 at 0.25 M, then collapsed into hovering (3 % success, 34 s episodes). The step to 0.1 made the top start 4× higher. |
| D | 5–15 m, 1–3 m/s | equal | – | 0.1 | 0.999 | 0.41 M | Level 0.1 at 0.28 M, then fell to 8 % success. |
| E | 5–15 m, 1–3 m/s | equal | – | 0.3 | 0.9999 | 0.40 M | Never left level 0: hovered from the start (reward −96). |
| F | 20–50 m, 2–5 m/s | equal | – | 0.3 | 0.9999 | 0.40 M | Never left level 0: hovered (reward −100). |
| G | 5–15 m, 1–3 m/s | equal | – | 0.1 | 0.9999 | 0.35 M | Never left level 0 (1 % success). |
| H | 20–50 m, 2–5 m/s | equal | 0.5 | 0.1 | 0.999 | 0.59 M | Level 0.1 at 0.40 M, 18 % success there. |
| J | 5–15 m, 1–3 m/s | same factor | – | 0.1 | 0.999 | 0.88 M | Levels 0.1, 0.2, 0.3 at 0.28, 0.41, 0.57 M; 33 % at 0.3, and 13 of the last 100 episodes ran the tank dry. |
| **I** | 5–15 m, 1–3 m/s | same factor | 0.5 | 0.1 | 0.999 | 1.33 M | Levels 0.1, 0.2 and 0.3 at 0.27, 0.35 and 0.53 M; 56 % success at 0.3 and rising, 17 s episodes, 66 of the last 100 episodes landed. Full-range validation at 1 M steps: 0 % (it had only trained on starts up to 43 m). **Now the defaults.** | |

The trap is hovering: an agent that holds its attitude but can't land yet stays up until the tank runs dry, and with
the discount that looks cheaper than a failed touchdown. A low start, small noise, levels that grow by a constant
factor and half the episodes at easier levels kept pilot I landing. If episodes grow long overnight, that is the
sign; `training_report` prints how the last 100 episodes ended.

**The first night** (2026-10-11, `runs/train/20261011-1617`, five seeds, 18 M steps each) did not converge.
- None of the 8,800 validation flights landed.
- The curriculum stalled after about 3 M steps, at levels 0 to 0.5. Level 0.5 starts 32–87 m up, falling 3–9.5 m/s.
- The main measured cause: the updates grew too big as the exploration noise shrank. Approx KL rose to 0.05–0.4,
  against 0.015 in the first hour, so each seed's skill swung instead of growing. Seed 4 landed 80 % of the easiest
  starts at 5 M steps and 0 % at 18 M.
- Since 2026-10-12 each update is capped (`PPO_TARGET_KL`, D32).

The two other causes, hovering and an engine cut left to noise, are in [docs/env_variables.md](docs/env_variables.md),
with the details.

**The speed-up** (your choice, D31). The physics step now works in plain floats instead of small numpy arrays
(`rotation_rows`, `mat_vec`, `Vehicle.step`):
- The environment steps 2.7× faster: 1.03 against 2.80 ms per step, best of three runs taken in turn on the same
  busy machine. The M3 evaluation took 171 s on 4 processes instead of 362 s on 6.
- Every result is unchanged to the last bit. Checked four ways:
  - 51 reference cases (23,289 steps; every observation, reward, event and telemetry field) are identical;
  - all 21 saved episode logs replay bit for bit;
  - the M3 evaluation re-run wrote a byte-identical `episodes.csv` (1,000 episodes);
  - all tests pass.
- The one non-obvious step: numpy hands 3×3 matrix-vector products to OpenBLAS. On this machine that rounds each
  row as fma(a₂, v₂, fma(a₀, v₀, a₁·v₁)), found by testing every evaluation order on 120,000 random rows.
  `mat_vec` reproduces it with `math.fma`, which rounds the same on every machine.

## 8. Validation

| Check | Simulator | Reference |
|---|---|---|
| CG height / footpad radius | 0.5658 | 0.566 (S4 p5) |
| Engine shutdown: thrust down to 10 % | 54 ms after the command | t90 53 ms (S4 Fig 6b) |
| Level hover, no misalignment, 10 s open loop | tilt 0.000°, drift 0.000 m, height ±2 cm | stays level (fixed CG) |
| Nominal 2.44 m/s touchdown, engines cut on contact | lands, settles after 0.72 s; rebounds at 0.29 m/s (12 %) | — |
| Tip-over on a 45° slope (24 drops) | 1.00 (rigid legs: feet slide, μ 0.6 < tan 45°) | 0.85 with Viking's crushable legs (S4 p15) |
| Tip-over on 10°, 20°, 30° slopes | 0.00 | — |
| Free fall vs closed form | error ≤ g·t·dt/2 (integrator order) | — |
| Propellant used vs ∫T/(Isp g₀) dt | equal to 1e-12 | brief D6 |
| Storm push, 30 m/s | 14.5 N | 57.6 N at 51 m/s in dense air calibrates C_d·A (S4 Table II) |
| Descent drag, 30 m/s | 14.5 N (one C_d·A since 2026-10-10) | 13 N with the old axial C_d·A |
| Storm drift after 30 s, uncorrected | 0.22 / 0.48 / 0.69 m/s at 17 / 25 / 30 m/s | — |
| Radar noise | matches the published 1σ | S3 p109, p112 |
| Navigation errors at 300 m (40 resets, 1σ) | height 4.4 m, vertical speed 0.049 m/s, horizontal 0.21 m/s | 5.0 m, 0.054, 0.19 (S3 accuracies through the beam geometry) |
| Stand-in policy, gentle low starts | lands in the green zone, reward ~99 | — |
| Replay of logged episodes | bit-identical: 6 of 6 at M2 (landings and failures); 5 of 5 at M3, two of them logged before a fact was added | D10 |
| Settling on the training terrain (1,000 Viking-like touchdowns, engines cut on contact) | 0 not settled within 5 s, 0 tipped (before D26 and D27: 16 not settled) | — |
| Classical controller, 500 calm + 500 storm test episodes | 1,000 landed, all within 2.3 m of the target (section 6) | no fixed target (D25) |
| Classical controller, hardest start | lands 0.5 m off with 6.1 of 60 kg left | the 60 kg tank (D16) |
| Speed-up of the physics step (D31) | 51 reference cases bit-identical; 21 of 21 logs replay; M3 evaluation byte-identical | the code before 2026-10-11 |
| Curriculum level 1 | exactly the episodes before the curriculum (the 51 cases again) | D29 |

**Speed on this laptop** (`python -m scripts.benchmark`, 2026-10-11 after the speed-up; the machine was 17 % busy at
the start):

| Processes | Simulator control steps/s | Real time (all processes) | Speed-up |
|---|---|---|---|
| 1 | 2,482 | 99× | 1× |
| 2 | 4,599 | 184× | 1.9× |
| 4 | 6,814 | 272× | 2.7× |
| 6 | 8,946 | 357× | 3.6× |
| 8 | 9,139 | 365× | 3.7× |

- The benchmark flies the bare simulator, sensors and navigation included.
- The training environment adds the observation and the judging. One process stepped it 2.7× faster than before the
  speed-up: 1.03 against 2.80 ms per step, in runs taken in turn on a busy machine.
- PPO training, updates included, with five seeds of two environment workers each: 10.0 M steps an hour in total,
  2.0 M per seed. Before the speed-up it was 7.0 M and 1.4 M. Three workers per seed gave 10.3 M, within noise.
- Re-run the benchmark on a new machine.

## 9. Where to run: locally

- **Local (recommended):**
  - The live viewer needs a local window.
  - PPO with small networks is limited by CPU-side simulation (D9). This laptop's 8 cores beat free Colab's 2 vCPUs,
    and the GPU barely helps there. The GPU is still used for rendering.
  - Keep the laptop on mains power in "performance" mode, with games and other heavy programs closed, during
    training and benchmarks.
- **Colab:** optional extra capacity for headless seeds or fuzzing batches (`MUJOCO_GL=egl`).
  - Pins may need adjusting for Colab's Python version.
  - A replay is bit-identical only on the machine and MuJoCo version that recorded it.

## 10. Decisions and simplifications

**Your decisions:**
- No human or manual flight. The brief is updated: §8, the M1 row and §14.
- Fixed thrust envelope; blowdown not modelled.
- Notebooks used alongside the package, not instead of it.
- Run locally. The project has been a git repo since 2026-10-06.
- **The 2026-10-05 simplification** (details and reasons in [docs/env_variables.md](docs/env_variables.md)):
  - rigid legs that break above 3.34 m/s down or 1.22 m/s sideways;
  - the agent cuts the engines (permanent, must be off by the end);
  - episodes start at 200–500 m;
  - one agent learns calm and storm landings;
  - CG fixed on the centre line, mass from fuel burned, constant inertia;
  - misalignment kept, roll jets removed;
  - Viking's radar errors on;
  - one engine lag.
- **2026-10-10:**
  - The three closest-call variables are cut.
  - The agent's actions are the three valve settings directly, plus engines-off (O2).
  - You asked me to set the rest of M2 (brief D19–D22) and gave the reward ideas: green and yellow zones, a
    time cost, a tilt cost.
  - Engines-off in training (D23): the agent keeps the right to cut at any time. Early PPO training starts low
    (curriculum), with a policy biased toward keeping the engines on.
  - M3 (D24, D25): the classical controller flies the agent's observations; results are reported with confidence
    intervals and no fixed pass mark.
  - Found in M3 (D26, D27): friction is drawn from at least 1.25 × tan(slope), so a lander at rest holds; the contact
    time constant is 10 ms instead of 5 ms.
- **2026-10-11 (M4):**
  - The curriculum scales the whole start together (D29).
  - Training runs at night (D30).
  - The simulator was sped up first, with every result unchanged (D31).

**Choices made while building:**
- MJCF is generated from `.env` at run time (`lander/mjcf.py`); there is no hand-written XML.
- Control rate is 25 Hz.
- The ground is a mocap body, so slope and direction can change per episode without recompiling.
- The scripted scenarios cut the engines on the latched first contact, because the rigid legs rebound slightly.
- **M2:**
  - The agent sees the onboard navigation estimate, never the truth: the radars set it and the ideal IMU bridges
    the gaps.
  - Horizontal observations are in the heading frame.
  - Leg switches are latched.
  - The reward is paid only at the end, so no per-step cost tempts an agent to crash early.
  - A reset can carry per-episode overrides; the simulator is rebuilt when the config changes.
  - Logs are JSON + NPZ, and the replay checks bit-identity.
- **M3:**
  - The controller's design (D28): Viking's contour, gate and leg-switch cut, plus target steering, a descent that
    waits for the sideways move, and a fuel reserve. Its gains are `.env` facts, so episode logs record them.
  - The controller is built from the env's nominal config: per-episode overrides (faults, fuzzing) change only the
    simulated world.
  - Test sets start at seed 1,000,000; training stays below.
  - A log written before a fact was added to `.env` still replays: the new facts come from today's `.env`, with a
    warning.
- **M4** (D32):
  - The PPO settings, the starting policy (hover trim, engines on) and the curriculum's easiest start are `.env`
    facts.
  - Observation scaling is calibrated once per run and then fixed; reward scaling adapts.
  - Five seeds in parallel with two environment workers each; below-normal priority; the laptop is kept awake.
  - Validation uses its own seeds (900,000+) at level 1, with the policy's mean action.

**Simplifications** (also in `.env` notes and `docs/env_variables.md`):
- Footpad contact is a 3.2 cm sphere at the junction (the real pad is a 45 cm cone). The disc is visual only.
- No aero torque; no gusts; constant air density.
- One radar altimeter (Viking had two).
- Navigation trusts each new radar reading instead of filtering; carrying the height forward assumes flat ground.
- The hydrazine plume is a visual aid.

## 11. Code map: every file and function

### `lander/` — the simulator package

**`lander/config.py`** — loads `.env` into typed, read-only dataclasses and enforces the metadata.
- `ConfigError`: raised for any malformed, missing, unknown or inconsistent value.
- `Param`: one fact (`key`, `value`, `unit`, `status`, `source`, `note`, `origin`).
- `G0_MPS2`: standard gravity (a definition, so a constant rather than a fact).
- `EngineCfg`, `MassCfg`, `LegCfg`, `AeroCfg`, `RadarCfg`, `RequirementsCfg`, `SequenceCfg`, `MarsCfg`, `WindCfg`,
  `TerrainCfg`, `SimCfg`, `EpisodeCfg`, `RewardCfg`, `ControllerCfg`: one dataclass per key prefix. Field name = key
  minus prefix, lower case.
- `SimCfg.control_dt_s`: control period (1 / `SIM_CONTROL_HZ`).
- `SimCfg.substeps`: physics steps per control step.
- `EpisodeCfg.pad_radius_m`: the middle zone radius (a success must settle inside it).
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
- `validate`: cross-field checks (3 engines, 3 legs, thrust order, inertia triangle inequality, whole substeps,
  ranges low ≤ high, growing zones, start below the ceiling, controller braking within the axial limit, …).
- `derived`: computed sanity numbers `{name: (value, unit, formula)}`, shown by `check_config`.
- `CurriculumCfg`, `PpoCfg`: the `CURRICULUM_*` and `PPO_*` groups (M4).

**`lander/geometry.py`** — fixed positions from the config (numpy only).
- `unit_azimuth`: horizontal unit vector at an azimuth.
- `cross3`: fast 3-vector cross product (np.cross was a bottleneck).
- `mat_vec`: a 3×3 matrix times a 3-vector in plain floats, for the physics step. Rounded exactly like numpy's
  BLAS call on the machine M1–M3 ran on, so the 2026-10-11 speed-up changed no result.
- `LanderGeometry`: footpads, CG, engine positions, radar antenna position, Doppler beam directions.
- `hexagon_vertices`: body outline from the S2 side lengths (an equilateral triangle with the corners cut).
- `build_geometry`: builds `LanderGeometry` from a `Config`.

**`lander/vehicle.py`** — engines, propellant and mass (numpy, plain floats in the per-step code).
- `Command`: actuator command held for one control period. `throttle` = 3 valve settings 0..1; `engines_off` = shut
  the engines down for good.
- `MassModel`: dry mass + propellant, fixed CG, constant inertia.
  - `MassModel.total`: whole-vehicle mass for the propellant left.
- `EngineBank`: the three engines; per-engine state as lists of floats.
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
- `drag_force`: quadratic drag, ½ ρ C_d·A |v| v, the same in every direction.
- `wind_vector`: the episode's steady wind from a speed and the direction it blows toward.

**`lander/sensors.py`** — IMU, radar altimeter, Doppler landing radar (numpy only; terrain queries injected).
- `SensorReadings`: one sample (`accel`, `gyro`, `altitude`, `doppler_beam`, `doppler_vel`, `radar_t` = when the
  radars last sampled; NaN = invalid).
  - `SensorReadings.altitude_valid` / `SensorReadings.doppler_valid`: validity.
- `SensorSuite`: all sensors.
  - `SensorSuite.reset`: new noise stream, clears the radar holds.
  - `SensorSuite.sample`: reads every sensor from the truth state and a terrain object.

**`lander/navigation.py`** — the onboard estimate the agent sees (numpy only).
- `NavEstimate`: position relative to the target, velocity, height of the feet, and whether the latest radar
  readings were valid.
- `Navigator`: radars set the estimate, the ideal IMU carries it between readings and in the blind zones.
  - `Navigator.reset`: starts from the handed-over state, then applies the readings taken at the start.
  - `Navigator.update`: carries the estimate forward one control step and applies any new readings.

**`lander/touchdown.py`** — the brief §7 outcome for the simplified lander (numpy only).
- `stability_distance`: CG distance to the nearest vertical plane through two footpads (negative = tipping).
- `TouchdownReport`: everything measured (first-contact speeds, tilt and pitch/yaw rate, settling, …) plus `criteria`.
  - `TouchdownReport.done`: settled, tipped over, broken leg, or body strike after touchdown.
  - `TouchdownReport.success`: done and every decided criterion passed (engines off, attitude limits, pad radius, …).
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
  - A mocap ground with the target marker and dashed rings at the green-zone and pad radii.
- `build_model`: compiles the XML and collects a `ModelIndex`.
- `set_terrain`: tilts the ground (mocap body) and sets footpad friction for an episode.

**`lander/sim.py`** — `LanderSim`, the MuJoCo glue.
- `rotation_rows`: rows of the rotation matrix of a quaternion, as plain numbers (the physics step's form).
- `quat_to_mat`, `quat_mul`, `quat_from_axis_angle`: quaternion helpers.
- `quat_from_tilt`: attitude from heading + tilt + tilt direction.
- `tilt_of`: angle of the thrust axis from vertical.
- `plane_height`: ground height of a plane at (x, y).
- `InitialState`: full initial state (CG position and velocity, attitude, body rates, propellant, starting throttles).
- `ResetOptions`: per-episode conditions and faults: wind speed and direction, slope, friction, disturbance
  force/torque, engine fault scale, engines off at the start, sensor noise.
  - `ResetOptions.from_any`: builds it from a dict (rejects unknown keys).
- `Telemetry`: ground-truth snapshot each control step, including the latched leg switches (`pads_touched`).
  - `Telemetry.vz`: vertical speed.
  - `Telemetry.vh`: horizontal speed.
  - `Telemetry.spin_dps`: spin about the lander's own vertical axis.
- `StepResult`: telemetry + sensor readings + touchdown report + events + navigation estimate (`nav`).
- `PlaneTerrain`: ground queries for the sensors.
  - `PlaneTerrain.height`: ground height at (x, y).
  - `PlaneTerrain.height_above`: vertical height of a point.
  - `PlaneTerrain.nearest`: closest ground point.
  - `PlaneTerrain.raycast`: MuJoCo ray against the ground.
- `LanderSim`: the simulator.
  - `LanderSim.t`: simulation time.
  - `LanderSim.engines_off`: true once the engines are shut down.
  - `LanderSim.last_thrust`: actual thrust of each engine in the last physics step.
  - `LanderSim.hover_trim`: torque-free valve settings for a total thrust (default: the current weight).
  - `LanderSim.state_above_ground`: `InitialState` with the lowest footpad a given height above the ground.
  - `LanderSim.reset`: new episode from a seed, initial state and options.
  - `LanderSim.step`: one control period.
  - `LanderSim.telemetry`: current truth snapshot.
  - `LanderSim.read_sensors`: current sensor readings.

**`lander/episode_log.py`** — episode log files (D10, D22).
- `LOG_FORMAT`, `TRAJECTORY_COLUMNS`: the file format version and the trajectory columns.
- `code_version`: git commit and an uncommitted-changes flag ("unversioned" without git); read once per process.
- `versions`: Python, platform, MuJoCo, numpy and Gymnasium versions.
- `trajectory_row`: one trajectory row from a `Telemetry`.
- `to_jsonable`: numpy and tuples → plain JSON types.
- `episode_record`: everything the JSON file holds about one episode.
- `write_episode`: writes `<stem>.json` and `<stem>.npz`.
- `read_episode`: reads both back.
- `keys_missing_from_log`: facts added to `.env` after a log was written.
- `config_from_log`: rebuilds the exact config an episode ran with; facts newer than the log come from today's `.env`.

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
- `status_text`: one-line flight / landing status (including "leg broken", "engines still firing" and "off the pad").
- `hud_lines`: (label, value) rows: altitude radar / onboard / true, speeds, target distance, tilt, spin, propellant,
  thrusts, engines, wind, status.
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
- `watch`: runs a policy in real time (or slow motion) with plumes and HUD; Space pauses, Enter restarts;
  `done_fn` decides when an episode is over (default: the landing is decided).

### `envs/` — the Gymnasium layer (M2)

Importing `envs` registers `MarsLander-v0`.

**`envs/mars_lander_env.py`** — the training environment (section 5).
- `OBS_NAMES`: the 22 observation names, in order.
- `FAILURES`, `IN_FLIGHT`: every failure reason with its meaning; the ones that happen in flight.
- `EPISODE_KEYS`: the reset options that set the start.
- `action_to_command`: agent action → `Command` (valves (a + 1) / 2, engines off if a[3] > 0).
- `heading_axes`: cos and sin of the lander's heading.
- `to_heading`: a world vector's horizontal part in the heading frame.
- `observe`: the observation from a `StepResult` (navigation, IMU, own state).
- `truth`: ground truth for `info`.
- `start_ranges`: the start ranges at a curriculum level (0 = `CURRICULUM_START_*`, 1 = `EPISODE_START_*` exactly).
- `sample_episode`: draws a start in a fixed order, from the ranges of a curriculum level, and resolves given options.
- `Outcome`: how an episode ended (result, reason, reward and its parts, distance, time).
  - `Outcome.success`: a safe landing inside the pad radius.
  - `Outcome.safe`: landed safely, on or off the pad.
  - `Outcome.as_dict`: for `info` and logs.
- `landing_reward`: zone score minus the time and peak-tilt costs.
- `impact_ratio`: first-contact speed as a multiple of the leg limits.
- `failure_reward`: −100 in flight; −50 to −100 at touchdown, by impact ratio.
- `judge`: the outcome after a step, or None while the episode goes on.
- `MarsLanderEnv`: the Gymnasium env.
  - `MarsLanderEnv.last_result`: the latest `StepResult` (truth, sensors, navigation, report).
  - `MarsLanderEnv.set_curriculum_level`: the level later episodes start at (the reset option "level" wins).
  - `MarsLanderEnv.reset`: samples or takes the start (options, per-episode overrides).
  - `MarsLanderEnv.step`: one control step; the reward comes at the end.
  - `MarsLanderEnv.render`: an RGB frame with plumes and HUD (`render_mode="rgb_array"`).
  - `MarsLanderEnv.close`: releases the renderer.

**`envs/episode_logger.py`** — writing episodes to disk.
- `EpisodeLogger`: Gymnasium wrapper; every `every`-th episode → JSON + NPZ (actions, rewards, trajectory if asked).
  - `EpisodeLogger.reset` / `EpisodeLogger.step`: record as the episode runs; the file is written when it ends.

### `controllers/` — policies that fly the environment (M3, M4)

**`controllers/classical.py`** — the classical controller (section 6).
- `GuidanceState`: what it decided at its last step (phase, descent speed it steered to, waiting for the sideways
  move, fuel reserve, tilt command, engine thrusts).
- `ClassicalController`: callable policy over the 22 observations; knows only fixed vehicle facts.
  - `ClassicalController.reset`: forgets the last episode.
  - `ClassicalController.contour`: the descent speed allowed at a height (brake to the gate, then 2.44 m/s).
  - `ClassicalController.side_time`: seconds the sideways move to the target needs.
  - `ClassicalController.fuel_to_land`: propellant a straight-down landing from here needs.
  - `ClassicalController.mix`: total thrust and pitch/yaw torques → three engine thrusts (attitude first).

**`controllers/ppo.py`** — a trained PPO agent as a controller (section 7).
- `FixedMeanStd`: VecNormalize's observation statistics, fixed once calibrated.
  - `FixedMeanStd.update`: ignores new samples.
- `calibrate_obs_scaling`: mean and variance of every observation over classical landings from the full start range.
- `PpoPolicy`: a checkpoint folder as a controller: raw observation → the policy's mean action.
  - `PpoPolicy.reset`: start of an episode (no state).
- `load_ppo_policy`: `PpoPolicy` for a folder, loaded once per process.

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
- **`scripts/run_episodes.py`**
  - `POLICIES`: `classical`, `random`, `descend`.
  - `GENTLE`: the overrides for gentle low starts.
  - `descend`: stand-in policy: valves for 95 % of the weight, engines off once a leg touches.
  - `make_policy`: a fresh policy for one episode (the controller gets the env's nominal config; `ppo:FOLDER` a
    trained agent).
  - `policy_arg`: checks `--policy` (a name from `POLICIES`, or `ppo:FOLDER` with a `model.zip`).
  - `main`: runs episodes, prints outcomes, logs them with `--log`.
- **`scripts/evaluate.py`**
  - `TEST_SETS`: calm and storm, with their first seeds (1,000,000 and 2,000,000).
  - `wilson`: 95 % Wilson interval of a success rate.
  - `run_episode`: flies one test episode; returns one row of results.
  - `summarize`: per test set: rates with intervals, failure causes, distributions.
  - `evaluate`: runs the test sets in parallel; returns the rows and the summary.
  - `print_report`: the summary as text.
  - `main`: CLI (`--episodes`, `--procs`, `--out`, `--log-failures`).
- **`scripts/replay.py`**
  - `ReplayCheck`: how a replay compares with its log.
    - `ReplayCheck.identical`: same steps, outcome, return and trajectory, to the last bit.
  - `environment_warnings`: version or code differences that can break bit-identical replay, and facts newer than
    the log.
  - `replay`: re-runs a logged episode and compares.
  - `watch_replay`: shows a logged episode in the viewer.
  - `main`: CLI (`--video`, `--watch`).
- **`scripts/benchmark.py`**
  - `run`: aggregate control steps/s with N worker processes.
  - `main`: CLI.
- **`scripts/train.py`** (section 7)
  - `RunSettings`: how a run trains (seeds, envs per seed, checkpoint and validation intervals, overrides);
    in `run.json`.
  - `SeedState`: where one seed stands (steps, hours, sessions, curriculum level and window, best validation);
    in each checkpoint's `state.json`.
  - `parse_seeds`: `"0-4"` → `[0, 1, 2, 3, 4]`.
  - `run_config`: `.env` plus the run's `--set` overrides.
  - `keep_awake`: asks Windows not to sleep while training runs.
  - `lower_priority`: below-normal priority for training and the processes it starts.
  - `append_csv`: appends a row to a CSV.
  - `config_changes`: facts that changed since the run started (printed on `--resume`).
  - `make_model`: a fresh PPO agent with the `PPO_*` settings.
  - `start_engines_on`: D23's untrained policy: valves at hover trim, engines lit.
  - `save_checkpoint`: model + scaling + state, written aside and then swapped in.
  - `validate`: the policy's mean action on the validation episodes.
  - `train_seed`: trains or resumes one seed for a time or step budget (curriculum, validation, checkpoints).
  - `start_run`: a new run folder: `run.json` (settings, config, code version) and `obs_scaling.json`.
  - `main`: CLI; one process per seed.
- **`scripts/training_report.py`**
  - `read_rows`: a training CSV as rows of numbers.
  - `seed_dirs`: a run's seed folders.
  - `seed_status`: a seed's latest training numbers and validation.
  - `print_report`: the table, then `learning_curves.png`.
  - `plot_curves`: level, success at the level and validation success per seed.
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
| `docs/Mars_Lander_RL_code_guide_v2.pdf` | Code guide v2 (2026-10-06): every `.env` variable of that day with why it was kept and the functions that read it (Section 2, including where each key of the old 155-key file went), then the call tree from `main` with every node explained in in-order (LVR). |
| `docs/Mars_Lander_RL_code_guide.pdf` | Code guide v1: the same call tree without the variables section. |
| `tools/code_guide/` | The scripts that build both guides and the old 155-key `.env` (its README says how to run them). |
| `runs/` | Outputs: videos, episode logs, evaluations (`runs/eval/`), training runs (`runs/train/`). |

## 12. Tests (`python -m pytest`)

- **`tests/conftest.py`**: fixtures. `cfg` ignores `LANDER_*` overrides; `make_sim(overrides, seed)`; `CLEAN` = no air,
  no misalignment, perfect radar readings.
- **`tests/test_config.py`**:
  - Every fact has a status and source; parked facts are comments only.
  - Missing metadata, a bad status or a sourceless "published" value are rejected.
  - Unknown and missing keys are rejected; contradicting episode and reward settings are rejected.
  - Env-var and CLI overrides are recorded; contradicting controller settings are rejected.
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
  - Drag: a tenth of a newton in calm descent, ~14.5 N at 30 m/s, 5–15 N in a dust storm.
  - Storm drift after 30 s matches the review.
  - Ideal radar altimeter, including the 41 m cut-off.
  - Ideal Doppler velocity, and no lock in hover.
  - Ideal IMU.
  - Radar noise matches Viking's published levels.
- **`tests/test_navigation.py`**:
  - Perfect sensors give the true position, velocity and height.
  - Below 41 m the IMU carries the height (exact on flat ground).
  - On a slope the carried height drifts by the ground's fall since the last reading.
  - Radar errors reach the estimate at the expected 1σ (height, vertical and horizontal speed).
- **`tests/test_touchdown.py`**:
  - Stability-distance geometry.
  - Nominal drop succeeds and events come in order.
  - The hard drop breaks a leg.
  - Free fall crashes.
  - A landing only counts once the engines are cut.
  - Tilt above 5° or pitch/yaw rate above 7°/s at first contact fails; spin does not; off the pad fails only the pad check.
  - Running out of propellant airborne is flagged.
- **`tests/test_env.py`**:
  - Passes Gymnasium's environment checker; `MarsLander-v0` is registered.
  - Same seed → same episode, bit for bit; `info["episode"]` replays it.
  - The start distribution matches the `EPISODE_*` ranges (storm share, slope distribution, …).
  - Options set any part of the start without moving the rest; per-episode overrides rebuild and revert.
  - Action mapping; the engines-off command is permanent.
  - Gentle landings: in the green zone (reward 100 − time cost) and off the pad.
  - Every in-flight failure ends the episode with −100.
  - Touchdown failures are graded by impact speed (engines left on: −50; a free fall: between −50 and −100).
  - The reward functions; a safe landing always beats a failure.
  - The observation is the onboard view (and holds no wind).
  - `render()` gives a frame.
- **`tests/test_episode_log.py`**:
  - A logged landing holds the config, versions, code version and resolved start, and replays bit-identically.
  - A failed episode replays too; a one-bit change to one action is caught.
  - A log older than a new fact (no `CTRL_*` keys) still replays, with a warning.
  - The logger can write every n-th episode.
  - The code version outside a git repo is "unversioned".
- **`tests/test_controller.py`**:
  - The mixer gives the total thrust and torques asked for (the torque-free split equals the simulator's trim);
    when an engine saturates, attitude comes first.
  - The descent plan: the contour, the sideways time budget, the fuel estimate (the hardest start fits the tank).
  - Engines off once a leg switch trips.
  - Lands calm and in a storm (the first episode of each test set), within the touchdown limits.
  - The hardest start (500 m, 30 m/s down, 50 m off moving 10 m/s away, 30 m/s storm) lands with propellant left.
- **`tests/test_evaluate.py`**:
  - Wilson intervals match known values.
  - The test sets use their fixed seeds and weather; the summary counts; a logged failure replays bit-identically.
- **`tests/test_training.py`**:
  - Curriculum ranges: level 1 is exactly `EPISODE_START_*`, level 0 the `CURRICULUM_START_*` ranges.
  - Geometric levels end exactly at D19 and grow by one factor per level; mixing happens only in training, and an
    explicit level in the reset options wins.
  - Level 1 gives the episode without a curriculum; level-0 starts fall inside the easy ranges; the level and the
    weather come back in the last step's `info`.
  - Contradicting `CURRICULUM_*` and `PPO_*` settings are rejected; `--seeds` parsing.
  - The curriculum rises only after a full window at the current level reaches the success rate.
  - The untrained agent holds hover trim with the engines on: its cut chance per step is `PPO_INIT_CUT_PROB`.
  - A tiny run trains, validates, checkpoints, resumes (keeping the update cap `PPO_TARGET_KL`), and its checkpoint
    flies a test episode; the observation scaling never moves.
- **`tests/test_video_docs.py`**:
  - Offscreen frame and MP4 work.
  - This README names every module, class and function (`lander/`, `envs/`, `controllers/`, `scripts/`).
  - The vehicle layer never imports MuJoCo.

## 13. Milestones

| M | Scope | State |
|---|---|---|
| M1 | Physics, Viking model, Mars air + wind, legs, viewer (plumes + HUD), video, scripted scenarios; simplified 2026-10-05 | **done** |
| M2 | Gymnasium env, `reset(seed, options)`, logging + replay; O2–O5 decided | **done** 2026-10-10 |
| M3 | Classical controller (Viking's contour → 2.44 m/s from 16.8 m → cut engines on contact; attitude by differential throttle; target steering) and the evaluation script; D24–D28 | **done** 2026-10-10 |
| M4 | PPO (SAC if needed), `VecNormalize`, checkpoints, TensorBoard, CPU, ≥ 5 seeds; curriculum; simulator speed-up; D29–D32 | **built** 2026-10-11. The first night did not converge; each update is capped since 2026-10-12 |
| M5 | Evaluation with 95 % Wilson CIs on calm and storm test sets, comparison with M3, videos | |
