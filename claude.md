# Mars Lander RL — project brief

Viking-class lander · Mars terminal descent · MuJoCo · Stable-Baselines3. Decisions were made with the user in a design discussion (Oct 2026).
**Locked** = fixed. **Open** = ask the user before the milestone that needs it. Sources: §15 (S1–S11).

## 0. How to work here
- Build the milestones (§13) in order, **one at a time**. Before each: state assumptions in a few lines, ask ≤3 questions only if essential, and resolve the Open decisions it depends on.
- After each: summarise what was built, what was run/tested and what could **not** be verified. Then stop and wait for the user's go-ahead.
- The user wants the *why*: for any design choice not covered here, give the main alternatives in a line or two and ask.
- Run and test what you can. If a window can't open, test headless and give exact Windows commands to run the viewer locally.
- Every numeric parameter in config carries `source` and `status` (`published` / `derived` / `assumed`). Never present an assumed number as published.
- Never silently change the task. If RL doesn't converge, try in this order and record each change: curriculum (lower start altitude) → potential-based reward shaping → action smoothing.

## 1. Goal
1. Realistic 3D Mars terminal-landing simulator with continuous controls and a live viewer.
2. RL agent that learns to land it.
3. Later: fuzz-testing study that searches for initial conditions/environments where the agent fails; the classical controller is the validity check.

## 2. Setup
- Windows 11 · Python **3.13.5** (project `.venv`, pinned in README) · GPU RTX 3050 6 GB or Google Colab. **Training runs on CPU.**
- MuJoCo from pip; **pin the exact version** in `requirements.txt` and write it into every episode log. Headless/Colab: `MUJOCO_GL=egl`.

## 3. Decisions
### Locked
| # | Topic | Decision |
|---|---|---|
| D1 | Physics | MuJoCo, hybrid: a backend-agnostic Python vehicle layer computes engine and aero forces and sensor readings; MuJoCo integrates rigid-body motion, resolves contacts and renders. |
| D2 | World | **Mars**, chosen as the least favourable landing environment. |
| D3 | Vehicle | **NASA Viking Lander (1976)** with exact published numbers (§4); estimates labelled; simplified per D15. |
| D4 | Actuators | **n = 3 fixed throttleable engines**, all lit in powered descent, individually throttled for pitch/yaw. No gimbal. **No roll thrusters** (2026-10-05): spin is not controlled; thrust misalignment spins the lander 2–9°/s over a 30 s descent, harmless with rigid legs. |
| D5 | Lag | One engine lag: first order, t90 = 53 ms up and down (`VIKING_ENGINE_T90_S`, Viking's shutdown trace). No rate limit and no `lag_scale` (2026-10-05): degrade by overriding the lag (env var or CLI); the episode log must record it. |
| D6 | Fuel | Per engine ṁ = T_actual / (Isp·g₀), using **actual** post-lag thrust; mass = dry + propellant left (CG and inertia fixed, D15); empty tank → thrust 0 → failure. |
| D7 | Engine cutoff | **The agent commands engines off** (2026-10-05, replaces Viking's TESS leg switches): permanent, no relight; thrust decays with the D5 lag. A landing counts only if the engines are off by the end of the episode. |
| D8 | Realism priority | Sensors, touchdown dynamics and discrete logic (where real landers fail) before integrator fidelity. |
| D9 | Training | SB3 PPO (SAC alternative), `VecNormalize`, checkpoints, TensorBoard, CPU, ≥5 seeds. |
| D10 | Determinism | Same seed + config (+ same machine and MuJoCo version) → identical episode. Log the full config per episode; a `replay` script reproduces it. |
| D11 | Control input | **No human/manual flight** (2026-10-05): only the RL agent's commands (scripted scenarios in M1) fly the lander, against the environment conditions. |
| D12 | Legs | **Rigid legs** ending in three footpad contact spheres. A leg breaks if first contact is faster than 3.34 m/s down or 1.22 m/s sideways (Viking's leg design limits, S5). (2026-10-05; replaces the hinge + crush-strut model; no TESS, strut or load limiters.) |
| D13 | Thrust envelope | Fixed 276–2,667 N per engine; S3's blowdown thrust decay **not modelled** (2026-10-05). |
| D14 | Tooling | Facts live in `.env` (value · unit · status · source); only what the code reads, each explained in `docs/env_variables.md`, later facts parked as comments. Run locally (i5-12450H, CPU training); notebooks only for exploration/analysis next to the package. No git repo: logs record `"unversioned"`. |
| D15 | Simplified vehicle | (2026-10-05) CG on the centre line at 0.79 m, never moves; mass = 573.1 kg dry + propellant; constant inertia (S4 touchdown values). Constant air density; one steady wind per episode (calm = none; storms 17–30 m/s), no gusts. Ideal IMU at the CG; one radar altimeter; Viking's radar errors on by default. Thrust misalignment (0.044°) is kept as the only steady disturbance. |
| D16 | Episode start | (2026-10-05, resolves O1) Random height 200–500 m; Viking's earlier phases (parachute, 1.4 km ignition, handover) dropped. Speed, tilt and rate ranges proposed at M2; 60 kg propellant at the start (assumed, check at M3). |
| D17 | Weather in training | (2026-10-05) One agent learns calm and dust-storm landings from mixed episodes (~30 % storms, assumed) and is not told which; scored separately on calm and storm test sets. |

Rejected (context only): Moon/Vikram (possible future preset), single gimballed engine, PyBullet, GPU-parallel sims (they break SB3-on-CPU), game engines.
### Open
| # | Topic | Options | Recommendation | By |
|---|---|---|---|---|
| O2 | Agent actions | Always + engines-off (D7). (a) Raw: 3 throttles. (b) Allocator: total thrust + 2 torques → mixer → engines. | Undecided | M2 |
| O3 | Pad radius, arena bounds, timeout, tilt limit | — | Propose at M2 (Viking's 3σ touchdown attitude was under ~2°, S4; parked: S3's 5° and 7°/s) | M2 |
| O4 | Observations (including how the agent senses touchdown), reward | Sensor noise decided: Viking's radar errors on (D15). | Propose at M2 | M2 |

O1 (episode start) was decided on 2026-10-05: see D16.

## 4. Vehicle: Viking Lander
*Since M1, `.env` is the authority for every number the code uses (each with status + source, checked against the NASA PDFs; `docs/env_variables.md` explains each); facts for later milestones are parked as comments at its end. The table below is Viking as published. The simulated vehicle is simplified (D4, D7, D12, D15): CG on the centre line, no roll jets, rigid legs without TESS, struts or limiters, one engine lag. Where the sources corrected this table — propellant 83.9 kg (S3), constant-velocity phase from 16.8 m (S3) — see README §3.*
| Item | Value | Status | Src |
|---|---|---|---|
| Terminal engines | 3 × hydrazine monoprop, 18 small nozzles each, one per body side, 120° apart | published | S1 S3 |
| Thrust per engine | 276–2,667 N (10:1 throttle) | published | S1 S2 S9 |
| Pitch/yaw | each engine's thrust controlled individually | published | S4 |
| Roll | 4 × MR-50F, 8.5 lbf ≈ 37.8 N each | published (maker) | S6 |
| Propellant | 85 kg purified hydrazine; 2 spherical Ti tanks on the sides between legs 1–2 and 1–3 | published | S2 S3 |
| Mass | 657 kg fuelled; 575.8 kg at touchdown | published | S2 S4 |
| Principal inertias | 162.7, 216.9, 298.3 kg·m² | published | S4 |
| CG height ÷ footpad radius | 0.566 (undeformed legs) | published | S4 |
| Engine and leg coordinates | dimensioned drawings in TN D-7959 Figs 3 and 6 | published (transcribe) | S4 |
| Isp | ~195 s (MSL descendant measured 204–223 s, minus ~9% 18-nozzle loss) | derived | S7 S8 |
| Throttle response | 50 ms to 90% (MSL engine valve) | assumed proxy | S7 |
| Shutdown delay | 0.044 ± 0.016 s (3σ) | published | S4 |
| TESS | each leg switch trips after 1.3 cm compression, before strut crush; any footpad → all engines off | published | S4 S5 |
| Main strut | crushable honeycomb: 4,448 N at start, ramp over 8.9 cm, then ~11,200 N for ≥21.6 cm; ~445 N tension | published | S5 |
| Bipod load limiters | yield ~7,875 N | published | S5 |
| Landing gear mass | 20 kg | published | S5 |
| Sensors | strapdown IMU, 2 radar altimeters, 4-beam Doppler landing radar | published | S3 |
| Sequence | chute ~6 km / ~250 m/s → ~60 m/s in 45 s; engines ignite ~1.4 km; chute released +2 s; gravity-turn guidance; constant 2.4 m/s from ~12 m; ~40 s burn | published | S2 S3 S10 |
| Thrust misalignment | MSL engine measured 0.044° | published (proxy) | S7 |

Derived (Isp 195 s, g₀ 9.80665, g 3.71): ṁ per engine 1.39 kg/s at 2,667 N and 0.14 kg/s at 276 N · weight 2,437 N at 657 kg → T/W 3.28, hover throttle ≈ 30%, hover ṁ ≈ 1.28 kg/s · NASA's touchdown thrusts sum to 2,151 N ≈ weight at 575.8 kg (2,136 N), so the 2.4 m/s phase is near-hover (use as a sim check) · ~81 of 85 kg burned → thin reserve.

## 5. Environment: Mars, near surface
- g = 3.71 m/s² (Viking design 3.7, S3; fact sheet 3.69–3.73, S11). Flat, non-rotating local frame.
- Air density ~0.016 ± 0.006 kg/m³ (S11); design surface pressure 4–7.5 mbar (S3).
- Wind: Viking sites 2–7 m/s summer, 5–10 autumn, 17–30 dust storm (S11). Since 2026-10-05 (D15, D17): calm episodes have no wind (2–10 m/s moves the lander < 0.08 m/s in 30 s), dust storms are a steady 17–30 m/s, and training mixes the two; the fuzzer pushes beyond. Gusts are left out (< 0.1 m/s of drift even in a storm). Ground-survival design 90 m/s (S3).
- Drag: our own quadratic model (MuJoCo's fluid model off), constant density. The lateral C_d·A is calibrated to NASA's touchdown aero forces (a few N mean, 3σ ≈ 15–58 N per axis, S4); a 30 m/s storm pushes ~14.5 N.
- Terrain: sloped plane; slope ~ exponential with 90th percentile 11.7°, uniform direction (S4). Heightfield and rocks (Surveyor-based distributions, S4) later.
- Footpad friction μ ∈ [0.2, 1.0] (S4, S5). Plume–surface effects only as optional near-ground sensor degradation (Viking's 18 nozzles limited ground disturbance).

## 6. Simulation design
- Frames: world ENU, z up, origin at target; body +z along the thrust axis; quaternion attitude; body-frame rates; tilt = angle(body z, world z).
- Rates: physics dt ≤ 2 ms (check contact stability with the rigid legs; reduce if needed); control 20–25 Hz, zero-order hold.
- MJCF: lander = one free body with three rigid footpad contact spheres (D12, D15), generated from `.env` at run time; mass updated as propellant burns, CG and inertia fixed. Sites for 3 engines and the IMU (at the CG). Terrain plane.
- Vehicle layer, every physics step: command → first-order lag (t90 53 ms) → clamp 276–2,667 N → misalignment → force at engine site; engines-off command (permanent); mass flow; drag in the episode's steady wind; sensors (ideal IMU, radar altimeter, Doppler radar) with Viking's radar errors.
- Logging: per-episode JSON (config, seeds, MuJoCo version, git hash) + optional trajectory `.npz`; `replay` reproduces it.

## 7. Success / failure
Success = all of: at first footpad contact, vertical speed ≤ 3.34 m/s (Viking 2.44 ± 0.9) and horizontal ≤ 1.22 m/s, otherwise a leg breaks (S5, D12); stable, i.e. the CG never crosses a vertical plane through two footpads (TN D-7959 stability distance > 0), settled on all three feet when KE < 0.5% of impact KE (S4); no body strike; engines off by the end of the episode (D7); inside the pad radius; tilt within limit (O3). (Body clearance ≥ 22 cm and peak load ≤ 30 g were dropped on 2026-10-05: with rigid legs the body cannot near the ground before a tip-over, and peak g is set by MuJoCo's contact softness.)
Failure: tip-over, broken leg, any limit exceeded, body strike, engines still firing at the end, leaving the arena, timeout, propellant exhausted before touchdown.
Benchmark: Viking required ≥99% probability per requirement in the worst-case atmosphere (S3); its gear achieved ≥99.7% stability (S5).

## 8. Viewer (M1)
- MuJoCo passive viewer (local window on Windows) for watching. Plumes: 3 cones scaled by each engine's actual thrust.
- HUD: altitude (radar and true), vertical/horizontal speed, tilt, spin, propellant kg, per-engine thrust, engines on/cut, wind, time.
- No manual flight (D11). Commands come from scripted scenarios (M1) and later the agent; viewer keys only pause and restart.
- Video: offscreen renderer → MP4; must work headless.

## 9. Gymnasium env (M2)
`reset(seed, options)` takes the full initial state (position, velocity, attitude, angular rate; start at 200–500 m, D16) and parameters: mass/propellant, thrust limits, gravity, slope + direction, friction, wind (steady; calm or storm, D17), generic disturbance force/torque, misalignment, sensor noise, engine lag, faults. Observations come from sensor models (ground truth available for debugging).

## 10. Controllers, training, evaluation
- Classical (M3): Viking's scheme — guidance down from the start point → constant 2.4 m/s from ~12–17 m → cut the engines on contact; PD attitude via differential throttle (pitch/yaw; spin is not controlled). Reused as the fuzzing validity check.
- RL (M4): PPO first, SAC if needed; `VecNormalize`, checkpoints, TensorBoard; CPU; ≥5 seeds.
- Eval (M5): success rate with 95% Wilson CIs over ≥5 seeds, separately on calm and dust-storm test sets (D17); propellant used; touchdown speed/tilt distributions; comparison with the classical controller; MP4s.

## 11. Tests (pytest, headless)
Free fall (engines off, no air) matches z₀ + v₀t − ½gt² · hover: total thrust = m·g holds altitude and stays level without misalignment · mass flow Δm = ∫T/(Isp·g₀)dt and mass = dry + propellant · determinism: same seed + config → identical trajectory · engines-off is permanent and decays with the 53 ms lag · a leg breaks above 3.34 m/s down or 1.22 m/s sideways · storm push and drift · radar noise at the published levels · steep slopes tip over (TN D-7959: 0.85 probability at 45° with crushable legs).

## 12. Fuzzing knobs (stage 3)
engine lag (override) · engine underperformance/failure · thrust misalignment · CoM offset · wind beyond dust storm, gusts, density spread · slope ≥19° · friction · rocks · sensor noise/bias/dropouts · spurious early engine cut (Mars Polar Lander-style) · initial-state envelope.

## 13. Milestones (in order)
| M | Scope | Done when |
|---|---|---|
| M1 | Physics, Viking model, Mars air + wind, legs, viewer (plumes + HUD), video export, scripted scenarios. No RL, no manual flight (D11). Simplified 2026-10-05 (D15). | Scripted physics scenarios run in the viewer and export to MP4; headless tests pass; README commands work. |
| M2 | Gymnasium env + tests (resolve O2–O4 first; O1 decided, D16). | `reset(seed, options)` per §9; logging and replay verified. |
| M3 | Classical controller. | High success on the nominal distribution, with CI. |
| M4 | RL training (PPO; SAC if needed). | Learning curves, checkpoints, CPU only. |
| M5 | Evaluation + video. | Success rate ± CI over ≥5 seeds; videos exported. |

## 14. Repo layout (proposal)
- `lander/`: `config.py` (dataclasses; every param has value, source, status), `assets/viking.xml`, `vehicle.py`, `aero.py`, `legs.py`, `sensors.py`, `sim.py`, `viewer.py`, `episode_log.py`
- `envs/mars_lander_env.py` (M2) · `controllers/classical.py` (M3) · `scripts/`: `watch`, `export_video`, `train`, `eval`, `replay` · `tests/`
- `requirements.txt` (pinned: mujoco, numpy, gymnasium, stable-baselines3, tensorboard, imageio, imageio-ffmpeg, pytest) · `README.md` with commands for train, eval, watch, export video and replay.
- As built in M1 (see README §8): `.env` holds every fact the code uses (explained in `docs/env_variables.md`); `lander/` also has `geometry.py`, `mjcf.py` (MJCF generated from `.env`, no `assets/viking.xml`), `touchdown.py`, `scenarios.py`, `visuals.py`, `hud.py`, `video.py`; `legs.py` was removed with the rigid legs (D12); `episode_log.py` and `replay` come with M2.

## 15. Sources
- S1 Viking program (Wikipedia): https://en.wikipedia.org/wiki/Viking_program
- S2 Viking 1 Lander (NASA NSSDCA): https://nssdc.gsfc.nasa.gov/nmc/spacecraft/display.action?id=1975-075C
- S3 NASA RP-1027, Viking '75 Spacecraft Design and Test Summary, Vol. I: https://ntrs.nasa.gov/api/citations/19810001592/downloads/19810001592.pdf
- S4 NASA TN D-7959, A Monte Carlo Analysis of the Viking Lander Dynamics at Touchdown: https://ntrs.nasa.gov/api/citations/19750023452/downloads/19750023452.pdf
- S5 Pohlen et al., The Evolution of the Viking Landing Gear (JPL TM 33-777): https://ntrs.nasa.gov/api/citations/19760021205/downloads/19760021205.pdf
- S6 L3Harris, The Enduring Legacy of L3Harris Propulsion on Viking Missions: https://www.l3harris.com/newsroom/editorial/2025/08/enduring-legacy-l3harris-propulsion-viking-missions
- S7 Dawson et al., Monopropellant Hydrazine 700 lbf Throttling Terminal Descent Engine for MSL, AIAA 2007-5481: http://matthewwturner.com/uah/IPT2008_summer/baselines/LOW%20Files/Payload/Downloads/AIAA-2007-5481-979.pdf
- S8 Viking lander engine cutaway, 18-nozzle losses (ResearchGate): https://www.researchgate.net/figure/king-Lander-Engine-Cutaway-View-Rocket-Research-Primex-Aerospace_fig4_232617757
- S9 History and Status of Aerojet Mars Lander Propulsion, IAC-10.C4.1.7: http://iafastro.directory/iac/paper/id/8273/abstract-pdf/IAC-10.C4.1.7.brief.pdf
- S10 Viking landing sequence of events (ResearchGate): https://www.researchgate.net/figure/king-Landing-Sequence-of-Events-NASA-Press-Handbook-Release-76-116_fig1_232617757
- S11 NASA Mars Fact Sheet: https://nssdc.gsfc.nasa.gov/planetary/factsheet/marsfact.html