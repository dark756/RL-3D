# The `.env` variables: what each one does and why it is kept

`.env` holds every fact the simulator reads: **41 variables** (26 published, 5 derived, 10 assumed), each
with its unit, status and source. This page explains each one and records why the rest of the
original 155 keys went (decided 2026-10-05). Eight facts that later milestones will need are kept
as commented-out "parked" lines at the end of `.env`. Section 2 of
`docs/Mars_Lander_RL_code_guide_v2.pdf` has the same list, plus the functions that read each variable
and a key-by-key account of the old file. Of the 155 old keys, 26 were kept unchanged, 19 were merged or
renamed into 14 new keys, 8 were parked, 23 became constants in the code, and 79 were removed.

## Decisions behind the trim

| # | Decision | Effect on the brief (`claude.md`) |
|---|---|---|
| 1 | **Rigid legs.** A leg breaks if first contact is faster than 3.34 m/s down or 1.22 m/s sideways (Viking's leg design limits, S5). | Replaces D12. |
| 2 | **The agent shuts the engines off.** The cut is permanent (no relight), and a landing only counts if the engines are off by the end of the episode. | Replaces D7. |
| 3 | **Episodes start at a random height of 200–500 m.** Viking's earlier phases (parachute, ignition at 1.4 km, handover) are dropped. | Resolves O1. |
| 4 | **One agent, mixed weather.** Some episodes have a dust storm; the agent is not told which. | M2 and M4 approach. |
| 5 | **The CG sits on the centre line at 0.79 m and never moves.** Mass = dry mass + propellant left. Inertias are constant. | New D15. |
| 6 | **Misalignment stays; the roll jets go.** Spin is not controlled. It reaches 2–9°/s over a 30 s descent, which moves the feet at most 0.22 m/s at touchdown: harmless with rigid legs. The agent's actions are 3 throttles + engines off. | Replaces D4's roll thrusters; D6's jet fuel no longer applies. |
| 7 | **Radar readings carry Viking's published errors, switched on by default.** | Settles the sensor-noise part of O4. |
| 8 | **One engine lag (53 ms).** The rate limit and `lag_scale` go; slower engines are tested by overriding the lag, which the logs record. | Changes D5. |

## Error-data review

Each item was measured in 30 s of open-loop hover from 300 m: the engines are held at a fixed thrust
balance and there is no controller. Each run is compared with the same seed with the effect switched
off. A lander with no misalignment and no wind stays exactly level.

| Data | Size | What it changes | Verdict |
|---|---|---|---|
| Engine misalignment | 0.044° | A tiny torque (~0.2 N·m) that never stops. Uncorrected for 30 s (20 seeds): tilt 4–55° (median 19°), 3–34 m/s sideways, spin 2–9°/s. | **Keep:** the only steady disturbance; the agent must learn to hold attitude against it. |
| Engine lag, up / down | 50 / 53 ms | The two differ by 3 ms. | **Merged** into one lag of 53 ms. |
| Engine rate limit | 48,000 N/s | Adds 12 ms, only on full-range throttle jumps. Same cause as the lag; its value was guessed from it. | **Removed.** |
| Thrust noise | 0, no data | Nothing. | **Removed.** |
| Roll-jet valve delay, minimum firing | 10 ms, 20 ms | Roll-jet pulse timing only. | **Removed** (roll jets removed). |
| IMU position | 0.94 m off-centre | Hard turns add up to ~10 m/s² to the accelerometer reading: noise, not information. | **Removed:** the IMU sits at the CG. |
| Wind gusts (strength, duration) | 1 m/s, 2 s | Extra drift: 0.002 m/s in calm air, 0.06 m/s on average in a 25 m/s storm. | **Removed.** |
| Calm wind (range and default) | 2–10 m/s | 0.015–0.07 m/s of drift, at most 6 % of the sideways limit. | **Removed:** calm episodes have no wind. |
| Air-density spread | ±0.006 kg/m³ | The densest air raises a storm's push by 37 %; the 17–30 m/s storm range already spans a 3× range of push. | **Removed.** |
| `lag_scale` | 1.0 | With one lag left, the same as overriding that lag directly. | **Removed.** |
| Radar update period | 0.2 s for each radar | Readings up to 0.2 s old: up to 1 m/s of stale velocity while braking hard. | **Kept**, one shared value. |
| Radar errors and noise switch | altimeter ~1.1 m at 41 m, ~2 m at 100 m; Doppler ~0.1 m/s near touchdown (1σ) | The largest error sources left; the Doppler error is 8 % of the sideways limit. | **Kept**, switched on. |
| Radar blind zones | 41 m, 1 m/s | No altitude reading below 41 m; no velocity reading near hover. | **Kept:** this is where landings go wrong. |
| Cut in the first round | — | IMU bias, scale and misalignment errors (10⁴–10⁶× below the signals), engine cant, centre-of-pressure offset, wind shear, crush-force spread, shutdown-delay spread, second altimeter. | Removed. |

`DESIGN_99_MPS` and `REF_TD_AERO_3SIGMA_N` only re-checked `CDA_LATERAL` against the numbers it was
calibrated from. Their values now sit in `CDA_LATERAL`'s note.

## The 41 variables

### Engines `VIKING_ENGINE_*` (7)
| Key | Value | Status | What it is | Why it stays |
|---|---|---|---|---|
| THRUST_MIN_N | 276 N | published | Thrust of each engine at its lowest setting. | Viking couldn't throttle lower, so lit engines always give at least 828 N (about ⅓ of the weight). Only the engines-off command removes it. |
| THRUST_MAX_N | 2,667 N | published | Thrust of each engine at full setting. | How hard the lander can brake: 3.4 × its starting weight. |
| ISP_S | 195 s | derived | Fuel efficiency. | Fuel burned = thrust ÷ (Isp × 9.81). Decides when the tank runs dry. |
| X_M | 0.835, −0.835, 0 m | published | Engine positions, sideways. | Throttling the engines unequally tilts the lander: the only way to steer. |
| Y_M | 0.482, 0.482, −0.989 m | published | Engine positions, front/back. | Same. |
| T90_S | 0.053 s | derived | Time for thrust to complete 90 % of any change, including after the cut. | The one engine lag the agent must allow for. Taken from Viking's shutdown trace (S4 Fig 6b); using it for throttle-up too is assumed (the MSL engine valve's 50 ms, S7, agrees). |
| MISALIGN_DEG | 0.044° | assumed | Thrust slightly off the engine axis, in a random direction each episode. | The only steady disturbance (see the review). |

### Mass `VIKING_MASS_*` (6)
| Key | Value | Status | What it is | Why it stays |
|---|---|---|---|---|
| DRY_KG | 573.1 kg | derived | The lander with empty tanks: 657 kg fuelled (S2) − 83.9 kg propellant (S3). | Mass at every step = this + the propellant left. |
| START_PROPELLANT_KG | 60 kg | assumed | Propellant at the start of an episode. | The fuel budget. A careful landing from 500 m at 30 m/s needs about 45–55 kg; the M3 classical controller will check it. |
| IXX_KGM2, IYY_KGM2, IZZ_KGM2 | 162.7, 216.9, 298.3 | published | Resistance to turning about each axis. | How fast a torque turns or spins the lander. Held constant: fuel is under 10 % of the mass. |
| CG_Z_M | 0.7904 m | published | CG height above the feet, on the centre line. | Decides when the lander tips over on a slope. |

### Legs `VIKING_LEG_*` (3), rigid
| Key | Value | Status | What it is | Why it stays |
|---|---|---|---|---|
| AZIMUTH_DEG | 90, −30, 210 | published | Directions of the three legs. | The footprint shape: which way the lander can tip. |
| FOOTPAD_RADIUS_M | 1.397 m | published | Distance from the centre line to each foot. | The footprint size: the tip-over margin. |
| FOOTPAD_Z_M | 0.03175 m | published | Foot height, which is also the contact-sphere radius. | Where the feet meet the ground. |

The break rule has no variables of its own: it reuses the two speed limits under Landing requirements.

### Aerodynamics `VIKING_AERO_*` (2)
Drag depends on Cd·A, the drag coefficient times the area facing the air.

| Key | Value | Status | What it is | Why it stays |
|---|---|---|---|---|
| CDA_AXIAL_M2 | 1.80 m² | assumed | Cd·A for air moving along the thrust axis. | Drag from the descent itself: Cd 1.05 (assumed) × the 1.71 m² body outline (S2). |
| CDA_LATERAL_M2 | 2.01 m² | derived | Cd·A for sideways air. | **The storm push.** Calibrated so a 51 m/s wind in 0.022 kg/m³ air gives NASA's 57.58 N (S3, S4). |

### Radars `VIKING_RADAR_*` (9)
Both antennas sit at the centre of the underside. Published errors are treated as 3σ.

| Key | Value | Status | What it is | Why it stays |
|---|---|---|---|---|
| PERIOD_S | 0.2 s | published | Both radars give a new reading every 0.2 s. | Readings can be up to 0.2 s old. |
| ALT_MIN_RANGE_M | 41 m | published | Altimeter readings below 41 m are thrown away. | No altitude for the last 41 m (Viking's altimeter got false echoes off the lander). |
| ALT_ERR_FRAC | 0.045 | published | Altimeter error proportional to height. | Together with ALT_ERR_M: ~1.1 m (1σ) at 41 m, 2 m at 100 m. |
| ALT_ERR_M | 1.5 m | published | Fixed part of the altimeter error. | Same. |
| DOPPLER_SPLAY_DEG | 21.5° | published | Doppler beam angle from straight down. | Which beams still see the ground when tilted; how velocity is worked out. |
| DOPPLER_CLOCK_DEG | 45° | published | Rotation of the 4 beams about the vertical. | Same. |
| DOPPLER_MIN_SPEED_MPS | 1 m/s | published | A beam loses lock below this closing speed. | No velocity reading near hover, just as the agent slows for touchdown. |
| DOPPLER_ERR_FRAC | 0.015 | published | Velocity error proportional to speed. | Together with ERR_MPS: ~0.1 m/s (1σ) near touchdown. |
| DOPPLER_ERR_MPS | 0.3 m/s | published | Smallest velocity error. | Same. |

### Landing requirements `VIKING_REQ_*` (3)
| Key | Value | Status | What it is | Why it stays |
|---|---|---|---|---|
| VERTICAL_SPEED_MPS | 3.34 m/s | derived | Fastest allowed downward speed at first contact. | Faster breaks a leg (2.44 + 0.9 m/s, S5). |
| HORIZONTAL_SPEED_MPS | 1.22 m/s | published | Fastest allowed sideways speed at first contact. | Faster breaks a leg (S5). |
| SETTLE_KE_FRAC | 0.005 | published | "Settled" means resting on all three feet with motion energy below 0.5 % of the impact energy. | Decides when the landing is over and can be judged (S4). |

**Success** means all of these:
- no broken leg;
- no tip-over;
- no body strike;
- settled;
- the engines are off by the end of the episode;
- the fuel did not run out in the air.

Pad radius and tilt limit come at M2 (O3).

### Reference `VIKING_REF_*` (1)
| Key | Value | Status | What it is | Why it stays |
|---|---|---|---|---|
| TIPOVER_P_45DEG | 0.85 | published | NASA's probability of tipping over on a 45° slope (crushable legs). | The slope test prints ours next to it. With rigid legs on μ = 0.6 ground ours is 1.00, because the feet slide. |

### Mars `MARS_*` (2)
| Key | Value | Status | What it is | Why it stays |
|---|---|---|---|---|
| GRAVITY_MPS2 | 3.71 | published | Surface gravity. | The weight the engines fight. |
| AIR_DENSITY_KGPM3 | 0.016 | published | Air density, used at all heights (it changes less than 5 % over 500 m). | Sets every aerodynamic force. |

### Wind `MARS_WIND_*` (1)
| Key | Value | Status | What it is | Why it stays |
|---|---|---|---|---|
| STORM_MPS | 17, 30 m/s | published | Dust-storm wind range (S11). | M2 draws storm winds from it; calm episodes have no wind. |

### Terrain `MARS_TERRAIN_*` (3)
| Key | Value | Status | What it is | Why it stays |
|---|---|---|---|---|
| SLOPE_DEG | 0° | assumed | Default ground slope. | Read at every reset; M2 randomises it. |
| SLOPE_DIRECTION_DEG | 0° | assumed | Default downhill direction. | Same. |
| FRICTION | 0.6 | assumed | How much the feet grip. | Decides whether the feet slide at touchdown. |

### Simulation `SIM_*` (4)
| Key | Value | Status | What it is | Why it stays |
|---|---|---|---|---|
| PHYSICS_DT_S | 0.002 s | assumed | Physics step. | Accuracy and stability. |
| CONTROL_HZ | 25 | assumed | Agent decisions per second. | Viking's 40 ms guidance cycle (S3). |
| CONTACT_TIMECONST_S | 0.005 s | assumed | How hard the ground feels in MuJoCo. | With rigid legs this alone shapes the impact (a 2.44 m/s touchdown rebounds at ~0.29 m/s). |
| SENSOR_NOISE_SCALE | 1 | assumed | Radar-noise switch: 1 = Viking's errors, 0 = perfect readings. | On by default (decision 7); tests set 0 when they need exact readings. |

## Parked at the end of `.env` (8)
- **Viking's final-phase guidance, for M3:** a constant 2.44 m/s descent from 16.8 m; axial throttle limit 0.84.
- **Touchdown tilt and pitch/yaw-rate limits, for O3 at M2:** 5° and 7°/s.
- **Slope and friction distributions, for M2:** 90 % of slopes below 11.7°; friction 0.2–1.0.
- **Gear stability target, for M5:** 0.997.

To use one, remove its leading `# ` and add its field to `lander/config.py`.

## Moved into the code
These only shape the drawing, or cannot change, so they are constants with their sources in a
comment:
- **`lander/config.py`:** standard gravity, `G0_MPS2`.
- **`lander/geometry.py`:** the body underside height and the number of Doppler beams.
- **`lander/mjcf.py`:** body, tank, strut, bipod and footpad drawing sizes, and the contact damping
  ratio 1 (MuJoCo's default).

## Proposed for M2 (built with the training environment)
| What | Proposal | Status |
|---|---|---|
| Start | 200–500 m up, falling 10–30 m/s, sideways 0–10 m/s, tilt 0–15°, any heading, turn rates up to 2°/s | height: decided; the rest assumed |
| Weather | 30 % storms (steady 17–30 m/s, random direction), 70 % no wind | 30 % assumed; range published |
| Ground | slope and friction from the parked distributions | published |
| Actions | 3 throttles + engines off | decided; O2 (raw throttles or a mixer) still open |
| Still open | O3 (pad radius, arena, time limit, tilt limit); O4 (observations including how the agent senses touchdown; reward) | — |

## Training for calm air and dust storms
1. **The storm is a gentle, steady push.**
   - Mars air is thin: a 17–30 m/s storm pushes with 5–15 N, under 1 % of the weight.
   - Measured: uncorrected, that is 0.22–0.69 m/s of drift after 30 s, up to half of the 1.22 m/s sideways budget.
2. **What the agent must learn.**
   - Lean a tenth to a third of a degree into the wind, judging the wind only from its own drift.
   - The hardest part is the end. Below 41 m there is no altitude reading. Once the closing speed drops under 1 m/s there is no velocity reading either, and the push continues.
3. **Training.** One agent, with storms in 30 % of episodes. It is not told which.
4. **If learning stalls.** The brief's first fix is a curriculum: start with calm air and low starts, then raise the storm share and the start height as success climbs. Log each change.
5. **Scoring at M5.**
   - Two fixed test sets, calm and storm, of e.g. 500 episodes each, using seeds never seen in training.
   - For each set: success rate with 95 % confidence intervals over at least 5 training seeds, next to the classical controller on the same episodes, plus the sideways touchdown speed.
6. **Harder storms later (fuzzing).** Winds above 30 m/s. Gusts and density spread can return then if wanted.
