"""Text of the code guide v2's Section 2: every .env variable, why it is kept, and where the old keys went.

Numbers quoted here were computed from .env (lander.config.derived) or measured with the M1 code on
2026-10-05/06 (misalignment and storm runs, slope drops, drop-scenario rebound).
Mini-markup (make_code_guide_v2.fmt): `code`, **bold**, _{sub}, ^{sup}; @@name@@ = link to a subsection.
"""

# ------------------------------------------------------------------------------------------------ groups
GROUP_TEXT = {
    "VIKING_ENGINE_": (
        "Engines",
        "Three fixed engines, throttled one by one: the agent's only actuators (no gimbal, no roll jets). A valve "
        "setting u from 0 to 1 gives 276 N + u × 2,391 N per engine. Only the engines-off command, which is "
        "permanent, or an empty tank gives zero thrust."),
    "VIKING_MASS_": (
        "Mass and inertia",
        "Mass = dry mass + propellant left, updated every physics step. The CG and the inertia never change (D15)."),
    "VIKING_LEG_": (
        "Legs",
        "Rigid legs ending in three contact spheres. The break rule has no variables of its own: it uses the two "
        "speed limits under @@req@@."),
    "VIKING_AERO_": (
        "Aerodynamics",
        "Drag on each body axis = ½ × air density × Cd·A × (relative speed)^{2}, where relative speed = lander "
        "velocity − wind; it acts at the CG. Cd·A is the drag coefficient times the area facing the air. Only the "
        "product is ever used, so each axis has one number."),
    "VIKING_RADAR_": (
        "Radars",
        "Viking's radar altimeter and four-beam Doppler radar, both antennas at the centre of the underside. The "
        "published accuracies are treated as 3σ (assumed). `SIM_SENSOR_NOISE_SCALE` switches the noise on (1, "
        "the default) or off (0)."),
    "VIKING_REQ_": (
        "Landing requirements",
        "These decide success: no broken leg, no tip-over, no body strike, settled, engines off by the end of the "
        "episode, and propellant left at touchdown. The pad radius and tilt limit come at M2 (open decision O3)."),
    "VIKING_REF_": (
        "Reference",
        "Published results the simulation is compared with. Nothing in the physics reads them."),
    "MARS_": (
        "Mars",
        "Both are held constant over the last 500 m."),
    "MARS_WIND_": (
        "Wind",
        "One steady wind per episode: none in calm episodes, a storm wind from this range otherwise. Calm winds "
        "and gusts were measured and removed (@@gone@@)."),
    "MARS_TERRAIN_": (
        "Terrain",
        "The ground is a tilted plane through the target. These are the defaults a reset uses when its options "
        "give no slope or friction."),
    "SIM_": (
        "Simulation",
        "Numerical settings, not physics. All four are assumed."),
}

# order of the subsections (the order of .env)
GROUP_ORDER = ["VIKING_ENGINE_", "VIKING_MASS_", "VIKING_LEG_", "VIKING_AERO_", "VIKING_RADAR_", "VIKING_REQ_",
               "VIKING_REF_", "MARS_", "MARS_WIND_", "MARS_TERRAIN_", "SIM_"]

# ------------------------------------------------------------------------------------------------ the 41 variables
# key -> (what it is, why it is kept)
VARS = {
    # ---------------------------------------------------------------- engines
    "VIKING_ENGINE_THRUST_MIN_N": (
        "Thrust of one engine at its lowest lit setting (valve 0).",
        "Viking could not throttle lower (10:1 range), so three lit engines always push at least 828 N, 35 % of "
        "the start weight. Only the engines-off command gives zero, which is why the last metres need a cut-off "
        "decision. It also limits steering near hover: there the engines run at 770–790 N, so each can be "
        "throttled down only about 500 N but up about 1,900 N."),
    "VIKING_ENGINE_THRUST_MAX_N": (
        "Thrust of one engine at full setting (valve 1).",
        "How hard the lander can brake: three engines give 3.4 × the start weight, a net deceleration of "
        "8.9 m/s². From 30 m/s that stops the lander in about 50 m (ignoring the lag), so it sets how late braking "
        "can start. It also scales the plumes and the HUD's thrust percentages."),
    "VIKING_ENGINE_ISP_S": (
        "Specific impulse: the engines' fuel economy.",
        "Propellant flow = thrust ÷ (Isp × 9.807 m/s²). Hovering at the start burns 1.23 kg/s, so the 60 kg "
        "load lasts 49 s of hover or 14 s at full thrust. It turns every throttle choice into a fuel cost and "
        "decides when the tank runs dry, which fails the landing if the lander is still in the air."),
    "VIKING_ENGINE_X_M": (
        "Sideways (X) position of each engine: +0.835 and −0.835 m for engines 1 and 2, 0 for engine 3.",
        "There is no gimbal, so unequal throttles are the only way to tilt the lander. Engines 1 and 2 sit "
        "0.835 m either side of the centre line: 100 N more on one than the other gives 83.5 N·m about Y, an "
        "angular acceleration of 22 °/s²."),
    "VIKING_ENGINE_Y_M": (
        "Fore-aft (Y) position of each engine: 0.482 m for engines 1 and 2, −0.989 m for engine 3.",
        "Engine 3 working against engines 1 and 2 tilts the lander about X: 100 N more on engine 3 gives "
        "98.9 N·m, 35 °/s². The published positions are slightly unequal (2 × 0.482 ≠ 0.989), so equal throttles "
        "at hover leave a 20 N·m twist and engine 3 must run 2.5 % lower. `hover_trim` does this; the agent has "
        "to learn the same offset."),
    "VIKING_ENGINE_T90_S": (
        "Time for an engine to complete 90 % of any thrust change, up or down, including the tail-off after "
        "engines-off (a first-order lag, τ = 23 ms).",
        "The agent decides every 40 ms, but each command is only 82 % complete by the next decision, so it has "
        "to lead its commands rather than react. After the cut, thrust needs 53 ms to fall to 10 %, so the "
        "lander is still being braked for a moment after it touches. One value replaces the separate rise and "
        "fall lags (50 and 53 ms) and the rate limit, which repeated it."),
    "VIKING_ENGINE_MISALIGN_DEG": (
        "Each engine's thrust points 0.044° off its axis, in a random direction drawn every episode.",
        "The only steady disturbance in calm air. Uncorrected for 30 s it tilts the lander 4–55° (median 19°, "
        "20 seeds), pushes it 3–34 m/s sideways and spins it at 2–9 °/s. The direction changes every episode, so "
        "no fixed trim can cancel it: the agent must learn real attitude feedback. Without it every calm episode "
        "would be the same, and one fixed trim would hold the lander level (the hover test stays at 0.000° of "
        "tilt)."),
    # ---------------------------------------------------------------- mass
    "VIKING_MASS_DRY_KG": (
        "The lander with empty tanks: 657 kg fuelled (S2) minus 83.9 kg of terminal-descent propellant (S3).",
        "Mass at every physics step = 573.1 kg + the propellant left. As the tank empties, the weight falls by up "
        "to 9.5 % (2,349 to 2,126 N) and the thrust needed to hover falls with it, so the agent must keep "
        "adjusting."),
    "VIKING_MASS_START_PROPELLANT_KG": (
        "Propellant on board when an episode starts, 200–500 m up.",
        "The fuel budget: 49 s of hovering at the start weight. It decides whether a slow, careful descent is "
        "affordable. Assumed: our estimate is that a careful landing from 500 m at 30 m/s needs about 45–55 kg, "
        "and the M3 classical controller will check it. New, because episodes no longer start at engine ignition "
        "with the full 83.9 kg."),
    "VIKING_MASS_IXX_KGM2": (
        "Moment of inertia about the body X axis (S4, at touchdown mass).",
        "How quickly a torque about X turns the lander. That is the axis engine 3 controls against engines 1 and "
        "2. Held constant because the propellant is at most 9.5 % of the mass."),
    "VIKING_MASS_IYY_KGM2": (
        "Moment of inertia about the body Y axis.",
        "How quickly a torque about Y turns the lander. That is the axis engines 1 and 2 control by throttling "
        "against each other."),
    "VIKING_MASS_IZZ_KGM2": (
        "Moment of inertia about the thrust axis (Z).",
        "Nothing controls rotation about Z (there are no roll jets), so this alone sets how fast the misalignment "
        "spins the lander: 2–9 °/s after 30 s."),
    "VIKING_MASS_CG_Z_M": (
        "Height of the centre of gravity above the footpad plane, on the centre line.",
        "Sets how easily the lander tips: CG height ÷ footpad radius = 0.566 (S4). Resting on a slope, it tips "
        "once the slope passes 41.5° with an edge of the footprint facing downhill (atan of 0.70 m ÷ 0.79 m). It "
        "is also the 0.26 m lever arm through which the misaligned thrust tilts the lander."),
    # ---------------------------------------------------------------- legs
    "VIKING_LEG_AZIMUTH_DEG": (
        "Directions of the three legs around the centre line (leg 1 on +Y).",
        "With the footpad radius it places the three feet: the triangle the lander stands on and tips over, and "
        "the edges the stability distance is measured to. The published directions put the legs between the "
        "engines, as on Viking; without them we would have to invent positions."),
    "VIKING_LEG_FOOTPAD_RADIUS_M": (
        "Distance from the centre line to each foot (55.00 in).",
        "The size of the footprint. Its edges lie 0.70 m from the centre, and that distance is the tip-over "
        "margin behind the stability check and the 41.5° static limit."),
    "VIKING_LEG_FOOTPAD_Z_M": (
        "Height of each foot's junction above the ground plane, used as the contact-sphere radius (3.2 cm).",
        "Where the feet meet the ground. MuJoCo needs a contact radius, and the footpad altitude and the moment "
        "of touchdown depend on it. Its effect is small; it stays because the published size beats an invented "
        "one."),
    # ---------------------------------------------------------------- aero
    "VIKING_AERO_CDA_AXIAL_M2": (
        "Cd·A for air moving along the thrust axis: Cd 1.05 (assumed) × the 1.71 m² body outline (S2).",
        "Drag from the descent itself, plus the axial part of a storm acting on a tilted lander. Small: 13 N at "
        "30 m/s, 0.55 % of the weight. One of the @@close@@."),
    "VIKING_AERO_CDA_LATERAL_M2": (
        "Cd·A for air moving sideways, calibrated to NASA's numbers.",
        "**The storm's push.** A 17–30 m/s storm pushes with 4.6–14.5 N, which drifts an uncorrected lander "
        "0.22–0.69 m/s in 30 s: up to 57 % of the 1.22 m/s sideways limit. Calibrated so that a 51 m/s wind in "
        "0.022 kg/m³ air gives NASA's 57.58 N (S3, S4). Without it a storm would have no effect."),
    # ---------------------------------------------------------------- radars
    "VIKING_RADAR_PERIOD_S": (
        "Both radars give a new reading every 0.2 s and hold it in between.",
        "The agent decides 25 times a second but gets fresh radar data only 5 times a second, so it must act on "
        "readings up to 0.2 s old. At full braking (8.9 m/s²) a held velocity can be 1.8 m/s out of date. One "
        "value replaces the two equal radar periods."),
    "VIKING_RADAR_ALT_MIN_RANGE_M": (
        "Altimeter readings below 41 m are thrown away.",
        "Viking ignored them because of false echoes off its own structure. So the last 41 m, the riskiest part, "
        "have no altitude reading at all: the agent must carry its height forward using the Doppler velocity and "
        "the IMU."),
    "VIKING_RADAR_ALT_ERR_FRAC": (
        "The part of the altimeter error that grows with range: 4.5 % (3σ).",
        "Dominates high up. The 1σ error = (4.5 % × range + 1.5 m) ÷ 3 is 8 m at 500 m and 2 m at 100 m, so the "
        "starting height is known only to within metres."),
    "VIKING_RADAR_ALT_ERR_M": (
        "The fixed part of the altimeter error: 1.5 m (3σ).",
        "Dominates low down: together with ALT_ERR_FRAC the error is 1.1 m (1σ) at 41 m, the last valid "
        "reading."),
    "VIKING_RADAR_DOPPLER_SPLAY_DEG": (
        "Angle of each of the four Doppler beams from straight down.",
        "Decides which beams still hit the ground when the lander tilts, and how the beam speeds combine into a "
        "velocity. Each beam sees only cos 21.5° = 93 % of a vertical descent, so in a straight, level descent "
        "all four lose lock below 1.07 m/s."),
    "VIKING_RADAR_DOPPLER_CLOCK_DEG": (
        "Rotation of the beams about the thrust axis: 45° from the pitch and yaw axes.",
        "Fixes the beam directions in the body, and so which beam drops out first as the lander tilts about a "
        "given axis. One of the @@close@@: the uncontrolled spin turns the beams anyway."),
    "VIKING_RADAR_DOPPLER_MIN_SPEED_MPS": (
        "A beam loses lock when its closing speed falls below 1 m/s.",
        "No velocity reading near hover. If the agent slows below about 1 m/s of descent before touchdown, it "
        "loses its speed measurement, so it cannot creep down slowly on radar data alone."),
    "VIKING_RADAR_DOPPLER_ERR_FRAC": (
        "The part of the velocity error that grows with speed: 1.5 % (3σ).",
        "Dominates above 20 m/s: the 1σ error = max(1.5 % × speed, 0.3 m/s) ÷ 3 is 0.15 m/s at 30 m/s, early in "
        "the descent."),
    "VIKING_RADAR_DOPPLER_ERR_MPS": (
        "The smallest velocity error: 0.3 m/s (3σ).",
        "Dominates below 20 m/s, so for the whole final approach the error is 0.1 m/s (1σ): 8 % of the 1.22 m/s "
        "sideways limit, and the main sensor error near touchdown."),
    # ---------------------------------------------------------------- requirements
    "VIKING_REQ_VERTICAL_SPEED_MPS": (
        "Fastest allowed descent speed at first footpad contact: Viking's 2.44 m/s design speed + 0.9 m/s.",
        "Any faster and a leg breaks (Viking's leg design limit, S5), failing the landing on the spot. It defines "
        "how gently the agent must arrive. As the rigid-leg break rule it replaces the 22 removed leg values "
        "(crush struts, leg switches, load limiters, leg masses and the old leg criteria)."),
    "VIKING_REQ_HORIZONTAL_SPEED_MPS": (
        "Fastest allowed sideways speed at first footpad contact.",
        "Any faster and a leg breaks (S5). This is the budget that storm drift (up to 0.69 m/s uncorrected) and "
        "Doppler error (0.1 m/s) eat into."),
    "VIKING_REQ_SETTLE_KE_FRAC": (
        "Settled = resting on all three feet with kinetic energy below 0.5 % of the impact energy, held for "
        "0.5 s.",
        "Ends the episode and lets the landing be judged (S4's criterion). Without it there is no clear moment "
        "when a landing is over."),
    # ---------------------------------------------------------------- reference
    "VIKING_REF_TIPOVER_P_45DEG": (
        "NASA's probability that Viking tips over landing on a 45° slope (with crushable legs).",
        "A comparison target, not physics: the slope test prints our rate next to it. With rigid legs on "
        "μ = 0.6 ground ours is 1.00 (24 drops), because the feet slide; at 10–30° it is 0.00. One of the "
        "@@close@@."),
    # ---------------------------------------------------------------- Mars
    "MARS_GRAVITY_MPS2": (
        "Mars surface gravity.",
        "The weight the engines fight: 2,349 N at the start mass. It sets the hover thrust and the free-fall "
        "speed, and gives the free-fall test its exact answer."),
    "MARS_AIR_DENSITY_KGPM3": (
        "Air density, used at every height (it changes by only 4.4 % over 500 m).",
        "Scales every aerodynamic force, since drag is proportional to density. Thin Mars air is why even a "
        "30 m/s storm pushes with only 14.5 N. The ±0.006 spread was removed because the storm range already "
        "covers it."),
    "MARS_WIND_STORM_MPS": (
        "Dust-storm wind speeds at the Viking sites (S11).",
        "Defines the storm half of training: M2 will draw a steady storm wind from 17–30 m/s in about 30 % of "
        "episodes. In M1 only the storm-push estimate in `derived` and the tests read it; the scripts take "
        "`--wind`."),
    # ---------------------------------------------------------------- terrain
    "MARS_TERRAIN_SLOPE_DEG": (
        "Default ground slope.",
        "The ground every reset uses unless its options say otherwise (flat). M2 will draw slopes from the "
        "parked distribution (90 % below 11.7°)."),
    "MARS_TERRAIN_SLOPE_DIRECTION_DEG": (
        "Default downhill direction, measured from +X toward +Y.",
        "Together with the lander's heading it decides which edge of the footprint faces downhill, and so how "
        "easily the lander tips."),
    "MARS_TERRAIN_FRICTION": (
        "Default footpad friction coefficient, the middle of the published 0.2–1.0.",
        "Decides whether the feet grip or slide after contact. On a 45° slope μ = 0.6 is below tan 45° = 1, so "
        "the feet slide and every drop tips over. M2 will draw it from 0.2–1.0."),
    # ---------------------------------------------------------------- simulation
    "SIM_PHYSICS_DT_S": (
        "MuJoCo's integration step.",
        "Small enough for stable contact with rigid legs and the 5 ms contact time constant, which must span at "
        "least 2 steps. Twenty physics steps make one control step."),
    "SIM_CONTROL_HZ": (
        "Agent decisions per second; each command is held for 40 ms.",
        "Matches Viking's 40 ms guidance cycle (S3). With the 53 ms engine lag and the 0.2 s radar period it "
        "sets how quickly the agent can react. Also the video frame rate."),
    "SIM_CONTACT_TIMECONST_S": (
        "How soft MuJoCo makes the ground contact (its time constant).",
        "With rigid legs this alone shapes the impact: a 2.44 m/s touchdown rebounds at about 0.29 m/s. It must "
        "stay at least 2 physics steps."),
    "SIM_SENSOR_NOISE_SCALE": (
        "Radar-noise switch: 1 = Viking's published errors, 0 = perfect readings.",
        "On by default (decided on 5 October). Tests set 0 when they need exact readings, and a reset option can "
        "change it per episode, for curricula or fuzzing."),
}

# new key -> how it was formed from the 155-key file (shown under the key)
WAS = {
    "VIKING_ENGINE_T90_S": "merged: T90_RISE_S 0.050 + T90_FALL_S 0.053",
    "VIKING_MASS_DRY_KG": "merged: FUELLED_KG 657 − PROPELLANT_KG 83.9",
    "VIKING_MASS_START_PROPELLANT_KG": "new key",
    "VIKING_AERO_CDA_AXIAL_M2": "merged: CD_AXIAL 1.05 × AREA_AXIAL_M2 1.71",
    "VIKING_AERO_CDA_LATERAL_M2": "merged: CD_LATERAL 1.0 × AREA_LATERAL_M2 2.01",
    "VIKING_RADAR_PERIOD_S": "merged: RADALT_PERIOD_S + TDLR_PERIOD_S",
    "VIKING_RADAR_ALT_MIN_RANGE_M": "was RADALT_MIN_RANGE_M",
    "VIKING_RADAR_ALT_ERR_FRAC": "was RADALT_ERR_FRAC",
    "VIKING_RADAR_ALT_ERR_M": "was RADALT_ERR_M",
    "VIKING_RADAR_DOPPLER_SPLAY_DEG": "was TDLR_SPLAY_DEG",
    "VIKING_RADAR_DOPPLER_CLOCK_DEG": "was TDLR_CLOCK_DEG",
    "VIKING_RADAR_DOPPLER_MIN_SPEED_MPS": "was TDLR_MIN_SPEED_MPS",
    "VIKING_RADAR_DOPPLER_ERR_FRAC": "was TDLR_ERR_FRAC",
    "VIKING_RADAR_DOPPLER_ERR_MPS": "was TDLR_ERR_MPS",
    "MARS_WIND_STORM_MPS": "was DUSTSTORM_MPS",
}

# new key -> the old keys it was built from (for the accounting check)
MERGED_FROM = {
    "VIKING_ENGINE_T90_S": ["VIKING_ENGINE_T90_RISE_S", "VIKING_ENGINE_T90_FALL_S"],
    "VIKING_MASS_DRY_KG": ["VIKING_MASS_FUELLED_KG", "VIKING_MASS_PROPELLANT_KG"],
    "VIKING_AERO_CDA_AXIAL_M2": ["VIKING_AERO_CD_AXIAL", "VIKING_AERO_AREA_AXIAL_M2"],
    "VIKING_AERO_CDA_LATERAL_M2": ["VIKING_AERO_CD_LATERAL", "VIKING_AERO_AREA_LATERAL_M2"],
    "VIKING_RADAR_PERIOD_S": ["VIKING_RADALT_PERIOD_S", "VIKING_TDLR_PERIOD_S"],
    "VIKING_RADAR_ALT_MIN_RANGE_M": ["VIKING_RADALT_MIN_RANGE_M"],
    "VIKING_RADAR_ALT_ERR_FRAC": ["VIKING_RADALT_ERR_FRAC"],
    "VIKING_RADAR_ALT_ERR_M": ["VIKING_RADALT_ERR_M"],
    "VIKING_RADAR_DOPPLER_SPLAY_DEG": ["VIKING_TDLR_SPLAY_DEG"],
    "VIKING_RADAR_DOPPLER_CLOCK_DEG": ["VIKING_TDLR_CLOCK_DEG"],
    "VIKING_RADAR_DOPPLER_MIN_SPEED_MPS": ["VIKING_TDLR_MIN_SPEED_MPS"],
    "VIKING_RADAR_DOPPLER_ERR_FRAC": ["VIKING_TDLR_ERR_FRAC"],
    "VIKING_RADAR_DOPPLER_ERR_MPS": ["VIKING_TDLR_ERR_MPS"],
    "MARS_WIND_STORM_MPS": ["MARS_WIND_DUSTSTORM_MPS"],
}

# ------------------------------------------------------------------------------------------------ the tests
CRITERIA = [
    "**It acts.** The code reads it, and it changes the physics, the sensors or how a landing is judged. "
    "Reference-only values were dropped, except NASA's 45° tip-over figure, which the slope test compares "
    "against.",
    "**It is not a duplicate.** No other variable already carries its effect. Dropped for this: the engine rate "
    "limit (repeats the lag), `LAG_SCALE` (repeats overriding the lag), the air-density spread (inside the storm "
    "range), and separate drag coefficients and areas (only their product is used).",
    "**It is big enough to matter.** Each candidate was measured in 30 s of open-loop hover (engines held at a "
    "fixed balance, no controller) against the landing limits of 1.22 m/s sideways and 3.34 m/s down. Calm wind "
    "(< 0.08 m/s of drift), gusts (≤ 0.09 m/s) and the IMU errors (10^{4}–10^{6}× smaller than the signals) "
    "failed; misalignment (4–55° of tilt) passed easily.",
    "**A published number beats an invented one.** Where the simulation needs some value anyway, such as a foot "
    "radius or a beam angle, the published one stays even when its effect is small.",
]

DECISIONS = [
    ("D12", "Rigid legs. A leg breaks if first contact is faster than 3.34 m/s down or 1.22 m/s sideways.",
     "The two speed limits replace 22 leg values: crush struts, leg switches, load limiters, leg masses and the "
     "clearance and peak-load criteria."),
    ("D7", "The agent commands engines off. The cut is permanent, and a landing counts only if the engines are off "
           "by the end of the episode.",
     "Viking's leg switches (TESS) and their delay go; `engines_off` joins the actions."),
    ("D16", "Episodes start at a random height of 200–500 m.",
     "8 values for the parachute, ignition and handover go; START_PROPELLANT_KG is new."),
    ("D17", "One agent learns calm and dust-storm landings from mixed episodes and is not told which.",
     "The storm range stays; calm winds and gusts go."),
    ("D15", "The CG sits on the centre line and never moves; mass = dry + propellant; inertia constant; ideal IMU at "
            "the CG; one altimeter; constant air density; one steady wind.",
     "CG offsets, touchdown mass, IMU errors and position, the second altimeter, density spread and scale height "
     "go."),
    ("D4", "Thrust misalignment stays; the roll jets go, so spin is not controlled.",
     "7 roll-jet values and the roll-rate limit go."),
    ("D15", "Viking's radar errors are on by default (settles the noise part of O4).",
     "SENSOR_NOISE_SCALE = 1."),
    ("D5", "One engine lag of 53 ms; no rate limit, no lag scale.",
     "The rise and fall lags merge into T90_S; the rate limit and LAG_SCALE go."),
]

# ------------------------------------------------------------------------------------------------ closest calls
CLOSE_CALLS = [
    ("VIKING_REF_TIPOVER_P_45DEG",
     "Nothing in the simulation. The value would move into the slope test as a constant with its source.",
     "`.env` is where every published number lives with its source (D14), and the README's validation table "
     "quotes it."),
    ("VIKING_AERO_CDA_AXIAL_M2",
     "Using the lateral 2.01 m² on every axis would raise the axial drag by 12 %: about 1.5 N at 30 m/s, under "
     "0.1 % of the weight.",
     "It is the only number for drag along the thrust axis; merging would apply a value calibrated for sideways "
     "flow."),
    ("VIKING_RADAR_DOPPLER_CLOCK_DEG",
     "Setting the beams to 0° would change only which beam drops out first as the lander tilts; the spin turns "
     "the beams in any case.",
     "The beams need some angle, and Viking's is published (S3 Fig 60)."),
]

# ------------------------------------------------------------------------------------------------ parked
PARKED = {   # key -> (needed at, what it will do)
    "VIKING_SEQ_TERMINAL_SPEED_MPS": ("M3", "Viking's constant final descent speed; the classical controller's last "
                                            "phase follows it."),
    "VIKING_SEQ_TERMINAL_ALT_M": ("M3", "Height where that constant-speed phase begins."),
    "VIKING_SEQ_AXIAL_THROTTLE_LIMIT": ("M3", "Average valve limit for the controller's descent-speed loop."),
    "VIKING_REQ_TILT_DEG": ("M2 (O3)", "Candidate tilt limit at touchdown."),
    "VIKING_REQ_RATE_PITCHYAW_DPS": ("M2 (O3)", "Candidate turn-rate limit at touchdown."),
    "MARS_TERRAIN_SLOPE_P90_DEG": ("M2", "Slope distribution for training resets: exponential, 90 % below 11.7°."),
    "MARS_TERRAIN_FRICTION_RANGE": ("M2", "Friction range for training resets."),
    "VIKING_REQ_STABILITY_PROB": ("M5", "Viking's landing-gear stability: a benchmark for the evaluation."),
}

# ------------------------------------------------------------------------------------------------ moved into the code
MOVED = [   # (old keys, where the value lives now, file)
    (["VIKING_ENGINE_COUNT", "VIKING_LEG_COUNT"],
     "Implied by the lists: `validate` requires 3 engine coordinates and 3 leg directions.", "lander/config.py"),
    (["VIKING_ENGINE_Z_M", "VIKING_BODY_BOTTOM_Z_M"],
     "`BODY_BOTTOM_Z_M` = 0.5265 m: the engines and both radar antennas sit on the underside.", "lander/geometry.py"),
    (["VIKING_TDLR_BEAMS"], "`DOPPLER_BEAMS` = 4.", "lander/geometry.py"),
    (["VIKING_BODY_LONG_SIDE_M", "VIKING_BODY_SHORT_SIDE_M", "VIKING_BODY_HEIGHT_M", "VIKING_BODY_TDLR_DROP_M",
      "VIKING_BODY_TDLR_SIZE_M", "VIKING_MASS_TANK_AZIMUTH_DEG", "VIKING_MASS_TANK_RADIUS_M", "VIKING_MASS_TANK_Z_M",
      "VIKING_MASS_TANK_DIAMETER_M", "VIKING_LEG_FOOTPAD_DIAMETER_M", "VIKING_LEG_STRUT_TOP_RADIUS_M",
      "VIKING_LEG_STRUT_TOP_Z_M", "VIKING_LEG_BIPOD_RADIUS_M", "VIKING_LEG_BIPOD_HALF_SPAN_M", "VIKING_LEG_BIPOD_Z_M"],
     "Drawing constants `BODY_*`, `RADAR_HOUSING_M`, `TANKS`, `FOOTPAD_DIAMETER_M`, `STRUT_TOP` and `BIPOD`, each "
     "with its source. They shape the model you see; the hull, radar housing, engines and tanks also count for "
     "body strikes.", "lander/mjcf.py"),
    (["SIM_G0_MPS2"], "`G0_MPS2` = 9.80665: standard gravity is a definition, not a setting.", "lander/config.py"),
    (["SIM_CONTACT_DAMPRATIO"], "The 1 in the contact `solref`: MuJoCo's default damping ratio.", "lander/mjcf.py"),
    (["VIKING_REF_CG_HEIGHT_RATIO"], "The test asserts CG height ÷ footpad radius = 0.566.", "tests/test_model.py"),
]

# ------------------------------------------------------------------------------------------------ removed
REMOVED = [   # (theme, old keys, why)
    ("Roll jets and the roll-rate limit",
     ["VIKING_ROLLJET_COUNT", "VIKING_ROLLJET_THRUST_N", "VIKING_ROLLJET_ISP_S", "VIKING_ROLLJET_ARM_M",
      "VIKING_ROLLJET_Z_M", "VIKING_ROLLJET_VALVE_DELAY_S", "VIKING_ROLLJET_MIN_ON_S", "VIKING_REQ_RATE_ROLL_DPS"],
     "D4: spin is not controlled. The misalignment spins the lander 2–9 °/s over 30 s, which moves the feet at "
     "most 0.22 m/s at touchdown: harmless with rigid legs. The actions shrink to 3 throttles + engines off."),
    ("Crushable legs, leg switches (TESS) and the leg criteria",
     ["VIKING_STRUT_PRESTROKE_FORCE_N", "VIKING_STRUT_CRUSH_ONSET_N", "VIKING_STRUT_CRUSH_RAMP_M",
      "VIKING_STRUT_CRUSH_PLATEAU_N", "VIKING_STRUT_PLATEAU_STROKE_M", "VIKING_STRUT_TENSION_N",
      "VIKING_STRUT_STIFFNESS_NPM", "VIKING_STRUT_DAMPING_NSPM", "VIKING_STRUT_SCATTER_3SIGMA",
      "VIKING_TESS_TRIP_STROKE_M", "VIKING_TESS_DELAY_MEAN_S", "VIKING_TESS_DELAY_3SIGMA_S",
      "VIKING_MASS_LEG_MOVING_KG", "VIKING_MASS_GEAR_KG", "VIKING_REQ_STRUT_STROKE_M", "VIKING_REQ_PEAK_LOAD_G",
      "VIKING_REQ_CLEARANCE_M", "VIKING_REF_STROKE_3SIGMA_M", "VIKING_REF_LIMITER_YIELD_N",
      "VIKING_REF_STRUT_LENGTH_M", "VIKING_REF_CLEARANCE_MEAN_M", "VIKING_REF_CLEARANCE_3SIGMA_M"],
     "D12 and D7: rigid legs that break above the two speed limits, and the agent, not leg switches, cuts the "
     "engines. A force limit on rigid legs would depend on MuJoCo's contact softness. The body cannot near the "
     "ground before a tip-over, so no clearance limit; peak g is set by the contact softness. The 20 kg of leg "
     "mass is part of the dry mass."),
    ("Moving CG",
     ["VIKING_MASS_TOUCHDOWN_KG", "VIKING_MASS_CG_X_M", "VIKING_MASS_CG_Y_M"],
     "D15: the CG sits on the centre line (Viking's was 4 cm off). The touchdown mass only anchored the published "
     "CG and inertia values."),
    ("Viking's earlier flight phases",
     ["VIKING_SEQ_IGNITION_ALT_M", "VIKING_SEQ_CHUTE_RELEASE_S", "VIKING_SEQ_WARMUP_S", "VIKING_SEQ_WARMUP_THROTTLE",
      "VIKING_SEQ_HANDOVER_SPEED_MPS", "VIKING_SEQ_HANDOVER_TILT_DEG", "VIKING_SEQ_PROBABLE_SPEED_MPS",
      "VIKING_SEQ_DESCENT_TIME_S"],
     "D16: episodes start 200–500 m up, after the parachute, engine ignition and the handover to terminal "
     "guidance."),
    ("Limits never reached below 500 m",
     ["VIKING_RADALT_MAX_RANGE_M", "VIKING_TDLR_MAX_RANGE_M", "VIKING_TDLR_MAX_SPEED_MPS"],
     "137 km, 6 km and 225 m/s; an episode starts at most 500 m up and falls at tens of m/s."),
    ("Engine settings with no effect, or that repeat the lag",
     ["VIKING_ENGINE_RATE_LIMIT_NPS", "SIM_LAG_SCALE", "VIKING_ENGINE_CANT_DEG", "VIKING_ENGINE_NOISE_FRAC"],
     "D5: the rate limit added 12 ms, only on full-range jumps, and its value was guessed from the lag; "
     "LAG_SCALE does what overriding the lag does. Cant and thrust noise were both 0 (no data)."),
    ("IMU errors and position",
     ["VIKING_IMU_POS_M", "VIKING_IMU_RATE_HZ", "VIKING_IMU_GYRO_BIAS_DPH", "VIKING_IMU_GYRO_SCALE_PPM",
      "VIKING_IMU_ACCEL_BIAS_G", "VIKING_IMU_ACCEL_SCALE_PPM", "VIKING_IMU_MISALIGN_MRAD"],
     "The IMU is ideal, sits at the CG and is read every control step. Its published errors are "
     "10^{4}–10^{6}× smaller than the signals; the off-centre position only added up to ~10 m/s² of "
     "turn-induced acceleration (noise, not information)."),
    ("Second altimeter and antenna positions",
     ["VIKING_RADALT_COUNT", "VIKING_RADALT_POS_M", "VIKING_TDLR_POS_M"],
     "The two altimeters were identical, so one is modelled. Both antennas sit at the centre of the underside; "
     "the old positions were guesses (not dimensioned in the sources)."),
    ("Calm wind, gusts, shear and air variations",
     ["MARS_WIND_SPEED_MPS", "MARS_WIND_DIRECTION_DEG", "MARS_WIND_GUST_SIGMA_MPS", "MARS_WIND_GUST_TAU_S",
      "MARS_WIND_SHEAR_EXPONENT", "MARS_WIND_REF_HEIGHT_M", "MARS_WIND_SUMMER_MPS", "MARS_WIND_FALL_MPS",
      "MARS_AIR_DENSITY_SIGMA_KGPM3", "MARS_SCALE_HEIGHT_M", "MARS_SURFACE_PRESSURE_PA", "MARS_WIND_SURVIVAL_MPS",
      "VIKING_AERO_COP_OFFSET_Z_M"],
     "Calm winds (2–10 m/s) drift the lander < 0.08 m/s in 30 s and gusts ≤ 0.09 m/s; shear and the "
     "centre-of-pressure offset were 0. The density spread (+37 % push) sits inside the storm range (3× push); "
     "density changes 4.4 % over 500 m, and surface pressure was never used. The 90 m/s survival wind applies "
     "after landing."),
    ("Calibration inputs and NASA's touchdown results",
     ["MARS_WIND_DESIGN_99_MPS", "VIKING_REF_TD_AERO_3SIGMA_N", "VIKING_REF_TD_AERO_MEAN_N", "VIKING_REF_TD_THRUST_N",
      "VIKING_REF_TD_VZ_MPS", "VIKING_REF_TD_VZ_3SIGMA_MPS", "VIKING_REF_TD_VH_MPS", "VIKING_REF_TD_VH_3SIGMA_MPS"],
     "The 51 m/s design wind and NASA's 57.58 N aero force only calibrated CDA_LATERAL; their values sit in its "
     "note. The rest were outputs of NASA's own touchdown simulation that no code read; the thrusts also "
     "reflected the 4 cm CG offset (engine 3 ran 15 % low, now 2.5 %)."),
]
