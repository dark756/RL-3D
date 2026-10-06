"""Rigid legs (brief D12): the leg-break speed limits (S5) and touchdown on slopes (S4 p15)."""

from __future__ import annotations

import numpy as np

from conftest import CLEAN
from lander.scenarios import LOW_PROPELLANT_KG, hover_then_cut


def _touchdown(sim, velocity):
    st = sim.state_above_ground(0.2, velocity=velocity, propellant_kg=LOW_PROPELLANT_KG)
    sim.reset(initial_state=st)
    while sim.t < 6.0 and not sim.monitor.report.done:
        sim.step(hover_then_cut(sim))
    return sim.monitor.report


def test_leg_breaks_above_the_design_speeds(make_sim, cfg):
    sim = make_sim(CLEAN)
    ok = _touchdown(sim, (0.0, 0.0, -3.0))
    assert ok.success and not ok.leg_broken                      # 3.0 m/s down is within 3.34 m/s
    hard = _touchdown(sim, (0.0, 0.0, -3.6))
    assert hard.leg_broken and hard.done and "vertical_speed" in hard.failures()   # decided at first contact
    sideways = _touchdown(sim, (1.5, 0.0, -2.4))
    assert sideways.leg_broken and "horizontal_speed" in sideways.failures()
    assert cfg.req.vertical_speed_mps == 3.34 and cfg.req.horizontal_speed_mps == 1.22


def _slope_drops(make_sim, slope, n, seed0):
    sim = make_sim()
    rng = np.random.default_rng(seed0)
    tipped = 0
    for i in range(n):
        heading = float(rng.uniform(0.0, 360.0))
        v = (float(rng.normal(0, 0.1)), float(rng.normal(0, 0.1)), -float(rng.normal(2.43, 0.13)))
        st = sim.state_above_ground(0.2, velocity=v, heading_deg=heading, propellant_kg=LOW_PROPELLANT_KG,
                                    slope_deg=slope, slope_direction_deg=0.0)
        sim.reset(seed=seed0 + i, initial_state=st, options={"slope_deg": slope, "slope_direction_deg": 0.0})
        while sim.t < 6.0 and not sim.monitor.report.done:
            sim.step(hover_then_cut(sim))
        tipped += sim.monitor.report.tipped_over
    return tipped / n


def test_steep_slopes_tip_over_gentle_slopes_do_not(make_sim, cfg):
    p45 = _slope_drops(make_sim, 45.0, 12, 100)
    p10 = _slope_drops(make_sim, 10.0, 6, 200)
    print(f"tip-over fraction: 45 deg {p45:.2f} (S4, crushable legs: {cfg.ref.tipover_p_45deg}), 10 deg {p10:.2f}")
    assert p45 >= 0.6
    assert p10 == 0.0
