"""Touchdown judging and the section 7 outcome on the scripted scenarios."""

from __future__ import annotations

import math

import numpy as np
import pytest

from conftest import CLEAN
from lander.scenarios import make_scenario, run_scenario
from lander.touchdown import stability_distance
from lander.vehicle import Command


def test_stability_distance_geometry():
    r = 1.0
    pads = np.array([[r * math.cos(a), r * math.sin(a), 0.0] for a in np.radians([90, 210, 330])])
    assert stability_distance(np.zeros(3), pads) == pytest.approx(0.5 * r)        # inradius
    assert stability_distance(np.array([0.0, -0.6, 0.8]), pads) == pytest.approx(-0.1)   # outside edge
    # 0.2 m toward a footpad: the two edges meeting there get 0.2 * sin(30 deg) closer; height is ignored
    assert stability_distance(np.array([0.0, 0.2, 5.0]), pads) == pytest.approx(0.4)


def _outcome(make_sim, name):
    sim = make_sim()
    res = run_scenario(sim, make_scenario(name, sim))
    return res[-1].report, res


def test_nominal_drop_succeeds(make_sim):
    r, res = _outcome(make_sim, "drop")
    assert r.success and r.settled and not r.tipped_over and r.engines_off
    assert r.vz_at_contact == pytest.approx(2.44, abs=0.05)
    events = [e for x in res for e in x.events]
    assert events.index("touchdown") < events.index("engines_cut") < events.index("settled")


def test_hard_drop_breaks_a_leg(make_sim):
    r, res = _outcome(make_sim, "hard_drop")
    assert not r.success and r.leg_broken and "vertical_speed" in r.failures()
    assert "leg_broken" in [e for x in res for e in x.events]


def test_free_fall_crashes(make_sim):
    r, _ = _outcome(make_sim, "freefall")
    assert not r.success and r.leg_broken and r.vz_at_contact > 20


def test_engines_must_be_cut_before_the_landing_counts(make_sim):
    sim = make_sim(CLEAN)
    st = sim.state_above_ground(1.0, velocity=(0.0, 0.0, -1.0), propellant_kg=5.0)
    sim.reset(initial_state=st)
    while not sim.monitor.report.done:                   # minimum thrust (828 N < weight), never cut
        res = sim.step(Command((0.0, 0.0, 0.0)))
    r = res.report
    assert r.settled and not r.success and r.failures() == ["engines_off"]
    res = sim.step(Command((0.0, 0.0, 0.0), engines_off=True))
    assert res.report.success


def test_out_of_propellant_is_flagged(make_sim):
    sim = make_sim()
    st = sim.state_above_ground(40.0, propellant_kg=0.5)
    sim.reset(initial_state=st)
    res = None
    while sim.t < 12 and not sim.monitor.report.done:
        res = sim.step()
    assert "propellant_exhausted" in [e for e in (res.events if res else [])] or sim.vehicle.empty
    assert sim.monitor.report.propellant_exhausted_airborne
    assert "propellant_left_at_touchdown" in sim.monitor.report.failures()
