"""The generated MuJoCo model: one rigid body, mass = dry + propellant, fixed CG and inertia (D15)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from lander.geometry import build_geometry
from lander.vehicle import Command


def test_model_dimensions(make_sim):
    sim = make_sim()
    assert sim.model.nq == 7 and sim.model.nv == 6          # one free body: rigid legs, no joints
    assert len(sim.idx.pad_geoms) == 3 and len(sim.idx.engine_sites) == 3


def test_mass_is_dry_plus_propellant_with_fixed_cg_and_inertia(make_sim, cfg):
    sim = make_sim()
    b, m = sim.idx.lander_body, cfg.mass
    for prop in (60.0, 20.0, 0.0):
        sim.reset(initial_state=sim.state_above_ground(5.0, propellant_kg=prop), options={"engines_enabled": False})
        assert sim.model.body_mass[b] == pytest.approx(m.dry_kg + prop)
    sim.reset(initial_state=sim.state_above_ground(100.0))
    for _ in range(50):                                     # 2 s of hover burns ~2.5 kg
        sim.step(Command(sim.hover_trim()))
    assert sim.vehicle.propellant < m.start_propellant_kg - 2.0
    assert sim.model.body_mass[b] == pytest.approx(m.dry_kg + sim.vehicle.propellant)
    assert sim.model.body_ipos[b] == pytest.approx([0.0, 0.0, m.cg_z_m])          # CG never moves
    assert sim.model.body_inertia[b] == pytest.approx([m.ixx_kgm2, m.iyy_kgm2, m.izz_kgm2])


def test_footprint_matches_drawings(cfg):
    g = build_geometry(cfg)
    for pad, az in zip(g.footpads, cfg.leg.azimuth_deg):
        assert math.hypot(pad[0], pad[1]) == pytest.approx(cfg.leg.footpad_radius_m)
        assert math.degrees(math.atan2(pad[1], pad[0])) % 360 == pytest.approx(az % 360)
        assert pad[2] == pytest.approx(cfg.leg.footpad_z_m)
    # CG height / footpad radius (S4 p5)
    assert cfg.mass.cg_z_m / cfg.leg.footpad_radius_m == pytest.approx(0.566, abs=0.001)


def test_model_is_built_from_env(make_sim):
    """Editing a geometry fact changes the compiled model (no hidden XML copy)."""
    a, b = make_sim(), make_sim({"VIKING_LEG_FOOTPAD_RADIUS_M": 1.5})
    pa = a.model.geom_pos[a.idx.pad_geoms[0]]
    pb = b.model.geom_pos[b.idx.pad_geoms[0]]
    assert not np.allclose(pa, pb)
