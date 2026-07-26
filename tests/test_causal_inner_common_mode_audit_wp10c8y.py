from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"
for path in (SCRIPTS, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_common_mode_audit_wp10c8y as wp10c8y


def test_continuum_profile_basis_is_compact_and_field_limited() -> None:
    radius = np.asarray((1.7, 1.8, 2.4, 6.648, 7.0))
    basis = wp10c8y._continuum_profile_basis(
        radius,
        inner_rg=1.8,
        outer_rg=6.648,
    )
    assert basis.shape == (
        radius.size,
        5,
        wp10c8y.PROFILE_COLUMNS,
    )
    assert np.array_equal(basis[0], np.zeros_like(basis[0]))
    assert np.array_equal(basis[1], np.zeros_like(basis[1]))
    assert np.array_equal(basis[-2], np.zeros_like(basis[-2]))
    assert np.array_equal(basis[-1], np.zeros_like(basis[-1]))
    assert np.array_equal(basis[:, 0], np.zeros_like(basis[:, 0]))
    assert np.array_equal(basis[:, 2], np.zeros_like(basis[:, 2]))
    assert np.array_equal(basis[:, 3], np.zeros_like(basis[:, 3]))
    assert np.any(basis[:, 1])
    assert np.any(basis[:, 4])


def test_common_direction_selector_accepts_identical_nested_maps() -> None:
    rng = np.random.default_rng(20260726)
    fine = rng.normal(size=(8, 5, wp10c8y.PROFILE_COLUMNS))
    fine_measures = np.ones(8)
    medium = wp10c8y._restrict_basis(fine, fine_measures)
    medium_measures = np.ones(4) * 2.0
    coarse = wp10c8y._restrict_basis(medium, medium_measures)
    state_maps = {64: coarse, 128: medium, 256: fine}
    rate_maps = {
        family: {
            mesh: (index + 1.0) * state_maps[mesh]
            for mesh in wp10c8y.MESHES
        }
        for index, family in enumerate(wp10c8y.FAMILIES)
    }
    coefficients, report = wp10c8y._select_common_direction(
        state_maps=state_maps,
        rate_maps=rate_maps,
        measures={
            64: np.ones(2) * 4.0,
            128: medium_measures,
            256: fine_measures,
        },
        radii={
            64: np.asarray((2.5, 5.0)),
            128: np.linspace(2.0, 6.0, 4),
            256: np.linspace(1.9, 6.1, 8),
        },
        active_outer_rg=6.648,
    )
    assert coefficients.shape == (wp10c8y.PROFILE_COLUMNS,)
    assert report["selected_linear_passed"]
    assert report["passing_candidate_labels"]
    assert report["selected_linear_gate_score"] <= 1.0


def test_machine_evidence_keeps_architecture_blocked() -> None:
    if not wp10c8y.DEFAULT_OUTPUT.exists():
        return
    payload = json.loads(
        wp10c8y.DEFAULT_OUTPUT.read_text(encoding="utf-8")
    )
    assert payload["work_package"] == "WP10c8y"
    assert payload["classification"] == (
        "common_mode_passed_boundary_insensitive_underresolution"
    )
    assert payload["common_initial_profile_passed"]
    assert payload["selection"]["selected_linear_passed"]
    assert payload["selection"]["selected_template_cosine"] >= (
        wp10c8y.MINIMUM_TEMPLATE_COSINE
    )
    assert payload["passed_history_families"] == []
    for row in payload["pair_contracts"].values():
        assert row["passed"]
        assert row["maximum_pairwise_coordinate_defect"] <= (
            wp10c8y.MAXIMUM_PAIR_COORDINATE_DEFECT
        )
    for region in payload["initial_profile"]["binding_regions"]:
        for metric in payload["initial_profile"]["pairwise"][
            "N128_N256"
        ][region].values():
            assert wp10c8y._initial_metric_passed(metric)
    for family in wp10c8y.FAMILIES:
        result = payload["history_families"][family]
        assert result["available"]
        assert not result["passed"]
        assert result["state_observed_order"] < (
            wp10c8y.MINIMUM_HISTORY_SPATIAL_ORDER
        )
        assert result["rate_observed_order"] < (
            wp10c8y.MINIMUM_HISTORY_SPATIAL_ORDER
        )
    assert set(payload["mode_diagnosis"]) == set(wp10c8y.FAMILIES)
    assert payload["boundary_insensitive_underresolution"]
    assert payload["boundary_family_comparison"]["passed"]
    assert payload["decision"] == {
        "bounded_history_passed": False,
        "bounded_history_run": True,
        "common_equal_coordinate_lift_passed": True,
        "common_initial_profile_passed": True,
        "embedded_patch_preflight_authorized": True,
        "fixed_q_averaging_authorized": False,
        "n512_local_confirmation_authorized": False,
        "production_embedded_patch_authorized": False,
        "production_boundary_replacement_authorized": False,
    }
    assert payload["scope"]["formal_fast_average_certified"] is False
    assert payload["scope"]["reduced_architecture_selected"] is False
