from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_causal_inner_reaction_free_field_architecture_diagnosis_wp10c9d6c7c3b5c4f25f6 as target


def test_committed_rate_and_full_model_secant_inputs_are_locked() -> None:
    locked = target._validate_inputs(require_clean=False)
    assert set(locked["input_hashes"]) == {
        "cold_anchor",
        "hybrid_architecture",
        "primary_anchor",
        "arclength_segment",
    }


def test_free_field_not_fixed_q_reaction_matches_physical_subspace() -> None:
    metrics, arrays, architecture = target._evaluate(
        target._validate_inputs(require_clean=False)
    )
    assert metrics["passed"]
    assert all(metrics["gates"].values())
    assert metrics["gates"]["free_rate_matches_physical_subspace"]
    assert metrics["gates"]["fixed_Q_rate_rejected_by_physical_subspace"]
    assert metrics["gates"]["reaction_aligned_with_fixed_Q_tangent"]
    assert arrays["free_coordinate_rates4x470_per_s"].shape == (4, 470)
    assert architecture["fixed_Q_interpretation"]["physical_phase_clock_rejected"]


def test_rate_split_is_additive_at_every_anchor() -> None:
    metrics, arrays, _architecture = target._evaluate(
        target._validate_inputs(require_clean=False)
    )
    np.testing.assert_allclose(
        arrays["free_coordinate_rates4x470_per_s"]
        + arrays["reaction_coordinate_actions4x470_per_s"],
        arrays["fixed_Q_coordinate_rates4x470_per_s"],
        rtol=target.MAXIMUM_ADDITIVE_CLOSURE_DEFECT,
        atol=1.0e-12,
    )
    assert metrics["gate_values"]["free_direction_rank_two_energy"] >= 0.9999


def test_canonical_package_when_present() -> None:
    if not target.CANONICAL_DIRECTORY.exists():
        return
    helper = target._helper()
    helper._validate_checksums(target.CANONICAL_DIRECTORY)
    summary = helper._read(target.CANONICAL_DIRECTORY / "summary.json")
    assert summary["passed"]
    assert not summary["fixed_Q_physical_phase_authorized"]
    assert summary["conservative_free_field_hidden_amplitude_rom_selected"]
    assert summary["authorized_next"] == target.AUTHORIZED_NEXT
