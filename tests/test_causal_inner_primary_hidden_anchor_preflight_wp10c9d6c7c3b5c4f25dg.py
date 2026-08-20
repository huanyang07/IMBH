from __future__ import annotations

import numpy as np

import run_causal_inner_primary_hidden_anchor_preflight_wp10c9d6c7c3b5c4f25dg as f25dg


def test_manifest_authorizes_the_fail_fast_anchor_execution() -> None:
    frozen = f25dg._validate_manifest(require_clean=False)
    assert frozen["manifest_hashes"]
    stages = frozen["contract"]["prospective_execution"]["stage_order"]
    assert stages[1] == "evaluate_one_fresh_exact_fixed_Q_rate_at_the_exact_20ms_anchor"
    assert "stop_without_generator_or_root" in stages[3]


def test_dual_consistent_rate_decomposition_is_exact() -> None:
    geometry = f25dg._geometry()
    rng = np.random.default_rng(26082026)
    coordinate_rate = rng.standard_normal(f25dg.manifest.COORDINATE_DIMENSION)
    result = f25dg._decompose_rate(coordinate_rate, geometry)
    assert result["decomposition_relative_defect"] <= f25dg.manifest.DUAL_GEOMETRY_GATE
    assert result["hidden_projection_relative_defect"] <= f25dg.manifest.DUAL_GEOMETRY_GATE
    assert np.allclose(result["H"], geometry["Q"] @ coordinate_rate)
    assert not np.allclose(result["H"], geometry["Z"].T @ coordinate_rate)


def test_fail_fast_threshold_and_budgets_are_unchanged() -> None:
    assert f25dg.HIDDEN_FRACTION_GATE == 0.25
    assert f25dg.PASS_AUTHORIZED_NEXT == "WP10c9d6c7c3b5c4f25dh"


def test_canonical_evidence_if_present() -> None:
    if not f25dg.CANONICAL_DIRECTORY.exists():
        return
    f25dg._checksums(f25dg.CANONICAL_DIRECTORY)
    summary = f25dg._read(f25dg.CANONICAL_DIRECTORY / "summary.json")
    payload = f25dg._read(
        f25dg.CANONICAL_DIRECTORY / "primary_anchor_rate_metrics.json"
    )
    assert summary["new_exact_fixed_Q_rate_evaluations"] == 1
    assert summary["new_complete_generator_assemblies"] == 0
    assert summary["new_intrinsic_hidden_roots"] == 0
    assert not summary["hidden_root_attempted"]
    assert not summary["sealed_16ms_opened"]
    assert all(payload["checks"].values())
    assert payload["hidden_fraction_gate_passed"] == summary[
        "anchor_hidden_fraction_gate_passed"
    ]
