from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    ROOT / "scripts/run_causal_inner_extended_localization_wp10c9d5c1.py"
)
CANONICAL = (
    ROOT
    / "results/canonical/causal_inner_extended_localization_wp10c9d5c1"
)
SUMMARY = CANONICAL / "summary.json"
DECISIVE = CANONICAL / "decisive_arrays.npz"

SPEC = importlib.util.spec_from_file_location("wp10c9d5c1_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def test_wp10c9d5c1_contract_is_non_tautological() -> None:
    assert RUNNER.ANALYZED_BASE_COMMIT == (
        "f409244f0f9b487b918d4e93f49e8bcf41049af1"
    )
    assert "outer_shared_face" not in RUNNER.EXPLANATORY_TERMS
    assert all(
        "outer_shared_face" not in names
        for names in RUNNER.GROUPS.values()
    )
    grouped = tuple(
        name
        for names in RUNNER.GROUPS.values()
        for name in names
    )
    assert set(grouped) == set(RUNNER.EXPLANATORY_TERMS)
    assert len(grouped) == len(set(grouped))
    assert RUNNER.MINIMUM_RECOVERY_ORDER == 0.75
    assert RUNNER.MINIMUM_ERROR_COSINE == 0.90
    assert RUNNER.REQUIRED_CONSECUTIVE_SURFACES == 2


def test_pair_attribution_preserves_signed_fixed_coefficients() -> None:
    target = np.asarray([[1.0, -2.0, 0.5], [0.5, -1.0, 0.25]])
    terms = {
        name: np.zeros_like(target)
        for name in RUNNER.EXPLANATORY_TERMS
    }
    terms["inner_shared_face"] = -target
    report, gram = RUNNER._pair_attribution(
        target,
        terms,
        np.ones(3),
    )
    inner = report["groups"]["inner_boundary"]
    assert np.isclose(inner["target_aligned_fraction"], 1.0)
    assert np.isclose(inner["fixed_coefficient_residual"], 0.0)
    assert report["complete_explanatory_closure_defect"] == 0.0
    assert report["complete_explanatory_relative_closure_defect"] == 0.0
    assert gram.shape == (
        len(RUNNER.EXPLANATORY_TERMS),
        len(RUNNER.EXPLANATORY_TERMS),
    )


def test_inactive_target_uses_fixed_physical_closure() -> None:
    target = np.full((4, 3), 1.0e-20)
    terms = {
        name: np.zeros_like(target)
        for name in RUNNER.EXPLANATORY_TERMS
    }
    terms["inner_shared_face"] = -target
    terms["geometry"] = np.full_like(target, 2.0e-16)
    terms["cooling"] = -terms["geometry"]
    report, _gram = RUNNER._pair_attribution(
        target,
        terms,
        np.ones(3),
    )
    assert not report["active"]
    assert report["complete_explanatory_closure_defect"] < 1.0e-30
    assert report["complete_explanatory_relative_closure_defect"] is None


def test_wp10c9d5c1_canonical_evidence_is_self_consistent() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert summary["work_package"] == "WP10c9d5c1"
    assert summary["method_passed"]
    assert summary["parent_wp10c9d5_candidate_remains_rejected"]
    assert summary["analytic_tangent_physical_sensitivity_remains_passed"]
    assert summary["attribution_target"].startswith("direct outer-face")
    assert not summary["production_operator_authorized"]
    assert not summary["nonlinear_candidate_authorized"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]
    assert _sha256(DECISIVE) == summary["decisive_arrays_sha256"]
    with np.load(DECISIVE, allow_pickle=False) as source:
        assert set(source.files) == set(summary["decisive_array_hashes"])
        for name in source.files:
            assert (
                _array_sha256(source[name])
                == summary["decisive_array_hashes"][name]
            )
    branches = {
        "A_recovery_before_coupling",
        "B_stable_boundary_contribution",
        "C_stable_storage_anchor_contribution",
        "D_no_recovery_or_stable_non_target_mechanism",
        "E_stable_principal_or_lower_source_contribution",
    }
    assert summary["classification"] in branches
    authorizations = (
        summary["conservative_extraction_surface_audit_authorized"],
        summary["boundary_half_cell_audit_authorized"],
        summary["self_consistent_tangent_audit_authorized"],
        summary["targeted_source_path_audit_authorized"],
        summary["monolithic_replacement_authorized"],
    )
    assert sum(bool(value) for value in authorizations) == 1
