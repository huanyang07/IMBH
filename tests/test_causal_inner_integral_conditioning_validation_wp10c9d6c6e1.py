from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = (
    ROOT
    / "results/canonical/"
    "causal_inner_integral_conditioning_validation_wp10c9d6c6e1"
)
SUMMARY = CANONICAL / "summary.json"
DECISIVE = CANONICAL / "decisive_arrays.npz"


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


def _summary() -> dict:
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


def test_wp10c9d6c6e1_preserves_manifest_and_historical_status() -> None:
    summary = _summary()
    assert summary["manifest_sha256"] == (
        "7eee9c710df8ee48418e0e54007d2f5a02360c07f42af2a750df5d15b3cc9f92"
    )
    assert summary["parent_classification_preserved"]
    assert summary["c6c_rejection_preserved"]
    assert not summary["operator_changed"]


def test_wp10c9d6c6e1_respects_eligibility_stop() -> None:
    summary = _summary()
    eligibility = summary["eligibility_report"]
    if not eligibility["passed"]:
        assert not summary["propagation_executed"]
        assert summary["classification"] == (
            "frozen_integral_profiles_ineligible"
        )
    else:
        assert summary["propagation_executed"]
        assert summary["method_passed"]
        assert (
            summary["maximum_exact_integral_relative_solve_residual"]
            <= 1.0e-12
        )


def test_wp10c9d6c6e1_rejects_only_the_frozen_balanced_spectra() -> None:
    summary = _summary()
    assert summary["classification"] == "frozen_integral_profiles_ineligible"
    assert not summary["propagation_executed"]
    profiles = summary["eligibility_report"]["profile_reports"]
    failed = {name for name, report in profiles.items() if not report["passed"]}
    assert failed == {
        "balanced_p2_p4__inward_shear",
        "balanced_p2_p4__outward_shear",
    }
    for name in failed:
        report = profiles[name]
        assert not report["spectral_passed"]
        assert report["purity_passed"]
        assert report["projection_passed"]
        assert report["endpoint_passed"]
        assert report["balance_passed"]
    assert all(
        report["passed"]
        for name, report in profiles.items()
        if name not in failed
    )


def test_wp10c9d6c6e1_never_authorizes_downstream_physics() -> None:
    summary = _summary()
    assert not summary["nonlinear_physical_trajectory_authorized"]
    assert not summary["production_operator_authorized"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]


def test_wp10c9d6c6e1_canonical_hashes() -> None:
    summary = _summary()
    assert _sha256(DECISIVE) == summary["decisive_arrays_sha256"]
    for relative, expected in summary[
        "implementation_source_hashes"
    ].items():
        assert _sha256(ROOT / relative) == expected
    with np.load(DECISIVE, allow_pickle=False) as source:
        assert set(source.files) == set(summary["decisive_array_hashes"])
        for name in source.files:
            assert (
                _array_sha256(source[name])
                == summary["decisive_array_hashes"][name]
            )
