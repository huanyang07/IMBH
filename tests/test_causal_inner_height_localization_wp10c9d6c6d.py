from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = (
    ROOT
    / "results/canonical/"
    "causal_inner_height_localization_wp10c9d6c6d"
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


def test_wp10c9d6c6d_preserves_scope_and_parent_classification() -> None:
    summary = _summary()
    assert summary["work_package"] == "WP10c9d6c6d"
    assert not summary["operator_changed"]
    assert summary["parent_classification_preserved"]
    assert summary["c6c_failure_preserved"]
    assert not summary["embedded_export_discrimination_authorized"]
    assert not summary["nonlinear_physical_trajectory_authorized"]
    assert not summary["production_operator_authorized"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]


def test_wp10c9d6c6d_direct_and_continuum_gates() -> None:
    summary = _summary()
    assert summary["method_passed"]
    assert summary["direct_cell_sum_parity_defect"] <= 1.0e-12
    assert summary["maximum_parent_history_replay_defect"] <= 1.0e-10
    assert summary["continuum_reference_report"]["passed"]


def test_wp10c9d6c6d_selects_cancellation_not_an_operator_change() -> None:
    summary = _summary()
    assert summary["classification"] == (
        "convergent_bands_noncontracting_cancellation_remainder"
    )
    assert summary["authorized_next"] == (
        "prospective_integral_conditioning_audit"
    )
    mechanism = summary["mechanism_selection"]
    assert mechanism["all_failed_profile_bands_converge"]
    assert not mechanism["common_failed_bands"]
    assert not mechanism["selected_bands"]
    assert not summary["targeted_operator_intervention_authorized"]


def test_wp10c9d6c6d_canonical_hashes() -> None:
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
