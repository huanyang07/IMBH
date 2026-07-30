from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from imri_qpe.layer3_minidisk_1d.causal_inner_packet_manifest import (
    causal_canonical_json_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = (
    ROOT
    / "results/canonical/"
    "causal_inner_bandlimited_balance_feasibility_wp10c9d6c6e2b"
)
SUMMARY = CANONICAL / "summary.json"
DECISIVE = CANONICAL / "decisive_arrays.npz"
SELECTED = CANONICAL / "selected_profile_manifest.json"


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


def test_wp10c9d6c6e2b_preserves_search_and_no_propagation() -> None:
    summary = _summary()
    assert summary["search_manifest_sha256"] == (
        "7dba2fd9db6cc093eff8a3307dfe851036fee3fbb243a5100b1f0819d4b44c02"
    )
    assert summary["evaluation_executed"]
    assert not summary["propagation_executed"]
    assert not summary["operator_changed"]
    assert summary["parent_classification_preserved"]
    assert summary["c6c_c6d_c6e0_c6e1_status_preserved"]


def test_wp10c9d6c6e2b_selected_manifest_hashes() -> None:
    summary = _summary()
    selected = json.loads(SELECTED.read_text(encoding="utf-8"))
    stored = selected.pop("selected_profile_manifest_sha256")
    assert causal_canonical_json_sha256(selected) == stored
    assert summary["selected_profile_manifest_sha256"] == stored


def test_wp10c9d6c6e2b_finds_no_eligible_candidate() -> None:
    summary = _summary()
    assert summary["classification"] == (
        "no_eligible_bandlimited_balance_profile"
    )
    report = summary["feasibility_report"]
    assert not report["eligible_candidates"]
    assert report["selected_candidate"] is None
    assert not report["passed"]
    assert not any(
        item["passed"]
        for item in report["candidate_pair_reports"].values()
    )
    closest = report["candidate_pair_reports"]["p2_cos1"]
    assert closest["maximum_theta_99"] <= 0.30
    assert closest["maximum_alias_fraction"] > 1.0e-3


def test_wp10c9d6c6e2b_never_authorizes_downstream_physics() -> None:
    summary = _summary()
    assert not summary["embedded_export_discrimination_authorized"]
    assert not summary["nonlinear_physical_trajectory_authorized"]
    assert not summary["production_operator_authorized"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]


def test_wp10c9d6c6e2b_canonical_hashes() -> None:
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
