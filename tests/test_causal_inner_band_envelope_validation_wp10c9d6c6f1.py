from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = (
    ROOT
    / "results/canonical/"
    "causal_inner_band_envelope_validation_wp10c9d6c6f1"
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


def test_wp10c9d6c6f1_preserves_manifest_and_operator() -> None:
    summary = _summary()
    assert summary["manifest_sha256"] == (
        "221a271dd861226bbc09eaf430dfc6bef47ad39a5b5d7e6e53520f9d75fcb643"
    )
    assert summary["parent_classification_preserved"]
    assert summary["historical_classifications_preserved"]
    assert summary["c6c_rejection_preserved"]
    assert not summary["operator_changed"]
    assert summary["propagation_executed"]
    assert summary["projection_replay_report"]["passed"]


def test_wp10c9d6c6f1_all_variants_receive_one_frozen_route() -> None:
    summary = _summary()
    decision = summary["prospective_decision"]
    assert len(decision["variant_reports"]) == 20
    assert (
        decision["direct_variant_count"]
        + decision["alternate_variant_count"]
        + len(decision["failed_variants"])
        == 20
    )
    assert all(
        item["route"]
        in {
            "historical_direct_contract",
            "proof_style_cancellation_conditioned_band_envelope",
            "failed",
        }
        for item in decision["variant_reports"].values()
    )


def test_wp10c9d6c6f1_certifies_every_variant_directly() -> None:
    summary = _summary()
    assert (
        summary["classification"]
        == "prospective_band_envelope_uniform_validation_certified"
    )
    assert (
        summary["authorized_next"]
        == "WP10c9d6c7_embedded_discrimination"
    )
    assert summary["passed"]
    decision = summary["prospective_decision"]
    assert decision["all_variants_passed"]
    assert decision["direct_variant_count"] == 20
    assert decision["alternate_variant_count"] == 0
    assert not decision["alternate_base_profiles"]
    assert not decision["failed_variants"]
    assert not summary["historical_direct_contract_report"]["failed_packets"]


def test_wp10c9d6c6f1_downstream_authorization_is_consistent() -> None:
    summary = _summary()
    assert (
        summary["embedded_export_discrimination_authorized"]
        == summary["passed"]
    )
    assert (
        summary["uniform_profile_class_certified"]
        == summary["passed"]
    )
    assert not summary["nonlinear_physical_trajectory_authorized"]
    assert not summary["production_operator_authorized"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]


def test_wp10c9d6c6f1_canonical_hashes() -> None:
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
