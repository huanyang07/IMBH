from __future__ import annotations

import hashlib
import json
from pathlib import Path

from imri_qpe.layer3_minidisk_1d.causal_inner_packet_manifest import (
    causal_canonical_json_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = (
    ROOT
    / "results/canonical/"
    "causal_inner_integral_conditioning_manifest_wp10c9d6c6e0"
)
SUMMARY = CANONICAL / "summary.json"
MANIFEST = CANONICAL / "conditioning_manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _summary() -> dict:
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_wp10c9d6c6e0_freezes_profiles_before_propagation() -> None:
    summary = _summary()
    manifest = _manifest()
    assert summary["base_profile_count"] == 7
    assert summary["profile_variant_count"] == 28
    assert summary["cancellation_stress_profile_count"] == 2
    assert not summary["propagation_executed"]
    assert not summary["eligibility_evaluated"]
    assert not manifest["propagation_executed"]
    assert not manifest["eligibility_evaluated"]


def test_wp10c9d6c6e0_manifest_hash_and_contract() -> None:
    summary = _summary()
    manifest = _manifest()
    stored_hash = manifest.pop("manifest_sha256")
    assert causal_canonical_json_sha256(manifest) == stored_hash
    assert summary["manifest_sha256"] == stored_hash
    contract = manifest["integral_conditioning_contract"]
    assert contract["historical_direct_component_route_preserved"]
    assert contract["maximum_absolute_band_error_envelope"] == 0.05
    assert (
        contract["maximum_cancellation_ratio_each_grid_pair"]
        == 0.25
    )
    assert contract["no_retroactive_application_to_wp10c9d6c6c"]


def test_wp10c9d6c6e0_authorizes_only_frozen_eligibility_run() -> None:
    summary = _summary()
    assert summary["classification"] == (
        "integral_conditioning_contract_and_profiles_frozen_"
        "eligibility_audit_authorized"
    )
    assert summary["parent_classification_preserved"]
    assert summary["c6c_rejection_preserved"]
    assert not summary["embedded_export_discrimination_authorized"]
    assert not summary["nonlinear_physical_trajectory_authorized"]
    assert not summary["production_operator_authorized"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]


def test_wp10c9d6c6e0_source_hashes() -> None:
    summary = _summary()
    for relative, expected in summary[
        "implementation_source_hashes"
    ].items():
        assert _sha256(ROOT / relative) == expected
