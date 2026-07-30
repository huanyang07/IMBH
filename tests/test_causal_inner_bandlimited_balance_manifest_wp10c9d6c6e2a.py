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
    "causal_inner_bandlimited_balance_manifest_wp10c9d6c6e2a"
)
SUMMARY = CANONICAL / "summary.json"
MANIFEST = CANONICAL / "search_manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _summary() -> dict:
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


def test_wp10c9d6c6e2a_freezes_search_before_evaluation() -> None:
    summary = _summary()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    stored = manifest.pop("manifest_sha256")
    assert causal_canonical_json_sha256(manifest) == stored
    assert summary["manifest_sha256"] == stored
    assert summary["candidate_count"] == 6
    assert not summary["evaluation_executed"]
    assert not summary["propagation_executed"]
    assert not manifest["evaluation_executed"]
    assert not manifest["propagation_executed"]


def test_wp10c9d6c6e2a_forbids_history_selection_and_threshold_changes() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    contract = manifest["search_contract"]
    assert contract["propagated_history_objective_forbidden"]
    assert contract["maximum_theta_99"] == 0.30
    assert contract["maximum_nyquist_alias_fraction"] == 1.0e-3
    assert contract["pair_must_pass_both_shear_families"]
    assert contract["selection_is_lexicographic_ascending"]


def test_wp10c9d6c6e2a_authorizes_only_feasibility() -> None:
    summary = _summary()
    assert summary["classification"] == (
        "bandlimited_conditioning_search_frozen_"
        "feasibility_authorized"
    )
    assert summary["parent_classification_preserved"]
    assert summary["c6c_c6d_c6e0_c6e1_status_preserved"]
    assert not summary["embedded_export_discrimination_authorized"]
    assert not summary["nonlinear_physical_trajectory_authorized"]
    assert not summary["production_operator_authorized"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]


def test_wp10c9d6c6e2a_source_hashes() -> None:
    summary = _summary()
    for relative, expected in summary[
        "implementation_source_hashes"
    ].items():
        assert _sha256(ROOT / relative) == expected
