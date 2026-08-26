from __future__ import annotations

import hashlib

import run_causal_inner_entropy_complete_thermodynamic_chart_atlas_manifest_wp10c9d6c7c3b5c4f25fizfh as target


def test_raw_coordinate_rejection_is_preserved() -> None:
    validated = target._validate_parent(require_clean=False)
    assert validated["metrics"]["failure"]["stage"] == "first_colored_raw_M_coordinate_plus_lift"
    assert validated["metrics"]["new_truth_operator_calls"] == 0


def test_chart_recovery_keeps_exact_macro_storage() -> None:
    contract = target._contract()
    redesign = contract["coordinate_redesign"]
    assert redesign["stored_online_state"] == "exact_16_cell_(M,J,E,beta_r,chi)"
    assert redesign["exact_MJE_storage_and_face_flux_conservation_unchanged"]
    assert contract["atlas_audit"]["maximum_new_truth_operator_calls"] == 38
    assert not contract["claim_boundary"]["state_propagation_authorized"]


def test_canonical_package_closes_if_present() -> None:
    directory = target.CANONICAL_DIRECTORY
    if not directory.exists(): return
    for line in (directory / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1); assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected
    summary = target._utils()._read_json(directory / "summary.json")
    assert summary["definitions_only"] and summary["raw_coordinate_atlas_rejection_preserved"]
    assert not summary["state_propagation_authorized"]
