from __future__ import annotations

import hashlib

import run_causal_inner_entropy_complete_fixed_q_invariant_object_manifest_wp10c9d6c7c3b5c4f25fizep as target


def test_corrected_crossing_is_locked() -> None:
    validated = target._validate_parent(require_clean=False)
    assert validated["summary"]["classification"] == target.parent.PASS_CLASSIFICATION
    assert validated["summary"]["crossed_old_rejected_time"]
    assert validated["metrics"]["first_failure"] is None


def test_split_is_exact_and_not_legacy_global_fixed_q() -> None:
    split = target._contract()["mathematical_split"]
    assert split["slow_exact_rows"] == [0, 2, 3]
    assert split["fast_equation_rows"] == [1, 4, 5, 6]
    assert split["slow_dimension"] == 336
    assert split["fast_dimension"] == 448
    assert split["identity"] == "784_equals_336_plus_448"
    assert split["fixed_Q_means_cellwise_exact_M_J_E_not_only_three_global_ledgers"]
    assert not split["legacy_global_fixed_Q_reaction_used"]


def test_fast_vector_field_and_coloring_are_frozen() -> None:
    contract = target._contract()
    assert contract["fast_vector_field"]["constraints"] == "D_Q_times_primitive_rate_equals_zero"
    assert contract["sparse_derivative"]["cell_coloring_count"] == 3
    assert contract["sparse_derivative"]["fast_field_count"] == 4
    assert contract["sparse_derivative"]["forward_colored_residual_evaluations_per_assembly"] == 12


def test_claim_boundary_stops_before_roots() -> None:
    claim = target._contract()["claim_boundary"]
    assert claim["implementation_authorized"]
    assert not claim["nonlinear_root_execution_authorized"]
    assert not claim["slow_flux_atlas_authorized"]
    assert not claim["complete_cycle_execution_authorized"]


def test_canonical_package_closes_if_present() -> None:
    directory = target.CANONICAL_DIRECTORY
    if not directory.exists(): return
    for line in (directory / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected
    summary = target._utils()._read_json(directory / "summary.json")
    assert summary["definitions_only"]
    assert summary["cellwise_fixed_Q_split_frozen"]
    assert not summary["nonlinear_root_execution_authorized"]
