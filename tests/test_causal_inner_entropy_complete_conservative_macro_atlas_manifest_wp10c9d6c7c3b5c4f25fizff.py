from __future__ import annotations

import hashlib

import run_causal_inner_entropy_complete_conservative_macro_atlas_manifest_wp10c9d6c7c3b5c4f25fizff as target


def test_inverse_certificate_authorizes_only_atlas_manifest() -> None:
    validated = target._validate_parent(require_clean=False)
    assert validated["summary"]["hydrostatic_implicit_inverse_tangent_certified"]
    assert validated["summary"]["local_slow_flux_atlas_manifest_authorized"]
    assert not validated["summary"]["complete_cycle_execution_authorized"]


def test_macro_architecture_is_conservative_and_cost_bounded() -> None:
    contract = target._contract()
    architecture = contract["selected_architecture"]
    atlas = contract["atlas_construction"]
    assert architecture["truth_radial_cells"] == 7 * architecture["online_radial_cells"]
    assert architecture["online_state_dimension"] == 80
    assert atlas["colored_truth_calls"] + atlas["independent_JVP_truth_calls"] == 38
    assert len(atlas["strict_blind_profiles"]) == 2
    assert contract["online_cost"]["truth_calls_per_macrostep"] == 0
    assert not contract["claim_boundary"]["state_propagation_authorized"]


def test_canonical_package_closes_if_present() -> None:
    directory = target.CANONICAL_DIRECTORY
    if not directory.exists(): return
    for line in (directory / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected
    summary = target._utils()._read_json(directory / "summary.json")
    assert summary["definitions_only"]
    assert summary["conservative_16_cell_macro_atlas_selected"]
    assert not summary["state_propagation_authorized"]
