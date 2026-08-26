from __future__ import annotations

import hashlib

import run_causal_inner_entropy_complete_path_conservative_interface_audit_wp10c9d6c7c3b5c4f25fizeg as target


def test_spatial_manifest_is_hash_locked_and_authorizes_only_interfaces() -> None:
    validated = target._validate_parent(require_clean=False)
    assert validated["summary"]["classification"] == target.parent.CLASSIFICATION
    assert validated["summary"]["nonpropagating_interface_audit_authorized"]
    assert not validated["summary"]["semidiscrete_cell_operator_authorized"]
    assert validated["local_summary"]["complete_reduced_principal_certified"]


def test_audit_ladders_and_exact_rows_are_frozen() -> None:
    assert target.QUADRATURE_ORDERS == (4, 8, 16)
    assert target.SMOOTH_AMPLITUDES == (1.0e-3, 5.0e-4, 2.5e-4)
    assert tuple(target.EXACT_ROWS) == (0, 1, 2, 3, 5, 6)


def test_failure_classification_is_fail_closed() -> None:
    assert target._classification([]) == target.PASS_CLASSIFICATION
    assert target._classification(["flux:shared"]) == target.FLUX_FAILURE
    assert target._classification(["split:closure"]) == target.SPLIT_FAILURE
    assert (
        target._classification(["hyperbolicity:imaginary_speed"])
        == target.HYPERBOLICITY_FAILURE
    )


def test_canonical_package_closes_if_present() -> None:
    directory = target.CANONICAL_DIRECTORY
    if not directory.exists():
        return
    for line in (directory / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        assert actual == expected
    summary = target._utils()._read_json(directory / "summary.json")
    assert summary["classification"] in (
        target.PASS_CLASSIFICATION,
        target.FLUX_FAILURE,
        target.SPLIT_FAILURE,
        target.HYPERBOLICITY_FAILURE,
    )
    assert summary["isolated_interfaces_only"]
    assert summary["new_trajectory_steps"] == 0
    assert not summary["semidiscrete_cell_operator_authorized"]
    assert not summary["seven_field_trajectory_authorized"]
