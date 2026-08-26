from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

import run_causal_inner_advective_subspace_path_continuity_certificate_wp10c9d6c7c3b5c4f25fizee5 as target


def test_parent_manifest_and_physical_source_are_hash_locked() -> None:
    validated = target._validate_parent(require_clean=False)
    assert validated["summary"]["classification"] == target.parent.CLASSIFICATION
    assert validated["summary"]["parent_negative_result_preserved"]


def test_resolution_metric_detects_smooth_and_discontinuous_paths() -> None:
    angles = np.linspace(0.0, 0.3, 9)
    bases = np.empty((len(angles), 7, 3))
    for index, angle in enumerate(angles):
        basis = np.zeros((7, 3))
        basis[0, 0] = np.cos(angle)
        basis[3, 0] = np.sin(angle)
        basis[1, 1] = 1.0
        basis[2, 2] = 1.0
        bases[index] = basis
    projectors = np.asarray([basis @ basis.T for basis in bases])
    smooth = target._resolution_metrics(bases, projectors)
    assert smooth["minimum_adjacent_subspace_cosine"] > 0.99
    broken = np.array(bases, copy=True)
    broken[4, :, 0] = 0.0
    broken[4, 6, 0] = 1.0
    broken_projectors = np.asarray([basis @ basis.T for basis in broken])
    discontinuous = target._resolution_metrics(broken, broken_projectors)
    assert discontinuous["minimum_adjacent_subspace_cosine"] < 0.99


def test_canonical_package_closes_if_present() -> None:
    directory = target.CANONICAL_DIRECTORY
    if not directory.exists():
        return
    for line in (directory / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        assert actual == expected
    summary = target._utils()._read_json(directory / "summary.json")
    metrics = target._utils()._read_json(directory / "path_metrics.json")
    assert summary["classification"] in {
        target.PASS_CLASSIFICATION,
        target.FAIL_CLASSIFICATION,
    }
    assert summary["parent_negative_result_preserved"]
    assert summary["new_trajectory_steps"] == 0
    if summary["passed"]:
        assert summary["smooth_uniformly_bounded_advective_subspace_on_saved_path"]
        assert summary["authorized_next"] == target.AUTHORIZED_NEXT_ON_PASS
        assert metrics["minimum_cluster_complement_gap_over_c"] >= 1.0e-4
        assert max(metrics["maximum_projector_jump_refinement_ratios"]) <= 0.60
    else:
        assert summary["authorized_next"] is None


def test_declared_paths_are_workspace_relative() -> None:
    for relative in (
        target.THIS_RUNNER,
        target.THIS_TEST,
        target.PHYSICAL_SOURCE,
        target.PHYSICAL_TEST,
        target.REPORT_RELATIVE,
    ):
        assert not Path(relative).is_absolute()
