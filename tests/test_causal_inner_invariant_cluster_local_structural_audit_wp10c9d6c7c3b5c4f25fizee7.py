from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

import run_causal_inner_invariant_cluster_local_structural_audit_wp10c9d6c7c3b5c4f25fizee7 as target
from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo import (
    generalized_maxwell_cattaneo_principal,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_geometry import (
    kerr_schild_column_geometry,
)


def _fixture_principal():
    geometry = kerr_schild_column_geometry(5.599841633135499e9, 1.48e9)
    chart = np.asarray(
        [
            4.74082887,
            -0.330628060,
            0.662598339,
            14.9471713,
            2.13041458e-4,
            20.1048472,
            0.0,
        ]
    )
    return generalized_maxwell_cattaneo_principal(
        geometry,
        chart,
        proper_vertical_frequency=2.7491520839259703,
        alpha=0.1,
    )


def test_parent_correction_and_envelope_are_hash_locked() -> None:
    validated = target._validate_parent(require_clean=False)
    assert validated["summary"]["classification"] == target.parent.CLASSIFICATION
    assert validated["summary"]["corrected_full_local_audit_authorized"]
    assert validated["envelope_hashes"]


def test_representative_advective_cluster_passes_invariant_gates() -> None:
    metrics, reasons, selected = target._cluster_metrics(_fixture_principal())
    assert len(selected) == 3
    assert not reasons
    assert metrics["maximum_advective_cluster_transport_offset_over_c"] <= 1.0e-6
    assert metrics["minimum_advective_cluster_complement_gap_over_c"] >= 1.0e-4


def test_classification_is_fail_closed() -> None:
    assert target._classify(()) == target.PASS_CLASSIFICATION
    assert target._classify(("causality:light_cone",)) == target.CAUSALITY_FAILURE
    assert target._classify(("strong_hyperbolicity:advective_cluster_gap",)) == target.HYPERBOLICITY_FAILURE
    assert target._classify(("ledger:entropy",)) == target.LEDGER_FAILURE
    assert target._classify(("derivation:ladder",)) == target.DERIVATION_FAILURE


def test_canonical_package_closes_if_present() -> None:
    directory = target.CANONICAL_DIRECTORY
    if not directory.exists():
        return
    for line in (directory / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        assert actual == expected
    summary = target._utils()._read_json(directory / "summary.json")
    metrics = target._utils()._read_json(directory / "audit_metrics.json")
    assert summary["classification"] in {
        target.PASS_CLASSIFICATION,
        target.CAUSALITY_FAILURE,
        target.HYPERBOLICITY_FAILURE,
        target.LEDGER_FAILURE,
        target.DERIVATION_FAILURE,
    }
    assert summary["all_parent_results_preserved"]
    assert summary["new_trajectory_steps"] == 0
    if summary["passed"]:
        assert summary["complete_reduced_principal_certified"]
        assert summary["advective_cluster_certified"]
        assert metrics["base_points_audited"] == metrics["base_points_planned"]
        assert metrics["first_failure"] is None
        assert summary["authorized_next"] == target.AUTHORIZED_NEXT_ON_PASS
    else:
        assert not summary["complete_reduced_principal_certified"]
        assert metrics["first_failure"] is not None
        assert summary["authorized_next"] is None


def test_declared_paths_are_workspace_relative() -> None:
    for relative in (
        target.THIS_RUNNER,
        target.THIS_TEST,
        target.FROZEN_AUDIT_RUNNER,
        target.PHYSICAL_SOURCE,
        target.PHYSICAL_TEST,
        target.REPORT_RELATIVE,
    ):
        assert not Path(relative).is_absolute()
