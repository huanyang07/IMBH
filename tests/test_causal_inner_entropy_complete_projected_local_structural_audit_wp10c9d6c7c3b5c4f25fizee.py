from __future__ import annotations

import hashlib

import numpy as np

import run_causal_inner_entropy_complete_projected_local_structural_audit_wp10c9d6c7c3b5c4f25fizee as target
from imri_qpe.layer3_minidisk_1d.causal_inner_generalized_maxwell_cattaneo import (
    generalized_maxwell_cattaneo_principal,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_geometry import (
    kerr_schild_column_geometry,
)


def _principal():
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
        ],
        dtype=float,
    )
    return generalized_maxwell_cattaneo_principal(
        geometry,
        chart,
        proper_vertical_frequency=2.7491520839259703,
        alpha=0.1,
    )


def test_representative_point_passes_every_binding_gate() -> None:
    metrics, reasons = target._point_metrics(_principal(), alpha=0.1)
    assert reasons == ()
    assert metrics["maximum_imaginary_speed_over_c"] <= 1.0e-10
    assert metrics["maximum_light_cone_excess_over_c"] <= 1.0e-10
    assert metrics["reference_causality_minimum_margin"] >= 1.0e-8
    assert metrics["minimum_entropy_production_rate"] >= 0.0


def test_advective_eigenspace_has_exact_three_mode_cluster() -> None:
    basis, maximum_gap = target._advective_basis(_principal())
    assert basis.shape == (7, 3)
    np.testing.assert_allclose(
        basis.conj().T @ basis,
        np.eye(3),
        rtol=0.0,
        atol=1.0e-12,
    )
    assert maximum_gap <= 1.0e-7


def test_failure_classification_is_fail_closed() -> None:
    assert target._classification(()) == target.PASS_CLASSIFICATION
    assert target._classification(("ledger:entropy_sign",)) == (
        target.LEDGER_FAILURE
    )
    assert target._classification(("causality:light_cone",)) == (
        target.CAUSALITY_FAILURE
    )
    assert target._classification(
        ("strong_hyperbolicity:complex_speed",)
    ) == target.HYPERBOLICITY_FAILURE
    assert target._classification(("derivation:temporal_condition",)) == (
        target.DERIVATION_FAILURE
    )


def test_parent_authorization_and_frozen_envelope_close() -> None:
    validated = target._validate_parent(require_clean=False)
    assert validated["summary"]["authorized_next"] == target.WORK_PACKAGE
    assert validated["contract"]["binding_gates"]["fail_closed"]
    assert "audit_envelope.npz" in validated["envelope_hashes"]


def test_canonical_result_closes_if_present() -> None:
    directory = target.CANONICAL_DIRECTORY
    if not directory.exists():
        return
    for line in (directory / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        expected, name = line.split("  ", 1)
        actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        assert actual == expected
    summary = target.parent.parent._read_json(directory / "summary.json")
    assert summary["classification"] in {
        target.PASS_CLASSIFICATION,
        target.CAUSALITY_FAILURE,
        target.HYPERBOLICITY_FAILURE,
        target.LEDGER_FAILURE,
        target.DERIVATION_FAILURE,
    }
    assert summary["new_trajectory_steps"] == 0
    assert not summary["seven_field_trajectory_authorized"]
    assert not summary["complete_cycle_execution_authorized"]
