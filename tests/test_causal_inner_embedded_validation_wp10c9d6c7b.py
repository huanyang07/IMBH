from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = (
    ROOT
    / "results/canonical/"
    "causal_inner_embedded_validation_wp10c9d6c7b"
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


def test_wp10c9d6c7b_preserves_manifest_and_operator() -> None:
    summary = _summary()
    assert summary["manifest_sha256"] == (
        "c465f284dd2991fa0241b2bb268fc723a89bc111bedd59c3cf5a5830346e554a"
    )
    assert summary["parent_classification_preserved"]
    assert summary["uniform_certification_preserved"]
    assert summary["historical_classifications_preserved"]
    assert not summary["operator_changed"]
    assert summary["propagation_executed"]


def test_wp10c9d6c7b_method_and_ledger_gates_pass() -> None:
    summary = _summary()
    assert summary["method_passed"]
    assert all(
        item["passed"] for item in summary["method_reports"].values()
    )
    assert (
        summary["maximum_exact_integral_relative_solve_residual"]
        <= 1.0e-12
    )
    for report in summary["method_reports"].values():
        assert report["maximum_active_directional_export_defect"] <= 2.0e-6
        assert report["conservative_transport_telescoping_defect"] <= 1.0e-12
        assert report["active_prefix_ledger_defect"] <= 1.0e-12


def test_wp10c9d6c7b_rejects_only_the_p3_shear_variants() -> None:
    summary = _summary()
    assert (
        summary["classification"]
        == "prospective_embedded_profile_validation_failed"
    )
    assert not summary["passed"]
    assert summary["authorized_next"] is None
    decision = summary["prospective_decision"]
    expected = {
        f"{profile}::a{amplitude:.2f}::{sign}"
        for profile in ("p3__inward_shear", "p3__outward_shear")
        for amplitude in (0.5, 1.0)
        for sign in ("minus", "plus")
    }
    assert set(decision["failed_variants"]) == expected
    assert decision["direct_variant_count"] == 12
    assert decision["alternate_variant_count"] == 0
    assert not decision["alternate_base_profiles"]


def test_wp10c9d6c7b_failure_is_direction_only_and_interface_localized() -> None:
    summary = _summary()
    reports = summary["historical_direct_contract_report"][
        "packet_reports"
    ]
    for profile in ("p3__inward_shear", "p3__outward_shear"):
        report = reports[f"{profile}::a1.00::plus"]
        instantaneous = report["instantaneous_exports"]
        assert instantaneous["observed_rms_order"] >= 0.75
        assert instantaneous["observed_maximum_order"] >= 0.75
        assert instantaneous["minimum_significant_component_order"] >= 0.75
        assert instantaneous["maximum_fine_normalized_difference"] <= 0.05
        assert instantaneous["history_cosine"] >= 0.90
        assert instantaneous["refinement_error_cosine"] < 0.90
        assert report["cumulative_exports"]["passed"]
        assert report["state_reference"]["passed"]

    for profile in (
        "p5__inward_shear",
        "p5__outward_shear",
        "p3__material",
    ):
        assert reports[f"{profile}::a1.00::plus"]["passed"]

    localization = summary["interface_localization_report"]
    assert not localization["binding"]
    assert localization["post_result_diagnostic_only"]
    for profile in ("p3__inward_shear", "p3__outward_shear"):
        payload = localization["profiles"][profile]
        assert payload["inner_face_and_distributed"] >= 0.99
        assert payload["coupling_face_and_net_drive"] < 0.90
        assert payload["coupling_face_only"] < 0.90
    for profile in ("p5__inward_shear", "p5__outward_shear"):
        payload = localization["profiles"][profile]
        assert payload["inner_face_and_distributed"] >= 0.98
        assert payload["coupling_face_and_net_drive"] >= 0.98


def test_wp10c9d6c7b_coupling_diagnostics_remain_failed() -> None:
    coupling = _summary()["coupling_diagnostic_report"]
    assert not coupling["passed"]
    assert not coupling["coupling_face_flux_convergence_passed"]
    assert not coupling["interface_state_convergence_passed"]
    assert not coupling["energy_convergence_passed"]
    assert not coupling["absolute_reflection_threshold_applied"]
    assert coupling["maximum_characteristic_eigenpair_defect"] <= 1.0e-12
    assert (
        coupling["characteristic_energy_method"]
        == "complete_coordinate_descriptor_pencil_eigenvectors_"
        "normalized_by_fixed_physical_field_scales"
    )


def test_wp10c9d6c7b_downstream_work_is_blocked() -> None:
    summary = _summary()
    assert not summary["embedded_profile_class_certified"]
    assert not summary["bounded_nonlinear_common_mode_authorized"]
    assert not summary["nonlinear_physical_trajectory_authorized"]
    assert not summary["production_operator_authorized"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]


def test_wp10c9d6c7b_canonical_hashes() -> None:
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
