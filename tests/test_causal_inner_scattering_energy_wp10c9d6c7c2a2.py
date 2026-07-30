from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from imri_qpe.layer3_minidisk_1d.causal_inner_packet_manifest import (
    causal_array_sha256,
    causal_canonical_json_sha256,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_scattering_energy import (
    causal_manufactured_energy_ledger,
    causal_normalization_invariant_scattering_energy,
)


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = (
    ROOT
    / "results/canonical/"
    "causal_inner_scattering_energy_wp10c9d6c7c2a2"
)
SUMMARY = CANONICAL / "summary.json"
MANIFEST = CANONICAL / "method_manifest.json"
DECISIVE = CANONICAL / "decisive_arrays.npz"
PROVENANCE = CANONICAL / "provenance.json"


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


def test_invariant_energy_is_unchanged_by_generalized_vector_scaling() -> None:
    temporal = np.diag((2.0, 3.0, 4.0, 5.0, 6.0))
    mixing = np.asarray(
        [
            [1.0, 0.1, 0.0, 0.0, 0.0],
            [0.2, 1.0, 0.1, 0.0, 0.0],
            [0.0, 0.1, 1.0, 0.1, 0.0],
            [0.0, 0.0, 0.2, 1.0, 0.1],
            [0.1, 0.0, 0.0, 0.1, 1.0],
        ]
    )
    speeds = np.asarray((-0.8, -0.4, -0.1, 0.3, 0.7))
    evolution = mixing @ np.diag(speeds) @ np.linalg.inv(mixing)
    spatial = temporal @ evolution
    audit = causal_normalization_invariant_scattering_energy(
        temporal,
        spatial,
        np.asarray((1.0, 2.0, 3.0, 4.0, 5.0)),
    )
    assert audit.minimum_energy_eigenvalue > 0.0
    assert audit.maximum_projector_idempotence_defect <= 1.0e-12
    assert audit.maximum_cross_projector_defect <= 1.0e-12
    assert audit.maximum_energy_orthogonality_defect <= 1.0e-12
    assert audit.maximum_symmetrizer_defect <= 1.0e-12
    assert audit.maximum_rescaling_invariance_defect <= 1.0e-12


def test_complete_energy_ledger_closes_with_all_lower_blocks() -> None:
    rng = np.random.default_rng(9417)
    points = 7
    state = rng.normal(size=(points, 5))
    time_derivative = rng.normal(size=(points, 5))
    spatial_derivative = rng.normal(size=(points, 5))
    energy = np.repeat(np.eye(5)[None, :, :], points, axis=0)
    evolution = np.repeat(
        np.diag((-0.7, -0.4, -0.1, 0.2, 0.6))[None, :, :],
        points,
        axis=0,
    )
    lower = {
        "stress_relaxation": np.repeat(
            (-0.2 * np.eye(5))[None, :, :],
            points,
            axis=0,
        ),
        "responsive_height": np.repeat(
            (0.03 * np.eye(5))[None, :, :],
            points,
            axis=0,
        ),
    }
    total_lower = np.sum(np.asarray(tuple(lower.values())), axis=0)
    forcing = (
        time_derivative
        + np.einsum("nij,nj->ni", evolution, spatial_derivative)
        - np.einsum("nij,nj->ni", total_lower, state)
    )
    flux_derivative = rng.normal(size=(points, 5, 5))
    flux_derivative = 0.5 * (
        flux_derivative
        + np.swapaxes(flux_derivative, 1, 2)
    )
    ledger = causal_manufactured_energy_ledger(
        state,
        time_derivative,
        spatial_derivative,
        forcing,
        energy,
        evolution,
        flux_derivative,
        lower,
    )
    assert set(ledger.lower_source_work_by_block) == set(lower)
    assert ledger.maximum_relative_closure_defect <= 2.0e-15


def test_wp10c9d6c7c2a2_energy_method_passes_but_route_is_rejected() -> None:
    summary = _summary()
    assert not summary["passed"]
    assert summary["classification"] == (
        "manufactured_interface_patch_rejected_"
        "unidirectional_characteristic_core"
    )
    assert summary["extension_report"]["passed"]
    assert summary["energy_report"]["passed"]
    assert summary["physical_core_parity"]["passed"]
    assert summary["independent_balance_reference"]["passed"]
    assert not summary["operator_changed"]
    assert not summary["propagation_executed"]


def test_wp10c9d6c7c2a2_C4_extension_and_core_parity() -> None:
    summary = _summary()
    extension = summary["extension_report"]
    assert extension["admissible_cell_count"] == extension["cell_count"] == 98
    assert extension["maximum_core_replay_defect"] <= 1.0e-12
    assert extension["maximum_scaled_C4_join_defect"] <= 1.0e-12
    assert extension["maximum_scaled_C4_far_defect"] <= 1.0e-12
    assert extension["minimum_characteristic_speed_gap"] >= 1.0e-6
    core = summary["physical_core_parity"]
    assert core["maximum_defect"] <= 1.0e-12
    assert core["defects"]["spatial_principal_matrix"] <= 1.0e-12
    assert core["defects"]["temporal_storage_matrix"] <= 1.0e-12


def test_wp10c9d6c7c2a2_invariant_energy_and_reference_gates() -> None:
    summary = _summary()
    energy = summary["energy_report"]
    assert energy["normalization_and_sign_invariant"]
    assert energy["descriptor_compatible"]
    assert not energy["thermodynamic_entropy_claimed"]
    assert energy["minimum_energy_eigenvalue"] > 0.0
    assert energy["maximum_projector_idempotence_defect"] <= 1.0e-12
    assert energy["maximum_cross_projector_defect"] <= 1.0e-12
    assert energy["maximum_energy_ledger_relative_defect"] <= 1.0e-10
    assert energy["constant_state_residual"] <= 1.0e-12
    assert energy["minimum_packet_signal_to_uncertainty_ratio"] >= 5.0
    reference = summary["independent_balance_reference"]
    assert reference["reference_uncertainty_to_fine_difference"] <= 0.10
    assert reference["fourth_order_product_rule_relative_l2"][0] > (
        reference["fourth_order_product_rule_relative_l2"][1]
    )
    assert reference["fourth_order_product_rule_relative_l2"][1] > (
        reference["fourth_order_product_rule_relative_l2"][2]
    )


def test_wp10c9d6c7c2a2_rejects_bidirectional_packet_claim() -> None:
    packet = _summary()["packet_and_characteristic_preflight"]
    speeds = np.asarray(packet["interface_characteristic_speeds_over_c"])
    assert np.all(speeds < 0.0)
    assert packet["interface_positive_characteristic_count"] == 0
    assert packet["interface_negative_characteristic_count"] == 5
    assert not packet["fine_to_coarse_incidence_available"]
    assert packet["coarse_to_fine_incidence_available"]
    assert not packet["bidirectional_incidence_passed"]
    assert not packet["travel_windows_frozen"]
    assert not _summary()["uniform_scattering_propagation_authorized"]
    assert _summary()["authorized_next"] == (
        "WP10c9d6c7c2a3_definitions_only_scattering_scope_revision"
    )


def test_wp10c9d6c7c2a2_decisive_arrays_and_hashes() -> None:
    summary = _summary()
    with np.load(DECISIVE, allow_pickle=False) as source:
        assert set(source.files) == set(summary["decisive_array_hashes"])
        assert source["patch_edges"].shape == (99,)
        assert source["manufactured_primitive_charts"].shape == (98, 5)
        assert source["normalization_invariant_projectors"].shape == (
            98,
            5,
            5,
            5,
        )
        assert source["primitive_energy_metrics"].shape == (98, 5, 5)
        assert source["incidence_direction_available_flags"].tolist() == [
            0,
            1,
        ]
        for name in source.files:
            assert (
                causal_array_sha256(source[name])
                == summary["decisive_array_hashes"][name]
            )
    assert _sha256(DECISIVE) == summary["decisive_arrays_sha256"]


def test_wp10c9d6c7c2a2_manifest_and_provenance_are_canonical() -> None:
    summary = _summary()
    manifest = _manifest()
    payload = {
        key: value
        for key, value in manifest.items()
        if key != "manifest_sha256"
    }
    assert causal_canonical_json_sha256(payload) == (
        manifest["manifest_sha256"]
    )
    assert summary["manifest_sha256"] == manifest["manifest_sha256"]
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    assert provenance["source_parent_commit"] == (
        "de29e71f05be20c979c52354584b7b694fb26c6e"
    )
    assert provenance["scientific_status"] == "REJECTED"
    assert provenance["classification"] == summary["classification"]
    assert not manifest["binding_decision"]["all_c2a2_method_gates_passed"]
    assert not manifest["binding_decision"]["uniform_c2b_authorized"]
