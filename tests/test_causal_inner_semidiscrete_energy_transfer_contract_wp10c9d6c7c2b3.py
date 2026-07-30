import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CASE = (
    ROOT
    / "results/canonical/"
    "causal_inner_semidiscrete_energy_transfer_contract_wp10c9d6c7c2b3"
)
SOURCE_PARENT = "6b144ecf325efbd428b25b14f0f388d8e0369515"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _json(name: str) -> dict:
    return json.loads((CASE / name).read_text(encoding="utf-8"))


def test_canonical_positive_transfer_contract_and_stop_gates():
    summary = _json("summary.json")
    decision = summary["binding_decision"]
    assert summary["analyzed_base_commit"] == SOURCE_PARENT
    assert summary["passed"]
    assert (
        summary["classification"]
        == "positive_fixed_band_arrival_energy_contract_frozen_"
        "uniform_validation_authorized"
    )
    assert decision["c2b1_rejection_preserved"]
    assert decision["c2b2_interpretation_preserved"]
    assert decision["positive_initial_reference_certified"]
    assert decision["positive_fixed_band_arrival_contract_frozen"]
    assert not decision["local_face_transmission_contract_certified"]
    assert decision["uniform_c2b4_authorized"]
    assert not decision["embedded_c2c1_authorized"]
    assert not decision["embedded_c2c2_authorized"]
    assert not decision["operator_or_interface_redesign_authorized"]
    assert not decision["nonlinear_authorized"]
    assert not decision["fixed_Q_or_reduction_authorized"]
    assert (
        summary["authorized_next"]
        == "WP10c9d6c7c2b4_one_way_uniform_arrival_energy_validation"
    )


def test_positive_reference_bands_windows_and_partition():
    summary = _json("summary.json")
    reference = summary["positive_reference"]
    assert reference["passed"]
    assert reference["minimum_positive_initial_energy"] > 0.0
    assert reference["zero_null_initial_energy"] == 0.0
    assert reference["maximum_initial_receiving_band_energy"] == 0.0
    assert reference["minimum_energy_metric_eigenvalue"] > 0.0
    assert reference["maximum_family_partition_relative_defect"] <= 1.0e-10
    assert reference["source_band_faces_N98"] == [52, 95]
    assert reference["receiving_band_faces_N98"] == [6, 49]
    assert reference["upstream_diagnostic_band_faces_N98"] == [95, 98]
    assert reference["arrival_windows_derived_before_propagation"]
    assert reference["bands_fixed_in_physical_space_and_nested_across_levels"]

    manifest = _json("transfer_manifest.json")
    observable = manifest["prospective_observable"]
    assert observable["strictly_nonnegative"]
    assert observable["need_not_be_bounded_by_one"]
    assert manifest["energy_ledger"]["local_face_ratio_is_not_reintroduced"]
    assert manifest["uncertainty_contract"]["no_slow_impact_threshold"]
    assert (
        "no root-sum-square"
        in manifest["uncertainty_contract"]["combination"]
    )

    with np.load(CASE / "decisive_arrays.npz", allow_pickle=False) as arrays:
        assert np.array_equal(arrays["reference_levels"], (98, 196, 392))
        assert np.array_equal(
            arrays["level_band_faces"],
            (
                (98, 6, 49, 52, 95, 95, 98),
                (196, 12, 98, 104, 190, 190, 196),
                (392, 24, 196, 208, 380, 380, 392),
            ),
        )
        assert arrays["primary_arrival_windows_seconds"].shape == (3, 2)
        assert arrays["arrival_window_nuisance_seconds"].shape == (3, 3, 2)
        assert np.all(arrays["initial_total_energy"][:4] > 0.0)
        assert arrays["initial_total_energy"][4] == 0.0
        assert np.all(arrays["initial_receiving_band_energy"] == 0.0)
        assert np.all(arrays["initial_upstream_band_energy"] == 0.0)
        assert np.max(
            arrays["family_partition_relative_defect"][:4]
        ) <= 1.0e-10


def test_canonical_hashes_and_sources_are_current():
    summary = _json("summary.json")
    provenance = _json("provenance.json")
    assert provenance["source_parent_commit"] == SOURCE_PARENT
    assert provenance["scientific_status"] == "CERTIFIED"
    assert summary["decisive_arrays_sha256"] == _sha256(
        CASE / "decisive_arrays.npz"
    )
    assert summary["config_sha256"] == _sha256(CASE / "config.json")
    for relative, expected in summary["implementation_source_hashes"].items():
        assert _sha256(ROOT / relative) == expected

    declared = {}
    for line in (CASE / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        digest, name = line.split("  ", 1)
        declared[name] = digest
    for name in (
        "config.json",
        "decisive_arrays.npz",
        "provenance.json",
        "summary.json",
        "transfer_manifest.json",
    ):
        assert declared[name] == _sha256(CASE / name)
