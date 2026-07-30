import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CASE = (
    ROOT
    / "results/canonical/"
    "causal_inner_one_way_transmission_interpretation_wp10c9d6c7c2b2"
)
SOURCE_PARENT = "51a32ff686cea3b91d7f5056c464004399318172"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _json(name: str) -> dict:
    return json.loads((CASE / name).read_text(encoding="utf-8"))


def test_canonical_interpretation_and_stop_gates():
    summary = _json("summary.json")
    decision = summary["binding_decision"]
    assert summary["analyzed_base_commit"] == SOURCE_PARENT
    assert (
        summary["classification"]
        == "exact_semidiscrete_energy_identity_certified_"
        "local_face_transmission_not_certifiable"
    )
    assert summary["passed"]
    assert decision["semidiscrete_energy_identity_passed"]
    assert decision["c2b1_rejection_preserved"]
    assert not decision["all_numerical_transmission_channels_passed"]
    assert not decision["all_numerical_incident_signs_stable_and_positive"]
    assert not decision["local_face_transmission_contract_certified"]
    assert not decision["genuine_uniform_transport_error_selected"]
    assert not decision["embedded_c2c1_authorized"]
    assert not decision["operator_or_interface_redesign_authorized"]
    assert not decision["nonlinear_authorized"]
    assert not decision["fixed_Q_or_reduction_authorized"]
    assert (
        summary["authorized_next"]
        == "WP10c9d6c7c2b3_definitions_only_semidiscrete_energy_"
        "transfer_contract"
    )


def test_exact_semidiscrete_ledgers_and_negative_face_result():
    summary = _json("summary.json")
    maximum_block = max(
        summary["per_level"][f"N{cells}"]["method"][
            "maximum_block_power_defect"
        ]
        for cells in (98, 196, 392)
    )
    maximum_face = max(
        summary["per_level"][f"N{cells}"]["method"][
            "maximum_face_power_defect"
        ]
        for cells in (98, 196, 392)
    )
    maximum_integrated = max(
        summary["per_level"][f"N{cells}"]["method"][
            "maximum_time_integrated_energy_defect"
        ]
        for cells in (98, 196, 392)
    )
    assert maximum_block <= 5.0e-13
    assert maximum_face <= 7.0e-15
    assert maximum_integrated <= 1.1e-11

    expected = {
        "acoustic": (400.66406329076614, 1281.272348867819, 1977.4423623058642),
        "shear": (67.36274768563125, 101.8411072769529, 113.5755926048719),
        "mixed_shear_acoustic": (
            132.09717463156198,
            101.42853970298825,
            93.61930795864798,
        ),
    }
    for family, values in expected.items():
        result = summary["transmission_comparison"][family]
        assert np.allclose(result["numerical"]["values"], values)
        assert not result["numerical"]["passed"]
        assert (
            result["maximum_selected_boundary_face_absolute_fraction"]
            < 0.063
        )
    assert not summary["transmission_comparison"]["acoustic"][
        "numerical_incident_sign_stable_and_positive"
    ]


def test_canonical_hashes_and_sources_are_current():
    summary = _json("summary.json")
    provenance = _json("provenance.json")
    assert provenance["source_parent_commit"] == SOURCE_PARENT
    assert (
        provenance["scientific_status"]
        == "SUPPORTED BUT NOT FULLY CERTIFIED"
    )
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
    for name in ("config.json", "decisive_arrays.npz", "provenance.json", "summary.json"):
        assert declared[name] == _sha256(CASE / name)

    with np.load(CASE / "decisive_arrays.npz", allow_pickle=False) as arrays:
        assert np.array_equal(arrays["reference_levels"], (98, 196, 392))
        assert arrays["N392__stored_energy"].shape == (1025, 3)
        assert arrays["N392__conservative_face_power"].shape == (
            1025,
            3,
            393,
        )
