from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from imri_qpe.layer3_minidisk_1d.causal_inner_packet_manifest import (
    causal_array_sha256,
    causal_canonical_json_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = (
    ROOT
    / "results/canonical/"
    "causal_inner_embedded_regularity_manifest_wp10c9d6c7c0"
)
SUMMARY = CANONICAL / "summary.json"
MANIFEST = CANONICAL / "regularity_manifest.json"
DECISIVE = CANONICAL / "decisive_arrays.npz"


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


def test_wp10c9d6c7c0_freezes_four_prospective_controls() -> None:
    summary = _summary()
    manifest = _manifest()
    assert summary["profile_count"] == 4
    assert summary["profile_variant_count"] == 16
    assert set(manifest["profile_definitions"]) == {
        "p4__inward_shear",
        "p4__outward_shear",
        "p3_buffer45__inward_shear",
        "p3_buffer45__outward_shear",
    }
    assert all(
        item["binding"] for item in manifest["profile_variants"]
    )
    assert not manifest["operator_changed"]
    assert not manifest["propagation_executed"]
    assert manifest["c7b_rejection_preserved"]


def test_wp10c9d6c7c0_manifest_hash_is_canonical() -> None:
    summary = _summary()
    manifest = _manifest()
    payload = {
        key: value
        for key, value in manifest.items()
        if key != "manifest_sha256"
    }
    assert (
        causal_canonical_json_sha256(payload)
        == manifest["manifest_sha256"]
        == summary["manifest_sha256"]
    )


def test_wp10c9d6c7c0_profiles_are_uniformly_eligible() -> None:
    summary = _summary()
    assert summary["all_uniform_profiles_eligible"]
    assert all(
        report["passed"]
        for report in summary["uniform_eligibility_reports"].values()
    )
    extrema = summary["measured_extrema"]
    assert extrema["maximum_theta_99"] <= 0.30
    assert extrema["maximum_nyquist_alias_fraction"] <= 1.0e-3
    assert extrema["maximum_projection_defect"] <= 2.0e-12
    assert extrema["maximum_support_endpoint_cell_fraction"] <= 5.0e-3
    assert extrema["minimum_global_family_purity"] >= 0.995
    assert extrema["minimum_active_cell_family_purity"] >= 0.98


def test_wp10c9d6c7c0_trace_controls_are_factorized() -> None:
    summary = _summary()
    assert summary["all_coupling_trace_expectations_passed"]
    for layout in summary["layout_reports"].values():
        for profile_name, report in layout["profile_reports"].items():
            if profile_name.startswith("p4__"):
                assert report["coupling_trace_expectation"] == "active"
                assert report["maximum_coupling_trace_fraction"] >= 1.0e-10
            else:
                assert report["coupling_trace_expectation"] == "inactive"
                assert report["zero_buffer_cell_count"] >= (
                    3 * layout["refinement_ratio"]
                )
                assert report["zero_buffer_norm"] == 0.0
                assert report["maximum_coupling_trace_fraction"] <= 1.0e-15
            assert report["coupling_trace_expectation_passed"]


def test_wp10c9d6c7c0_preserves_historical_controls() -> None:
    controls = _summary()["historical_controls"]
    assert not controls["p3__inward_shear"]["passed"]
    assert not controls["p3__outward_shear"]["passed"]
    assert controls["p5__inward_shear"]["passed"]
    assert controls["p5__outward_shear"]["passed"]
    assert (
        controls["p3__inward_shear"][
            "instantaneous_refinement_error_cosine"
        ]
        < 0.90
    )
    assert (
        controls["p5__inward_shear"][
            "instantaneous_refinement_error_cosine"
        ]
        >= 0.90
    )


def test_wp10c9d6c7c0_authorizes_only_uniform_preflight() -> None:
    summary = _summary()
    assert summary["passed"]
    assert (
        summary["classification"]
        == "endpoint_interface_regularity_manifest_frozen_"
        "uniform_control_preflight_authorized"
    )
    assert (
        summary["authorized_next"]
        == "WP10c9d6c7c1_uniform_then_embedded_regularity_"
        "discrimination"
    )
    assert summary["uniform_control_propagation_authorized"]
    assert not summary["embedded_control_propagation_authorized"]
    assert not summary["bounded_nonlinear_common_mode_authorized"]
    assert not summary["production_operator_authorized"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]


def test_wp10c9d6c7c0_canonical_hashes() -> None:
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
                causal_array_sha256(source[name])
                == summary["decisive_array_hashes"][name]
            )
