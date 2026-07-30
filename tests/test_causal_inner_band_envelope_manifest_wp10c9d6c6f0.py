from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from imri_qpe.layer3_minidisk_1d.causal_inner_packet_manifest import (
    causal_canonical_json_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = (
    ROOT
    / "results/canonical/"
    "causal_inner_band_envelope_manifest_wp10c9d6c6f0"
)
SUMMARY = CANONICAL / "summary.json"
MANIFEST = CANONICAL / "band_envelope_manifest.json"
E1_ARRAYS = (
    ROOT
    / "results/canonical/"
    "causal_inner_integral_conditioning_validation_wp10c9d6c6e1/"
    "decisive_arrays.npz"
)


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


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_wp10c9d6c6f0_freezes_only_unpropagated_eligible_profiles() -> None:
    summary = _summary()
    manifest = _manifest()
    assert summary["base_profile_count"] == 5
    assert summary["profile_variant_count"] == 20
    assert summary["all_inherited_profiles_eligible"]
    assert summary["eligibility_reused_not_reoptimized"]
    assert not summary["propagation_executed"]
    assert not manifest["propagation_executed"]
    assert all(
        item["passed"]
        for item in manifest["profile_eligibility_reports"].values()
    )


def test_wp10c9d6c6f0_manifest_hash_and_exact_profile_set() -> None:
    summary = _summary()
    manifest = _manifest()
    stored = manifest.pop("manifest_sha256")
    assert causal_canonical_json_sha256(manifest) == stored
    assert summary["manifest_sha256"] == stored
    assert set(manifest["base_profile_definitions"]) == {
        "p3__inward_shear",
        "p3__outward_shear",
        "p5__inward_shear",
        "p5__outward_shear",
        "p3__material",
    }
    assert len(manifest["profile_variants"]) == 20
    assert all(item["binding"] for item in manifest["profile_variants"])


def test_wp10c9d6c6f0_limits_alternate_route_precisely() -> None:
    contract = _manifest()["component_route_contract"]
    assert contract["historical_direct_component_route_preserved"]
    assert contract["standard_route_applies_to_every_significant_component"]
    assert contract["alternate_route_scope"] == {
        "physical_block": "lower_height_work",
        "conservative_channel": "angular_momentum",
        "history_types": ["instantaneous", "cumulative"],
    }
    assert contract["alternate_route_forbidden_for_all_other_components"]
    assert contract["minimum_direct_or_band_rms_order"] == 0.75
    assert contract["minimum_direct_or_band_maximum_order"] == 0.75
    assert contract["minimum_active_band_refinement_error_cosine"] == 0.90
    assert contract["maximum_absolute_band_error_envelope"] == 0.05
    assert contract["maximum_cancellation_ratio_each_grid_pair"] == 0.25
    assert contract["triangle_inequality_is_binding_not_a_fitted_model"]
    assert contract["no_fitted_coefficients"]
    assert contract["minimum_profiles_required_to_use_alternate_route"] == 0
    assert contract["no_retroactive_application_to_wp10c9d6c6c"]


def test_wp10c9d6c6f0_authorizes_only_uniform_propagation() -> None:
    summary = _summary()
    assert summary["classification"] == (
        "band_envelope_contract_and_heldout_profiles_frozen_"
        "uniform_propagation_authorized"
    )
    assert summary["historical_classifications_preserved"]
    assert summary["c6c_rejection_preserved"]
    assert not summary["embedded_export_discrimination_authorized"]
    assert not summary["nonlinear_physical_trajectory_authorized"]
    assert not summary["production_operator_authorized"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]


def test_wp10c9d6c6f0_inherited_projection_hashes_are_exact() -> None:
    manifest = _manifest()
    expected = manifest["profile_projection_hashes"]
    with np.load(E1_ARRAYS, allow_pickle=False) as source:
        assert set(expected).issubset(source.files)
        for name, digest in expected.items():
            assert _array_sha256(source[name]) == digest


def test_wp10c9d6c6f0_source_hashes() -> None:
    summary = _summary()
    for relative, expected in summary[
        "implementation_source_hashes"
    ].items():
        assert _sha256(ROOT / relative) == expected
