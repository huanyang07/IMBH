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
    "causal_inner_embedded_manifest_wp10c9d6c7a"
)
SUMMARY = CANONICAL / "summary.json"
MANIFEST = CANONICAL / "embedded_manifest.json"
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


def test_wp10c9d6c7a_freezes_exact_embedded_layouts() -> None:
    summary = _summary()
    assert (
        summary["classification"]
        == "embedded_layout_and_profile_manifest_frozen_"
        "propagation_authorized"
    )
    assert summary["passed"]
    reports = summary["layout_reports"]
    assert tuple(reports) == (
        "N128_exterior_N128_inner_c48",
        "N128_exterior_N256_inner_c48",
        "N128_exterior_N512_inner_c48",
    )
    assert [reports[label]["n_cells"] for label in reports] == [
        64,
        112,
        208,
    ]
    assert [
        reports[label]["n_refined_cells"] for label in reports
    ] == [48, 96, 192]
    assert all(
        report["coupling_radius_over_rg"] == 12.777241939756358
        for report in reports.values()
    )


def test_wp10c9d6c7a_profiles_are_exact_zero_exterior_and_nested() -> None:
    summary = _summary()
    extrema = summary["measured_extrema"]
    assert extrema["maximum_grid_replay_defect"] == 0.0
    assert extrema["maximum_profile_exterior_norm"] == 0.0
    assert extrema["maximum_profile_restriction_defect"] <= 2.0e-12
    assert extrema["maximum_background_restriction_defect"] <= 2.0e-12
    assert extrema["maximum_exterior_replay_defect"] <= 2.0e-12
    assert extrema["maximum_reconstruction_factor_change"] == 0.0
    assert extrema["maximum_normalized_coupling_trace_jump"] <= 1.0e-4
    assert summary["base_profile_count"] == 5
    assert summary["profile_variant_count"] == 20


def test_wp10c9d6c7a_preserves_scientific_stops() -> None:
    summary = _summary()
    assert summary["uniform_certification_preserved"]
    assert summary["historical_c6c_rejection_preserved"]
    assert summary["embedded_propagation_authorized"]
    assert (
        summary["authorized_next"]
        == "WP10c9d6c7b_prospective_embedded_propagation"
    )
    assert not summary["operator_changed"]
    assert not summary["propagation_executed"]
    assert not summary["nonlinear_physical_trajectory_authorized"]
    assert not summary["production_operator_authorized"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]


def test_wp10c9d6c7a_manifest_and_canonical_hashes() -> None:
    summary = _summary()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest_hash = manifest.pop("manifest_sha256")
    assert causal_canonical_json_sha256(manifest) == manifest_hash
    assert manifest_hash == summary["manifest_sha256"]
    assert _sha256(DECISIVE) == summary["decisive_arrays_sha256"]
    with np.load(DECISIVE, allow_pickle=False) as source:
        assert set(source.files) == set(summary["decisive_array_hashes"])
        for name in source.files:
            assert (
                _array_sha256(source[name])
                == summary["decisive_array_hashes"][name]
            )
    for relative, expected in summary[
        "implementation_source_hashes"
    ].items():
        assert _sha256(ROOT / relative) == expected
