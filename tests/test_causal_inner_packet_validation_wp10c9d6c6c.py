from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    ROOT
    / "scripts/"
    "run_causal_inner_packet_validation_wp10c9d6c6c.py"
)
CANONICAL = (
    ROOT
    / "results/canonical/"
    "causal_inner_packet_validation_wp10c9d6c6c"
)
SUMMARY = CANONICAL / "summary.json"
DECISIVE = CANONICAL / "decisive_arrays.npz"

SPEC = importlib.util.spec_from_file_location(
    "wp10c9d6c6c_runner",
    RUNNER_PATH,
)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


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


def test_wp10c9d6c6c_preserves_manifest_and_gates() -> None:
    summary = _summary()
    assert RUNNER.FROZEN_MANIFEST_SHA256 == (
        "c908494d0886e126c4c8f4a6ef80e872e7df6161cf8937bc39cfbbe0a65811fc"
    )
    assert RUNNER.TIME_HORIZON_S == 0.125
    assert RUNNER.TIME_SAMPLE_COUNT == 65
    assert summary["manifest_report"]["verified"]
    assert summary["manifest_report"]["packet_variant_count"] == 44
    assert summary["configuration"]["frozen_contract"][
        "all_manifest_variants_binding"
    ]
    assert summary["configuration"]["frozen_contract"][
        "exact_boundary_semigroup_integral_required"
    ]


def test_wp10c9d6c6c_freezes_only_the_two_failed_shear_bases() -> None:
    summary = _summary()
    comparison = summary["comparison_report"]
    expected = {
        f"p2__{family}::a{amplitude:.2f}::{sign}"
        for family in ("inward_shear", "outward_shear")
        for amplitude in (0.5, 1.0)
        for sign in ("minus", "plus")
    }
    assert set(comparison["failed_packets"]) == expected
    assert comparison["packet_count"] == 44
    assert sum(
        report["passed"]
        for report in comparison["packet_reports"].values()
    ) == 36
    assert comparison["maximum_propagation_scaling_defect"] == 0.0


def test_wp10c9d6c6c_failure_is_only_the_frozen_component_order() -> None:
    reports = _summary()["comparison_report"]["packet_reports"]
    for packet_id, report in reports.items():
        assert report["state_reference"]["passed"]
        for metric_name in (
            "instantaneous_exports",
            "cumulative_exports",
        ):
            metric = report[metric_name]
            assert metric["observed_rms_order"] >= 0.75
            assert metric["observed_maximum_order"] >= 0.75
            assert metric["maximum_fine_normalized_difference"] <= 0.05
            assert metric["history_cosine"] >= 0.90
            assert metric["refinement_error_cosine"] >= 0.90
        if packet_id in _summary()["comparison_report"]["failed_packets"]:
            assert (
                report["instantaneous_exports"]["component_orders"][
                    "vertical_work_angular_momentum"
                ]
                < 0.75
            )
            assert (
                report["cumulative_exports"]["component_orders"][
                    "vertical_work_angular_momentum"
                ]
                < 0.75
            )
        else:
            assert report["instantaneous_exports"]["passed"]
            assert report["cumulative_exports"]["passed"]


def test_wp10c9d6c6c_rejects_embedded_promotion() -> None:
    summary = _summary()
    assert summary["classification"] == (
        "prospective_uniform_packet_validation_failed"
    )
    assert summary["authorized_next"] == (
        "freeze_failed_variant_and_localize"
    )
    assert not summary["passed"]
    assert not summary["embedded_export_discrimination_authorized"]
    assert not summary["nonlinear_physical_trajectory_authorized"]
    assert not summary["production_operator_authorized"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]


def test_wp10c9d6c6c_exact_integrals_and_canonical_hashes() -> None:
    summary = _summary()
    assert max(
        report["maximum_exact_integral_relative_solve_residual"]
        for report in summary["propagation_report"].values()
    ) <= 1.0e-12
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
