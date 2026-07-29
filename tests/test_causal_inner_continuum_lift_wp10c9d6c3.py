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
    "run_causal_inner_continuum_lift_wp10c9d6c3.py"
)
CANONICAL = (
    ROOT
    / "results/canonical/"
    "causal_inner_continuum_lift_wp10c9d6c3"
)
SUMMARY = CANONICAL / "summary.json"
DECISIVE = CANONICAL / "decisive_arrays.npz"

SPEC = importlib.util.spec_from_file_location(
    "wp10c9d6c3_runner",
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


def test_wp10c9d6c3_contract_changes_no_operator_or_historical_gate() -> None:
    assert RUNNER.MESHES == (64, 128, 256, 512)
    assert RUNNER.BACKGROUND_DEGREES == (5, 7)
    assert RUNNER.PRIMARY_BACKGROUND_DEGREE == 5
    assert RUNNER.PRIMARY_PROJECTION_ORDER == 24
    assert RUNNER.SECONDARY_PROJECTION_ORDER == 12
    assert RUNNER.MINIMUM_EXPORT_ORDER == 0.75
    assert RUNNER.MAXIMUM_FINE_PHYSICAL_DIFFERENCE == 0.05
    assert RUNNER.MINIMUM_HISTORY_COSINE == 0.90
    assert RUNNER.MINIMUM_ERROR_COSINE == 0.90


def test_wp10c9d6c3_builds_smooth_proper_measure_common_lift() -> None:
    configurations, decisive, report = (
        RUNNER._build_continuum_configurations()
    )
    assert report["passed"]
    assert report["state_semantics"] == (
        "proper-measure finite-volume cell averages"
    )
    assert report["spline_reports"]["5"]["continuity_order"] == 4
    assert report["spline_reports"]["7"]["continuity_order"] == 6
    assert (
        report["maximum_scaled_constraint_defect"]
        <= RUNNER.MAXIMUM_BACKGROUND_CONSTRAINT_DEFECT
    )
    assert (
        report["maximum_boundary_defect"]
        <= RUNNER.MAXIMUM_BACKGROUND_BOUNDARY_DEFECT
    )
    assert (
        report["maximum_background_representation_difference"]
        <= RUNNER.MAXIMUM_BACKGROUND_REPRESENTATION_DIFFERENCE
    )
    assert report["maximum_reconstruction_factor_change"] == 0.0
    assert decisive["continuum_background_coefficients"].shape == (24, 5)
    for label, active_cells in zip(
        RUNNER.LABELS,
        RUNNER.ACTIVE_CELLS,
        strict=True,
    ):
        configuration = configurations[label]
        assert configuration["base_primitives"].shape == (
            active_cells,
            5,
        )
        assert configuration["physical_directions"][
            "calibration_mixed"
        ].shape == (active_cells, 5)
        difference = RUNNER._relative_difference(
            configuration["initial_directions"]["calibration_mixed"],
            configuration["initial_directions"][
                "calibration_mixed__projection_order_12"
            ],
        )
        assert (
            difference
            <= RUNNER.MAXIMUM_LIFT_STATE_RELATIVE_DIFFERENCE
        )


def test_wp10c9d6c3_conditioned_metric_records_peak_migration() -> None:
    times = np.linspace(0.0, 1.0, 17)
    shape = (times.size, len(RUNNER.OBSERVABLE_NAMES))
    coarse = np.zeros(shape)
    first = np.zeros(shape)
    second = np.zeros(shape)
    first[4, 8] = 1.0
    second[0, 8] = 0.25
    histories = {
        "coarse": {"times": times, "signals": coarse},
        "medium": {"times": times, "signals": coarse + first},
        "fine": {
            "times": times,
            "signals": coarse + first + second,
        },
    }
    report = RUNNER._conditioned_metrics(
        histories,
        np.ones(len(RUNNER.OBSERVABLE_NAMES)),
        labels=("coarse", "medium", "fine"),
        stride=1,
    )
    assert report["peak_migrated"]
    assert report["coarse_medium_argmax"]["time_index"] == 4
    assert report["medium_fine_argmax"]["time_index"] == 0
    assert report["linfinity_order"] == 2.0


def test_wp10c9d6c3_canonical_evidence_is_self_consistent() -> None:
    assert SUMMARY.exists()
    assert DECISIVE.exists()
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert summary["work_package"] == "WP10c9d6c3"
    assert summary["parent_wp10c9d6c2_classification_preserved"]
    assert summary["parent_classification"] == (
        "four_level_uniform_asymptotic_direction_rejected"
    )
    assert not summary["operator_changed"]
    assert not summary["direct_operator_redesign_authorized"]
    assert not summary["embedded_export_discrimination_authorized"]
    assert not summary["nonlinear_physical_trajectory_authorized"]
    assert not summary["production_operator_authorized"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]
    assert _sha256(DECISIVE) == summary["decisive_arrays_sha256"]
    with np.load(DECISIVE, allow_pickle=False) as source:
        assert set(source.files) == set(summary["decisive_array_hashes"])
        for name in source.files:
            assert (
                _array_sha256(source[name])
                == summary["decisive_array_hashes"][name]
            )
