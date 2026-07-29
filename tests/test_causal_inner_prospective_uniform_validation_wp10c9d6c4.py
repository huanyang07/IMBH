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
    "run_causal_inner_prospective_uniform_validation_wp10c9d6c4.py"
)
CANONICAL = (
    ROOT
    / "results/canonical/"
    "causal_inner_prospective_uniform_validation_wp10c9d6c4"
)
SUMMARY = CANONICAL / "summary.json"
DECISIVE = CANONICAL / "decisive_arrays.npz"

SPEC = importlib.util.spec_from_file_location(
    "wp10c9d6c4_runner",
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


def test_wp10c9d6c4_freezes_c3_operator_lift_and_gates() -> None:
    assert RUNNER.MESHES == (64, 128, 256, 512)
    assert RUNNER.PRIMARY_PROJECTION_ORDER == 24
    assert RUNNER.SECONDARY_PROJECTION_ORDER == 12
    assert RUNNER.MINIMUM_EXPORT_ORDER == 0.75
    assert RUNNER.MAXIMUM_FINE_PHYSICAL_DIFFERENCE == 0.05
    assert RUNNER.MINIMUM_HISTORY_COSINE == 0.90
    assert RUNNER.MINIMUM_ERROR_COSINE == 0.90
    assert RUNNER.MAXIMUM_LIFT_TO_FINE_EXPORT_RATIO == 0.10
    assert RUNNER.CALIBRATION_PROFILES == (
        "historical_common_smooth_fit",
    )
    assert len(RUNNER.HELDOUT_PROFILES) >= 3
    assert "heldout_broad_outer_inner" in RUNNER.HELDOUT_PROFILES
    assert "heldout_first_cell_outgoing" in RUNNER.HELDOUT_PROFILES
    assert set(RUNNER.AMPLITUDE_SIGN_FACTORS) == {-1.0, 0.5}


def test_wp10c9d6c4_profiles_are_fixed_smooth_and_admissible() -> None:
    _parent, arrays = RUNNER._load_parent()
    configurations, decisive, report = RUNNER._build_configurations(
        arrays
    )
    assert report["passed"]
    assert report["maximum_frozen_background_defect"] == 0.0
    assert report["historical_calibration_fit"]["passed"]
    assert (
        report["historical_calibration_fit"]["relative_l2_fit_defect"]
        <= RUNNER.MAXIMUM_HISTORICAL_FIT_RELATIVE_L2_DEFECT
    )
    assert (
        report["historical_calibration_fit"]["fit_cosine"]
        >= RUNNER.MINIMUM_HISTORICAL_FIT_COSINE
    )
    outgoing = report["first_cell_outgoing"]
    assert outgoing["coordinate_speed_over_c"] < 0.0
    assert outgoing["first_cell_dominated_on_every_grid"]
    assert (
        report["maximum_perturbed_reconstruction_factor_change"]
        == 0.0
    )
    assert report["profile_definition_sha256"] == (
        RUNNER._fixed_profile_hash()
    )
    for label, active_cells in zip(
        RUNNER.LABELS,
        RUNNER.ACTIVE_CELLS,
        strict=True,
    ):
        configuration = configurations[label]
        for profile in RUNNER.PERTURBATIONS:
            assert configuration["physical_directions"][profile].shape == (
                active_cells,
                5,
            )
            assert configuration["initial_directions"][profile].shape == (
                5 * active_cells,
            )
            secondary = (
                profile + "__projection_order_12"
            )
            assert configuration["physical_directions"][secondary].shape == (
                active_cells,
                5,
            )
            assert (
                f"{label}__{profile}__physical_direction"
                in decisive
            )


def test_wp10c9d6c4_first_cell_profile_is_grid_independent() -> None:
    _parent, arrays = RUNNER._load_parent()
    _configurations, _decisive, report = (
        RUNNER._build_configurations(arrays)
    )
    outgoing = report["first_cell_outgoing"]
    coarse_outer = outgoing["coarse_first_cell_outer_radius_over_rg"]
    assert all(
        radius <= coarse_outer
        for radius in outgoing["peak_radii_over_rg"].values()
    )
    assert outgoing["maximum_eigenpair_defect"] <= 1.0e-12


def test_wp10c9d6c4_canonical_evidence_is_self_consistent() -> None:
    assert SUMMARY.exists()
    assert DECISIVE.exists()
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert summary["work_package"] == "WP10c9d6c4"
    assert summary["parent_wp10c9d6c3_classification_preserved"]
    assert summary["parent_classification"] == (
        "smooth_continuum_four_level_export_direction_certified"
    )
    assert summary["profile_definition_sha256"] == (
        RUNNER._fixed_profile_hash()
    )
    assert summary["classification"] == (
        "prospective_heldout_uniform_validation_failed"
    )
    assert summary["authorized_next"] == (
        "smooth_profile_local_truncation_audit"
    )
    assert summary["method_passed"]
    assert summary["lift_uncertainty_passed"]
    assert not summary["calibration_passed"]
    assert not summary["prospective_heldout_passed"]
    assert summary["smooth_profile_local_truncation_audit_authorized"]
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
