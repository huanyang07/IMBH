from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    ROOT / "scripts/run_causal_inner_monolithic_manufactured_wp10c9d6b.py"
)
CANONICAL = (
    ROOT
    / "results/canonical/causal_inner_monolithic_manufactured_wp10c9d6b"
)
SUMMARY = CANONICAL / "summary.json"
DECISIVE = CANONICAL / "decisive_arrays.npz"

SPEC = importlib.util.spec_from_file_location("wp10c9d6b_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
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


def test_outgoing_wave_has_exact_declared_radial_derivative() -> None:
    base_profile, context = RUNNER._base_profile_and_context()
    wave = RUNNER._outgoing_wave(base_profile, context)
    profile = RUNNER._OutgoingWaveProfile(
        base_profile,
        RUNNER.TIMESTEP_SECONDS,
        wave["direction"],
        wave["speed_over_c"],
    )
    radius = 2.35 * context.grid.gravitational_radius
    step = 1.0e-6
    numerical = (
        profile.chart(radius * np.exp(step))
        - profile.chart(radius * np.exp(-step))
    ) / (radius * (np.exp(step) - np.exp(-step)))
    np.testing.assert_allclose(
        profile.derivative(radius),
        numerical,
        rtol=3.0e-8,
        atol=1.0e-12,
    )
    assert wave["speed_over_c"] < 0.0
    assert wave["incoming_inner_characteristics"] == 0


def test_wp10c9d6b_canonical_evidence_is_self_consistent() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert summary["work_package"] == "WP10c9d6b"
    assert summary["method_passed"]
    assert summary["classification"] == (
        "monolithic_manufactured_balance_and_outgoing_wave_passed_"
        "uniform_export_preflight_authorized"
    )
    assert summary["manufactured_balance"]["passed"]
    assert not summary["manufactured_balance"]["forcing_inserted_into_operator"]
    assert not summary["manufactured_balance"][
        "residual_subtraction_used_in_operator"
    ]
    assert summary["outgoing_near_horizon_wave"]["passed"]
    assert summary["outgoing_near_horizon_wave"]["speed_over_c"] < 0.0
    assert summary["temporal_refinement"]["passed"]
    assert summary["method_ledger"][
        "maximum_affine_reconstruction_path_defect"
    ] <= RUNNER.MAXIMUM_AFFINE_RECONSTRUCTION_PATH_DEFECT
    assert summary["method_ledger"][
        "exact_affine_reconstruction_path_derivative_used"
    ]
    assert summary["declared_temporal_path_product_required"]
    assert not summary["strict_endpoint_storage_potential_authorized"]
    assert summary["uniform_grid_physical_export_preflight_authorized"]
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
