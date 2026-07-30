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
    "run_causal_inner_packet_resolution_wp10c9d6c6a.py"
)
CANONICAL = (
    ROOT
    / "results/canonical/"
    "causal_inner_packet_resolution_wp10c9d6c6a"
)
SUMMARY = CANONICAL / "summary.json"
DECISIVE = CANONICAL / "decisive_arrays.npz"

SPEC = importlib.util.spec_from_file_location(
    "wp10c9d6c6a_runner",
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


def test_wp10c9d6c6a_freezes_prospective_symbol_contract() -> None:
    assert RUNNER.LABELS == (
        "uniform_N128",
        "uniform_N256",
        "uniform_N512",
    )
    assert RUNNER.SYMBOL_RADII_OVER_RG == (2.2, 3.0, 5.0, 8.0, 11.0)
    assert RUNNER.SYMBOL_TIMES_S[-1] == 0.125
    assert RUNNER.SPECTRAL_ENERGY_QUANTILE == 0.99
    assert RUNNER.MAXIMUM_COMPLETE_SEMIGROUP_ERROR == 0.025
    assert RUNNER.MAXIMUM_PRINCIPAL_SEMIGROUP_ERROR == 0.025
    assert RUNNER.MAXIMUM_FAMILY_LEAKAGE == 0.010
    assert RUNNER.MINIMUM_CERTIFIED_THETA == 0.20
    assert RUNNER.MINIMUM_CROSS_GRID_SYMBOL_ORDER == 1.50


def test_wp10c9d6c6a_canonical_classification_and_stops() -> None:
    summary = _summary()
    assert summary["work_package"] == "WP10c9d6c6a"
    assert summary["parent_classification_preserved"]
    assert summary["parent_classification"] == (
        "narrow_profile_preasymptotic_width_crossover_no_redesign"
    )
    expected = bool(
        summary["symbol_contract"]["passed"]
        and summary["cross_grid_symbol_report"]["passed"]
    )
    assert summary["passed"] is expected
    assert (
        summary["prospective_packet_manifest_authorized"]
        is expected
    )
    assert not summary["operator_changed"]
    assert not summary["uniform_packet_propagation_authorized"]
    assert not summary["embedded_export_discrimination_authorized"]
    assert not summary["nonlinear_physical_trajectory_authorized"]
    assert not summary["production_operator_authorized"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]


def test_wp10c9d6c6a_symbol_and_boundary_contracts_are_distinct() -> None:
    summary = _summary()
    assert summary["boundary_packet_requires_additional_DAE_contract"]
    report = summary["packet_resolution_report"]
    assert report[
        "boundary_packets_require_spectral_and_dae_contracts"
    ]
    assert not report["historical_results_used_to_set_threshold"]
    for profile in report["profiles"].values():
        assert profile["boundary_dae_eligible"]
        assert profile[
            "combined_boundary_packet_eligible"
        ] == (
            profile["spectral_eligible"]
            and profile["boundary_dae_eligible"]
        )


def test_wp10c9d6c6a_stencils_and_reference_are_certified() -> None:
    summary = _summary()
    contract = summary["symbol_contract"]
    for report in contract["radius_reports"].values():
        assert not report["touches_boundary"]
        assert (
            report["maximum_row_symbol_parity_defect"]
            <= RUNNER.MAXIMUM_ROW_SYMBOL_PARITY_DEFECT
        )
        assert (
            report["maximum_omitted_stencil_fraction"]
            <= RUNNER.MAXIMUM_OMITTED_STENCIL_FRACTION
        )
    cross = summary["cross_grid_symbol_report"]
    assert not cross["any_selected_stencil_touches_boundary"]
    assert (
        cross["maximum_row_symbol_parity_defect"]
        <= RUNNER.MAXIMUM_ROW_SYMBOL_PARITY_DEFECT
    )


def test_wp10c9d6c6a_canonical_arrays_and_hashes() -> None:
    summary = _summary()
    assert SUMMARY.exists()
    assert DECISIVE.exists()
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
        theta = np.asarray(source["theta_values"], dtype=float)
        mask = np.asarray(
            source["global_contiguous_passed"],
            dtype=bool,
        )
        certified = (
            float(theta[np.flatnonzero(mask)[-1]])
            if np.any(mask)
            else 0.0
        )
        assert certified == summary["symbol_contract"]["certified_theta"]
