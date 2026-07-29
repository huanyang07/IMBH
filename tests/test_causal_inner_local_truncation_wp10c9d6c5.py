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
    "run_causal_inner_local_truncation_wp10c9d6c5.py"
)
CANONICAL = (
    ROOT
    / "results/canonical/"
    "causal_inner_local_truncation_wp10c9d6c5"
)
SUMMARY = CANONICAL / "summary.json"
DECISIVE = CANONICAL / "decisive_arrays.npz"

SPEC = importlib.util.spec_from_file_location(
    "wp10c9d6c5_runner",
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


def test_wp10c9d6c5_freezes_operator_profiles_and_decision_gates() -> None:
    assert RUNNER.MESHES == (64, 128, 256, 512)
    assert RUNNER.PRIMARY_CONTINUUM_NODES == 769
    assert RUNNER.SECONDARY_CONTINUUM_NODES == 513
    assert RUNNER.PROJECTION_ORDER == 24
    assert RUNNER.MINIMUM_CLEAN_TRUNCATION_ORDER == 0.75
    assert RUNNER.MINIMUM_PHASE_EXPLAINED_FRACTION == 0.80
    assert RUNNER.MINIMUM_ERROR_COSINE == 0.90
    assert RUNNER.BOUNDARY_BAND_COARSE_EDGE_INDICES == (1, 2, 3)
    assert set(RUNNER.BOUNDARY_PROFILES) == {
        "boundary_band_outgoing_original",
        "boundary_band_outgoing_wider",
        "boundary_band_outgoing_shifted",
        "boundary_band_outgoing_shifted_wider",
    }
    assert RUNNER.BOUNDARY_PROFILE_DEFINITIONS[
        "boundary_band_outgoing_original"
    ]["log_width"] == 0.065
    assert RUNNER.BOUNDARY_PROFILE_DEFINITIONS[
        "boundary_band_outgoing_wider"
    ]["log_width"] == 0.130


def test_wp10c9d6c5_canonical_classification_and_stops() -> None:
    summary = _summary()
    assert summary["work_package"] == "WP10c9d6c5"
    assert summary["passed"]
    assert summary["parent_wp10c9d6c4_classification_preserved"]
    assert summary["parent_classification"] == (
        "prospective_heldout_uniform_validation_failed"
    )
    assert summary["classification"] == (
        "narrow_profile_preasymptotic_width_crossover_no_redesign"
    )
    assert summary["authorized_next"] == (
        "prospective_transport_packet_validation"
    )
    selection = summary["mechanism_selection"]
    assert selection["local_truncation_cleanly_contracts"]
    assert selection["narrow_profile_width_crossover_selected"]
    assert not selection["phase_crossover_selected"]
    assert not selection["causality_claimed"]
    assert not summary["direct_operator_redesign_authorized"]
    assert not summary["embedded_export_discrimination_authorized"]
    assert not summary["nonlinear_physical_trajectory_authorized"]
    assert not summary["production_operator_authorized"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]
    assert not summary["operator_changed"]


def test_wp10c9d6c5_continuum_and_discrete_ledgers_close() -> None:
    summary = _summary()
    ledger = summary["ledger_report"]
    assert ledger["passed"]
    assert (
        ledger["maximum_discrete_ledger_relative_defect"]
        <= RUNNER.MAXIMUM_DISCRETE_LEDGER_DEFECT
    )
    assert (
        ledger["maximum_continuum_ledger_relative_defect"]
        <= RUNNER.MAXIMUM_CONTINUUM_LEDGER_DEFECT
    )
    assert (
        ledger["maximum_truncation_ledger_relative_defect"]
        <= RUNNER.MAXIMUM_DISCRETE_LEDGER_DEFECT
    )
    assert all(
        report["passed"]
        for report in summary["continuum_reference_reports"].values()
    )
    assert summary["maximum_restart_defect"] <= (
        RUNNER.wp10c9d6c4.MAXIMUM_RESTART_DEFECT
    )


def test_wp10c9d6c5_width_not_boundary_shift_controls_failure() -> None:
    reports = _summary()["boundary_profile_reports"]

    def instantaneous(name: str) -> dict:
        return reports[name]["historical"]["primary_fine"][
            "instantaneous"
        ]

    assert not instantaneous("boundary_band_outgoing_original")["passed"]
    assert not instantaneous("boundary_band_outgoing_shifted")["passed"]
    assert instantaneous("boundary_band_outgoing_wider")["passed"]
    assert instantaneous("boundary_band_outgoing_shifted_wider")[
        "passed"
    ]
    assert (
        instantaneous("boundary_band_outgoing_original")[
            "observed_rms_order"
        ]
        >= 1.9
    )
    for name in RUNNER.BOUNDARY_PROFILES:
        cumulative = reports[name]["historical"]["primary_fine"][
            "cumulative"
        ]
        assert cumulative["passed"]


def test_wp10c9d6c5_local_truncation_contracts_in_all_three_bands() -> None:
    attribution = _summary()["band_attribution"]
    assert attribution[
        "original_profile_local_truncation_cleanly_contracts"
    ]
    original = attribution["profile_reports"][
        "boundary_band_outgoing_original"
    ]
    assert len(original) == 3
    for band in original.values():
        assert band["cleanly_contracting"]
        assert (
            band["minimum_fine_pair_order"]
            >= RUNNER.MINIMUM_CLEAN_TRUNCATION_ORDER
        )
        assert band["fine_direction_cosine"] >= 0.99
        assert band["group_stability"]["boundary"]["stable"]


def test_wp10c9d6c5_phase_and_historical_fit_remain_nonbinding() -> None:
    summary = _summary()
    phase = summary["phase_amplitude_report"][
        "boundary_band_outgoing_original"
    ]
    assert not phase["phase_crossover_selected"]
    assert (
        phase["coarse_medium"]["explained_energy_fraction"]
        < RUNNER.MINIMUM_PHASE_EXPLAINED_FRACTION
    )
    assert (
        phase["medium_fine"]["explained_energy_fraction"]
        < RUNNER.MINIMUM_PHASE_EXPLAINED_FRACTION
    )
    historical = summary["historical_representation_report"]
    assert not historical["binding_attribution_eligible"]
    assert (
        historical["maximum_representation_to_fine_spatial_ratio"]
        > RUNNER.MAXIMUM_HISTORICAL_REPRESENTATION_TO_FINE_RATIO
    )


def test_wp10c9d6c5_canonical_arrays_and_block_sum_are_exact() -> None:
    summary = _summary()
    assert SUMMARY.exists()
    assert DECISIVE.exists()
    assert _sha256(DECISIVE) == summary["decisive_arrays_sha256"]
    source_hashes = summary["implementation_source_hashes"]
    assert (
        "tests/test_causal_inner_local_truncation_wp10c9d6c5.py"
        in source_hashes
    )
    for relative, expected in source_hashes.items():
        assert _sha256(ROOT / relative) == expected
    with np.load(DECISIVE, allow_pickle=False) as source:
        assert set(source.files) == set(summary["decisive_array_hashes"])
        for name in source.files:
            assert (
                _array_sha256(source[name])
                == summary["decisive_array_hashes"][name]
            )
        prefix = (
            "boundary_band_outgoing_original__uniform_N512__"
        )
        total = np.asarray(
            source[prefix + "total_truncation_rows"],
            dtype=float,
        )
        reconstructed = sum(
            (
                np.asarray(
                    source[
                        prefix + "truncation_block__" + block
                    ],
                    dtype=float,
                )
                for block in RUNNER.TRUNCATION_BLOCK_NAMES
            ),
            start=np.zeros_like(total),
        )
        scale = max(float(np.linalg.norm(total)), np.finfo(float).tiny)
        assert np.linalg.norm(reconstructed - total) / scale <= 1.0e-12
