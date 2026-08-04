from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_causal_inner_nonlinear_paired_base_replay_validation_wp10c9d6c7c3b5c2c1 as runner


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def test_paired_replay_passes_every_separate_bitwise_gate() -> None:
    summary = _read(runner.SUMMARY_PATH)
    assert summary["paired_replay_bitwise"] is True
    assert all(summary["paired_replay_report"].values())
    assert summary["maximum_accumulated_time_spacing_units"] <= 1.0


def test_method_and_fresh_process_envelopes_pass() -> None:
    summary = _read(runner.SUMMARY_PATH)
    method = summary["method_report"]
    assert method["all_steps_accepted"] is True
    assert method["maximum_scaled_residual"] <= 1.0e-10
    assert method["maximum_discrete_ledger_defect"] <= 1.0e-12
    explanatory = summary["fresh_process_committed_main_comparison"]
    assert explanatory["maximum_scaled_state_difference"] <= 1.0e-12
    assert explanatory["maximum_scaled_export_difference"] <= 1.0e-12


def test_only_missing_perturbed_second_rung_is_authorized() -> None:
    summary = _read(runner.SUMMARY_PATH)
    assert summary["passed"] is True
    assert summary["authorized_next"] == (
        "WP10c9d6c7c3b5c2d_second_rung_perturbed_completion"
    )
    assert summary["historical_c2_failure_preserved"] is True
    assert summary["perturbed_second_rung_authorized"] is True
    assert summary["later_duration_rungs_authorized"] is False
    assert summary["fixed_q_micro_solver_authorized"] is False
    assert summary["reduced_slow_evolution_authorized"] is False


def test_decisive_arrays_close_bitwise() -> None:
    import numpy as np

    with np.load(runner.DECISIVE_ARRAYS, allow_pickle=False) as arrays:
        assert np.array_equal(arrays["direct_states"], arrays["serialized_states"])
        assert np.array_equal(arrays["direct_exports"], arrays["serialized_exports"])
        assert np.array_equal(
            arrays["direct_primitive_history"], arrays["serialized_primitive_history"]
        )
        assert np.array_equal(
            arrays["direct_mapped_history"], arrays["serialized_mapped_history"]
        )
        assert np.array_equal(
            arrays["direct_height_history"], arrays["serialized_height_history"]
        )


def test_canonical_hashes_close() -> None:
    for line in (runner.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert hashlib.sha256(
            (runner.CANONICAL_DIRECTORY / name).read_bytes()
        ).hexdigest() == digest
