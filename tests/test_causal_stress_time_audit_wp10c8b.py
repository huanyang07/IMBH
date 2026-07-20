from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_causal_stress_time_audit_wp10c8b as wp10c8b
from imri_qpe.layer3_minidisk_1d import (
    CausalFiveFieldAdaptiveBDF2Restart,
    CausalFiveFieldBDFHistory,
)


def _restart(
    previous_timestep_seconds: float,
    older_timestep_seconds: float,
) -> CausalFiveFieldAdaptiveBDF2Restart:
    state = np.zeros(1, dtype=float)
    ledger = np.zeros(5, dtype=float)
    return CausalFiveFieldAdaptiveBDF2Restart(
        state_vector=state,
        history=CausalFiveFieldBDFHistory(
            previous_physical_increment=np.zeros_like(state),
            previous_vertical_killing_increment=np.zeros((1, 4)),
            previous_timestep_seconds=previous_timestep_seconds,
        ),
        older_physical_increment=np.zeros_like(state),
        older_timestep_seconds=older_timestep_seconds,
        cumulative_actual_conserved_storage=np.array(ledger, copy=True),
        cumulative_actual_vertical_storage=np.array(ledger, copy=True),
        cumulative_boundary_transport=np.array(ledger, copy=True),
        cumulative_endogenous_source=np.array(ledger, copy=True),
        cumulative_stream_source=np.array(ledger, copy=True),
        cumulative_closure_defect=np.array(ledger, copy=True),
        elapsed_time=0.125,
        dt_next=2.0 * previous_timestep_seconds,
        next_order=2,
        accepted_steps=10,
        accepted_bdf2_steps=9,
        rejected_attempts=0,
        audit_count=2,
        provenance={"test": True},
    )


def test_condition_multistep_start_preserves_well_scaled_history() -> None:
    restart = _restart(1.0, 1.0)

    conditioned = wp10c8b._condition_multistep_start(restart)

    assert conditioned is restart
    assert conditioned.next_order == 2


def test_condition_multistep_start_uses_bdf1_outside_ratio_band() -> None:
    for previous, older in ((0.49, 1.0), (2.01, 1.0)):
        restart = _restart(previous, older)

        conditioned = wp10c8b._condition_multistep_start(restart)

        assert conditioned.next_order == 1
        assert conditioned.elapsed_time == restart.elapsed_time
        assert conditioned.dt_next == restart.dt_next
        np.testing.assert_array_equal(
            conditioned.state_vector,
            restart.state_vector,
        )


def test_normalized_balance_rows_are_signed_cancellation_safe() -> None:
    values = wp10c8b._normalized_balance_rows(
        np.asarray([2.0, -3.0, 0.0]),
        np.asarray([1.0, -3.0, 0.0]),
    )

    np.testing.assert_allclose(values, [1.0 / 3.0, 0.0, 0.0])


def test_weighted_summary_uses_normalized_measures() -> None:
    summary = wp10c8b._weighted_summary(
        np.asarray([2.0, 4.0]),
        np.asarray([1.0, 3.0]),
    )

    assert summary["maximum"] == 4.0
    assert summary["median"] == 3.0
    assert summary["weighted_mean"] == 3.5
    assert summary["weighted_rms"] == np.sqrt(13.0)
