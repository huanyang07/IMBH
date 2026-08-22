from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from imri_qpe.layer3_minidisk_1d.conservative_free_field_rom import HiddenAmplitudeState
import run_causal_inner_truth_free_hot_mode_engine_wp10c9d6c7c3b5c4f25fc as target


def test_state_roundtrip_is_bitwise() -> None:
    state = HiddenAmplitudeState(
        macro=np.asarray((1.0, 2.0)),
        amplitudes=np.asarray((3.0, -4.0)),
        forcing_phase=0.25,
        mode="hot",
        elapsed_seconds=0.5,
    )
    restored = target._state_roundtrip(state)
    assert target._bitwise_state(state, restored)


def test_canonical_package_when_present() -> None:
    if not target.CANONICAL_DIRECTORY.exists():
        return
    helper = target._helper()
    helper._validate_checksums(target.CANONICAL_DIRECTORY)
    summary = helper._read(target.CANONICAL_DIRECTORY / "summary.json")
    metrics = helper._read(target.CANONICAL_DIRECTORY / "hot_engine_metrics.json")
    assert summary["passed"] == metrics["passed"]
    expected = target.PASS_CLASSIFICATION if metrics["passed"] else target.FAIL_CLASSIFICATION
    assert summary["classification"] == expected
    assert metrics["gate_values"]["online_truth_calls"] == 0
    assert metrics["gate_values"]["online_fixed_Q_reaction_calls"] == 0
    assert summary["authorized_next"] == (
        target.AUTHORIZED_NEXT if metrics["passed"] else None
    )
