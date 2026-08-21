from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_hybrid_phase_memory_architecture_selection_wp10c9d6c7c3b5c4f25e1 as selection


def test_rank_at_energy_detects_one_direction() -> None:
    matrix = np.asarray([[1.0, 0.0], [2.0, 0.0], [-3.0, 0.0]])
    rank, singular, energy = selection._rank_at_energy(matrix, 0.9999)
    assert rank == 1
    assert singular[0] > 0.0
    assert energy[-1] == 1.0


def test_unit_rows_are_normalized() -> None:
    rows = selection._unit_rows(np.asarray([[3.0, 4.0], [0.0, 2.0]]))
    assert np.allclose(np.linalg.norm(rows, axis=1), 1.0)


def test_classification_preserves_missing_cycle_truth() -> None:
    assert "complete_cycle_truth_missing" in selection.PASS_CLASSIFICATION


def test_execution_repair_is_pre_evidence_and_truth_free() -> None:
    assert selection.LOCK_ARTIFACT.endswith("execution_lock_v5")
    source = (ROOT / selection.THIS_RUNNER).read_text(encoding="utf-8")
    assert 'tube_summary["transition_dynamic_dimension"]' in source
    assert 'tube_metrics["gate_values"]["macro_decoder_closure"]' in source


def test_v2_result_uses_local_atomic_npz_writer() -> None:
    assert selection.ARTIFACT.endswith("_v2")
    assert selection.PARTIAL_ARTIFACT != selection.ARTIFACT
    assert callable(selection._write_npz)
