from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_adaptive_hot_exit_phase_atlas_manifest_v2_wp10c9d6c7c3b5c4f25ed as manifest


def test_v2_recovers_all_interrupted_truth_records_without_recomputation() -> None:
    metrics, arrays = manifest._recover_cache()
    assert metrics["record_count"] == 14
    assert metrics["canonical_window_result_previously_written"] is False
    assert arrays["coordinate470"].shape == (14, 470)
    assert arrays["coordinate_rate470_per_s"].shape == (14, 470)
    assert np.all(np.isfinite(arrays["coordinate_rate470_per_s"]))


def test_v2_changes_only_runtime_packaging_semantics() -> None:
    assert manifest.INITIAL_DURATION_SECONDS == manifest.original.INITIAL_DURATION_SECONDS
    assert manifest.RATE_BASIS_RANKS == manifest.original.RATE_BASIS_RANKS
    assert manifest.HIDDEN_SECANT_FRACTION_MAX == manifest.original.HIDDEN_SECANT_FRACTION_MAX
    assert manifest.MAXIMUM_Q3_RELATIVE_DRIFT == manifest.original.MAXIMUM_Q3_RELATIVE_DRIFT
