from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    ROOT
    / "scripts/"
    "run_causal_inner_monolithic_four_level_wp10c9d6c2.py"
)
CANONICAL = (
    ROOT
    / "results/canonical/"
    "causal_inner_monolithic_four_level_wp10c9d6c2"
)
SUMMARY = CANONICAL / "summary.json"
DECISIVE = CANONICAL / "decisive_arrays.npz"

SPEC = importlib.util.spec_from_file_location(
    "wp10c9d6c2_runner",
    RUNNER_PATH,
)
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


def test_wp10c9d6c2_contract_preserves_all_physical_gates() -> None:
    assert RUNNER.MESHES == (64, 128, 256, 512)
    assert RUNNER.LABELS == (
        "uniform_N64",
        "uniform_N128",
        "uniform_N256",
        "uniform_N512",
    )
    assert RUNNER.TRIPLETS["fine_N128_N256_N512"] == (
        "uniform_N128",
        "uniform_N256",
        "uniform_N512",
    )
    assert RUNNER.MINIMUM_EXPORT_ORDER == 0.75
    assert RUNNER.MAXIMUM_FINE_PHYSICAL_DIFFERENCE == 0.05
    assert RUNNER.MINIMUM_HISTORY_COSINE == 0.90
    assert RUNNER.MINIMUM_ERROR_COSINE == 0.90
    assert RUNNER.PRIMARY_CONTINUATION == "from_N256"
    assert RUNNER.SECONDARY_CONTINUATION == "from_N128"


def test_wp10c9d6c2_history_metric_requires_error_alignment() -> None:
    times = np.linspace(0.0, 1.0, 21)
    coarse = np.column_stack(
        tuple(
            (1.0 + 0.1 * field) * np.sin(times)
            for field in range(13)
        )
    )
    error = np.column_stack(
        tuple(
            (0.02 + 0.001 * field) * np.cos(times)
            for field in range(13)
        )
    )
    aligned = {
        "coarse": coarse,
        "medium": coarse + error,
        "fine": coarse + error + 0.25 * error,
    }
    report = RUNNER._history_metrics(
        aligned,
        ("coarse", "medium", "fine"),
        np.ones(13),
    )
    assert report["passed"]
    rotated = dict(aligned)
    rotated_error = np.column_stack(
        tuple(
            (0.02 + 0.001 * field) * np.sin(4.0 * np.pi * times)
            for field in range(13)
        )
    )
    rotated["fine"] = coarse + error + 0.25 * rotated_error
    failed = RUNNER._history_metrics(
        rotated,
        ("coarse", "medium", "fine"),
        np.ones(13),
    )
    assert not failed["passed"]
    assert failed["refinement_error_cosine"] < 0.90


def test_wp10c9d6c2_builds_one_nested_fine_grid_without_clipping() -> None:
    configurations, _decisive, report = (
        RUNNER._build_four_configurations()
    )
    assert report["passed"]
    assert report["fine_active_cells"] == 192
    assert report["grid_nesting_defect"] <= 1.0e-14
    assert report["reference_background_defect"] <= 1.0e-14
    assert report["maximum_reconstruction_factor_change"] == 0.0
    fine = configurations["uniform_N512"]
    assert fine["base_primitives"].shape == (192, 5)
    assert fine["context"].stream_sources is not None
    assert np.all(fine["context"].stream_sources.matrix == 0.0)
    assert report["continuation_scaled_cosine"] > 0.99


def test_wp10c9d6c2_canonical_evidence_is_self_consistent() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert summary["work_package"] == "WP10c9d6c2"
    assert summary["parent_wp10c9d6c1_classification_preserved"]
    assert not summary["embedded_export_discrimination_authorized"]
    assert not summary["nonlinear_physical_trajectory_authorized"]
    assert not summary["production_operator_authorized"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]
    assert not summary["uses_production_generator"]
    assert not summary["uses_production_anchor_storage_derivative"]
    assert _sha256(DECISIVE) == summary["decisive_arrays_sha256"]
    with np.load(DECISIVE, allow_pickle=False) as source:
        assert set(source.files) == set(summary["decisive_array_hashes"])
        for name in source.files:
            assert (
                _array_sha256(source[name])
                == summary["decisive_array_hashes"][name]
            )
