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
    "run_causal_inner_monolithic_anchor_audit_wp10c9d6c1.py"
)
CANONICAL = (
    ROOT
    / "results/canonical/"
    "causal_inner_monolithic_anchor_audit_wp10c9d6c1"
)
SUMMARY = CANONICAL / "summary.json"
DECISIVE = CANONICAL / "decisive_arrays.npz"

SPEC = importlib.util.spec_from_file_location("wp10c9d6c1_runner", RUNNER_PATH)
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


def test_wp10c9d6c1_contract_preserves_the_binding_stop() -> None:
    assert RUNNER.LABELS == (
        "uniform_N64",
        "uniform_N128",
        "uniform_N256",
    )
    assert RUNNER.REFERENCE_LABEL == "uniform_N128"
    assert RUNNER.PRIMARY_STRIDE == 2
    assert RUNNER.MINIMUM_PROFILE_ORDER == 0.75
    assert RUNNER.MINIMUM_PROFILE_ERROR_COSINE == 0.90
    assert RUNNER.MINIMUM_ERROR_COSINE_IMPROVEMENT == 0.50
    assert "target_inner_face" not in RUNNER.EXPLANATORY_TERMS
    assert all(
        "target_inner_face" not in names
        for names in RUNNER.GROUPS.values()
    )


def test_wp10c9d6c1_group_metrics_require_fixed_coefficient_alignment() -> None:
    target = np.asarray([[1.0, 2.0, -1.0], [0.5, -0.5, 1.5]])
    exact = RUNNER._group_metrics(target, target)
    opposite = RUNNER._group_metrics(target, -target)
    partial = RUNNER._group_metrics(target, 0.2 * target)
    assert exact["passed"]
    assert exact["target_aligned_fraction"] == 1.0
    assert exact["fixed_coefficient_residual"] == 0.0
    assert not opposite["passed"]
    assert not partial["passed"]


def test_wp10c9d6c1_common_lift_keeps_the_reference_anchor_exact() -> None:
    payload, arrays = RUNNER.wp10c9d6c._load_replay_inputs()
    native = RUNNER.wp10c9d6c._configurations(payload, arrays)
    common, _decisive, report = (
        RUNNER._common_continuum_configurations(native)
    )
    assert report["passed"]
    assert report["maximum_reference_anchor_defect"] <= 1.0e-14
    assert np.array_equal(
        common[RUNNER.REFERENCE_LABEL]["base_primitives"],
        native[RUNNER.REFERENCE_LABEL]["base_primitives"],
    )
    reference_outer = native[RUNNER.REFERENCE_LABEL][
        "context"
    ].outer_boundary_frozen_exterior_chart
    for label in RUNNER.LABELS:
        assert np.array_equal(
            common[label][
                "context"
            ].outer_boundary_frozen_exterior_chart,
            reference_outer,
        )


def test_wp10c9d6c1_canonical_evidence_is_self_consistent() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert summary["work_package"] == "WP10c9d6c1"
    assert summary["parent_wp10c9d6c_classification_preserved"]
    assert not summary["embedded_export_discrimination_authorized"]
    assert not summary["nonlinear_physical_trajectory_authorized"]
    assert not summary["production_operator_authorized"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]
    assert not summary["uses_production_generator"]
    assert not summary["uses_production_anchor_storage_derivative"]
    assert summary["common_attribution"][
        "target_excluded_from_explanatory_groups"
    ]
    assert _sha256(DECISIVE) == summary["decisive_arrays_sha256"]
    with np.load(DECISIVE, allow_pickle=False) as source:
        assert set(source.files) == set(summary["decisive_array_hashes"])
        for name in source.files:
            assert (
                _array_sha256(source[name])
                == summary["decisive_array_hashes"][name]
            )
