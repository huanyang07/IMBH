from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNNER = (
    ROOT
    / "scripts/run_causal_inner_dynamic_localization_wp10c9d5b.py"
)
CANONICAL = (
    ROOT
    / "results/canonical/"
    "causal_inner_dynamic_localization_wp10c9d5b"
)


def _module():
    spec = importlib.util.spec_from_file_location(
        "wp10c9d5b_runner",
        RUNNER,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_localization_gates_and_decision_branches_are_predeclared() -> None:
    module = _module()
    assert module.ANALYZED_BASE_COMMIT == (
        "cb10412aef66ff5e1e2724f8bd702b2c17a5f734"
    )
    assert module.MAXIMUM_RECOVERY_RADIUS_OVER_RG == 5.0
    assert module.MINIMUM_RECOVERY_ORDER == 0.75
    assert module.MAXIMUM_FINE_NORMALIZED_DIFFERENCE == 0.05
    assert module.MINIMUM_FINE_SIGNED_COSINE == 0.90
    assert module.REQUIRED_CONSECUTIVE_RECOVERY_SURFACES == 2
    assert module.MINIMUM_DOMINANT_BLOCK_FRACTION == 0.50
    assert module.MINIMUM_DOMINANT_BLOCK_COSINE == 0.90


def test_canonical_localization_evidence_is_self_consistent() -> None:
    module = _module()
    required = (
        CANONICAL / "config.json",
        CANONICAL / "decisive_arrays.npz",
        CANONICAL / "provenance.json",
        CANONICAL / "summary.json",
        CANONICAL / "SHA256SUMS.txt",
    )
    assert all(path.exists() for path in required)
    for line in (CANONICAL / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        expected, name = line.split("  ", maxsplit=1)
        assert _sha256(CANONICAL / name) == expected
    summary = json.loads(
        (CANONICAL / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["work_package"] == "WP10c9d5b"
    assert summary["parent_wp10c9d5_candidate_remains_rejected"] is True
    assert (
        summary["parent_wp10c9d5a_global_hardening_remains_rejected"]
        is True
    )
    assert summary["method_passed"] is True
    assert summary["binding_branch"] in {
        "A_compact_recovery_radius",
        "B_first_face_or_first_cell_dominance",
        "C_descriptor_dominance",
        "D_no_compact_recovery_or_stable_dominant_term",
    }
    assert summary["frozen_candidate_recertification_authorized"] is False
    assert summary["nonlinear_candidate_authorized"] is False
    assert summary["production_operator_authorized"] is False
    assert summary["fixed_q_micro_solver_authorized"] is False
    assert summary["reduced_slow_evolution_authorized"] is False
    assert summary["decisive_arrays_sha256"] == _sha256(
        CANONICAL / "decisive_arrays.npz"
    )
    with np.load(CANONICAL / "decisive_arrays.npz") as archive:
        assert set(archive.files) == set(summary["decisive_array_hashes"])
        for name in archive.files:
            assert (
                module._array_sha256(archive[name])
                == summary["decisive_array_hashes"][name]
            )
