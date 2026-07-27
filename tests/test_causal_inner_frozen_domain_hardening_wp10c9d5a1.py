from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNNER = (
    ROOT
    / "scripts/"
    "run_causal_inner_frozen_domain_hardening_wp10c9d5a1.py"
)
CANONICAL = (
    ROOT
    / "results/canonical/"
    "causal_inner_frozen_domain_hardening_wp10c9d5a1"
)


def _module():
    spec = importlib.util.spec_from_file_location(
        "wp10c9d5a1_runner",
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


def test_domain_audit_is_predeclared_and_parent_rejection_is_preserved() -> None:
    module = _module()
    assert module.ANALYZED_BASE_COMMIT == (
        "155e18339076fd2b27d419173b92e1d5d608963b"
    )
    assert module.INNER_RADIUS_OVER_RG == 5.0
    assert module.STENCIL_HALO_CELLS == 3
    assert module.HELD_OUT_SEEDS == (91051, 91052, 91053, 91054)
    assert module.MAXIMUM_SELECTED_JVP_DEFECT == 5.0e-5
    assert module.MAXIMUM_PLATEAU_ADJACENT_CHANGE == 2.0e-5


def test_canonical_domain_hardening_evidence_is_self_consistent() -> None:
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
    assert summary["work_package"] == "WP10c9d5a1"
    assert summary["parent_wp10c9d5a_remains_globally_rejected"] is True
    assert summary["domain_scoped_derivative_passed"] is True
    assert summary["wp10c9d5b_inner_localization_authorized"] is True
    assert (
        summary["global_frozen_candidate_recertification_authorized"]
        is False
    )
    assert summary["production_operator_authorized"] is False
    assert summary["fixed_q_micro_solver_authorized"] is False
    assert summary["reduced_slow_evolution_authorized"] is False
    localization = summary["spatial_localization"]
    assert localization["outermost_cell_squared_fraction"] >= 0.95
    assert localization["inner_through_5rg_squared_fraction"] <= 0.01
    regions = summary["original_random_0_region_audits"]
    assert regions["inner_through_5rg"]["passed"] is True
    assert regions["inner_plus_stencil_halo"]["passed"] is True
    assert regions["complete_grid"]["passed"] is False
    assert summary["branch_comparison"]["passed"] is True
    assert summary["dense_report"]["passed"] is True
    assert all(
        report["passed"]
        for report in summary["held_out_reports"].values()
    )
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
