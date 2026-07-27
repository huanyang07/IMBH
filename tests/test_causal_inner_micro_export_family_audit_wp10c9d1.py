from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = (
    ROOT
    / "scripts/run_causal_inner_micro_export_family_audit_wp10c9d1.py"
)
OUTPUT = (
    ROOT
    / "outputs/tables/"
    "causal_inner_micro_export_family_audit_wp10c9d1.json"
)


def _module():
    spec = importlib.util.spec_from_file_location("wp10c9d1_runner", RUNNER)
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


def test_family_attribution_selects_stable_dominant_family() -> None:
    module = _module()
    labels = ("coarse", "medium", "fine")
    times = np.linspace(0.0, 1.0, 9)
    n_observables = len(module.wp10c9d0.OBSERVABLE_NAMES)
    family = {}
    total = {}
    baseline = {
        label: np.zeros(n_observables, dtype=float) for label in labels
    }
    errors = {"coarse": 4.0, "medium": 2.0, "fine": 1.0}
    for label in labels:
        values = np.zeros((5, times.size, n_observables), dtype=float)
        profile = (1.0 + times)[:, None]
        values[0, :, :3] = errors[label] * profile
        values[1, :, :3] = 0.1 * errors[label] * profile
        values[2, :, :3] = 0.05 * errors[label] * profile
        values[3, :, :3] = 0.05 * errors[label] * profile
        values[4, :, :3] = 0.05 * errors[label] * profile
        family[label] = values
        total[label] = np.sum(values, axis=0)
    result = module._error_attribution(
        family,
        total,
        baseline,
        labels,
        selected_observables=np.arange(0, 6, dtype=int),
    )
    assert result["single_family_selected"] is True
    assert result["selected_family"] == "inward_acoustic"
    assert result["coarse_medium"][
        "maximum_family_sum_closure_defect"
    ] < 1.0e-14


def test_inactive_components_are_removed_before_normalization() -> None:
    module = _module()
    labels = ("coarse", "medium", "fine")
    family = {
        label: np.zeros((5, 4, len(module.wp10c9d0.OBSERVABLE_NAMES)))
        for label in labels
    }
    total = {
        label: np.zeros((4, len(module.wp10c9d0.OBSERVABLE_NAMES)))
        for label in labels
    }
    baseline = {
        label: np.ones(len(module.wp10c9d0.OBSERVABLE_NAMES))
        for label in labels
    }
    for index, label in enumerate(labels):
        family[label][2, :, 0] = float(index + 1)
        total[label] = np.sum(family[label], axis=0)
    result = module._error_attribution(
        family,
        total,
        baseline,
        labels,
        selected_observables=np.arange(0, 6, dtype=int),
    )
    assert result["coarse_medium"]["leading_family"] == "material"
    assert np.isfinite(
        result["coarse_medium"]["leading_activity_fraction"]
    )


def test_machine_evidence_hash_and_decision_are_self_consistent() -> None:
    if not OUTPUT.exists():
        pytest.skip("WP10c9d1 evidence has not been generated")
    payload = json.loads(OUTPUT.read_text(encoding="utf-8"))
    arrays = ROOT / payload["arrays_path"]
    assert payload["work_package"] == "WP10c9d1"
    assert payload["arrays_sha256"] == _sha256(arrays)
    if payload["single_subsystem_selected"]:
        assert payload["selected_family"] is not None
    else:
        assert payload["selected_family"] is None
        assert payload["classification"] == (
            "conservative_export_defect_is_multifamily_full_coupled_"
            "operator_required"
        )
