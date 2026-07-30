from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

from imri_qpe.layer3_minidisk_1d.causal_inner_packet_manifest import (
    causal_canonical_json_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    ROOT
    / "scripts/"
    "run_causal_inner_packet_manifest_wp10c9d6c6b.py"
)
CANONICAL = (
    ROOT
    / "results/canonical/"
    "causal_inner_packet_manifest_wp10c9d6c6b"
)
SUMMARY = CANONICAL / "summary.json"
MANIFEST = CANONICAL / "packet_manifest.json"
DECISIVE = CANONICAL / "decisive_arrays.npz"

SPEC = importlib.util.spec_from_file_location(
    "wp10c9d6c6b_runner",
    RUNNER_PATH,
)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
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


def _summary() -> dict:
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_wp10c9d6c6b_freezes_signs_amplitudes_and_purity_gates() -> None:
    assert RUNNER.AMPLITUDE_FACTORS == (0.5, 1.0)
    assert RUNNER.SIGNS == (-1, 1)
    assert RUNNER.MINIMUM_GLOBAL_FAMILY_PURITY == 0.995
    assert RUNNER.MINIMUM_ACTIVE_CELL_FAMILY_PURITY == 0.98
    assert RUNNER.MINIMUM_MIXED_COEFFICIENT_COSINE == 0.99
    assert RUNNER.MAXIMUM_REPLAY_DEFECT == 2.0e-12


def test_wp10c9d6c6b_manifest_hash_and_counts_are_binding() -> None:
    summary = _summary()
    manifest = _manifest()
    declared = manifest.pop("manifest_sha256")
    assert causal_canonical_json_sha256(manifest) == declared
    assert declared == summary["manifest_report"]["manifest_sha256"]
    assert len(manifest["base_profiles"]) == 11
    assert len(manifest["packet_variants"]) == 44
    assert len(manifest["nonbinding_stress_controls"]) == 4
    assert all(
        entry["eligible_for_prospective_propagation"]
        for entry in manifest["base_profiles"]
    )
    assert all(
        not entry["spectrally_eligible"]
        and not entry["propagate_in_prospective_uniform_suite"]
        for entry in manifest["nonbinding_stress_controls"]
    )


def test_wp10c9d6c6b_authorizes_propagation_only() -> None:
    summary = _summary()
    assert summary["classification"] == (
        "packet_definition_manifest_frozen_"
        "uniform_propagation_authorized"
    )
    assert summary["authorized_next"] == (
        "WP10c9d6c6c_prospective_uniform_packet_propagation"
    )
    assert summary["parent_classification_preserved"]
    assert not summary["operator_changed"]
    assert not summary["propagation_executed"]
    assert summary["prospective_uniform_packet_propagation_authorized"]
    assert not summary["embedded_export_discrimination_authorized"]
    assert not summary["nonlinear_physical_trajectory_authorized"]
    assert not summary["production_operator_authorized"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]


def test_wp10c9d6c6b_projection_replay_and_purity_pass() -> None:
    report = _summary()["manifest_report"]
    assert report["passed"]
    assert report["maximum_parent_projection_replay_defect"] == 0.0
    assert report["maximum_scaling_defect"] == 0.0
    for purity in report["purity_reports"].values():
        assert purity["passed"]


def test_wp10c9d6c6b_canonical_arrays_and_hashes() -> None:
    summary = _summary()
    assert _sha256(MANIFEST) == summary["manifest_file_sha256"]
    assert _sha256(DECISIVE) == summary["decisive_arrays_sha256"]
    for relative, expected in summary[
        "implementation_source_hashes"
    ].items():
        assert _sha256(ROOT / relative) == expected
    with np.load(DECISIVE, allow_pickle=False) as source:
        assert set(source.files) == set(summary["decisive_array_hashes"])
        for name in source.files:
            assert (
                _array_sha256(source[name])
                == summary["decisive_array_hashes"][name]
            )
