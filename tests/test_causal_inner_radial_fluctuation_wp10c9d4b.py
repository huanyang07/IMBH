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
    "run_causal_inner_radial_fluctuation_audit_wp10c9d4b.py"
)
CANONICAL = (
    ROOT
    / "results/canonical/"
    "causal_inner_radial_fluctuation_wp10c9d4b"
)


def _module():
    spec = importlib.util.spec_from_file_location("wp10c9d4b_runner", RUNNER)
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


def test_manufactured_profile_is_common_and_has_exact_derivative() -> None:
    module = _module()
    inner = np.asarray([4.7, -0.33, 0.67, 15.2, 3.0e-4])
    outer = np.asarray([4.6, -0.37, 0.62, 14.7, 1.5e-4])
    profile = module._ManufacturedProfile(1.8, 6.648, inner, outer)
    radius = 3.1
    step = 1.0e-6
    numerical = (
        profile.chart(radius * np.exp(step))
        - profile.chart(radius * np.exp(-step))
    ) / (radius * (np.exp(step) - np.exp(-step)))
    np.testing.assert_allclose(
        profile.derivative(radius),
        numerical,
        rtol=2.0e-8,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        profile.chart(profile.inner_radius),
        inner,
        rtol=0.0,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        profile.chart(profile.outer_radius),
        outer,
        rtol=0.0,
        atol=1.0e-15,
    )


def test_canonical_evidence_is_present_and_self_consistent() -> None:
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
    provenance = json.loads(
        (CANONICAL / "provenance.json").read_text(encoding="utf-8")
    )
    assert summary["work_package"] == "WP10c9d4b"
    assert summary["analyzed_base_commit"] == (
        "10546da78561ccb4a5f60a203b8b80a47fa26be3"
    )
    assert provenance["source_parent_commit"] == (
        "10546da78561ccb4a5f60a203b8b80a47fa26be3"
    )
    assert summary["radial_candidate_gate_passed"] is True
    assert summary[
        "wp10c9d5_frozen_linear_discrimination_authorized"
    ] is True
    assert summary["production_operator_authorized"] is False
    assert summary["nonlinear_candidate_authorized"] is False
    assert summary["fixed_q_micro_solver_authorized"] is False
    assert summary["reduced_slow_evolution_authorized"] is False
    assert summary["classification"] == (
        "radial_five_field_candidate_gate_passed_"
        "frozen_linear_discrimination_authorized"
    )
    assert summary["decisive_arrays_sha256"] == _sha256(
        CANONICAL / "decisive_arrays.npz"
    )
    assert summary["manufactured_family"]["residual_subtraction_used"] is False
    assert (
        summary["jacobian_audit"]["production_equality_required"] is False
    )
    assert (
        provenance["implementation_source_manifest_sha256"]
        == summary["implementation_source_manifest_sha256"]
    )
    source_manifest = hashlib.sha256()
    for relative, expected in sorted(
        summary["implementation_source_hashes"].items()
    ):
        assert (ROOT / relative).exists()
        source_manifest.update(relative.encode("utf-8"))
        source_manifest.update(expected.encode("ascii"))
    assert (
        source_manifest.hexdigest()
        == summary["implementation_source_manifest_sha256"]
    )

    with np.load(CANONICAL / "decisive_arrays.npz") as archive:
        assert set(archive.files) == set(summary["decisive_array_hashes"])
        for name in archive.files:
            assert (
                module._array_sha256(archive[name])
                == summary["decisive_array_hashes"][name]
            )
