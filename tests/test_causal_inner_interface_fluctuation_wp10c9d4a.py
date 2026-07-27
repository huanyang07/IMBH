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
    "run_causal_inner_interface_fluctuation_audit_wp10c9d4a.py"
)
CANONICAL = (
    ROOT
    / "results/canonical/"
    "causal_inner_interface_fluctuation_wp10c9d4a"
)


def _module():
    spec = importlib.util.spec_from_file_location("wp10c9d4a_runner", RUNNER)
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


def test_cell_average_wave_is_exact_for_constant_and_sine_mean() -> None:
    module = _module()
    base = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
    direction = np.asarray([0.2, -0.1, 0.3, 0.1, -0.2])
    averages, edges, centers, spacing = module._cell_average_wave(
        base,
        direction,
        16,
    )
    expected_factor = np.sin(0.5 * spacing) / (0.5 * spacing)
    expected = (
        base[None, :]
        + module.AMPLITUDE
        * expected_factor
        * np.sin(centers)[:, None]
        * direction[None, :]
    )
    np.testing.assert_allclose(averages, expected, rtol=0.0, atol=0.0)
    assert edges.shape == (17,)
    assert centers.shape == (16,)


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
    assert summary["work_package"] == "WP10c9d4a"
    assert summary["analyzed_base_commit"] == (
        "f0b4dcc1715647fb7300c3840546cc61ef4482b7"
    )
    assert provenance["source_parent_commit"] == (
        "f0b4dcc1715647fb7300c3840546cc61ef4482b7"
    )
    assert summary["decisive_arrays_sha256"] == _sha256(
        CANONICAL / "decisive_arrays.npz"
    )
    assert summary["production_operator_authorized"] is False
    assert summary["nonlinear_candidate_authorized"] is False
    assert summary["fixed_q_micro_solver_authorized"] is False
    assert summary["classification"] == (
        "interface_inclusive_fixed_geometry_gate_passed_"
        "radial_well_balance_authorized"
    )
    assert summary["interface_inclusive_gate_passed"] is True
    assert summary["radial_well_balance_audit_authorized"] is True
    expected_pass = all(
        case["passed"] for case in summary["cases"].values()
    )
    assert summary["interface_inclusive_gate_passed"] is expected_pass
    assert (
        summary["radial_well_balance_audit_authorized"]
        is expected_pass
    )
    assert (
        provenance["implementation_source_manifest_sha256"]
        == summary["implementation_source_manifest_sha256"]
    )
    for relative, expected in summary[
        "implementation_source_hashes"
    ].items():
        assert _sha256(ROOT / relative) == expected

    with np.load(CANONICAL / "decisive_arrays.npz") as archive:
        assert set(archive.files) == set(summary["decisive_array_hashes"])
        for name in archive.files:
            assert (
                module._array_sha256(archive[name])
                == summary["decisive_array_hashes"][name]
            )
