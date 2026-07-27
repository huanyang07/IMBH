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
    "run_causal_inner_frozen_discrimination_wp10c9d5.py"
)
CANONICAL = (
    ROOT
    / "results/canonical/"
    "causal_inner_frozen_discrimination_wp10c9d5"
)


def _module():
    spec = importlib.util.spec_from_file_location("wp10c9d5_runner", RUNNER)
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


def test_held_out_packets_are_predeclared_and_distinct() -> None:
    module = _module()
    assert tuple(module.HELD_OUT_COEFFICIENTS) == (
        "heldout_shear_acoustic",
        "heldout_material_shear",
        "heldout_five_family",
    )
    supports = [
        tuple(sorted(item.items()))
        for item in module.HELD_OUT_COEFFICIENTS.values()
    ]
    assert len(set(supports)) == 3


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
    assert summary["work_package"] == "WP10c9d5"
    assert summary["analyzed_base_commit"] == (
        "42dd7f1d4ca048fcbd2faa02e71e0a66db300891"
    )
    assert summary["scientific_implementation_commit"] == (
        "038ba35659e76aff0605fffa5fb457e99362063d"
    )
    assert summary["scientific_implementation_tree_sha"] == (
        "a1e4e33378154d91d17afe001479b063b74ca27f"
    )
    assert summary["scientific_arrays_unchanged"] is True
    assert summary["production_operator_authorized"] is False
    assert summary["fixed_q_micro_solver_authorized"] is False
    assert summary["reduced_slow_evolution_authorized"] is False
    assert summary["decisive_arrays_sha256"] == _sha256(
        CANONICAL / "decisive_arrays.npz"
    )
    assert (
        provenance["implementation_source_manifest_sha256"]
        == summary["implementation_source_manifest_sha256"]
    )
    with np.load(CANONICAL / "decisive_arrays.npz") as archive:
        assert set(archive.files) == set(summary["decisive_array_hashes"])
        for name in archive.files:
            assert (
                module._array_sha256(archive[name])
                == summary["decisive_array_hashes"][name]
            )
