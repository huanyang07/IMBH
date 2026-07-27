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
    "run_causal_inner_frozen_hardening_wp10c9d5a.py"
)
CANONICAL = (
    ROOT
    / "results/canonical/"
    "causal_inner_frozen_hardening_wp10c9d5a"
)
D5_CANONICAL = (
    ROOT
    / "results/canonical/"
    "causal_inner_frozen_discrimination_wp10c9d5"
)


def _module():
    spec = importlib.util.spec_from_file_location("wp10c9d5a_runner", RUNNER)
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


def test_scientific_git_identity_is_exact() -> None:
    module = _module()
    assert module._validate_scientific_git_identity() == {
        "scientific_implementation_commit": (
            "038ba35659e76aff0605fffa5fb457e99362063d"
        ),
        "scientific_implementation_parent_commit": (
            "42dd7f1d4ca048fcbd2faa02e71e0a66db300891"
        ),
        "scientific_implementation_tree_sha": (
            "a1e4e33378154d91d17afe001479b063b74ca27f"
        ),
    }


def test_replay_contexts_reproduce_the_stored_base_residuals() -> None:
    module = _module()
    payload, arrays = module._load_replay_inputs()
    for label in module.REPLAY_LABELS:
        prefix = f"{label}__"
        context = module._context_from_payload(
            payload["contexts"][label],
            arrays,
        )
        residual = module._scaled_delta_function(
            context,
            arrays[prefix + "base_primitives"],
            arrays[prefix + "primitive_column_scales"],
            arrays[prefix + "conservation_row_scales"],
        )
        actual = residual(
            np.zeros(arrays[prefix + "base_primitives"].size)
        )
        np.testing.assert_allclose(
            actual,
            arrays[prefix + "base_scaled_delta"],
            rtol=0.0,
            atol=0.0,
        )


def test_canonical_hardening_evidence_and_corrected_d5_are_consistent() -> None:
    module = _module()
    required = (
        CANONICAL / "config.json",
        CANONICAL / "decisive_arrays.npz",
        CANONICAL / "provenance.json",
        CANONICAL / "replay_contexts.json",
        CANONICAL / "replay_inputs.npz",
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
    d5_summary = json.loads(
        (D5_CANONICAL / "summary.json").read_text(encoding="utf-8")
    )
    d5_provenance = json.loads(
        (D5_CANONICAL / "provenance.json").read_text(encoding="utf-8")
    )
    assert summary["work_package"] == "WP10c9d5a"
    assert summary["production_operator_authorized"] is False
    assert summary["nonlinear_candidate_authorized"] is False
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
    assert d5_summary["analyzed_base_commit"] == (
        "42dd7f1d4ca048fcbd2faa02e71e0a66db300891"
    )
    assert d5_provenance["source_parent_commit"] == (
        "42dd7f1d4ca048fcbd2faa02e71e0a66db300891"
    )
    assert d5_summary["scientific_arrays_unchanged"] is True
    assert _sha256(D5_CANONICAL / "decisive_arrays.npz") == (
        module.WP10C9D5_DECISIVE_ARRAYS_SHA256
    )
