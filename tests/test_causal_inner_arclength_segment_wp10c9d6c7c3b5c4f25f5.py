from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_causal_inner_arclength_segment_wp10c9d6c7c3b5c4f25f5 as target


def test_manifest_when_present_is_frozen() -> None:
    if not target.manifest.CANONICAL_DIRECTORY.exists():
        return
    locked = target._validate_manifest(require_clean=False)
    assert locked["contract"]["phase_system"]["arclength_span"] == 2.5e-2


def test_seed_is_exact_Window_five_endpoint() -> None:
    base = target._source()._base_inputs()
    seed = target._seed(base)
    assert seed["time_seconds"] == 2.499999999999999e-6
    assert seed["previous_metrics"]["window_index"] == 5


def test_canonical_segment_when_present() -> None:
    if not target.CANONICAL_DIRECTORY.exists():
        return
    helper = target._helper()
    helper._validate_checksums(target.CANONICAL_DIRECTORY)
    summary = helper._read(target.CANONICAL_DIRECTORY / "summary.json")
    metrics = helper._read(target.CANONICAL_DIRECTORY / "arclength_segment_metrics.json")
    assert summary["passed"]
    assert summary["checkpoint_roundtrip_bitwise"]
    assert metrics["passed"]
    assert metrics["gates"]["time_mapping"]
    assert metrics["new_nonlinear_fixed_Q_roots"] == 0
