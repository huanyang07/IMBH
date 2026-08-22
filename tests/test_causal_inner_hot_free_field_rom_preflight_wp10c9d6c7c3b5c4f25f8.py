from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_causal_inner_hot_free_field_rom_preflight_wp10c9d6c7c3b5c4f25f8 as target


def test_five_exact_hot_states_are_identified_without_projection() -> None:
    nodes, coordinates, states = target._states_and_coordinates()
    assert nodes.shape == (5,)
    assert coordinates.shape == (5, 470)
    assert states.shape[0] == 5
    assert np.all(np.diff(nodes) > 0.0)


def test_hidden_basis_selector_is_fail_closed() -> None:
    x = np.linspace(0.0, 1.0, 5)
    rates = np.stack((1.0 + x, 2.0 - x, x * x), axis=1)
    basis, rank, attempts = target._select_hidden_basis(rates)
    assert rank in target.manifest.HIDDEN_RATE_RANKS
    assert basis.shape == (3, rank)
    assert str(rank) in attempts


def test_canonical_package_when_present() -> None:
    if not target.CANONICAL_DIRECTORY.exists():
        return
    helper = target._helper()
    helper._validate_checksums(target.CANONICAL_DIRECTORY)
    summary = helper._read(target.CANONICAL_DIRECTORY / "summary.json")
    metrics = helper._read(
        target.CANONICAL_DIRECTORY / "hot_free_field_metrics.json"
    )
    assert summary["passed"]
    assert summary["truth_free_hidden_amplitude_engine_manifest_authorized"]
    assert not summary["fixed_Q_physical_phase_authorized"]
    assert metrics["gate_values"]["new_fixed_Q_reaction_calls"] == 0
    assert summary["authorized_next"] == target.AUTHORIZED_NEXT
