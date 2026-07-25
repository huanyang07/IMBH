from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"
for path in (SCRIPTS, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_mode_healing_wp10c8t as wp10c8t
import run_causal_inner_mode_n128_confirmation_wp10c8t as confirmation


def _localization() -> dict:
    row = {
        "controlling_shell": 0,
        "controlling_shell_l1_fraction": 0.95,
    }
    return {
        "fine": {
            "final_state_controlling_shell": 0,
            "final_rate_controlling_shell": 0,
            "state_support": (row, row),
            "primitive_rate_support": (row, row),
        }
    }


def test_wp10c8t_paths_are_mesh_aware(tmp_path: Path) -> None:
    contract = {
        "context": SimpleNamespace(
            grid=SimpleNamespace(centers=np.zeros(128))
        ),
        "checkpoint_directory": tmp_path,
    }
    restart = wp10c8t._restart_path(
        contract,
        "fine",
        "plus",
        0.125,
    )
    trajectory = wp10c8t._trajectory_path(
        contract,
        "fine",
        "plus",
    )
    assert restart.name == "N128_fine_plus_t0p125_restart.npz"
    assert trajectory.name == "N128_fine_plus_trajectory_t0p125.npz"
    assert restart.parent == tmp_path


def test_signed_direction_comparison_requires_shape_and_amplitude() -> None:
    passed = confirmation._signed_direction_comparison(
        np.asarray((1.0, 2.0)),
        np.asarray((1.1, 2.2)),
    )
    failed = confirmation._signed_direction_comparison(
        np.asarray((1.0, 0.0)),
        np.asarray((0.0, 1.0)),
    )
    assert passed["passed"]
    assert not failed["passed"]


def test_n128_decision_accepts_mesh_supported_persistence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    n64_arrays = tmp_path / "n64.npz"
    np.savez_compressed(
        n64_arrays,
        # The full observable schema is intentionally mesh-specific.  Only
        # the physical coordinate-rate schema must agree across N64/N128.
        fine_pair_full_names=np.asarray(
            ("n64_native_cell_diagnostic", "coordinate_rate_0"),
            dtype="U",
        ),
        fine_minus_coordinate_names=np.asarray(
            ("coordinate_rate_0",),
            dtype="U",
        ),
        fine_pair_signed_slow_rate_half_difference=np.asarray(
            ((1.0, 0.5), (0.5, 0.25))
        ),
    )
    monkeypatch.setattr(confirmation, "N64_ARRAYS", n64_arrays)
    coarse = {
        "times": np.asarray((0.0, 0.125)),
        "full_names": np.asarray(("coordinate_rate_0",), dtype="U"),
        "coordinate_names": np.asarray(("coordinate_rate_0",), dtype="U"),
        "full_spreads": np.asarray(((2.0,), (0.51,))),
        "signed_slow_rate_half_difference": np.asarray(
            ((1.0, 0.5), (0.51, 0.255))
        ),
    }
    fine = {
        **coarse,
        "full_spreads": np.asarray(((2.0,), (0.50,))),
        "signed_slow_rate_half_difference": np.asarray(
            ((1.0, 0.5), (0.50, 0.25))
        ),
    }
    decision, arrays = confirmation._n128_decision(
        pair_arrays={"coarse": coarse, "fine": fine},
        localizations=_localization(),
        all_contracts_passed=True,
    )
    assert (
        decision["classification"]
        == "mesh_supported_persistent_localized_inner_mode_through_0p125s"
    )
    assert decision["architecture_confirmation_passed"]
    assert arrays["uncertainty_exclusive_lower_spreads"][-1, 0] == 0.49
