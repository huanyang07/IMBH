from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_causal_inner_nonlinear_third_duration_rung_spatial_confirmation_manifest_wp10c9d6c7c3b5c3g as runner


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def test_manifest_freezes_only_middle_fine_generic_propagation() -> None:
    manifest = _read(runner.MANIFEST_PATH)
    summary = _read(runner.SUMMARY_PATH)
    experiment = manifest["experiment"]
    assert manifest["propagation_executed"] is False
    assert experiment["profile"] == runner.GENERIC_PROFILE
    assert tuple(experiment["new_layouts"]) == runner.NEW_LAYOUTS
    assert experiment["coarse_c3d_base_and_perturbed_main_replay_strict_reused_by_hash"] is True
    assert summary["middle_fine_generic_spatial_confirmation_authorized"] is True
    assert summary["middle_fine_generic_spatial_confirmation_executed"] is False
    assert summary["third_duration_rung_spatial_convergence_certified"] is False


def test_active_faces_targets_and_common_parent_are_exact() -> None:
    manifest = _read(runner.MANIFEST_PATH)
    experiment = manifest["experiment"]
    state = manifest["state_contract"]
    tier = manifest["Tier_I_contract"]
    assert experiment["active_coupling_face_indices"] == {
        runner.COARSE_LAYOUT: 48,
        runner.MIDDLE_LAYOUT: 96,
        runner.FINE_LAYOUT: 192,
    }
    assert experiment["main_target_microseconds"] == runner.MAIN_TARGET_MICROSECONDS.tolist()
    assert experiment["replay_target_microseconds"] == runner.REPLAY_TARGET_MICROSECONDS.tolist()
    assert experiment["strict_target_microseconds"] == runner.STRICT_TARGET_MICROSECONDS.tolist()
    assert "common_64_cell_parent" in state["restriction"]
    assert tier["interface_flux_must_use_layout_active_face"] == experiment["active_coupling_face_indices"]


def test_uncertainty_and_downstream_stops_remain_binding() -> None:
    manifest = _read(runner.MANIFEST_PATH)
    summary = _read(runner.SUMMARY_PATH)
    uncertainty = manifest["temporal_uncertainty_contract"]
    assert uncertainty["maximum_strict_to_observable_medium_fine_spatial_error_ratio"] == 0.10
    assert uncertainty["observability_factor"] == 5.0
    assert summary["fourth_duration_rung_manifest_authorized"] is False
    assert summary["fixed_q_micro_solver_authorized"] is False
    assert summary["reduced_slow_evolution_authorized"] is False
    assert manifest["positive_branch"]["fixed_q_and_reduced_evolution_still_blocked"] is True


def test_canonical_hashes_close() -> None:
    for line in (runner.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert hashlib.sha256((runner.CANONICAL_DIRECTORY / name).read_bytes()).hexdigest() == digest
