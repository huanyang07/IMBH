from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_nonlinear_5ms_spatial_certificate_wp10c9d6c7c3b5c3h2f as h2f  # noqa: E402


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_state_is_bounded_but_order_is_temporally_unobservable():
    summary = _read(h2f.SUMMARY_PATH)
    state = summary["analysis"]["state"]
    temporal = state["temporal_classification"]
    assert state["raw_spatial_contract_passed"]
    assert state["observed_rms_order"] > 1.9
    assert not temporal["spatial_difference_observable"]
    assert not temporal["spatial_orders_and_error_direction_certifying"]
    assert temporal["passed"]
    assert state["binding_channel_passed"]


def test_observable_instantaneous_exports_reject_certificate():
    summary = _read(h2f.SUMMARY_PATH)
    instant = summary["analysis"]["instantaneous_Tier_I"]
    temporal = instant["temporal_classification"]
    assert temporal["spatial_difference_observable"]
    assert temporal["temporal_uncertainty_to_medium_fine_difference_ratio"] < 0.10
    assert instant["observed_rms_order"] < 0.0
    assert instant["observed_maximum_order"] < 0.0
    assert instant["refinement_error_cosine"] < 0.90
    assert set(instant["failed_component_orders"]) == {
        "inner_flux_mass",
        "inner_flux_angular_momentum",
        "net_drive_mass",
        "net_drive_angular_momentum",
    }
    assert not instant["binding_channel_passed"]


def test_rejected_scope_and_authorization_are_binding():
    summary = _read(h2f.SUMMARY_PATH)
    assert not summary["passed"]
    assert summary["classification"] == (
        "five_ms_spatial_certificate_rejected_Tier_I_exports_"
        "nonconvergent_later_duration_blocked"
    )
    assert summary["middle_fine_5ms_spatial_certificate_issued"]
    assert not summary["third_duration_rung_spatial_convergence_certified"]
    assert not summary["fourth_duration_rung_manifest_authorized"]
    assert not summary["physical_failure_detected"]
    assert summary["numerical_spatial_export_failure_detected"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]


def test_canonical_hashes_close():
    entries = {}
    for line in (h2f.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        digest, name = line.split("  ", 1)
        entries[name] = digest
    assert entries
    for name, digest in entries.items():
        assert _sha256(h2f.CANONICAL_DIRECTORY / name) == digest
