from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"
for path in (SCRIPTS, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_shear_root_cause_audit_wp10c9c0 as wp10c9c0


def test_wp10c9c0_machine_evidence_stops_before_path_candidate() -> None:
    if not wp10c9c0.DEFAULT_OUTPUT.exists():
        return
    payload = json.loads(
        wp10c9c0.DEFAULT_OUTPUT.read_text(encoding="utf-8")
    )

    assert payload["work_package"] == "WP10c9c0"
    assert payload["base_commit"] == wp10c9c0.BASE_COMMIT
    assert payload["classification"] == (
        "path_inconsistency_not_proved_selected_shear_damping_persists"
    )
    assert not payload["passed"]

    algebra = payload["sign_derivative_and_shear_energy"]
    assert algebra["passed"]
    assert (
        algebra["maximum_path_small_jump_defect"]
        <= wp10c9c0.MAXIMUM_PATH_LINEARIZATION_DEFECT
    )
    assert (
        algebra["maximum_path_reversal_defect"]
        <= wp10c9c0.MAXIMUM_PATH_REVERSAL_DEFECT
    )
    assert (
        algebra["maximum_derivative_step_defect"]
        <= wp10c9c0.MAXIMUM_DERIVATIVE_PLATEAU_DEFECT
    )
    assert algebra["minimum_energy_eigenvalue"] > 0.0

    fourier = payload["constant_coefficient_fourier"]
    assert fourier["passed"]
    assert all(
        value >= wp10c9c0.MINIMUM_LOCAL_ORDER
        for value in fourier["minimum_observed_orders"].values()
    )
    manufactured = payload["variable_coefficient_manufactured_wave"]
    assert manufactured["passed"]
    assert (
        manufactured["minimum_observed_order"]
        >= wp10c9c0.MINIMUM_LOCAL_ORDER
    )

    packets = payload["corrected_full_operator_shear_packets"]
    assert packets is not None
    assert not packets["passed"]
    assert not packets["by_family"]["inward_shear"]["passed"]
    assert packets["by_family"]["outward_shear"]["passed"]
    inward = packets["by_family"]["inward_shear"]
    assert (
        inward["observed_orders"]["phase_centroid"]
        >= wp10c9c0.MINIMUM_PACKET_PHASE_ORDER
    )
    assert (
        inward["observed_orders"]["characteristic_amplitude_damping"]
        < wp10c9c0.MINIMUM_PACKET_DAMPING_ORDER
    )
    assert (
        inward["observed_orders"]["physical_total_shear_energy_damping"]
        >= wp10c9c0.MINIMUM_PACKET_DAMPING_ORDER
    )
    assert (
        inward["fine_minimum_signed_cosine"]
        >= wp10c9c0.MINIMUM_SIGNED_COSINE
    )

    decision = payload["root_cause_decision"]
    assert not decision["current_split_locally_failed"]
    assert decision["monolithic_locally_passed"]
    assert not decision["corrected_full_packet_passed"]
    assert not decision["path_inconsistency_proved"]
    assert not decision["wp10c9c1_path_candidate_authorized"]

    scope = payload["scope"]
    assert not scope["production_operator_changed"]
    assert not scope["nonlinear_path_flux_implemented"]
    assert not scope["nonlinear_truth_run"]
    assert not scope["fixed_q_averaging_run"]
    assert not scope["reduced_model_run"]
