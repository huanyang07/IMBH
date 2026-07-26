from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"
for path in (SCRIPTS, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_embedded_patch_preflight_wp10c8z as wp10c8z


def test_observed_order_requires_positive_nested_differences() -> None:
    assert wp10c8z._observed_order(4.0, 1.0) == pytest.approx(2.0)
    assert wp10c8z._observed_order(0.0, 1.0) is None
    assert wp10c8z._observed_order(1.0, 0.0) is None


def test_common_profile_interpolation_reproduces_its_nodes() -> None:
    radius = np.geomspace(1.8, 20.0, 9)
    values = np.column_stack(
        (
            np.sin(np.log(radius)),
            np.cos(np.log(radius)),
        )
    )
    evaluate = wp10c8z._common_profile_function(radius, values)
    positive = wp10c8z._common_positive_profile_function(
        radius,
        np.exp(values),
    )

    np.testing.assert_allclose(
        evaluate(radius),
        values,
        rtol=3.0e-16,
        atol=3.0e-17,
    )
    np.testing.assert_allclose(
        positive(radius),
        np.exp(values),
        rtol=3.0e-16,
        atol=0.0,
    )


def test_machine_evidence_rejects_patch_truth_but_passes_kernel() -> None:
    if not wp10c8z.DEFAULT_OUTPUT.exists():
        return
    payload = json.loads(
        wp10c8z.DEFAULT_OUTPUT.read_text(encoding="utf-8")
    )

    assert payload["work_package"] == "WP10c8z"
    assert payload["classification"] == (
        "embedded_patch_inner_phase_not_converged"
    )
    assert not payload["passed"]
    assert payload["method_certification"]["passed"]
    assert payload["method_certification"][
        "maximum_shared_flux_defect"
    ] == 0.0
    assert payload["method_certification"][
        "maximum_telescoping_defect"
    ] == 0.0
    assert payload["method_certification"][
        "restart_split_relative_defect"
    ] <= wp10c8z.MAXIMUM_PROPAGATION_RESTART_DEFECT

    configurations = payload["layout"]["configurations"]
    assert set(configurations) == {
        "N128_exterior_N128_inner_c48",
        "N128_exterior_N256_inner_c48",
        "N128_exterior_N512_inner_c48",
        "N128_exterior_N512_inner_c56_matched",
    }
    for row in configurations.values():
        assert row["operator"]["passed"]
        assert row["pair"]["passed"]
        assert row["shared_flux"]["passed"]
        assert row["shared_flux"]["maximum_state_flux_defect"] == 0.0
        assert row["shared_flux"]["maximum_telescoping_defect"] == 0.0
        assert (
            row["operator"]["maximum_relative_storage_action_defect"]
            <= wp10c8z.MAXIMUM_STORAGE_ACTION_DEFECT
        )

    orders = payload["history"]["observed_orders_by_region"][
        "active_core"
    ]
    assert orders["state"] < wp10c8z.MINIMUM_SPATIAL_ORDER
    assert orders["rate"] < wp10c8z.MINIMUM_SPATIAL_ORDER
    assert not payload["history"]["spatial_gate_passed"]
    assert payload["history"]["signal_gate_passed"]
    fine = payload["history"]["pairwise_by_region"]["active_core"][
        "N256patch_N512patch"
    ]
    assert fine["rate"]["minimum_signed_cosine"] < (
        wp10c8z.MINIMUM_SIGNED_COSINE
    )

    coupling = payload["coupling_location"]
    assert coupling["passed"]
    assert not coupling["response_reached_coupling"]
    assert coupling["maximum_history_defect"] <= (
        wp10c8z.MAXIMUM_COUPLING_LOCATION_HISTORY_DEFECT
    )
    assert max(coupling["signal_fraction_by_configuration"].values()) <= (
        wp10c8z.MAXIMUM_COUPLING_SIGNAL_FRACTION
    )

    decision = payload["decision"]
    assert decision["bulk_near_horizon_operator_redesign_required"]
    assert not decision[
        "bounded_nonlinear_embedded_patch_truth_authorized"
    ]
    assert not decision["one_more_inner_patch_refinement_authorized"]
    assert not decision["production_embedded_patch_authorized"]
    assert not decision["fixed_q_averaging_authorized"]
    assert not decision["reduced_coordinate_selection_authorized"]
    assert not decision["macrostep_authorized"]
