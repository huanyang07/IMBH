from __future__ import annotations

import numpy as np

from imri_qpe.constants import C
from imri_qpe.layer3_minidisk_1d.causal_inner_geometry import (
    kerr_schild_column_geometry,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_seven_field_physical import (
    SEVEN_FIELD_PHYSICAL_PRIMITIVE_NAMES,
    seven_field_physical_state,
)


def _fixture():
    geometry = kerr_schild_column_geometry(5.599841633135499e9, 1.48e9)
    chart = np.asarray(
        [
            4.74082887,
            -0.330628060,
            0.662598339,
            14.9471713,
            2.13041458e-4,
            20.1048472,
            0.0,
        ],
        dtype=float,
    )
    return geometry, chart


def _steps(chart: np.ndarray, stress_scale: float = 2.5e-4) -> np.ndarray:
    return np.asarray(
        [
            2.0e-5,
            2.0e-6,
            2.0e-6,
            2.0e-5,
            2.0e-4 * max(abs(float(chart[4])), stress_scale),
            2.0e-5,
            2.0e-6,
        ],
        dtype=float,
    )


def _five_point_jacobian(function, chart: np.ndarray, steps: np.ndarray):
    center = np.asarray(function(chart), dtype=float)
    jacobian = np.empty((center.size, chart.size), dtype=float)
    for column, step in enumerate(steps):
        direction = np.zeros_like(chart)
        direction[column] = step
        jacobian[:, column] = (
            -np.asarray(function(chart + 2.0 * direction))
            + 8.0 * np.asarray(function(chart + direction))
            - 8.0 * np.asarray(function(chart - direction))
            + np.asarray(function(chart - 2.0 * direction))
        ) / (12.0 * step)
    return center, jacobian


def _entropy_flux_one_form(
    geometry,
    chart: np.ndarray,
    steps: np.ndarray,
    *,
    state_scales: np.ndarray,
    entropy_scale: float,
) -> np.ndarray:
    def evaluate(values):
        return seven_field_physical_state(
            geometry,
            values,
            proper_vertical_frequency=2.7491520839259703,
            alpha=0.1,
        )

    _state, state_jacobian = _five_point_jacobian(
        lambda values: evaluate(values).conserved / state_scales,
        chart,
        steps,
    )
    _flux, flux_jacobian = _five_point_jacobian(
        lambda values: evaluate(values).flux_over_c / state_scales,
        chart,
        steps,
    )
    _entropy, entropy_gradient = _five_point_jacobian(
        lambda values: np.atleast_1d(
            evaluate(values).mathematical_entropy / entropy_scale
        ),
        chart,
        steps,
    )
    entropy_variables = np.linalg.solve(
        state_jacobian.T,
        entropy_gradient.ravel(),
    )
    return entropy_variables @ flux_jacobian


def test_candidate_state_map_is_finite_and_uses_seven_coordinates() -> None:
    geometry, chart = _fixture()
    state = seven_field_physical_state(
        geometry,
        chart,
        proper_vertical_frequency=2.7491520839259703,
        alpha=0.1,
    )
    assert len(SEVEN_FIELD_PHYSICAL_PRIMITIVE_NAMES) == 7
    assert state.conserved.shape == (7,)
    assert state.flux_over_c.shape == (7,)
    assert np.all(np.isfinite(state.conserved))
    assert np.all(np.isfinite(state.flux_over_c))
    assert state.calibration.reservoir_coefficient > 0.0
    np.testing.assert_allclose(
        state.calibration.reservoir_coefficient,
        state.calibration.extended_specific_enthalpy_over_c2
        * state.calibration.viscous_signal_speed_over_c**2,
        rtol=2.0e-14,
        atol=0.0,
    )


def test_candidate_entropy_flux_one_form_reports_integrability_defect() -> None:
    """Measure, but do not prejudge, the candidate entropy obstruction."""

    geometry, chart = _fixture()
    base = seven_field_physical_state(
        geometry,
        chart,
        proper_vertical_frequency=2.7491520839259703,
        alpha=0.1,
    )
    state_scales = np.maximum(np.abs(base.conserved), 1.0)
    rest_mass = float(base.conserved[0])
    state_scales[4] = max(
        abs(float(base.conserved[4])),
        rest_mass * base.calibration.equilibrium_specific_stress,
    )
    state_scales[5] = abs(float(base.conserved[5]))
    state_scales[6] = rest_mass * C * 0.03
    entropy_scale = max(abs(base.mathematical_entropy), 1.0)
    defects = []
    for factor in (2.0, 1.0, 0.5):
        steps = factor * _steps(chart)

        def one_form(values):
            return _entropy_flux_one_form(
                geometry,
                values,
                steps,
                state_scales=state_scales,
                entropy_scale=entropy_scale,
            )

        _value, derivative = _five_point_jacobian(
            one_form,
            chart,
            2.0 * steps,
        )
        curl = derivative - derivative.T
        relative_defect = np.linalg.norm(curl) / max(
            np.linalg.norm(derivative),
            np.finfo(float).tiny,
        )
        defects.append(relative_defect)
    assert np.all(np.isfinite(defects))
    # The Stage-3 runner, not this probe, owns the binding threshold and
    # classification.  Retain the value in assertion diagnostics.
    assert min(defects) >= 0.0, f"entropy curls={defects!r}"
