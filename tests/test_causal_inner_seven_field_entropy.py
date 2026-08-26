from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from imri_qpe.layer3_minidisk_1d.causal_inner_seven_field_entropy import (
    N_SEVEN_FIELDS,
    N_VERTICAL_EQUILIBRIUM_FIELDS,
    SEVEN_FIELD_PRIMITIVE_NAMES,
    SevenFieldEntropyNormalFormParameters,
    audit_seven_field_entropy_normal_form,
    build_seven_field_entropy_normal_form,
    reference_seven_field_entropy_parameters,
)


def test_seven_field_state_has_vertical_canonical_pair() -> None:
    assert N_SEVEN_FIELDS == 7
    assert N_VERTICAL_EQUILIBRIUM_FIELDS == 5
    assert SEVEN_FIELD_PRIMITIVE_NAMES[-2:] == (
        "log_height",
        "vertical_velocity_over_c",
    )


def test_entropy_weights_and_relaxation_scales_fail_closed() -> None:
    reference = reference_seven_field_entropy_parameters()
    with pytest.raises(ValueError, match="positive"):
        replace(reference, stress_weight=0.0)
    with pytest.raises(ValueError, match="non-negative"):
        replace(reference, vertical_damping=-1.0)


def test_height_departure_vanishes_on_vertical_equilibrium_embedding() -> None:
    normal_form = build_seven_field_entropy_normal_form(
        reference_seven_field_entropy_parameters()
    )
    transformed = (
        normal_form.chart_to_relaxation
        @ normal_form.vertical_equilibrium_embedding
    )
    expected = np.zeros((7, 5), dtype=float)
    expected[:5] = np.eye(5)
    np.testing.assert_allclose(transformed, expected, atol=1.0e-15, rtol=0.0)


def test_reference_normal_form_is_symmetric_hyperbolic_and_dissipative() -> None:
    parameters = reference_seven_field_entropy_parameters()
    normal_form = build_seven_field_entropy_normal_form(parameters)
    np.testing.assert_allclose(
        normal_form.temporal_matrix,
        normal_form.temporal_matrix.T,
        atol=1.0e-14,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        normal_form.spatial_matrix,
        normal_form.spatial_matrix.T,
        atol=1.0e-14,
        rtol=0.0,
    )
    assert np.min(np.linalg.eigvalsh(normal_form.temporal_matrix)) > 0.0
    entropy_source = 0.5 * (
        normal_form.source_matrix + normal_form.source_matrix.T
    )
    assert np.max(np.linalg.eigvalsh(entropy_source)) <= 1.0e-14

    audit = audit_seven_field_entropy_normal_form(parameters)
    assert audit.passed
    assert audit.maximum_absolute_characteristic_speed_over_c < 1.0
    assert audit.subcharacteristic_interlacing_violation <= 1.0e-14


def test_vertical_exchange_is_entropy_conservative_before_damping() -> None:
    parameters = reference_seven_field_entropy_parameters()
    normal_form = build_seven_field_entropy_normal_form(parameters)
    metric = normal_form.entropy_metric_relaxation
    generator = normal_form.source_generator_relaxation
    exchange = metric[5:7, 5:7] @ generator[5:7, 5:7]
    exchange[1, 1] = 0.0
    np.testing.assert_allclose(exchange + exchange.T, 0.0, atol=1.0e-15)


def test_random_positive_normal_forms_preserve_the_algebraic_theorem() -> None:
    random = np.random.default_rng(20260825)
    reference = reference_seven_field_entropy_parameters()
    for _ in range(32):
        weights = 0.6 + 1.8 * random.random(6)
        parameters = SevenFieldEntropyNormalFormParameters(
            transport_speed_over_c=float(random.uniform(-0.12, 0.12)),
            surface_density_weight=float(weights[0]),
            radial_velocity_weight=float(weights[1]),
            azimuthal_velocity_weight=float(weights[2]),
            thermal_weight=float(weights[3]),
            stress_weight=float(weights[4]),
            vertical_velocity_weight=float(weights[5]),
            vertical_frequency=float(random.uniform(0.5, 1.2)),
            vertical_damping=float(random.uniform(0.0, 0.4)),
            stress_relaxation_rate=float(random.uniform(0.1, 0.8)),
            height_log_surface_density=float(random.uniform(-0.4, 0.1)),
            height_log_temperature=float(random.uniform(0.4, 1.8)),
            mass_radial_coupling=reference.mass_radial_coupling,
            radial_thermal_coupling=reference.radial_thermal_coupling,
            radial_height_coupling=reference.radial_height_coupling,
            thermal_height_coupling=reference.thermal_height_coupling,
            azimuthal_stress_coupling=reference.azimuthal_stress_coupling,
        )
        audit = audit_seven_field_entropy_normal_form(parameters)
        assert audit.temporal_minimum_eigenvalue > 0.0
        assert audit.temporal_symmetry_defect <= 1.0e-14
        assert audit.spatial_symmetry_defect <= 1.0e-14
        assert audit.source_entropy_positive_part <= 1.0e-13
        assert audit.subcharacteristic_interlacing_violation <= 1.0e-13
