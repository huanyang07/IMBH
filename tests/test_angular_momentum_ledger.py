from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from imri_qpe.layer3_minidisk_1d.angular_momentum_ledger import (
    algebraic_flux_ledger,
    angular_flux,
    angular_flux_prime,
    evaluate_angular_momentum_ledger,
)


def test_angular_flux_and_product_derivative() -> None:
    assert angular_flux(10.0, 5.0, 20.0) == pytest.approx(30.0)
    assert angular_flux_prime(10.0, -2.0, 3.0, 0.4) == pytest.approx(-2.0)


@pytest.mark.parametrize(
    "wind_prime,stream_prime,l_wind,l_stream,external_torque",
    [
        (0.0, 2.0, 5.0, 5.0, 0.0),
        (2.0, 0.0, 5.0, 5.0, 0.0),
        (3.0, 5.0, 5.0, 5.0, 4.0),
    ],
)
def test_manufactured_mass_loading_ledger_closes(
    wind_prime: float,
    stream_prime: float,
    l_wind: float,
    l_stream: float,
    external_torque: float,
) -> None:
    mdot_prime = wind_prime - stream_prime
    expected_prime = mdot_prime * 5.0
    external_torque = expected_prime - (wind_prime * l_wind - stream_prime * l_stream)
    row = evaluate_angular_momentum_ledger(
        mdot=10.0,
        mdot_prime=mdot_prime,
        specific_l=5.0,
        viscous_torque=0.0,
        flux_specific_l_prime=0.0,
        wind_prime=wind_prime,
        stream_prime=stream_prime,
        l_wind=l_wind,
        l_stream=l_stream,
        external_torque=external_torque,
    )
    assert row.residual == pytest.approx(0.0, abs=1.0e-14)
    assert row.angular_flux_prime == pytest.approx(expected_prime)


def test_local_disk_carried_l_requires_viscous_loading_correction() -> None:
    params = SimpleNamespace(stream_torque_delta_l_fraction=0.0)
    state = SimpleNamespace(R=2.0, W=3.0, l=8.0, l_K=9.0)
    mdot = 10.0
    wind_prime = 1.0
    stream_prime = 3.0
    mdot_prime = wind_prime - stream_prime
    viscous_torque = 2.0 * np.pi * state.R**2 * state.W

    representation = algebraic_flux_ledger(
        0.0,
        state,
        params,
        mdot=mdot,
        mdot_prime=mdot_prime,
        wind_prime=wind_prime,
        stream_prime=stream_prime,
        closure="representation",
    )
    provisional = algebraic_flux_ledger(
        0.0,
        state,
        params,
        mdot=mdot,
        mdot_prime=mdot_prime,
        wind_prime=wind_prime,
        stream_prime=stream_prime,
        closure="local_disk_prescribed",
    )
    required = algebraic_flux_ledger(
        0.0,
        state,
        params,
        mdot=mdot,
        mdot_prime=mdot_prime,
        wind_prime=wind_prime,
        stream_prime=stream_prime,
        closure="local_disk_required",
    )

    assert representation.residual == pytest.approx(0.0, abs=1.0e-13)
    assert provisional.residual == pytest.approx(
        -mdot_prime * viscous_torque / mdot, rel=1.0e-13
    )
    assert required.residual == pytest.approx(0.0, abs=1.0e-13)
    assert required.external_torque - provisional.external_torque == pytest.approx(
        provisional.residual
    )


def test_ledger_rejects_nonpositive_mass_flux() -> None:
    with pytest.raises(ValueError, match="mdot"):
        evaluate_angular_momentum_ledger(
            mdot=0.0,
            mdot_prime=0.0,
            specific_l=1.0,
            viscous_torque=0.0,
            flux_specific_l_prime=0.0,
            wind_prime=0.0,
            stream_prime=0.0,
            l_wind=1.0,
            l_stream=1.0,
            external_torque=0.0,
        )
