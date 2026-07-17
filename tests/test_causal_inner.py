from __future__ import annotations

import numpy as np
import pytest

from imri_qpe.constants import C
from imri_qpe.layer3_minidisk_1d import (
    audit_causal_inner_characteristics,
    gas_radiation_adiabatic_sound_speed_squared,
    gas_radiation_relativistic_sound_speed_squared,
    special_relativistic_radial_characteristic_speeds,
)


def test_relativistic_sound_speed_recovers_cold_gas_limit() -> None:
    rho = 1.0
    temperature = 1.0e4
    newtonian = gas_radiation_adiabatic_sound_speed_squared(rho, temperature)
    relativistic = gas_radiation_relativistic_sound_speed_squared(
        rho, temperature
    )

    assert relativistic == pytest.approx(newtonian, rel=1.0e-7)


def test_relativistic_sound_speed_has_radiation_limit() -> None:
    sound_squared = gas_radiation_relativistic_sound_speed_squared(
        1.0e-10, 1.0e10
    )

    assert sound_squared / C**2 == pytest.approx(1.0 / 3.0, rel=1.0e-8)


def test_relativistic_sound_speed_is_subluminal_for_arrays() -> None:
    density = np.geomspace(1.0e-12, 1.0e2, 32)
    temperature = np.geomspace(1.0e3, 1.0e11, 32)
    sound_squared = gas_radiation_relativistic_sound_speed_squared(
        density, temperature
    )

    assert np.all(sound_squared > 0.0)
    assert np.all(sound_squared < C**2)


def test_subsonic_inflow_has_one_incoming_inner_characteristic() -> None:
    audit = audit_causal_inner_characteristics(-0.1 * C, 0.2 * C)

    assert audit.incoming_characteristics == 1
    assert not audit.causally_outgoing


def test_supersonic_inflow_has_no_incoming_inner_characteristic() -> None:
    audit = audit_causal_inner_characteristics(-0.5 * C, 0.2 * C)

    assert audit.incoming_characteristics == 0
    assert audit.causally_outgoing
    assert max(abs(value) for value in audit.characteristic_speeds) < C


def test_characteristics_reject_noncausal_inputs() -> None:
    with pytest.raises(ValueError, match="radial velocity"):
        special_relativistic_radial_characteristic_speeds(C, 0.1 * C)
    with pytest.raises(ValueError, match="sound speed"):
        special_relativistic_radial_characteristic_speeds(0.0, C)
