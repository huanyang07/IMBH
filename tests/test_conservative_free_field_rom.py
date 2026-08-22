from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from imri_qpe.layer3_minidisk_1d.conservative_free_field_rom import (
    ConservativeCoordinateSplit,
    ConservativeHiddenAmplitudeModel,
    HiddenAmplitudeState,
    canonical_rate_basis,
    polynomial_holdout,
    relative_projection_defects,
)


def _split() -> ConservativeCoordinateSplit:
    return ConservativeCoordinateSplit(
        macro_restriction=np.asarray(((1.0, 0.0, 0.0),)),
        macro_lift=np.asarray(((1.0,), (0.0,), (0.0,))),
        hidden_dual=np.asarray(((0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
        hidden_lift=np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
    )


def test_conservative_split_roundtrips_state_and_rate() -> None:
    split = _split()
    coordinate = np.asarray((2.0, -3.0, 4.0))
    macro, hidden = split.split(coordinate)
    np.testing.assert_array_equal(split.compose(macro, hidden), coordinate)
    assert max(split.identity_defects.values()) == 0.0


def test_hidden_amplitude_model_never_projects_macro_ledger() -> None:
    model = ConservativeHiddenAmplitudeModel(
        split=_split(),
        hidden_origin=np.asarray((1.0, -1.0)),
        hidden_basis=np.asarray(((1.0,), (0.0,))),
    )
    state = HiddenAmplitudeState(
        macro=np.asarray((2.0,)),
        amplitudes=np.asarray((3.0,)),
        forcing_phase=0.25,
        mode="cold",
    )
    np.testing.assert_array_equal(model.decode(state), np.asarray((2.0, 4.0, -1.0)))
    macro_rate, amplitude_rate, unresolved = model.project_rate(
        np.asarray((5.0, 7.0, 0.0))
    )
    np.testing.assert_array_equal(macro_rate, np.asarray((5.0,)))
    np.testing.assert_array_equal(amplitude_rate, np.asarray((7.0,)))
    assert unresolved == 0.0


def test_rate_basis_and_polynomial_holdout() -> None:
    samples = np.asarray(((1.0, 0.0), (2.0, 0.0), (3.0, 0.0)))
    basis, _singular, energy = canonical_rate_basis(samples, rank=1)
    assert np.max(relative_projection_defects(samples, basis)) < 1.0e-14
    assert energy[0] == 1.0
    nodes = np.linspace(0.0, 1.0, 5)
    values = np.stack((1.0 + nodes + nodes * nodes, 2.0 - nodes), axis=1)
    heldout, predictions, relative = polynomial_holdout(
        nodes, values, np.asarray((0, 2, 4))
    )
    np.testing.assert_array_equal(heldout, np.asarray((1, 3)))
    np.testing.assert_allclose(predictions, values[heldout], atol=1.0e-14)
    assert np.max(relative) < 1.0e-14
