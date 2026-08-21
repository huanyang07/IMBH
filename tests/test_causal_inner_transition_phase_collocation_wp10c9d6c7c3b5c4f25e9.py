from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
for path in (ROOT/"src",ROOT/"scripts"):
    if str(path) not in sys.path:sys.path.insert(0,str(path))
import run_causal_inner_transition_phase_collocation_wp10c9d6c7c3b5c4f25e9 as transition


def test_transition_models_hold_out_all_odd_interior_states() -> None:
    geometry=transition._helper()._load_npz(transition.manifest.manifest_geometry_path())
    fine,coarse=transition._models(geometry["trajectory_times_seconds"],geometry["trajectory_coordinates470"])
    assert len(fine.segments)==4
    assert len(coarse.segments)==2
    assert transition.manifest.HELDOUT_INDICES==(1,3,5,7,9,11,13,15)
    assert np.max(fine.interface_value_defects())<1.0e-12


def test_transition_state_lineage_is_complete_without_propagation() -> None:
    assert transition._states().shape==(18,112,5)
    assert transition.manifest.MAXIMUM_EXACT_CONTINUOUS_RATE_CALLS==8
