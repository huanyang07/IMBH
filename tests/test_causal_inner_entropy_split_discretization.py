import numpy as np

from imri_qpe.layer3_minidisk_1d.causal_inner_entropy_split_discretization import (
    audit_frozen_split_operators,
    build_frozen_split_operators,
    midpoint_cayley_step,
    strang_split_step,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_full_port_atlas import (
    build_full_port_atlas_anchor,
)


def _operators():
    anchor = build_full_port_atlas_anchor(
        sound_speed=2.287047318996709e9,
        temperature=4.436398409641123e6,
        proper_half_thickness=2.45860382301911e8,
        proper_vertical_frequency=8.279018646718441,
        alpha=0.1,
        shear_relaxation_time=0.06262958217858178,
        transport_speed_over_c=-0.27,
    )
    return build_frozen_split_operators(
        anchor, cell_count=7, cell_light_crossing_seconds=0.04
    )


def test_frozen_operators_are_entropy_stable_and_constant_preserving():
    assert audit_frozen_split_operators(_operators()).passed


def test_midpoint_ledgers_close_for_transport_and_source():
    operators = _operators()
    state = np.linspace(-0.21, 0.19, 77)
    for generator in (operators.transport_generator, operators.source_generator):
        step = midpoint_cayley_step(generator, state, 2e-3)
        assert step.dissipated_energy >= -1e-14
        assert step.ledger_relative_defect <= 2e-14


def test_strang_step_is_dissipative_and_closes_combined_ledger():
    step = strang_split_step(_operators(), np.linspace(-0.21, 0.19, 77), 2e-3)
    assert step.energy_after <= step.energy_before + 1e-14
    assert step.source_heat_deposit >= -1e-14
    assert step.interface_entropy_dissipation >= -1e-14
    assert step.total_ledger_relative_defect <= 3e-14
