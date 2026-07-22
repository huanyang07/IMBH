from __future__ import annotations

import numpy as np
import pytest

from imri_qpe.constants import C
from imri_qpe.layer3_minidisk_1d import (
    make_causal_five_field_regression_context,
    make_causal_five_field_seed,
    pack_causal_five_field_state,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_moment_audit import (
    causal_five_field_moment_coordinate_ladder,
    causal_five_field_moment_coordinate_values,
    causal_mesh_coincident_moment_shells,
)


def _synthetic_schur_response(n_cells: int) -> dict:
    reduced_count = 5 * n_cells
    algebraic_count = reduced_count + 5 * (n_cells + 1)
    primitive_scales = np.linspace(0.5, 2.0, reduced_count)
    response = np.zeros((algebraic_count, reduced_count), dtype=float)
    for cell in range(n_cells):
        for component in range(5):
            column = 5 * cell + component
            response[column, column] = float(component + 1)
    face_start = reduced_count
    for face in range(n_cells + 1):
        cell = min(face, n_cells - 1)
        for component in range(5):
            row = face_start + 5 * face + component
            column = 5 * cell + component
            response[row, column] = float(face + component + 1)
    return {
        "primitive_column_scales": primitive_scales,
        "algebraic_column_scales": np.ones(algebraic_count),
        "algebraic_response_scaled": response,
    }


def _five_shell_fixture():
    n_cells = 10
    context = make_causal_five_field_regression_context(n_cells)
    state = make_causal_five_field_seed(context)
    vector = pack_causal_five_field_state(state)
    grid_edges_rg = (
        context.grid.edges / context.grid.gravitational_radius
    )
    edge_indices = np.asarray((0, 2, 4, 6, 8, 10))
    shell_edges_rg = grid_edges_rg[edge_indices]
    shape_bands = (
        (
            "luminous_test_shell",
            float(shell_edges_rg[1]),
            float(shell_edges_rg[2]),
        ),
        (
            "source_shell",
            float(shell_edges_rg[3]),
            float(shell_edges_rg[4]),
        ),
    )
    reduced = _synthetic_schur_response(n_cells)
    return context, state, vector, shell_edges_rg, shape_bands, reduced


def test_mesh_coincident_shells_return_exact_cell_masks() -> None:
    (
        context,
        _state,
        _vector,
        shell_edges_rg,
        _shape_bands,
        _reduced,
    ) = _five_shell_fixture()

    geometry = causal_mesh_coincident_moment_shells(
        context,
        shell_edges_rg,
    )

    assert geometry.shell_count == 5
    assert np.array_equal(
        geometry.edge_indices,
        np.asarray((0, 2, 4, 6, 8, 10)),
    )
    assert all(np.count_nonzero(mask) == 2 for mask in geometry.cell_masks)
    assert np.array_equal(
        np.sum(np.asarray(geometry.cell_masks), axis=0),
        np.ones(10, dtype=int),
    )


def test_moment_ladder_is_incremental_and_uses_schur_storage_rows() -> None:
    (
        context,
        state,
        vector,
        shell_edges_rg,
        shape_bands,
        reduced,
    ) = _five_shell_fixture()

    ladder = causal_five_field_moment_coordinate_ladder(
        context,
        vector,
        reduced,
        shell_edges_rg,
        shape_bands_rg=shape_bands,
    )

    assert tuple(level.coordinate_count for level in ladder.levels) == (
        15,
        20,
        25,
        30,
        34,
    )
    assert ladder.storage_semantics == (
        "instantaneous_conserved_storage_without_cumulative_vertical_work"
    )
    assert tuple(level.name for level in ladder.levels) == (
        "instantaneous_shell_mje",
        "plus_shell_mean_log_temperature",
        "plus_shell_radial_momentum",
        "plus_shell_stress_storage",
        "plus_targeted_shape_moments",
    )
    for previous, current in zip(
        ladder.levels[:-1],
        ladder.levels[1:],
        strict=True,
    ):
        assert current.coordinate_names[: previous.coordinate_count] == (
            previous.coordinate_names
        )
        assert np.array_equal(
            current.raw_constraint_matrix[: previous.coordinate_count],
            previous.raw_constraint_matrix,
        )

    base = ladder.level("instantaneous_shell_mje")
    measures = context.grid.cell_measures
    mask = ladder.geometry.cell_masks[0]
    expected_mass = float(np.sum(measures[mask] * state.conserved[mask, 0]))
    assert base.coordinate_names[:3] == (
        "shell_0_rest_mass",
        "shell_0_angular_momentum",
        "shell_0_killing_energy",
    )
    assert base.coordinate_values[0] == pytest.approx(expected_mass)
    expected_mass_row = np.zeros(50)
    mass_columns = np.arange(0, 10, 5)
    expected_mass_row[mass_columns] = measures[mask]
    assert np.array_equal(
        base.raw_constraint_matrix[0],
        expected_mass_row,
    )
    assert np.all(base.coordinate_scales > 0.0)
    assert np.all(np.isfinite(base.conditioned_constraint_matrix))


def test_value_only_ladder_exactly_matches_descriptor_ladder() -> None:
    (
        context,
        _state,
        vector,
        shell_edges_rg,
        shape_bands,
        reduced,
    ) = _five_shell_fixture()

    value_only = causal_five_field_moment_coordinate_values(
        context,
        vector,
        shell_edges_rg,
        shape_bands_rg=shape_bands,
    )
    ladder = causal_five_field_moment_coordinate_ladder(
        context,
        vector,
        reduced,
        shell_edges_rg,
        shape_bands_rg=shape_bands,
    )

    assert tuple(level.coordinate_count for level in value_only.levels) == (
        15,
        20,
        25,
        30,
        34,
    )
    assert value_only.storage_semantics == ladder.storage_semantics
    assert np.array_equal(
        value_only.geometry.edge_indices,
        ladder.geometry.edge_indices,
    )
    for expected, actual in zip(
        ladder.levels,
        value_only.levels,
        strict=True,
    ):
        assert actual.name == expected.name
        assert actual.coordinate_names == expected.coordinate_names
        assert actual.coordinate_families == expected.coordinate_families
        assert np.array_equal(
            actual.coordinate_values,
            expected.coordinate_values,
        )
        assert np.array_equal(
            actual.coordinate_scales,
            expected.coordinate_scales,
        )
    assert value_only.interface_flux_names == ladder.interface_flux_names
    assert np.array_equal(
        value_only.interface_flux_values,
        ladder.interface_flux_values,
    )
    assert np.array_equal(
        value_only.interface_flux_scales,
        ladder.interface_flux_scales,
    )
    assert value_only.level("plus_targeted_shape_moments") is (
        value_only.levels[-1]
    )
    with pytest.raises(KeyError):
        value_only.level("not_a_level")


def test_primitive_moments_act_on_scaled_tangents_and_shapes_remove_means() -> None:
    (
        context,
        _state,
        vector,
        shell_edges_rg,
        shape_bands,
        reduced,
    ) = _five_shell_fixture()
    ladder = causal_five_field_moment_coordinate_ladder(
        context,
        vector,
        reduced,
        shell_edges_rg,
        shape_bands_rg=shape_bands,
    )

    thermal = ladder.level("plus_shell_mean_log_temperature")
    thermal_row = thermal.raw_constraint_matrix[15]
    mask = ladder.geometry.cell_masks[0]
    weights = (
        context.grid.cell_measures[mask]
        / np.sum(context.grid.cell_measures[mask])
    )
    columns = 5 * np.flatnonzero(mask) + 3
    assert np.allclose(
        thermal_row[columns],
        weights * ladder.primitive_column_scales[columns],
    )

    final = ladder.level("plus_targeted_shape_moments")
    scaled_constant_temperature = np.zeros(50)
    scaled_constant_temperature[3::5] = (
        1.0 / ladder.primitive_column_scales[3::5]
    )
    scaled_constant_density = np.zeros(50)
    scaled_constant_density[0::5] = (
        1.0 / ladder.primitive_column_scales[0::5]
    )
    for name, row in zip(
        final.coordinate_names[-4:],
        final.raw_constraint_matrix[-4:],
        strict=True,
    ):
        direction = (
            scaled_constant_temperature
            if "log_temperature" in name
            else scaled_constant_density
        )
        assert abs(float(row @ direction)) < 2.0e-15


def test_interface_flux_outputs_use_physical_c_times_face_response() -> None:
    (
        context,
        state,
        vector,
        shell_edges_rg,
        shape_bands,
        reduced,
    ) = _five_shell_fixture()
    ladder = causal_five_field_moment_coordinate_ladder(
        context,
        vector,
        reduced,
        shell_edges_rg,
        shape_bands_rg=shape_bands,
    )

    assert len(ladder.interface_flux_names) == 12
    assert ladder.interface_flux_names[:3] == (
        "interface_1_rest_mass",
        "interface_1_angular_momentum",
        "interface_1_killing_energy",
    )
    first_face = ladder.geometry.edge_indices[1]
    assert ladder.interface_flux_values[0] == pytest.approx(
        C * state.weighted_face_fluxes_over_c[first_face, 0]
    )
    expected = np.zeros(50)
    expected[5 * first_face] = C * float(first_face + 1)
    assert np.array_equal(ladder.raw_interface_flux_jacobian[0], expected)
    assert np.all(ladder.interface_flux_scales > 0.0)
    assert np.all(np.isfinite(ladder.interface_flux_jacobian))


def test_moment_ladder_rejects_noncoincident_edges_and_shape_bands() -> None:
    (
        context,
        _state,
        vector,
        shell_edges_rg,
        shape_bands,
        reduced,
    ) = _five_shell_fixture()
    bad_edges = shell_edges_rg.copy()
    bad_edges[2] += 1.0e-3
    with pytest.raises(ValueError, match="not mesh coincident"):
        causal_five_field_moment_coordinate_ladder(
            context,
            vector,
            reduced,
            bad_edges,
            shape_bands_rg=shape_bands,
        )

    bad_band = (("not_a_shell", 2.0, 3.0),)
    with pytest.raises(ValueError, match="resolve to one declared shell"):
        causal_five_field_moment_coordinate_ladder(
            context,
            vector,
            reduced,
            shell_edges_rg,
            shape_bands_rg=bad_band,
        )


def test_default_nominal_shape_bands_resolve_to_snapped_shell_edges() -> None:
    n_cells = 64
    context = make_causal_five_field_regression_context(n_cells)
    state = make_causal_five_field_seed(context)
    vector = pack_causal_five_field_state(state)
    grid_edges_rg = (
        context.grid.edges / context.grid.gravitational_radius
    )
    target_indices = [
        0,
        *(
            int(np.argmin(np.abs(grid_edges_rg - target)))
            for target in (6.0, 60.0, 200.0, 280.0)
        ),
        n_cells,
    ]
    snapped_edges_rg = grid_edges_rg[np.asarray(target_indices)]
    assert not np.array_equal(
        snapped_edges_rg[1:-1],
        np.asarray((6.0, 60.0, 200.0, 280.0)),
    )

    ladder = causal_five_field_moment_coordinate_ladder(
        context,
        vector,
        _synthetic_schur_response(n_cells),
        snapped_edges_rg,
    )

    final_names = ladder.level(
        "plus_targeted_shape_moments"
    ).coordinate_names
    assert final_names[-4:] == (
        "shape_6_to_60rg_log_temperature_first",
        "shape_6_to_60rg_log_surface_density_first",
        "shape_source_shell_log_temperature_first",
        "shape_source_shell_log_surface_density_first",
    )
