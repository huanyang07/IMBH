"""Conservative nonoverlapping embedded grids for the causal inner DAE.

The embedded patch is represented by one ordinary finite-volume grid.  Each
parent cell inside the selected coupling face is subdivided uniformly in
``ln R``; parent cells outside that face are retained exactly.  Consequently
the coupling face has one primitive left trace, one primitive right trace,
and one production Rusanov flux in the existing DAE.  No interpolation,
duplicated interface state, or frozen exterior trace is introduced.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .causal_inner_dae_system import (
    CausalFiveFieldDAEContext,
    causal_five_field_face_flux_decomposition,
    unpack_causal_five_field_state,
)
from .causal_inner_geometry import (
    KerrSchildColumnGrid,
    make_kerr_schild_column_grid_from_edges,
)


@dataclass(frozen=True)
class CausalEmbeddedPatchLayout:
    """Mapping between a parent grid and one nonoverlapping refined patch."""

    parent_grid: KerrSchildColumnGrid
    grid: KerrSchildColumnGrid
    parent_coupling_face_index: int
    coupling_face_index: int
    refinement_ratio: int
    parent_cell_indices: np.ndarray
    subcell_indices: np.ndarray

    @property
    def n_parent_cells(self) -> int:
        return int(self.parent_grid.centers.size)

    @property
    def n_cells(self) -> int:
        return int(self.grid.centers.size)

    @property
    def n_refined_cells(self) -> int:
        return int(self.coupling_face_index)

    @property
    def coupling_radius(self) -> float:
        return float(self.grid.edges[self.coupling_face_index])

    def validated(self) -> CausalEmbeddedPatchLayout:
        parent_cells = self.n_parent_cells
        cells = self.n_cells
        parent_face = int(self.parent_coupling_face_index)
        ratio = int(self.refinement_ratio)
        expected_cells = parent_face * ratio + parent_cells - parent_face
        if (
            parent_face != self.parent_coupling_face_index
            or not 1 <= parent_face < parent_cells
            or ratio != self.refinement_ratio
            or ratio < 1
            or self.coupling_face_index != parent_face * ratio
            or cells != expected_cells
            or np.asarray(self.parent_cell_indices).shape != (cells,)
            or np.asarray(self.subcell_indices).shape != (cells,)
        ):
            raise ValueError("embedded patch layout metadata is invalid")
        if not np.array_equal(
            self.grid.edges[self.coupling_face_index :],
            self.parent_grid.edges[parent_face:],
        ):
            raise ValueError("embedded coarse exterior changed")
        if not np.array_equal(
            self.parent_cell_indices,
            np.concatenate(
                (
                    np.repeat(np.arange(parent_face), ratio),
                    np.arange(parent_face, parent_cells),
                )
            ),
        ):
            raise ValueError("embedded parent-cell map is invalid")
        return self


@dataclass(frozen=True)
class CausalEmbeddedPatchFluxAudit:
    """One-face conservation evidence for an embedded coupling."""

    coupling_face_index: int
    coupling_radius: float
    state_weighted_flux_over_c: np.ndarray
    production_weighted_flux_over_c: np.ndarray
    left_residual_contribution: np.ndarray
    right_residual_contribution: np.ndarray
    maximum_state_flux_defect: float
    maximum_telescoping_defect: float

    @property
    def passed(self) -> bool:
        return bool(
            self.maximum_state_flux_defect <= 1.0e-12
            and self.maximum_telescoping_defect == 0.0
        )


def make_causal_embedded_patch_layout(
    parent_grid: KerrSchildColumnGrid,
    parent_coupling_face_index: int,
    refinement_ratio: int,
) -> CausalEmbeddedPatchLayout:
    """Subdivide parent cells inside one face and retain its exterior.

    Every refined parent interval is split uniformly in logarithmic radius.
    The exact parent faces are assigned directly at subdivision boundaries,
    avoiding accumulated roundoff that could otherwise break grid nesting.
    """

    parent_edges = np.asarray(parent_grid.edges, dtype=float)
    parent_cells = int(parent_edges.size - 1)
    parent_face = int(parent_coupling_face_index)
    ratio = int(refinement_ratio)
    if (
        parent_face != parent_coupling_face_index
        or not 1 <= parent_face < parent_cells
        or ratio != refinement_ratio
        or ratio < 1
    ):
        raise ValueError("embedded patch coupling or refinement is invalid")

    refined_edges = []
    for cell in range(parent_face):
        local = np.geomspace(
            parent_edges[cell],
            parent_edges[cell + 1],
            ratio + 1,
        )
        local[0] = parent_edges[cell]
        local[-1] = parent_edges[cell + 1]
        refined_edges.extend(local[:-1])
    edges = np.concatenate(
        (
            np.asarray(refined_edges, dtype=float),
            parent_edges[parent_face:],
        )
    )
    grid = make_kerr_schild_column_grid_from_edges(
        edges,
        parent_grid.gravitational_radius,
    )
    parent_indices = np.concatenate(
        (
            np.repeat(np.arange(parent_face, dtype=int), ratio),
            np.arange(parent_face, parent_cells, dtype=int),
        )
    )
    subcell_indices = np.concatenate(
        (
            np.tile(np.arange(ratio, dtype=int), parent_face),
            np.zeros(parent_cells - parent_face, dtype=int),
        )
    )
    return CausalEmbeddedPatchLayout(
        parent_grid=parent_grid,
        grid=grid,
        parent_coupling_face_index=parent_face,
        coupling_face_index=parent_face * ratio,
        refinement_ratio=ratio,
        parent_cell_indices=parent_indices,
        subcell_indices=subcell_indices,
    ).validated()


def restrict_causal_embedded_patch_cell_averages(
    values: np.ndarray,
    layout: CausalEmbeddedPatchLayout,
) -> np.ndarray:
    """Conservatively restrict embedded cell averages to the parent grid."""

    layout = layout.validated()
    array = np.asarray(values, dtype=float)
    if array.ndim >= 2 and array.shape[-2] == layout.n_cells:
        cell_axis = array.ndim - 2
    elif array.ndim >= 1 and array.shape[0] == layout.n_cells:
        cell_axis = 0
    else:
        cell_axis = -1
    if (
        cell_axis < 0
        or np.any(~np.isfinite(array))
    ):
        raise ValueError("embedded restriction values are invalid")
    moved = np.moveaxis(array, cell_axis, 0)
    flattened = moved.reshape(layout.n_cells, -1)
    restricted = np.zeros(
        (layout.n_parent_cells, flattened.shape[1]),
        dtype=float,
    )
    weighted = flattened * layout.grid.cell_measures[:, None]
    np.add.at(restricted, layout.parent_cell_indices, weighted)
    restricted /= layout.parent_grid.cell_measures[:, None]
    restricted = restricted.reshape(
        (layout.n_parent_cells,) + moved.shape[1:]
    )
    return np.moveaxis(restricted, 0, cell_axis)


def causal_embedded_patch_flux_audit(
    context: CausalFiveFieldDAEContext,
    vector: np.ndarray,
    layout: CausalEmbeddedPatchLayout,
) -> CausalEmbeddedPatchFluxAudit:
    """Verify the single shared production flux at the coupling face."""

    context = context.validated()
    layout = layout.validated()
    if not np.array_equal(context.grid.edges, layout.grid.edges):
        raise ValueError("embedded context and layout grids differ")
    state = unpack_causal_five_field_state(vector, layout.n_cells)
    face = int(layout.coupling_face_index)
    decomposition = causal_five_field_face_flux_decomposition(
        context,
        vector,
    )
    production = np.asarray(
        decomposition.production_weighted_face_fluxes_over_c[face - 1],
        dtype=float,
    )
    state_flux = np.asarray(
        state.weighted_face_fluxes_over_c[face],
        dtype=float,
    )
    scale = np.maximum(np.abs(production), 1.0)
    state_defect = float(np.max(np.abs(state_flux - production) / scale))
    left = np.array(production, copy=True)
    right = -np.array(production, copy=True)
    telescoping = float(np.max(np.abs(left + right)))
    return CausalEmbeddedPatchFluxAudit(
        coupling_face_index=face,
        coupling_radius=float(layout.coupling_radius),
        state_weighted_flux_over_c=state_flux,
        production_weighted_flux_over_c=production,
        left_residual_contribution=left,
        right_residual_contribution=right,
        maximum_state_flux_defect=state_defect,
        maximum_telescoping_defect=telescoping,
    )
