"""Conservative signed-flux outer-disk evolution with independent surface density.

This module is the wind-free, nearly Keplerian bridge to a time-dependent
stream-fed minidisk.  It deliberately does not eliminate ``Sigma`` through
``Mdot/u`` and therefore admits finite-density stagnation and decretion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import diags, eye, lil_matrix
from scipy.sparse.linalg import spsolve

from .grid import RadialGrid
from .transonic_potential import PaczynskiWiitaPotential


@dataclass(frozen=True)
class SignedFluxBoundary:
    """Mass/torque conditions at the two radial boundaries."""

    inner_mode: str = "zero_torque"
    outer_mode: str = "tidal_wall"

    def __post_init__(self) -> None:
        if self.inner_mode not in {"zero_torque", "tidal_wall"}:
            raise ValueError("inner_mode must be 'zero_torque' or 'tidal_wall'")
        if self.outer_mode not in {"zero_torque", "tidal_wall"}:
            raise ValueError("outer_mode must be 'zero_torque' or 'tidal_wall'")


@dataclass(frozen=True)
class StreamInjectionState:
    """Cell-integrated mass, angular-momentum, and energy injection rates."""

    mass_rate_cells: np.ndarray
    angular_momentum_rate_cells: np.ndarray
    total_energy_rate_cells: np.ndarray

    def __post_init__(self) -> None:
        arrays = {}
        for name in (
            "mass_rate_cells",
            "angular_momentum_rate_cells",
            "total_energy_rate_cells",
        ):
            array = np.array(getattr(self, name), dtype=float, copy=True)
            if array.ndim != 1 or np.any(~np.isfinite(array)):
                raise ValueError(f"{name} must be a finite one-dimensional array")
            array.setflags(write=False)
            arrays[name] = array
        if len({array.shape for array in arrays.values()}) != 1:
            raise ValueError("stream injection arrays must have identical shapes")
        if np.any(arrays["mass_rate_cells"] < 0.0):
            raise ValueError("mass_rate_cells must be non-negative")
        for name, array in arrays.items():
            object.__setattr__(self, name, array)

    def validated_for(self, grid: RadialGrid) -> StreamInjectionState:
        if self.mass_rate_cells.shape != grid.centers.shape:
            raise ValueError("stream injection arrays must match the radial grid")
        return self


@dataclass(frozen=True)
class SignedFluxTransport:
    """Face fluxes and exact integrated budget rates for one disk state."""

    surface_density: np.ndarray
    viscosity: np.ndarray
    specific_angular_momentum: np.ndarray
    omega: np.ndarray
    viscous_torque_centers: np.ndarray
    viscous_torque_faces: np.ndarray
    mdot_faces: np.ndarray
    angular_flux_faces: np.ndarray
    source_mass_rate_cells: np.ndarray
    source_angular_rate_cells: np.ndarray
    source_total_energy_rate_cells: np.ndarray
    external_angular_rate_cells: np.ndarray
    mass_rate_cells: np.ndarray
    mass_budget_rate: float
    angular_momentum_rate_from_state: float
    angular_momentum_budget_rate: float
    angular_momentum_budget_defect: float


@dataclass(frozen=True)
class SignedFluxStepResult:
    """Accepted explicit finite-volume step without positivity clipping."""

    surface_density: np.ndarray
    transport: SignedFluxTransport
    dt: float


def _require_closed_step_angular_ledger(transport: SignedFluxTransport) -> None:
    scale = max(
        float(np.sum(np.abs(transport.source_angular_rate_cells))),
        float(np.sum(np.abs(transport.external_angular_rate_cells))),
        abs(transport.angular_momentum_budget_rate),
        abs(transport.angular_momentum_rate_from_state),
        1.0,
    )
    if abs(transport.angular_momentum_budget_defect) > 1.0e-10 * scale:
        raise ValueError(
            "time evolution with nonlocal stream angular momentum requires the "
            "coupled angular IMEX operator"
        )


def _grid_array(name: str, values, grid: RadialGrid, *, nonnegative: bool = False) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        array = np.full_like(grid.centers, float(array), dtype=float)
    if array.shape != grid.centers.shape or np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must be finite and match the radial grid")
    if nonnegative and np.any(array < 0.0):
        raise ValueError(f"{name} must be non-negative")
    return array


def normalized_stream_cell_rates(
    grid: RadialGrid,
    total_mass_rate: float,
    *,
    center: float,
    log_width: float,
    specific_angular_momentum: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exactly normalized absolute stream mass and angular cell rates."""

    if not np.isfinite(total_mass_rate) or total_mass_rate < 0.0:
        raise ValueError("total_mass_rate must be finite and non-negative")
    if not np.isfinite(center) or center <= 0.0:
        raise ValueError("stream center must be positive and finite")
    if not np.isfinite(log_width) or log_width <= 0.0:
        raise ValueError("log_width must be positive and finite")
    if specific_angular_momentum is not None and not np.isfinite(specific_angular_momentum):
        raise ValueError("specific_angular_momentum must be finite when supplied")
    coordinate = np.log(grid.centers / float(center)) / float(log_width)
    weights = np.exp(-0.5 * coordinate**2)
    weights /= np.sum(weights)
    mass = float(total_mass_rate) * weights
    angular = mass * (
        0.0 if specific_angular_momentum is None else float(specific_angular_momentum)
    )
    return mass, angular


def normalized_stream_injection_state(
    grid: RadialGrid,
    total_mass_rate: float,
    *,
    center: float,
    log_width: float,
    specific_angular_momentum: float,
    specific_total_energy: float,
) -> StreamInjectionState:
    """Return one exactly normalized, immutable stream source state."""

    mass, angular = normalized_stream_cell_rates(
        grid,
        total_mass_rate,
        center=center,
        log_width=log_width,
        specific_angular_momentum=specific_angular_momentum,
    )
    if not np.isfinite(specific_total_energy):
        raise ValueError("specific_total_energy must be finite")
    return StreamInjectionState(
        mass_rate_cells=mass,
        angular_momentum_rate_cells=angular,
        total_energy_rate_cells=mass * float(specific_total_energy),
    )


def _stream_state_from_legacy_arguments(
    grid: RadialGrid,
    specific_l: np.ndarray,
    *,
    stream_state: StreamInjectionState | None,
    source_mass_rate_cells,
    source_specific_angular_momentum,
) -> StreamInjectionState:
    if stream_state is not None:
        if (
            source_mass_rate_cells is not None
            or source_specific_angular_momentum is not None
        ):
            raise ValueError("stream_state cannot be combined with legacy source arguments")
        return stream_state.validated_for(grid)
    if source_mass_rate_cells is None:
        mass = np.zeros_like(grid.centers)
    else:
        mass = _grid_array(
            "source_mass_rate_cells", source_mass_rate_cells, grid, nonnegative=True
        )
    if source_specific_angular_momentum is None:
        angular = mass * specific_l
    else:
        source_l = _grid_array(
            "source_specific_angular_momentum",
            source_specific_angular_momentum,
            grid,
        )
        angular = mass * source_l
    return StreamInjectionState(
        mass_rate_cells=mass,
        angular_momentum_rate_cells=angular,
        total_energy_rate_cells=np.zeros_like(mass),
    )


def signed_flux_transport(
    grid: RadialGrid,
    surface_density,
    viscosity,
    M_g: float,
    *,
    boundary: SignedFluxBoundary | None = None,
    stream_state: StreamInjectionState | None = None,
    source_mass_rate_cells=None,
    source_specific_angular_momentum=None,
    external_angular_rate_cells=None,
) -> SignedFluxTransport:
    """Evaluate signed viscous transport and conservative global budgets.

    ``mdot_faces`` is inward-positive and ordered from the inner to outer
    boundary. A negative value is decretion. The nearly Keplerian closure uses
    ``Mdot=dG/dl`` at interior faces and never divides by ``Mdot`` or radial
    velocity.
    """

    sigma = _grid_array("surface_density", surface_density, grid, nonnegative=True)
    nu = _grid_array("viscosity", viscosity, grid, nonnegative=True)
    if np.any(sigma <= 0.0):
        raise ValueError("surface_density must be strictly positive")
    boundary = SignedFluxBoundary() if boundary is None else boundary
    potential = PaczynskiWiitaPotential(float(M_g))
    omega = np.asarray(potential.omega_k(grid.centers), dtype=float)
    specific_l = np.asarray(potential.l_k(grid.centers), dtype=float)
    edge_l = np.asarray(potential.l_k(grid.edges), dtype=float)
    shear = np.asarray(
        omega * potential.dln_omega_k_dlnR(grid.centers) / grid.centers,
        dtype=float,
    )
    torque = np.asarray(-2.0 * np.pi * grid.centers**3 * nu * sigma * shear, dtype=float)

    torque_faces = np.empty(grid.edges.size, dtype=float)
    torque_faces[1:-1] = 0.5 * (torque[:-1] + torque[1:])
    torque_faces[0] = 0.0 if boundary.inner_mode == "zero_torque" else torque[0]
    torque_faces[-1] = 0.0 if boundary.outer_mode == "zero_torque" else torque[-1]

    mdot_faces = np.empty(grid.edges.size, dtype=float)
    mdot_faces[1:-1] = np.diff(torque) / np.diff(specific_l)
    if boundary.inner_mode == "zero_torque":
        mdot_faces[0] = (torque[0] - torque_faces[0]) / (specific_l[0] - edge_l[0])
    else:
        mdot_faces[0] = 0.0
    if boundary.outer_mode == "zero_torque":
        mdot_faces[-1] = (torque_faces[-1] - torque[-1]) / (edge_l[-1] - specific_l[-1])
    else:
        mdot_faces[-1] = 0.0

    source = _stream_state_from_legacy_arguments(
        grid,
        specific_l,
        stream_state=stream_state,
        source_mass_rate_cells=source_mass_rate_cells,
        source_specific_angular_momentum=source_specific_angular_momentum,
    )
    if external_angular_rate_cells is None:
        external_angular = np.zeros_like(sigma)
    else:
        external_angular = _grid_array(
            "external_angular_rate_cells",
            external_angular_rate_cells,
            grid,
        )

    mass_rate = mdot_faces[1:] - mdot_faces[:-1] + source.mass_rate_cells
    angular_flux = mdot_faces * edge_l - torque_faces
    state_angular_rate = float(np.sum(mass_rate * specific_l))
    budget_angular_rate = float(
        angular_flux[-1]
        - angular_flux[0]
        + np.sum(source.angular_momentum_rate_cells + external_angular)
    )
    return SignedFluxTransport(
        surface_density=sigma,
        viscosity=nu,
        specific_angular_momentum=specific_l,
        omega=omega,
        viscous_torque_centers=torque,
        viscous_torque_faces=torque_faces,
        mdot_faces=mdot_faces,
        angular_flux_faces=angular_flux,
        source_mass_rate_cells=source.mass_rate_cells,
        source_angular_rate_cells=source.angular_momentum_rate_cells,
        source_total_energy_rate_cells=source.total_energy_rate_cells,
        external_angular_rate_cells=external_angular,
        mass_rate_cells=mass_rate,
        mass_budget_rate=float(
            mdot_faces[-1] - mdot_faces[0] + np.sum(source.mass_rate_cells)
        ),
        angular_momentum_rate_from_state=state_angular_rate,
        angular_momentum_budget_rate=budget_angular_rate,
        angular_momentum_budget_defect=float(state_angular_rate - budget_angular_rate),
    )


def advance_signed_flux_explicit(
    grid: RadialGrid,
    surface_density,
    viscosity,
    M_g: float,
    dt: float,
    **transport_kwargs,
) -> SignedFluxStepResult:
    """Advance annular masses once, rejecting rather than clipping negativity."""

    if not np.isfinite(dt) or dt < 0.0:
        raise ValueError("dt must be finite and non-negative")
    transport = signed_flux_transport(
        grid,
        surface_density,
        viscosity,
        M_g,
        **transport_kwargs,
    )
    _require_closed_step_angular_ledger(transport)
    mass = transport.surface_density * grid.area
    updated_mass = mass + float(dt) * transport.mass_rate_cells
    if np.any(updated_mass <= 0.0):
        raise ValueError("signed-flux timestep would produce non-positive annular mass")
    return SignedFluxStepResult(
        surface_density=np.asarray(updated_mass / grid.area, dtype=float),
        transport=transport,
        dt=float(dt),
    )


def signed_flux_linear_operator(
    grid: RadialGrid,
    viscosity,
    M_g: float,
    *,
    boundary: SignedFluxBoundary | None = None,
):
    """Return sparse ``A`` such that ``dSigma/dt=A Sigma`` without sources."""

    nu = _grid_array("viscosity", viscosity, grid, nonnegative=True)
    boundary = SignedFluxBoundary() if boundary is None else boundary
    potential = PaczynskiWiitaPotential(float(M_g))
    omega = np.asarray(potential.omega_k(grid.centers), dtype=float)
    specific_l = np.asarray(potential.l_k(grid.centers), dtype=float)
    edge_l = np.asarray(potential.l_k(grid.edges), dtype=float)
    shear = np.asarray(
        omega * potential.dln_omega_k_dlnR(grid.centers) / grid.centers,
        dtype=float,
    )
    torque_coefficient = -2.0 * np.pi * grid.centers**3 * nu * shear
    n = grid.centers.size
    face_from_sigma = lil_matrix((n + 1, n), dtype=float)
    if boundary.inner_mode == "zero_torque":
        face_from_sigma[0, 0] = torque_coefficient[0] / (
            specific_l[0] - edge_l[0]
        )
    for face in range(1, n):
        delta_l = specific_l[face] - specific_l[face - 1]
        face_from_sigma[face, face - 1] = -torque_coefficient[face - 1] / delta_l
        face_from_sigma[face, face] = torque_coefficient[face] / delta_l
    if boundary.outer_mode == "zero_torque":
        face_from_sigma[n, n - 1] = -torque_coefficient[n - 1] / (
            edge_l[n] - specific_l[n - 1]
        )
    divergence = lil_matrix((n, n + 1), dtype=float)
    for cell in range(n):
        divergence[cell, cell] = -1.0
        divergence[cell, cell + 1] = 1.0
    inverse_area = diags(1.0 / np.asarray(grid.area, dtype=float))
    return (inverse_area @ divergence.tocsr() @ face_from_sigma.tocsr()).tocsr()


def advance_signed_flux_implicit(
    grid: RadialGrid,
    surface_density,
    viscosity,
    M_g: float,
    dt: float,
    *,
    boundary: SignedFluxBoundary | None = None,
    source_mass_rate_cells=None,
    source_specific_angular_momentum=None,
) -> SignedFluxStepResult:
    """Advance prescribed-viscosity transport with backward Euler."""

    if not np.isfinite(dt) or dt < 0.0:
        raise ValueError("dt must be finite and non-negative")
    sigma = _grid_array("surface_density", surface_density, grid, nonnegative=True)
    if np.any(sigma <= 0.0):
        raise ValueError("surface_density must be strictly positive")
    if source_mass_rate_cells is None:
        source_mass = np.zeros_like(sigma)
    else:
        source_mass = _grid_array(
            "source_mass_rate_cells", source_mass_rate_cells, grid, nonnegative=True
        )
    operator = signed_flux_linear_operator(
        grid, viscosity, M_g, boundary=boundary
    )
    right = sigma + float(dt) * source_mass / grid.area
    updated = np.asarray(
        spsolve(eye(sigma.size, format="csr") - float(dt) * operator, right),
        dtype=float,
    )
    if np.any(~np.isfinite(updated)) or np.any(updated <= 0.0):
        raise ValueError("implicit signed-flux step produced non-positive surface density")
    transport = signed_flux_transport(
        grid,
        updated,
        viscosity,
        M_g,
        boundary=boundary,
        source_mass_rate_cells=source_mass,
        source_specific_angular_momentum=source_specific_angular_momentum,
    )
    _require_closed_step_angular_ledger(transport)
    return SignedFluxStepResult(surface_density=updated, transport=transport, dt=float(dt))


def solve_signed_flux_steady_legacy(
    grid: RadialGrid,
    viscosity,
    M_g: float,
    *,
    boundary: SignedFluxBoundary | None = None,
    source_mass_rate_cells,
    source_specific_angular_momentum=None,
) -> SignedFluxTransport:
    """Reproduce the pre-WP1 mass-only steady closure from commit 53566fa."""

    source_mass = _grid_array(
        "source_mass_rate_cells", source_mass_rate_cells, grid, nonnegative=True
    )
    if not np.any(source_mass > 0.0):
        raise ValueError("a steady supplied disk requires a nonzero mass source")
    operator = signed_flux_linear_operator(
        grid, viscosity, M_g, boundary=boundary
    )
    right = -source_mass / grid.area
    sigma = np.asarray(spsolve(operator, right), dtype=float)
    if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0.0):
        raise ValueError("steady signed-flux solve did not produce positive surface density")
    return signed_flux_transport(
        grid,
        sigma,
        viscosity,
        M_g,
        boundary=boundary,
        source_mass_rate_cells=source_mass,
        source_specific_angular_momentum=source_specific_angular_momentum,
    )


def solve_signed_flux_steady(
    grid: RadialGrid,
    viscosity,
    M_g: float,
    *,
    boundary: SignedFluxBoundary | None = None,
    stream_state: StreamInjectionState | None = None,
    source_mass_rate_cells=None,
    source_specific_angular_momentum=None,
    external_angular_rate_cells=None,
) -> SignedFluxTransport:
    """Solve steady mass and angular momentum from one conservative ledger.

    The stream is deposited at cell centers. Face mass and angular fluxes are
    integrated across each cell, so both source moments affect the solved flow.
    ``zero_torque`` is an open boundary and ``tidal_wall`` is a zero-mass-flux
    boundary whose required torque is an output.
    """

    boundary = SignedFluxBoundary() if boundary is None else boundary
    nu = _grid_array("viscosity", viscosity, grid, nonnegative=True)
    if np.any(nu <= 0.0):
        raise ValueError("viscosity must be strictly positive for a steady disk")
    potential = PaczynskiWiitaPotential(float(M_g))
    omega = np.asarray(potential.omega_k(grid.centers), dtype=float)
    specific_l = np.asarray(potential.l_k(grid.centers), dtype=float)
    edge_l = np.asarray(potential.l_k(grid.edges), dtype=float)
    source = _stream_state_from_legacy_arguments(
        grid,
        specific_l,
        stream_state=stream_state,
        source_mass_rate_cells=source_mass_rate_cells,
        source_specific_angular_momentum=source_specific_angular_momentum,
    )
    if not np.any(source.mass_rate_cells > 0.0):
        raise ValueError("a steady supplied disk requires a nonzero mass source")
    if external_angular_rate_cells is None:
        external_angular = np.zeros_like(grid.centers)
    else:
        external_angular = _grid_array(
            "external_angular_rate_cells", external_angular_rate_cells, grid
        )
    angular_injection = source.angular_momentum_rate_cells + external_angular
    total_mass = float(np.sum(source.mass_rate_cells))
    total_angular = float(np.sum(angular_injection))

    if (
        boundary.inner_mode == "tidal_wall"
        and boundary.outer_mode == "tidal_wall"
    ):
        raise ValueError("a supplied steady disk cannot have mass walls at both boundaries")
    if boundary.outer_mode == "tidal_wall":
        mdot_inner = total_mass
    elif boundary.inner_mode == "tidal_wall":
        mdot_inner = 0.0
    else:
        denominator = float(edge_l[-1] - edge_l[0])
        if denominator <= 0.0:
            raise ValueError(
                "open steady disk requires increasing boundary angular momentum"
            )
        mdot_inner = float((total_mass * edge_l[-1] - total_angular) / denominator)

    mdot_faces = np.empty(grid.edges.size, dtype=float)
    if boundary.outer_mode == "tidal_wall":
        mdot_faces[-1] = 0.0
        for cell in range(grid.centers.size - 1, -1, -1):
            mdot_faces[cell] = mdot_faces[cell + 1] + source.mass_rate_cells[cell]
    else:
        mdot_faces[0] = mdot_inner
        cumulative_mass = np.cumsum(source.mass_rate_cells)
        mdot_faces[1:] = mdot_inner - cumulative_mass

    angular_flux = np.empty(grid.edges.size, dtype=float)
    if boundary.inner_mode == "zero_torque":
        angular_flux[0] = mdot_faces[0] * edge_l[0]
        for cell in range(grid.centers.size):
            angular_flux[cell + 1] = angular_flux[cell] - angular_injection[cell]
    else:
        angular_flux[-1] = mdot_faces[-1] * edge_l[-1]
        for cell in range(grid.centers.size - 1, -1, -1):
            angular_flux[cell] = angular_flux[cell + 1] + angular_injection[cell]

    torque_faces = mdot_faces * edge_l - angular_flux
    tolerance_scale = max(
        abs(total_angular), total_mass * float(np.max(edge_l)), 1.0
    )
    if (
        boundary.inner_mode == "zero_torque"
        and abs(torque_faces[0]) > 1.0e-12 * tolerance_scale
    ):
        raise ValueError("inner zero-torque boundary did not close")
    if (
        boundary.outer_mode == "zero_torque"
        and abs(torque_faces[-1]) > 1.0e-12 * tolerance_scale
    ):
        raise ValueError("outer zero-torque boundary did not close")
    if (
        boundary.inner_mode == "tidal_wall"
        and abs(mdot_faces[0]) > 1.0e-12 * total_mass
    ):
        raise ValueError("inner tidal wall did not close")
    if (
        boundary.outer_mode == "tidal_wall"
        and abs(mdot_faces[-1]) > 1.0e-12 * total_mass
    ):
        raise ValueError("outer tidal wall did not close")

    mdot_centers = mdot_faces[:-1] - 0.5 * source.mass_rate_cells
    angular_flux_centers = angular_flux[:-1] - 0.5 * angular_injection
    torque_centers = mdot_centers * specific_l - angular_flux_centers
    shear = np.asarray(
        omega * potential.dln_omega_k_dlnR(grid.centers) / grid.centers,
        dtype=float,
    )
    torque_coefficient = -2.0 * np.pi * grid.centers**3 * nu * shear
    sigma = np.asarray(torque_centers / torque_coefficient, dtype=float)
    if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0.0):
        raise ValueError(
            "conservative steady angular ledger did not produce positive surface density"
        )

    mass_rate = mdot_faces[1:] - mdot_faces[:-1] + source.mass_rate_cells
    state_angular_rate = float(np.sum(mass_rate * specific_l))
    budget_angular_rate = float(
        angular_flux[-1] - angular_flux[0] + np.sum(angular_injection)
    )
    return SignedFluxTransport(
        surface_density=sigma,
        viscosity=nu,
        specific_angular_momentum=specific_l,
        omega=omega,
        viscous_torque_centers=torque_centers,
        viscous_torque_faces=torque_faces,
        mdot_faces=mdot_faces,
        angular_flux_faces=angular_flux,
        source_mass_rate_cells=source.mass_rate_cells,
        source_angular_rate_cells=source.angular_momentum_rate_cells,
        source_total_energy_rate_cells=source.total_energy_rate_cells,
        external_angular_rate_cells=external_angular,
        mass_rate_cells=mass_rate,
        mass_budget_rate=float(
            mdot_faces[-1] - mdot_faces[0] + np.sum(source.mass_rate_cells)
        ),
        angular_momentum_rate_from_state=state_angular_rate,
        angular_momentum_budget_rate=budget_angular_rate,
        angular_momentum_budget_defect=float(state_angular_rate - budget_angular_rate),
    )
