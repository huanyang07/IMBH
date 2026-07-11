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


def signed_flux_transport(
    grid: RadialGrid,
    surface_density,
    viscosity,
    M_g: float,
    *,
    boundary: SignedFluxBoundary | None = None,
    source_mass_rate_cells=None,
    source_specific_angular_momentum=None,
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

    if source_mass_rate_cells is None:
        source_mass = np.zeros_like(sigma)
    else:
        source_mass = _grid_array(
            "source_mass_rate_cells", source_mass_rate_cells, grid, nonnegative=True
        )
    if source_specific_angular_momentum is None:
        source_angular = source_mass * specific_l
    else:
        source_l = _grid_array(
            "source_specific_angular_momentum",
            source_specific_angular_momentum,
            grid,
        )
        source_angular = source_mass * source_l

    mass_rate = mdot_faces[1:] - mdot_faces[:-1] + source_mass
    angular_flux = mdot_faces * edge_l - torque_faces
    state_angular_rate = float(np.sum(mass_rate * specific_l))
    budget_angular_rate = float(
        angular_flux[-1] - angular_flux[0] + np.sum(source_angular)
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
        source_mass_rate_cells=source_mass,
        source_angular_rate_cells=source_angular,
        mass_rate_cells=mass_rate,
        mass_budget_rate=float(
            mdot_faces[-1] - mdot_faces[0] + np.sum(source_mass)
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
    return SignedFluxStepResult(surface_density=updated, transport=transport, dt=float(dt))


def solve_signed_flux_steady(
    grid: RadialGrid,
    viscosity,
    M_g: float,
    *,
    boundary: SignedFluxBoundary | None = None,
    source_mass_rate_cells,
    source_specific_angular_momentum=None,
) -> SignedFluxTransport:
    """Solve the prescribed-viscosity steady mass equation exactly."""

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
