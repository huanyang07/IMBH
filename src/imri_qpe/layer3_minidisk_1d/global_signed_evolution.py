"""One-domain conservative state and flux-primary evolution primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.sparse import block_diag, csr_matrix, eye, lil_matrix

from imri_qpe.constants import C, DEFAULT_KAPPA_ES, DEFAULT_MU_MOL

from .energy_identity import enthalpy_vertical_work
from .grid import RadialGrid
from .signed_flux_common_stress import positive_edge_reconstruction
from .transonic_local import stream_annulus_shape_and_derivative
from .transonic_potential import PaczynskiWiitaPotential
from .transonic_thermo import (
    TransonicVerticalState,
    integrated_stress,
    radiative_cooling,
    vertical_state,
)


_COMPONENTS = ("mass", "radial_momentum", "angular_momentum", "total_energy")


def _finite_vector(name: str, values, size: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (size,) or np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must be a finite vector of length {size}")
    return array


@dataclass(frozen=True)
class GlobalConservativeState:
    """Cell-integrated conserved state on one radial domain."""

    mass: np.ndarray
    radial_momentum: np.ndarray
    angular_momentum: np.ndarray
    total_energy: np.ndarray

    def validated(self) -> GlobalConservativeState:
        size = np.asarray(self.mass).size
        values = {
            name: _finite_vector(name, getattr(self, name), size)
            for name in _COMPONENTS
        }
        if np.any(values["mass"] <= 0.0):
            raise ValueError("cell-integrated mass must stay positive")
        return GlobalConservativeState(**values)

    @property
    def n_cells(self) -> int:
        return int(np.asarray(self.mass).size)


@dataclass(frozen=True)
class GlobalFaceFluxes:
    """Outward-oriented face fluxes for all conserved components."""

    mass: np.ndarray
    radial_momentum: np.ndarray
    angular_momentum: np.ndarray
    total_energy: np.ndarray

    def validated_for(self, n_cells: int) -> GlobalFaceFluxes:
        values = {
            name: _finite_vector(name, getattr(self, name), n_cells + 1)
            for name in _COMPONENTS
        }
        return GlobalFaceFluxes(**values)


@dataclass(frozen=True)
class GlobalCellSources:
    """Cell-integrated source rates for all conserved components."""

    mass: np.ndarray
    radial_momentum: np.ndarray
    angular_momentum: np.ndarray
    total_energy: np.ndarray

    @classmethod
    def zeros(cls, n_cells: int) -> GlobalCellSources:
        zero = np.zeros(n_cells, dtype=float)
        return cls(*(np.array(zero, copy=True) for _ in _COMPONENTS))

    def validated_for(self, n_cells: int) -> GlobalCellSources:
        values = {
            name: _finite_vector(name, getattr(self, name), n_cells)
            for name in _COMPONENTS
        }
        return GlobalCellSources(**values)


def combine_global_cell_sources(
    *sources: GlobalCellSources,
) -> GlobalCellSources:
    """Add named conservative cell sources component by component."""

    if not sources:
        raise ValueError("at least one source is required")
    n_cells = np.asarray(sources[0].mass).size
    validated = [source.validated_for(n_cells) for source in sources]
    return GlobalCellSources(
        **{
            name: np.sum(
                np.vstack([getattr(source, name) for source in validated]),
                axis=0,
            )
            for name in _COMPONENTS
        }
    ).validated_for(n_cells)


def global_compact_stream_cell_sources(
    grid: RadialGrid,
    total_mass_rate: float,
    *,
    center: float,
    log_width: float,
    specific_radial_velocity: float,
    specific_angular_momentum: float,
    specific_total_energy: float,
    shape: str = "compact_c2",
) -> GlobalCellSources:
    """Return exact compact-source moments for one injected stream state."""

    values = {
        "total_mass_rate": total_mass_rate,
        "center": center,
        "log_width": log_width,
        "specific_radial_velocity": specific_radial_velocity,
        "specific_angular_momentum": specific_angular_momentum,
        "specific_total_energy": specific_total_energy,
    }
    if any(not np.isfinite(value) for value in values.values()):
        raise ValueError("stream source parameters must be finite")
    if total_mass_rate < 0.0 or center <= 0.0 or log_width <= 0.0:
        raise ValueError("stream mass rate and geometry must be non-negative")
    shape_name = str(shape).strip().lower()
    if shape_name not in {"compact_c2", "c2", "compact_c4", "c4"}:
        raise ValueError("global exact stream shape must be compact_c2 or compact_c4")
    outer_radius = float(grid.edges[-1])
    center_fraction = float(center) / outer_radius
    cumulative = np.array(
        [
            stream_annulus_shape_and_derivative(
                float(log_radius),
                center_fraction,
                float(log_width),
                outer_radius,
                shape=shape_name,
            )[0]
            for log_radius in np.log(grid.edges)
        ],
        dtype=float,
    )
    tolerance = 1.0e-13
    if cumulative[0] > tolerance or cumulative[-1] < 1.0 - tolerance:
        raise ValueError("compact stream support must lie fully inside the grid")
    weights = np.diff(cumulative)
    if np.any(weights < -tolerance):
        raise ValueError("compact stream cumulative profile must not decrease")
    weights = np.maximum(weights, 0.0)
    if not np.isclose(np.sum(weights), 1.0, rtol=0.0, atol=2.0e-13):
        raise ValueError("compact stream cell weights do not sum to unity")
    mass = float(total_mass_rate) * weights
    return GlobalCellSources(
        mass=mass,
        radial_momentum=mass * float(specific_radial_velocity),
        angular_momentum=mass * float(specific_angular_momentum),
        total_energy=mass * float(specific_total_energy),
    ).validated_for(grid.centers.size)


@dataclass(frozen=True)
class GlobalLedgerAudit:
    """Independent global defects for one conservative time step."""

    defects: dict[str, float]
    relative_defects: dict[str, float]

    @property
    def maximum_relative_defect(self) -> float:
        return max(self.relative_defects.values(), default=0.0)


@dataclass(frozen=True)
class GlobalPrimitiveState:
    """Primitive and one-zone thermodynamic state recovered from conservation."""

    surface_density: np.ndarray
    radial_velocity: np.ndarray
    omega: np.ndarray
    temperature: np.ndarray
    specific_total_energy: np.ndarray
    specific_internal_energy: np.ndarray
    vertical: TransonicVerticalState


@dataclass(frozen=True)
class GlobalOuterCharacteristicAudit:
    """Radial characteristic count at one domain edge."""

    radial_velocity: float
    effective_sound_speed: float
    radial_mach_number: float
    eigenvalues: tuple[float, float, float, float]
    incoming_characteristics: int


@dataclass(frozen=True)
class GlobalInnerCharacteristicProjectionAudit:
    """Linear acoustic projection used by the inner absorbing boundary."""

    incoming_amplitude_before: float
    incoming_amplitude_after: float
    outgoing_amplitude_before: float
    outgoing_amplitude_after: float
    projected_radial_velocity: float
    projected_integrated_pressure: float
    projected_temperature: float


@dataclass(frozen=True)
class GlobalInviscidProfile:
    """Smooth reconstructed inviscid fluxes and cylindrical source terms."""

    primitives: GlobalPrimitiveState
    face_fluxes: GlobalFaceFluxes
    cell_sources: GlobalCellSources
    viscous_torque_faces: np.ndarray | None = None
    vertical_work_rate_cells: np.ndarray | None = None
    inner_characteristic_projection: (
        GlobalInnerCharacteristicProjectionAudit | None
    ) = None


@dataclass(frozen=True)
class GlobalInviscidStepResult:
    """One explicit shock-capable inviscid step without hidden clipping."""

    state: GlobalConservativeState
    profile: GlobalInviscidProfile
    ledger: GlobalLedgerAudit
    accepted: bool
    dt: float
    message: str


@dataclass(frozen=True)
class GlobalImplicitStressStepResult:
    """One conservative backward-Euler alpha-stress substep."""

    state: GlobalConservativeState
    face_fluxes: GlobalFaceFluxes
    viscous_torque_faces: np.ndarray
    ledger: GlobalLedgerAudit
    accepted: bool
    dt: float
    nfev: int
    maximum_scaled_residual: float
    maximum_storage_scaled_ledger_defect: float
    message: str


@dataclass(frozen=True)
class GlobalIMEXStepResult:
    """One explicit-inviscid, implicit-stress conservative step."""

    state: GlobalConservativeState
    inviscid: GlobalInviscidStepResult
    stress: GlobalImplicitStressStepResult | None
    ledger: GlobalLedgerAudit
    accepted: bool
    dt: float
    maximum_storage_scaled_ledger_defect: float
    message: str


@dataclass(frozen=True)
class GlobalJacobianAudit:
    """Certification of one sparse physical Jacobian."""

    directions: int
    pattern_nonzeros: int
    maximum_absolute_defect: float
    maximum_relative_defect: float
    accepted: bool


@dataclass(frozen=True)
class GlobalBackwardEulerStepResult:
    """One monolithic physical-state backward-Euler step."""

    state: GlobalConservativeState
    profile: GlobalInviscidProfile
    ledger: GlobalLedgerAudit
    accepted: bool
    dt: float
    nfev: int
    maximum_scaled_residual: float
    maximum_storage_scaled_ledger_defect: float
    message: str
    jacobian_audit: GlobalJacobianAudit | None = None


@dataclass(frozen=True)
class GlobalCoolingStepResult:
    """One local backward-Euler radiative-cooling substep."""

    state: GlobalConservativeState
    cooling_rate_cells: np.ndarray
    ledger: GlobalLedgerAudit
    accepted: bool
    dt: float
    maximum_storage_scaled_ledger_defect: float
    message: str


@dataclass(frozen=True)
class GlobalFluxPrimaryLayout:
    """Exact state and residual layout for a four-field global DAE."""

    n_cells: int

    def __post_init__(self) -> None:
        if int(self.n_cells) != self.n_cells or self.n_cells < 1:
            raise ValueError("n_cells must be a positive integer")

    @property
    def differential_size(self) -> int:
        return 4 * self.n_cells

    @property
    def algebraic_size(self) -> int:
        return 4 * (self.n_cells + 1)

    @property
    def state_size(self) -> int:
        return self.differential_size + self.algebraic_size

    @property
    def residual_size(self) -> int:
        return self.state_size

    def state_slices(self) -> dict[str, slice]:
        start = 0
        slices: dict[str, slice] = {}
        for prefix, width in (
            ("cell", self.n_cells),
            ("face", self.n_cells + 1),
        ):
            for component in _COMPONENTS:
                name = f"{prefix}_{component}"
                slices[name] = slice(start, start + width)
                start += width
        if start != self.state_size:
            raise RuntimeError("global DAE layout is inconsistent")
        return slices


def global_backward_euler_jacobian_sparsity(n_cells: int) -> csr_matrix:
    """Return the nearest-neighbor primitive Jacobian pattern.

    Unknown and residual blocks are ordered as four complete cell fields. Each
    finite-volume row consumes its left and right face closures plus a local
    source, so it can depend on every primitive in the local cell and its
    immediate neighbors.
    """

    if int(n_cells) != n_cells or n_cells < 1:
        raise ValueError("n_cells must be a positive integer")
    n_cells = int(n_cells)
    pattern = lil_matrix((4 * n_cells, 4 * n_cells), dtype=np.int8)
    for residual_component in range(4):
        for cell in range(n_cells):
            row = residual_component * n_cells + cell
            for neighbor in range(max(0, cell - 1), min(n_cells, cell + 2)):
                for primitive_component in range(4):
                    column = primitive_component * n_cells + neighbor
                    pattern[row, column] = 1
    return pattern.tocsr()


def _colored_finite_difference_jacobian(
    residual,
    values: np.ndarray,
    pattern: csr_matrix,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    relative_step: float = 1.0e-6,
) -> np.ndarray:
    """Evaluate the nearest-neighbor Jacobian in twelve disjoint colors."""

    values = np.asarray(values, dtype=float)
    base = np.asarray(residual(values), dtype=float)
    n_cells = values.size // 4
    jacobian = np.zeros((base.size, values.size), dtype=float)
    columns = pattern.tocsc()
    for primitive_component in range(4):
        for cell_color in range(3):
            group = [
                primitive_component * n_cells + cell
                for cell in range(cell_color, n_cells, 3)
            ]
            if not group:
                continue
            trial = np.array(values, copy=True)
            steps: dict[int, float] = {}
            for column in group:
                step = relative_step * max(abs(float(values[column])), 1.0)
                if values[column] + step >= upper[column]:
                    step = -step
                if values[column] + step <= lower[column]:
                    raise ValueError("finite-difference state lies on both bounds")
                trial[column] += step
                steps[column] = step
            difference = np.asarray(residual(trial), dtype=float) - base
            for column in group:
                start = columns.indptr[column]
                stop = columns.indptr[column + 1]
                rows = columns.indices[start:stop]
                jacobian[rows, column] = difference[rows] / steps[column]
    return jacobian


def _colored_central_finite_difference_jacobian(
    residual,
    values: np.ndarray,
    pattern: csr_matrix,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    relative_step: float = 4.0e-6,
) -> csr_matrix:
    """Evaluate the local Jacobian with symmetric disjoint color groups."""

    values = np.asarray(values, dtype=float)
    n_cells = values.size // 4
    jacobian = lil_matrix((values.size, values.size), dtype=float)
    columns = pattern.tocsc()
    for primitive_component in range(4):
        for cell_color in range(3):
            group = [
                primitive_component * n_cells + cell
                for cell in range(cell_color, n_cells, 3)
            ]
            if not group:
                continue
            plus = np.array(values, copy=True)
            minus = np.array(values, copy=True)
            steps: dict[int, float] = {}
            for column in group:
                step = relative_step * max(abs(float(values[column])), 1.0)
                if (
                    values[column] + step >= upper[column]
                    or values[column] - step <= lower[column]
                ):
                    raise ValueError(
                        "central finite-difference state reaches a bound"
                    )
                plus[column] += step
                minus[column] -= step
                steps[column] = step
            difference = (
                np.asarray(residual(plus), dtype=float)
                - np.asarray(residual(minus), dtype=float)
            )
            for column in group:
                start = columns.indptr[column]
                stop = columns.indptr[column + 1]
                rows = columns.indices[start:stop]
                jacobian[rows, column] = (
                    difference[rows] / (2.0 * steps[column])
                )
    return jacobian.tocsr()


def _sparse_central_finite_difference_jacobian(
    residual,
    values: np.ndarray,
    pattern: csr_matrix,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    relative_step: float = 4.0e-6,
) -> csr_matrix:
    """Evaluate each local sparse column with an independent centered step."""

    values = np.asarray(values, dtype=float)
    jacobian = lil_matrix((values.size, values.size), dtype=float)
    columns = pattern.tocsc()
    for column in range(values.size):
        step = relative_step * max(abs(float(values[column])), 1.0)
        if (
            values[column] + step >= upper[column]
            or values[column] - step <= lower[column]
        ):
            raise ValueError("central finite-difference state reaches a bound")
        plus = np.array(values, copy=True)
        minus = np.array(values, copy=True)
        plus[column] += step
        minus[column] -= step
        difference = (
            np.asarray(residual(plus), dtype=float)
            - np.asarray(residual(minus), dtype=float)
        ) / (2.0 * step)
        start = columns.indptr[column]
        stop = columns.indptr[column + 1]
        rows = columns.indices[start:stop]
        jacobian[rows, column] = difference[rows]
    return jacobian.tocsr()


def _sparse_forward_finite_difference_jacobian(
    residual,
    values: np.ndarray,
    pattern: csr_matrix,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    relative_step: float = 1.0e-6,
) -> csr_matrix:
    """Match the accepted dense one-sided derivative on the local pattern."""

    values = np.asarray(values, dtype=float)
    base = np.asarray(residual(values), dtype=float)
    jacobian = lil_matrix((values.size, values.size), dtype=float)
    columns = pattern.tocsc()
    for column in range(values.size):
        step = relative_step * max(abs(float(values[column])), 1.0)
        if values[column] + step >= upper[column]:
            step = -step
        if values[column] + step <= lower[column]:
            raise ValueError("forward finite-difference state reaches a bound")
        trial = np.array(values, copy=True)
        trial[column] += step
        difference = (np.asarray(residual(trial), dtype=float) - base) / step
        start = columns.indptr[column]
        stop = columns.indptr[column + 1]
        rows = columns.indices[start:stop]
        jacobian[rows, column] = difference[rows]
    return jacobian.tocsr()


def _dense_forward_finite_difference_jacobian(
    residual,
    values: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    relative_step: float = 1.0e-6,
) -> np.ndarray:
    """Evaluate all one-sided columns using the production perturbation."""

    values = np.asarray(values, dtype=float)
    base = np.asarray(residual(values), dtype=float)
    jacobian = np.empty((values.size, values.size), dtype=float)
    for column in range(values.size):
        step = relative_step * max(abs(float(values[column])), 1.0)
        if values[column] + step >= upper[column]:
            step = -step
        if values[column] + step <= lower[column]:
            raise ValueError("forward finite-difference state reaches a bound")
        trial = np.array(values, copy=True)
        trial[column] += step
        jacobian[:, column] = (
            np.asarray(residual(trial), dtype=float) - base
        ) / step
    return jacobian


def _audit_sparse_pattern_against_dense_columns(
    dense_jacobian: np.ndarray,
    pattern: csr_matrix,
    *,
    relative_tolerance: float = 1.0e-10,
) -> tuple[csr_matrix, GlobalJacobianAudit]:
    """Certify locality by measuring every derivative outside the pattern."""

    dense_jacobian = np.asarray(dense_jacobian, dtype=float)
    allowed = pattern.toarray().astype(bool)
    sparse_values = np.where(allowed, dense_jacobian, 0.0)
    omitted = np.where(allowed, 0.0, dense_jacobian)
    maximum_absolute = float(np.max(np.abs(omitted)))
    scale = max(float(np.max(np.abs(dense_jacobian))), 1.0e-14)
    maximum_relative = maximum_absolute / scale
    sparse = csr_matrix(sparse_values)
    return sparse, GlobalJacobianAudit(
        directions=int(dense_jacobian.shape[1]),
        pattern_nonzeros=int(sparse.nnz),
        maximum_absolute_defect=maximum_absolute,
        maximum_relative_defect=maximum_relative,
        accepted=bool(maximum_relative <= relative_tolerance),
    )


def _audit_sparse_jacobian_directions(
    residual,
    values: np.ndarray,
    jacobian: csr_matrix,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    directions: int = 3,
    relative_step: float = 4.0e-6,
    relative_tolerance: float = 3.0e-5,
) -> GlobalJacobianAudit:
    """Compare sparse products with independent centered directional slopes."""

    values = np.asarray(values, dtype=float)
    maximum_absolute = 0.0
    maximum_relative = 0.0
    coordinates = np.arange(values.size, dtype=float) + 1.0
    for index in range(directions):
        direction = np.sin((index + 1.0) * coordinates)
        direction += 0.5 * np.cos((index + 2.0) * coordinates)
        direction /= max(float(np.max(np.abs(direction))), 1.0)
        step = relative_step
        positive = direction > 0.0
        negative = direction < 0.0
        if np.any(positive & np.isfinite(upper)):
            step = min(
                step,
                0.45
                * float(
                    np.min(
                        (upper[positive & np.isfinite(upper)]
                        - values[positive & np.isfinite(upper)])
                        / direction[positive & np.isfinite(upper)]
                    )
                ),
            )
        if np.any(negative & np.isfinite(lower)):
            step = min(
                step,
                0.45
                * float(
                    np.min(
                        (values[negative & np.isfinite(lower)]
                        - lower[negative & np.isfinite(lower)])
                        / -direction[negative & np.isfinite(lower)]
                    )
                ),
            )
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("directional finite-difference step is invalid")
        finite_difference = (
            np.asarray(residual(values + step * direction), dtype=float)
            - np.asarray(residual(values - step * direction), dtype=float)
        ) / (2.0 * step)
        product = np.asarray(jacobian @ direction, dtype=float)
        defect = product - finite_difference
        absolute = float(np.max(np.abs(defect)))
        scale = max(
            float(np.max(np.abs(product))),
            float(np.max(np.abs(finite_difference))),
            1.0e-14,
        )
        maximum_absolute = max(maximum_absolute, absolute)
        maximum_relative = max(maximum_relative, absolute / scale)
    return GlobalJacobianAudit(
        directions=int(directions),
        pattern_nonzeros=int(jacobian.nnz),
        maximum_absolute_defect=maximum_absolute,
        maximum_relative_defect=maximum_relative,
        accepted=bool(maximum_relative <= relative_tolerance),
    )


def state_from_primitives(
    grid: RadialGrid,
    surface_density,
    radial_velocity,
    omega,
    specific_total_energy,
) -> GlobalConservativeState:
    """Construct integrated conserved variables from cell primitives."""

    n_cells = grid.centers.size
    sigma = _finite_vector("surface_density", surface_density, n_cells)
    velocity = _finite_vector("radial_velocity", radial_velocity, n_cells)
    rotation = _finite_vector("omega", omega, n_cells)
    energy = _finite_vector(
        "specific_total_energy", specific_total_energy, n_cells
    )
    if np.any(sigma <= 0.0):
        raise ValueError("surface_density must be positive")
    mass = grid.area * sigma
    return GlobalConservativeState(
        mass=mass,
        radial_momentum=mass * velocity,
        angular_momentum=mass * grid.centers**2 * rotation,
        total_energy=mass * energy,
    ).validated()


def state_from_thermodynamic_primitives(
    grid: RadialGrid,
    surface_density,
    radial_velocity,
    omega,
    temperature,
    M_g: float,
    *,
    mu_mol: float = DEFAULT_MU_MOL,
    kappa: float = DEFAULT_KAPPA_ES,
    gamma_gas: float = 5.0 / 3.0,
    specific_mechanical_energy_correction=None,
) -> GlobalConservativeState:
    """Construct total energy with the shared potential and vertical closure."""

    n_cells = grid.centers.size
    sigma = _finite_vector("surface_density", surface_density, n_cells)
    velocity = _finite_vector("radial_velocity", radial_velocity, n_cells)
    rotation = _finite_vector("omega", omega, n_cells)
    temperature = _finite_vector("temperature", temperature, n_cells)
    potential = PaczynskiWiitaPotential(float(M_g))
    vertical = vertical_state(
        sigma,
        temperature,
        grid.centers,
        potential,
        mu_mol=mu_mol,
        kappa=kappa,
        gamma_gas=gamma_gas,
    )
    mechanical_correction = (
        np.zeros(n_cells, dtype=float)
        if specific_mechanical_energy_correction is None
        else _finite_vector(
            "specific_mechanical_energy_correction",
            specific_mechanical_energy_correction,
            n_cells,
        )
    )
    specific_total = (
        np.asarray(potential.phi(grid.centers), dtype=float)
        + 0.5 * velocity**2
        + 0.5 * (grid.centers * rotation) ** 2
        + mechanical_correction
        + np.asarray(vertical.e, dtype=float)
    )
    return state_from_primitives(
        grid, sigma, velocity, rotation, specific_total
    )


def recover_global_primitives(
    grid: RadialGrid,
    state: GlobalConservativeState,
    M_g: float,
    *,
    temperature_bounds: tuple[float, float] = (1.0e3, 1.0e12),
    mu_mol: float = DEFAULT_MU_MOL,
    kappa: float = DEFAULT_KAPPA_ES,
    gamma_gas: float = 5.0 / 3.0,
    specific_mechanical_energy_correction=None,
) -> GlobalPrimitiveState:
    """Recover a positive thermal state without dividing by radial mass flux."""

    state = state.validated()
    if state.n_cells != grid.centers.size:
        raise ValueError("global state and radial grid have different sizes")
    lower_temperature, upper_temperature = map(float, temperature_bounds)
    if not 0.0 < lower_temperature < upper_temperature:
        raise ValueError("temperature bounds must be positive and ordered")
    potential = PaczynskiWiitaPotential(float(M_g))
    sigma = state.mass / grid.area
    radial_velocity = state.radial_momentum / state.mass
    omega = state.angular_momentum / (state.mass * grid.centers**2)
    specific_total = state.total_energy / state.mass
    mechanical_correction = (
        np.zeros(state.n_cells, dtype=float)
        if specific_mechanical_energy_correction is None
        else _finite_vector(
            "specific_mechanical_energy_correction",
            specific_mechanical_energy_correction,
            state.n_cells,
        )
    )
    target_internal = specific_total - (
        np.asarray(potential.phi(grid.centers), dtype=float)
        + 0.5 * radial_velocity**2
        + 0.5 * (grid.centers * omega) ** 2
        + mechanical_correction
    )
    if np.any(~np.isfinite(target_internal)):
        raise ValueError("recovered internal energy target is not finite")

    def internal_energy(index: int, log_temperature: float) -> float:
        local = vertical_state(
            float(sigma[index]),
            float(np.exp(log_temperature)),
            float(grid.centers[index]),
            potential,
            mu_mol=mu_mol,
            kappa=kappa,
            gamma_gas=gamma_gas,
        )
        return float(local.e)

    log_lower = float(np.log(lower_temperature))
    log_upper = float(np.log(upper_temperature))
    temperature = np.empty(state.n_cells, dtype=float)
    for index in range(state.n_cells):
        lower_residual = internal_energy(index, log_lower) - target_internal[index]
        upper_residual = internal_energy(index, log_upper) - target_internal[index]
        if lower_residual > 0.0 or upper_residual < 0.0:
            raise ValueError(
                f"cell {index} internal energy lies outside temperature bounds"
            )
        root = brentq(
            lambda value: internal_energy(index, value) - target_internal[index],
            log_lower,
            log_upper,
            xtol=1.0e-12,
            rtol=1.0e-12,
        )
        temperature[index] = np.exp(root)
    vertical = vertical_state(
        sigma,
        temperature,
        grid.centers,
        potential,
        mu_mol=mu_mol,
        kappa=kappa,
        gamma_gas=gamma_gas,
    )
    return GlobalPrimitiveState(
        surface_density=np.asarray(sigma, dtype=float),
        radial_velocity=np.asarray(radial_velocity, dtype=float),
        omega=np.asarray(omega, dtype=float),
        temperature=temperature,
        specific_total_energy=np.asarray(specific_total, dtype=float),
        specific_internal_energy=np.asarray(vertical.e, dtype=float),
        vertical=vertical,
    )


def _signed_edge_reconstruction(grid: RadialGrid, values) -> np.ndarray:
    values = _finite_vector("signed cell values", values, grid.centers.size)
    log_centers = np.log(grid.centers)
    log_edges = np.log(grid.edges)
    edges = np.interp(log_edges, log_centers, values)
    if values.size > 1:
        left_slope = (values[1] - values[0]) / (
            log_centers[1] - log_centers[0]
        )
        right_slope = (values[-1] - values[-2]) / (
            log_centers[-1] - log_centers[-2]
        )
        edges[0] = values[0] + left_slope * (log_edges[0] - log_centers[0])
        edges[-1] = values[-1] + right_slope * (
            log_edges[-1] - log_centers[-1]
        )
    return np.asarray(edges, dtype=float)


def global_inviscid_face_fluxes(
    grid: RadialGrid,
    primitives: GlobalPrimitiveState,
    M_g: float,
) -> GlobalFaceFluxes:
    """Return smooth outward Euler fluxes reconstructed at physical faces."""

    potential = PaczynskiWiitaPotential(float(M_g))
    sigma_faces = positive_edge_reconstruction(
        grid, primitives.surface_density
    )
    velocity_faces = _signed_edge_reconstruction(
        grid, primitives.radial_velocity
    )
    omega_faces = positive_edge_reconstruction(grid, primitives.omega)
    pressure_faces = positive_edge_reconstruction(
        grid, np.asarray(primitives.vertical.Pi, dtype=float)
    )
    internal_energy_faces = positive_edge_reconstruction(
        grid, primitives.specific_internal_energy
    )
    radius = grid.edges
    mass_flux = 2.0 * np.pi * radius * sigma_faces * velocity_faces
    specific_l = radius**2 * omega_faces
    specific_total = (
        np.asarray(potential.phi(radius), dtype=float)
        + 0.5 * velocity_faces**2
        + 0.5 * (radius * omega_faces) ** 2
        + internal_energy_faces
    )
    bernoulli = specific_total + pressure_faces / sigma_faces
    return GlobalFaceFluxes(
        mass=np.asarray(mass_flux, dtype=float),
        radial_momentum=np.asarray(
            mass_flux * velocity_faces + 2.0 * np.pi * radius * pressure_faces,
            dtype=float,
        ),
        angular_momentum=np.asarray(mass_flux * specific_l, dtype=float),
        total_energy=np.asarray(mass_flux * bernoulli, dtype=float),
    ).validated_for(grid.centers.size)


def apply_global_conserved_donor_outer_flux(
    grid: RadialGrid,
    fluxes: GlobalFaceFluxes,
    primitives: GlobalPrimitiveState,
) -> GlobalFaceFluxes:
    """Use one donor mass flux for every advected outer-face quantity.

    The complete cylindrical mass flux is reconstructed as a conserved object
    from the final cell rather than as a product of independently extrapolated
    surface density and radial velocity. The donor specific state is then
    carried by that same mass flux. Viscous torque and torque work are added by
    the shared stress operator after this boundary reconstruction.
    """

    fluxes = fluxes.validated_for(grid.centers.size)
    sigma = np.asarray(primitives.surface_density, dtype=float)
    velocity = np.asarray(primitives.radial_velocity, dtype=float)
    omega = np.asarray(primitives.omega, dtype=float)
    integrated_pressure = np.asarray(primitives.vertical.Pi, dtype=float)
    if any(values.shape != grid.centers.shape for values in (
        sigma,
        velocity,
        omega,
        integrated_pressure,
    )):
        raise ValueError("outer donor primitives must match the radial grid")
    cell = -1
    mass_flux = float(
        2.0
        * np.pi
        * grid.centers[cell]
        * sigma[cell]
        * velocity[cell]
    )
    specific_l = float(grid.centers[cell] ** 2 * omega[cell])
    bernoulli = float(
        primitives.specific_total_energy[cell]
        + integrated_pressure[cell] / sigma[cell]
    )
    values = {
        name: np.array(getattr(fluxes, name), copy=True)
        for name in _COMPONENTS
    }
    values["mass"][-1] = mass_flux
    values["radial_momentum"][-1] = (
        mass_flux * velocity[cell]
        + 2.0 * np.pi * grid.edges[-1] * integrated_pressure[cell]
    )
    values["angular_momentum"][-1] = mass_flux * specific_l
    values["total_energy"][-1] = mass_flux * bernoulli
    return GlobalFaceFluxes(**values).validated_for(grid.centers.size)


def global_vertical_work_rate_cells(
    grid: RadialGrid,
    outward_mass_flux_faces,
    primitives: GlobalPrimitiveState,
) -> np.ndarray:
    """Return radial one-zone column work paired with enthalpy transport.

    ``enthalpy_vertical_work`` uses the legacy inward-positive accretion rate,
    while this module orients every face flux outward.  The sign conversion is
    therefore explicit here.  Surface-density and midplane-density increments
    use the same positive edge reconstruction as the physical flux closure.
    """

    mass_flux = _finite_vector(
        "outward_mass_flux_faces",
        outward_mass_flux_faces,
        grid.centers.size + 1,
    )
    sigma = _finite_vector(
        "surface_density",
        primitives.surface_density,
        grid.centers.size,
    )
    rho = _finite_vector(
        "density", primitives.vertical.rho, grid.centers.size
    )
    if np.any(sigma <= 0.0) or np.any(rho <= 0.0):
        raise ValueError("surface density and density must be positive")
    inward_mdot_centers = -0.5 * (mass_flux[:-1] + mass_flux[1:])
    sigma_edges = positive_edge_reconstruction(grid, sigma)
    rho_edges = positive_edge_reconstruction(grid, rho)
    work = enthalpy_vertical_work(
        inward_mdot_centers,
        sigma,
        primitives.vertical.Pi,
        sigma_edges[1:] - sigma_edges[:-1],
        primitives.vertical.P_tot,
        rho,
        rho_edges[1:] - rho_edges[:-1],
    )
    return _finite_vector(
        "vertical_work_rate_cells", work, grid.centers.size
    )


def global_temporal_vertical_work_cells(
    old_state: GlobalConservativeState,
    old_primitives: GlobalPrimitiveState,
    new_state: GlobalConservativeState,
    new_primitives: GlobalPrimitiveState,
) -> np.ndarray:
    """Trapezoid the column work ``M*(Pi/Sigma)*dln(H)`` over one step."""

    old_state = old_state.validated()
    new_state = new_state.validated()
    if old_state.n_cells != new_state.n_cells:
        raise ValueError("old and new states must share one mesh")
    size = old_state.n_cells
    old_sigma = _finite_vector(
        "old surface density", old_primitives.surface_density, size
    )
    new_sigma = _finite_vector(
        "new surface density", new_primitives.surface_density, size
    )
    old_H = _finite_vector("old H", old_primitives.vertical.H, size)
    new_H = _finite_vector("new H", new_primitives.vertical.H, size)
    if np.any(old_H <= 0.0) or np.any(new_H <= 0.0):
        raise ValueError("old and new H must be positive")
    old_enthalpy = _finite_vector(
        "old column enthalpy", old_primitives.vertical.Pi / old_sigma, size
    )
    new_enthalpy = _finite_vector(
        "new column enthalpy", new_primitives.vertical.Pi / new_sigma, size
    )
    return np.asarray(
        0.5
        * (
            old_state.mass * old_enthalpy
            + new_state.mass * new_enthalpy
        )
        * np.log(new_H / old_H),
        dtype=float,
    )


def global_alpha_stress_torque_faces(
    grid: RadialGrid,
    primitives: GlobalPrimitiveState,
    *,
    alpha: float,
    mu_stress: float = 0.0,
    stress_factor: float = 1.0,
    boundary_mode: str = "extrapolated",
) -> np.ndarray:
    """Return outward viscous torque from the shared alpha-stress closure."""

    if not np.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be non-negative and finite")
    if alpha == 0.0:
        return np.zeros(grid.centers.size + 1, dtype=float)
    stress = np.asarray(
        integrated_stress(
            primitives.vertical,
            alpha,
            mu_stress=mu_stress,
            stress_factor=stress_factor,
        ),
        dtype=float,
    )
    torque_centers = 2.0 * np.pi * grid.centers**2 * stress
    torque_faces = positive_edge_reconstruction(grid, torque_centers)
    if boundary_mode == "zero_torque":
        torque_faces[[0, -1]] = 0.0
    elif boundary_mode == "outer_zero_torque":
        torque_faces[-1] = 0.0
    elif boundary_mode == "inner_zero_torque":
        torque_faces[0] = 0.0
    elif boundary_mode != "extrapolated":
        raise ValueError(
            "stress boundary_mode must be extrapolated, zero_torque, "
            "outer_zero_torque, or inner_zero_torque"
        )
    return np.asarray(torque_faces, dtype=float)


def add_global_alpha_stress_fluxes(
    grid: RadialGrid,
    fluxes: GlobalFaceFluxes,
    primitives: GlobalPrimitiveState,
    *,
    alpha: float,
    mu_stress: float = 0.0,
    stress_factor: float = 1.0,
    boundary_mode: str = "extrapolated",
) -> tuple[GlobalFaceFluxes, np.ndarray]:
    """Add paired angular and torque-work fluxes in outward orientation.

    The older steady modules use inward-positive fluxes ``Mdot*l-G`` and
    ``Mdot*B-Omega*G``.  These global face fluxes are outward positive, so the
    same physical transport enters as ``+G`` and ``+Omega*G``.
    """

    fluxes = fluxes.validated_for(grid.centers.size)
    torque = global_alpha_stress_torque_faces(
        grid,
        primitives,
        alpha=alpha,
        mu_stress=mu_stress,
        stress_factor=stress_factor,
        boundary_mode=boundary_mode,
    )
    omega_faces = positive_edge_reconstruction(grid, primitives.omega)
    return (
        GlobalFaceFluxes(
            mass=np.array(fluxes.mass, copy=True),
            radial_momentum=np.array(fluxes.radial_momentum, copy=True),
            angular_momentum=np.asarray(
                fluxes.angular_momentum + torque, dtype=float
            ),
            total_energy=np.asarray(
                fluxes.total_energy + omega_faces * torque, dtype=float
            ),
        ).validated_for(grid.centers.size),
        torque,
    )


def global_effective_sound_speed(
    primitives: GlobalPrimitiveState,
    *,
    gamma_gas: float = 5.0 / 3.0,
) -> np.ndarray:
    """Return the gas-radiation acoustic speed used by the Rusanov flux."""

    if gamma_gas <= 1.0:
        raise ValueError("gamma_gas must exceed one")
    pressure = (
        gamma_gas * np.asarray(primitives.vertical.P_gas, dtype=float)
        + (4.0 / 3.0) * np.asarray(primitives.vertical.P_rad, dtype=float)
    )
    density = np.asarray(primitives.vertical.rho, dtype=float)
    sound_speed_squared = pressure / density
    if np.any(~np.isfinite(sound_speed_squared)) or np.any(
        sound_speed_squared <= 0.0
    ):
        raise ValueError("effective sound speed is not physical")
    return np.sqrt(sound_speed_squared)


def global_outer_characteristic_audit(
    primitives: GlobalPrimitiveState,
    *,
    gamma_gas: float = 5.0 / 3.0,
) -> GlobalOuterCharacteristicAudit:
    """Count characteristics entering through the outer radial boundary.

    Positive speeds point out of the modeled domain.  The four-field radial
    Euler block has acoustic speeds ``v-c`` and ``v+c`` plus two advected
    speeds ``v``.  A negative eigenvalue therefore requires exterior data.
    """

    velocity = _finite_vector(
        "radial velocity",
        primitives.radial_velocity,
        np.asarray(primitives.radial_velocity).size,
    )
    if velocity.size < 1:
        raise ValueError("characteristic audit requires at least one cell")
    sound_speed = global_effective_sound_speed(
        primitives, gamma_gas=gamma_gas
    )
    outer_velocity = float(velocity[-1])
    outer_sound_speed = float(sound_speed[-1])
    eigenvalues = (
        outer_velocity - outer_sound_speed,
        outer_velocity,
        outer_velocity,
        outer_velocity + outer_sound_speed,
    )
    return GlobalOuterCharacteristicAudit(
        radial_velocity=outer_velocity,
        effective_sound_speed=outer_sound_speed,
        radial_mach_number=outer_velocity / outer_sound_speed,
        eigenvalues=eigenvalues,
        incoming_characteristics=sum(value < 0.0 for value in eigenvalues),
    )


def global_inner_characteristic_audit(
    primitives: GlobalPrimitiveState,
    *,
    gamma_gas: float = 5.0 / 3.0,
) -> GlobalOuterCharacteristicAudit:
    """Count radial characteristics entering through the inner boundary.

    Positive coordinate speeds travel from the inner ghost region into the
    modeled domain. A causally outgoing plunge therefore has no positive
    eigenvalues.
    """

    velocity = _finite_vector(
        "radial velocity",
        primitives.radial_velocity,
        np.asarray(primitives.radial_velocity).size,
    )
    if velocity.size < 1:
        raise ValueError("characteristic audit requires at least one cell")
    sound_speed = global_effective_sound_speed(
        primitives, gamma_gas=gamma_gas
    )
    inner_velocity = float(velocity[0])
    inner_sound_speed = float(sound_speed[0])
    eigenvalues = (
        inner_velocity - inner_sound_speed,
        inner_velocity,
        inner_velocity,
        inner_velocity + inner_sound_speed,
    )
    return GlobalOuterCharacteristicAudit(
        radial_velocity=inner_velocity,
        effective_sound_speed=inner_sound_speed,
        radial_mach_number=inner_velocity / inner_sound_speed,
        eigenvalues=eigenvalues,
        incoming_characteristics=sum(value > 0.0 for value in eigenvalues),
    )


def _single_cell_inviscid_flux(
    grid: RadialGrid,
    primitives: GlobalPrimitiveState,
    *,
    cell: int,
    face: int,
) -> np.ndarray:
    radius = float(grid.edges[face])
    sigma = float(primitives.surface_density[cell])
    velocity = float(primitives.radial_velocity[cell])
    omega = float(primitives.omega[cell])
    integrated_pressure = float(
        np.asarray(primitives.vertical.Pi, dtype=float).reshape(-1)[cell]
    )
    mass_flux = 2.0 * np.pi * radius * sigma * velocity
    specific_l = float(grid.centers[cell] ** 2 * omega)
    bernoulli = float(
        primitives.specific_total_energy[cell]
        + integrated_pressure / sigma
    )
    return np.asarray(
        [
            mass_flux,
            mass_flux * velocity + 2.0 * np.pi * radius * integrated_pressure,
            mass_flux * specific_l,
            mass_flux * bernoulli,
        ],
        dtype=float,
    )


def apply_global_reference_characteristic_inner_flux(
    grid: RadialGrid,
    fluxes: GlobalFaceFluxes,
    primitives: GlobalPrimitiveState,
    reference_primitives: GlobalPrimitiveState,
    M_g: float,
    *,
    temperature_bounds: tuple[float, float] = (1.0e3, 1.0e12),
    mu_mol: float = DEFAULT_MU_MOL,
    kappa: float = DEFAULT_KAPPA_ES,
    gamma_gas: float = 5.0 / 3.0,
) -> tuple[GlobalFaceFluxes, GlobalInnerCharacteristicProjectionAudit]:
    """Remove the incoming inner acoustic perturbation relative to a reference.

    The outgoing acoustic invariant and the two inward-advected/contact fields
    are inherited from the interior. Only the incoming ``v+c`` perturbation is
    set to zero. The correction is applied to the existing well-balanced face
    flux, so the supplied transonic reference remains exactly unchanged.
    """

    fluxes = fluxes.validated_for(grid.centers.size)
    cell = 0
    sigma = float(primitives.surface_density[cell])
    velocity = float(primitives.radial_velocity[cell])
    pressure = float(primitives.vertical.Pi[cell])
    reference_sigma = float(reference_primitives.surface_density[cell])
    reference_velocity = float(reference_primitives.radial_velocity[cell])
    reference_pressure = float(reference_primitives.vertical.Pi[cell])
    reference_sound = float(
        global_effective_sound_speed(
            reference_primitives, gamma_gas=gamma_gas
        )[cell]
    )
    velocity_perturbation = velocity - reference_velocity
    pressure_velocity = (
        pressure - reference_pressure
    ) / (reference_sigma * reference_sound)
    incoming_before = velocity_perturbation + pressure_velocity
    outgoing_before = velocity_perturbation - pressure_velocity
    projected_velocity = velocity - 0.5 * incoming_before
    projected_pressure = (
        pressure
        - 0.5
        * reference_sigma
        * reference_sound
        * incoming_before
    )
    if incoming_before == 0.0:
        audit = GlobalInnerCharacteristicProjectionAudit(
            incoming_amplitude_before=0.0,
            incoming_amplitude_after=0.0,
            outgoing_amplitude_before=float(outgoing_before),
            outgoing_amplitude_after=float(outgoing_before),
            projected_radial_velocity=velocity,
            projected_integrated_pressure=pressure,
            projected_temperature=float(primitives.temperature[cell]),
        )
        return fluxes, audit
    if not np.isfinite(projected_pressure) or projected_pressure <= 0.0:
        raise ValueError("inner characteristic projection gives nonpositive pressure")

    potential = PaczynskiWiitaPotential(float(M_g))
    radius = float(grid.centers[cell])
    lower_temperature, upper_temperature = map(float, temperature_bounds)

    def pressure_residual(log_temperature: float) -> float:
        local = vertical_state(
            sigma,
            float(np.exp(log_temperature)),
            radius,
            potential,
            mu_mol=mu_mol,
            kappa=kappa,
            gamma_gas=gamma_gas,
        )
        return float(local.Pi - projected_pressure)

    log_lower = float(np.log(lower_temperature))
    log_upper = float(np.log(upper_temperature))
    if pressure_residual(log_lower) > 0.0 or pressure_residual(log_upper) < 0.0:
        raise ValueError(
            "inner characteristic pressure lies outside temperature bounds"
        )
    projected_temperature = float(
        np.exp(
            brentq(
                pressure_residual,
                log_lower,
                log_upper,
                xtol=1.0e-12,
                rtol=1.0e-12,
            )
        )
    )
    projected_vertical = vertical_state(
        sigma,
        projected_temperature,
        radius,
        potential,
        mu_mol=mu_mol,
        kappa=kappa,
        gamma_gas=gamma_gas,
    )
    projected_specific_total = float(
        potential.phi(radius)
        + 0.5 * projected_velocity**2
        + 0.5 * (radius * float(primitives.omega[cell])) ** 2
        + projected_vertical.e
    )
    projected_primitives = GlobalPrimitiveState(
        surface_density=np.asarray([sigma]),
        radial_velocity=np.asarray([projected_velocity]),
        omega=np.asarray([float(primitives.omega[cell])]),
        temperature=np.asarray([projected_temperature]),
        specific_total_energy=np.asarray([projected_specific_total]),
        specific_internal_energy=np.asarray([float(projected_vertical.e)]),
        vertical=projected_vertical,
    )
    projected_flux = _single_cell_inviscid_flux(
        grid, projected_primitives, cell=0, face=0
    )
    current_flux = _single_cell_inviscid_flux(
        grid, primitives, cell=cell, face=0
    )
    values = {
        name: np.array(getattr(fluxes, name), copy=True)
        for name in _COMPONENTS
    }
    for index, name in enumerate(_COMPONENTS):
        values[name][0] += projected_flux[index] - current_flux[index]

    projected_velocity_perturbation = (
        projected_velocity - reference_velocity
    )
    projected_pressure_velocity = (
        float(projected_vertical.Pi) - reference_pressure
    ) / (reference_sigma * reference_sound)
    audit = GlobalInnerCharacteristicProjectionAudit(
        incoming_amplitude_before=float(incoming_before),
        incoming_amplitude_after=float(
            projected_velocity_perturbation + projected_pressure_velocity
        ),
        outgoing_amplitude_before=float(outgoing_before),
        outgoing_amplitude_after=float(
            projected_velocity_perturbation - projected_pressure_velocity
        ),
        projected_radial_velocity=float(projected_velocity),
        projected_integrated_pressure=float(projected_vertical.Pi),
        projected_temperature=projected_temperature,
    )
    return (
        GlobalFaceFluxes(**values).validated_for(grid.centers.size),
        audit,
    )


def global_rusanov_face_fluxes(
    grid: RadialGrid,
    state: GlobalConservativeState,
    primitives: GlobalPrimitiveState,
    M_g: float,
    *,
    gamma_gas: float = 5.0 / 3.0,
) -> GlobalFaceFluxes:
    """Return first-order local Lax-Friedrichs fluxes for all four fields."""

    state = state.validated()
    if state.n_cells != grid.centers.size:
        raise ValueError("global state and radial grid have different sizes")
    sigma = primitives.surface_density
    velocity = primitives.radial_velocity
    specific_l = state.angular_momentum / state.mass
    specific_total = state.total_energy / state.mass
    integrated_pressure = np.asarray(primitives.vertical.Pi, dtype=float)
    potential = PaczynskiWiitaPotential(float(M_g))
    potential_centers = np.asarray(potential.phi(grid.centers), dtype=float)
    nonpotential_specific = specific_total - potential_centers
    sound_speed = global_effective_sound_speed(
        primitives, gamma_gas=gamma_gas
    )
    conserved = np.vstack(
        (
            sigma,
            sigma * velocity,
            sigma * specific_l,
            sigma * specific_total,
        )
    )
    physical_flux = np.vstack(
        (
            sigma * velocity,
            sigma * velocity**2 + integrated_pressure,
            sigma * velocity * specific_l,
            sigma
            * velocity
            * (specific_total + integrated_pressure / sigma),
        )
    )
    nonpotential_energy = sigma * nonpotential_specific
    nonpotential_energy_flux = sigma * velocity * (
        nonpotential_specific + integrated_pressure / sigma
    )
    n_cells = state.n_cells
    face_flux = np.empty((4, n_cells + 1), dtype=float)
    face_flux[:, 0] = physical_flux[:, 0]
    face_flux[:, -1] = physical_flux[:, -1]
    for face in range(1, n_cells):
        left = face - 1
        right = face
        maximum_speed = max(
            abs(float(velocity[left])) + float(sound_speed[left]),
            abs(float(velocity[right])) + float(sound_speed[right]),
        )
        face_flux[:, face] = 0.5 * (
            physical_flux[:, left] + physical_flux[:, right]
        ) - 0.5 * maximum_speed * (
            conserved[:, right] - conserved[:, left]
        )
        numerical_mass_flux = face_flux[0, face]
        face_flux[3, face] = (
            0.5
            * (
                nonpotential_energy_flux[left]
                + nonpotential_energy_flux[right]
            )
            - 0.5
            * maximum_speed
            * (nonpotential_energy[right] - nonpotential_energy[left])
            + float(potential.phi(grid.edges[face])) * numerical_mass_flux
        )
    face_flux *= 2.0 * np.pi * grid.edges[np.newaxis, :]
    return GlobalFaceFluxes(
        mass=face_flux[0],
        radial_momentum=face_flux[1],
        angular_momentum=face_flux[2],
        total_energy=face_flux[3],
    ).validated_for(n_cells)


def global_equilibrium_corrected_rusanov_fluxes(
    grid: RadialGrid,
    state: GlobalConservativeState,
    primitives: GlobalPrimitiveState,
    reference_state: GlobalConservativeState,
    reference_primitives: GlobalPrimitiveState,
    M_g: float,
    *,
    gamma_gas: float = 5.0 / 3.0,
) -> GlobalFaceFluxes:
    """Add Rusanov dissipation only to deviations from a reference equilibrium."""

    current = global_rusanov_face_fluxes(
        grid, state, primitives, M_g, gamma_gas=gamma_gas
    )
    reference_rusanov = global_rusanov_face_fluxes(
        grid,
        reference_state,
        reference_primitives,
        M_g,
        gamma_gas=gamma_gas,
    )
    reference_smooth = global_inviscid_face_fluxes(
        grid, reference_primitives, M_g
    )
    values = {
        name: getattr(current, name)
        - getattr(reference_rusanov, name)
        + getattr(reference_smooth, name)
        for name in _COMPONENTS
    }
    return GlobalFaceFluxes(**values).validated_for(grid.centers.size)


def global_open_no_inflow_boundary_fluxes(
    grid: RadialGrid,
    fluxes: GlobalFaceFluxes,
    primitives: GlobalPrimitiveState,
) -> GlobalFaceFluxes:
    """Block unconfigured advective inflow while retaining pressure traction."""

    fluxes = fluxes.validated_for(grid.centers.size)
    values = {
        name: np.array(getattr(fluxes, name), copy=True) for name in _COMPONENTS
    }
    integrated_pressure = np.asarray(primitives.vertical.Pi, dtype=float)
    blocked_faces = []
    if values["mass"][0] > 0.0:
        blocked_faces.append((0, 0))
    if values["mass"][-1] < 0.0:
        blocked_faces.append((-1, -1))
    for face, cell in blocked_faces:
        values["mass"][face] = 0.0
        values["angular_momentum"][face] = 0.0
        values["total_energy"][face] = 0.0
        values["radial_momentum"][face] = (
            2.0 * np.pi * grid.edges[face] * integrated_pressure[cell]
        )
    return GlobalFaceFluxes(**values).validated_for(grid.centers.size)


def global_inviscid_cell_sources(
    grid: RadialGrid,
    primitives: GlobalPrimitiveState,
    M_g: float,
    *,
    include_radiative_cooling: bool = False,
    kappa: float = DEFAULT_KAPPA_ES,
) -> GlobalCellSources:
    """Return cylindrical pressure, centrifugal, and gravity source rates."""

    potential = PaczynskiWiitaPotential(float(M_g))
    sigma = primitives.surface_density
    omega = primitives.omega
    omega_k = np.asarray(potential.omega_k(grid.centers), dtype=float)
    integrated_pressure = np.asarray(primitives.vertical.Pi, dtype=float)
    pressure_geometry = 2.0 * np.pi * grid.widths * integrated_pressure
    orbital_force = (
        grid.area
        * sigma
        * grid.centers
        * (omega**2 - omega_k**2)
    )
    zero = np.zeros(grid.centers.size, dtype=float)
    cooling = (
        global_radiative_cooling_rate_cells(
            grid, primitives, kappa=kappa
        )
        if include_radiative_cooling
        else zero
    )
    return GlobalCellSources(
        mass=np.array(zero, copy=True),
        radial_momentum=np.asarray(
            pressure_geometry + orbital_force, dtype=float
        ),
        angular_momentum=np.array(zero, copy=True),
        total_energy=-np.asarray(cooling, dtype=float),
    ).validated_for(grid.centers.size)


def _add_global_vertical_work_source(
    sources: GlobalCellSources,
    vertical_work_rate_cells,
) -> GlobalCellSources:
    """Add physical column work only to the total-energy source ledger."""

    size = np.asarray(sources.mass).size
    sources = sources.validated_for(size)
    work = _finite_vector(
        "vertical_work_rate_cells", vertical_work_rate_cells, size
    )
    return GlobalCellSources(
        mass=np.array(sources.mass, copy=True),
        radial_momentum=np.array(sources.radial_momentum, copy=True),
        angular_momentum=np.array(sources.angular_momentum, copy=True),
        total_energy=np.asarray(sources.total_energy + work, dtype=float),
    ).validated_for(size)


def global_radiative_cooling_rate_cells(
    grid: RadialGrid,
    primitives: GlobalPrimitiveState,
    *,
    kappa: float = DEFAULT_KAPPA_ES,
) -> np.ndarray:
    """Return positive two-face radiative energy loss per cell."""

    rate = (
        np.asarray(radiative_cooling(primitives.vertical, kappa=kappa), dtype=float)
        * grid.area
    )
    if rate.shape != grid.centers.shape or np.any(~np.isfinite(rate)):
        raise ValueError("radiative cooling rate must be finite and match the grid")
    if np.any(rate <= 0.0):
        raise ValueError("radiative cooling rate must be positive")
    return rate


def evaluate_global_inviscid_profile(
    grid: RadialGrid,
    state: GlobalConservativeState,
    M_g: float,
    *,
    temperature_bounds: tuple[float, float] = (1.0e3, 1.0e12),
    mu_mol: float = DEFAULT_MU_MOL,
    kappa: float = DEFAULT_KAPPA_ES,
    gamma_gas: float = 5.0 / 3.0,
    include_radiative_cooling: bool = False,
    include_vertical_column_work: bool = False,
    external_sources: GlobalCellSources | None = None,
    specific_mechanical_energy_correction=None,
) -> GlobalInviscidProfile:
    """Recover primitives and evaluate the smooth inviscid balance operator."""

    primitives = recover_global_primitives(
        grid,
        state,
        M_g,
        temperature_bounds=temperature_bounds,
        mu_mol=mu_mol,
        kappa=kappa,
        gamma_gas=gamma_gas,
        specific_mechanical_energy_correction=(
            specific_mechanical_energy_correction
        ),
    )
    face_fluxes = global_inviscid_face_fluxes(grid, primitives, M_g)
    cell_sources = global_inviscid_cell_sources(
        grid,
        primitives,
        M_g,
        include_radiative_cooling=include_radiative_cooling,
        kappa=kappa,
    )
    vertical_work = (
        global_vertical_work_rate_cells(
            grid, face_fluxes.mass, primitives
        )
        if include_vertical_column_work
        else np.zeros(grid.centers.size, dtype=float)
    )
    if include_vertical_column_work:
        cell_sources = _add_global_vertical_work_source(
            cell_sources, vertical_work
        )
    if external_sources is not None:
        cell_sources = combine_global_cell_sources(
            cell_sources, external_sources.validated_for(state.n_cells)
        )
    return GlobalInviscidProfile(
        primitives=primitives,
        face_fluxes=face_fluxes,
        cell_sources=cell_sources,
        viscous_torque_faces=np.zeros(grid.centers.size + 1, dtype=float),
        vertical_work_rate_cells=vertical_work,
    )


def evaluate_global_rusanov_profile(
    grid: RadialGrid,
    state: GlobalConservativeState,
    M_g: float,
    *,
    reference_state: GlobalConservativeState | None = None,
    boundary_mode: str = "transmissive",
    temperature_bounds: tuple[float, float] = (1.0e3, 1.0e12),
    mu_mol: float = DEFAULT_MU_MOL,
    kappa: float = DEFAULT_KAPPA_ES,
    gamma_gas: float = 5.0 / 3.0,
    alpha: float = 0.0,
    mu_stress: float = 0.0,
    stress_factor: float = 1.0,
    stress_boundary_mode: str = "extrapolated",
    include_radiative_cooling: bool = False,
    include_vertical_column_work: bool = False,
    external_sources: GlobalCellSources | None = None,
    primitives: GlobalPrimitiveState | None = None,
    reference_primitives: GlobalPrimitiveState | None = None,
    open_face_reconstruction: str = "primitive_product",
    specific_mechanical_energy_correction=None,
) -> GlobalInviscidProfile:
    """Evaluate Rusanov transport with optional paired alpha-stress fluxes."""

    if primitives is None:
        primitives = recover_global_primitives(
            grid,
            state,
            M_g,
            temperature_bounds=temperature_bounds,
            mu_mol=mu_mol,
            kappa=kappa,
            gamma_gas=gamma_gas,
            specific_mechanical_energy_correction=(
                specific_mechanical_energy_correction
            ),
        )
    if reference_state is None:
        if reference_primitives is not None:
            raise ValueError(
                "reference_primitives requires a reference_state"
            )
        face_fluxes = global_rusanov_face_fluxes(
            grid, state, primitives, M_g, gamma_gas=gamma_gas
        )
    else:
        if reference_primitives is None:
            reference_primitives = recover_global_primitives(
                grid,
                reference_state,
                M_g,
                temperature_bounds=temperature_bounds,
                mu_mol=mu_mol,
                kappa=kappa,
                gamma_gas=gamma_gas,
                specific_mechanical_energy_correction=(
                    specific_mechanical_energy_correction
                ),
            )
        face_fluxes = global_equilibrium_corrected_rusanov_fluxes(
            grid,
            state,
            primitives,
            reference_state,
            reference_primitives,
            M_g,
            gamma_gas=gamma_gas,
        )
    if open_face_reconstruction == "conserved_donor":
        face_fluxes = apply_global_conserved_donor_outer_flux(
            grid, face_fluxes, primitives
        )
    elif open_face_reconstruction != "primitive_product":
        raise ValueError(
            "open_face_reconstruction must be primitive_product or "
            "conserved_donor"
        )
    inner_projection = None
    if boundary_mode == "characteristic_inner_open_outer":
        if reference_primitives is None:
            raise ValueError(
                "characteristic inner boundary requires a reference state"
            )
        face_fluxes, inner_projection = (
            apply_global_reference_characteristic_inner_flux(
                grid,
                face_fluxes,
                primitives,
                reference_primitives,
                M_g,
                temperature_bounds=temperature_bounds,
                mu_mol=mu_mol,
                kappa=kappa,
                gamma_gas=gamma_gas,
            )
        )
        face_fluxes = global_open_no_inflow_boundary_fluxes(
            grid, face_fluxes, primitives
        )
    elif boundary_mode == "open_no_inflow":
        face_fluxes = global_open_no_inflow_boundary_fluxes(
            grid, face_fluxes, primitives
        )
    elif boundary_mode != "transmissive":
        raise ValueError(
            "boundary_mode must be transmissive, open_no_inflow, or "
            "characteristic_inner_open_outer"
        )
    face_fluxes, viscous_torque = add_global_alpha_stress_fluxes(
        grid,
        face_fluxes,
        primitives,
        alpha=alpha,
        mu_stress=mu_stress,
        stress_factor=stress_factor,
        boundary_mode=stress_boundary_mode,
    )
    cell_sources = global_inviscid_cell_sources(
        grid,
        primitives,
        M_g,
        include_radiative_cooling=include_radiative_cooling,
        kappa=kappa,
    )
    vertical_work = (
        global_vertical_work_rate_cells(
            grid, face_fluxes.mass, primitives
        )
        if include_vertical_column_work
        else np.zeros(grid.centers.size, dtype=float)
    )
    if include_vertical_column_work:
        cell_sources = _add_global_vertical_work_source(
            cell_sources, vertical_work
        )
    if external_sources is not None:
        cell_sources = combine_global_cell_sources(
            cell_sources, external_sources.validated_for(state.n_cells)
        )
    return GlobalInviscidProfile(
        primitives=primitives,
        face_fluxes=face_fluxes,
        cell_sources=cell_sources,
        viscous_torque_faces=viscous_torque,
        vertical_work_rate_cells=vertical_work,
        inner_characteristic_projection=inner_projection,
    )


def global_inviscid_cfl_timestep(
    grid: RadialGrid,
    primitives: GlobalPrimitiveState,
    *,
    cfl: float = 0.25,
    gamma_gas: float = 5.0 / 3.0,
) -> float:
    """Return the explicit acoustic CFL step for the current primitive state."""

    if not 0.0 < cfl <= 1.0:
        raise ValueError("cfl must lie in (0,1]")
    signal_speed = np.abs(primitives.radial_velocity) + global_effective_sound_speed(
        primitives, gamma_gas=gamma_gas
    )
    return float(cfl * np.min(grid.widths / signal_speed))


def advance_global_inviscid_rusanov(
    grid: RadialGrid,
    state: GlobalConservativeState,
    M_g: float,
    dt: float,
    *,
    reference_state: GlobalConservativeState | None = None,
    boundary_mode: str = "transmissive",
    temperature_bounds: tuple[float, float] = (1.0e3, 1.0e12),
    mu_mol: float = DEFAULT_MU_MOL,
    kappa: float = DEFAULT_KAPPA_ES,
    gamma_gas: float = 5.0 / 3.0,
    alpha: float = 0.0,
    mu_stress: float = 0.0,
    stress_factor: float = 1.0,
    stress_boundary_mode: str = "extrapolated",
    include_radiative_cooling: bool = False,
    external_sources: GlobalCellSources | None = None,
) -> GlobalInviscidStepResult:
    """Advance one explicit step and reject non-positive thermal states."""

    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive and finite")
    state = state.validated()
    profile = evaluate_global_rusanov_profile(
        grid,
        state,
        M_g,
        reference_state=reference_state,
        boundary_mode=boundary_mode,
        temperature_bounds=temperature_bounds,
        mu_mol=mu_mol,
        kappa=kappa,
        gamma_gas=gamma_gas,
        alpha=alpha,
        mu_stress=mu_stress,
        stress_factor=stress_factor,
        stress_boundary_mode=stress_boundary_mode,
        include_radiative_cooling=include_radiative_cooling,
        external_sources=external_sources,
    )
    rhs = global_conservative_rhs(profile.face_fluxes, profile.cell_sources)
    trial = GlobalConservativeState(
        **{
            name: getattr(state, name) + dt * getattr(rhs, name)
            for name in _COMPONENTS
        }
    )
    accepted = True
    message = "accepted"
    try:
        trial = trial.validated()
        recover_global_primitives(
            grid,
            trial,
            M_g,
            temperature_bounds=temperature_bounds,
            mu_mol=mu_mol,
            kappa=kappa,
            gamma_gas=gamma_gas,
        )
    except ValueError as error:
        accepted = False
        message = str(error)
        trial = state
    ledger = audit_global_backward_euler_ledgers(
        trial,
        state,
        dt,
        profile.face_fluxes,
        profile.cell_sources,
    )
    return GlobalInviscidStepResult(
        state=trial,
        profile=profile,
        ledger=ledger,
        accepted=accepted,
        dt=float(dt),
        message=message,
    )


def advance_global_alpha_stress_backward_euler(
    grid: RadialGrid,
    state: GlobalConservativeState,
    M_g: float,
    dt: float,
    *,
    alpha: float,
    mu_stress: float = 0.0,
    stress_factor: float = 1.0,
    stress_boundary_mode: str = "extrapolated",
    temperature_bounds: tuple[float, float] = (1.0e3, 1.0e12),
    mu_mol: float = DEFAULT_MU_MOL,
    kappa: float = DEFAULT_KAPPA_ES,
    gamma_gas: float = 5.0 / 3.0,
    residual_tolerance: float = 1.0e-9,
    ledger_tolerance: float = 1.0e-8,
    max_nfev: int = 200,
) -> GlobalImplicitStressStepResult:
    """Advance the paired alpha torque and work with a physical-state solve.

    This IMEX substep holds cell mass and radial momentum fixed. Positive
    rotation and temperature are the nonlinear unknowns, while cell angular
    momentum and total energy are reconstructed and required to satisfy the
    two backward-Euler conservation laws.
    """

    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive and finite")
    if not np.isfinite(residual_tolerance) or residual_tolerance <= 0.0:
        raise ValueError("residual_tolerance must be positive and finite")
    if not np.isfinite(ledger_tolerance) or ledger_tolerance <= 0.0:
        raise ValueError("ledger_tolerance must be positive and finite")
    if int(max_nfev) != max_nfev or max_nfev < 1:
        raise ValueError("max_nfev must be a positive integer")
    state = state.validated()
    old = recover_global_primitives(
        grid,
        state,
        M_g,
        temperature_bounds=temperature_bounds,
        mu_mol=mu_mol,
        kappa=kappa,
        gamma_gas=gamma_gas,
    )
    sigma = old.surface_density
    radial_velocity = old.radial_velocity
    n_cells = state.n_cells
    lower_temperature, upper_temperature = map(float, temperature_bounds)
    initial = np.concatenate((np.log(old.omega), np.log(old.temperature)))
    lower = np.concatenate(
        (
            np.full(n_cells, -np.inf),
            np.full(n_cells, np.log(lower_temperature)),
        )
    )
    upper = np.concatenate(
        (
            np.full(n_cells, np.inf),
            np.full(n_cells, np.log(upper_temperature)),
        )
    )
    angular_scale = np.maximum(np.abs(state.angular_momentum), 1.0)
    energy_scale = np.maximum(np.abs(state.total_energy), 1.0)

    def reconstruct(values: np.ndarray):
        omega = np.exp(values[:n_cells])
        temperature = np.exp(values[n_cells:])
        reconstructed = state_from_thermodynamic_primitives(
            grid,
            sigma,
            radial_velocity,
            omega,
            temperature,
            M_g,
            mu_mol=mu_mol,
            kappa=kappa,
            gamma_gas=gamma_gas,
        )
        trial = GlobalConservativeState(
            mass=np.array(state.mass, copy=True),
            radial_momentum=np.array(state.radial_momentum, copy=True),
            angular_momentum=reconstructed.angular_momentum,
            total_energy=reconstructed.total_energy,
        ).validated()
        primitives = recover_global_primitives(
            grid,
            trial,
            M_g,
            temperature_bounds=temperature_bounds,
            mu_mol=mu_mol,
            kappa=kappa,
            gamma_gas=gamma_gas,
        )
        torque = global_alpha_stress_torque_faces(
            grid,
            primitives,
            alpha=alpha,
            mu_stress=mu_stress,
            stress_factor=stress_factor,
            boundary_mode=stress_boundary_mode,
        )
        omega_faces = positive_edge_reconstruction(grid, primitives.omega)
        fluxes = GlobalFaceFluxes(
            mass=np.zeros(n_cells + 1, dtype=float),
            radial_momentum=np.zeros(n_cells + 1, dtype=float),
            angular_momentum=torque,
            total_energy=omega_faces * torque,
        )
        return trial, torque, fluxes

    def residual(values: np.ndarray) -> np.ndarray:
        trial, _torque, fluxes = reconstruct(values)
        angular_rhs = fluxes.angular_momentum[:-1] - fluxes.angular_momentum[1:]
        energy_rhs = fluxes.total_energy[:-1] - fluxes.total_energy[1:]
        return np.concatenate(
            (
                (
                    trial.angular_momentum
                    - state.angular_momentum
                    - dt * angular_rhs
                )
                / angular_scale,
                (
                    trial.total_energy
                    - state.total_energy
                    - dt * energy_rhs
                )
                / energy_scale,
            )
        )

    solve = least_squares(
        residual,
        initial,
        bounds=(lower, upper),
        xtol=1.0e-14,
        ftol=1.0e-14,
        gtol=1.0e-14,
        max_nfev=int(max_nfev),
    )
    trial, torque, fluxes = reconstruct(solve.x)
    maximum_residual = float(np.max(np.abs(residual(solve.x))))
    trial_ledger = audit_global_backward_euler_ledgers(
        trial,
        state,
        dt,
        fluxes,
        GlobalCellSources.zeros(n_cells),
    )
    storage_ledger_defect = maximum_storage_scaled_ledger_defect(
        trial, trial_ledger
    )
    accepted = bool(
        maximum_residual <= residual_tolerance
        and storage_ledger_defect <= ledger_tolerance
    )
    accepted_state = trial if accepted else state
    if accepted:
        ledger = trial_ledger
        message = "accepted"
    else:
        ledger = trial_ledger
        message = (
            f"implicit stress candidate residual {maximum_residual:.6e} "
            f"and storage-scaled ledger {storage_ledger_defect:.6e}; "
            f"gates are {residual_tolerance:.6e} and "
            f"{ledger_tolerance:.6e}: {solve.message}"
        )
    return GlobalImplicitStressStepResult(
        state=accepted_state,
        face_fluxes=fluxes,
        viscous_torque_faces=torque,
        ledger=ledger,
        accepted=accepted,
        dt=float(dt),
        nfev=int(solve.nfev),
        maximum_scaled_residual=maximum_residual,
        maximum_storage_scaled_ledger_defect=storage_ledger_defect,
        message=message,
    )


def advance_global_imex(
    grid: RadialGrid,
    state: GlobalConservativeState,
    M_g: float,
    dt: float,
    *,
    alpha: float,
    reference_state: GlobalConservativeState | None = None,
    boundary_mode: str = "transmissive",
    stress_boundary_mode: str = "extrapolated",
    temperature_bounds: tuple[float, float] = (1.0e3, 1.0e12),
    mu_mol: float = DEFAULT_MU_MOL,
    kappa: float = DEFAULT_KAPPA_ES,
    gamma_gas: float = 5.0 / 3.0,
    mu_stress: float = 0.0,
    stress_factor: float = 1.0,
    ledger_tolerance: float = 1.0e-8,
) -> GlobalIMEXStepResult:
    """Compose explicit Euler transport and backward-Euler alpha stress."""

    state = state.validated()
    inviscid = advance_global_inviscid_rusanov(
        grid,
        state,
        M_g,
        dt,
        reference_state=reference_state,
        boundary_mode=boundary_mode,
        temperature_bounds=temperature_bounds,
        mu_mol=mu_mol,
        kappa=kappa,
        gamma_gas=gamma_gas,
    )
    if not inviscid.accepted:
        return GlobalIMEXStepResult(
            state=state,
            inviscid=inviscid,
            stress=None,
            ledger=inviscid.ledger,
            accepted=False,
            dt=float(dt),
            maximum_storage_scaled_ledger_defect=(
                maximum_storage_scaled_ledger_defect(state, inviscid.ledger)
            ),
            message=f"explicit transport rejected: {inviscid.message}",
        )
    stress = advance_global_alpha_stress_backward_euler(
        grid,
        inviscid.state,
        M_g,
        dt,
        alpha=alpha,
        mu_stress=mu_stress,
        stress_factor=stress_factor,
        stress_boundary_mode=stress_boundary_mode,
        temperature_bounds=temperature_bounds,
        mu_mol=mu_mol,
        kappa=kappa,
        gamma_gas=gamma_gas,
        ledger_tolerance=ledger_tolerance,
    )
    if not stress.accepted:
        return GlobalIMEXStepResult(
            state=state,
            inviscid=inviscid,
            stress=stress,
            ledger=stress.ledger,
            accepted=False,
            dt=float(dt),
            maximum_storage_scaled_ledger_defect=(
                stress.maximum_storage_scaled_ledger_defect
            ),
            message=f"implicit stress rejected: {stress.message}",
        )
    combined_fluxes = GlobalFaceFluxes(
        **{
            name: getattr(inviscid.profile.face_fluxes, name)
            + getattr(stress.face_fluxes, name)
            for name in _COMPONENTS
        }
    )
    ledger = audit_global_backward_euler_ledgers(
        stress.state,
        state,
        dt,
        combined_fluxes,
        inviscid.profile.cell_sources,
    )
    storage_ledger_defect = maximum_storage_scaled_ledger_defect(
        stress.state, ledger
    )
    accepted = bool(storage_ledger_defect <= ledger_tolerance)
    return GlobalIMEXStepResult(
        state=stress.state if accepted else state,
        inviscid=inviscid,
        stress=stress,
        ledger=ledger,
        accepted=accepted,
        dt=float(dt),
        maximum_storage_scaled_ledger_defect=storage_ledger_defect,
        message=(
            "accepted"
            if accepted
            else (
                "combined IMEX ledger defect "
                f"{storage_ledger_defect:.6e} exceeds "
                f"{ledger_tolerance:.6e}"
            )
        ),
    )


def advance_global_radiative_cooling_backward_euler(
    grid: RadialGrid,
    state: GlobalConservativeState,
    M_g: float,
    dt: float,
    *,
    temperature_bounds: tuple[float, float] = (1.0e3, 1.0e12),
    mu_mol: float = DEFAULT_MU_MOL,
    kappa: float = DEFAULT_KAPPA_ES,
    gamma_gas: float = 5.0 / 3.0,
    ledger_tolerance: float = 1.0e-12,
) -> GlobalCoolingStepResult:
    """Cool each cell implicitly while preserving the other conserved fields."""

    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive and finite")
    state = state.validated()
    old = recover_global_primitives(
        grid,
        state,
        M_g,
        temperature_bounds=temperature_bounds,
        mu_mol=mu_mol,
        kappa=kappa,
        gamma_gas=gamma_gas,
    )
    lower_temperature, _upper_temperature = map(float, temperature_bounds)
    potential = PaczynskiWiitaPotential(float(M_g))
    new_temperature = np.empty(state.n_cells, dtype=float)
    for index in range(state.n_cells):
        sigma = float(old.surface_density[index])
        radius = float(grid.centers[index])
        old_internal = float(old.specific_internal_energy[index])

        def cell_residual(log_temperature: float) -> float:
            local = vertical_state(
                sigma,
                float(np.exp(log_temperature)),
                radius,
                potential,
                mu_mol=mu_mol,
                kappa=kappa,
                gamma_gas=gamma_gas,
            )
            cooling_rate = float(
                radiative_cooling(local, kappa=kappa) * grid.area[index]
            )
            return float(
                state.mass[index] * (float(local.e) - old_internal)
                + dt * cooling_rate
            )

        lower = float(np.log(lower_temperature))
        upper = float(np.log(old.temperature[index]))
        if cell_residual(lower) >= 0.0:
            zero = np.zeros(state.n_cells, dtype=float)
            ledger = GlobalLedgerAudit(
                defects={name: 0.0 for name in _COMPONENTS},
                relative_defects={name: 0.0 for name in _COMPONENTS},
            )
            return GlobalCoolingStepResult(
                state=state,
                cooling_rate_cells=zero,
                ledger=ledger,
                accepted=False,
                dt=float(dt),
                maximum_storage_scaled_ledger_defect=0.0,
                message=(
                    f"cell {index} cooling root falls below the temperature "
                    "bound"
                ),
            )
        root = brentq(cell_residual, lower, upper, xtol=1.0e-12, rtol=1.0e-12)
        new_temperature[index] = np.exp(root)
    reconstructed = state_from_thermodynamic_primitives(
        grid,
        old.surface_density,
        old.radial_velocity,
        old.omega,
        new_temperature,
        M_g,
        mu_mol=mu_mol,
        kappa=kappa,
        gamma_gas=gamma_gas,
    )
    trial = GlobalConservativeState(
        mass=np.array(state.mass, copy=True),
        radial_momentum=np.array(state.radial_momentum, copy=True),
        angular_momentum=np.array(state.angular_momentum, copy=True),
        total_energy=reconstructed.total_energy,
    ).validated()
    new_primitives = recover_global_primitives(
        grid,
        trial,
        M_g,
        temperature_bounds=temperature_bounds,
        mu_mol=mu_mol,
        kappa=kappa,
        gamma_gas=gamma_gas,
    )
    cooling_rate = global_radiative_cooling_rate_cells(
        grid, new_primitives, kappa=kappa
    )
    zero_faces = np.zeros(state.n_cells + 1, dtype=float)
    fluxes = GlobalFaceFluxes(
        mass=np.array(zero_faces, copy=True),
        radial_momentum=np.array(zero_faces, copy=True),
        angular_momentum=np.array(zero_faces, copy=True),
        total_energy=np.array(zero_faces, copy=True),
    )
    sources = GlobalCellSources(
        mass=np.zeros(state.n_cells, dtype=float),
        radial_momentum=np.zeros(state.n_cells, dtype=float),
        angular_momentum=np.zeros(state.n_cells, dtype=float),
        total_energy=-cooling_rate,
    )
    ledger = audit_global_backward_euler_ledgers(
        trial, state, dt, fluxes, sources
    )
    storage_defect = maximum_storage_scaled_ledger_defect(trial, ledger)
    accepted = bool(storage_defect <= ledger_tolerance)
    return GlobalCoolingStepResult(
        state=trial if accepted else state,
        cooling_rate_cells=cooling_rate,
        ledger=ledger,
        accepted=accepted,
        dt=float(dt),
        maximum_storage_scaled_ledger_defect=storage_defect,
        message=(
            "accepted"
            if accepted
            else (
                f"cooling ledger defect {storage_defect:.6e} exceeds "
                f"{ledger_tolerance:.6e}"
            )
        ),
    )


def advance_global_backward_euler(
    grid: RadialGrid,
    state: GlobalConservativeState,
    M_g: float,
    dt: float,
    *,
    alpha: float = 0.0,
    reference_state: GlobalConservativeState | None = None,
    boundary_mode: str = "transmissive",
    stress_boundary_mode: str = "extrapolated",
    temperature_bounds: tuple[float, float] = (1.0e3, 1.0e12),
    mu_mol: float = DEFAULT_MU_MOL,
    kappa: float = DEFAULT_KAPPA_ES,
    gamma_gas: float = 5.0 / 3.0,
    mu_stress: float = 0.0,
    stress_factor: float = 1.0,
    residual_tolerance: float = 1.0e-8,
    ledger_tolerance: float = 1.0e-8,
    max_nfev: int = 300,
    use_sparse_jacobian: bool = False,
    jacobian_mode: str | None = None,
    jacobian_relative_tolerance: float = 3.0e-5,
    include_radiative_cooling: bool = False,
    include_vertical_column_work: bool = False,
    external_sources: GlobalCellSources | None = None,
    open_face_reconstruction: str = "primitive_product",
    specific_mechanical_energy_correction=None,
) -> GlobalBackwardEulerStepResult:
    """Solve all four conservation laws together at the new time level."""

    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive and finite")
    state = state.validated()
    old = recover_global_primitives(
        grid,
        state,
        M_g,
        temperature_bounds=temperature_bounds,
        mu_mol=mu_mol,
        kappa=kappa,
        gamma_gas=gamma_gas,
        specific_mechanical_energy_correction=(
            specific_mechanical_energy_correction
        ),
    )
    n_cells = state.n_cells
    initial = np.concatenate(
        (
            np.log(old.surface_density),
            old.radial_velocity / C,
            np.log(old.omega),
            np.log(old.temperature),
        )
    )
    lower_temperature, upper_temperature = map(float, temperature_bounds)
    lower = np.concatenate(
        (
            np.full(n_cells, -np.inf),
            np.full(n_cells, -np.inf),
            np.full(n_cells, -np.inf),
            np.full(n_cells, np.log(lower_temperature)),
        )
    )
    upper = np.concatenate(
        (
            np.full(n_cells, np.inf),
            np.full(n_cells, np.inf),
            np.full(n_cells, np.inf),
            np.full(n_cells, np.log(upper_temperature)),
        )
    )
    sound_speed = global_effective_sound_speed(old, gamma_gas=gamma_gas)
    scales = {
        "mass": np.maximum(np.abs(state.mass), 1.0),
        "radial_momentum": np.maximum(
            state.mass * (np.abs(old.radial_velocity) + sound_speed), 1.0
        ),
        "angular_momentum": np.maximum(
            np.abs(state.angular_momentum), 1.0
        ),
        # Orbital binding dominates total energy but cancels in the thermal
        # balance. Scale this row by thermal storage so cooling and torque
        # heating cannot hide beneath the orbital-energy magnitude.
        "total_energy": np.maximum(
            state.mass * np.abs(old.specific_internal_energy), 1.0
        ),
    }
    fixed_reference_primitives = (
        None
        if reference_state is None
        else recover_global_primitives(
            grid,
            reference_state,
            M_g,
            temperature_bounds=temperature_bounds,
            mu_mol=mu_mol,
            kappa=kappa,
            gamma_gas=gamma_gas,
            specific_mechanical_energy_correction=(
                specific_mechanical_energy_correction
            ),
        )
    )

    def reconstruct(values: np.ndarray):
        sigma = np.exp(values[:n_cells])
        radial_velocity = C * values[n_cells : 2 * n_cells]
        omega = np.exp(values[2 * n_cells : 3 * n_cells])
        temperature = np.exp(values[3 * n_cells :])
        trial = state_from_thermodynamic_primitives(
            grid,
            sigma,
            radial_velocity,
            omega,
            temperature,
            M_g,
            mu_mol=mu_mol,
            kappa=kappa,
            gamma_gas=gamma_gas,
            specific_mechanical_energy_correction=(
                specific_mechanical_energy_correction
            ),
        )
        vertical = vertical_state(
            sigma,
            temperature,
            grid.centers,
            PaczynskiWiitaPotential(float(M_g)),
            mu_mol=mu_mol,
            kappa=kappa,
            gamma_gas=gamma_gas,
        )
        trial_primitives = GlobalPrimitiveState(
            surface_density=np.asarray(sigma, dtype=float),
            radial_velocity=np.asarray(radial_velocity, dtype=float),
            omega=np.asarray(omega, dtype=float),
            temperature=np.asarray(temperature, dtype=float),
            specific_total_energy=np.asarray(
                trial.total_energy / trial.mass, dtype=float
            ),
            specific_internal_energy=np.asarray(vertical.e, dtype=float),
            vertical=vertical,
        )
        profile = evaluate_global_rusanov_profile(
            grid,
            trial,
            M_g,
            reference_state=reference_state,
            boundary_mode=boundary_mode,
            temperature_bounds=temperature_bounds,
            mu_mol=mu_mol,
            kappa=kappa,
            gamma_gas=gamma_gas,
            alpha=alpha,
            mu_stress=mu_stress,
            stress_factor=stress_factor,
            stress_boundary_mode=stress_boundary_mode,
            include_radiative_cooling=include_radiative_cooling,
            include_vertical_column_work=include_vertical_column_work,
            external_sources=external_sources,
            primitives=trial_primitives,
            reference_primitives=fixed_reference_primitives,
            open_face_reconstruction=open_face_reconstruction,
            specific_mechanical_energy_correction=(
                specific_mechanical_energy_correction
            ),
        )
        return trial, profile

    def residual(values: np.ndarray) -> np.ndarray:
        trial, profile = reconstruct(values)
        temporal_work = (
            global_temporal_vertical_work_cells(
                state, old, trial, profile.primitives
            )
            if include_vertical_column_work
            else None
        )
        unscaled = global_backward_euler_residual(
            trial,
            state,
            dt,
            profile.face_fluxes,
            profile.cell_sources,
            energy_storage_correction=temporal_work,
        )
        return np.concatenate(
            tuple(
                unscaled[index * n_cells : (index + 1) * n_cells]
                / scales[name]
                for index, name in enumerate(_COMPONENTS)
            )
        )

    if jacobian_mode is None:
        jacobian_mode = (
            "colored_forward" if use_sparse_jacobian else "dense"
        )
    elif use_sparse_jacobian:
        raise ValueError(
            "use_sparse_jacobian cannot be combined with jacobian_mode"
        )
    if jacobian_mode not in {
        "dense",
        "colored_forward",
        "colored_central",
        "sparse_central",
        "sparse_forward",
    }:
        raise ValueError(
            "jacobian_mode must be dense, colored_forward, colored_central, "
            "sparse_central, or sparse_forward"
        )
    jacobian_audit = None
    jacobian_options = {"jac": "2-point", "diff_step": 1.0e-6}
    if jacobian_mode == "colored_forward":
        pattern = global_backward_euler_jacobian_sparsity(n_cells)
        jacobian_options = {
            "jac": lambda values: _colored_finite_difference_jacobian(
                residual,
                values,
                pattern,
                lower,
                upper,
            )
        }
    elif jacobian_mode in {
        "colored_central",
        "sparse_central",
        "sparse_forward",
    }:
        pattern = global_backward_euler_jacobian_sparsity(n_cells)
        jacobian_builder = {
            "colored_central": _colored_central_finite_difference_jacobian,
            "sparse_central": _sparse_central_finite_difference_jacobian,
            "sparse_forward": _sparse_forward_finite_difference_jacobian,
        }[jacobian_mode]
        if jacobian_mode == "sparse_forward":
            dense_initial_jacobian = (
                _dense_forward_finite_difference_jacobian(
                    residual, initial, lower, upper
                )
            )
            initial_jacobian, jacobian_audit = (
                _audit_sparse_pattern_against_dense_columns(
                    dense_initial_jacobian,
                    pattern,
                    relative_tolerance=jacobian_relative_tolerance,
                )
            )
        else:
            initial_jacobian = jacobian_builder(
                residual,
                initial,
                pattern,
                lower,
                upper,
            )
            jacobian_audit = _audit_sparse_jacobian_directions(
                residual,
                initial,
                initial_jacobian,
                lower,
                upper,
                relative_tolerance=jacobian_relative_tolerance,
            )
        if not jacobian_audit.accepted:
            trial, profile = reconstruct(initial)
            ledger = audit_global_backward_euler_ledgers(
                trial,
                state,
                dt,
                profile.face_fluxes,
                profile.cell_sources,
                energy_storage_correction=(
                    global_temporal_vertical_work_cells(
                        state, old, trial, profile.primitives
                    )
                    if include_vertical_column_work
                    else None
                ),
            )
            return GlobalBackwardEulerStepResult(
                state=state,
                profile=profile,
                ledger=ledger,
                accepted=False,
                dt=float(dt),
                nfev=0,
                maximum_scaled_residual=float(
                    np.max(np.abs(residual(initial)))
                ),
                maximum_storage_scaled_ledger_defect=(
                    maximum_storage_scaled_ledger_defect(state, ledger)
                ),
                message=(
                    f"{jacobian_mode} Jacobian directional defect "
                    f"{jacobian_audit.maximum_relative_defect:.6e} exceeds "
                    f"{jacobian_relative_tolerance:.6e}"
                ),
                jacobian_audit=jacobian_audit,
            )
        cached_values = np.array(initial, copy=True)
        cached_jacobian = initial_jacobian

        def certified_jacobian(values):
            nonlocal cached_values, cached_jacobian
            values = np.asarray(values, dtype=float)
            if np.array_equal(values, cached_values):
                return cached_jacobian.toarray()
            cached_values = np.array(values, copy=True)
            cached_jacobian = jacobian_builder(
                residual,
                values,
                pattern,
                lower,
                upper,
            )
            return cached_jacobian.toarray()

        jacobian_options = {"jac": certified_jacobian, "tr_solver": "exact"}
    solve = least_squares(
        residual,
        initial,
        bounds=(lower, upper),
        xtol=1.0e-15,
        ftol=None,
        gtol=1.0e-12,
        x_scale="jac",
        max_nfev=int(max_nfev),
        **jacobian_options,
    )
    trial, profile = reconstruct(solve.x)
    maximum_residual = float(np.max(np.abs(residual(solve.x))))
    ledger = audit_global_backward_euler_ledgers(
        trial,
        state,
        dt,
        profile.face_fluxes,
        profile.cell_sources,
        energy_storage_correction=(
            global_temporal_vertical_work_cells(
                state, old, trial, profile.primitives
            )
            if include_vertical_column_work
            else None
        ),
    )
    storage_ledger_defect = maximum_storage_scaled_ledger_defect(trial, ledger)
    accepted = bool(
        maximum_residual <= residual_tolerance
        and storage_ledger_defect <= ledger_tolerance
    )
    return GlobalBackwardEulerStepResult(
        state=trial if accepted else state,
        profile=profile,
        ledger=ledger,
        accepted=accepted,
        dt=float(dt),
        nfev=int(solve.nfev),
        maximum_scaled_residual=maximum_residual,
        maximum_storage_scaled_ledger_defect=storage_ledger_defect,
        message=(
            "accepted"
            if accepted
            else (
                f"global backward-Euler residual {maximum_residual:.6e}, "
                f"storage ledger {storage_ledger_defect:.6e}: {solve.message}"
            )
        ),
        jacobian_audit=jacobian_audit,
    )


def pack_global_flux_primary_state(
    state: GlobalConservativeState,
    fluxes: GlobalFaceFluxes,
) -> np.ndarray:
    """Pack four differential fields followed by four face-flux fields."""

    state = state.validated()
    fluxes = fluxes.validated_for(state.n_cells)
    return np.concatenate(
        tuple(getattr(state, name) for name in _COMPONENTS)
        + tuple(getattr(fluxes, name) for name in _COMPONENTS)
    )


def unpack_global_flux_primary_state(
    values,
    layout: GlobalFluxPrimaryLayout,
) -> tuple[GlobalConservativeState, GlobalFaceFluxes]:
    """Unpack a global mixed differential/algebraic state."""

    values = _finite_vector("global state", values, layout.state_size)
    slices = layout.state_slices()
    state = GlobalConservativeState(
        **{
            name: np.asarray(values[slices[f"cell_{name}"]], dtype=float)
            for name in _COMPONENTS
        }
    ).validated()
    fluxes = GlobalFaceFluxes(
        **{
            name: np.asarray(values[slices[f"face_{name}"]], dtype=float)
            for name in _COMPONENTS
        }
    ).validated_for(layout.n_cells)
    return state, fluxes


def global_conservative_rhs(
    fluxes: GlobalFaceFluxes,
    sources: GlobalCellSources,
) -> GlobalConservativeState:
    """Return exact finite-volume rates using outward-oriented fluxes."""

    n_cells = np.asarray(sources.mass).size
    fluxes = fluxes.validated_for(n_cells)
    sources = sources.validated_for(n_cells)
    rates = {
        name: getattr(fluxes, name)[:-1]
        - getattr(fluxes, name)[1:]
        + getattr(sources, name)
        for name in _COMPONENTS
    }
    return GlobalConservativeState(**rates)


def global_backward_euler_residual(
    new_state: GlobalConservativeState,
    old_state: GlobalConservativeState,
    dt: float,
    new_fluxes: GlobalFaceFluxes,
    new_sources: GlobalCellSources,
    *,
    energy_storage_correction=None,
) -> np.ndarray:
    """Return unscaled backward-Euler conservation rows."""

    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive and finite")
    old_state = old_state.validated()
    new_state = new_state.validated()
    if new_state.n_cells != old_state.n_cells:
        raise ValueError("old and new global states use different meshes")
    rhs = global_conservative_rhs(new_fluxes, new_sources)
    correction = (
        np.zeros(new_state.n_cells, dtype=float)
        if energy_storage_correction is None
        else _finite_vector(
            "energy_storage_correction",
            energy_storage_correction,
            new_state.n_cells,
        )
    )
    return np.concatenate(
        tuple(
            getattr(new_state, name)
            - getattr(old_state, name)
            - dt * getattr(rhs, name)
            + (correction if name == "total_energy" else 0.0)
            for name in _COMPONENTS
        )
    )


def audit_global_backward_euler_ledgers(
    new_state: GlobalConservativeState,
    old_state: GlobalConservativeState,
    dt: float,
    new_fluxes: GlobalFaceFluxes,
    new_sources: GlobalCellSources,
    *,
    energy_storage_correction=None,
) -> GlobalLedgerAudit:
    """Audit telescoped boundary fluxes independently of cell residuals."""

    old_state = old_state.validated()
    new_state = new_state.validated()
    fluxes = new_fluxes.validated_for(old_state.n_cells)
    sources = new_sources.validated_for(old_state.n_cells)
    correction = (
        np.zeros(old_state.n_cells, dtype=float)
        if energy_storage_correction is None
        else _finite_vector(
            "energy_storage_correction",
            energy_storage_correction,
            old_state.n_cells,
        )
    )
    defects: dict[str, float] = {}
    relative: dict[str, float] = {}
    for name in _COMPONENTS:
        cell_change = getattr(new_state, name) - getattr(old_state, name)
        corrected_change = cell_change + (
            correction if name == "total_energy" else 0.0
        )
        change = float(np.sum(corrected_change))
        expected = dt * float(
            getattr(fluxes, name)[0]
            - getattr(fluxes, name)[-1]
            + np.sum(getattr(sources, name))
        )
        defect = change - expected
        activity = float(
            np.sum(np.abs(corrected_change))
            + dt
            * (
                abs(getattr(fluxes, name)[0])
                + abs(getattr(fluxes, name)[-1])
                + np.sum(np.abs(getattr(sources, name)))
            )
        )
        scale = max(activity, abs(change), abs(expected), 1.0)
        defects[name] = float(defect)
        relative[name] = float(abs(defect) / scale)
    return GlobalLedgerAudit(defects=defects, relative_defects=relative)


def maximum_storage_scaled_ledger_defect(
    state: GlobalConservativeState,
    ledger: GlobalLedgerAudit,
) -> float:
    """Scale global defects by conserved storage for near-equilibrium gates."""

    state = state.validated()
    scales = {
        "mass": max(float(np.sum(np.abs(state.mass))), 1.0),
        "radial_momentum": max(float(np.sum(state.mass)) * C, 1.0),
        "angular_momentum": max(
            float(np.sum(np.abs(state.angular_momentum))), 1.0
        ),
        "total_energy": max(
            float(np.sum(np.abs(state.total_energy))), 1.0
        ),
    }
    return max(
        abs(float(ledger.defects[name])) / scales[name]
        for name in _COMPONENTS
    )


def global_descriptor_mass_matrix(layout: GlobalFluxPrimaryLayout) -> csr_matrix:
    """Return identity differential storage and zero algebraic storage."""

    return block_diag(
        (
            eye(layout.differential_size, format="csr"),
            csr_matrix((layout.algebraic_size, layout.algebraic_size)),
        ),
        format="csr",
    )


def manufactured_flux_closure_jacobian(
    layout: GlobalFluxPrimaryLayout,
) -> csr_matrix:
    """Return the identity face-flux closure used only for rank tests."""

    return block_diag(
        (
            csr_matrix((layout.differential_size, layout.differential_size)),
            eye(layout.algebraic_size, format="csr"),
        ),
        format="csr",
    )


def manufactured_backward_euler_jacobian(
    layout: GlobalFluxPrimaryLayout,
    dt: float,
) -> csr_matrix:
    """Return a square full-rank Jacobian for the manufactured descriptor test."""

    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive and finite")
    differential = eye(layout.differential_size, format="csr") / dt
    algebraic = eye(layout.algebraic_size, format="csr")
    return block_diag((differential, algebraic), format="csr")
