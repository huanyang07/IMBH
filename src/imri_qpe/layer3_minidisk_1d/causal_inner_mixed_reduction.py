"""Conservation-constrained mixed-mode reduction for the causal descriptor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _scipy_linalg():
    try:
        from scipy.linalg import solve
        from scipy.sparse.linalg import expm_multiply
    except ImportError as exc:  # pragma: no cover - exercised without solver extra
        raise RuntimeError(
            "scipy is required for causal mixed-mode reduction"
        ) from exc
    return solve, expm_multiply


def _finite_matrix(
    values: np.ndarray,
    *,
    name: str,
    rows: int | None = None,
) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or np.any(~np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite matrix")
    if rows is not None and matrix.shape[0] != rows:
        raise ValueError(f"{name} has an incompatible row count")
    return matrix


def causal_descriptor_explicit_matrices(
    descriptor: np.ndarray,
    stationary: np.ndarray,
    inputs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return ``dx/dt = L x + G u`` from ``E dx/dt + K x = R u``."""

    descriptor_matrix = _finite_matrix(descriptor, name="descriptor")
    stationary_matrix = _finite_matrix(
        stationary,
        name="stationary",
        rows=descriptor_matrix.shape[0],
    )
    input_matrix = _finite_matrix(
        inputs,
        name="inputs",
        rows=descriptor_matrix.shape[0],
    )
    if (
        descriptor_matrix.shape[0] != descriptor_matrix.shape[1]
        or stationary_matrix.shape != descriptor_matrix.shape
    ):
        raise ValueError("descriptor and stationary matrices must be square")
    solve, _expm_multiply = _scipy_linalg()
    right = np.column_stack((-stationary_matrix, input_matrix))
    solved = solve(
        descriptor_matrix,
        right,
        assume_a="gen",
        check_finite=True,
    )
    dynamic = np.asarray(
        solved[:, : descriptor_matrix.shape[0]],
        dtype=float,
    )
    explicit_inputs = np.asarray(
        solved[:, descriptor_matrix.shape[0] :],
        dtype=float,
    )
    defect = descriptor_matrix @ solved - right
    relative_defect = float(
        np.linalg.norm(defect)
        / max(float(np.linalg.norm(right)), np.finfo(float).tiny)
    )
    return dynamic, explicit_inputs, relative_defect


def causal_stream_descriptor_inputs(
    weighted_stream_source_per_ct: np.ndarray,
    conservation_row_scales: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Map the linked stream amplitude and moment variations into DAE rows.

    The stationary conservation residual is ``flux divergence - source``.
    Therefore the explicit right-hand side receives the positive scaled source.
    The first column varies the complete physical stream state.  The remaining
    columns are infinitesimal fixed-mass variations of the three injected
    specific moments and are intended only as robustness inputs.
    """

    stream = _finite_matrix(
        weighted_stream_source_per_ct,
        name="weighted_stream_source_per_ct",
    )
    if stream.shape[1] != 4:
        raise ValueError("stream source must contain four Killing moments")
    row_scales = np.asarray(conservation_row_scales, dtype=float)
    expected_rows = 5 * stream.shape[0]
    if (
        row_scales.shape != (expected_rows,)
        or np.any(~np.isfinite(row_scales))
        or np.any(row_scales <= 0.0)
    ):
        raise ValueError("conservation row scales are invalid")

    physical = np.zeros((stream.shape[0], 5), dtype=float)
    physical[:, :4] = stream
    columns = [physical.ravel() / row_scales]
    names = ["physical_stream_amplitude"]
    for component, name in (
        (1, "specific_radial_momentum_variation"),
        (2, "specific_angular_momentum_variation"),
        (3, "specific_killing_energy_variation"),
    ):
        variation = np.zeros_like(physical)
        variation[:, component] = stream[:, component]
        columns.append(variation.ravel() / row_scales)
        names.append(name)
    return np.column_stack(columns), tuple(names)


def causal_log_time_quadrature(
    horizon_seconds: float,
    *,
    sample_count: int = 18,
    earliest_fraction: float = 1.0e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a nonuniform trapezoidal quadrature resolving early transients."""

    horizon = float(horizon_seconds)
    count = int(sample_count)
    fraction = float(earliest_fraction)
    if not np.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("time horizon must be positive and finite")
    if count < 3:
        raise ValueError("time quadrature requires at least three samples")
    if not np.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError("earliest time fraction must lie between zero and one")
    positive = np.geomspace(
        horizon * fraction,
        horizon,
        count - 1,
    )
    times = np.concatenate(([0.0], positive))
    weights = np.empty_like(times)
    weights[0] = 0.5 * (times[1] - times[0])
    weights[-1] = 0.5 * (times[-1] - times[-2])
    weights[1:-1] = 0.5 * (times[2:] - times[:-2])
    return times, weights


@dataclass(frozen=True)
class CausalMixedModeROM:
    """One explicit Petrov-Galerkin model with exact ledger coordinates."""

    trial_basis: np.ndarray
    test_basis: np.ndarray
    dynamic_matrix: np.ndarray
    input_matrix: np.ndarray
    output_matrix: np.ndarray
    hankel_singular_values: np.ndarray
    protected_coordinate_count: int
    horizon_seconds: float
    biorthogonality_defect: float
    protected_value_defect: float
    protected_dynamics_defect: float

    @property
    def order(self) -> int:
        return int(self.dynamic_matrix.shape[0])


@dataclass(frozen=True)
class CausalLyapunovMetricAudit:
    """Numerical certificate for one full-system Lyapunov energy metric."""

    metric: np.ndarray
    forcing: np.ndarray
    minimum_eigenvalue: float
    maximum_eigenvalue: float
    normalized_minimum_eigenvalue: float
    relative_residual: float
    positive_definite: bool
    residual_passed: bool
    accepted: bool


@dataclass(frozen=True)
class CausalStableRationalKrylovROM:
    """Observable-targeted rational model with ledger-safe stabilization."""

    trial_basis: np.ndarray
    test_basis: np.ndarray
    raw_dynamic_matrix: np.ndarray
    dynamic_matrix: np.ndarray
    input_matrix: np.ndarray
    output_matrix: np.ndarray
    protected_operators: np.ndarray
    rational_timescales_seconds: np.ndarray
    protected_coordinate_count: int
    raw_maximum_real_eigenvalue: float
    maximum_real_eigenvalue: float
    stabilization_succeeded: bool
    stabilization_penalty: float | None
    stabilization_correction_fraction: float
    biorthogonality_defect: float
    protected_value_defect: float
    protected_dynamics_defect: float
    candidate_singular_values: np.ndarray

    @property
    def order(self) -> int:
        return int(self.dynamic_matrix.shape[0])


def causal_lyapunov_metric_audit(
    dynamic: np.ndarray,
    *,
    state_weights: np.ndarray | None = None,
    positivity_tolerance: float = 1.0e-12,
    residual_tolerance: float = 1.0e-8,
) -> CausalLyapunovMetricAudit:
    """Audit a balanced dense Lyapunov metric without repairing its spectrum.

    A Hurwitz matrix has a positive-definite solution of
    ``A.T H + H A = -Q`` for every positive-definite ``Q``.  Strong
    non-normality can nevertheless make the dense numerical solve unusable.
    This routine reports that failure directly rather than clipping negative
    metric eigenvalues and calling the result a certificate.
    """

    system = _finite_matrix(dynamic, name="dynamic")
    if system.shape[0] != system.shape[1]:
        raise ValueError("dynamic matrix must be square")
    n_state = system.shape[0]
    if state_weights is None:
        weights = np.ones(n_state, dtype=float)
    else:
        weights = np.asarray(state_weights, dtype=float)
        if (
            weights.shape != (n_state,)
            or np.any(~np.isfinite(weights))
            or np.any(weights <= 0.0)
        ):
            raise ValueError("state weights must be positive and finite")
    positivity = float(positivity_tolerance)
    residual_gate = float(residual_tolerance)
    if (
        not np.isfinite(positivity)
        or positivity <= 0.0
        or not np.isfinite(residual_gate)
        or residual_gate <= 0.0
    ):
        raise ValueError("Lyapunov tolerances must be positive and finite")

    try:
        from scipy.linalg import (
            eigvalsh,
            matrix_balance,
            solve_continuous_lyapunov,
        )
    except ImportError as exc:  # pragma: no cover - solver extra missing
        raise RuntimeError(
            "scipy is required for the Lyapunov metric audit"
        ) from exc

    square_root = np.sqrt(weights)
    weighted = (
        square_root[:, None]
        * system
        / square_root[None, :]
    )
    balanced, transform = matrix_balance(
        weighted,
        permute=True,
        scale=True,
        separate=False,
    )
    balanced_metric = solve_continuous_lyapunov(
        balanced.T,
        -np.eye(n_state),
    )
    balanced_metric = 0.5 * (
        balanced_metric + balanced_metric.T
    )
    inverse_transform = np.linalg.inv(transform)
    metric = (
        inverse_transform.T
        @ balanced_metric
        @ inverse_transform
    )
    metric = 0.5 * (metric + metric.T)
    forcing = inverse_transform.T @ inverse_transform
    eigenvalues = eigvalsh(metric, driver="evr")
    minimum = float(eigenvalues[0])
    maximum = float(eigenvalues[-1])
    normalized_minimum = minimum / max(
        abs(maximum),
        np.finfo(float).tiny,
    )
    left = weighted.T @ metric
    right = metric @ weighted
    residual = left + right + forcing
    relative_residual = float(
        np.linalg.norm(residual)
        / max(
            np.linalg.norm(left)
            + np.linalg.norm(right)
            + np.linalg.norm(forcing),
            np.finfo(float).tiny,
        )
    )
    positive_definite = bool(
        minimum > positivity * max(abs(maximum), np.finfo(float).tiny)
    )
    residual_passed = bool(relative_residual <= residual_gate)
    return CausalLyapunovMetricAudit(
        metric=np.asarray(metric, dtype=float),
        forcing=np.asarray(forcing, dtype=float),
        minimum_eigenvalue=minimum,
        maximum_eigenvalue=maximum,
        normalized_minimum_eigenvalue=normalized_minimum,
        relative_residual=relative_residual,
        positive_definite=positive_definite,
        residual_passed=residual_passed,
        accepted=bool(positive_definite and residual_passed),
    )


def _weighted_rational_candidates(
    dynamic: np.ndarray,
    directions: np.ndarray,
    outputs: np.ndarray,
    timescales_seconds: np.ndarray,
) -> np.ndarray:
    try:
        from scipy.linalg import solve
    except ImportError as exc:  # pragma: no cover - solver extra missing
        raise RuntimeError(
            "scipy is required for rational Krylov reduction"
        ) from exc
    identity = np.eye(dynamic.shape[0])
    blocks = [directions, outputs.T]
    for timescale in timescales_seconds:
        shift = 1.0 / float(timescale)
        blocks.append(
            solve(
                shift * identity - dynamic,
                directions,
                assume_a="gen",
                check_finite=True,
            )
        )
        blocks.append(
            solve(
                shift * identity - dynamic.T,
                outputs.T,
                assume_a="gen",
                check_finite=True,
            )
        )
    candidates = np.column_stack(blocks)
    norms = np.linalg.norm(candidates, axis=0)
    active = norms > np.finfo(float).eps * max(
        float(np.max(norms)),
        1.0,
    )
    if not np.any(active):
        raise ValueError("rational Krylov candidates have zero rank")
    return candidates[:, active] / norms[active][None, :]


def _ledger_safe_lqr_stabilization(
    dynamic: np.ndarray,
    protected_coordinates: np.ndarray,
    *,
    stability_tolerance: float,
    penalties: tuple[float, ...],
) -> tuple[np.ndarray, bool, float | None, float]:
    try:
        from scipy.linalg import eigvals, null_space, solve_continuous_are
    except ImportError as exc:  # pragma: no cover - solver extra missing
        raise RuntimeError(
            "scipy is required for ledger-safe ROM stabilization"
        ) from exc

    raw = np.asarray(dynamic, dtype=float)
    maximum_real = float(np.max(np.real(eigvals(raw))))
    if maximum_real <= stability_tolerance:
        return raw.copy(), True, None, 0.0
    control = null_space(protected_coordinates)
    if control.shape[1] == 0:
        return raw.copy(), False, None, float("inf")

    candidates = []
    for penalty in penalties:
        rho = float(penalty)
        if not np.isfinite(rho) or rho <= 0.0:
            raise ValueError(
                "stabilization penalties must be positive and finite"
            )
        try:
            riccati = solve_continuous_are(
                raw,
                control,
                np.eye(raw.shape[0]),
                rho * np.eye(control.shape[1]),
            )
            correction = (
                control @ control.T @ riccati / rho
            )
            stabilized = raw - correction
            stabilized_maximum = float(
                np.max(np.real(eigvals(stabilized)))
            )
            fraction = float(
                np.linalg.norm(correction)
                / max(np.linalg.norm(raw), np.finfo(float).tiny)
            )
            if (
                np.all(np.isfinite(stabilized))
                and stabilized_maximum <= stability_tolerance
            ):
                candidates.append(
                    (fraction, rho, np.asarray(stabilized, dtype=float))
                )
        except (ValueError, np.linalg.LinAlgError):
            continue
    if not candidates:
        return raw.copy(), False, None, float("inf")
    fraction, penalty, stabilized = min(
        candidates,
        key=lambda row: row[0],
    )
    return stabilized, True, penalty, fraction


def causal_stable_rational_krylov_rom(
    dynamic: np.ndarray,
    inputs: np.ndarray,
    outputs: np.ndarray,
    protected_operators: np.ndarray,
    *,
    order: int,
    timescales_seconds: np.ndarray,
    state_weights: np.ndarray | None = None,
    initial_directions: np.ndarray | None = None,
    stability_tolerance: float = 1.0e-8,
    stabilization_penalties: tuple[float, ...] = (
        1.0e12,
        1.0e10,
        1.0e8,
        1.0e6,
        1.0e4,
        1.0e2,
        1.0,
    ),
) -> CausalStableRationalKrylovROM:
    """Build a rational Krylov model and stabilize only ledger-null rows.

    The basis is an orthogonal Galerkin basis in a cell-measure-weighted
    coordinate chart.  Exact ledger representers are inserted first.  If
    projection creates unstable poles, an LQR correction may act only in the
    null space of the reduced ledger coordinates.  The correction magnitude
    is returned as an authorization diagnostic; stability alone is not a
    fidelity claim.
    """

    system = _finite_matrix(dynamic, name="dynamic")
    if system.shape[0] != system.shape[1]:
        raise ValueError("dynamic matrix must be square")
    n_state = system.shape[0]
    input_matrix = _finite_matrix(inputs, name="inputs", rows=n_state)
    output_matrix = _finite_matrix(outputs, name="outputs")
    protected = _finite_matrix(
        protected_operators,
        name="protected_operators",
    )
    if (
        output_matrix.shape[1] != n_state
        or protected.shape[1] != n_state
    ):
        raise ValueError("output operators have incompatible dimensions")
    if initial_directions is None:
        directions = input_matrix
    else:
        initial = _finite_matrix(
            initial_directions,
            name="initial_directions",
            rows=n_state,
        )
        directions = np.column_stack((input_matrix, initial))
    timescales = np.asarray(timescales_seconds, dtype=float)
    if (
        timescales.ndim != 1
        or timescales.size == 0
        or np.any(~np.isfinite(timescales))
        or np.any(timescales <= 0.0)
    ):
        raise ValueError("rational timescales must be positive and finite")
    if state_weights is None:
        weights = np.ones(n_state, dtype=float)
    else:
        weights = np.asarray(state_weights, dtype=float)
        if (
            weights.shape != (n_state,)
            or np.any(~np.isfinite(weights))
            or np.any(weights <= 0.0)
        ):
            raise ValueError("state weights must be positive and finite")
    target_order = int(order)
    protected_count = int(protected.shape[0])
    if not protected_count <= target_order <= n_state:
        raise ValueError("order cannot represent protected coordinates")

    try:
        from scipy.linalg import eigvals, qr, svd
    except ImportError as exc:  # pragma: no cover - solver extra missing
        raise RuntimeError(
            "scipy is required for rational Krylov reduction"
        ) from exc
    square_root = np.sqrt(weights)
    inverse_square_root = 1.0 / square_root
    weighted_dynamic = (
        square_root[:, None]
        * system
        * inverse_square_root[None, :]
    )
    weighted_directions = square_root[:, None] * directions
    weighted_outputs = output_matrix * inverse_square_root[None, :]
    weighted_protected = protected * inverse_square_root[None, :]
    protected_trial, _ = qr(
        weighted_protected.T,
        mode="economic",
    )
    if protected_trial.shape[1] != protected_count:
        raise ValueError("protected operators are linearly dependent")
    projector = np.eye(n_state) - protected_trial @ protected_trial.T
    candidates = projector @ _weighted_rational_candidates(
        weighted_dynamic,
        weighted_directions,
        weighted_outputs,
        timescales,
    )
    candidate_norms = np.linalg.norm(candidates, axis=0)
    active = candidate_norms > np.finfo(float).eps * max(
        float(np.max(candidate_norms)),
        1.0,
    )
    candidates = (
        candidates[:, active] / candidate_norms[active][None, :]
    )
    left, singular, _right = svd(candidates, full_matrices=False)
    dynamic_count = target_order - protected_count
    threshold = (
        np.finfo(float).eps
        * max(candidates.shape)
        * max(float(singular[0]), 1.0)
    )
    available = int(np.count_nonzero(singular > threshold))
    if available < dynamic_count:
        raise ValueError(
            "rational Krylov candidates have insufficient numerical rank "
            f"({available} available, {dynamic_count} requested)"
        )
    weighted_trial = np.column_stack(
        (protected_trial, left[:, :dynamic_count])
    )
    weighted_trial, _ = qr(weighted_trial, mode="economic")
    trial = inverse_square_root[:, None] * weighted_trial
    test = square_root[:, None] * weighted_trial
    raw_dynamic = test.T @ system @ trial
    reduced_protected = protected @ trial
    stabilized, succeeded, penalty, correction = (
        _ledger_safe_lqr_stabilization(
            raw_dynamic,
            reduced_protected,
            stability_tolerance=float(stability_tolerance),
            penalties=stabilization_penalties,
        )
    )
    raw_maximum = float(np.max(np.real(eigvals(raw_dynamic))))
    maximum = float(np.max(np.real(eigvals(stabilized))))
    projection = trial @ test.T
    protected_scale = max(
        float(np.linalg.norm(protected)),
        np.finfo(float).tiny,
    )
    protected_dynamics_scale = max(
        float(np.linalg.norm(reduced_protected @ raw_dynamic)),
        np.finfo(float).tiny,
    )
    return CausalStableRationalKrylovROM(
        trial_basis=np.asarray(trial, dtype=float),
        test_basis=np.asarray(test, dtype=float),
        raw_dynamic_matrix=np.asarray(raw_dynamic, dtype=float),
        dynamic_matrix=np.asarray(stabilized, dtype=float),
        input_matrix=np.asarray(test.T @ input_matrix, dtype=float),
        output_matrix=np.asarray(output_matrix @ trial, dtype=float),
        protected_operators=np.asarray(protected, dtype=float),
        rational_timescales_seconds=timescales.copy(),
        protected_coordinate_count=protected_count,
        raw_maximum_real_eigenvalue=raw_maximum,
        maximum_real_eigenvalue=maximum,
        stabilization_succeeded=succeeded,
        stabilization_penalty=penalty,
        stabilization_correction_fraction=correction,
        biorthogonality_defect=float(
            np.max(
                np.abs(
                    test.T @ trial - np.eye(target_order)
                )
            )
        ),
        protected_value_defect=float(
            np.linalg.norm(protected @ projection - protected)
            / protected_scale
        ),
        protected_dynamics_defect=float(
            np.linalg.norm(
                reduced_protected @ (stabilized - raw_dynamic)
            )
            / protected_dynamics_scale
        ),
        candidate_singular_values=np.asarray(singular, dtype=float),
    )


def causal_truncate_mixed_mode_rom(
    rom: CausalMixedModeROM,
    dynamic: np.ndarray,
    inputs: np.ndarray,
    outputs: np.ndarray,
    *,
    order: int,
) -> CausalMixedModeROM:
    """Truncate one BPOD ladder without recomputing its snapshots."""

    system = _finite_matrix(dynamic, name="dynamic")
    input_matrix = _finite_matrix(
        inputs,
        name="inputs",
        rows=system.shape[0],
    )
    output_matrix = _finite_matrix(outputs, name="outputs")
    target_order = int(order)
    if not rom.protected_coordinate_count <= target_order <= rom.order:
        raise ValueError("truncated order is incompatible with the ROM")
    if (
        system.shape != (rom.trial_basis.shape[0],) * 2
        or output_matrix.shape[1] != system.shape[0]
    ):
        raise ValueError("full matrices and ROM dimensions differ")
    trial = np.asarray(rom.trial_basis[:, :target_order], dtype=float)
    test = np.asarray(rom.test_basis[:, :target_order], dtype=float)
    reduced_dynamic = test.T @ system @ trial
    protected = rom.protected_coordinate_count
    return CausalMixedModeROM(
        trial_basis=trial,
        test_basis=test,
        dynamic_matrix=reduced_dynamic,
        input_matrix=test.T @ input_matrix,
        output_matrix=output_matrix @ trial,
        hankel_singular_values=rom.hankel_singular_values,
        protected_coordinate_count=protected,
        horizon_seconds=rom.horizon_seconds,
        biorthogonality_defect=float(
            np.max(np.abs(test.T @ trial - np.eye(target_order)))
        ),
        protected_value_defect=rom.protected_value_defect,
        protected_dynamics_defect=float(
            np.max(
                np.abs(
                    reduced_dynamic[:protected]
                    - rom.dynamic_matrix[:protected, :target_order]
                )
            )
        ),
    )


def _snapshot_blocks(
    dynamic: np.ndarray,
    directions: np.ndarray,
    times: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    _solve, expm_multiply = _scipy_linalg()
    blocks = []
    for time_seconds, weight in zip(times, weights, strict=True):
        evolved = expm_multiply(
            float(time_seconds) * dynamic,
            directions,
            traceA=float(time_seconds) * float(np.trace(dynamic)),
        )
        blocks.append(np.sqrt(float(weight)) * evolved)
    return np.column_stack(blocks)


def causal_conservation_constrained_balanced_rom(
    dynamic: np.ndarray,
    inputs: np.ndarray,
    outputs: np.ndarray,
    protected_operators: np.ndarray,
    *,
    order: int,
    horizon_seconds: float,
    state_weights: np.ndarray | None = None,
    initial_directions: np.ndarray | None = None,
    sample_count: int = 18,
    singular_value_tolerance: float = 1.0e-12,
    allow_rank_truncation: bool = False,
) -> CausalMixedModeROM:
    """Build a finite-horizon BPOD model with exact protected coordinates."""

    system = _finite_matrix(dynamic, name="dynamic")
    if system.shape[0] != system.shape[1]:
        raise ValueError("dynamic matrix must be square")
    n_state = system.shape[0]
    input_matrix = _finite_matrix(inputs, name="inputs", rows=n_state)
    output_matrix = _finite_matrix(outputs, name="outputs")
    if output_matrix.shape[1] != n_state:
        raise ValueError("outputs have an incompatible column count")
    protected = _finite_matrix(
        protected_operators,
        name="protected_operators",
    )
    if protected.shape[1] != n_state:
        raise ValueError("protected operators have an incompatible column count")
    protected_count = protected.shape[0]
    target_order = int(order)
    if not protected_count <= target_order <= n_state:
        raise ValueError("reduced order cannot represent protected coordinates")
    if initial_directions is None:
        primal_directions = input_matrix
    else:
        initial = _finite_matrix(
            initial_directions,
            name="initial_directions",
            rows=n_state,
        )
        primal_directions = np.column_stack((input_matrix, initial))
    if primal_directions.shape[1] == 0 or output_matrix.shape[0] == 0:
        raise ValueError("reduction requires inputs and outputs")

    if state_weights is None:
        weights = np.ones(n_state, dtype=float)
    else:
        weights = np.asarray(state_weights, dtype=float)
        if (
            weights.shape != (n_state,)
            or np.any(~np.isfinite(weights))
            or np.any(weights <= 0.0)
        ):
            raise ValueError("state weights must be positive and finite")
    square_root = np.sqrt(weights)
    inverse_square_root = 1.0 / square_root
    weighted_dynamic = (
        square_root[:, None]
        * system
        * inverse_square_root[None, :]
    )
    weighted_primal = square_root[:, None] * primal_directions
    weighted_outputs = output_matrix * inverse_square_root[None, :]
    weighted_protected = protected * inverse_square_root[None, :]

    gram = weighted_protected @ weighted_protected.T
    protected_rank = np.linalg.matrix_rank(gram)
    if protected_rank != protected_count:
        raise ValueError("protected operators are linearly dependent")
    protected_trial = weighted_protected.T @ np.linalg.inv(gram)
    null_projector = (
        np.eye(n_state) - protected_trial @ weighted_protected
    )

    times, time_weights = causal_log_time_quadrature(
        horizon_seconds,
        sample_count=sample_count,
    )
    primal_snapshots = null_projector @ _snapshot_blocks(
        weighted_dynamic,
        weighted_primal,
        times,
        time_weights,
    )
    adjoint_snapshots = null_projector @ _snapshot_blocks(
        weighted_dynamic.T,
        weighted_outputs.T,
        times,
        time_weights,
    )
    cross_gram = adjoint_snapshots.T @ primal_snapshots
    left, singular, right_h = np.linalg.svd(
        cross_gram,
        full_matrices=False,
    )
    dynamic_count = target_order - protected_count
    if dynamic_count:
        largest = float(singular[0]) if singular.size else 0.0
        threshold = max(
            float(singular_value_tolerance) * largest,
            np.finfo(float).eps * max(cross_gram.shape) * largest,
        )
        available = int(np.count_nonzero(singular > threshold))
        if available < dynamic_count:
            if not allow_rank_truncation:
                raise ValueError(
                    "snapshot Hankel matrix has insufficient numerical rank "
                    f"({available} available, {dynamic_count} requested)"
                )
            dynamic_count = available
            target_order = protected_count + dynamic_count
        inverse_sqrt = 1.0 / np.sqrt(singular[:dynamic_count])
        dynamic_trial = (
            primal_snapshots
            @ right_h.T[:, :dynamic_count]
            * inverse_sqrt[None, :]
        )
        dynamic_test = (
            adjoint_snapshots
            @ left[:, :dynamic_count]
            * inverse_sqrt[None, :]
        )
        dynamic_trial = null_projector @ dynamic_trial
        dynamic_test = null_projector @ dynamic_test
        dynamic_overlap = dynamic_test.T @ dynamic_trial
        dynamic_test = dynamic_test @ np.linalg.inv(dynamic_overlap.T)
    else:
        dynamic_trial = np.empty((n_state, 0), dtype=float)
        dynamic_test = np.empty((n_state, 0), dtype=float)

    weighted_trial = np.column_stack(
        (protected_trial, dynamic_trial)
    )
    weighted_test = np.column_stack(
        (weighted_protected.T, dynamic_test)
    )
    trial = inverse_square_root[:, None] * weighted_trial
    test = square_root[:, None] * weighted_test
    reduced_dynamic = test.T @ system @ trial
    reduced_inputs = test.T @ input_matrix
    reduced_outputs = output_matrix @ trial
    identity = np.eye(target_order)
    protected_selector = np.zeros((protected_count, target_order))
    protected_selector[:, :protected_count] = np.eye(protected_count)
    biorthogonality_defect = float(
        np.max(np.abs(test.T @ trial - identity))
    )
    protected_value_defect = float(
        np.max(np.abs(protected @ trial - protected_selector))
    )
    protected_dynamics_defect = float(
        np.max(
            np.abs(
                reduced_dynamic[:protected_count]
                - protected @ system @ trial
            )
        )
    )
    return CausalMixedModeROM(
        trial_basis=trial,
        test_basis=test,
        dynamic_matrix=reduced_dynamic,
        input_matrix=reduced_inputs,
        output_matrix=reduced_outputs,
        hankel_singular_values=np.asarray(singular, dtype=float),
        protected_coordinate_count=protected_count,
        horizon_seconds=float(horizon_seconds),
        biorthogonality_defect=biorthogonality_defect,
        protected_value_defect=protected_value_defect,
        protected_dynamics_defect=protected_dynamics_defect,
    )


def causal_linear_initial_response(
    dynamic: np.ndarray,
    initial_state: np.ndarray,
    times_seconds: np.ndarray,
) -> np.ndarray:
    """Return state responses with shape ``(time, state, direction)``."""

    system = _finite_matrix(dynamic, name="dynamic")
    initial = _finite_matrix(
        initial_state,
        name="initial_state",
        rows=system.shape[0],
    )
    times = np.asarray(times_seconds, dtype=float)
    if times.ndim != 1 or np.any(~np.isfinite(times)) or np.any(times < 0.0):
        raise ValueError("response times must be finite and nonnegative")
    _solve, expm_multiply = _scipy_linalg()
    return np.stack(
        [
            expm_multiply(
                float(time_seconds) * system,
                initial,
                traceA=float(time_seconds) * float(np.trace(system)),
            )
            for time_seconds in times
        ],
        axis=0,
    )


def causal_rom_initial_response(
    rom: CausalMixedModeROM,
    initial_state: np.ndarray,
    times_seconds: np.ndarray,
) -> np.ndarray:
    """Return reconstructed ROM responses for full-state initial directions."""

    initial = _finite_matrix(
        initial_state,
        name="initial_state",
        rows=rom.trial_basis.shape[0],
    )
    reduced_initial = rom.test_basis.T @ initial
    reduced = causal_linear_initial_response(
        rom.dynamic_matrix,
        reduced_initial,
        times_seconds,
    )
    return np.einsum(
        "nr,trd->tnd",
        rom.trial_basis,
        reduced,
        optimize=True,
    )


def causal_stable_rom_initial_response(
    rom: CausalStableRationalKrylovROM,
    initial_state: np.ndarray,
    times_seconds: np.ndarray,
) -> np.ndarray:
    """Return reconstructed stable-ROM responses to full-state directions."""

    initial = _finite_matrix(
        initial_state,
        name="initial_state",
        rows=rom.trial_basis.shape[0],
    )
    reduced = causal_linear_initial_response(
        rom.dynamic_matrix,
        rom.test_basis.T @ initial,
        times_seconds,
    )
    return np.einsum(
        "nr,trd->tnd",
        rom.trial_basis,
        reduced,
        optimize=True,
    )


def causal_linear_transfer_response(
    dynamic: np.ndarray,
    directions: np.ndarray,
    outputs: np.ndarray,
    timescales_seconds: np.ndarray,
) -> np.ndarray:
    """Return ``C (s I - A)^-1 D`` at ``s=1/timescale``."""

    system = _finite_matrix(dynamic, name="dynamic")
    if system.shape[0] != system.shape[1]:
        raise ValueError("dynamic matrix must be square")
    direction_matrix = _finite_matrix(
        directions,
        name="directions",
        rows=system.shape[0],
    )
    output_matrix = _finite_matrix(outputs, name="outputs")
    if output_matrix.shape[1] != system.shape[0]:
        raise ValueError("outputs have an incompatible column count")
    timescales = np.asarray(timescales_seconds, dtype=float)
    if (
        timescales.ndim != 1
        or timescales.size == 0
        or np.any(~np.isfinite(timescales))
        or np.any(timescales <= 0.0)
    ):
        raise ValueError("transfer timescales must be positive and finite")
    solve, _expm_multiply = _scipy_linalg()
    identity = np.eye(system.shape[0])
    return np.stack(
        [
            output_matrix
            @ solve(
                identity / float(timescale) - system,
                direction_matrix,
                assume_a="gen",
                check_finite=True,
            )
            for timescale in timescales
        ],
        axis=0,
    )


def causal_stable_rom_transfer_response(
    rom: CausalStableRationalKrylovROM,
    directions: np.ndarray,
    timescales_seconds: np.ndarray,
    *,
    outputs: np.ndarray | None = None,
) -> np.ndarray:
    """Return stable-ROM transfer responses for full-state directions."""

    direction_matrix = _finite_matrix(
        directions,
        name="directions",
        rows=rom.trial_basis.shape[0],
    )
    if outputs is None:
        reduced_outputs = rom.output_matrix
    else:
        full_outputs = _finite_matrix(outputs, name="outputs")
        if full_outputs.shape[1] != rom.trial_basis.shape[0]:
            raise ValueError("outputs have an incompatible column count")
        reduced_outputs = full_outputs @ rom.trial_basis
    return causal_linear_transfer_response(
        rom.dynamic_matrix,
        rom.test_basis.T @ direction_matrix,
        reduced_outputs,
        timescales_seconds,
    )


def causal_rom_memory_kernel_actions(
    dynamic: np.ndarray,
    rom: CausalMixedModeROM,
    times_seconds: np.ndarray,
) -> np.ndarray:
    """Return exact oblique unresolved feedback actions on retained states."""

    system = _finite_matrix(dynamic, name="dynamic")
    if system.shape != (
        rom.trial_basis.shape[0],
        rom.trial_basis.shape[0],
    ):
        raise ValueError("dynamic matrix and ROM dimensions differ")
    times = np.asarray(times_seconds, dtype=float)
    if times.ndim != 1 or np.any(~np.isfinite(times)) or np.any(times < 0.0):
        raise ValueError("kernel times must be finite and nonnegative")
    projection = rom.trial_basis @ rom.test_basis.T
    complement = np.eye(system.shape[0]) - projection
    orthogonal_dynamic = complement @ system @ complement
    unresolved_coupling = complement @ system @ rom.trial_basis
    return np.stack(
        [
            rom.test_basis.T
            @ system
            @ complement
            @ evolved
            for evolved in causal_linear_initial_response(
                orthogonal_dynamic,
                unresolved_coupling,
                times,
            )
        ],
        axis=0,
    )
