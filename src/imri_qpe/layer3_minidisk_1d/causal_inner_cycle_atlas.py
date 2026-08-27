"""Structure-preserving interpolation for the hybrid reduced cycle atlas.

All interpolants use nonnegative simplex weights.  The continuous branch
state is corrected in the certified conservation normal so that its four
retained invariants are exact.  Symmetric principal matrices, dissipative
sources with a common four-coordinate kernel, driver ledgers, and event
reset ledgers are therefore preserved algebraically.  No extrapolating or
nearest-neighbour fallback is provided.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .causal_inner_entropy_characteristic_boundary import (
    build_outward_entropy_characteristic_boundary,
)


Array = np.ndarray
TWO_PI = 2.0 * np.pi


def _finite(value, *, ndim: int, name: str, dtype=float) -> Array:
    result = np.asarray(value, dtype=dtype)
    if result.ndim != ndim or (dtype is not int and np.any(~np.isfinite(result))):
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    return result


def _relative(left, right) -> float:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    return float(
        np.linalg.norm(a - b)
        / max(np.linalg.norm(a), np.linalg.norm(b), np.finfo(float).tiny)
    )


def _unwrap_around(values: Array, center: float) -> Array:
    raw = np.asarray(values, dtype=float)
    return float(center) + (raw - float(center) + np.pi) % TWO_PI - np.pi


@dataclass(frozen=True)
class SimplexLocation:
    simplex_index: int
    vertex_indices: Array
    weights: Array
    minimum_weight: float
    weight_sum_defect: float
    coordinate_reproduction_defect: float
    maximum_vertex_distance: float


@dataclass(frozen=True)
class GuardSheetLocation:
    simplex_index: int
    vertex_indices: Array
    weights: Array
    oriented_normal: Array
    signed_guard_distance: float
    affine_hull_reproduction_defect: float
    minimum_weight: float
    weight_sum_defect: float


@dataclass(frozen=True)
class CycleDriverValue:
    phase_rate_per_second: float
    slow_forcing_per_second: Array
    distributed_ledger_rate: Array
    boundary_ledger_rate: Array
    outer_incoming_characteristics: Array
    q_location: SimplexLocation
    phase_indices: Array
    phase_weights: Array
    forcing_ledger_relative_defect: float


@dataclass(frozen=True)
class CycleBranchValue:
    state: Array
    radial_matrices: Array
    source_matrices: Array
    forcing_per_second: Array
    trust_radius: float
    fast_spectral_gap_per_second: float
    guard_margins: Array
    location: SimplexLocation
    invariant_relative_defect: float
    maximum_radial_symmetry_defect: float
    maximum_source_entropy_positive_eigenvalue: float
    minimum_source_nullity: int
    inner_incoming_count: int
    outer_incoming_count: int


@dataclass(frozen=True)
class CycleEventValue:
    duration_seconds: float
    ledger_impulse: Array
    ledger_null_constitutive_jump: Array
    post_state: Array
    source_mode_index: int
    destination_mode_index: int
    transition_class_index: int
    guard: GuardSheetLocation
    transversality: float
    reset_ledger_relative_defect: float
    constitutive_null_relative_defect: float


def _barycentric(vertices: Array, query: Array) -> tuple[Array, float, float]:
    values = _finite(vertices, ndim=2, name="simplex vertices")
    target = _finite(query, ndim=1, name="simplex query")
    if values.shape != (len(target) + 1, len(target)):
        raise ValueError("a full simplex needs dimension+1 vertices")
    for index, vertex in enumerate(values):
        if np.array_equal(vertex, target):
            weights = np.zeros(len(values)); weights[index] = 1.0
            return weights, 0.0, 0.0
    augmented = np.vstack((values.T, np.ones(len(values))))
    rhs = np.concatenate((target, [1.0]))
    try:
        weights = np.linalg.solve(augmented, rhs)
    except np.linalg.LinAlgError as error:
        raise ValueError("simplex is affinely singular") from error
    reproduced = weights @ values
    defect = _relative(reproduced, target)
    return weights, defect, abs(float(np.sum(weights)) - 1.0)


def locate_full_simplex(
    nodes,
    simplices,
    query,
    *,
    simplex_labels=None,
    label: int | None = None,
    tolerance: float = 2.0e-12,
) -> SimplexLocation:
    points = _finite(nodes, ndim=2, name="nodes")
    cells = _finite(simplices, ndim=2, name="simplices", dtype=int)
    target = _finite(query, ndim=1, name="query")
    if points.shape[1] != len(target) or cells.shape[1] != len(target) + 1:
        raise ValueError("simplex dimensions disagree")
    if np.any(cells < 0) or np.any(cells >= len(points)):
        raise ValueError("simplex vertex index is out of range")
    labels = None
    if simplex_labels is not None:
        labels = _finite(simplex_labels, ndim=1, name="simplex labels", dtype=int)
        if labels.shape != (len(cells),) or label is None:
            raise ValueError("simplex labels need one requested label")
    candidates: list[tuple[float, int, SimplexLocation]] = []
    for simplex_index, indices in enumerate(cells):
        if labels is not None and int(labels[simplex_index]) != int(label):
            continue
        vertices = points[indices]
        try:
            weights, defect, sum_defect = _barycentric(vertices, target)
        except ValueError:
            continue
        minimum = float(np.min(weights))
        if minimum < -float(tolerance) or defect > float(tolerance) or sum_defect > float(tolerance):
            continue
        weights = np.where(np.abs(weights) <= tolerance, 0.0, weights)
        weights /= np.sum(weights)
        distance = float(np.max(np.linalg.norm(vertices - target, axis=1)))
        location = SimplexLocation(
            simplex_index,
            np.array(indices, copy=True),
            weights,
            float(np.min(weights)),
            abs(float(np.sum(weights)) - 1.0),
            _relative(weights @ vertices, target),
            distance,
        )
        candidates.append((-location.minimum_weight, simplex_index, location))
    if not candidates:
        raise ValueError("query lies outside every admissible simplex")
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def periodic_phase_weights(phase_nodes, phase: float) -> tuple[Array, Array]:
    nodes = _finite(phase_nodes, ndim=1, name="phase nodes")
    if len(nodes) < 3 or nodes[0] != 0.0 or nodes[-1] != TWO_PI or np.any(np.diff(nodes) <= 0.0):
        raise ValueError("periodic phase nodes must increase from zero to two pi")
    raw = float(phase)
    if not np.isfinite(raw):
        raise ValueError("phase must be finite")
    value = raw % TWO_PI
    if value == 0.0:
        return np.asarray((0,), dtype=int), np.asarray((1.0,))
    exact = np.flatnonzero(nodes == value)
    if len(exact):
        return np.asarray((int(exact[0]),), dtype=int), np.asarray((1.0,))
    left = int(np.searchsorted(nodes, value, side="right") - 1)
    right = left + 1
    fraction = (value - nodes[left]) / (nodes[right] - nodes[left])
    return np.asarray((left, right), dtype=int), np.asarray((1.0 - fraction, fraction))


def _weighted(values: Array, indices: Array, weights: Array) -> Array:
    exact = np.flatnonzero(weights == 1.0)
    if len(exact) == 1 and np.count_nonzero(weights) == 1:
        return np.array(np.asarray(values)[indices[int(exact[0])]], copy=True)
    return np.tensordot(weights, np.asarray(values)[indices], axes=(0, 0))


def interpolate_cycle_driver(
    driver,
    *,
    q_simplices,
    q_scales,
    query_invariants,
    phase: float,
    mode_index: int,
    conservation_map,
) -> CycleDriverValue:
    phases = _finite(driver["phase_nodes"], ndim=1, name="phase nodes")
    q_nodes = _finite(driver["retained_invariant_nodes4"], ndim=2, name="q nodes")
    scales = _finite(q_scales, ndim=1, name="q scales")
    target = _finite(query_invariants, ndim=1, name="query invariants")
    conservation = _finite(conservation_map, ndim=2, name="conservation map")
    if q_nodes.shape[1] != 4 or scales.shape != (4,) or target.shape != (4,) or np.any(scales <= 0.0):
        raise ValueError("driver invariant coordinates must have dimension four")
    q_location = locate_full_simplex(
        q_nodes / scales,
        q_simplices,
        target / scales,
    )
    phase_indices, phase_weights = periodic_phase_weights(phases, phase)
    mode = int(mode_index)

    def tensor(name: str):
        value = np.asarray(driver[name])
        if value.shape[0] != len(phases) or value.shape[1] != len(q_nodes) or mode < 0 or mode >= value.shape[2]:
            raise ValueError(f"driver field {name} has incompatible dimensions")
        phase_values = np.asarray(
            [
                _weighted(value[index, :, mode], q_location.vertex_indices, q_location.weights)
                for index in phase_indices
            ]
        )
        return _weighted(phase_values, np.arange(len(phase_values)), phase_weights)

    rates = _finite(driver["phase_rate_per_second"], ndim=1, name="phase rate")
    rate = float(_weighted(rates, phase_indices, phase_weights))
    forcing = np.asarray(tensor("slow_forcing1232_per_second"), dtype=float)
    distributed = np.asarray(tensor("distributed_source_ledger_rate4"), dtype=float)
    boundary = np.asarray(tensor("boundary_ledger_rate4"), dtype=float)
    incoming = np.asarray(tensor("outer_incoming_characteristics11"), dtype=float)
    if rate <= 0.0 or conservation.shape[1] != len(forcing):
        raise ValueError("interpolated driver is not physically oriented")
    ledger_defect = _relative(conservation @ forcing, distributed + boundary)
    if ledger_defect > 2.0e-12:
        raise ValueError("interpolated driver forcing does not close its ledger")
    return CycleDriverValue(
        rate,
        forcing,
        distributed,
        boundary,
        incoming,
        q_location,
        phase_indices,
        phase_weights,
        ledger_defect,
    )


def locate_periodic_branch_simplex(
    anchor_invariants,
    anchor_phase,
    simplices,
    simplex_modes,
    *,
    q_scales,
    phase_scale: float,
    query_invariants,
    phase: float,
    mode_index: int,
    tolerance: float = 2.0e-12,
) -> SimplexLocation:
    q = _finite(anchor_invariants, ndim=2, name="anchor invariants")
    phases = _finite(anchor_phase, ndim=1, name="anchor phase")
    cells = _finite(simplices, ndim=2, name="branch simplices", dtype=int)
    modes = _finite(simplex_modes, ndim=1, name="branch simplex modes", dtype=int)
    scales = _finite(q_scales, ndim=1, name="branch q scales")
    target_q = _finite(query_invariants, ndim=1, name="query invariants")
    pscale = float(phase_scale)
    center = float(phase) % TWO_PI
    if q.shape[1] != 4 or target_q.shape != (4,) or scales.shape != (4,) or np.any(scales <= 0.0) or pscale <= 0.0 or cells.shape[1] != 6 or modes.shape != (len(cells),):
        raise ValueError("branch simplex coordinate dimensions disagree")
    target = np.concatenate((target_q / scales, [center / pscale]))
    candidates = []
    for simplex_index, indices in enumerate(cells):
        if int(modes[simplex_index]) != int(mode_index):
            continue
        unwrapped = _unwrap_around(phases[indices], center)
        vertices = np.column_stack((q[indices] / scales, unwrapped / pscale))
        try:
            weights, defect, sum_defect = _barycentric(vertices, target)
        except ValueError:
            continue
        if np.min(weights) < -tolerance or defect > tolerance or sum_defect > tolerance:
            continue
        weights = np.where(np.abs(weights) <= tolerance, 0.0, weights); weights /= np.sum(weights)
        location = SimplexLocation(simplex_index, np.array(indices, copy=True), weights, float(np.min(weights)), abs(float(np.sum(weights)) - 1.0), _relative(weights @ vertices, target), float(np.max(np.linalg.norm(vertices - target, axis=1))))
        candidates.append((-location.minimum_weight, simplex_index, location))
    if not candidates: raise ValueError("branch query lies outside every mode-pure periodic simplex")
    candidates.sort(key=lambda item: (item[0], item[1])); return candidates[0][2]


def interpolate_cycle_branch(
    branch,
    *,
    branch_simplices,
    branch_simplex_modes,
    q_scales,
    phase_scale: float,
    query_invariants,
    phase: float,
    mode_index: int,
    conservation_map,
    minimum_norm_normal,
    trust_tolerance: float = 2.0e-12,
) -> CycleBranchValue:
    conservation = _finite(conservation_map, ndim=2, name="conservation map")
    normal = _finite(minimum_norm_normal, ndim=2, name="minimum-norm normal")
    target = _finite(query_invariants, ndim=1, name="query invariants")
    states = _finite(branch["anchor_states1232"], ndim=2, name="anchor states")
    invariants = _finite(branch["anchor_invariants4"], ndim=2, name="anchor invariants")
    phases = _finite(branch["anchor_phase"], ndim=1, name="anchor phase")
    modes = _finite(branch["anchor_mode_index"], ndim=1, name="anchor modes", dtype=int)
    if conservation.shape != (4, states.shape[1]) or normal.shape != (states.shape[1], 4) or target.shape != (4,):
        raise ValueError("branch conservation geometry is inconsistent")
    if _relative(conservation @ normal, np.eye(4)) > 2.0e-12:
        raise ValueError("minimum-norm normal does not invert the conservation map")
    location = locate_periodic_branch_simplex(
        invariants,
        phases,
        branch_simplices,
        branch_simplex_modes,
        q_scales=q_scales,
        phase_scale=phase_scale,
        query_invariants=target,
        phase=phase,
        mode_index=mode_index,
    )
    indices = location.vertex_indices; weights = location.weights
    if np.any(modes[indices] != int(mode_index)):
        raise ValueError("branch simplex mixes discrete modes")
    trust_values = _finite(branch["trust_radii"], ndim=1, name="trust radii")
    if location.maximum_vertex_distance > float(np.min(trust_values[indices])) + trust_tolerance:
        raise ValueError("branch query exceeds a contributing anchor trust radius")
    state = _weighted(states, indices, weights)
    raw_invariant_defect = _relative(conservation @ state, target)
    if not (
        np.count_nonzero(weights) == 1
        and np.any(weights == 1.0)
        and raw_invariant_defect <= 2.0e-12
    ):
        state = state + normal @ (target - conservation @ state)
    radial = _weighted(_finite(branch["radial_matrices112x11x11"], ndim=4, name="radial matrices"), indices, weights)
    source = _weighted(_finite(branch["source_matrices112x11x11"], ndim=4, name="source matrices"), indices, weights)
    forcing = _weighted(_finite(branch["forcing1232_per_second"], ndim=2, name="branch forcing"), indices, weights)
    gap = float(_weighted(_finite(branch["stable_spectral_gaps_per_second"], ndim=1, name="spectral gaps"), indices, weights))
    trust = float(_weighted(trust_values, indices, weights))
    margins = _weighted(_finite(branch["guard_margins"], ndim=2, name="guard margins"), indices, weights)
    if gap <= 0.0:
        raise ValueError("interpolated branch lost normal hyperbolicity")
    if radial.shape != (112, 11, 11) or source.shape != (112, 11, 11):
        raise ValueError("interpolated branch has the wrong native port shape")
    if max(np.linalg.norm(source[:, :4, :]), np.linalg.norm(source[:, :, :4])) > 2.0e-12:
        raise ValueError("local source matrices do not share the four-coordinate kernel")
    radial_symmetry = float(np.max(np.linalg.norm(radial - radial.transpose(0, 2, 1), axis=(1, 2))))
    source_positive = float(max(np.max(np.linalg.eigvalsh(0.5 * (value + value.T))) for value in source))
    nullity = min(11 - np.linalg.matrix_rank(value, tol=1.0e-10) for value in source)
    inner = build_outward_entropy_characteristic_boundary(radial[0], outward_normal=-1.0)
    outer = build_outward_entropy_characteristic_boundary(radial[-1], outward_normal=1.0)
    invariant_defect = _relative(conservation @ state, target)
    if radial_symmetry > 2.0e-12 or source_positive > 2.0e-12 or nullity < 4 or inner.incoming_count != 0 or outer.incoming_count != 11 or invariant_defect > 2.0e-12:
        raise ValueError("interpolated branch failed a structure gate")
    return CycleBranchValue(state, radial, source, forcing, trust, gap, margins, location, invariant_defect, radial_symmetry, source_positive, int(nullity), inner.incoming_count, outer.incoming_count)


def locate_guard_sheet(
    nodes,
    simplices,
    truth_normals,
    query,
    *,
    simplex_classes=None,
    transition_class: int | None = None,
    tolerance: float = 2.0e-12,
) -> GuardSheetLocation:
    points = _finite(nodes, ndim=2, name="guard nodes")
    cells = _finite(simplices, ndim=2, name="guard simplices", dtype=int)
    normals = _finite(truth_normals, ndim=2, name="truth guard normals")
    target = _finite(query, ndim=1, name="guard query")
    if points.shape[1] != 5 or target.shape != (5,) or cells.shape[1] != 5 or normals.shape != points.shape:
        raise ValueError("guard sheet needs five-dimensional points and five-vertex elements")
    classes = None
    if simplex_classes is not None:
        classes = _finite(simplex_classes, ndim=1, name="guard simplex classes", dtype=int)
        if classes.shape != (len(cells),) or transition_class is None:
            raise ValueError("guard classes need one requested transition class")
    candidates = []
    for simplex_index, indices in enumerate(cells):
        if classes is not None and int(classes[simplex_index]) != int(transition_class): continue
        vertices = points[indices]; edges = vertices[1:] - vertices[0]
        _u, singular, vh = np.linalg.svd(edges, full_matrices=True)
        if len(singular) != 4 or singular[-1] <= tolerance * max(singular[0], 1.0): continue
        normal = vh[-1]; truth = np.mean(normals[indices], axis=0)
        if float(normal @ truth) < 0.0: normal = -normal
        signed = float(normal @ (target - vertices[0])); projection = target - signed * normal
        augmented = np.vstack((vertices.T, np.ones(5))); rhs = np.concatenate((projection, [1.0]))
        weights, *_ = np.linalg.lstsq(augmented, rhs, rcond=None)
        reproduced = weights @ vertices; hull_defect = _relative(reproduced, projection); sum_defect = abs(float(np.sum(weights)) - 1.0)
        if np.min(weights) < -tolerance or hull_defect > tolerance or sum_defect > tolerance: continue
        weights = np.where(np.abs(weights) <= tolerance, 0.0, weights); weights /= np.sum(weights)
        location = GuardSheetLocation(simplex_index, np.array(indices, copy=True), weights, normal, signed, _relative(weights @ vertices, projection), float(np.min(weights)), abs(float(np.sum(weights)) - 1.0))
        candidates.append((abs(signed), -location.minimum_weight, simplex_index, location))
    if not candidates: raise ValueError("query is outside every admissible guard-sheet element")
    candidates.sort(key=lambda item: (item[0], item[1], item[2])); return candidates[0][3]


def interpolate_cycle_event(
    events,
    *,
    event_simplices,
    event_simplex_classes,
    q_scales,
    phase_scale: float,
    query_invariants,
    phase: float,
    transition_class: int,
    reduced_flow_scaled,
    pre_state,
    conservation_map,
    minimum_norm_normal,
    require_on_guard: bool = True,
    guard_tolerance: float = 2.0e-10,
) -> CycleEventValue:
    conservation = _finite(conservation_map, ndim=2, name="conservation map")
    normal_map = _finite(minimum_norm_normal, ndim=2, name="minimum norm normal")
    q = _finite(events["pre_invariants4"], ndim=2, name="event invariants")
    phases = _finite(events["phase"], ndim=1, name="event phases")
    qscale = _finite(q_scales, ndim=1, name="event q scales")
    pscale = float(phase_scale); center = float(phase) % TWO_PI
    target_q = _finite(query_invariants, ndim=1, name="query invariants")
    if q.shape[1] != 4 or target_q.shape != (4,) or qscale.shape != (4,) or np.any(qscale <= 0.0) or pscale <= 0.0:
        raise ValueError("event reduced coordinates disagree")
    unwrapped = _unwrap_around(phases, center)
    nodes = np.column_stack((q / qscale, unwrapped / pscale))
    target = np.concatenate((target_q / qscale, [center / pscale]))
    truth_normals = _finite(events["reduced_guard_normals5"], ndim=2, name="guard normals")
    guard = locate_guard_sheet(nodes, event_simplices, truth_normals, target, simplex_classes=event_simplex_classes, transition_class=transition_class)
    if require_on_guard and abs(guard.signed_guard_distance) > float(guard_tolerance):
        raise ValueError("event query has not reached the guard sheet")
    indices = guard.vertex_indices; weights = guard.weights
    offsets = _finite(events["reduced_guard_offsets"], ndim=1, name="guard offsets")
    anchor_guard_residual = np.einsum(
        "ij,ij->i", truth_normals[indices], nodes[indices]
    ) + offsets[indices]
    if float(np.max(np.abs(anchor_guard_residual))) > 2.0e-12:
        raise ValueError("event truth normals and offsets do not contain their guard points")
    classes = _finite(events["transition_class_index"], ndim=1, name="transition classes", dtype=int)
    source_modes = _finite(events["source_mode_index"], ndim=1, name="source modes", dtype=int)
    destination_modes = _finite(events["destination_mode_index"], ndim=1, name="destination modes", dtype=int)
    if np.any(classes[indices] != int(transition_class)) or len(np.unique(source_modes[indices])) != 1 or len(np.unique(destination_modes[indices])) != 1:
        raise ValueError("event element mixes transition classes or modes")
    flow = _finite(reduced_flow_scaled, ndim=1, name="reduced flow")
    if flow.shape != (5,): raise ValueError("reduced event flow must have dimension five")
    transversality = float(guard.oriented_normal @ flow)
    if abs(transversality) < 1.0e-8: raise ValueError("event crossing is not transverse")
    duration = float(_weighted(_finite(events["duration_seconds"], ndim=1, name="event duration"), indices, weights))
    impulse = _weighted(_finite(events["integrated_ledger_impulse4"], ndim=2, name="event impulse"), indices, weights)
    constitutive = _weighted(_finite(events["ledger_null_constitutive_jump1232"], ndim=2, name="constitutive jump"), indices, weights)
    before = _finite(pre_state, ndim=1, name="pre-event state")
    if conservation.shape != (4, len(before)) or normal_map.shape != (len(before), 4) or duration <= 0.0:
        raise ValueError("event reset geometry or duration is invalid")
    constitutive = constitutive - normal_map @ (conservation @ constitutive)
    post = before + normal_map @ impulse + constitutive
    reset_defect = _relative(conservation @ (post - before), impulse)
    null_defect = float(np.linalg.norm(conservation @ constitutive) / max(np.linalg.norm(constitutive), 1.0))
    if reset_defect > 2.0e-12 or null_defect > 2.0e-12: raise ValueError("interpolated event reset does not close its ledger")
    return CycleEventValue(duration, impulse, constitutive, post, int(source_modes[indices[0]]), int(destination_modes[indices[0]]), int(transition_class), guard, transversality, reset_defect, null_defect)


__all__ = [
    "CycleBranchValue",
    "CycleDriverValue",
    "CycleEventValue",
    "GuardSheetLocation",
    "SimplexLocation",
    "interpolate_cycle_branch",
    "interpolate_cycle_driver",
    "interpolate_cycle_event",
    "locate_full_simplex",
    "locate_guard_sheet",
    "locate_periodic_branch_simplex",
    "periodic_phase_weights",
]
