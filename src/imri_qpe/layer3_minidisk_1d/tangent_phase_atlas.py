"""Local phase charts built from conservative-metric unit tangents.

The atlas is an event/phase observer, not a replacement vector field.  Each
chart fits an oriented circle in the leading affine two-plane of a trailing
window of already accepted physical free-field tangents.  A subsequent exact
tangent can therefore be assessed prospectively before it is admitted to the
next chart.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Array = np.ndarray


def _finite(value, *, ndim: int, name: str) -> Array:
    result = np.asarray(value, dtype=float)
    if result.ndim != ndim or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    return result.copy()


def normalized_metric_tangents(rates: Array, transform: Array) -> Array:
    """Return row-wise unit tangents after a frozen conservative transform."""

    vectors = _finite(rates, ndim=2, name="rates")
    metric = _finite(transform, ndim=2, name="transform")
    if metric.shape != (vectors.shape[1], vectors.shape[1]):
        raise ValueError("transform and tangent dimensions disagree")
    transformed = vectors @ metric.T
    speeds = np.linalg.norm(transformed, axis=1)
    if np.any(speeds <= np.finfo(float).tiny):
        raise ValueError("metric tangent speed is zero")
    return transformed / speeds[:, None]


def _canonicalize_plane_signs(basis: Array) -> Array:
    result = np.asarray(basis, dtype=float).copy()
    for column in range(result.shape[1]):
        pivot = int(np.argmax(np.abs(result[:, column])))
        if result[pivot, column] < 0.0:
            result[:, column] *= -1.0
    return result


@dataclass(frozen=True)
class TangentPhaseChart:
    """One oriented local circle chart for unit tangent directions."""

    mean_tangent: Array
    plane_basis: Array
    circle_center: Array
    circle_radius: float
    orientation_sign: int
    oriented_angle_origin: float
    training_phases: Array
    predicted_phase_increment: float
    two_plane_energy_fraction: float
    training_relative_radial_rms: float
    circle_solve_condition_number: float

    def __post_init__(self) -> None:
        mean = _finite(self.mean_tangent, ndim=1, name="mean_tangent")
        basis = _finite(self.plane_basis, ndim=2, name="plane_basis")
        center = _finite(self.circle_center, ndim=1, name="circle_center")
        phases = _finite(self.training_phases, ndim=1, name="training_phases")
        if basis.shape != (len(mean), 2) or center.shape != (2,):
            raise ValueError("tangent phase chart dimensions disagree")
        if np.linalg.norm(basis.T @ basis - np.eye(2), ord=np.inf) > 1.0e-10:
            raise ValueError("tangent phase plane is not orthonormal")
        if len(phases) < 4 or not np.all(np.diff(phases) > 0.0):
            raise ValueError("training phase must be strictly increasing")
        radius = float(self.circle_radius)
        prediction = float(self.predicted_phase_increment)
        energy = float(self.two_plane_energy_fraction)
        radial = float(self.training_relative_radial_rms)
        condition = float(self.circle_solve_condition_number)
        if (
            not np.isfinite(radius)
            or radius <= 0.0
            or not np.isfinite(prediction)
            or prediction <= 0.0
            or not 0.0 <= energy <= 1.0
            or not np.isfinite(radial)
            or radial < 0.0
            or not np.isfinite(condition)
            or condition < 1.0
            or int(self.orientation_sign) not in (-1, 1)
        ):
            raise ValueError("tangent phase chart diagnostics are invalid")
        for value in (mean, basis, center, phases):
            value.setflags(write=False)
        object.__setattr__(self, "mean_tangent", mean)
        object.__setattr__(self, "plane_basis", basis)
        object.__setattr__(self, "circle_center", center)
        object.__setattr__(self, "training_phases", phases)
        object.__setattr__(self, "circle_radius", radius)
        object.__setattr__(self, "predicted_phase_increment", prediction)
        object.__setattr__(self, "two_plane_energy_fraction", energy)
        object.__setattr__(self, "training_relative_radial_rms", radial)
        object.__setattr__(self, "circle_solve_condition_number", condition)

    def _unit(self, tangent: Array) -> Array:
        value = _finite(tangent, ndim=1, name="unit_tangent")
        if value.shape != self.mean_tangent.shape:
            raise ValueError("unit tangent has the wrong dimension")
        norm = float(np.linalg.norm(value))
        if norm <= np.finfo(float).tiny:
            raise ValueError("unit tangent is zero")
        return value / norm

    def evaluate(self, tangent: Array) -> dict[str, float]:
        """Evaluate a tangent in the chart, unwrapped near its trailing phase."""

        unit = self._unit(tangent)
        centered = unit - self.mean_tangent
        plane = centered @ self.plane_basis
        radial_vector = plane - self.circle_center
        radius = float(np.linalg.norm(radial_vector))
        raw_angle = float(np.arctan2(radial_vector[1], radial_vector[0]))
        phase = self.orientation_sign * raw_angle - self.oriented_angle_origin
        reference = float(self.training_phases[-1])
        phase += 2.0 * np.pi * np.round((reference - phase) / (2.0 * np.pi))
        reconstructed_plane = plane @ self.plane_basis.T
        return {
            "phase": float(phase),
            "phase_increment": float(phase - reference),
            "relative_radial_defect": float(
                abs(radius - self.circle_radius) / self.circle_radius
            ),
            "out_of_plane_defect": float(
                np.linalg.norm(centered - reconstructed_plane)
            ),
        }

    def predicted_unit_tangent(self) -> Array:
        phase = float(self.training_phases[-1] + self.predicted_phase_increment)
        oriented_angle = phase + self.oriented_angle_origin
        raw_angle = self.orientation_sign * oriented_angle
        plane = self.circle_center + self.circle_radius * np.asarray(
            (np.cos(raw_angle), np.sin(raw_angle))
        )
        tangent = self.mean_tangent + self.plane_basis @ plane
        return tangent / np.linalg.norm(tangent)

    def evaluate_next(self, tangent: Array) -> dict[str, float]:
        result = self.evaluate(tangent)
        exact = self._unit(tangent)
        predicted = self.predicted_unit_tangent()
        result["direction_prediction_defect_radians"] = float(
            np.arccos(np.clip(predicted @ exact, -1.0, 1.0))
        )
        result["predicted_phase_increment"] = self.predicted_phase_increment
        return result


def fit_tangent_phase_chart(
    unit_tangents: Array, *, predictor_increment_count: int = 4
) -> TangentPhaseChart:
    """Fit one deterministic affine-plane circle chart to ordered tangents."""

    tangents = _finite(unit_tangents, ndim=2, name="unit_tangents")
    if len(tangents) < 6:
        raise ValueError("a tangent phase chart requires at least six samples")
    norms = np.linalg.norm(tangents, axis=1)
    if np.any(norms <= np.finfo(float).tiny):
        raise ValueError("unit tangent training data contain a zero")
    tangents /= norms[:, None]
    mean = np.mean(tangents, axis=0)
    centered = tangents - mean
    _left, singular, right = np.linalg.svd(centered, full_matrices=False)
    if len(singular) < 2 or singular[1] <= np.finfo(float).tiny:
        raise ValueError("tangent phase plane is rank deficient")
    basis = _canonicalize_plane_signs(right[:2].T)
    plane = centered @ basis
    matrix = np.column_stack(
        (2.0 * plane[:, 0], 2.0 * plane[:, 1], np.ones(len(plane)))
    )
    right_hand_side = np.sum(plane * plane, axis=1)
    solution, _residuals, rank, _singular = np.linalg.lstsq(
        matrix, right_hand_side, rcond=None
    )
    if rank != 3:
        raise ValueError("tangent phase circle fit is rank deficient")
    center = solution[:2]
    radius_squared = float(solution[2] + center @ center)
    if radius_squared <= 0.0:
        raise ValueError("tangent phase circle radius is not positive")
    radius = float(np.sqrt(radius_squared))
    radial = np.linalg.norm(plane - center, axis=1)
    raw = np.unwrap(
        np.arctan2(plane[:, 1] - center[1], plane[:, 0] - center[0])
    )
    orientation = 1 if float(np.median(np.diff(raw))) > 0.0 else -1
    oriented = orientation * raw
    phases = oriented - oriented[0]
    if not np.all(np.diff(phases) > 0.0):
        raise ValueError("tangent phase training direction is not monotone")
    count = min(int(predictor_increment_count), len(phases) - 1)
    if count < 1:
        raise ValueError("predictor increment count is invalid")
    energy = float(np.sum(singular[:2] ** 2) / np.sum(singular**2))
    return TangentPhaseChart(
        mean_tangent=mean,
        plane_basis=basis,
        circle_center=center,
        circle_radius=radius,
        orientation_sign=orientation,
        oriented_angle_origin=float(oriented[0]),
        training_phases=phases,
        predicted_phase_increment=float(np.median(np.diff(phases)[-count:])),
        two_plane_energy_fraction=energy,
        training_relative_radial_rms=float(
            np.sqrt(np.mean((radial - radius) ** 2)) / radius
        ),
        circle_solve_condition_number=float(np.linalg.cond(matrix)),
    )


def rolling_tangent_phase_audit(
    unit_tangents: Array, *, window_size: int, predictor_increment_count: int = 4
) -> tuple[dict, dict[str, Array]]:
    """Apply strictly trailing-window fits to every available holdout tangent."""

    tangents = _finite(unit_tangents, ndim=2, name="unit_tangents")
    window = int(window_size)
    if window < 6 or len(tangents) <= window:
        raise ValueError("rolling tangent phase audit has insufficient samples")
    records = []
    energies = []
    training_radial = []
    circle_conditions = []
    for index in range(window, len(tangents)):
        chart = fit_tangent_phase_chart(
            tangents[index - window : index],
            predictor_increment_count=predictor_increment_count,
        )
        records.append(chart.evaluate_next(tangents[index]))
        energies.append(chart.two_plane_energy_fraction)
        training_radial.append(chart.training_relative_radial_rms)
        circle_conditions.append(chart.circle_solve_condition_number)
    names = tuple(records[0])
    arrays = {
        name: np.asarray([record[name] for record in records], dtype=float)
        for name in names
    }
    metrics = {
        "window_size": window,
        "predictor_increment_count": int(predictor_increment_count),
        "prediction_count": len(records),
        "all_phase_increments_positive": bool(
            np.all(arrays["phase_increment"] > 0.0)
        ),
        "minimum_phase_increment": float(np.min(arrays["phase_increment"])),
        "maximum_phase_increment": float(np.max(arrays["phase_increment"])),
        "maximum_relative_radial_defect": float(
            np.max(arrays["relative_radial_defect"])
        ),
        "maximum_out_of_plane_defect": float(
            np.max(arrays["out_of_plane_defect"])
        ),
        "maximum_direction_prediction_defect_radians": float(
            np.max(arrays["direction_prediction_defect_radians"])
        ),
        "rms_direction_prediction_defect_radians": float(
            np.sqrt(
                np.mean(arrays["direction_prediction_defect_radians"] ** 2)
            )
        ),
        "minimum_training_two_plane_energy_fraction": float(np.min(energies)),
        "maximum_training_relative_radial_rms": float(np.max(training_radial)),
        "maximum_circle_solve_condition_number": float(
            np.max(circle_conditions)
        ),
    }
    return metrics, arrays
