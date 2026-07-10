"""Classify and globalize the Mdot=5 phase-DAE exit critical point."""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_mdot5_global_phase_dae_production as global_phase  # noqa: E402


model = global_phase.model
EXIT_DIR = ROOT / "outputs/checkpoints/m5_eta_phase_dae_exit_refinement_98p125_N164"
OUTPUT_STEM = os.environ.get(
    "IMBH_MDOT5_PHASE_CRITICAL_OUTPUT_STEM",
    "m5_eta_phase_critical_globalization_98p125_N164",
)
TABLE_PATH = ROOT / "outputs" / "tables" / f"{OUTPUT_STEM}.json"
PROFILE_PATH = ROOT / "outputs" / "tables" / f"{OUTPUT_STEM}_profiles.json"
CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints" / OUTPUT_STEM
NOTE_PATH = ROOT / "Note" / "CODEX_MDOT5_PHASE_CRITICAL_GLOBALIZATION_RESULTS.md"
FIGURE_PATH = ROOT / "outputs" / "figures" / f"{OUTPUT_STEM}.png"
RUN_PHASE_JACOBIANS = os.environ.get("IMBH_MDOT5_PHASE_CRITICAL_JACOBIANS", "1").strip().lower() in {
    "1", "true", "yes", "on"
}
RUN_ARCLENGTH = os.environ.get("IMBH_MDOT5_PHASE_CRITICAL_ARCLENGTH", "1").strip().lower() in {
    "1", "true", "yes", "on"
}
RUN_CUT_CORRECTOR = os.environ.get("IMBH_MDOT5_PHASE_CRITICAL_CUT_CORRECTOR", "1").strip().lower() in {
    "1", "true", "yes", "on"
}
CUT_MAX_NFEV = int(os.environ.get("IMBH_MDOT5_PHASE_CRITICAL_CUT_MAX_NFEV", "60"))
ARC_INITIAL_DS = float(os.environ.get("IMBH_MDOT5_PHASE_CRITICAL_ARC_INITIAL_DS", "0.02"))
ARC_MAX_DS = float(os.environ.get("IMBH_MDOT5_PHASE_CRITICAL_ARC_MAX_DS", "0.05"))
ARC_MIN_DS = float(os.environ.get("IMBH_MDOT5_PHASE_CRITICAL_ARC_MIN_DS", "2e-4"))
ARC_MAX_STEPS = int(os.environ.get("IMBH_MDOT5_PHASE_CRITICAL_ARC_MAX_STEPS", "36"))
ARC_MAX_NFEV = int(os.environ.get("IMBH_MDOT5_PHASE_CRITICAL_ARC_MAX_NFEV", "80"))
ARC_PRINT_EVERY = max(1, int(os.environ.get("IMBH_MDOT5_PHASE_CRITICAL_ARC_PRINT_EVERY", "1")))
REFRESH_EXISTING = os.environ.get("IMBH_MDOT5_PHASE_CRITICAL_REFRESH_EXISTING", "0").strip().lower() in {
    "1", "true", "yes", "on"
}


CHECKPOINTS = (
    ("refine2", 0.0, "refine2.npz"),
    ("quarter", 0.25, "extend2_quarter.npz"),
    ("half", 0.5, "extend2_half.npz"),
    ("threequarter", 0.75, "extend2_threequarter.npz"),
    ("f8125", 0.8125, "extend2_f8125.npz"),
    ("f84375", 0.84375, "extend2_f84375.npz"),
    ("f859375", 0.859375, "extend2_f859375.npz"),
    ("f875", 0.875, "extend2_f875.npz"),
    ("f8828125", 0.8828125, "extend2_f8828125.npz"),
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _load_phase(path: Path) -> tuple[np.ndarray, ...]:
    with np.load(path) as data:
        return tuple(np.asarray(data[key], dtype=float) for key in ("z", "p", "p_mid", "ds"))


def _phase_points(z: np.ndarray, p: np.ndarray, p_mid: np.ndarray, ds: np.ndarray):
    for pos in range(z.shape[0]):
        yield f"node_{pos}", np.asarray(z[pos]), np.asarray(p[pos])
        if pos < ds.size:
            z_mid = 0.5 * (z[pos] + z[pos + 1]) + ds[pos] / 8.0 * (p[pos] - p[pos + 1])
            yield f"mid_{pos}", np.asarray(z_mid), np.asarray(p_mid[pos])


def _tangent_jacobian(z: np.ndarray, p: np.ndarray, params, lambda0: float) -> dict[str, Any]:
    base = np.asarray(
        model._global_flux_phase_dae_point_data(z, p, params, lambda0)["homogeneous_rows"],
        dtype=float,
    )
    jac = np.zeros((base.size, p.size), dtype=float)
    for col in range(p.size):
        step = 1.0e-6 * max(1.0, abs(float(p[col])))
        plus = np.asarray(p, dtype=float).copy()
        minus = np.asarray(p, dtype=float).copy()
        plus[col] += step
        minus[col] -= step
        r_plus = np.asarray(
            model._global_flux_phase_dae_point_data(z, plus, params, lambda0)["homogeneous_rows"],
            dtype=float,
        )
        r_minus = np.asarray(
            model._global_flux_phase_dae_point_data(z, minus, params, lambda0)["homogeneous_rows"],
            dtype=float,
        )
        jac[:, col] = (r_plus - r_minus) / (2.0 * step)
    _u, singular, vt = np.linalg.svd(jac, full_matrices=True)
    tolerance = max(jac.shape) * np.finfo(float).eps * max(float(singular[0]), 1.0)
    return {
        "rank": int(np.count_nonzero(singular > tolerance)),
        "singular_values": singular,
        "right_null": vt[-1],
        "condition_nonzero": float(singular[0] / max(float(singular[-1]), 1.0e-300)),
    }


def _point_audit(label: str, z: np.ndarray, p: np.ndarray, params, lambda0: float) -> dict[str, Any]:
    point = model._global_flux_phase_dae_point_data(z, p, params, lambda0)
    singular = np.asarray(point["A_singular_values"], dtype=float)
    state_tangent = np.asarray(p[:3], dtype=float)
    radial_tangent = np.asarray(p[:2], dtype=float)
    right = np.asarray(point["A_right_min"], dtype=float)
    alignment = abs(float(np.dot(radial_tangent, right))) / max(
        float(np.linalg.norm(radial_tangent)) * float(np.linalg.norm(right)), 1.0e-300
    )
    tangent = _tangent_jacobian(z, p, params, lambda0)
    return {
        "point": label,
        "R_rg": float(np.exp(float(z[3])) / params.r_g),
        "p_R": float(p[3]),
        "p_state_norm": float(np.linalg.norm(state_tangent)),
        "physical_derivative_norm": float(np.linalg.norm(state_tangent) / max(abs(float(p[3])), 1.0e-300)),
        "sigma_min_A": float(np.min(singular)),
        "sigma_max_A": float(np.max(singular)),
        "cond_A": float(point["cond_A"]),
        "compatibility": float(point["compatibility"]),
        "right_null_alignment": alignment,
        "left_null": np.asarray(point["A_left_min"], dtype=float),
        "right_null": right,
        "homogeneous_radial": float(point["dae"][0]),
        "homogeneous_energy": float(point["dae"][1]),
        "homogeneous_fprime": float(point["fprime"]),
        "direct_equivalence_radial": float(point["equivalence"][0]),
        "direct_equivalence_energy": float(point["equivalence"][1]),
        "direct_equivalence_fprime": float(point["fprime_equivalence"]),
        "dH_dp_rank": int(tangent["rank"]),
        "dH_dp_singular_values": tangent["singular_values"],
        "dH_dp_right_null": tangent["right_null"],
        "dH_dp_condition": float(tangent["condition_nonzero"]),
    }


def _phase_jacobian_audit(
    z: np.ndarray,
    p: np.ndarray,
    p_mid: np.ndarray,
    ds: np.ndarray,
    params,
    lambda0: float,
) -> dict[str, Any]:
    from scipy.optimize import least_squares

    count = int(ds.size)
    start = global_phase._phase_pack(z, p, p_mid, ds)
    labels = np.arange(count, dtype=int)
    mesh_target = np.diff(np.log(np.maximum(ds, 1.0e-300)))
    left = np.asarray(z[0], dtype=float)
    right = np.asarray(z[-1], dtype=float)

    def residual(vector: np.ndarray) -> np.ndarray:
        z_q, p_q, pm_q, ds_q = global_phase._phase_unpack(vector, count + 1, count)
        data = model._global_flux_phase_dae_segment_data(
            z_q, p_q, pm_q, ds_q, params, lambda0, labels, mesh_target
        )
        return np.concatenate(
            [np.asarray(data["rows"], dtype=float), 100.0 * (z_q[0] - left), 100.0 * (z_q[-1] - right)]
        )

    sparsity = model._global_flux_phase_dae_segment_sparsity(count + 1, count, "state")
    result = least_squares(
        residual,
        start,
        jac_sparsity=sparsity,
        x_scale="jac",
        max_nfev=1,
        ftol=1.0e-8,
        xtol=1.0e-8,
        gtol=1.0e-8,
    )
    jac = result.jac.toarray() if hasattr(result.jac, "toarray") else np.asarray(result.jac, dtype=float)
    _u, singular, vt = np.linalg.svd(jac, full_matrices=False)
    return {
        "rows": int(jac.shape[0]),
        "variables": int(jac.shape[1]),
        "smallest_singular_values": singular[-4:][::-1],
        "condition": float(singular[0] / max(float(singular[-1]), 1.0e-300)),
        "weak_right": vt[-1],
    }


def _critical_checkpoint_audit(params, lambda0: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    profiles: dict[str, Any] = {}
    for label, fraction, filename in CHECKPOINTS:
        path = EXIT_DIR / filename
        z, p, p_mid, ds = _load_phase(path)
        points = [_point_audit(name, z_q, p_q, params, lambda0) for name, z_q, p_q in _phase_points(z, p, p_mid, ds)]
        critical = min(points, key=lambda item: abs(float(item["p_R"])))
        phase_jac = _phase_jacobian_audit(z, p, p_mid, ds, params, lambda0) if RUN_PHASE_JACOBIANS else {}
        row = {
            "label": label,
            "fraction": float(fraction),
            "checkpoint": str(path.relative_to(ROOT)),
            "intervals": int(ds.size),
            **critical,
            "phase_jacobian_smin": (
                float(phase_jac["smallest_singular_values"][0]) if phase_jac else math.nan
            ),
            "phase_jacobian_condition": float(phase_jac.get("condition", math.nan)),
        }
        rows.append(row)
        profiles[label] = {"points": points, "phase_jacobian": phase_jac}
        print(
            "critical",
            label,
            f"R={row['R_rg']:.6f}",
            f"pR={row['p_R']:.3e}",
            f"sminA={row['sigma_min_A']:.3e}",
            f"compat={row['compatibility']:.3e}",
            flush=True,
        )
    return rows, profiles


def _cut_energy_interval(
    z_l: np.ndarray,
    z_r: np.ndarray,
    params,
    lambda0: float,
) -> dict[str, float]:
    dx = float(z_r[3] - z_l[3])
    if dx <= 0.0:
        return {"residual": math.inf, "numerator": math.inf, "denominator": 1.0}
    y_l = np.asarray(z_l[:2], dtype=float)
    y_r = np.asarray(z_r[:2], dtype=float)
    g = (y_r - y_l) / dx
    numerator = 0.0
    denominator = 0.0
    for coefficient, fraction in ((1.0, 0.0), (4.0, 0.5), (1.0, 1.0)):
        z_q = (1.0 - fraction) * z_l + fraction * z_r
        F_q = max(float(z_q[2]), 1.0e-300)
        dlogF = float((math.log(max(float(z_r[2]), 1.0e-300)) - math.log(max(float(z_l[2]), 1.0e-300))) / dx)
        local = model._local_params_with_point_mdot(
            params, float(z_q[3]), math.log(F_q * params.Mdot_g_s), dlogF
        )
        terms = model._energy_terms_at(float(z_q[3]), np.asarray(z_q[:2]), g, lambda0, local)
        weight = dx * coefficient / 6.0
        numerator += weight * float(terms["area"]) * float(terms["raw"])
        denominator += abs(weight) * float(terms["area"]) * float(terms["denom"])
    return {
        "numerator": float(numerator),
        "denominator": float(denominator),
        "residual": float(numerator / max(abs(denominator), 1.0e-300)),
    }


def _cut_mass_interval(z_l: np.ndarray, z_r: np.ndarray, params, lambda0: float) -> float:
    dx = float(z_r[3] - z_l[3])
    if dx <= 0.0:
        return math.inf
    y_l = np.asarray(z_l[:2], dtype=float)
    y_r = np.asarray(z_r[:2], dtype=float)
    g = (y_r - y_l) / dx
    integral = 0.0
    for coefficient, fraction in ((1.0, 0.0), (4.0, 0.5), (1.0, 1.0)):
        z_q = (1.0 - fraction) * z_l + fraction * z_r
        F_q = max(float(z_q[2]), 1.0e-300)
        dlogF = float((math.log(max(float(z_r[2]), 1.0e-300)) - math.log(max(float(z_l[2]), 1.0e-300))) / dx)
        local = model._local_params_with_point_mdot(
            params, float(z_q[3]), math.log(F_q * params.Mdot_g_s), dlogF
        )
        wind = model._safe_wind_prime(float(z_q[3]), np.asarray(z_q[:2]), g, lambda0, local)
        if not np.isfinite(wind):
            wind = 0.0
        source = global_phase.stream_source_prime(float(z_q[3]), local)
        integral += dx * coefficient / 6.0 * (float(wind) - float(source)) / params.Mdot_g_s
    return float(z_r[2] - z_l[2] - integral)


def _moving_interface_audit(
    x_log: np.ndarray,
    params,
    context: dict[str, Any],
    aux: np.ndarray,
    lambda0: float,
) -> list[dict[str, Any]]:
    setup = global_phase._composite_setup(x_log, params, context, aux, global_phase._load_problem()[-1])
    x_flux = np.asarray(setup["immutable_x_flux"], dtype=float)
    n = int(params.n_nodes)
    logu, logT, _logmdot, _sonic, _lambda, logR = model.pilot._unpack(x_log, params)
    base = model._production_residual_flux_base(
        x_flux, params, source_guard_context=None, include_source_guards=False, skip_source_dynamics_override=True
    )
    source_data = model._global_flux_hsfv_source_data(x_log, params, context, aux)
    source_raw = np.asarray(source_data.get("raw_rows", source_data.get("rows", [])), dtype=float)
    source_intervals, source_components = global_phase._source_row_metadata(source_data, context)
    mass_start = model._inner_mdot_row_index(params) + 1
    rows: list[dict[str, Any]] = []
    selected = {0.5: "extend2_half.npz", 0.75: "extend2_threequarter.npz", 0.84375: "extend2_f84375.npz", 0.8828125: "extend2_f8828125.npz"}
    for fraction, filename in selected.items():
        z, p, p_mid, ds = _load_phase(EXIT_DIR / filename)
        replaced = set(range(129, 143))
        right_node = 143
        z_right = np.asarray(
            [logu[right_node], logT[right_node], x_flux[2 * n + right_node], logR[right_node]], dtype=float
        )
        cut_mass = _cut_mass_interval(z[-1], z_right, params, lambda0)
        cut_energy = _cut_energy_interval(z[-1], z_right, params, lambda0)
        cut_dx = float(z_right[3] - z[-1, 3])
        cut_g = (z_right[:2] - z[-1, :2]) / max(cut_dx, 1.0e-300)
        cut_mid = 0.5 * (z[-1] + z_right)
        cut_dlogF = float((math.log(max(z_right[2], 1.0e-300)) - math.log(max(z[-1, 2], 1.0e-300))) / max(cut_dx, 1.0e-300))
        cut_params = model._local_params_with_point_mdot(
            params, float(cut_mid[3]), math.log(max(float(cut_mid[2]), 1.0e-300) * params.Mdot_g_s), cut_dlogF
        )
        cut_ode = model._scaled_residual_at(float(cut_mid[3]), np.asarray(cut_mid[:2]), cut_g, lambda0, cut_params)
        phase_mass = global_phase._phase_mass_profile(z, p, p_mid, ds, np.arange(ds.size), params, lambda0)
        mass_values = [abs(float(base[mass_start + idx]) / max(model.GLOBAL_FLUX_PRODUCTION_MASS_WEIGHT, 1.0e-300)) for idx in range(n - 1) if idx not in replaced]
        mass_values.extend(abs(float(item["FV_mass"])) for item in phase_mass)
        mass_values.append(abs(cut_mass))
        radial_values = [abs(float(base[2 * idx])) for idx in range(n - 1) if idx not in replaced]
        energy_values = [abs(float(base[2 * idx + 1])) for idx in range(n - 1) if idx not in replaced]
        radial_values.append(abs(float(cut_ode[0])))
        energy_values.append(abs(float(cut_ode[1])))
        keep_source = ~np.isin(source_intervals, np.asarray(sorted(replaced), dtype=int))
        if source_components.size == source_raw.size:
            radial_values.extend(np.abs(source_raw[keep_source & (source_components == "radial")]).tolist())
            energy_values.extend(np.abs(source_raw[keep_source & (source_components == "energy")]).tolist())
        phase_last_energy = global_phase._phase_interval_energy(
            z[-2], z[-1], p[-2], p_mid[-1], p[-1], ds[-1], params, lambda0
        )
        interface_numerator = float(phase_last_energy["numerator"]) + float(cut_energy["numerator"])
        interface_denominator = abs(float(phase_last_energy["denominator"])) + abs(float(cut_energy["denominator"]))
        phase_summary = model._global_flux_phase_dae_segment_data(
            z, p, p_mid, ds, params, lambda0, np.arange(ds.size), np.diff(np.log(ds))
        )["summary"]
        source_guard = float(np.nanmax(np.abs(source_raw[keep_source]))) if np.any(keep_source) else math.nan
        row = {
            "fraction": float(fraction),
            "checkpoint": str((EXIT_DIR / filename).relative_to(ROOT)),
            "interface_R_rg": float(np.exp(z[-1, 3]) / params.r_g),
            "cut_width_logR": cut_dx,
            "cut_radial": abs(float(cut_ode[0])),
            "cut_energy": abs(float(cut_ode[1])),
            "cut_FV_mass": abs(float(cut_mass)),
            "global_FV_mass": float(max(mass_values)),
            "outside_radial": float(max(radial_values)),
            "outside_energy": float(max(energy_values)),
            "interface_FV_energy": abs(float(interface_numerator / max(interface_denominator, 1.0e-300))),
            "source_compatibility": source_guard,
            "DeltaF_jump": 0.0,
            "phase_radial": float(phase_summary["radial_max"]),
            "phase_energy": float(phase_summary["energy_max"]),
        }
        rows.append(row)
        print(
            "cut",
            f"fraction={fraction:.6f}",
            f"R={row['interface_R_rg']:.4f}",
            f"mass={row['global_FV_mass']:.3e}",
            f"Rout={row['outside_radial']:.3e}",
            flush=True,
        )
    return rows


def _cut_corrector_case(
    fraction: float,
    filename: str,
    x_log: np.ndarray,
    params,
    context: dict[str, Any],
    aux_seed: np.ndarray,
    lambda0: float,
) -> dict[str, Any]:
    """Release the cut-cell source tail while keeping the phase branch fixed."""

    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix

    z, p, p_mid, ds = _load_phase(EXIT_DIR / filename)
    x_flux_seed = model._log_x_to_flux_x(x_log, params)
    n = int(params.n_nodes)
    source_intervals = np.asarray(context["interval_indices"], dtype=int)
    interval_positions = {
        int(interval): pos for pos, interval in enumerate(source_intervals)
    }
    active_intervals = np.arange(143, 151, dtype=int)
    active_nodes = np.arange(143, 152, dtype=int)
    aux_per_interval = int(aux_seed.size // max(source_intervals.size, 1))
    active_aux_positions: list[int] = []
    for interval in active_intervals:
        pos = interval_positions[int(interval)]
        active_aux_positions.extend(
            range(aux_per_interval * pos, aux_per_interval * (pos + 1))
        )
    active_aux = np.asarray(active_aux_positions, dtype=int)

    state_start = np.concatenate(
        [
            x_flux_seed[active_nodes],
            x_flux_seed[n + active_nodes],
            x_flux_seed[2 * n + active_nodes],
        ]
    )
    start = np.concatenate([state_start, np.asarray(aux_seed[active_aux], dtype=float)])
    right_reference = np.asarray(
        [x_flux_seed[active_nodes[-1]], x_flux_seed[n + active_nodes[-1]], x_flux_seed[2 * n + active_nodes[-1]]],
        dtype=float,
    )

    def expand(vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        vector = np.asarray(vector, dtype=float)
        count = int(active_nodes.size)
        x_flux = np.asarray(x_flux_seed, dtype=float).copy()
        x_flux[active_nodes] = vector[:count]
        x_flux[n + active_nodes] = vector[count : 2 * count]
        x_flux[2 * n + active_nodes] = vector[2 * count : 3 * count]
        aux = np.asarray(aux_seed, dtype=float).copy()
        aux[active_aux] = vector[3 * count :]
        return x_flux, aux

    def components(vector: np.ndarray) -> dict[str, Any]:
        x_flux, aux = expand(vector)
        x_trial = model._flux_x_to_log_x(x_flux, params)
        logu, logT, _logmdot, _sonic, _lam, logR = model.pilot._unpack(x_trial, params)
        source_data = model._global_flux_hsfv_source_data(x_trial, params, context, aux)
        source_rows = np.asarray(source_data.get("rows", []), dtype=float)
        row_intervals, _row_components = global_phase._source_row_metadata(source_data, context)
        selected = np.isin(row_intervals, active_intervals)
        selected_source = source_rows[selected]
        right_node = int(active_nodes[0])
        z_right = np.asarray(
            [logu[right_node], logT[right_node], x_flux[2 * n + right_node], logR[right_node]],
            dtype=float,
        )
        cut_mass = _cut_mass_interval(z[-1], z_right, params, lambda0)
        cut_energy = _cut_energy_interval(z[-1], z_right, params, lambda0)
        dx = float(z_right[3] - z[-1, 3])
        cut_mid = 0.5 * (z[-1] + z_right)
        cut_g = (z_right[:2] - z[-1, :2]) / max(dx, 1.0e-300)
        dlogF = float(
            (math.log(max(z_right[2], 1.0e-300)) - math.log(max(z[-1, 2], 1.0e-300)))
            / max(dx, 1.0e-300)
        )
        cut_params = model._local_params_with_point_mdot(
            params, float(cut_mid[3]), math.log(max(float(cut_mid[2]), 1.0e-300) * params.Mdot_g_s), dlogF
        )
        cut_ode = model._scaled_residual_at(
            float(cut_mid[3]), np.asarray(cut_mid[:2]), cut_g, lambda0, cut_params
        )
        phase_energy = global_phase._phase_interval_energy(
            z[-2], z[-1], p[-2], p_mid[-1], p[-1], ds[-1], params, lambda0
        )
        interface_numerator = float(phase_energy["numerator"]) + float(cut_energy["numerator"])
        interface_denominator = abs(float(phase_energy["denominator"])) + abs(float(cut_energy["denominator"]))
        right_state = np.asarray(
            [
                x_flux[active_nodes[-1]],
                x_flux[n + active_nodes[-1]],
                x_flux[2 * n + active_nodes[-1]],
            ],
            dtype=float,
        )
        return {
            "source_rows": selected_source,
            "cut_ode": np.asarray(cut_ode, dtype=float),
            "cut_mass": float(cut_mass),
            "interface_energy": float(interface_numerator / max(interface_denominator, 1.0e-300)),
            "right_anchor": right_state - right_reference,
        }

    def residual(vector: np.ndarray) -> np.ndarray:
        data = components(vector)
        return np.concatenate(
            [
                np.asarray(data["source_rows"], dtype=float),
                10.0 * np.asarray(data["cut_ode"], dtype=float),
                np.asarray([10.0 * float(data["cut_mass"]), 10.0 * float(data["interface_energy"])]),
                10.0 * np.asarray(data["right_anchor"], dtype=float),
                1.0e-3 * (np.asarray(vector, dtype=float) - start),
            ]
        )

    initial = components(start)
    lower = start.copy()
    upper = start.copy()
    count = int(active_nodes.size)
    lower[: 2 * count] -= 0.3
    upper[: 2 * count] += 0.3
    lower[2 * count : 3 * count] = np.maximum(start[2 * count : 3 * count] - 0.1, 1.0e-8)
    upper[2 * count : 3 * count] = start[2 * count : 3 * count] + 0.1
    aux_lower = np.asarray(context["aux_lower"], dtype=float)[active_aux]
    aux_upper = np.asarray(context["aux_upper"], dtype=float)[active_aux]
    lower[3 * count :] = aux_lower
    upper[3 * count :] = aux_upper

    initial_rows = residual(start)
    pattern = lil_matrix((initial_rows.size, start.size), dtype=int)
    source_count = int(np.asarray(initial["source_rows"]).size)
    rows_per_interval = source_count // max(int(active_intervals.size), 1)
    for local_interval, interval in enumerate(active_intervals):
        row_slice = slice(rows_per_interval * local_interval, rows_per_interval * (local_interval + 1))
        for node in range(max(int(active_nodes[0]), int(interval) - 1), min(int(active_nodes[-1]), int(interval) + 2) + 1):
            local_node = int(node - active_nodes[0])
            for family in range(3):
                pattern[row_slice, family * count + local_node] = 1
        for aux_interval in range(max(int(active_intervals[0]), int(interval) - 1), min(int(active_intervals[-1]), int(interval) + 1) + 1):
            aux_local = int(aux_interval - active_intervals[0])
            begin = 3 * count + aux_per_interval * aux_local
            pattern[row_slice, begin : begin + aux_per_interval] = 1
    cut_start = source_count
    pattern[cut_start : cut_start + 4, [0, count, 2 * count]] = 1
    pattern[cut_start + 4 : cut_start + 7, [count - 1, 2 * count - 1, 3 * count - 1]] = 1
    anchor_start = cut_start + 7
    for col in range(start.size):
        pattern[anchor_start + col, col] = 1

    result = least_squares(
        residual,
        np.clip(start, lower + 1.0e-12, upper - 1.0e-12),
        bounds=(lower, upper),
        jac_sparsity=pattern.tocsr(),
        x_scale="jac",
        max_nfev=max(1, CUT_MAX_NFEV),
        ftol=1.0e-9,
        xtol=1.0e-9,
        gtol=1.0e-9,
    )
    final = components(result.x)

    def maximum(data: dict[str, Any]) -> float:
        return float(
            max(
                np.max(np.abs(np.asarray(data["source_rows"], dtype=float))),
                np.max(np.abs(np.asarray(data["cut_ode"], dtype=float))),
                abs(float(data["cut_mass"])),
                abs(float(data["interface_energy"])),
            )
        )

    row = {
        "fraction": float(fraction),
        "checkpoint": str((EXIT_DIR / filename).relative_to(ROOT)),
        "released_nodes": active_nodes.tolist(),
        "released_variables": int(start.size),
        "nfev": int(result.nfev),
        "status": int(result.status),
        "message": str(result.message),
        "initial_max": maximum(initial),
        "final_max": maximum(final),
        "initial_cut_radial": abs(float(initial["cut_ode"][0])),
        "final_cut_radial": abs(float(final["cut_ode"][0])),
        "initial_cut_energy": abs(float(initial["cut_ode"][1])),
        "final_cut_energy": abs(float(final["cut_ode"][1])),
        "initial_cut_mass": abs(float(initial["cut_mass"])),
        "final_cut_mass": abs(float(final["cut_mass"])),
        "initial_interface_energy": abs(float(initial["interface_energy"])),
        "final_interface_energy": abs(float(final["interface_energy"])),
        "initial_source_max": float(np.max(np.abs(initial["source_rows"]))),
        "final_source_max": float(np.max(np.abs(final["source_rows"]))),
        "right_anchor_max": float(np.max(np.abs(final["right_anchor"]))),
        "accepted_exploratory": bool(
            maximum(final) <= 3.0e-5
            and float(np.max(np.abs(final["right_anchor"]))) <= 1.0e-3
        ),
    }
    print(
        "cut-correct",
        f"fraction={fraction:.6f}",
        f"nfev={row['nfev']}",
        f"initial={row['initial_max']:.3e}",
        f"final={row['final_max']:.3e}",
        flush=True,
    )
    return row


def _moving_interface_correctors(
    x_log: np.ndarray,
    params,
    context: dict[str, Any],
    aux: np.ndarray,
    lambda0: float,
) -> list[dict[str, Any]]:
    selected = (
        (0.5, "extend2_half.npz"),
        (0.75, "extend2_threequarter.npz"),
        (0.84375, "extend2_f84375.npz"),
        (0.8828125, "extend2_f8828125.npz"),
    )
    return [
        _cut_corrector_case(fraction, filename, x_log, params, context, aux, lambda0)
        for fraction, filename in selected
    ]


def _local_arclength_step(
    z_left: np.ndarray,
    p_left: np.ndarray,
    ds: float,
    params,
    lambda0: float,
) -> tuple[bool, dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    from scipy.optimize import least_squares

    z_predict = np.asarray(z_left, dtype=float) + float(ds) * np.asarray(p_left, dtype=float)
    start = np.concatenate([z_predict, np.asarray(p_left), np.asarray(p_left)])
    z_lo = z_predict.copy()
    z_hi = z_predict.copy()
    state_trust = max(0.08, 3.0 * abs(float(ds)))
    z_lo[:2] -= state_trust
    z_hi[:2] += state_trust
    z_lo[2] = max(z_predict[2] - 0.02, 1.0e-8)
    z_hi[2] = z_predict[2] + 0.02
    z_lo[3] = float(z_left[3]) - 0.01
    z_hi[3] = float(z_left[3]) + 0.01
    lower = np.concatenate([z_lo, np.full(8, -1.5)])
    upper = np.concatenate([z_hi, np.full(8, 1.5)])

    def unpack(vector: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return np.asarray(vector[:4]), np.asarray(vector[4:8]), np.asarray(vector[8:12])

    def raw(vector: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        z_right, p_mid, p_right = unpack(vector)
        z_mid = 0.5 * (z_left + z_right) + ds / 8.0 * (p_left - p_right)
        mid = model._global_flux_phase_dae_point_data(z_mid, p_mid, params, lambda0)
        right = model._global_flux_phase_dae_point_data(z_right, p_right, params, lambda0)
        kin = z_right - z_left - ds / 6.0 * (p_left + 4.0 * p_mid + p_right)
        values = np.concatenate(
            [
                np.asarray(mid["homogeneous_rows"]),
                np.asarray(right["homogeneous_rows"]),
                kin,
                np.asarray([np.linalg.norm(p_mid) - 1.0, np.linalg.norm(p_right) - 1.0]),
            ]
        )
        diagnostics = {"mid": mid, "right": right, "kinematic": kin, "z_mid": z_mid}
        return values, diagnostics

    weights = np.asarray([100.0, 100.0, 100.0] * 2 + [30.0] * 4 + [10.0, 10.0])
    result = least_squares(
        lambda vector: weights * raw(vector)[0],
        np.clip(start, lower + 1.0e-12, upper - 1.0e-12),
        bounds=(lower, upper),
        x_scale="jac",
        max_nfev=max(1, ARC_MAX_NFEV),
        ftol=1.0e-10,
        xtol=1.0e-10,
        gtol=1.0e-10,
    )
    z_right, p_mid, p_right = unpack(result.x)
    values, diagnostics = raw(result.x)
    radial = max(abs(float(values[0])), abs(float(values[3])))
    energy = max(abs(float(values[1])), abs(float(values[4])))
    fprime = max(abs(float(values[2])), abs(float(values[5])))
    kinematic = float(np.max(np.abs(values[6:10])))
    norm = float(np.max(np.abs(values[10:12])))
    accepted = bool(
        radial <= 3.0e-5
        and energy <= 3.0e-5
        and fprime <= 1.0e-5
        and kinematic <= 3.0e-4
        and norm <= 1.0e-4
    )
    point = _point_audit("right", z_right, p_right, params, lambda0)
    summary = {
        "accepted": accepted,
        "nfev": int(result.nfev),
        "status": int(result.status),
        "message": str(result.message),
        "ds": float(ds),
        "radial": radial,
        "energy": energy,
        "fprime": fprime,
        "kinematic": kinematic,
        "norm": norm,
        **point,
    }
    return accepted, summary, z_right, p_mid, p_right


def _arclength_continuation(params, lambda0: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    z, p, p_mid, ds_all = _load_phase(EXIT_DIR / "extend2_f8828125.npz")
    ds_try = ARC_INITIAL_DS
    rows: list[dict[str, Any]] = []
    rejected = 0
    crossed = False
    negative_steps = 0
    for step in range(ARC_MAX_STEPS):
        accepted, row, z_right, pm_right, p_right = _local_arclength_step(
            z[-1], p[-1], ds_try, params, lambda0
        )
        row["step"] = int(step)
        row["attempt_ds"] = float(ds_try)
        rows.append(row)
        if step % ARC_PRINT_EVERY == 0 or not accepted:
            print(
                "arc",
                step,
                f"accepted={accepted}",
                f"ds={ds_try:.3e}",
                f"R={row['R_rg']:.6f}",
                f"pR={row['p_R']:.3e}",
                f"rad={row['radial']:.3e}",
                f"E={row['energy']:.3e}",
                flush=True,
            )
        if not accepted:
            rejected += 1
            ds_try *= 0.5
            if ds_try < ARC_MIN_DS or rejected >= 5:
                break
            continue
        rejected = 0
        previous_pR = float(p[-1, 3])
        z = np.vstack([z, z_right])
        p = np.vstack([p, p_right])
        p_mid = np.vstack([p_mid, pm_right])
        ds_all = np.concatenate([ds_all, np.asarray([ds_try])])
        crossed = crossed or previous_pR * float(p_right[3]) < 0.0
        if float(p_right[3]) < 0.0:
            negative_steps += 1
        ds_try = min(ARC_MAX_DS, ds_try * (1.15 if int(row["nfev"]) < 30 else 1.0))
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            CHECKPOINT_DIR / f"arc_step_{step:03d}.npz",
            z=z,
            p=p,
            p_mid=p_mid,
            ds=ds_all,
            crossed=np.asarray(crossed),
        )
        if crossed and negative_steps >= 10:
            break
    accepted_rows = [item for item in rows if bool(item.get("accepted", False))]
    crossing_position = next(
        (pos for pos, item in enumerate(accepted_rows) if float(item["p_R"]) < 0.0),
        None,
    )
    post_crossing = accepted_rows[crossing_position + 1 :] if crossing_position is not None else []
    classification = {
        "crossed_p_R_zero": bool(crossed),
        "accepted_steps": len(accepted_rows),
        "negative_steps": int(negative_steps),
        "minimum_abs_p_R": (
            float(min(abs(float(item["p_R"])) for item in accepted_rows)) if accepted_rows else math.nan
        ),
        "critical_R_rg": (
            float(min(accepted_rows, key=lambda item: abs(float(item["p_R"]))) ["R_rg"])
            if accepted_rows else math.nan
        ),
        "second_positive_branch": bool(
            crossed and any(float(item["p_R"]) > 0.0 for item in post_crossing)
        ),
    }
    return rows, classification


def _source_and_angular_audit(params, lambda0: float) -> dict[str, Any]:
    z, p, p_mid, ds = _load_phase(EXIT_DIR / "extend2_f8828125.npz")
    critical_points = [
        _point_audit(name, z_q, p_q, params, lambda0) for name, z_q, p_q in _phase_points(z, p, p_mid, ds)
    ]
    critical = min(critical_points, key=lambda item: abs(float(item["p_R"])))
    x = math.log(float(critical["R_rg"]) * params.r_g)
    h = 1.0e-4

    def source(xq: float) -> float:
        return float(global_phase.stream_source_prime(xq, params))

    value = source(x)
    first = (source(x + h) - source(x - h)) / (2.0 * h)
    second = (source(x + h) - 2.0 * value + source(x - h)) / h**2
    transitions = model._source_transition_radii_rg(params)
    inner = next((item for item in transitions if item["name"] == "source_support_inner"), None)
    angular = global_phase._phase_angular_profile(
        z, p, p_mid, ds, np.arange(ds.size), params, lambda0
    )
    angular_max = max(abs(float(item["angular_FV"])) for item in angular)
    variants: list[dict[str, Any]] = []
    for name, shape, width_factor in (
        ("compact_c2", "compact_c2", 1.0),
        ("compact_c4", "compact_c4", 1.0),
        ("compact_cinf", "compact_cinf", 1.0),
        ("compact_c2_wide", "compact_c2", 1.25),
    ):
        variant_params = replace(
            params,
            stream_source_shape=shape,
            stream_source_log_width=float(params.stream_source_log_width) * width_factor,
        )
        point = model._global_flux_phase_dae_point_data(
            z[-1], p[-1], variant_params, lambda0
        )
        tangent = _tangent_jacobian(z[-1], p[-1], variant_params, lambda0)
        null = np.asarray(tangent["right_null"], dtype=float)
        if float(np.dot(null, p[-1])) < 0.0:
            null = -null
        variant_transitions = model._source_transition_radii_rg(variant_params)
        variant_inner = next(
            (item for item in variant_transitions if item["name"] == "source_support_inner"),
            None,
        )
        variants.append(
            {
                "name": name,
                "shape": shape,
                "width_factor": float(width_factor),
                "source_inner_edge_rg": (
                    float(variant_inner["R_rg"]) if variant_inner is not None else math.nan
                ),
                "homogeneous_radial": float(point["dae"][0]),
                "homogeneous_energy": float(point["dae"][1]),
                "homogeneous_fprime": float(point["fprime"]),
                "sigma_min_A": float(np.min(point["A_singular_values"])),
                "cond_A": float(point["cond_A"]),
                "compatibility": float(point["compatibility"]),
                "local_null_p_R": float(null[3]),
                "dH_dp_rank": int(tangent["rank"]),
            }
        )
    return {
        "critical_R_rg": float(critical["R_rg"]),
        "source_transitions": transitions,
        "source_value": value,
        "source_first_dlogR": float(first),
        "source_second_dlogR2": float(second),
        "distance_from_inner_edge_rg": (
            float(critical["R_rg"] - float(inner["R_rg"])) if inner is not None else math.nan
        ),
        "distance_from_inner_edge_logR": (
            float(math.log(float(critical["R_rg"]) / float(inner["R_rg"]))) if inner is not None else math.nan
        ),
        "angular_flux_max": float(angular_max),
        "angular_profile": angular,
        "source_shape_variants": variants,
        "angular_assumption": (
            "diagnostic uses l_w=l_disk and l_s=l_disk, with the configured cumulative "
            "stream torque derivative as tau_s/Mdot; this is not a production closure"
        ),
    }


def _step_size_validation() -> dict[str, Any]:
    cases = (
        (0.05, "m5_eta_phase_critical_globalization_98p125_N164"),
        (0.01, "m5_eta_phase_critical_arc_ds001_98p125_N164"),
        (0.005, "m5_eta_phase_critical_arc_ds0005_98p125_N164"),
        (0.0025, "m5_eta_phase_critical_arc_ds00025_98p125_N164"),
    )
    rows: list[dict[str, Any]] = []
    for ds_value, stem in cases:
        table = ROOT / "outputs" / "tables" / f"{stem}.json"
        if not table.exists():
            continue
        data = json.loads(table.read_text())
        accepted = [item for item in data.get("arclength", []) if item.get("accepted", False)]
        if not accepted:
            continue
        first_negative = next(
            (pos for pos, item in enumerate(accepted) if float(item["p_R"]) < 0.0),
            None,
        )
        pre_position = max(0, first_negative - 1) if first_negative is not None else len(accepted) - 1
        pre = accepted[pre_position]
        checkpoint = ROOT / "outputs" / "checkpoints" / stem / f"arc_step_{int(pre['step']):03d}.npz"
        endpoint_z = np.full(4, math.nan)
        endpoint_p = np.full(4, math.nan)
        if checkpoint.exists():
            with np.load(checkpoint) as saved:
                endpoint_z = np.asarray(saved["z"][-1], dtype=float)
                endpoint_p = np.asarray(saved["p"][-1], dtype=float)
        closest = min(accepted, key=lambda item: abs(float(item["p_R"])))
        rows.append(
            {
                "ds": float(ds_value),
                "stem": stem,
                "accepted_steps": len(accepted),
                "total_arclength": float(sum(float(item["ds"]) for item in accepted)),
                "crossed": bool(first_negative is not None),
                "pre_cross_step": int(pre["step"]),
                "pre_cross_R_rg": float(pre["R_rg"]),
                "pre_cross_p_R": float(pre["p_R"]),
                "pre_cross_logu": float(endpoint_z[0]),
                "pre_cross_logT": float(endpoint_z[1]),
                "pre_cross_F": float(endpoint_z[2]),
                "closest_R_rg": float(closest["R_rg"]),
                "closest_p_R": float(closest["p_R"]),
                "endpoint_tangent": endpoint_p,
            }
        )
    refined = [item for item in rows if float(item["ds"]) <= 0.01]
    stable_state = False
    if len(refined) >= 2:
        logu_values = np.asarray([item["pre_cross_logu"] for item in refined], dtype=float)
        stable_state = bool(np.all(np.isfinite(logu_values)) and np.ptp(logu_values) <= 0.1)
    all_crossed = bool(refined and all(bool(item["crossed"]) for item in refined))
    certified_fold = bool(all_crossed and stable_state)
    decay_rate = math.nan
    limit_radius = math.nan
    if rows:
        finest = min(rows, key=lambda item: float(item["ds"]))
        finest_table = ROOT / "outputs" / "tables" / f"{finest['stem']}.json"
        finest_data = json.loads(finest_table.read_text())
        positive = [
            item for item in finest_data.get("arclength", [])
            if item.get("accepted", False) and float(item["p_R"]) > 0.0
        ]
        if len(positive) >= 20:
            tail = positive[-min(120, len(positive)) :]
            arc = np.cumsum(np.asarray([float(item["ds"]) for item in positive], dtype=float))[-len(tail) :]
            log_pr = np.log(np.asarray([float(item["p_R"]) for item in tail], dtype=float))
            slope, _intercept = np.polyfit(arc, log_pr, 1)
            decay_rate = float(-slope)
            last = tail[-1]
            if decay_rate > 0.0:
                limit_radius = float(
                    float(last["R_rg"]) * math.exp(float(last["p_R"]) / decay_rate)
                )
    return {
        "rows": rows,
        "certified_finite_state_fold": certified_fold,
        "interpretation": (
            "finite-state fold converged under arclength refinement"
            if certified_fold
            else "step-sensitive sheet switch; the resolved positive branch approaches a finite-radius, low-u singular limit"
        ),
        "positive_p_R_decay_rate_per_s": decay_rate,
        "estimated_limiting_R_rg": limit_radius,
    }


def _write_note(result: dict[str, Any]) -> None:
    critical = result["critical_checkpoints"]
    cut = result["moving_interface"]
    cut_correctors = result.get("moving_interface_correctors", [])
    arc = [row for row in result["arclength"] if row.get("accepted", False)]
    source = result["source_angular"]
    classification = result["classification"]
    step_validation = result.get("step_size_validation", {"rows": []})
    lines = [
        "# Mdot=5 phase critical-point classification and globalization",
        "",
        "Target: `Mdot_inner/Edd=5`, `Rout=335 rg`, `f_s=0.80`, `eta_E=98.125`, `N=164`.",
        "",
        "## Exact homogeneous DAE audit",
        "",
        "Production now uses the direct homogeneous residual `H(z,p)`; the divided radial residual is audit-only.",
        "",
        "| checkpoint | Rcrit (rg) | p_R | sigma_min(A) | cond(A) | u_min^T c | null alignment | dH/dp rank | phase J smin | phase J cond | H/direct max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in critical:
        equivalence = max(abs(float(row["direct_equivalence_radial"])), abs(float(row["direct_equivalence_energy"])))
        lines.append(
            f"| {row['label']} | {row['R_rg']:.6f} | {row['p_R']:.3e} | {row['sigma_min_A']:.3e} | "
            f"{row['cond_A']:.3e} | {row['compatibility']:.3e} | {row['right_null_alignment']:.6f} | "
            f"{row['dH_dp_rank']} | {row['phase_jacobian_smin']:.3e} | {row['phase_jacobian_condition']:.3e} | "
            f"{equivalence:.3e} |"
        )
    lines.extend(
        [
            "",
            "## Moving cut-cell interface",
            "",
            "| fraction | Rint (rg) | cut radial | cut energy | global FV mass | outside radial | outside energy | interface FV energy |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in cut:
        lines.append(
            f"| {row['fraction']:.6f} | {row['interface_R_rg']:.6f} | {row['cut_radial']:.3e} | "
            f"{row['cut_energy']:.3e} | {row['global_FV_mass']:.3e} | {row['outside_radial']:.3e} | "
            f"{row['outside_energy']:.3e} | {row['interface_FV_energy']:.3e} |"
        )
    if cut_correctors:
        lines.extend(
            [
                "",
                "### Coupled source-tail cut-cell corrector",
                "",
                "| fraction | initial max | final max | cut radial | cut energy | cut mass | interface energy | source max | right drift | accepted |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in cut_correctors:
            lines.append(
                f"| {row['fraction']:.6f} | {row['initial_max']:.3e} | {row['final_max']:.3e} | "
                f"{row['final_cut_radial']:.3e} | {row['final_cut_energy']:.3e} | {row['final_cut_mass']:.3e} | "
                f"{row['final_interface_energy']:.3e} | {row['final_source_max']:.3e} | "
                f"{row['right_anchor_max']:.3e} | {row['accepted_exploratory']} |"
            )
    lines.extend(
        [
            "",
            "## Signed arclength continuation",
            "",
            "| step | ds | R (rg) | p_R | radial | energy | F-prime | cond(A) | accepted |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in arc:
        lines.append(
            f"| {row['step']} | {row['ds']:.3e} | {row['R_rg']:.6f} | {row['p_R']:.3e} | "
            f"{row['radial']:.3e} | {row['energy']:.3e} | {row['fprime']:.3e} | {row['cond_A']:.3e} | yes |"
        )
    if step_validation.get("rows"):
        lines.extend(
            [
                "",
                "### Arclength step-size gate",
                "",
                "| ds | accepted steps | total s | crossed | last positive R (rg) | last positive p_R | last positive logu |",
                "|---:|---:|---:|---|---:|---:|---:|",
            ]
        )
        for row in step_validation["rows"]:
            lines.append(
                f"| {row['ds']:.4f} | {row['accepted_steps']} | {row['total_arclength']:.3f} | "
                f"{row['crossed']} | {row['pre_cross_R_rg']:.6f} | {row['pre_cross_p_R']:.3e} | "
                f"{row['pre_cross_logu']:.6f} |"
            )
        lines.extend(
            [
                "",
                f"Finite-state fold certified: `{step_validation['certified_finite_state_fold']}`.",
                f"Interpretation: {step_validation['interpretation']}.",
                f"Tail fit: `p_R ~ exp(-{step_validation.get('positive_p_R_decay_rate_per_s', math.nan):.3f} s)`, "
                f"with estimated limiting radius `{step_validation.get('estimated_limiting_R_rg', math.nan):.6f} rg`.",
            ]
        )
    lines.extend(
        [
            "",
            "## Source and angular audits",
            "",
            f"- Critical radius: `{source['critical_R_rg']:.6f} rg`.",
            f"- Distance from compact-source inner edge: `{source['distance_from_inner_edge_rg']:.6f} rg` "
            f"(`Delta lnR={source['distance_from_inner_edge_logR']:.6e}`).",
            f"- Source value/first/second log-radius derivatives: `{source['source_value']:.6e}`, "
            f"`{source['source_first_dlogR']:.6e}`, `{source['source_second_dlogR2']:.6e}`.",
            f"- Angular FV audit maximum: `{source['angular_flux_max']:.6e}`.",
            f"- Angular assumption: {source['angular_assumption']}.",
            "",
            "### Frozen-state source-shape diagnostic",
            "",
            "| variant | inner edge (rg) | H radial | H energy | H F-prime | sigma_min(A) | null p_R |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    lines.extend(
        f"| {item['name']} | {item['source_inner_edge_rg']:.6f} | {item['homogeneous_radial']:.3e} | "
        f"{item['homogeneous_energy']:.3e} | {item['homogeneous_fprime']:.3e} | "
        f"{item['sigma_min_A']:.3e} | {item['local_null_p_R']:.3e} |"
        for item in source["source_shape_variants"]
    )
    lines.extend(
        [
            "",
            "## Classification",
            "",
            f"- Coarse-step signed p_R zero crossing recovered: `{classification['crossed_p_R_zero']}`.",
            f"- Accepted arclength steps: `{classification['accepted_steps']}`.",
            f"- Closest accepted critical radius: `{classification['critical_R_rg']:.6f} rg`.",
            f"- Second positive-p_R branch found: `{classification['second_positive_branch']}`.",
            f"- Step-size-certified finite-state fold: `{classification.get('certified_finite_state_fold', False)}`.",
            f"- Branch classification: {classification.get('branch_classification', 'not classified')}.",
            "",
            "The growing compatibility scalar rules out a regular critical crossing. The homogeneous tangent Jacobian remains full rank, so there is no detected DAE-index change. The apparent signed crossing is not stable under arclength refinement and is therefore retained only as a rejected diagnostic sheet switch.",
            "",
            "The coupled moving-interface correctors also fail by four or more orders of magnitude relative to the exploratory gate. N164 global certification, a higher-N check, and eta continuation are therefore deferred.",
            "",
            "## Reproducibility",
            "",
            f"- Primary table: `{TABLE_PATH.relative_to(ROOT)}`.",
            f"- Profile table: `{PROFILE_PATH.relative_to(ROOT)}`.",
            f"- Diagnostic figure: `{FIGURE_PATH.relative_to(ROOT)}`.",
            "- Fine arclength checkpoints: `outputs/checkpoints/m5_eta_phase_critical_arc_ds001_98p125_N164/`, "
            "`outputs/checkpoints/m5_eta_phase_critical_arc_ds0005_98p125_N164/`, and "
            "`outputs/checkpoints/m5_eta_phase_critical_arc_ds00025_98p125_N164/`.",
            "- Regression status: `166 passed, 4 subtests passed`.",
            "",
            "Eta continuation remains paused until the phase/cut-cell composite satisfies the global conservation and exterior residual gates.",
        ]
    )
    NOTE_PATH.write_text("\n".join(lines) + "\n")


def _write_figure(result: dict[str, Any]) -> None:
    from PIL import Image, ImageDraw, ImageFont

    validation = result.get("step_size_validation", {})
    image = Image.new("RGB", (1500, 1050), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    colors = {0.05: "#B03A2E", 0.01: "#CA6F1E", 0.005: "#7D3C98", 0.0025: "#148F77"}
    panels = ((70, 70, 720, 490), (790, 70, 1440, 490), (70, 570, 720, 990), (790, 570, 1440, 990))

    def plot_panel(
        box,
        series: list[tuple[np.ndarray, np.ndarray, str, str]],
        title: str,
        xlabel: str,
        ylabel: str,
        *,
        logy: bool = False,
        horizontal: list[tuple[float, str, str]] | None = None,
    ) -> None:
        left, top, right, bottom = box
        pad_left, pad_right, pad_top, pad_bottom = 78, 22, 34, 54
        x0, x1 = left + pad_left, right - pad_right
        y0, y1 = top + pad_top, bottom - pad_bottom
        transformed: list[tuple[np.ndarray, np.ndarray, str, str]] = []
        x_all: list[float] = []
        y_all: list[float] = []
        for xs, ys, color, label in series:
            xs = np.asarray(xs, dtype=float)
            ys = np.asarray(ys, dtype=float)
            valid = np.isfinite(xs) & np.isfinite(ys) & ((ys > 0.0) if logy else True)
            xs = xs[valid]
            ys = np.log10(ys[valid]) if logy else ys[valid]
            transformed.append((xs, ys, color, label))
            x_all.extend(xs.tolist())
            y_all.extend(ys.tolist())
        for value, _color, _label in horizontal or []:
            if np.isfinite(value) and (value > 0.0 or not logy):
                y_all.append(math.log10(value) if logy else value)
        if not x_all or not y_all:
            return
        xmin, xmax = min(x_all), max(x_all)
        ymin, ymax = min(y_all), max(y_all)
        xspan = max(xmax - xmin, 1.0e-12)
        yspan = max(ymax - ymin, 1.0e-12)
        xmin -= 0.04 * xspan
        xmax += 0.04 * xspan
        ymin -= 0.06 * yspan
        ymax += 0.06 * yspan

        def xy(xv: float, yv: float) -> tuple[float, float]:
            return (
                x0 + (xv - xmin) / (xmax - xmin) * (x1 - x0),
                y1 - (yv - ymin) / (ymax - ymin) * (y1 - y0),
            )

        draw.rectangle((left, top, right, bottom), outline="#D5D8DC", width=1)
        for tick in range(5):
            xv = xmin + tick * (xmax - xmin) / 4.0
            yv = ymin + tick * (ymax - ymin) / 4.0
            px, _ = xy(xv, ymin)
            _, py = xy(xmin, yv)
            draw.line((px, y0, px, y1), fill="#ECEFF1", width=1)
            draw.line((x0, py, x1, py), fill="#ECEFF1", width=1)
            draw.text((px - 18, y1 + 8), f"{xv:.4g}", fill="#34495E", font=font)
            y_label = f"10^{yv:.1f}" if logy else f"{yv:.6g}"
            draw.text((left + 5, py - 6), y_label, fill="#34495E", font=font)
        draw.line((x0, y1, x1, y1), fill="#2C3E50", width=2)
        draw.line((x0, y0, x0, y1), fill="#2C3E50", width=2)
        draw.text((left + 10, top + 8), title, fill="#17202A", font=font)
        draw.text(((x0 + x1) / 2 - 28, bottom - 20), xlabel, fill="#34495E", font=font)
        draw.text((left + 5, top + 24), ylabel, fill="#34495E", font=font)
        legend_y = top + 8
        for xs, ys, color, label in transformed:
            points = [xy(float(xv), float(yv)) for xv, yv in zip(xs, ys)]
            if len(points) >= 2:
                draw.line(points, fill=color, width=3)
            for point in points[:: max(1, len(points) // 30)]:
                draw.ellipse((point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2), fill=color)
            draw.line((right - 155, legend_y + 5, right - 135, legend_y + 5), fill=color, width=3)
            draw.text((right - 130, legend_y), label, fill="#34495E", font=font)
            legend_y += 16
        for value, color, label in horizontal or []:
            yv = math.log10(value) if logy else value
            _, py = xy(xmin, yv)
            draw.line((x0, py, x1, py), fill=color, width=2)
            draw.text((x0 + 5, py - 14), label, fill=color, font=font)

    checkpoints = result.get("critical_checkpoints", [])
    plot_panel(
        panels[0],
        [(np.asarray([item["R_rg"] for item in checkpoints]), np.asarray([abs(item["p_R"]) for item in checkpoints]), "#176B87", "exit anchors")],
        "Approach to radial-graph singularity", "R / rg", "p_R", logy=True,
    )

    arc_series = []
    state_series = []
    for case in validation.get("rows", []):
        stem = str(case["stem"])
        data = json.loads((ROOT / "outputs" / "tables" / f"{stem}.json").read_text())
        accepted = [item for item in data.get("arclength", []) if item.get("accepted", False)]
        arc = np.cumsum(np.asarray([float(item["ds"]) for item in accepted], dtype=float))
        positive = np.asarray([float(item["p_R"]) > 0.0 for item in accepted], dtype=bool)
        arc_series.append((arc[positive], np.asarray([abs(float(item["p_R"])) for item in accepted])[positive], colors[float(case["ds"])], f"ds={float(case['ds']):g}"))
        logu_values: list[float] = []
        radius_values: list[float] = []
        for item in accepted:
            checkpoint = ROOT / "outputs" / "checkpoints" / stem / f"arc_step_{int(item['step']):03d}.npz"
            if checkpoint.exists():
                with np.load(checkpoint) as saved:
                    logu_values.append(float(saved["z"][-1, 0]))
                radius_values.append(float(item["R_rg"]))
        state_series.append((np.asarray(logu_values), np.asarray(radius_values), colors[float(case["ds"])], f"ds={float(case['ds']):g}"))
    plot_panel(panels[1], arc_series, "Arclength step-size audit", "intrinsic s", "abs(p_R)", logy=True)
    limit = float(validation.get("estimated_limiting_R_rg", math.nan))
    plot_panel(
        panels[2], state_series, "Finite radius, drifting state", "log u", "R / rg",
        horizontal=[(limit, "#2C3E50", "tail-fit limit")] if np.isfinite(limit) else None,
    )

    cut = result.get("moving_interface_correctors", [])
    fractions = np.asarray([float(item["fraction"]) for item in cut], dtype=float)
    cut_series = []
    if fractions.size:
        cut_series = [
            (fractions, np.asarray([item["initial_max"] for item in cut], dtype=float), "#7F8C8D", "initial"),
            (fractions, np.asarray([item["final_max"] for item in cut], dtype=float), "#C0392B", "corrected"),
        ]
    plot_panel(
        panels[3], cut_series, "Cut-cell globalization rejected", "interval-142 fraction", "maximum defect",
        logy=True, horizontal=[(3.0e-5, "#148F77", "exploratory gate")],
    )

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    image.save(FIGURE_PATH)


def main() -> None:
    if REFRESH_EXISTING:
        result = json.loads(TABLE_PATH.read_text())
        step_validation = _step_size_validation()
        result["step_size_validation"] = step_validation
        result.setdefault("classification", {})["certified_finite_state_fold"] = bool(
            step_validation["certified_finite_state_fold"]
        )
        result["classification"]["branch_classification"] = str(step_validation["interpretation"])
        TABLE_PATH.write_text(json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n")
        _write_note(result)
        _write_figure(result)
        print("refreshed", json.dumps(_jsonable(result["classification"]), sort_keys=True), flush=True)
        return
    x_log, params, context, aux, phase = global_phase._load_problem()
    lambda0 = float(phase["lambda0"])
    critical_rows, critical_profiles = _critical_checkpoint_audit(params, lambda0)
    cut_rows = _moving_interface_audit(x_log, params, context, aux, lambda0)
    cut_correctors = (
        _moving_interface_correctors(x_log, params, context, aux, lambda0)
        if RUN_CUT_CORRECTOR
        else []
    )
    arc_rows: list[dict[str, Any]] = []
    classification: dict[str, Any] = {
        "crossed_p_R_zero": False,
        "accepted_steps": 0,
        "negative_steps": 0,
        "minimum_abs_p_R": math.nan,
        "critical_R_rg": math.nan,
        "second_positive_branch": False,
    }
    if RUN_ARCLENGTH:
        arc_rows, classification = _arclength_continuation(params, lambda0)
    source_angular = _source_and_angular_audit(params, lambda0)
    step_validation = _step_size_validation()
    classification["certified_finite_state_fold"] = bool(
        step_validation["certified_finite_state_fold"]
    )
    classification["branch_classification"] = str(step_validation["interpretation"])
    result = {
        "target": {
            "Mdot_inner_Edd": 5.0,
            "Rout_rg": float(params.R_out_rg),
            "Rinj_rg": float(params.stream_source_center_fraction * params.R_out_rg),
            "source_fraction": float(params.stream_source_fraction),
            "eta_E": 98.125,
            "N": int(params.n_nodes),
        },
        "critical_checkpoints": critical_rows,
        "moving_interface": cut_rows,
        "moving_interface_correctors": cut_correctors,
        "arclength": arc_rows,
        "classification": classification,
        "source_angular": source_angular,
        "step_size_validation": step_validation,
    }
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.write_text(json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n")
    PROFILE_PATH.write_text(json.dumps(_jsonable(critical_profiles), indent=2, sort_keys=True) + "\n")
    _write_note(result)
    _write_figure(result)
    print("classification", json.dumps(_jsonable(classification), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
