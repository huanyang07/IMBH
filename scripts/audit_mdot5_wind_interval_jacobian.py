"""Wind-specific directional Jacobian audit for the Mdot=5 energy-wind branch."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_standard_slim_stream_mass_annulus_scan as scan  # noqa: E402
from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    algebraic_state,
    jacobian_directional_error,
    select_sonic_compatibility_pivot,
    square_collocation_jacobian,
    square_collocation_residual,
    state_bounds,
    stream_heating_rate,
    unpack_state,
    wind_energy_loss_rate,
)
from imri_qpe.layer3_minidisk_1d.transonic_collocation import _heating_terms_from_gradient  # noqa: E402
from imri_qpe.layer3_minidisk_1d.winds import energy_limited_wind_derivatives, q_edd_vertical  # noqa: E402
from imri_qpe.parameters import FiducialParams  # noqa: E402
from imri_qpe.scales import eddington_mdot  # noqa: E402


ANCHORS: tuple[tuple[str, str], ...] = (
    (
        "ewind_0",
        "outputs/checkpoints/high_mdot_stream_m5_compact_N896_050_to080_no_energy_merit/"
        "m5n896fast2_mass_0p8_torque_0p005_mdot_5_N896.npz",
    ),
    (
        "ewind_0p98",
        "outputs/checkpoints/high_mdot_stream_m5_fs080_smooth_energy_wind_chi099_w005_eta0_windjac_098_N896/"
        "m5smooth_ewind_eta0_chi099_w005_windjac098_mass_0p8_wind_0_heat_0_ewind_0p98_chi_0p99_wfrac_0p005_torque_0p005_mdot_5_N896.npz",
    ),
    (
        "ewind_0p997",
        "outputs/checkpoints/high_mdot_stream_m5_fs080_smooth_energy_wind_chi099_w005_eta0_windjac_0997_0999_N896/"
        "m5smooth_ewind_eta0_chi099_w005_windjac0997_0999_mass_0p8_wind_0_heat_0_ewind_0p997_chi_0p99_wfrac_0p005_torque_0p005_mdot_5_N896.npz",
    ),
    (
        "eta_6p20",
        "outputs/checkpoints/high_mdot_stream_m5_fs080_smooth_energy_wind_chi099_w005_eta0_eta_590_620_N896/"
        "m5smooth_ewind_eta0_chi099_w005_eta590_620_mass_0p8_wind_0_heat_0_ewind_0p997970569_chi_0p99_wfrac_0p005_torque_0p005_mdot_5_N896.npz",
    ),
    (
        "eta_6p35",
        "outputs/checkpoints/high_mdot_stream_m5_fs080_smooth_energy_wind_chi099_w005_eta0_eta_635_N896/"
        "m5smooth_ewind_eta0_chi099_w005_eta635_mass_0p8_wind_0_heat_0_ewind_0p998253253_chi_0p99_wfrac_0p005_torque_0p005_mdot_5_N896.npz",
    ),
)

STEPS = (3.0e-5, 1.0e-5, 3.0e-6, 1.0e-6)
JSON_OUTPUT = ROOT / "outputs/tables/m5_energy_wind_interval_jacobian_audit.json"
MD_OUTPUT = ROOT / "outputs/tables/m5_energy_wind_interval_jacobian_audit.md"


def _eta_from_epsilon(epsilon: float) -> float:
    if not 0.0 <= epsilon < 1.0:
        return math.inf
    return float(-math.log1p(-epsilon))


def _format(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        number = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(number):
        return "nan"
    if number == 0.0:
        return "0"
    if abs(number) < 1.0e-3 or abs(number) >= 1.0e4:
        return f"{number:.3e}"
    return f"{number:.6g}"


def _square_blocks(params) -> dict[str, slice]:
    n_interval = 2 * (int(params.n_nodes) - 1)
    return {
        "interval_R": slice(0, n_interval, 2),
        "interval_E": slice(1, n_interval, 2),
        "outer": slice(n_interval, n_interval + 2),
        "sonic": slice(n_interval + 2, n_interval + 4),
        "all": slice(0, n_interval + 4),
    }


def _safe_direction(z: np.ndarray, direction: np.ndarray, params) -> np.ndarray:
    lower, upper = state_bounds(params)
    v = np.asarray(direction, dtype=float)
    norm = float(np.linalg.norm(v))
    if norm <= 0.0:
        raise ValueError("zero direction")
    v = v / norm
    max_step = np.inf
    positive = v > 0.0
    negative = v < 0.0
    if np.any(positive):
        max_step = min(max_step, float(np.min((upper[positive] - z[positive]) / v[positive])))
    if np.any(negative):
        max_step = min(max_step, float(np.min((lower[negative] - z[negative]) / v[negative])))
    if np.isfinite(max_step) and max_step > 0.0:
        v = v * min(1.0, 0.25 * max_step / max(STEPS))
    return v


def _wind_midpoint_profiles(z: np.ndarray, params) -> dict[str, np.ndarray]:
    logu, logT, _logR_son, lambda0, logR = unpack_state(z, params)
    rows: dict[str, list[float]] = {
        "R_rg": [],
        "Qwind": [],
        "Qavail": [],
        "Qedd": [],
        "activation": [],
        "dQwind_dQavail": [],
        "dQwind_dQedd": [],
    }
    for idx in range(len(logR) - 1):
        dx = float(logR[idx + 1] - logR[idx])
        xm = float(0.5 * (logR[idx] + logR[idx + 1]))
        ym = np.array([0.5 * (logu[idx] + logu[idx + 1]), 0.5 * (logT[idx] + logT[idx + 1])], dtype=float)
        gm = np.array([(logu[idx + 1] - logu[idx]) / dx, (logT[idx + 1] - logT[idx]) / dx], dtype=float)
        qv, _qr, qa, _qe = _heating_terms_from_gradient(xm, ym, gm, lambda0, params)
        qs = stream_heating_rate(xm, params)
        state = algebraic_state(xm, float(ym[0]), float(ym[1]), lambda0, params)
        qwind = wind_energy_loss_rate(state, qv, qs, qa, params)
        qavail = float(qv + qs - qa)
        qedd = float(q_edd_vertical(state.Omega_K, state.H, kappa=params.kappa))
        width_fraction = float(params.wind_activation_width_fraction)
        dqa, dqe = energy_limited_wind_derivatives(
            qavail,
            qedd,
            float(params.wind_energy_limited_epsilon),
            chi_edd=float(params.wind_eddington_chi),
            activation_width=width_fraction * qedd,
            activation_width_dQedd=width_fraction,
        )
        rows["R_rg"].append(float(np.exp(xm) / params.r_g))
        rows["Qwind"].append(float(qwind))
        rows["Qavail"].append(float(qavail))
        rows["Qedd"].append(float(qedd))
        rows["activation"].append(float(qavail - float(params.wind_eddington_chi) * qedd))
        rows["dQwind_dQavail"].append(float(dqa))
        rows["dQwind_dQedd"].append(float(dqe))
    return {key: np.asarray(values, dtype=float) for key, values in rows.items()}


def _localized_directions(z: np.ndarray, params, wind_profiles: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    n = int(params.n_nodes)
    directions: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(20260705)
    directions["random"] = rng.normal(size=z.size)
    lambda_direction = np.zeros_like(z)
    lambda_direction[-1] = 1.0
    directions["lambda0"] = lambda_direction

    qwind = wind_profiles["Qwind"]
    active = np.flatnonzero(qwind > 0.0)
    if active.size:
        peak_interval = int(active[np.argmax(qwind[active])])
    else:
        peak_interval = int(np.argmin(np.abs(wind_profiles["activation"])))
    center_node = min(max(peak_interval + 1, 0), n - 1)
    sigma_nodes = max(2.0, 0.01 * n)
    nodes = np.arange(n, dtype=float)
    weight = np.exp(-0.5 * ((nodes - center_node) / sigma_nodes) ** 2)
    temp = np.zeros_like(z)
    temp[n : 2 * n] = weight
    directions["logT_wind_local"] = temp
    velo = np.zeros_like(z)
    velo[:n] = weight
    directions["logu_wind_local"] = velo
    return {name: _safe_direction(z, direction, params) for name, direction in directions.items()}


def _directional_errors(z: np.ndarray, params, pivot: str, directions: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    jac = square_collocation_jacobian(z, params, pivot=pivot, rel_step=3.0e-5)
    blocks = _square_blocks(params)
    out: list[dict[str, Any]] = []
    for name, direction in directions.items():
        jv = np.asarray(jac @ direction, dtype=float)
        best: dict[str, Any] | None = None
        for step in STEPS:
            plus = square_collocation_residual(z + step * direction, params, pivot=pivot)
            minus = square_collocation_residual(z - step * direction, params, pivot=pivot)
            fd = (plus - minus) / (2.0 * step)
            row: dict[str, Any] = {
                "direction": name,
                "step": float(step),
            }
            for block_name, block_slice in blocks.items():
                lhs = jv[block_slice]
                rhs = fd[block_slice]
                diff = lhs - rhs
                denom = float(np.linalg.norm(lhs) + np.linalg.norm(rhs) + 1.0e-300)
                row[f"{block_name}_rel_l2"] = float(np.linalg.norm(diff) / denom)
                row[f"{block_name}_abs_inf"] = float(np.linalg.norm(diff, ord=np.inf))
            if best is None or row["all_rel_l2"] < best["all_rel_l2"]:
                best = row
        if best is not None:
            out.append(best)
    return out


def audit_anchor(label: str, rel_path: str, fiducial: FiducialParams, mdot_edd: float) -> dict[str, Any]:
    path = ROOT / rel_path
    z, params = scan.load_anchor(path, fiducial, mdot_edd)
    pivot = select_sonic_compatibility_pivot(z, params)
    wind_profiles = _wind_midpoint_profiles(z, params)
    directions = _localized_directions(z, params, wind_profiles)
    t0 = time.time()
    directional = _directional_errors(z, params, pivot, directions)
    elapsed = time.time() - t0
    generic = jacobian_directional_error(
        z,
        params,
        pivot=pivot,
        steps=STEPS,
        n_directions=3,
        seed=7305,
        jacobian_rel_step=3.0e-5,
    )
    qwind = wind_profiles["Qwind"]
    activation = wind_profiles["activation"]
    dqa = wind_profiles["dQwind_dQavail"]
    dqe = wind_profiles["dQwind_dQedd"]
    active = qwind > 0.0
    transition = np.abs(activation) <= np.maximum(0.01 * np.maximum(wind_profiles["Qedd"], 1.0e-300), 1.0e-300)
    return {
        "label": label,
        "checkpoint": rel_path,
        "epsilon_w": float(params.wind_energy_limited_epsilon),
        "eta": _eta_from_epsilon(float(params.wind_energy_limited_epsilon)),
        "N": int(params.n_nodes),
        "pivot": str(pivot),
        "full_residual": scan.max_residual(z, params),
        "generic_best_step": float(generic.best_step),
        "generic_best_median_relative_error": float(generic.best_median_error),
        "generic_max_relative_error_at_best_step": float(
            generic.max_relative_error[int(np.argmin(generic.median_relative_error))]
        ),
        "active_interval_fraction": float(np.count_nonzero(active) / max(qwind.size, 1)),
        "transition_interval_fraction": float(np.count_nonzero(transition) / max(qwind.size, 1)),
        "peak_Qwind_R_rg": float(wind_profiles["R_rg"][int(np.argmax(qwind))]) if qwind.size and np.max(qwind) > 0.0 else np.nan,
        "max_dQwind_dQavail": float(np.max(dqa)) if dqa.size else np.nan,
        "min_dQwind_dQedd": float(np.min(dqe)) if dqe.size else np.nan,
        "directional_errors": directional,
        "elapsed_s": float(elapsed),
    }


def write_markdown(rows: list[dict[str, Any]]) -> None:
    columns = [
        "label",
        "epsilon_w",
        "eta",
        "full_residual",
        "generic_best_median_relative_error",
        "generic_max_relative_error_at_best_step",
        "active_interval_fraction",
        "transition_interval_fraction",
        "peak_Qwind_R_rg",
        "max_dQwind_dQavail",
        "min_dQwind_dQedd",
        "elapsed_s",
    ]
    lines = [
        "# Mdot=5 Energy-Wind Interval Jacobian Audit",
        "",
        "Generated by `scripts/audit_mdot5_wind_interval_jacobian.py`.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format(row.get(column, "")) for column in columns) + " |")
    lines.append("")
    lines.append("## Directional Error Details")
    lines.append("")
    for row in rows:
        lines.append(f"### {row['label']}")
        lines.append("")
        detail_columns = ["direction", "step", "all_rel_l2", "interval_E_rel_l2", "interval_E_abs_inf", "outer_rel_l2", "sonic_rel_l2"]
        lines.append("| " + " | ".join(detail_columns) + " |")
        lines.append("| " + " | ".join("---" for _ in detail_columns) + " |")
        for detail in row["directional_errors"]:
            lines.append("| " + " | ".join(_format(detail.get(column, "")) for column in detail_columns) + " |")
        lines.append("")
    MD_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    MD_OUTPUT.write_text("\n".join(lines) + "\n")


def main() -> None:
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)
    rows = [audit_anchor(label, rel_path, fiducial, mdot_edd) for label, rel_path in ANCHORS]
    JSON_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUTPUT.write_text(json.dumps(scan.json_safe(rows), indent=2, sort_keys=True) + "\n")
    write_markdown(rows)
    print(f"wrote {scan.relative_root_path(JSON_OUTPUT)}")
    print(f"wrote {scan.relative_root_path(MD_OUTPUT)}")


if __name__ == "__main__":
    main()
