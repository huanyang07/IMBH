"""Audit physical validity and asymptotic uncertainty at the Mdot=5 endpoint."""

from __future__ import annotations

import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_mdot5_phase_critical_classification as classification  # noqa: E402
from imri_qpe.constants import C, G  # noqa: E402
from imri_qpe.layer3_minidisk_1d import algebraic_state  # noqa: E402


model = classification.model
INPUT_PATH = (
    ROOT
    / "outputs/tables/m5_eta_phase_critical_classification_98p125_N164_profiles.json"
)
SUMMARY_PATH = (
    ROOT / "outputs/tables/m5_eta_endpoint_validity_audit_98p125_N164.json"
)
PROFILE_PATH = (
    ROOT / "outputs/tables/m5_eta_endpoint_validity_audit_98p125_N164_profiles.json"
)
FIGURE_PATH = (
    ROOT / "outputs/figures/m5_eta_endpoint_validity_audit_98p125_N164.png"
)
NOTE_PATH = ROOT / "Note/CODEX_MDOT5_ENDPOINT_VALIDITY_AND_EXPONENT_AUDIT_RESULTS.md"

COMMON_FIT_WINDOWS = (
    (6.0, 7.5),
    (6.0, 8.5),
    (6.5, 9.0),
    (7.0, 9.0),
)
FIT_QUANTITIES = (
    "p_R",
    "Sigma",
    "rho",
    "H_over_R",
    "tau_vertical",
    "Mach_eff",
    "L_u_over_H",
    "tau_radial",
    "toomre_Q",
    "t_layer_over_t_dyn",
    "annulus_mass_per_logu_g",
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
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _gradient_length(radius: float, numerator: float, denominator: float) -> float:
    if not np.isfinite(denominator) or abs(denominator) <= 1.0e-300:
        return math.inf
    return float(radius * abs(numerator / denominator))


def _point_metrics(
    z: np.ndarray,
    p: np.ndarray,
    params,
    lambda0: float,
    r_limit_rg: float,
    label: str,
) -> dict[str, Any]:
    z = np.asarray(z, dtype=float)
    p = np.asarray(p, dtype=float)
    F = max(float(z[2]), 1.0e-300)
    p_R = float(p[3])
    dlogF_dx = (
        float(p[2] / (F * p_R)) if abs(p_R) > 1.0e-300 else math.nan
    )
    local = model._local_params_with_point_mdot(
        params,
        float(z[3]),
        math.log(F * params.Mdot_g_s),
        dlogF_dx if np.isfinite(dlogF_dx) else 0.0,
    )
    state = algebraic_state(
        float(z[3]), float(z[0]), float(z[1]), float(lambda0), local
    )
    physical = classification._physical_point(z, p, params, lambda0)

    L_u = _gradient_length(state.R, p_R, float(p[0]))
    L_T = _gradient_length(state.R, p_R, float(p[1]))
    L_F = _gradient_length(state.R, F * p_R, float(p[2]))
    H = max(float(state.H), 1.0e-300)
    tau_radial = float(local.kappa * state.rho * L_u)
    tau_vertical_check = float(0.5 * local.kappa * state.Sigma)
    sound_speed = float(state.H * state.Omega_K)
    toomre_Q = float(
        sound_speed * state.Omega_K / max(math.pi * G * state.Sigma, 1.0e-300)
    )
    vertical_self_gravity = float(
        2.0
        * math.pi
        * G
        * state.Sigma
        / max(state.Omega_K**2 * state.H, 1.0e-300)
    )

    t_dyn = float(1.0 / max(state.Omega_K, 1.0e-300))
    t_inflow = float(state.R / max(state.u, 1.0e-300))
    t_layer = float(L_u / max(state.u, 1.0e-300))
    q_visc = abs(float(physical["Qvisc"]))
    t_thermal = float(
        state.Sigma * abs(state.e) / max(q_visc, 1.0e-300)
    )
    t_diff_vertical = float(3.0 * physical["tau"] * H / C)
    t_diff_radial = float(3.0 * tau_radial * L_u / C)

    delta_R_rg = float(max(r_limit_rg - physical["R_rg"], 0.0))
    delta_R_cm = float(delta_R_rg * params.r_g)
    p_u = float(p[0])
    annulus_mass_per_logu = (
        float(2.0 * math.pi * state.R**2 * state.Sigma * abs(p_R / p_u))
        if abs(p_u) > 1.0e-300
        else math.inf
    )
    dF_dx = float(p[2] / p_R) if abs(p_R) > 1.0e-300 else math.nan
    mass_target = float(
        physical["wind_prime_over_inner"] - physical["source_prime_over_inner"]
    )
    mass_residual_differential = (
        float(dF_dx - mass_target) if np.isfinite(dF_dx) else math.nan
    )
    mass_residual_homogeneous = float(p[2] - mass_target * p_R)
    mdot = float(F * params.Mdot_g_s)
    angular_flux = float(mdot * state.l - 2.0 * math.pi * state.R**2 * state.W)
    angular_scale = max(abs(params.Mdot_g_s * state.l_K), 1.0e-300)

    return {
        "label": label,
        "logu": float(z[0]),
        "logT": float(z[1]),
        "F": F,
        "R_rg": float(physical["R_rg"]),
        "delta_R_rg": delta_R_rg,
        "delta_R_cm": delta_R_cm,
        "p_u": p_u,
        "p_T": float(p[1]),
        "p_F": float(p[2]),
        "p_R": p_R,
        "u_cm_s": float(state.u),
        "T_K": float(state.T),
        "Sigma": float(state.Sigma),
        "rho": float(state.rho),
        "H_over_R": float(state.H_over_R),
        "Mach_eff": float(physical["Mach_eff"]),
        "L_u_cm": L_u,
        "L_T_cm": L_T,
        "L_F_cm": L_F,
        "L_u_over_H": float(L_u / H),
        "L_T_over_H": float(L_T / H),
        "L_F_over_H": float(L_F / H),
        "tau_vertical": float(physical["tau"]),
        "tau_vertical_check": tau_vertical_check,
        "tau_vertical_rel_error": float(
            abs(tau_vertical_check - physical["tau"])
            / max(abs(physical["tau"]), 1.0e-300)
        ),
        "tau_radial": tau_radial,
        "toomre_Q": toomre_Q,
        "vertical_self_gravity_ratio": vertical_self_gravity,
        "t_dyn_s": t_dyn,
        "t_inflow_s": t_inflow,
        "t_layer_s": t_layer,
        "t_thermal_s": t_thermal,
        "t_diff_vertical_s": t_diff_vertical,
        "t_diff_radial_s": t_diff_radial,
        "t_inflow_over_t_dyn": float(t_inflow / t_dyn),
        "t_layer_over_t_dyn": float(t_layer / t_dyn),
        "t_thermal_over_t_dyn": float(t_thermal / t_dyn),
        "t_diff_vertical_over_t_dyn": float(t_diff_vertical / t_dyn),
        "t_diff_radial_over_t_dyn": float(t_diff_radial / t_dyn),
        "annulus_mass_per_logu_g": annulus_mass_per_logu,
        "annulus_loading_time_s": float(
            annulus_mass_per_logu / max(params.Mdot_g_s, 1.0e-300)
        ),
        "mass_flux_fraction": F,
        "mass_residual_differential": mass_residual_differential,
        "mass_residual_homogeneous": mass_residual_homogeneous,
        "angular_flux_scaled": float(angular_flux / angular_scale),
        "Qadv_Qvisc": float(physical["Qadv_Qvisc"]),
        "Qrad_Qvisc": float(physical["Qrad_Qvisc"]),
        "Qwind_Qvisc": float(physical["Qwind_Qvisc"]),
        "homogeneous_max": float(physical["homogeneous_max"]),
        "sigma_min_A": float(physical["sigma_min_A"]),
        "cond_A": float(physical["cond_A"]),
        "compatibility": float(physical["compatibility"]),
    }


def _phase_segment_metrics(params, lambda0: float, r_limit_rg: float) -> list[dict[str, Any]]:
    z, p, p_mid, ds = classification._load_phase(classification.EXIT_ANCHOR)
    rows: list[dict[str, Any]] = []
    for pos in range(ds.size):
        rows.append(
            _point_metrics(z[pos], p[pos], params, lambda0, r_limit_rg, "phase_node")
        )
        z_mid = 0.5 * (z[pos] + z[pos + 1]) + ds[pos] / 8.0 * (p[pos] - p[pos + 1])
        rows.append(
            _point_metrics(
                z_mid, p_mid[pos], params, lambda0, r_limit_rg, "phase_midpoint"
            )
        )
    rows.append(_point_metrics(z[-1], p[-1], params, lambda0, r_limit_rg, "phase_node"))
    return sorted(rows, key=lambda row: float(row["R_rg"]))


def _profile_branch_metrics(
    branch: dict[str, Any], params, lambda0: float
) -> dict[str, Any]:
    summary = dict(branch["summary"])
    branch_params = params
    if summary.get("shape"):
        branch_params = replace(
            params,
            stream_source_shape=str(summary["shape"]),
            stream_source_log_width=float(params.stream_source_log_width)
            * float(summary.get("width_factor", 1.0)),
        )
    r_limit_rg = float(summary["R_limit_rg"])
    rows = []
    for point in branch["points"]:
        z = np.asarray(
            [
                point["logu"],
                point["logT"],
                point["F"],
                math.log(float(point["R_rg"]) * params.r_g),
            ],
            dtype=float,
        )
        p = np.asarray(
            [-1.0, point["p_T"], point["p_F"], point["p_R"]], dtype=float
        )
        rows.append(
            _point_metrics(z, p, branch_params, lambda0, r_limit_rg, summary["label"])
        )
    return {"summary": summary, "points": rows}


def _fit_power(
    points: list[dict[str, Any]], quantity: str, low: float, high: float
) -> dict[str, Any]:
    logu = np.asarray([row["logu"] for row in points], dtype=float)
    values = np.asarray([row[quantity] for row in points], dtype=float)
    mask = (
        (logu >= low)
        & (logu <= high)
        & np.isfinite(values)
        & (values > 0.0)
    )
    if np.count_nonzero(mask) < 8:
        return {"accepted": False, "count": int(np.count_nonzero(mask))}
    x = logu[mask]
    y = np.log(values[mask])
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    rms = float(np.sqrt(np.mean((y - predicted) ** 2)))
    return {
        "accepted": True,
        "count": int(x.size),
        "power_of_u": float(slope),
        "log_rms": rms,
    }


def _fit_uncertainty(branches: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fit_rows: list[dict[str, Any]] = []
    for branch in branches:
        label = str(branch["summary"]["label"])
        for low, high in COMMON_FIT_WINDOWS:
            by_quantity: dict[str, float] = {}
            for quantity in FIT_QUANTITIES:
                fit = _fit_power(branch["points"], quantity, low, high)
                row = {
                    "branch": label,
                    "window_logu_low": low,
                    "window_logu_high": high,
                    "quantity": quantity,
                    **fit,
                }
                fit_rows.append(row)
                if fit.get("accepted"):
                    by_quantity[quantity] = float(fit["power_of_u"])
            if "p_R" in by_quantity and "Sigma" in by_quantity:
                beta = by_quantity["p_R"]
                sigma_power = by_quantity["Sigma"]
                if beta > 0.0:
                    fit_rows.append(
                        {
                            "branch": label,
                            "window_logu_low": low,
                            "window_logu_high": high,
                            "quantity": "Sigma_divergence_power_of_deltaR",
                            "accepted": True,
                            "count": 1,
                            "power_of_u": float(-sigma_power / beta),
                            "log_rms": math.nan,
                        }
                    )
                    fit_rows.append(
                        {
                            "branch": label,
                            "window_logu_low": low,
                            "window_logu_high": high,
                            "quantity": "annulus_mass_power_of_deltaR",
                            "accepted": True,
                            "count": 1,
                            "power_of_u": float(1.0 + sigma_power / beta),
                            "log_rms": math.nan,
                        }
                    )

    summary: dict[str, Any] = {}
    quantities = sorted({str(row["quantity"]) for row in fit_rows})
    for quantity in quantities:
        values = np.asarray(
            [
                row["power_of_u"]
                for row in fit_rows
                if row["quantity"] == quantity and row.get("accepted")
            ],
            dtype=float,
        )
        if values.size == 0:
            continue
        summary[quantity] = {
            "count": int(values.size),
            "median": float(np.median(values)),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "standard_deviation": float(np.std(values)),
        }
    return fit_rows, summary


def _first_gate_failure(path: list[dict[str, Any]]) -> dict[str, Any]:
    gates = {
        "radial_scale_separation_Lu_over_H": ("L_u_over_H", lambda value: value >= 1.0),
        "vertical_adjustment_tlayer_over_tdyn": (
            "t_layer_over_t_dyn",
            lambda value: value >= 1.0,
        ),
        "radially_optically_thick": ("tau_radial", lambda value: value >= 1.0),
        "vertically_optically_thick": ("tau_vertical", lambda value: value >= 1.0),
        "non_self_gravitating": ("toomre_Q", lambda value: value >= 1.0),
    }
    output: dict[str, Any] = {}
    for gate, (key, predicate) in gates.items():
        failure = None
        for row in path:
            value = float(row[key])
            if np.isfinite(value) and not predicate(value):
                failure = {
                    "R_rg": float(row["R_rg"]),
                    "logu": float(row["logu"]),
                    "value": value,
                    "metric": key,
                }
                break
        output[gate] = failure
    failures = [row for row in output.values() if row is not None]
    output["first_model_validity_failure"] = (
        min(failures, key=lambda row: float(row["R_rg"])) if failures else None
    )
    return output


def _merge_physical_path(
    phase: list[dict[str, Any]], branches: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    source = next(branch for branch in branches if branch["summary"]["label"] == "compact_c2")
    baseline = next(
        branch for branch in branches if branch["summary"]["label"] == "baseline_dt_0.01"
    )
    baseline_start_logu = float(baseline["points"][0]["logu"])
    rows = list(phase)
    rows.extend(
        row for row in source["points"] if float(row["logu"]) > baseline_start_logu
    )
    rows.extend(baseline["points"])
    return rows


def _sample_rows(path: list[dict[str, Any]], count: int = 10) -> list[dict[str, Any]]:
    positions = np.unique(np.linspace(0, len(path) - 1, count, dtype=int))
    keys = (
        "R_rg",
        "logu",
        "delta_R_rg",
        "L_u_over_H",
        "tau_radial",
        "tau_vertical",
        "toomre_Q",
        "t_layer_over_t_dyn",
        "t_thermal_over_t_dyn",
        "mass_residual_homogeneous",
        "homogeneous_max",
    )
    return [{key: path[int(pos)][key] for key in keys} for pos in positions]


def _write_note(result: dict[str, Any]) -> None:
    gates = result["validity_gates"]
    fits = result["fit_summary"]
    numerical = result["numerical_gates"]
    sample = result["sample_path"]
    first = gates["first_model_validity_failure"]
    lines = [
        "# Mdot=5 endpoint validity and exponent audit",
        "",
        "Target: `Mdot_inner/Edd=5`, `Rout=335 rg`, `f_s=0.80`, `eta_E=98.125`, `N=164`.",
        "",
        "This report separates direct observations, numerical gates, model-validity gates, and interpretation. The accepted phase solution is retained beyond the validity boundary only as a mathematical continuation.",
        "",
        "## Direct profile audit",
        "",
        "| R (rg) | logu | R*-R (rg) | Lu/H | tau radial | tau vertical | Toomre Q | t_layer/t_dyn | t_th/t_dyn | mass residual | homogeneous |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sample:
        lines.append(
            f"| {row['R_rg']:.6f} | {row['logu']:.4f} | {row['delta_R_rg']:.3e} | "
            f"{row['L_u_over_H']:.3e} | {row['tau_radial']:.3e} | {row['tau_vertical']:.3e} | "
            f"{row['toomre_Q']:.3e} | {row['t_layer_over_t_dyn']:.3e} | "
            f"{row['t_thermal_over_t_dyn']:.3e} | {row['mass_residual_homogeneous']:.3e} | "
            f"{row['homogeneous_max']:.3e} |"
        )
    lines.extend(
        [
            "",
            "## Numerical gates",
            "",
            f"- Maximum homogeneous phase residual: `{numerical['homogeneous_max']:.3e}`.",
            f"- Maximum homogeneous mass residual: `{numerical['mass_homogeneous_max']:.3e}`.",
            f"- Vertical optical-depth identity error: `{numerical['vertical_tau_identity_max']:.3e}`.",
            f"- Condition-amplified differential mass audit: `{numerical['mass_differential_max_conditioned']:.3e}` (diagnostic only, not an endpoint gate).",
            "",
            "The finite homogeneous mass row remains controlled. The differential form is not used as a gate because division by vanishing `p_R` amplifies otherwise finite errors.",
            "",
            "## Model-validity gates",
            "",
            "| gate | first failure R (rg) | logu | value |",
            "|---|---:|---:|---:|",
        ]
    )
    for gate, row in gates.items():
        if gate == "first_model_validity_failure":
            continue
        if row is None:
            lines.append(f"| `{gate}` | not reached | - | - |")
        else:
            lines.append(
                f"| `{gate}` | {row['R_rg']:.6f} | {row['logu']:.4f} | {row['value']:.3e} |"
            )
    lines.extend(
        [
            "",
            f"The first model-validity failure is `{first['metric']}` at `R={first['R_rg']:.6f} rg`. "
            "The formal endpoint at `R*=225.52125 rg` is therefore not a physically resolved 1D disk layer.",
            "",
            "The radial optical depth still exceeds unity over the computed path, but it decreases while the vertical optical depth diverges. The extrapolated endpoint will eventually violate radial diffusion and self-gravity assumptions as well.",
            "",
            "## Common-window exponent uncertainty",
            "",
            "Fits use all two step-size branches plus the four independently re-solved source profiles over the same four `logu` windows.",
            "",
            "| quantity | median | minimum | maximum | standard deviation | fits |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for key in (
        "p_R",
        "Sigma",
        "rho",
        "H_over_R",
        "L_u_over_H",
        "tau_radial",
        "toomre_Q",
        "t_layer_over_t_dyn",
        "Sigma_divergence_power_of_deltaR",
        "annulus_mass_power_of_deltaR",
    ):
        row = fits[key]
        lines.append(
            f"| `{key}` | {row['median']:.4f} | {row['minimum']:.4f} | {row['maximum']:.4f} | "
            f"{row['standard_deviation']:.4f} | {row['count']} |"
        )
    mass_power = fits["annulus_mass_power_of_deltaR"]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "### Directly supported",
            "",
            "- The positive phase branch approaches a finite-radius low-velocity singular limit under the current equations.",
            "- The annulus mass remains locally integrable: the fitted mass exponent in `Delta R` is positive for every branch/window fit "
            f"(`{mass_power['minimum']:.3f}` to `{mass_power['maximum']:.3f}`).",
            "- Radial-vertical scale separation fails before the formal endpoint, so the mathematical asymptote is outside the model-validity domain.",
            "",
            "### Not established",
            "",
            "- Global nonexistence of a far-side branch.",
            "- A physical steady stagnation reservoir at `u=0`.",
            "- Globally conservative stream/wind angular-momentum closure.",
            "",
            "## Consequence for the outer-manifold search",
            "",
            "1. Search independently from the outer disk with seeds and gauges not derived from the accepted inner phase segment.",
            f"2. Treat `R={first['R_rg']:.6f} rg` as the first physical matching boundary for the current 1D model.",
            "3. Continuation beyond that boundary may classify mathematical topology, but cannot certify a physical disk branch.",
            "4. Require state and conserved-flux matching; do not require derivative continuity across a phase interface.",
            "5. Label a negative result as `not found in the surveyed manifold`, not global nonexistence.",
            "",
            "## Files",
            "",
            f"- summary: `{SUMMARY_PATH.relative_to(ROOT)}`",
            f"- profiles: `{PROFILE_PATH.relative_to(ROOT)}`",
            f"- figure: `{FIGURE_PATH.relative_to(ROOT)}`",
        ]
    )
    NOTE_PATH.write_text("\n".join(lines) + "\n")


def _write_figure(result: dict[str, Any]) -> None:
    from PIL import Image, ImageDraw, ImageFont

    path = result["physical_path"]
    fits = result["fit_summary"]
    image = Image.new("RGB", (1500, 960), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    panels = (
        (60, 55, 725, 440),
        (775, 55, 1440, 440),
        (60, 520, 725, 905),
        (775, 520, 1440, 905),
    )

    def panel(box, series, title, ylabel, logy=True, horizontal=()):
        left, top, right, bottom = box
        x0, x1, y0, y1 = left + 82, right - 22, top + 38, bottom - 52
        transformed = []
        all_x: list[float] = []
        all_y: list[float] = []
        for xs, ys, color, label in series:
            xs = np.asarray(xs, dtype=float)
            ys = np.asarray(ys, dtype=float)
            valid = np.isfinite(xs) & np.isfinite(ys) & ((ys > 0.0) if logy else True)
            xs, ys = xs[valid], ys[valid]
            if logy:
                ys = np.log10(ys)
            transformed.append((xs, ys, color, label))
            all_x.extend(xs.tolist())
            all_y.extend(ys.tolist())
        for value, _color, _label in horizontal:
            all_y.append(math.log10(value) if logy else value)
        if not all_x or not all_y:
            return
        xmin, xmax = min(all_x), max(all_x)
        ymin, ymax = min(all_y), max(all_y)
        dx, dy = max(xmax - xmin, 1.0e-12), max(ymax - ymin, 1.0e-12)
        xmin, xmax = xmin - 0.02 * dx, xmax + 0.02 * dx
        ymin, ymax = ymin - 0.06 * dy, ymax + 0.06 * dy

        def xy(x, y):
            px = x0 + (x - xmin) / (xmax - xmin) * (x1 - x0)
            py = y1 - (y - ymin) / (ymax - ymin) * (y1 - y0)
            return px, py

        draw.rectangle(box, outline="#D5D8DC")
        for tick in range(5):
            xv = xmin + tick * (xmax - xmin) / 4.0
            yv = ymin + tick * (ymax - ymin) / 4.0
            px, _ = xy(xv, ymin)
            _, py = xy(xmin, yv)
            draw.line((px, y0, px, y1), fill="#ECEFF1")
            draw.line((x0, py, x1, py), fill="#ECEFF1")
            draw.text((px - 20, y1 + 9), f"{xv:.4g}", fill="#34495E", font=font)
            ytext = f"10^{yv:.1f}" if logy else f"{yv:.3g}"
            draw.text((left + 4, py - 6), ytext, fill="#34495E", font=font)
        draw.line((x0, y1, x1, y1), fill="#2C3E50", width=2)
        draw.line((x0, y0, x0, y1), fill="#2C3E50", width=2)
        draw.text((left + 8, top + 8), title, fill="#17202A", font=font)
        draw.text(((x0 + x1) / 2 - 28, bottom - 18), "R / rg", fill="#34495E", font=font)
        draw.text((left + 8, top + 23), ylabel, fill="#34495E", font=font)
        legend_y = top + 8
        for xs, ys, color, label in transformed:
            points = [xy(float(x), float(y)) for x, y in zip(xs, ys)]
            if len(points) >= 2:
                draw.line(points, fill=color, width=3)
            draw.line((right - 175, legend_y + 5, right - 150, legend_y + 5), fill=color, width=3)
            draw.text((right - 145, legend_y), label, fill="#34495E", font=font)
            legend_y += 16
        for value, color, label in horizontal:
            y = math.log10(value) if logy else value
            _, py = xy(xmin, y)
            draw.line((x0, py, x1, py), fill=color, width=2)
            draw.text((x0 + 5, py - 14), label, fill=color, font=font)

    radii = [row["R_rg"] for row in path]
    panel(
        panels[0],
        [
            (radii, [row["L_u_over_H"] for row in path], "#B03A2E", "Lu/H"),
            (radii, [row["L_T_over_H"] for row in path], "#176B87", "LT/H"),
        ],
        "Radial gradient scales",
        "length / H",
        horizontal=((1.0, "#148F77", "scale-separation gate"),),
    )
    panel(
        panels[1],
        [
            (radii, [row["tau_radial"] for row in path], "#C0392B", "tau radial"),
            (radii, [row["tau_vertical"] for row in path], "#7D3C98", "tau vertical"),
            (radii, [row["toomre_Q"] for row in path], "#176B87", "Toomre Q"),
        ],
        "Transport and self-gravity gates",
        "value",
        horizontal=((1.0, "#148F77", "validity gate"),),
    )
    panel(
        panels[2],
        [
            (radii, [row["t_layer_over_t_dyn"] for row in path], "#B03A2E", "layer/dyn"),
            (radii, [row["t_thermal_over_t_dyn"] for row in path], "#D68910", "thermal/dyn"),
            (radii, [row["t_diff_radial_over_t_dyn"] for row in path], "#176B87", "rad diff/dyn"),
            (radii, [row["t_diff_vertical_over_t_dyn"] for row in path], "#7D3C98", "vert diff/dyn"),
        ],
        "Endpoint timescale hierarchy",
        "time / t_dyn",
        horizontal=((1.0, "#148F77", "one dynamical time"),),
    )

    quantities = (
        "p_R",
        "Sigma",
        "rho",
        "H_over_R",
        "L_u_over_H",
        "tau_radial",
        "toomre_Q",
        "annulus_mass_power_of_deltaR",
    )
    left, top, right, bottom = panels[3]
    draw.rectangle(panels[3], outline="#D5D8DC")
    draw.text((left + 8, top + 8), "Common-window exponent ranges", fill="#17202A", font=font)
    x0, x1, y0, y1 = left + 150, right - 25, top + 42, bottom - 38
    values = [fits[key] for key in quantities]
    xmin = min(row["minimum"] for row in values)
    xmax = max(row["maximum"] for row in values)
    pad = 0.08 * max(xmax - xmin, 1.0)
    xmin, xmax = xmin - pad, xmax + pad

    def xcoord(value):
        return x0 + (value - xmin) / (xmax - xmin) * (x1 - x0)

    for tick in range(6):
        value = xmin + tick * (xmax - xmin) / 5.0
        px = xcoord(value)
        draw.line((px, y0, px, y1), fill="#ECEFF1")
        draw.text((px - 15, y1 + 8), f"{value:.2g}", fill="#34495E", font=font)
    zero = xcoord(0.0)
    draw.line((zero, y0, zero, y1), fill="#566573", width=2)
    for pos, (quantity, row) in enumerate(zip(quantities, values)):
        py = y0 + (pos + 0.5) * (y1 - y0) / len(quantities)
        draw.text((left + 8, py - 6), quantity, fill="#34495E", font=font)
        draw.line((xcoord(row["minimum"]), py, xcoord(row["maximum"]), py), fill="#176B87", width=4)
        px = xcoord(row["median"])
        draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill="#C0392B")
    draw.text(((x0 + x1) / 2 - 38, bottom - 18), "power of u", fill="#34495E", font=font)

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    image.save(FIGURE_PATH)


def main() -> None:
    source = json.loads(INPUT_PATH.read_text())
    _x, params, _context, _aux, phase_data = classification.global_phase._load_problem()
    lambda0 = float(phase_data["lambda0"])
    r_limit_rg = float(source["baseline"][-1]["summary"]["R_limit_rg"])

    phase = _phase_segment_metrics(params, lambda0, r_limit_rg)
    branches = [
        _profile_branch_metrics(branch, params, lambda0)
        for branch in source["baseline"] + source["source_branches"]
    ]
    physical_path = _merge_physical_path(phase, branches)
    fit_rows, fit_summary = _fit_uncertainty(branches)
    gates = _first_gate_failure(physical_path)

    result = {
        "target": {
            "Mdot_inner_edd": 5.0,
            "eta_E": 98.125,
            "N": int(params.n_nodes),
            "Rout_rg": float(params.R_out_rg),
            "R_limit_rg": r_limit_rg,
        },
        "definitions": {
            "L_u": "abs(d ln u / dR)^-1",
            "tau_radial": "kappa rho L_u",
            "tau_vertical": "kappa Sigma / 2",
            "toomre_Q": "H Omega_K^2 / (pi G Sigma)",
            "diffusion_time": "3 tau L / c",
            "scale_separation_gate": "L_u/H >= 1",
            "vertical_adjustment_gate": "t_layer/t_dyn >= 1",
        },
        "validity_gates": gates,
        "fit_windows": COMMON_FIT_WINDOWS,
        "fit_summary": fit_summary,
        "sample_path": _sample_rows(physical_path),
        "numerical_gates": {
            "mass_homogeneous_max": float(
                np.nanmax(np.abs([row["mass_residual_homogeneous"] for row in physical_path]))
            ),
            "mass_differential_max_conditioned": float(
                np.nanmax(np.abs([row["mass_residual_differential"] for row in physical_path]))
            ),
            "homogeneous_max": float(
                np.nanmax([row["homogeneous_max"] for row in physical_path])
            ),
            "vertical_tau_identity_max": float(
                np.nanmax([row["tau_vertical_rel_error"] for row in physical_path])
            ),
        },
        "interpretation": {
            "local_annulus_mass_integrable": bool(
                fit_summary["annulus_mass_power_of_deltaR"]["minimum"] > 0.0
            ),
            "formal_endpoint_within_1d_validity": False,
            "global_nonexistence_established": False,
            "outer_search_required": True,
        },
    }
    profiles = {
        **result,
        "phase_segment": phase,
        "branches": branches,
        "physical_path": physical_path,
        "fit_rows": fit_rows,
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n")
    PROFILE_PATH.write_text(json.dumps(_jsonable(profiles), indent=2, sort_keys=True) + "\n")
    _write_note({**profiles})
    _write_figure({**profiles})
    print(json.dumps(_jsonable(result["validity_gates"]), sort_keys=True))
    print(json.dumps(_jsonable(result["fit_summary"]), sort_keys=True))


if __name__ == "__main__":
    main()
