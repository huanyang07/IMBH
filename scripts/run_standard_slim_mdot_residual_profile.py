"""Residual localization for standard slim-disk Mdot continuation cases."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    TransonicSlimParams,
    collocation_residual,
    unpack_state,
)
from imri_qpe.layer3_minidisk_1d.transonic_collocation import _differential_interval_residual_from_unpacked
from imri_qpe.layer3_minidisk_1d.transonic_local import (
    algebraic_state,
    differential_residual,
    differential_residual_scales,
    entropy_gradient_log,
    sonic_diagnostics,
    state_partials,
    stream_heating_rate,
    stream_mass_rate_and_derivative,
    stream_source_prime,
    wind_sink_prime,
)
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot
from run_standard_slim_analytic_seed_audit import ALPHA, fmt, json_safe
from run_standard_slim_mdot_injection_ladder import ACCEPTANCE_TOL, STRESS_FACTOR, dominant


ROOT = Path(__file__).resolve().parents[1]
TABLE_OUTPUT = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_MDOT_RESIDUAL_PROFILE_TABLE",
    "outputs/tables/slim_benchmark_mdot_residual_profiles.md",
)
JSON_OUTPUT = TABLE_OUTPUT.with_suffix(".json")
FIGURE_OUTPUT = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_MDOT_RESIDUAL_PROFILE_FIGURE",
    "outputs/figures/slim_benchmark_mdot_residual_profiles.png",
)

DEFAULT_CASES = (
    "anchor:outputs/checkpoints/slim_benchmark_mdot_injection_rout10000_2pct_ladder/anchor_mdot_0p001.npz,"
    "accepted_down:outputs/checkpoints/slim_benchmark_mdot_injection_rout10000_2pct_ladder/down_mdot_0p00098.npz,"
    "accepted_up:outputs/checkpoints/slim_benchmark_mdot_injection_rout10000_2pct_ladder/up_mdot_0p00102.npz,"
    "failed_down:outputs/checkpoints/slim_benchmark_mdot_injection_rout10000_2pct_ladder/down_mdot_0p000941192.npz,"
    "failed_up:outputs/checkpoints/slim_benchmark_mdot_injection_rout10000_2pct_ladder/up_mdot_0p00110408.npz"
)
CASE_SPECS = tuple(
    piece.strip()
    for piece in os.environ.get("IMBH_STANDARD_SLIM_MDOT_RESIDUAL_PROFILE_CASES", DEFAULT_CASES).replace(";", ",").split(",")
    if piece.strip()
)
TOP_N = int(os.environ.get("IMBH_STANDARD_SLIM_MDOT_RESIDUAL_PROFILE_TOP_N", "8"))


def params_for_checkpoint(fiducial: FiducialParams, mdot_edd: float, data) -> TransonicSlimParams:
    custom_grid_xi = None
    if "custom_grid_xi" in data:
        candidate_grid = np.asarray(data["custom_grid_xi"], dtype=float)
        if candidate_grid.shape == (int(data["n_nodes"]),):
            custom_grid_xi = tuple(float(value) for value in candidate_grid)
    return TransonicSlimParams(
        M2_g=fiducial.M2_g,
        Mdot_g_s=float(data["ratio"]) * mdot_edd,
        alpha=ALPHA,
        mu_stress=0.0,
        stress_factor=STRESS_FACTOR,
        R_out_rg=float(data["R_out_rg"]),
        n_nodes=int(data["n_nodes"]),
        grid_power=float(data["grid_power"]) if "grid_power" in data else 1.0,
        custom_grid_xi=custom_grid_xi,
        outer_temperature_logT=float(data["outer_temperature_logT"])
        if "outer_temperature_logT" in data and np.isfinite(float(data["outer_temperature_logT"]))
        else None,
        outer_entropy_logK=float(data["outer_entropy_logK"])
        if "outer_entropy_logK" in data and np.isfinite(float(data["outer_entropy_logK"]))
        else None,
        outer_omega_log_offset=float(data["outer_omega_log_offset"])
        if "outer_omega_log_offset" in data and np.isfinite(float(data["outer_omega_log_offset"]))
        else 0.0,
        stream_torque_delta_l_fraction=float(data["stream_torque_delta_l_fraction"])
        if "stream_torque_delta_l_fraction" in data and np.isfinite(float(data["stream_torque_delta_l_fraction"]))
        else 0.0,
        stream_torque_center_fraction=float(data["stream_torque_center_fraction"])
        if "stream_torque_center_fraction" in data and np.isfinite(float(data["stream_torque_center_fraction"]))
        else 0.8,
        stream_torque_log_width=float(data["stream_torque_log_width"])
        if "stream_torque_log_width" in data and np.isfinite(float(data["stream_torque_log_width"]))
        else 0.08,
        stream_source_fraction=float(data["stream_source_fraction"])
        if "stream_source_fraction" in data and np.isfinite(float(data["stream_source_fraction"]))
        else 0.0,
        stream_source_center_fraction=float(data["stream_source_center_fraction"])
        if "stream_source_center_fraction" in data and np.isfinite(float(data["stream_source_center_fraction"]))
        else 0.8,
        stream_source_log_width=float(data["stream_source_log_width"])
        if "stream_source_log_width" in data and np.isfinite(float(data["stream_source_log_width"]))
        else 0.08,
        wind_sink_fraction=float(data["wind_sink_fraction"])
        if "wind_sink_fraction" in data and np.isfinite(float(data["wind_sink_fraction"]))
        else 0.0,
        wind_sink_center_fraction=float(data["wind_sink_center_fraction"])
        if "wind_sink_center_fraction" in data and np.isfinite(float(data["wind_sink_center_fraction"]))
        else 0.8,
        wind_sink_log_width=float(data["wind_sink_log_width"])
        if "wind_sink_log_width" in data and np.isfinite(float(data["wind_sink_log_width"]))
        else 0.08,
        stream_heating_efficiency=float(data["stream_heating_efficiency"])
        if "stream_heating_efficiency" in data and np.isfinite(float(data["stream_heating_efficiency"]))
        else 0.0,
        residual_tol=1.0e-8,
        max_nfev=1,
        outer_closure="thin_value",
        interval_residual_form="differential",
        integrated_residual_weighting="none",
    )


def load_case(label: str, path: Path, fiducial: FiducialParams, mdot_edd: float) -> tuple[np.ndarray, TransonicSlimParams]:
    if not path.exists():
        raise FileNotFoundError(f"{label}: checkpoint not found: {path}")
    data = np.load(path, allow_pickle=True)
    z = np.asarray(data["z"], dtype=float)
    params = params_for_checkpoint(fiducial, mdot_edd, data)
    if "outer_closure" in data:
        outer_closure = str(np.asarray(data["outer_closure"]).item())
        slopes = None
        if "outer_match_log_slopes" in data:
            candidate = np.asarray(data["outer_match_log_slopes"], dtype=float)
            if candidate.shape == (2,) and np.all(np.isfinite(candidate)):
                slopes = (float(candidate[0]), float(candidate[1]))
        if slopes is None and outer_closure == "pressure_supported_thin_energy":
            logu, logT, _logR_son, _lambda0, logR = unpack_state(z, params)
            dx = float(logR[-1] - logR[-2])
            slopes = (float((logu[-1] - logu[-2]) / dx), float((logT[-1] - logT[-2]) / dx))
        params = replace(params, outer_closure=outer_closure, outer_match_log_slopes=slopes)
    return z, params


def parse_case_specs() -> list[tuple[str, Path]]:
    cases = []
    for spec in CASE_SPECS:
        if ":" not in spec:
            raise ValueError(f"case spec must be label:path, got {spec!r}")
        label, path = spec.split(":", 1)
        cases.append((label.strip(), ROOT / path.strip()))
    return cases


def interval_residuals(z: np.ndarray, params: TransonicSlimParams) -> np.ndarray:
    logu, logT, _logR_son, lambda0, logR = unpack_state(z, params)
    return np.asarray(
        [
            _differential_interval_residual_from_unpacked(logu, logT, logR, lambda0, params, idx)
            for idx in range(len(logR) - 1)
        ],
        dtype=float,
    )


def midpoint_diagnostics(logR: np.ndarray, logu: np.ndarray, logT: np.ndarray, lambda0: float, params: TransonicSlimParams, idx: int) -> dict[str, float]:
    dx = float(logR[idx + 1] - logR[idx])
    x = float(0.5 * (logR[idx] + logR[idx + 1]))
    y = np.array([0.5 * (logu[idx] + logu[idx + 1]), 0.5 * (logT[idx] + logT[idx + 1])], dtype=float)
    g = np.array([(logu[idx + 1] - logu[idx]) / dx, (logT[idx + 1] - logT[idx]) / dx], dtype=float)
    state = algebraic_state(x, y[0], y[1], lambda0, params)
    partials = state_partials(x, y, lambda0, params, eps_x=params.partial_eps, eps_y=params.partial_eps)
    dPi_dx = partials.x["Pi"] + float(np.dot(partials.y["Pi"], g))
    dOmega_dx = partials.x["Omega"] + float(np.dot(partials.y["Omega"], g))
    Tdsdx = entropy_gradient_log(x, y, g, lambda0, params)
    raw = differential_residual(x, y, g, lambda0, params)
    radial_scale, energy_scale = differential_residual_scales(x, y, lambda0, params)
    Q_visc = -state.W * dOmega_dx
    Q_adv = -(state.Sigma * state.u / state.R) * Tdsdx
    Mdot_local, dMdot_dlnR = stream_mass_rate_and_derivative(x, params)
    source_prime = stream_source_prime(x, params)
    wind_prime = wind_sink_prime(x, params)
    Q_stream = stream_heating_rate(x, params)
    sonic = sonic_diagnostics(x, y, lambda0, params)
    return {
        "R_mid_rg": float(np.exp(x) / params.r_g),
        "dx": dx,
        "raw_R_scaled_check": float(raw[0] / radial_scale),
        "raw_E_scaled_check": float(raw[1] / energy_scale),
        "Omega_frac": float(state.Omega / state.Omega_K - 1.0),
        "pressure_frac": float((dPi_dx / state.Sigma) / (state.R**2 * state.Omega_K**2 + 1.0e-300)),
        "inertia_frac": float((state.u**2 * g[0]) / (state.R**2 * state.Omega_K**2 + 1.0e-300)),
        "Qadv_Qvisc": float(Q_adv / (Q_visc + 1.0e-300)),
        "Qstream_Qvisc": float(Q_stream / (Q_visc + 1.0e-300)),
        "Mdot_over_param": float(Mdot_local / params.Mdot_g_s),
        "dMdot_dlnR_over_param": float(dMdot_dlnR / params.Mdot_g_s),
        "stream_source_prime_over_param": float(source_prime / params.Mdot_g_s),
        "wind_sink_prime_over_param": float(wind_prime / params.Mdot_g_s),
        "qbalance": float((Q_visc - state.Q_rad) / (abs(Q_visc) + abs(state.Q_rad) + 1.0e-300)),
        "H_R": float(state.H_over_R),
        "smin_over_smax_A": float(sonic.smin_over_smax),
        "condA": float(1.0 / (sonic.smin_over_smax + 1.0e-300)),
        "D_mid": float(sonic.D),
        "C1_mid": float(sonic.C1),
        "C2_mid": float(sonic.C2),
        "K_mid": float(sonic.compatibility),
    }


def case_interval_rows(
    *,
    label: str,
    z: np.ndarray,
    params: TransonicSlimParams,
    anchor_z: np.ndarray,
    anchor_params: TransonicSlimParams,
) -> list[dict[str, object]]:
    logu, logT, _logR_son, lambda0, logR = unpack_state(z, params)
    anchor_logu, anchor_logT, _anchor_logR_son, _anchor_lambda0, anchor_logR = unpack_state(anchor_z, anchor_params)
    intervals = interval_residuals(z, params)
    rows: list[dict[str, object]] = []
    for idx, residual in enumerate(intervals):
        x = float(0.5 * (logR[idx] + logR[idx + 1]))
        y_mid = np.array([0.5 * (logu[idx] + logu[idx + 1]), 0.5 * (logT[idx] + logT[idx + 1])], dtype=float)
        anchor_y = np.array(
            [
                np.interp(x, anchor_logR, anchor_logu),
                np.interp(x, anchor_logR, anchor_logT),
            ],
            dtype=float,
        )
        diag = midpoint_diagnostics(logR, logu, logT, lambda0, params, idx)
        rows.append(
            {
                "case": label,
                "ratio": float(params.mdot_edd_ratio),
                "R_out_rg": float(params.R_out_rg),
                "N": int(params.n_nodes),
                "interval": int(idx),
                "R_mid_rg": diag["R_mid_rg"],
                "interval_R": float(residual[0]),
                "interval_E": float(residual[1]),
                "abs_interval_R": float(abs(residual[0])),
                "abs_interval_E": float(abs(residual[1])),
                "delta_logu_anchor": float(y_mid[0] - anchor_y[0]),
                "delta_logT_anchor": float(y_mid[1] - anchor_y[1]),
                **diag,
            }
        )
    return rows


def summary_row(label: str, z: np.ndarray, params: TransonicSlimParams, interval_rows: list[dict[str, object]]) -> dict[str, object]:
    residual = collocation_residual(z, params)
    intervals = np.asarray([[row["interval_R"], row["interval_E"]] for row in interval_rows], dtype=float)
    peak_idx = int(np.argmax(np.abs(intervals[:, 0])))
    peak_E_idx = int(np.argmax(np.abs(intervals[:, 1])))
    audit = __import__("imri_qpe.layer3_minidisk_1d", fromlist=["residual_audit_from_state_vector"]).residual_audit_from_state_vector(z, params)
    return {
        "case": label,
        "ratio": float(params.mdot_edd_ratio),
        "accepted": bool(float(np.max(np.abs(residual))) <= ACCEPTANCE_TOL),
        "full": float(np.max(np.abs(residual))),
        "dominant": dominant(audit),
        "max_abs_interval_R": float(np.max(np.abs(intervals[:, 0]))),
        "max_abs_interval_E": float(np.max(np.abs(intervals[:, 1]))),
        "peak_interval_R_index": int(peak_idx),
        "peak_interval_R_rg": float(interval_rows[peak_idx]["R_mid_rg"]),
        "peak_interval_R_value": float(interval_rows[peak_idx]["interval_R"]),
        "peak_interval_E_index": int(peak_E_idx),
        "peak_interval_E_rg": float(interval_rows[peak_E_idx]["R_mid_rg"]),
        "peak_interval_E_value": float(interval_rows[peak_E_idx]["interval_E"]),
        "median_abs_interval_R": float(np.median(np.abs(intervals[:, 0]))),
        "median_abs_interval_E": float(np.median(np.abs(intervals[:, 1]))),
        "p90_abs_interval_R": float(np.quantile(np.abs(intervals[:, 0]), 0.9)),
        "p90_abs_interval_E": float(np.quantile(np.abs(intervals[:, 1]), 0.9)),
        "max_condA": float(max(row["condA"] for row in interval_rows)),
        "min_smin_over_smax_A": float(min(row["smin_over_smax_A"] for row in interval_rows)),
        "max_abs_delta_logu_anchor": float(max(abs(row["delta_logu_anchor"]) for row in interval_rows)),
        "max_abs_delta_logT_anchor": float(max(abs(row["delta_logT_anchor"]) for row in interval_rows)),
    }


def write_table(summary_rows: list[dict[str, object]], interval_rows: list[dict[str, object]]) -> None:
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Standard Slim Mdot Residual Localization",
        "",
        "Generated by `scripts/run_standard_slim_mdot_residual_profile.py`.",
        "",
        "Rows localize differential interval residuals for accepted and failed Mdot-continuation checkpoints.",
        "",
        "| case | accepted | Mdot/Edd | full | dominant | max abs int R | max abs int E | peak R_R/rg | peak int R | peak R_E/rg | peak int E | median abs int E | p90 abs int E | max condA | max abs dlogu | max abs dlogT |",
        "|---|:---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        formatted = {key: fmt(value) if isinstance(value, (float, int, np.floating, np.integer)) else value for key, value in row.items()}
        lines.append(
            "| {case} | {accepted} | {ratio} | {full} | {dominant} | {max_abs_interval_R} | {max_abs_interval_E} | "
            "{peak_interval_R_rg} | {peak_interval_R_value} | {peak_interval_E_rg} | {peak_interval_E_value} | "
            "{median_abs_interval_E} | {p90_abs_interval_E} | {max_condA} | "
            "{max_abs_delta_logu_anchor} | {max_abs_delta_logT_anchor} |".format(**formatted)
        )
    lines.extend(
        [
            "",
            f"## Top {TOP_N} interval_R peaks per case",
            "",
            "| case | interval | R_mid/rg | interval_R | interval_E | dlogu anchor | dlogT anchor | Omega frac | pressure frac | Qadv/Qvisc | Qstream/Qvisc | Mdot/param | src prime/param | dMdot/dlnR/param | H/R | condA |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for case in [row["case"] for row in summary_rows]:
        selected = [row for row in interval_rows if row["case"] == case]
        selected = sorted(selected, key=lambda row: abs(float(row["interval_R"])), reverse=True)[:TOP_N]
        for row in selected:
            formatted = {key: fmt(value) if isinstance(value, (float, int, np.floating, np.integer)) else value for key, value in row.items()}
            lines.append(
                "| {case} | {interval} | {R_mid_rg} | {interval_R} | {interval_E} | {delta_logu_anchor} | "
                "{delta_logT_anchor} | {Omega_frac} | {pressure_frac} | {Qadv_Qvisc} | {Qstream_Qvisc} | "
                "{Mdot_over_param} | {stream_source_prime_over_param} | {dMdot_dlnR_over_param} | {H_R} | {condA} |".format(**formatted)
            )
    lines.extend(
        [
            "",
            f"## Top {TOP_N} interval_E peaks per case",
            "",
            "| case | interval | R_mid/rg | interval_R | interval_E | dlogu anchor | dlogT anchor | Omega frac | pressure frac | Qadv/Qvisc | Qstream/Qvisc | Mdot/param | src prime/param | dMdot/dlnR/param | H/R | condA |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for case in [row["case"] for row in summary_rows]:
        selected = [row for row in interval_rows if row["case"] == case]
        selected = sorted(selected, key=lambda row: abs(float(row["interval_E"])), reverse=True)[:TOP_N]
        for row in selected:
            formatted = {key: fmt(value) if isinstance(value, (float, int, np.floating, np.integer)) else value for key, value in row.items()}
            lines.append(
                "| {case} | {interval} | {R_mid_rg} | {interval_R} | {interval_E} | {delta_logu_anchor} | "
                "{delta_logT_anchor} | {Omega_frac} | {pressure_frac} | {Qadv_Qvisc} | {Qstream_Qvisc} | "
                "{Mdot_over_param} | {stream_source_prime_over_param} | {dMdot_dlnR_over_param} | {H_R} | {condA} |".format(**formatted)
            )
    TABLE_OUTPUT.write_text("\n".join(lines) + "\n")
    payload = {"summary": summary_rows, "intervals": interval_rows}
    JSON_OUTPUT.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n")


def _log_bounds(values: list[float]) -> tuple[float, float]:
    finite = np.asarray([value for value in values if np.isfinite(value) and value > 0.0], dtype=float)
    if finite.size == 0:
        return -16.0, 0.0
    y = np.log10(np.maximum(finite, 1.0e-16))
    lo = float(np.floor(np.min(y)))
    hi = float(np.ceil(np.max(y)))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def write_figure(interval_rows: list[dict[str, object]]) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:
        print(f"skipping figure: PIL unavailable ({exc})", flush=True)
        return
    cases = list(dict.fromkeys(str(row["case"]) for row in interval_rows))
    colors = {
        "anchor": (60, 60, 60),
        "accepted_down": (31, 119, 180),
        "accepted_up": (44, 160, 44),
        "failed_down": (214, 39, 40),
        "failed_up": (148, 103, 189),
    }
    width, height = 1200, 780
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    panels = [
        (90, 80, 1110, 355, "|interval_R|"),
        (90, 455, 1110, 730, "|interval_E|"),
    ]
    all_x = [float(row["R_mid_rg"]) for row in interval_rows]
    x_min, x_max = _log_bounds(all_x)
    for panel_idx, (x0, y0, x1, y1, title) in enumerate(panels):
        key = "abs_interval_R" if panel_idx == 0 else "abs_interval_E"
        y_min, y_max = _log_bounds([float(row[key]) for row in interval_rows])
        draw.rectangle((x0, y0, x1, y1), outline=(60, 60, 60), width=2)
        draw.text((x0, y0 - 28), title, fill=(20, 20, 20), font=font)
        draw.text((x0 + 4, y0 + 4), f"1e{int(y_max)}", fill=(80, 80, 80), font=font)
        draw.text((x0 + 4, y1 - 18), f"1e{int(y_min)}", fill=(80, 80, 80), font=font)
        for case in cases:
            rows = [row for row in interval_rows if row["case"] == case]
            points = []
            for row in rows:
                xx = np.log10(max(float(row["R_mid_rg"]), 1.0e-300))
                yy = np.log10(max(float(row[key]), 1.0e-16))
                px = x0 + int((xx - x_min) / (x_max - x_min) * (x1 - x0))
                py = y1 - int((yy - y_min) / (y_max - y_min) * (y1 - y0))
                points.append((px, py))
            color = colors.get(case, (20, 20, 20))
            if len(points) >= 2:
                draw.line(points, fill=color, width=2)
            if points:
                draw.ellipse((points[-1][0] - 3, points[-1][1] - 3, points[-1][0] + 3, points[-1][1] + 3), fill=color)
    legend_x = 90
    for idx, case in enumerate(cases):
        color = colors.get(case, (20, 20, 20))
        y = 25 + 18 * idx
        draw.line((legend_x, y + 6, legend_x + 30, y + 6), fill=color, width=3)
        draw.text((legend_x + 38, y), case, fill=(20, 20, 20), font=font)
    draw.text((700, 25), "x-axis: log10(R_mid/rg)", fill=(20, 20, 20), font=font)
    FIGURE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(FIGURE_OUTPUT)


def main() -> None:
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)
    loaded = []
    for label, path in parse_case_specs():
        z, params = load_case(label, path, fiducial, mdot_edd)
        loaded.append((label, z, params))
    anchor_matches = [item for item in loaded if item[0] == "anchor"]
    if not anchor_matches:
        raise RuntimeError("one case must be labeled 'anchor'")
    _anchor_label, anchor_z, anchor_params = anchor_matches[0]
    summaries = []
    all_intervals = []
    for label, z, params in loaded:
        rows = case_interval_rows(label=label, z=z, params=params, anchor_z=anchor_z, anchor_params=anchor_params)
        summaries.append(summary_row(label, z, params, rows))
        all_intervals.extend(rows)
        print(
            f"{label}: ratio={params.mdot_edd_ratio:.7g} full={summaries[-1]['full']:.3e} "
            f"peak_R={summaries[-1]['peak_interval_R_rg']:.4g} accepted={summaries[-1]['accepted']}",
            flush=True,
        )
    write_table(summaries, all_intervals)
    write_figure(all_intervals)
    print(f"wrote {TABLE_OUTPUT}", flush=True)
    print(f"wrote {FIGURE_OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
