"""Power-law mass-loss diagnostics for Mdot=5 energy-wind checkpoints."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_standard_slim_stream_mass_annulus_scan as scan  # noqa: E402
from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    algebraic_state,
    effective_wind_powerlaw_slope,
    required_wind_energy_for_powerlaw_slope,
    stream_heating_rate,
    stream_mass_rate_and_derivative,
    stream_source_prime,
    unpack_state,
    wind_energy_loss_rate,
    wind_energy_per_mass,
    wind_mass_loss_prime_from_energy,
)
from imri_qpe.layer3_minidisk_1d.transonic_collocation import _heating_terms_from_gradient  # noqa: E402
from imri_qpe.parameters import FiducialParams  # noqa: E402
from imri_qpe.scales import eddington_mdot  # noqa: E402


ANCHORS: tuple[tuple[str, str, str], ...] = (
    (
        "ewind_0p98",
        "epsilon_w=0.98",
        "outputs/checkpoints/high_mdot_stream_m5_fs080_smooth_energy_wind_chi099_w005_eta0_windjac_098_N896/"
        "m5smooth_ewind_eta0_chi099_w005_windjac098_mass_0p8_wind_0_heat_0_ewind_0p98_chi_0p99_wfrac_0p005_torque_0p005_mdot_5_N896.npz",
    ),
    (
        "ewind_0p997",
        "epsilon_w=0.997",
        "outputs/checkpoints/high_mdot_stream_m5_fs080_smooth_energy_wind_chi099_w005_eta0_windjac_0997_0999_N896/"
        "m5smooth_ewind_eta0_chi099_w005_windjac0997_0999_mass_0p8_wind_0_heat_0_ewind_0p997_chi_0p99_wfrac_0p005_torque_0p005_mdot_5_N896.npz",
    ),
    (
        "eta_6p20",
        "eta=6.20",
        "outputs/checkpoints/high_mdot_stream_m5_fs080_smooth_energy_wind_chi099_w005_eta0_eta_590_620_N896/"
        "m5smooth_ewind_eta0_chi099_w005_eta590_620_mass_0p8_wind_0_heat_0_ewind_0p997970569_chi_0p99_wfrac_0p005_torque_0p005_mdot_5_N896.npz",
    ),
    (
        "eta_6p35",
        "eta=6.35",
        "outputs/checkpoints/high_mdot_stream_m5_fs080_smooth_energy_wind_chi099_w005_eta0_eta_635_N896/"
        "m5smooth_ewind_eta0_chi099_w005_eta635_mass_0p8_wind_0_heat_0_ewind_0p998253253_chi_0p99_wfrac_0p005_torque_0p005_mdot_5_N896.npz",
    ),
    (
        "eta_6p425",
        "eta=6.425",
        "outputs/checkpoints/m5_energy_wind_eta_adaptive_manual_6425_N896/"
        "eta_adaptive_manual_6425_mass_0p8_wind_0_heat_0_ewind_0p998379467_chi_0p99_wfrac_0p005_torque_0p005_mdot_5_N896.npz",
    ),
)

TARGET_S = (0.3, 0.5, 0.7, 1.0)
JSON_OUTPUT = ROOT / "outputs/tables/m5_energy_wind_powerlaw_slope_diagnostics.json"
PROFILE_JSON_OUTPUT = ROOT / "outputs/tables/m5_energy_wind_powerlaw_slope_profiles.json"
MD_OUTPUT = ROOT / "outputs/tables/m5_energy_wind_powerlaw_slope_diagnostics.md"
FIG_OUTPUT = ROOT / "outputs/figures/m5_energy_wind_powerlaw_slope_diagnostics.png"


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


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if int(np.count_nonzero(mask)) == 0:
        return math.nan
    return float(np.sum(values[mask] * weights[mask]) / np.sum(weights[mask]))


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if int(np.count_nonzero(mask)) == 0:
        return math.nan
    v = np.asarray(values[mask], dtype=float)
    w = np.asarray(weights[mask], dtype=float)
    order = np.argsort(v)
    v = v[order]
    w = w[order]
    cumulative = np.cumsum(w)
    target = float(quantile) * cumulative[-1]
    return float(np.interp(target, cumulative, v))


def _active_fraction(values: np.ndarray, weights: np.ndarray, lo: float, hi: float) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if int(np.count_nonzero(mask)) == 0:
        return math.nan
    selected = mask & (values >= lo) & (values <= hi)
    return float(np.sum(weights[selected]) / np.sum(weights[mask]))


def _profile_for_checkpoint(label: str, state_label: str, rel_path: str, fiducial: FiducialParams, mdot_edd: float) -> tuple[dict[str, Any], dict[str, Any]]:
    z, params = scan.load_anchor(ROOT / rel_path, fiducial, mdot_edd)
    logu, logT, _logR_son, lambda0, logR = unpack_state(z, params)

    R_values: list[float] = []
    qwind_values: list[float] = []
    qvisc_values: list[float] = []
    mdot_values: list[float] = []
    stream_prime_values: list[float] = []
    mwind_prime_values: list[float] = []
    s_w_values: list[float] = []
    s_net_values: list[float] = []
    ebind_values: list[float] = []

    for idx in range(len(logR) - 1):
        dx = float(logR[idx + 1] - logR[idx])
        xm = float(0.5 * (logR[idx] + logR[idx + 1]))
        ym = np.array([0.5 * (logu[idx] + logu[idx + 1]), 0.5 * (logT[idx] + logT[idx + 1])], dtype=float)
        gm = np.array([(logu[idx + 1] - logu[idx]) / dx, (logT[idx + 1] - logT[idx]) / dx], dtype=float)
        qv, _qr, qa, _qe = _heating_terms_from_gradient(xm, ym, gm, lambda0, params)
        qs = stream_heating_rate(xm, params)
        state = algebraic_state(xm, float(ym[0]), float(ym[1]), lambda0, params)
        qw = wind_energy_loss_rate(state, qv, qs, qa, params)
        mdot, _dmdot = stream_mass_rate_and_derivative(xm, params)
        source_prime = stream_source_prime(xm, params)
        e_bind = wind_energy_per_mass(params.M2_g, state.R)
        mwind_prime = wind_mass_loss_prime_from_energy(qw, state.R, e_bind)
        s_w = effective_wind_powerlaw_slope(qw, state.R, mdot, e_bind)
        s_net = (float(mwind_prime) - source_prime) / mdot

        R_values.append(float(state.R))
        qwind_values.append(float(qw))
        qvisc_values.append(float(qv))
        mdot_values.append(float(mdot))
        stream_prime_values.append(float(source_prime))
        mwind_prime_values.append(float(mwind_prime))
        s_w_values.append(float(s_w))
        s_net_values.append(float(s_net))
        ebind_values.append(float(e_bind))

    R = np.asarray(R_values, dtype=float)
    logR_mid = np.log(R)
    qwind = np.asarray(qwind_values, dtype=float)
    qvisc = np.asarray(qvisc_values, dtype=float)
    mdot = np.asarray(mdot_values, dtype=float)
    source_prime = np.asarray(stream_prime_values, dtype=float)
    mwind_prime = np.asarray(mwind_prime_values, dtype=float)
    s_w = np.asarray(s_w_values, dtype=float)
    s_net = np.asarray(s_net_values, dtype=float)
    e_bind = np.asarray(ebind_values, dtype=float)
    wind_weights = np.maximum(mwind_prime, 0.0)
    active = qwind > max(float(np.nanmax(qwind)) * 1.0e-10, 0.0) if qwind.size else np.asarray([], dtype=bool)

    stream_info = scan.stream_diagnostic(z, params)
    wind_info = scan.wind_energy_diagnostic(z, params)
    adv_info = scan.advection_diagnostic(z, params)
    integrated_mwind = float(np.trapezoid(mwind_prime, logR_mid))
    integrated_stream_in_active = (
        float(np.trapezoid(source_prime[active], logR_mid[active])) if int(np.count_nonzero(active)) >= 2 else math.nan
    )

    row: dict[str, Any] = {
        "label": label,
        "state_label": state_label,
        "checkpoint": rel_path,
        "epsilon_w": float(params.wind_energy_limited_epsilon),
        "Qwind_Qvisc": float(wind_info["integrated_Qwind_Qvisc"]),
        "Lrad_LEdd": float(adv_info["Lrad_LEdd"]),
        "f_adv_global": float(adv_info["f_adv_global"]),
        "Mdot_outer_over_inner_current": float(stream_info["Mdot_outer_over_inner"]),
        "implied_Mwind_over_inner_etaesc1": float(integrated_mwind / params.Mdot_g_s),
        "required_Mdot_outer_over_inner_etaesc1": float(stream_info["Mdot_outer_over_inner"] + integrated_mwind / params.Mdot_g_s),
        "active_R_min_rg": float(np.nanmin(R[active] / params.r_g)) if int(np.count_nonzero(active)) else math.nan,
        "active_R_max_rg": float(np.nanmax(R[active] / params.r_g)) if int(np.count_nonzero(active)) else math.nan,
        "wind_mass_weighted_s_eff_etaesc1": _weighted_mean(s_w, wind_weights),
        "wind_mass_weighted_s_net_etaesc1": _weighted_mean(s_net, wind_weights),
        "s_eff_etaesc1_p10": _weighted_quantile(s_w, wind_weights, 0.10),
        "s_eff_etaesc1_p50": _weighted_quantile(s_w, wind_weights, 0.50),
        "s_eff_etaesc1_p90": _weighted_quantile(s_w, wind_weights, 0.90),
        "s_eff_etaesc1_max_active": float(np.nanmax(s_w[active])) if int(np.count_nonzero(active)) else math.nan,
        "wind_weight_fraction_s_0_to_1_etaesc1": _active_fraction(s_w, wind_weights, 0.0, 1.0),
        "stream_source_integral_active_over_inner": float(integrated_stream_in_active / params.Mdot_g_s),
    }

    for target_s in TARGET_S:
        e_req = required_wind_energy_for_powerlaw_slope(qwind, R, mdot, target_s)
        eta_req = np.asarray(e_req, dtype=float) / np.maximum(e_bind, 1.0e-300)
        safe = str(target_s).replace(".", "p")
        row[f"etaE_req_p50_s{safe}"] = _weighted_quantile(eta_req, wind_weights, 0.50)
        row[f"etaE_req_p10_s{safe}"] = _weighted_quantile(eta_req, wind_weights, 0.10)
        row[f"etaE_req_p90_s{safe}"] = _weighted_quantile(eta_req, wind_weights, 0.90)

    profile = {
        "label": label,
        "state_label": state_label,
        "R_rg": (R / params.r_g).tolist(),
        "Qwind_Qvisc_local": (qwind / np.maximum(np.abs(qvisc), 1.0e-300)).tolist(),
        "Mdot_over_inner": (mdot / params.Mdot_g_s).tolist(),
        "Mwind_prime_over_inner": (mwind_prime / params.Mdot_g_s).tolist(),
        "stream_prime_over_inner": (source_prime / params.Mdot_g_s).tolist(),
        "s_eff_etaesc1": s_w.tolist(),
        "s_net_etaesc1": s_net.tolist(),
    }
    for target_s in TARGET_S:
        e_req = required_wind_energy_for_powerlaw_slope(qwind, R, mdot, target_s)
        safe = str(target_s).replace(".", "p")
        profile[f"etaE_req_s{safe}"] = (np.asarray(e_req, dtype=float) / np.maximum(e_bind, 1.0e-300)).tolist()
    return row, profile


def _write_markdown(rows: list[dict[str, Any]]) -> None:
    columns = [
        "state_label",
        "epsilon_w",
        "Qwind_Qvisc",
        "Lrad_LEdd",
        "f_adv_global",
        "Mdot_outer_over_inner_current",
        "implied_Mwind_over_inner_etaesc1",
        "required_Mdot_outer_over_inner_etaesc1",
        "active_R_min_rg",
        "active_R_max_rg",
        "s_eff_etaesc1_p10",
        "s_eff_etaesc1_p50",
        "s_eff_etaesc1_p90",
        "wind_weight_fraction_s_0_to_1_etaesc1",
        "etaE_req_p50_s0p3",
        "etaE_req_p50_s0p5",
        "etaE_req_p50_s0p7",
        "etaE_req_p50_s1p0",
        "wind_mass_weighted_s_net_etaesc1",
    ]
    lines = [
        "# Mdot=5 Energy-Wind Power-Law Slope Diagnostics",
        "",
        "Generated by `scripts/audit_mdot5_wind_powerlaw_slope.py`.",
        "",
        "The audit interprets the existing energy-only wind profile as if it were coupled to mass loss with",
        "`E_w = GM/(2R)` and asks whether the implied wind-only slope resembles a simulation-style",
        "`Mdot(R) proportional to R^s` law.",
        "",
        "`s_eff` is wind-only. `s_net` also subtracts the compact stream source, which is useful for this",
        "stream-fed minidisk but should not be compared directly to isolated-disk wind simulations.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format(row.get(column, "")) for column in columns) + " |")
    lines.extend(
        [
            "",
            "## Interpretation Rule",
            "",
            "- If `s_eff` with `E_w=GM/(2R)` is near `0.3-1`, the energy-limited wind naturally matches simulation-style mass loss.",
            "- If `s_eff` is several times larger than one, a target-`s` closure would require a larger effective launch energy,",
            "  `E_w = eta_E GM/(2R)`, with `eta_E approximately s_eff/s_target`.",
            "- Values of `s_net` can be negative because the compact stream source adds mass inward; this is a stream-fed geometry effect,",
            "  not a contradiction of the wind-only slope.",
            "",
            f"Figure: `{scan.relative_root_path(FIG_OUTPUT)}`",
        ]
    )
    MD_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    MD_OUTPUT.write_text("\n".join(lines) + "\n")


def _plot_profiles(profiles: list[dict[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting is an optional audit artifact
        print(f"plot skipped: {exc}")
        _plot_profiles_pil(profiles)
        return

    FIG_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 7.0), sharex=True)
    ax_s, ax_q = axes
    for profile in profiles:
        R = np.asarray(profile["R_rg"], dtype=float)
        s_eff = np.asarray(profile["s_eff_etaesc1"], dtype=float)
        s_net = np.asarray(profile["s_net_etaesc1"], dtype=float)
        qratio = np.asarray(profile["Qwind_Qvisc_local"], dtype=float)
        label = str(profile["state_label"])
        active = qratio > max(float(np.nanmax(qratio)) * 1.0e-10, 0.0) if qratio.size else np.zeros_like(qratio, dtype=bool)
        if int(np.count_nonzero(active)) == 0:
            continue
        ax_s.plot(R[active], s_eff[active], label=label)
        ax_s.plot(R[active], s_net[active], linestyle="--", alpha=0.55)
        ax_q.plot(R[active], qratio[active], label=label)

    ax_s.axhspan(0.3, 1.0, color="0.85", alpha=0.6, label="simulation s~0.3-1")
    ax_s.axhline(0.0, color="0.3", linewidth=0.8)
    ax_s.set_xscale("log")
    ax_s.set_yscale("symlog", linthresh=0.1)
    ax_s.set_ylabel("s_eff (solid), s_net (dashed)")
    ax_s.legend(fontsize=8, ncol=2)
    ax_s.grid(True, which="both", alpha=0.25)

    ax_q.set_xscale("log")
    ax_q.set_yscale("log")
    ax_q.set_xlabel("R / rg")
    ax_q.set_ylabel("local Qwind / |Qvisc|")
    ax_q.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_OUTPUT, dpi=180)
    plt.close(fig)


def _plot_profiles_pil(profiles: list[dict[str, Any]]) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:  # pragma: no cover - optional artifact
        print(f"PIL plot skipped: {exc}")
        return

    width, height = 1200, 900
    margin_left, margin_right = 90, 30
    panel_h = 330
    panel_gap = 90
    top1 = 70
    top2 = top1 + panel_h + panel_gap
    plot_w = width - margin_left - margin_right
    x_min, x_max = 4.0, 335.0
    y1_min, y1_max = -1.0, 3.0
    y2_min, y2_max = 1.0e-4, 10.0
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("Arial.ttf", 18)
        small = ImageFont.truetype("Arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
        small = ImageFont.load_default()

    def xmap(r: float) -> float:
        return margin_left + (math.log10(r) - math.log10(x_min)) / (math.log10(x_max) - math.log10(x_min)) * plot_w

    def ymap_linear(y: float, top: int) -> float:
        y = min(max(y, y1_min), y1_max)
        return top + panel_h - (y - y1_min) / (y1_max - y1_min) * panel_h

    def ymap_log(y: float, top: int) -> float:
        y = min(max(y, y2_min), y2_max)
        return top + panel_h - (math.log10(y) - math.log10(y2_min)) / (math.log10(y2_max) - math.log10(y2_min)) * panel_h

    def draw_axes(top: int, ylabel: str, log_y: bool = False) -> None:
        left = margin_left
        right = margin_left + plot_w
        bottom = top + panel_h
        draw.rectangle((left, top, right, bottom), outline="#333333", width=1)
        x_ticks = [5, 10, 30, 100, 300]
        for tick in x_ticks:
            x = xmap(float(tick))
            draw.line((x, bottom, x, bottom + 6), fill="#333333")
            draw.text((x - 12, bottom + 10), str(tick), fill="#333333", font=small)
        if log_y:
            y_ticks = [1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0]
            for tick in y_ticks:
                y = ymap_log(tick, top)
                draw.line((left - 6, y, left, y), fill="#333333")
                draw.text((8, y - 8), f"{tick:g}", fill="#333333", font=small)
        else:
            y_ticks = [-1, 0, 0.3, 1, 2, 3]
            for tick in y_ticks:
                y = ymap_linear(float(tick), top)
                draw.line((left - 6, y, left, y), fill="#333333")
                draw.text((35, y - 8), f"{tick:g}", fill="#333333", font=small)
        draw.text((left + 5, top - 30), ylabel, fill="#111111", font=font)

    def draw_polyline(points: list[tuple[float, float]], color: str, dashed: bool = False) -> None:
        if len(points) < 2:
            return
        if not dashed:
            draw.line(points, fill=color, width=3, joint="curve")
            return
        for i in range(len(points) - 1):
            if i % 2 == 0:
                draw.line((points[i], points[i + 1]), fill=color, width=2)

    band_y0 = ymap_linear(1.0, top1)
    band_y1 = ymap_linear(0.3, top1)
    draw.rectangle((margin_left, band_y0, margin_left + plot_w, band_y1), fill="#e8e8e8")
    draw_axes(top1, "wind-only s_eff (solid), net with stream source (dashed)")
    draw_axes(top2, "local Qwind / |Qvisc|", log_y=True)
    draw.text((margin_left, top2 + panel_h + 45), "R / rg", fill="#111111", font=font)
    draw.text((margin_left + 10, top1 + 8), "simulation-like s=0.3-1 band", fill="#555555", font=small)

    legend_x = margin_left + plot_w - 260
    legend_y = 35
    for idx, profile in enumerate(profiles):
        color = colors[idx % len(colors)]
        R = np.asarray(profile["R_rg"], dtype=float)
        s_eff = np.asarray(profile["s_eff_etaesc1"], dtype=float)
        s_net = np.asarray(profile["s_net_etaesc1"], dtype=float)
        qratio = np.asarray(profile["Qwind_Qvisc_local"], dtype=float)
        if qratio.size == 0:
            continue
        active = qratio > max(float(np.nanmax(qratio)) * 1.0e-10, 0.0)
        if int(np.count_nonzero(active)) < 2:
            continue
        R_active = R[active]
        s_eff_active = s_eff[active]
        s_net_active = s_net[active]
        qratio_active = qratio[active]
        s_points = [(xmap(float(r)), ymap_linear(float(s), top1)) for r, s in zip(R_active, s_eff_active)]
        net_points = [(xmap(float(r)), ymap_linear(float(s), top1)) for r, s in zip(R_active, s_net_active)]
        q_points = [(xmap(float(r)), ymap_log(float(q), top2)) for r, q in zip(R_active, qratio_active)]
        draw_polyline(s_points, color)
        draw_polyline(net_points, color, dashed=True)
        draw_polyline(q_points, color)
        y = legend_y + idx * 22
        draw.line((legend_x, y + 8, legend_x + 28, y + 8), fill=color, width=3)
        draw.text((legend_x + 35, y), str(profile["state_label"]), fill="#111111", font=small)

    FIG_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(FIG_OUTPUT)


def main() -> None:
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)
    rows: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    for label, state_label, rel_path in ANCHORS:
        row, profile = _profile_for_checkpoint(label, state_label, rel_path, fiducial, mdot_edd)
        rows.append(row)
        profiles.append(profile)

    JSON_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUTPUT.write_text(json.dumps(scan.json_safe(rows), indent=2, sort_keys=True) + "\n")
    PROFILE_JSON_OUTPUT.write_text(json.dumps(scan.json_safe(profiles), indent=2, sort_keys=True) + "\n")
    _write_markdown(rows)
    _plot_profiles(profiles)
    print(f"wrote {scan.relative_root_path(JSON_OUTPUT)}")
    print(f"wrote {scan.relative_root_path(PROFILE_JSON_OUTPUT)}")
    print(f"wrote {scan.relative_root_path(MD_OUTPUT)}")
    if FIG_OUTPUT.exists():
        print(f"wrote {scan.relative_root_path(FIG_OUTPUT)}")


if __name__ == "__main__":
    main()
