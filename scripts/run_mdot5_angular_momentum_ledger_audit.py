"""Audit conservative stream/wind angular-momentum bookkeeping at Mdot=5."""

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

import run_mdot5_phase_critical_classification as classification  # noqa: E402
from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    algebraic_flux_ledger,
    algebraic_state,
    stream_source_prime,
)


model = classification.model
SUMMARY_PATH = ROOT / "outputs/tables/m5_eta_angular_momentum_ledger_98p125_N164.json"
PROFILE_PATH = ROOT / "outputs/tables/m5_eta_angular_momentum_ledger_98p125_N164_profiles.json"
FIGURE_PATH = ROOT / "outputs/figures/m5_eta_angular_momentum_ledger_98p125_N164.png"
NOTE_PATH = ROOT / "Note/CODEX_MDOT5_ANGULAR_MOMENTUM_LEDGER_RESULTS.md"

CLOSURES = (
    "representation",
    "local_disk_prescribed",
    "local_disk_required",
    "keplerian_injection_prescribed",
    "keplerian_injection_required",
    "keplerian_local_prescribed",
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


def _phase_point(z, p, params, lambda0: float) -> dict[str, Any]:
    z = np.asarray(z, dtype=float)
    p = np.asarray(p, dtype=float)
    F = max(float(z[2]), 1.0e-300)
    p_R = float(p[3])
    dlogF_dx = float(p[2] / (F * p_R))
    local = model._local_params_with_point_mdot(
        params, float(z[3]), math.log(F * params.Mdot_g_s), dlogF_dx
    )
    y = np.asarray(z[:2], dtype=float)
    g = np.asarray(p[:2], dtype=float) / p_R
    state = algebraic_state(float(z[3]), float(y[0]), float(y[1]), float(lambda0), local)
    wind_prime = model._safe_wind_prime(
        float(z[3]), y, g, float(lambda0), local
    )
    if not np.isfinite(wind_prime):
        wind_prime = 0.0
    source_prime = float(stream_source_prime(float(z[3]), local))
    mdot = float(F * params.Mdot_g_s)
    mdot_prime = float(wind_prime - source_prime)
    scale = max(abs(params.Mdot_g_s * state.l_K), 1.0e-300)
    ledgers = {
        closure: algebraic_flux_ledger(
            float(z[3]),
            state,
            local,
            mdot=mdot,
            mdot_prime=mdot_prime,
            wind_prime=float(wind_prime),
            stream_prime=source_prime,
            closure=closure,
        )
        for closure in CLOSURES
    }
    prescribed = ledgers["local_disk_prescribed"].external_torque
    return {
        "R_rg": float(state.R / params.r_g),
        "logu": float(z[0]),
        "F": F,
        "p_R": p_R,
        "wind_prime_over_inner": float(wind_prime / params.Mdot_g_s),
        "stream_prime_over_inner": float(source_prime / params.Mdot_g_s),
        "mdot_prime_over_inner": float(mdot_prime / params.Mdot_g_s),
        "viscous_torque_scaled": float(
            2.0 * np.pi * state.R**2 * state.W / scale
        ),
        "angular_scale": scale,
        "closures": {
            closure: {
                "angular_flux": float(row.angular_flux),
                "wind_carried": float(row.wind_carried),
                "stream_carried": float(row.stream_carried),
                "external_torque": float(row.external_torque),
                "angular_flux_scaled": float(row.angular_flux / scale),
                "flux_prime_scaled": float(row.angular_flux_prime / scale),
                "wind_carried_scaled": float(row.wind_carried / scale),
                "stream_carried_scaled": float(row.stream_carried / scale),
                "external_torque_scaled": float(row.external_torque / scale),
                "residual_scaled": float(row.residual / scale),
                "required_torque_correction_scaled": float(
                    (row.external_torque - prescribed) / scale
                ),
                "flux_specific_l_over_lK": float(row.flux_specific_l / state.l_K),
                "l_stream_over_lK": float(row.l_stream / state.l_K),
                "l_wind_over_lK": float(row.l_wind / state.l_K),
            }
            for closure, row in ledgers.items()
        },
    }


def _phase_audit(params, lambda0: float) -> dict[str, Any]:
    z, p, p_mid, ds = classification._load_phase(classification.EXIT_ANCHOR)
    intervals: list[dict[str, Any]] = []
    points: list[dict[str, Any]] = []
    for pos in range(ds.size):
        z_mid = 0.5 * (z[pos] + z[pos + 1]) + ds[pos] / 8.0 * (
            p[pos] - p[pos + 1]
        )
        left = _phase_point(z[pos], p[pos], params, lambda0)
        middle = _phase_point(z_mid, p_mid[pos], params, lambda0)
        right = _phase_point(z[pos + 1], p[pos + 1], params, lambda0)
        if pos == 0:
            points.append(left)
        points.extend([middle, right])
        row: dict[str, Any] = {
            "interval": pos,
            "R_mid_rg": float(middle["R_rg"]),
            "closures": {},
        }
        for closure in CLOSURES:
            lrow = left["closures"][closure]
            mrow = middle["closures"][closure]
            rrow = right["closures"][closure]
            source_integral_raw = float(ds[pos]) / 6.0 * (
                p[pos, 3]
                * (
                    lrow["wind_carried"]
                    - lrow["stream_carried"]
                    + lrow["external_torque"]
                )
                + 4.0
                * p_mid[pos, 3]
                * (
                    mrow["wind_carried"]
                    - mrow["stream_carried"]
                    + mrow["external_torque"]
                )
                + p[pos + 1, 3]
                * (
                    rrow["wind_carried"]
                    - rrow["stream_carried"]
                    + rrow["external_torque"]
                )
            )
            scale = max(float(middle["angular_scale"]), 1.0e-300)
            flux_jump = float((rrow["angular_flux"] - lrow["angular_flux"]) / scale)
            source_integral = float(source_integral_raw / scale)
            row["closures"][closure] = {
                "flux_jump_scaled": flux_jump,
                "source_integral_scaled": source_integral,
                "FV_residual_scaled": float(flux_jump - source_integral),
            }
        intervals.append(row)
    return {"points": points, "intervals": intervals}


def _global_audit(x_log, params) -> list[dict[str, Any]]:
    logu, logT, logMdot, _logR_son, lambda0, logR = model.pilot._unpack(
        np.asarray(x_log, dtype=float), params
    )
    slope_u = np.gradient(np.asarray(logu, dtype=float), np.asarray(logR, dtype=float))
    slope_T = np.gradient(np.asarray(logT, dtype=float), np.asarray(logR, dtype=float))
    slope_mdot = np.gradient(
        np.asarray(logMdot, dtype=float), np.asarray(logR, dtype=float)
    )
    output: list[dict[str, Any]] = []
    for pos, x in enumerate(np.asarray(logR, dtype=float)):
        local = model._local_params_with_point_mdot(
            params, float(x), float(logMdot[pos]), float(slope_mdot[pos])
        )
        y = np.asarray([logu[pos], logT[pos]], dtype=float)
        g = np.asarray([slope_u[pos], slope_T[pos]], dtype=float)
        state = algebraic_state(
            float(x), float(y[0]), float(y[1]), float(lambda0), local
        )
        wind_prime = model._safe_wind_prime(
            float(x), y, g, float(lambda0), local
        )
        if not np.isfinite(wind_prime):
            wind_prime = 0.0
        source_prime = float(stream_source_prime(float(x), local))
        mdot = float(np.exp(logMdot[pos]))
        mdot_prime = float(wind_prime - source_prime)
        scale = max(abs(params.Mdot_g_s * state.l_K), 1.0e-300)
        ledgers = {
            closure: algebraic_flux_ledger(
                float(x),
                state,
                local,
                mdot=mdot,
                mdot_prime=mdot_prime,
                wind_prime=float(wind_prime),
                stream_prime=source_prime,
                closure=closure,
            )
            for closure in CLOSURES
        }
        prescribed = ledgers["local_disk_prescribed"].external_torque
        output.append(
            {
                "R_rg": float(state.R / params.r_g),
                "wind_prime_over_inner": float(wind_prime / params.Mdot_g_s),
                "stream_prime_over_inner": float(source_prime / params.Mdot_g_s),
                "closures": {
                    closure: {
                        "residual_scaled": float(row.residual / scale),
                        "required_torque_correction_scaled": float(
                            (row.external_torque - prescribed) / scale
                        ),
                    }
                    for closure, row in ledgers.items()
                },
            }
        )
    return output


def _max_profile(rows, closure: str, key: str, radius_key: str) -> tuple[float, float, float]:
    values = np.asarray(
        [float(row["closures"][closure][key]) for row in rows], dtype=float
    )
    radii = np.asarray([float(row[radius_key]) for row in rows], dtype=float)
    pos = int(np.nanargmax(np.abs(values)))
    return (
        float(abs(values[pos])),
        float(np.sqrt(np.nanmean(values**2))),
        float(radii[pos]),
    )


def _summarize(phase: dict[str, Any], global_rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for closure in CLOSURES:
        phase_point = _max_profile(
            phase["points"], closure, "residual_scaled", "R_rg"
        )
        phase_fv = _max_profile(
            phase["intervals"], closure, "FV_residual_scaled", "R_mid_rg"
        )
        global_point = _max_profile(
            global_rows, closure, "residual_scaled", "R_rg"
        )
        correction = _max_profile(
            phase["points"],
            closure,
            "required_torque_correction_scaled",
            "R_rg",
        )
        output[closure] = {
            "phase_point_max": phase_point[0],
            "phase_point_rms": phase_point[1],
            "phase_point_peak_R_rg": phase_point[2],
            "phase_FV_max": phase_fv[0],
            "phase_FV_rms": phase_fv[1],
            "phase_FV_peak_R_rg": phase_fv[2],
            "global_point_max": global_point[0],
            "global_point_rms": global_point[1],
            "global_point_peak_R_rg": global_point[2],
            "required_torque_correction_max": correction[0],
            "required_torque_correction_rms": correction[1],
            "required_torque_correction_peak_R_rg": correction[2],
        }
    return output


def _write_note(result: dict[str, Any]) -> None:
    summary = result["closures"]
    lines = [
        "# Mdot=5 angular-momentum ledger results",
        "",
        "Target: `Mdot_inner/Edd=5`, `Rout=335 rg`, `f_s=0.80`, `eta_E=98.125`, `N=164`.",
        "",
        "Sign convention: `Mdot` is inward-positive, `dMdot/dlnR = Mwind' - Mstream'`, and the net inward angular flux is `J=Mdot*l-G`. The conservative ledger is",
        "",
        "```text",
        "dJ/dlnR = Mwind' * l_w - Mstream' * l_s + tau_ext.",
        "```",
        "",
        "## Closure comparison",
        "",
        "| closure | phase point max | phase FV max | global point max | required torque correction | peak R (rg) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for closure in CLOSURES:
        row = summary[closure]
        lines.append(
            f"| `{closure}` | {row['phase_point_max']:.3e} | {row['phase_FV_max']:.3e} | "
            f"{row['global_point_max']:.3e} | {row['required_torque_correction_max']:.3e} | "
            f"{row['phase_point_peak_R_rg']:.3f} |"
        )
    current = summary["local_disk_prescribed"]
    representation = summary["representation"]
    required = summary["local_disk_required"]
    lines.extend(
        [
            "",
            "## Finding",
            "",
            "The exact `representation` closure assigns source and wind material the specific angular momentum carried by the algebraic net flux, `J/Mdot=l-G/Mdot`, and treats `Mdot*d(stream_l)/dlnR` as a separate torque. Its pointwise ledger closes algebraically.",
            "",
            "The previous provisional audit instead assigned both source and wind the full local disk `l` while retaining the same explicit torque. When `Mdot` varies, this omits the viscous-loading correction `-Mdot' G/Mdot`.",
            "",
            f"For the accepted phase branch, that provisional point defect reaches `{current['phase_point_max']:.3e}` and its FV defect reaches `{current['phase_FV_max']:.3e}`. The exact representation FV floor is `{representation['phase_FV_max']:.3e}`.",
            "",
            f"Allowing the external torque to absorb the missing local-disk loading term restores pointwise conservation; the required correction reaches `{required['required_torque_correction_max']:.3e}` in units of `Mdot_inner*lK` per `dlnR`.",
            "",
            "## Production decision",
            "",
            "- Keep `representation` as the exact audit of the current algebraic model.",
            "- Do not call it a physical stream closure: its carried `l_s=l_w=J/Mdot` is a representation identity.",
            "- For a physical model, specify `l_s(R)`, `l_w(R)`, and `tau_ext(R)` independently and promote the angular flux equation to production.",
            "- The independent outer-manifold search can classify the current mathematical closure, but physical flux matching must report the explicit closure used.",
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

    image = Image.new("RGB", (1450, 850), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    boxes = ((65, 55, 700, 790), (750, 55, 1385, 790))
    closures = (
        ("representation", "#176B87"),
        ("local_disk_prescribed", "#C0392B"),
        ("keplerian_injection_prescribed", "#7D3C98"),
    )

    def panel(box, rows, radius_key, value_key, title):
        left, top, right, bottom = box
        x0, x1, y0, y1 = left + 82, right - 22, top + 42, bottom - 52
        series = []
        all_x, all_y = [], []
        for closure, color in closures:
            xs = np.asarray([row[radius_key] for row in rows], dtype=float)
            ys = np.asarray(
                [abs(row["closures"][closure][value_key]) for row in rows], dtype=float
            )
            valid = np.isfinite(xs) & np.isfinite(ys) & (ys > 0.0)
            xs, ys = xs[valid], np.log10(ys[valid])
            series.append((xs, ys, color, closure))
            all_x.extend(xs.tolist())
            all_y.extend(ys.tolist())
        xmin, xmax = min(all_x), max(all_x)
        ymin, ymax = min(all_y), max(all_y)
        dx, dy = max(xmax - xmin, 1.0e-12), max(ymax - ymin, 1.0e-12)
        xmin, xmax = xmin - 0.03 * dx, xmax + 0.03 * dx
        ymin, ymax = ymin - 0.06 * dy, ymax + 0.06 * dy

        def xy(x, y):
            return (
                x0 + (x - xmin) / (xmax - xmin) * (x1 - x0),
                y1 - (y - ymin) / (ymax - ymin) * (y1 - y0),
            )

        draw.rectangle(box, outline="#D5D8DC")
        draw.text((left + 8, top + 8), title, fill="#17202A", font=font)
        for tick in range(5):
            xv = xmin + tick * (xmax - xmin) / 4.0
            yv = ymin + tick * (ymax - ymin) / 4.0
            px, _ = xy(xv, ymin)
            _, py = xy(xmin, yv)
            draw.line((px, y0, px, y1), fill="#ECEFF1")
            draw.line((x0, py, x1, py), fill="#ECEFF1")
            draw.text((px - 18, y1 + 8), f"{xv:.4g}", fill="#34495E", font=font)
            draw.text((left + 4, py - 6), f"10^{yv:.1f}", fill="#34495E", font=font)
        draw.line((x0, y1, x1, y1), fill="#2C3E50", width=2)
        draw.line((x0, y0, x0, y1), fill="#2C3E50", width=2)
        draw.text(((x0 + x1) / 2 - 28, bottom - 18), "R / rg", fill="#34495E", font=font)
        legend_y = top + 8
        for xs, ys, color, label in series:
            points = [xy(float(x), float(y)) for x, y in zip(xs, ys)]
            if len(points) > 1:
                draw.line(points, fill=color, width=3)
            draw.line((right - 210, legend_y + 5, right - 185, legend_y + 5), fill=color, width=3)
            draw.text((right - 180, legend_y), label, fill="#34495E", font=font)
            legend_y += 16

    panel(
        boxes[0],
        result["phase"]["intervals"],
        "R_mid_rg",
        "FV_residual_scaled",
        "Phase finite-volume angular residual",
    )
    panel(
        boxes[1],
        result["global_points"],
        "R_rg",
        "residual_scaled",
        "Global pointwise angular ledger residual",
    )
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    image.save(FIGURE_PATH)


def main() -> None:
    x_log, params, _context, _aux, phase_seed = classification.global_phase._load_problem()
    lambda0 = float(phase_seed["lambda0"])
    phase = _phase_audit(params, lambda0)
    global_rows = _global_audit(x_log, params)
    closures = _summarize(phase, global_rows)
    result = {
        "target": {
            "eta_E": 98.125,
            "N": int(params.n_nodes),
            "Rout_rg": float(params.R_out_rg),
        },
        "sign_convention": {
            "mass": "dMdot/dlnR = wind_prime - stream_prime",
            "angular": "d(Mdot*l-G)/dlnR = wind_prime*l_w - stream_prime*l_s + tau_ext",
        },
        "closures": closures,
        "decision": {
            "representation_identity_certified": bool(
                closures["representation"]["phase_point_max"] <= 1.0e-12
            ),
            "previous_local_disk_audit_conservative": bool(
                closures["local_disk_prescribed"]["phase_point_max"] <= 3.0e-5
            ),
            "physical_stream_closure_specified": False,
            "outer_search_can_use_representation_for_topology": True,
            "physical_matching_requires_explicit_closure": True,
        },
    }
    profiles = {**result, "phase": phase, "global_points": global_rows}
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n")
    PROFILE_PATH.write_text(json.dumps(_jsonable(profiles), indent=2, sort_keys=True) + "\n")
    _write_note(profiles)
    _write_figure(profiles)
    print(json.dumps(_jsonable(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
