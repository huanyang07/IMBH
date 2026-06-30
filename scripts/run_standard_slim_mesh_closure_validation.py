"""Mesh and outer-closure validation for high-side standard slim checkpoints."""

from __future__ import annotations

import json
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    TransonicSlimParams,
    collocation_residual,
    pressure_supported_omega_target,
    remap_profile_to_new_sonic_grid,
    residual_audit_from_state_vector,
    solve_square_transonic_polish,
    transonic_profile_from_state_vector,
    unpack_state,
)
from imri_qpe.layer3_minidisk_1d.transonic_collocation import _differential_interval_residual_from_unpacked
from imri_qpe.layer3_minidisk_1d.transonic_local import local_gradient
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot
from run_standard_slim_adaptive_mdot_ladder import STRESS_FACTOR
from run_standard_slim_analytic_seed_audit import ALPHA, fmt, json_safe
from run_standard_slim_mdot_injection_ladder import dominant


ROOT = Path(__file__).resolve().parents[1]
TABLE_OUTPUT = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_MESH_CLOSURE_TABLE",
    "outputs/tables/slim_benchmark_mesh_closure_validation.md",
)
JSON_OUTPUT = TABLE_OUTPUT.with_suffix(".json")
FIGURE_OUTPUT = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_MESH_CLOSURE_FIGURE",
    "outputs/figures/slim_benchmark_mesh_closure_validation.png",
)
CHECKPOINT_DIR = ROOT / os.environ.get(
    "IMBH_STANDARD_SLIM_MESH_CLOSURE_CHECKPOINTS",
    "outputs/checkpoints/slim_benchmark_mesh_closure_validation",
)

DEFAULT_CASES = (
    "anchor_7p73e3:outputs/checkpoints/slim_benchmark_adaptive_mdot_ladder_newton_refresh_1e4_1e2/up_mdot_0p00772853.npz,"
    "above_8p37e3:outputs/checkpoints/slim_benchmark_adaptive_mdot_ladder_newton_refresh_1e4_1e2/up_mdot_0p00837222.npz,"
    "endpoint_1e2:outputs/checkpoints/slim_benchmark_adaptive_mdot_ladder_newton_refresh_1e4_1e2/up_mdot_0p01.npz"
)
CASE_SPECS = tuple(
    piece.strip()
    for piece in os.environ.get("IMBH_STANDARD_SLIM_MESH_CLOSURE_CASES", DEFAULT_CASES).replace(";", ",").split(",")
    if piece.strip()
)
N_VALUES = tuple(
    int(piece)
    for piece in os.environ.get("IMBH_STANDARD_SLIM_MESH_CLOSURE_N_VALUES", "128,160").replace(":", ",").split(",")
    if piece.strip()
)
CLOSURE_SPECS = tuple(
    piece.strip()
    for piece in os.environ.get(
        "IMBH_STANDARD_SLIM_MESH_CLOSURE_CLOSURES",
        "thin_value,pressure_one_sided",
    )
    .replace(":", ",")
    .split(",")
    if piece.strip()
)
PIVOTS = tuple(
    piece.strip()
    for piece in os.environ.get("IMBH_STANDARD_SLIM_MESH_CLOSURE_PIVOTS", "C2,C1").replace(":", ",").split(",")
    if piece.strip()
)
NEWTON_MAX_ITER = int(os.environ.get("IMBH_STANDARD_SLIM_MESH_CLOSURE_NEWTON_MAX_ITER", "24"))
NEWTON_MAX_NFEV = int(os.environ.get("IMBH_STANDARD_SLIM_MESH_CLOSURE_NEWTON_MAX_NFEV", "1200"))
NEWTON_MAX_STEP_NORM = float(os.environ.get("IMBH_STANDARD_SLIM_MESH_CLOSURE_NEWTON_MAX_STEP_NORM", "0.25"))
NEWTON_LINEAR_SOLVER = os.environ.get("IMBH_STANDARD_SLIM_MESH_CLOSURE_NEWTON_LINEAR_SOLVER", "regularized_lsmr")
LSQ_FALLBACK_NFEV = int(os.environ.get("IMBH_STANDARD_SLIM_MESH_CLOSURE_LSQ_FALLBACK_NFEV", "0"))
ACCEPTANCE_TOL = float(os.environ.get("IMBH_STANDARD_SLIM_MESH_CLOSURE_ACCEPTANCE_TOL", "1e-5"))
ANCHOR_TOL = float(os.environ.get("IMBH_STANDARD_SLIM_MESH_CLOSURE_ANCHOR_TOL", "3e-6"))


def parse_case_specs() -> list[tuple[str, Path]]:
    cases = []
    for spec in CASE_SPECS:
        if ":" not in spec:
            raise ValueError(f"case spec must be label:path, got {spec!r}")
        label, path = spec.split(":", 1)
        cases.append((label.strip(), ROOT / path.strip()))
    return cases


def params_for(
    fiducial: FiducialParams,
    mdot_edd: float,
    ratio: float,
    R_out_rg: float,
    n_nodes: int,
    *,
    outer_closure: str = "thin_value",
    outer_match_log_slopes: tuple[float, float] | None = None,
) -> TransonicSlimParams:
    return TransonicSlimParams(
        M2_g=fiducial.M2_g,
        Mdot_g_s=float(ratio) * mdot_edd,
        alpha=ALPHA,
        mu_stress=0.0,
        stress_factor=STRESS_FACTOR,
        R_out_rg=float(R_out_rg),
        n_nodes=int(n_nodes),
        max_nfev=max(NEWTON_MAX_NFEV, LSQ_FALLBACK_NFEV, 1),
        residual_tol=1.0e-8,
        outer_closure=outer_closure,
        outer_match_log_slopes=outer_match_log_slopes,
        interval_residual_form="differential",
        integrated_residual_weighting="none",
    )


def load_checkpoint(path: Path, fiducial: FiducialParams, mdot_edd: float) -> tuple[np.ndarray, TransonicSlimParams]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    z = np.asarray(data["z"], dtype=float)
    params = params_for(fiducial, mdot_edd, float(data["ratio"]), float(data["R_out_rg"]), int(data["n_nodes"]))
    return z, params


def max_residual(z: np.ndarray, params: TransonicSlimParams) -> float:
    return float(np.max(np.abs(collocation_residual(z, params))))


def one_sided_outer_slopes(z: np.ndarray, params: TransonicSlimParams) -> tuple[float, float]:
    logu, logT, _logR_son, _lambda0, logR = unpack_state(z, params)
    dx = float(logR[-1] - logR[-2])
    return float((logu[-1] - logu[-2]) / dx), float((logT[-1] - logT[-2]) / dx)


def polyfit_outer_slopes(z: np.ndarray, params: TransonicSlimParams, n_fit: int = 8) -> tuple[float, float]:
    logu, logT, _logR_son, _lambda0, logR = unpack_state(z, params)
    count = min(int(n_fit), len(logR))
    x = logR[-count:] - logR[-1]
    gu = float(np.polyder(np.poly1d(np.polyfit(x, logu[-count:], min(2, count - 1))))(0.0))
    gT = float(np.polyder(np.poly1d(np.polyfit(x, logT[-count:], min(2, count - 1))))(0.0))
    return gu, gT


def local_ode_outer_slopes(z: np.ndarray, params: TransonicSlimParams) -> tuple[float, float]:
    logu, logT, _logR_son, lambda0, logR = unpack_state(z, params)
    gradient = local_gradient(logR[-1], np.array([logu[-1], logT[-1]], dtype=float), lambda0, params)
    return float(gradient[0]), float(gradient[1])


def closure_config(name: str, source_z: np.ndarray, source_params: TransonicSlimParams) -> tuple[str, tuple[float, float] | None, str]:
    if name == "thin_value":
        return "thin_value", None, "-"
    if name == "pressure_one_sided":
        return "pressure_supported_thin_energy", one_sided_outer_slopes(source_z, source_params), "one_sided"
    if name == "pressure_polyfit":
        return "pressure_supported_thin_energy", polyfit_outer_slopes(source_z, source_params), "polyfit"
    if name == "pressure_local_ode":
        return "pressure_supported_thin_energy", local_ode_outer_slopes(source_z, source_params), "local_ode"
    raise ValueError(f"unknown closure spec {name!r}")


def remap_seed(source_z: np.ndarray, source_params: TransonicSlimParams, target_params: TransonicSlimParams) -> np.ndarray:
    if source_params.n_nodes == target_params.n_nodes and np.isclose(source_params.R_out_rg, target_params.R_out_rg):
        return np.array(source_z, copy=True)
    profile = transonic_profile_from_state_vector(source_z, source_params)
    return remap_profile_to_new_sonic_grid(profile, target_params, temperature_mdot_power=0.0)


def interval_peak(z: np.ndarray, params: TransonicSlimParams) -> tuple[int, float, float, float]:
    logu, logT, _logR_son, lambda0, logR = unpack_state(z, params)
    intervals = np.asarray(
        [
            _differential_interval_residual_from_unpacked(logu, logT, logR, lambda0, params, idx)
            for idx in range(len(logR) - 1)
        ],
        dtype=float,
    )
    idx = int(np.argmax(np.abs(intervals[:, 0])))
    R_mid_rg = float(np.exp(0.5 * (logR[idx] + logR[idx + 1])) / params.r_g)
    return idx, R_mid_rg, float(intervals[idx, 0]), float(intervals[idx, 1])


def outer_pressure_diagnostic(z: np.ndarray, params: TransonicSlimParams) -> dict[str, float]:
    logu, logT, _logR_son, lambda0, logR = unpack_state(z, params)
    slopes = one_sided_outer_slopes(z, params)
    y = np.array([logu[-1], logT[-1]], dtype=float)
    profile = transonic_profile_from_state_vector(z, params)
    ln_omega = float(np.log(profile.Omega[-1] / profile.Omega_K[-1]))
    target = pressure_supported_omega_target(float(logR[-1]), y, np.asarray(slopes, dtype=float), lambda0, params)
    return {
        "outer_lnOmega_OmegaK": ln_omega,
        "outer_pressure_target": float(target),
        "outer_pressure_residual": float(ln_omega - target),
        "outer_logu_slope": float(slopes[0]),
        "outer_logT_slope": float(slopes[1]),
    }


def polish_best(z0: np.ndarray, params: TransonicSlimParams):
    best = None
    best_full = np.inf
    for pivot in PIVOTS:
        result = solve_square_transonic_polish(
            params,
            z0,
            pivot=pivot,
            method="newton",
            max_iter=NEWTON_MAX_ITER,
            max_nfev=NEWTON_MAX_NFEV,
            residual_tol=1.0e-8,
            use_block_jacobian=True,
            linear_solver=NEWTON_LINEAR_SOLVER,
            max_step_norm=NEWTON_MAX_STEP_NORM,
        )
        full = max_residual(result.z, params)
        if full < best_full:
            best = result
            best_full = full
        if full <= ANCHOR_TOL:
            break
    if best is None:
        raise RuntimeError("no polish pivots configured")
    if best_full <= ACCEPTANCE_TOL or LSQ_FALLBACK_NFEV <= 0:
        return best
    lsq = solve_square_transonic_polish(
        params,
        best.z,
        pivot=best.pivot,
        method="least_squares",
        max_nfev=LSQ_FALLBACK_NFEV,
        residual_tol=1.0e-8,
        use_block_jacobian=False,
    )
    return lsq if max_residual(lsq.z, params) < best_full else best


def row_for_result(
    *,
    label: str,
    source_path: Path,
    source_params: TransonicSlimParams,
    target_params: TransonicSlimParams,
    closure_spec: str,
    slope_source: str,
    slopes: tuple[float, float] | None,
    initial_z: np.ndarray,
    polish,
    elapsed_s: float,
) -> dict[str, object]:
    z = np.asarray(polish.z, dtype=float)
    audit = residual_audit_from_state_vector(z, target_params)
    profile = transonic_profile_from_state_vector(z, target_params)
    legacy_params = replace(target_params, outer_closure="thin_value", outer_match_log_slopes=None)
    legacy_audit = residual_audit_from_state_vector(z, legacy_params)
    peak_idx, peak_R, peak_R_value, peak_E_value = interval_peak(z, target_params)
    initial_full = max_residual(initial_z, target_params)
    final_full = max_residual(z, target_params)
    return {
        "case": label,
        "source_path": str(source_path.relative_to(ROOT)),
        "ratio": float(target_params.mdot_edd_ratio),
        "R_out_rg": float(target_params.R_out_rg),
        "source_N": int(source_params.n_nodes),
        "N": int(target_params.n_nodes),
        "closure_spec": closure_spec,
        "outer_closure": target_params.outer_closure,
        "slope_source": slope_source,
        "g_u": np.nan if slopes is None else float(slopes[0]),
        "g_T": np.nan if slopes is None else float(slopes[1]),
        "initial_full": float(initial_full),
        "final_full": float(final_full),
        "accepted": bool(final_full <= ACCEPTANCE_TOL),
        "anchor_eligible": bool(final_full <= ANCHOR_TOL),
        "dominant": dominant(audit),
        "interval_R": float(audit.interval_radial_max),
        "interval_E": float(audit.interval_energy_max),
        "outer_omega": float(audit.outer_omega),
        "outer_energy": float(audit.outer_energy),
        "legacy_outer_omega": float(legacy_audit.outer_omega),
        "legacy_outer_energy": float(legacy_audit.outer_energy),
        "peak_interval_R_index": int(peak_idx),
        "peak_interval_R_rg": float(peak_R),
        "peak_interval_R": float(peak_R_value),
        "peak_interval_E": float(peak_E_value),
        "Rson_rg": float(profile.sonic_radius / target_params.r_g),
        "lambda0_over_lK_isco": float(audit.lambda0_over_lK_isco),
        "max_H_R": float(np.max(profile.H_over_R)),
        "integrated_adv": float(profile.integrated_advective_fraction),
        "outer_H_R": float(audit.outer_H_over_R),
        "outer_Qadv_Qvisc": float(audit.outer_Qadv_over_Qvisc),
        "pivot": str(polish.pivot),
        "method": str(polish.method),
        "nfev": int(polish.result.nfev),
        "iterations": int(polish.iterations),
        "line_search_reductions": int(polish.line_search_reductions),
        "optimizer_success": bool(polish.result.optimizer_success),
        "elapsed_s": float(elapsed_s),
        "message": str(polish.result.message),
        **outer_pressure_diagnostic(z, target_params),
        "z": z,
    }


def save_checkpoint(row: dict[str, object], params: TransonicSlimParams) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    safe_case = str(row["case"]).replace(".", "p").replace("/", "_")
    safe_closure = str(row["closure_spec"]).replace(".", "p").replace("/", "_")
    stem = f"{safe_case}_N{int(row['N'])}_{safe_closure}_mdot_{float(row['ratio']):.8g}".replace(".", "p")
    payload = {key: value for key, value in row.items() if key != "z"}
    np.savez_compressed(
        CHECKPOINT_DIR / f"{stem}.npz",
        z=np.asarray(row["z"], dtype=float),
        ratio=np.array(row["ratio"]),
        R_out_rg=np.array(row["R_out_rg"]),
        n_nodes=np.array(row["N"]),
        outer_closure=np.array(params.outer_closure),
        row_json=np.array(json.dumps(json_safe(payload), sort_keys=True)),
    )


def write_table(rows: list[dict[str, object]]) -> None:
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Standard Slim Mesh/Closure Validation",
        "",
        "Generated by `scripts/run_standard_slim_mesh_closure_validation.py`.",
        "",
        f"Config: N values `{','.join(str(value) for value in N_VALUES)}`, closures `{','.join(CLOSURE_SPECS)}`, "
        f"pivots `{','.join(PIVOTS)}`, Newton solver `{NEWTON_LINEAR_SOLVER}`, Newton max iter `{NEWTON_MAX_ITER}`, "
        f"LSQ fallback nfev `{LSQ_FALLBACK_NFEV}`.",
        "",
        "| case | Mdot/Edd | N | closure | slope | initial full | final full | accepted | anchor | dominant | interval R | outer omega | legacy outer omega | peak R/rg | peak int R | pressure mismatch | max H/R | int adv | pivot | nfev | elapsed s | message |",
        "|---|---:|---:|---|---|---:|---:|:---:|:---:|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---|",
    ]
    for row in rows:
        formatted = {key: fmt(value) if isinstance(value, (float, int, np.floating, np.integer)) else value for key, value in row.items()}
        lines.append(
            "| {case} | {ratio} | {N} | {closure_spec} | {slope_source} | {initial_full} | {final_full} | "
            "{accepted} | {anchor_eligible} | {dominant} | {interval_R} | {outer_omega} | {legacy_outer_omega} | "
            "{peak_interval_R_rg} | {peak_interval_R} | {outer_pressure_residual} | {max_H_R} | {integrated_adv} | "
            "{pivot} | {nfev} | {elapsed_s} | {message} |".format(**formatted).replace("\n", " ")
        )
    TABLE_OUTPUT.write_text("\n".join(lines) + "\n")
    JSON_OUTPUT.write_text(json.dumps(json_safe([{key: value for key, value in row.items() if key != "z"} for row in rows]), indent=2, sort_keys=True) + "\n")


def write_figure(rows: list[dict[str, object]]) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:
        print(f"skipping figure: PIL unavailable ({exc})", flush=True)
        return
    if not rows:
        return
    width, height = 1100, 680
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    x0, y0, x1, y1 = 90, 90, 1030, 560
    draw.rectangle((x0, y0, x1, y1), outline=(60, 60, 60), width=2)
    values = [float(row["final_full"]) for row in rows if float(row["final_full"]) > 0.0]
    y_log = np.log10(np.maximum(np.asarray(values, dtype=float), 1.0e-16))
    y_min, y_max = float(np.floor(np.min(y_log))), float(np.ceil(np.max(y_log)))
    if y_max <= y_min:
        y_max = y_min + 1.0
    cases = list(dict.fromkeys(str(row["case"]) for row in rows))
    closures = list(dict.fromkeys(str(row["closure_spec"]) for row in rows))
    colors = {
        "thin_value": (31, 119, 180),
        "pressure_one_sided": (214, 39, 40),
        "pressure_polyfit": (44, 160, 44),
        "pressure_local_ode": (148, 103, 189),
    }
    n_min, n_max = min(N_VALUES), max(N_VALUES)
    if n_max == n_min:
        n_max = n_min + 1
    panel_width = (x1 - x0) / max(len(cases), 1)
    for case_idx, case in enumerate(cases):
        px0 = x0 + panel_width * case_idx
        px1 = px0 + panel_width
        draw.line((int(px0), y0, int(px0), y1), fill=(220, 220, 220), width=1)
        draw.text((int(px0) + 8, y0 - 24), case, fill=(20, 20, 20), font=font)
        for closure in closures:
            selected = sorted(
                [row for row in rows if row["case"] == case and row["closure_spec"] == closure],
                key=lambda row: int(row["N"]),
            )
            points = []
            for row in selected:
                xx = (int(row["N"]) - n_min) / (n_max - n_min)
                yy = np.log10(max(float(row["final_full"]), 1.0e-16))
                px = int(px0 + 20 + xx * max(panel_width - 40, 1))
                py = y1 - int((yy - y_min) / (y_max - y_min) * (y1 - y0))
                points.append((px, py))
            color = colors.get(closure, (20, 20, 20))
            if len(points) >= 2:
                draw.line(points, fill=color, width=3)
            for point in points:
                draw.ellipse((point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4), fill=color)
    for tol, label in ((ACCEPTANCE_TOL, "accept"), (ANCHOR_TOL, "anchor")):
        yy = np.log10(tol)
        py = y1 - int((yy - y_min) / (y_max - y_min) * (y1 - y0))
        draw.line((x0, py, x1, py), fill=(120, 120, 120), width=1)
        draw.text((x1 - 72, py - 14), label, fill=(80, 80, 80), font=font)
    for idx, closure in enumerate(closures):
        y = 25 + 18 * idx
        color = colors.get(closure, (20, 20, 20))
        draw.line((90, y + 6, 130, y + 6), fill=color, width=3)
        draw.text((140, y), closure, fill=(20, 20, 20), font=font)
    draw.text((650, 25), "Final residual vs N for high-side checkpoints", fill=(20, 20, 20), font=font)
    draw.text((x0 + 4, y0 + 4), f"1e{int(y_max)}", fill=(80, 80, 80), font=font)
    draw.text((x0 + 4, y1 - 18), f"1e{int(y_min)}", fill=(80, 80, 80), font=font)
    FIGURE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(FIGURE_OUTPUT)


def main() -> None:
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)
    rows: list[dict[str, object]] = []
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    for label, path in parse_case_specs():
        source_z, source_params = load_checkpoint(path, fiducial, mdot_edd)
        for closure_spec in CLOSURE_SPECS:
            outer_closure, slopes, slope_source = closure_config(closure_spec, source_z, source_params)
            for n_nodes in N_VALUES:
                target_params = params_for(
                    fiducial,
                    mdot_edd,
                    source_params.mdot_edd_ratio,
                    source_params.R_out_rg,
                    n_nodes,
                    outer_closure=outer_closure,
                    outer_match_log_slopes=slopes,
                )
                seed = remap_seed(source_z, source_params, target_params)
                print(f"{label} closure={closure_spec} N={n_nodes} initial={max_residual(seed, target_params):.3e}", flush=True)
                t0 = time.perf_counter()
                polish = polish_best(seed, target_params)
                elapsed = time.perf_counter() - t0
                row = row_for_result(
                    label=label,
                    source_path=path,
                    source_params=source_params,
                    target_params=target_params,
                    closure_spec=closure_spec,
                    slope_source=slope_source,
                    slopes=slopes,
                    initial_z=seed,
                    polish=polish,
                    elapsed_s=elapsed,
                )
                rows.append(row)
                save_checkpoint(row, target_params)
                write_table(rows)
                write_figure(rows)
                print(
                    f"  final={row['final_full']:.3e} dom={row['dominant']} peak_R={row['peak_interval_R_rg']:.4g} "
                    f"accepted={row['accepted']} anchor={row['anchor_eligible']}",
                    flush=True,
                )
    write_table(rows)
    write_figure(rows)
    print(f"wrote {TABLE_OUTPUT}", flush=True)
    print(f"wrote {FIGURE_OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
