"""Search for true B=[c A] rank-defect points near the radial fold."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from imri_qpe.layer3_minidisk_1d.transonic_local import (
    B_rank_minors,
    algebraic_state,
    extended_phase_space_matrix,
    phase_space_null_tangent,
    sonic_diagnostics,
)

from run_transonic_barrier_critical_probe import load_seed_rows, solve_probe
from run_transonic_desingularized_barrier_flow import json_safe
from run_transonic_sonic_reverse_fit import load_context


ROOT = Path(__file__).resolve().parents[1]
TABLE_OUTPUT = ROOT / os.environ.get(
    "IMBH_B_RANK_SEARCH_TABLE",
    "outputs/tables/transonic_B_rank_defect_search.md",
)
JSON_OUTPUT = TABLE_OUTPUT.with_name(TABLE_OUTPUT.stem + ".json")
FIGURE_OUTPUT = ROOT / os.environ.get(
    "IMBH_B_RANK_SEARCH_FIGURE",
    "outputs/figures/transonic_B_rank_defect_search.png",
)

FOLD_JSON = ROOT / "outputs/tables/transonic_desingularized_fold_refinement.json"
TRACE_JSON = ROOT / "outputs/tables/transonic_desingularized_barrier_flow_ds5e4_trace.json"
LOGR_SCALE = float(os.environ.get("IMBH_B_RANK_ACCESS_LOGR_SCALE", "0.01"))
Y_SCALE = float(os.environ.get("IMBH_B_RANK_ACCESS_Y_SCALE", "0.05"))
LAMBDA_SCALE = float(os.environ.get("IMBH_B_RANK_ACCESS_LAMBDA_SCALE", "0.001"))
RANK_SB_ACCEPT = float(os.environ.get("IMBH_B_RANK_SB_ACCEPT", "1e-8"))
RANK_MINOR_ACCEPT = float(os.environ.get("IMBH_B_RANK_MINOR_ACCEPT", "1e-8"))
ACCESS_ACCEPT = float(os.environ.get("IMBH_B_RANK_ACCESS_ACCEPT", "1.0"))
MAX_NFEV = int(os.environ.get("IMBH_B_RANK_MAX_NFEV", "800"))
LOGR_HALF_WIDTH = float(os.environ.get("IMBH_B_RANK_LOGR_HALF_WIDTH", "0.035"))
LOGU_HALF_WIDTH = float(os.environ.get("IMBH_B_RANK_LOGU_HALF_WIDTH", "0.35"))
LOGT_HALF_WIDTH = float(os.environ.get("IMBH_B_RANK_LOGT_HALF_WIDTH", "0.45"))
LAMBDA_HALF_WIDTH = float(os.environ.get("IMBH_B_RANK_LAMBDA_HALF_WIDTH", "0.004"))


def fixed(value: object, digits: int) -> str:
    number = float(value)
    return f"{number:.{digits}f}" if np.isfinite(number) else "nan"


def sci(value: object, digits: int = 3) -> str:
    number = float(value)
    return f"{number:.{digits}e}" if np.isfinite(number) else "nan"


def rank_residual_vector(logR: float, y: np.ndarray, lambda0: float, params) -> np.ndarray:
    B, _A, _c = extended_phase_space_matrix(logR, y, lambda0, params)
    denom = np.linalg.norm(B[0]) * np.linalg.norm(B[1]) + 1.0e-300
    return B_rank_minors(logR, y, lambda0, params) / denom


def rank_metrics(logR: float, y: np.ndarray, lambda0: float, params) -> dict[str, object]:
    B, _A, _c = extended_phase_space_matrix(logR, y, lambda0, params)
    singular_values = np.linalg.svd(B, compute_uv=False)
    residual = rank_residual_vector(logR, y, lambda0, params)
    tangent = phase_space_null_tangent(logR, y, lambda0, params)
    sonic = sonic_diagnostics(logR, y, lambda0, params)
    state = algebraic_state(logR, float(y[0]), float(y[1]), lambda0, params)
    return {
        "R_rg": np.nan,
        "logR": float(logR),
        "logu": float(y[0]),
        "logT": float(y[1]),
        "lambda0": float(lambda0),
        "smin_over_smax_B": float(singular_values[-1] / (singular_values[0] + 1.0e-300)),
        "B_singular_values": np.asarray(singular_values, dtype=float),
        "rel_minor_norm": float(np.linalg.norm(residual)),
        "rel_minor_max": float(np.max(np.abs(residual))),
        "Bp_max": float(np.max(np.abs(tangent.residual))),
        "p_x": float(tangent.px),
        "D": float(sonic.D),
        "C1": float(sonic.C1),
        "C2": float(sonic.C2),
        "K": float(sonic.compatibility),
        "H_R": float(state.H_over_R),
        "Omega_over_K": float(state.Omega / state.Omega_K),
    }


def load_fold_center() -> dict[str, object]:
    rows = json.loads(FOLD_JSON.read_text())
    valid = [row for row in rows if row.get("metric") == "euclidean" and row.get("status") == "fold_found"]
    if not valid:
        raise RuntimeError(f"missing fold row in {FOLD_JSON}")
    best = min(valid, key=lambda row: float(row["ds"]))
    return {
        "label": "incoming_fold_refined",
        "logR": float(best["logR_fold"]),
        "logu": float(best["logu_fold"]),
        "logT": float(best["logT_fold"]),
        "lambda0": float(best["lambda0"]),
    }


def critical_centers(ctx) -> list[dict[str, object]]:
    seeds = [
        seed
        for seed in load_seed_rows()
        if seed.get("case") == "free_reverse_fit" and int(seed.get("branch", -1)) == 1
    ]
    if not seeds:
        raise RuntimeError("missing free_reverse_fit branch=1 seed")
    seed = seeds[0]
    rows: list[dict[str, object]] = []
    for free_lambda in (False, True):
        probe = solve_probe(ctx, seed, free_lambda=free_lambda)
        rows.append(
            {
                "label": "free_lambda_critical" if free_lambda else "fixed_lambda_critical",
                "logR": float(np.log(float(probe["R_rg"]) * ctx.params.r_g)),
                "logu": float(probe["logu"]),
                "logT": float(probe["logT"]),
                "lambda0": float(probe["lambda0"]),
            }
        )
    return rows


def incoming_trace(ctx) -> list[dict[str, float]]:
    _ = ctx
    if not TRACE_JSON.exists():
        return []
    rows = json.loads(TRACE_JSON.read_text())
    return [
        {
            "logR": float(row["logR"]),
            "logu": float(row["logu"]),
            "logT": float(row["logT"]),
            "lambda0": float(row["lambda0"]),
            "R_rg": float(row["R_rg"]),
        }
        for row in rows
        if row.get("label") == "incoming_R6_plus"
        and np.isfinite(float(row.get("logR", np.nan)))
    ]


def accessibility(logR: float, y: np.ndarray, lambda0: float, trace_rows: list[dict[str, float]]) -> dict[str, object]:
    if not trace_rows:
        return {
            "access_distance_z": np.nan,
            "access_distance_with_lambda": np.nan,
            "access_nearest_R_rg": np.nan,
            "access_dlogR": np.nan,
            "access_dlogu": np.nan,
            "access_dlogT": np.nan,
            "access_dlambda": np.nan,
            "accessible_current_branch": False,
        }
    best: tuple[float, float, dict[str, float], np.ndarray] | None = None
    for row in trace_rows:
        delta = np.array(
            [
                (float(row["logR"]) - logR) / LOGR_SCALE,
                (float(row["logu"]) - float(y[0])) / Y_SCALE,
                (float(row["logT"]) - float(y[1])) / Y_SCALE,
            ],
            dtype=float,
        )
        distance_z = float(np.linalg.norm(delta))
        distance_lambda = float(
            np.sqrt(distance_z**2 + ((float(row["lambda0"]) - lambda0) / LAMBDA_SCALE) ** 2)
        )
        if best is None or distance_lambda < best[1]:
            best = (distance_z, distance_lambda, row, delta)
    assert best is not None
    distance_z, distance_lambda, row, _delta = best
    return {
        "access_distance_z": distance_z,
        "access_distance_with_lambda": distance_lambda,
        "access_nearest_R_rg": float(row["R_rg"]),
        "access_dlogR": float(row["logR"] - logR),
        "access_dlogu": float(row["logu"] - float(y[0])),
        "access_dlogT": float(row["logT"] - float(y[1])),
        "access_dlambda": float(row["lambda0"] - lambda0),
        "accessible_current_branch": bool(distance_lambda < ACCESS_ACCEPT),
    }


def make_row(
    source: str,
    search_mode: str,
    center: dict[str, object],
    logR: float,
    y: np.ndarray,
    lambda0: float,
    ctx,
    trace_rows: list[dict[str, float]],
    result=None,
) -> dict[str, object]:
    metrics = rank_metrics(logR, y, lambda0, ctx.params.physics)
    metrics["R_rg"] = float(np.exp(logR) / ctx.params.r_g)
    metrics.update(accessibility(logR, y, lambda0, trace_rows))
    metrics.update(
        {
            "source": source,
            "search_mode": search_mode,
            "center_R_rg": float(np.exp(float(center["logR"])) / ctx.params.r_g),
            "center_lambda0": float(center["lambda0"]),
            "delta_logR": float(logR - float(center["logR"])),
            "delta_logu": float(y[0] - float(center["logu"])),
            "delta_logT": float(y[1] - float(center["logT"])),
            "delta_lambda0": float(lambda0 - float(center["lambda0"])),
            "rank_defect": bool(
                float(metrics["smin_over_smax_B"]) < RANK_SB_ACCEPT
                and float(metrics["rel_minor_norm"]) < RANK_MINOR_ACCEPT
            ),
        }
    )
    if result is not None:
        metrics.update(
            {
                "optimizer_success": bool(result.success),
                "optimizer_cost": float(result.cost),
                "optimizer_nfev": int(result.nfev),
                "optimizer_message": str(result.message),
            }
        )
    return metrics


def optimize_center(
    source: str,
    center: dict[str, object],
    mode: str,
    ctx,
    trace_rows: list[dict[str, float]],
) -> dict[str, object]:
    logR0 = float(center["logR"])
    y0 = np.array([float(center["logu"]), float(center["logT"])], dtype=float)
    lambda0 = float(center["lambda0"])
    if mode == "fixed_lambda_minors":
        x0 = np.array([logR0, y0[0], y0[1]], dtype=float)
        lower = np.array([logR0 - LOGR_HALF_WIDTH, y0[0] - LOGU_HALF_WIDTH, y0[1] - LOGT_HALF_WIDTH], dtype=float)
        upper = np.array([logR0 + LOGR_HALF_WIDTH, y0[0] + LOGU_HALF_WIDTH, y0[1] + LOGT_HALF_WIDTH], dtype=float)

        def residual(trial: np.ndarray) -> np.ndarray:
            try:
                return rank_residual_vector(float(trial[0]), np.asarray(trial[1:], dtype=float), lambda0, ctx.params.physics)
            except Exception:
                return np.full(3, 1.0e6)

        result = least_squares(
            residual,
            x0,
            bounds=(lower, upper),
            x_scale=np.array([0.02, 0.1, 0.1], dtype=float),
            ftol=1.0e-13,
            xtol=1.0e-13,
            gtol=1.0e-13,
            max_nfev=MAX_NFEV,
        )
        logR = float(result.x[0])
        y = np.asarray(result.x[1:], dtype=float)
        return make_row(source, mode, center, logR, y, lambda0, ctx, trace_rows, result)
    if mode == "free_lambda_minors":
        x0 = np.array([logR0, y0[0], y0[1], lambda0], dtype=float)
        lower = np.array(
            [logR0 - LOGR_HALF_WIDTH, y0[0] - LOGU_HALF_WIDTH, y0[1] - LOGT_HALF_WIDTH, lambda0 - LAMBDA_HALF_WIDTH],
            dtype=float,
        )
        upper = np.array(
            [logR0 + LOGR_HALF_WIDTH, y0[0] + LOGU_HALF_WIDTH, y0[1] + LOGT_HALF_WIDTH, lambda0 + LAMBDA_HALF_WIDTH],
            dtype=float,
        )

        def residual(trial: np.ndarray) -> np.ndarray:
            try:
                return rank_residual_vector(
                    float(trial[0]),
                    np.asarray(trial[1:3], dtype=float),
                    float(trial[3]),
                    ctx.params.physics,
                )
            except Exception:
                return np.full(3, 1.0e6)

        result = least_squares(
            residual,
            x0,
            bounds=(lower, upper),
            x_scale=np.array([0.02, 0.1, 0.1, 0.001], dtype=float),
            ftol=1.0e-13,
            xtol=1.0e-13,
            gtol=1.0e-13,
            max_nfev=MAX_NFEV,
        )
        return make_row(
            source,
            mode,
            center,
            float(result.x[0]),
            np.asarray(result.x[1:3], dtype=float),
            float(result.x[3]),
            ctx,
            trace_rows,
            result,
        )
    raise ValueError(f"unknown mode {mode!r}")


def write_table(rows: list[dict[str, object]]) -> None:
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# B Rank-Defect Search",
        "",
        "Generated by `scripts/run_transonic_B_rank_defect_search.py`.",
        "",
        f"Rank acceptance: `sB<{RANK_SB_ACCEPT:g}` and `relative minor norm<{RANK_MINOR_ACCEPT:g}`. "
        f"Accessibility threshold: scaled distance `<{ACCESS_ACCEPT:g}`.",
        "",
        "| source | mode | rank defect | accessible | R/rg | lambda0 | sB | rel minor | p_x | D | K | H/R | Omega/K | access d | nearest R | dlambda access | dlogR | dlogu | dlogT | opt cost | message |",
        "|---|---|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {source} | {mode} | {rank} | {access} | {R} | {lambda0} | {sB} | {minor} | {px} | {D} | {K} | "
            "{HR} | {omega} | {adist} | {nearR} | {dlambda} | {dlogR} | {dlogu} | {dlogT} | {cost} | {msg} |".format(
                source=row.get("source", ""),
                mode=row.get("search_mode", ""),
                rank="yes" if row.get("rank_defect") else "no",
                access="yes" if row.get("accessible_current_branch") else "no",
                R=fixed(row.get("R_rg", np.nan), 9),
                lambda0=fixed(row.get("lambda0", np.nan), 9),
                sB=sci(row.get("smin_over_smax_B", np.nan)),
                minor=sci(row.get("rel_minor_norm", np.nan)),
                px=sci(row.get("p_x", np.nan)),
                D=sci(row.get("D", np.nan)),
                K=sci(row.get("K", np.nan)),
                HR=sci(row.get("H_R", np.nan)),
                omega=fixed(row.get("Omega_over_K", np.nan), 6),
                adist=sci(row.get("access_distance_with_lambda", np.nan)),
                nearR=fixed(row.get("access_nearest_R_rg", np.nan), 6),
                dlambda=sci(row.get("access_dlambda", np.nan)),
                dlogR=sci(row.get("access_dlogR", np.nan)),
                dlogu=sci(row.get("access_dlogu", np.nan)),
                dlogT=sci(row.get("access_dlogT", np.nan)),
                cost=sci(row.get("optimizer_cost", np.nan)),
                msg=str(row.get("optimizer_message", "")).replace("|", "/")[:80],
            )
        )
    TABLE_OUTPUT.write_text("\n".join(lines) + "\n")


def write_json(rows: list[dict[str, object]]) -> None:
    def clean(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        return json_safe(value)

    JSON_OUTPUT.write_text(
        json.dumps([{key: clean(value) for key, value in row.items()} for row in rows], indent=2, sort_keys=True)
        + "\n"
    )


def write_figure(rows: list[dict[str, object]]) -> None:
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:
        print(f"figure skipped: {exc}", flush=True)
        return
    width, height = 1120, 660
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((70, 24), "B rank-defect search", fill=(20, 20, 20))
    panels = [
        ("smin/smax B", "smin_over_smax_B", True),
        ("relative minor norm", "rel_minor_norm", True),
        ("access distance incl. lambda", "access_distance_with_lambda", True),
    ]
    mode_colors = {
        "center_eval": (31, 119, 180),
        "fixed_lambda_minors": (214, 39, 40),
        "free_lambda_minors": (44, 160, 44),
    }
    plot_l, plot_r = 100, width - 50
    panel_h = 140
    x_values = np.asarray([float(row["R_rg"]) for row in rows if np.isfinite(float(row.get("R_rg", np.nan)))], dtype=float)
    if x_values.size == 0:
        return
    xmin, xmax = float(np.min(x_values)), float(np.max(x_values))
    for pidx, (title, key, logy) in enumerate(panels):
        top = 70 + 175 * pidx
        bottom = top + panel_h
        draw.rectangle((plot_l, top, plot_r, bottom), outline=(60, 60, 60), width=1)
        values = np.asarray([float(row.get(key, np.nan)) for row in rows], dtype=float)
        finite = values[np.isfinite(values) & ((values > 0.0) if logy else np.ones_like(values, dtype=bool))]
        if finite.size == 0:
            continue
        ymin = max(float(np.min(finite)) * 0.6, 1.0e-20) if logy else float(np.min(finite))
        ymax = float(np.max(finite)) * 1.5 if logy else float(np.max(finite))
        if not logy:
            pad = max(0.1 * (ymax - ymin), 1.0e-6)
            ymin -= pad
            ymax += pad

        def sx(value: float) -> float:
            return plot_l + (value - xmin) / (xmax - xmin + 1.0e-300) * (plot_r - plot_l)

        def sy(value: float) -> float:
            if logy:
                return bottom - (np.log10(value) - np.log10(ymin)) / (np.log10(ymax) - np.log10(ymin) + 1.0e-300) * (bottom - top)
            return bottom - (value - ymin) / (ymax - ymin + 1.0e-300) * (bottom - top)

        draw.text((plot_l + 8, top + 8), title, fill=(20, 20, 20))
        for midx, (mode, color) in enumerate(mode_colors.items()):
            points = []
            for row in rows:
                if row.get("search_mode") != mode:
                    continue
                x = float(row.get("R_rg", np.nan))
                value = float(row.get(key, np.nan))
                if np.isfinite(x) and np.isfinite(value) and ((value > 0.0) if logy else True):
                    points.append((sx(x), sy(value)))
            for point in points:
                draw.ellipse((point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4), fill=color)
            if pidx == 0:
                draw.text((plot_l + 12, top + 24 + 16 * midx), mode, fill=color)
    draw.text((plot_l, height - 26), "R/rg", fill=(20, 20, 20))
    FIGURE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(FIGURE_OUTPUT)


def main() -> None:
    ctx = load_context()
    centers = [load_fold_center(), *critical_centers(ctx)]
    trace_rows = incoming_trace(ctx)
    rows: list[dict[str, object]] = []
    for center in centers:
        label = str(center["label"])
        logR = float(center["logR"])
        y = np.array([float(center["logu"]), float(center["logT"])], dtype=float)
        lambda0 = float(center["lambda0"])
        center_row = make_row(label, "center_eval", center, logR, y, lambda0, ctx, trace_rows)
        rows.append(center_row)
        print(
            f"{label} center: rank={center_row['rank_defect']} sB={center_row['smin_over_smax_B']:.3e} "
            f"minor={center_row['rel_minor_norm']:.3e} access={center_row['access_distance_with_lambda']:.3g}",
            flush=True,
        )
        for mode in ("fixed_lambda_minors", "free_lambda_minors"):
            row = optimize_center(label, center, mode, ctx, trace_rows)
            rows.append(row)
            print(
                f"{label} {mode}: rank={row['rank_defect']} R={row['R_rg']:.6g} "
                f"sB={row['smin_over_smax_B']:.3e} minor={row['rel_minor_norm']:.3e} "
                f"access={row['access_distance_with_lambda']:.3g}",
                flush=True,
            )
    write_table(rows)
    write_json(rows)
    write_figure(rows)
    print(f"wrote {TABLE_OUTPUT}", flush=True)
    print(f"wrote {JSON_OUTPUT}", flush=True)
    if FIGURE_OUTPUT.exists():
        print(f"wrote {FIGURE_OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
