"""Search outer Mdot=5 phase manifolds without using the inner phase seed."""

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

import run_mdot5_phase_critical_classification as classification  # noqa: E402
from imri_qpe.constants import C  # noqa: E402
from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    algebraic_flux_ledger,
    algebraic_state,
    stream_source_prime,
)


model = classification.model
VALIDITY_R_RG = float(
    os.environ.get("IMBH_MDOT5_OUTER_SEARCH_MATCH_R_RG", "223.23642744192122")
)
ARC_TARGET = float(os.environ.get("IMBH_MDOT5_OUTER_SEARCH_ARC", "0.05"))
MAX_STEPS = int(os.environ.get("IMBH_MDOT5_OUTER_SEARCH_MAX_STEPS", "180"))
START_RADII = tuple(
    float(piece)
    for piece in os.environ.get(
        "IMBH_MDOT5_OUTER_SEARCH_START_RADII", "330,300,270,250,235,230"
    ).split(",")
    if piece.strip()
)
CONTINUE_LABELS = tuple(
    piece.strip()
    for piece in os.environ.get(
        "IMBH_MDOT5_OUTER_SEARCH_CONTINUE_LABELS", "R330,R300,R270,R250,R235,R230"
    ).split(",")
    if piece.strip()
)
SCOUT_LABELS = tuple(
    piece.strip()
    for piece in os.environ.get(
        "IMBH_MDOT5_OUTER_SEARCH_SCOUT_LABELS", "R235,R230"
    ).split(",")
    if piece.strip()
)
SCOUT_PERTURBATIONS = tuple(
    piece.strip()
    for piece in os.environ.get(
        "IMBH_MDOT5_OUTER_SEARCH_SCOUT_PERTURBATIONS", ""
    ).split(",")
    if piece.strip()
)

SUMMARY_PATH = ROOT / "outputs/tables/m5_eta_independent_outer_manifold_98p125_N164.json"
PROFILE_PATH = ROOT / "outputs/tables/m5_eta_independent_outer_manifold_98p125_N164_profiles.json"
FIGURE_PATH = ROOT / "outputs/figures/m5_eta_independent_outer_manifold_98p125_N164.png"
NOTE_PATH = ROOT / "Note/CODEX_MDOT5_INDEPENDENT_OUTER_MANIFOLD_RESULTS.md"


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


def _variant_params(params, shape: str, width_factor: float):
    return replace(
        params,
        stream_source_shape=shape,
        stream_source_log_width=float(params.stream_source_log_width) * width_factor,
    )


def _solve_logR_tangent_broad(
    z: np.ndarray, seeds: list[np.ndarray], params, lambda0: float
) -> list[dict[str, Any]]:
    from scipy.optimize import least_squares

    roots: list[dict[str, Any]] = []

    def residual(values: np.ndarray) -> np.ndarray:
        p = np.asarray([values[0], values[1], values[2], 1.0], dtype=float)
        return np.asarray(
            model._global_flux_phase_dae_point_data(z, p, params, lambda0)[
                "homogeneous_rows"
            ],
            dtype=float,
        )

    for seed in seeds:
        result = least_squares(
            residual,
            np.clip(np.asarray(seed[:3], dtype=float), [-1.0e7, -1.0e6, -100.0], [1.0e7, 1.0e6, 100.0]),
            bounds=([-1.0e7, -1.0e6, -100.0], [1.0e7, 1.0e6, 100.0]),
            x_scale="jac",
            max_nfev=250,
            ftol=1.0e-11,
            xtol=1.0e-11,
            gtol=1.0e-11,
        )
        p = np.asarray([result.x[0], result.x[1], result.x[2], 1.0], dtype=float)
        maximum = float(np.max(np.abs(residual(result.x))))
        if maximum > 3.0e-6:
            continue
        inward = -p / max(float(np.linalg.norm(p)), 1.0e-300)
        if inward[3] > 0.0:
            inward = -inward
        if any(
            abs(float(np.dot(inward, row["p_inward"]))) > 1.0 - 1.0e-7
            for row in roots
        ):
            continue
        point = model._global_flux_phase_dae_point_data(z, inward, params, lambda0)
        roots.append(
            {
                "p_logR": p,
                "p_inward": inward,
                "homogeneous_max": maximum,
                "nfev": int(result.nfev),
                "cond_A": float(point["cond_A"]),
                "sigma_min_A": float(np.min(point["A_singular_values"])),
            }
        )
    return roots


def _seed_atlas(x_log, params) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    logu, logT, logMdot, _logR_son, lambda0, logR = model.pilot._unpack(
        np.asarray(x_log, dtype=float), params
    )
    radii = np.exp(logR) / params.r_g
    slope_u = np.gradient(logu, logR)
    slope_T = np.gradient(logT, logR)
    slope_mdot = np.gradient(logMdot, logR)
    variants = (
        ("compact_c2", "compact_c2", 1.0),
        ("compact_c4", "compact_c4", 1.0),
        ("compact_cinf", "compact_cinf", 1.0),
        ("compact_c2_wide", "compact_c2", 1.25),
    )
    perturbations = (
        ("nominal", np.asarray([0.0, 0.0, 0.0])),
        ("u_plus", np.asarray([0.03, 0.0, 0.0])),
        ("u_minus", np.asarray([-0.03, 0.0, 0.0])),
        ("u_minus_045", np.asarray([-0.045, 0.0, 0.0])),
        ("u_minus_0525", np.asarray([-0.0525, 0.0, 0.0])),
        ("u_minus_060", np.asarray([-0.06, 0.0, 0.0])),
        ("T_plus", np.asarray([0.0, 0.03, 0.0])),
        ("T_minus", np.asarray([0.0, -0.03, 0.0])),
        ("F_plus", np.asarray([0.0, 0.0, 0.01])),
        ("F_minus", np.asarray([0.0, 0.0, -0.01])),
        ("u044_F0042", np.asarray([-0.044, 0.0, 0.0042])),
        ("u044_F0045", np.asarray([-0.044, 0.0, 0.0045])),
        ("u044_F0048", np.asarray([-0.044, 0.0, 0.0048])),
        ("u045_F0045", np.asarray([-0.045, 0.0, 0.0045])),
        ("u0435_F0043", np.asarray([-0.0435, 0.0, 0.0043])),
        ("u044_Tm0005_F0045", np.asarray([-0.044, -0.0005, 0.0045])),
        ("u044_Tm00065_F0045", np.asarray([-0.044, -0.00065, 0.0045])),
        ("u044_Tm0008_F0045", np.asarray([-0.044, -0.0008, 0.0045])),
        ("u04365_Tm00065_F00449", np.asarray([-0.04365, -0.00065, 0.00449])),
        ("u0437_Tm00065_F0045", np.asarray([-0.0437, -0.00065, 0.0045])),
        ("u0436_Tm00065_F0045", np.asarray([-0.0436, -0.00065, 0.0045])),
        ("u04418_F004501", np.asarray([-0.04418, 0.0, 0.004501])),
    )
    rows: list[dict[str, Any]] = []
    nominal: dict[str, Any] = {}
    tangent_guesses = (
        np.asarray([0.0, 0.0, 0.0, 1.0]),
        np.asarray([1.0e3, -1.0e2, 0.0, 1.0]),
        np.asarray([-1.0e3, 1.0e2, 0.0, 1.0]),
        np.asarray([1.0e4, -1.0e3, 1.0, 1.0]),
        np.asarray([-1.0e4, 1.0e3, -1.0, 1.0]),
    )
    for target in START_RADII:
        pos = int(np.argmin(np.abs(radii - target)))
        F0 = float(np.exp(logMdot[pos]) / params.Mdot_g_s)
        base = np.asarray([logu[pos], logT[pos], F0, logR[pos]], dtype=float)
        fd_seed = np.asarray(
            [slope_u[pos], slope_T[pos], F0 * slope_mdot[pos], 1.0], dtype=float
        )
        label = f"R{int(round(target))}"
        nominal[label] = {
            "z": base,
            "fd_seed": fd_seed,
            "actual_R_rg": float(radii[pos]),
            "lambda0": float(lambda0),
        }
        reference_roots = _solve_logR_tangent_broad(
            base,
            [fd_seed, *tangent_guesses],
            _variant_params(params, "compact_c2", 1.0),
            float(lambda0),
        )
        reference_seed = (
            np.asarray(reference_roots[0]["p_logR"], dtype=float)
            if reference_roots
            else fd_seed
        )
        for variant_name, shape, width_factor in variants:
            local_params = _variant_params(params, shape, width_factor)
            active_perturbations = (
                perturbations if variant_name == "compact_c2" else perturbations[:1]
            )
            for perturb_name, perturb in active_perturbations:
                z = base.copy()
                z[:3] += perturb
                z[2] = max(z[2], 1.0e-8)
                if variant_name == "compact_c2" and perturb_name == "nominal":
                    roots = reference_roots
                else:
                    roots = _solve_logR_tangent_broad(
                        z, [reference_seed], local_params, float(lambda0)
                    )
                rows.append(
                    {
                        "label": label,
                        "target_R_rg": target,
                        "actual_R_rg": float(radii[pos]),
                        "variant": variant_name,
                        "shape": shape,
                        "width_factor": width_factor,
                        "perturbation": perturb_name,
                        "z": z,
                        "root_count": len(roots),
                        "roots": roots,
                    }
                )
    return rows, nominal


def _trajectory_point(z: np.ndarray, p: np.ndarray, params, lambda0: float) -> dict[str, Any]:
    physical = classification._physical_point(z, p, params, lambda0)
    p_u = float(p[0])
    L_u_over_H = (
        abs(float(p[3] / p_u)) / max(float(physical["H_over_R"]), 1.0e-300)
        if abs(p_u) > 1.0e-300
        else math.inf
    )
    t_layer_over_dyn = float(L_u_over_H / max(physical["Mach_eff"], 1.0e-300))
    return {
        "z": np.asarray(z, dtype=float),
        "p": np.asarray(p, dtype=float),
        **physical,
        "L_u_over_H": L_u_over_H,
        "t_layer_over_t_dyn": t_layer_over_dyn,
    }


def _continue_arclength(
    label: str,
    z_start: np.ndarray,
    p_start: np.ndarray,
    params,
    lambda0: float,
    arc_initial: float,
) -> dict[str, Any]:
    z = np.asarray(z_start, dtype=float).copy()
    p = np.asarray(p_start, dtype=float).copy()
    p = p / max(float(np.linalg.norm(p)), 1.0e-300)
    if p[3] > 0.0:
        p = -p
    arc = float(arc_initial)
    points = [_trajectory_point(z, p, params, lambda0)]
    steps: list[dict[str, Any]] = []
    failures = 0
    outward_count = 0
    status = "max_steps"
    target_x = math.log(VALIDITY_R_RG * params.r_g)
    for step in range(MAX_STEPS):
        mode = "arclength"
        accepted = False
        diagnostics: dict[str, Any]
        ds = math.nan
        p_R = float(p[3])
        if abs(p_R) > 5.0e-2:
            p_logR = np.asarray(p, dtype=float) / p_R
            if (
                abs(float(p_logR[0])) < 450.0
                and abs(float(p_logR[1])) < 90.0
                and abs(float(p_logR[2])) < 1.8
            ):
                dx = -min(1.0e-3, max(float(z[3] - target_x), 1.0e-8))
                accepted, z_new, p_logR_new, logR_diagnostics = (
                    classification._implicit_logR_step(
                        z, p_logR, dx, params, lambda0
                    )
                )
                if accepted:
                    mode = "logR"
                    p_new = -np.asarray(p_logR_new, dtype=float)
                    p_new /= max(float(np.linalg.norm(p_new)), 1.0e-300)
                    diagnostics = {
                        **logR_diagnostics,
                        "arc_residual": math.nan,
                        "norm_max": abs(float(np.linalg.norm(p_new)) - 1.0),
                    }
                    ds = abs(dx) / max(abs(float(p_new[3])), 1.0e-300)
        if not accepted:
            accepted, z_new, _p_mid, p_new, ds, diagnostics = classification._bordered_step(
                z, p, arc, params, lambda0
            )
        if not accepted:
            arc *= 0.5
            failures += 1
            if arc < 0.0025 or failures >= 5:
                status = "solver_failure"
                break
            continue
        failures = 0
        point = _trajectory_point(z_new, p_new, params, lambda0)
        delta_R = float(point["R_rg"] - points[-1]["R_rg"])
        outward_count = outward_count + 1 if delta_R > 0.0 else 0
        steps.append(
            {
                "step": step,
                "mode": mode,
                "arc_target": arc,
                "ds": ds,
                "delta_R_rg": delta_R,
                **diagnostics,
            }
        )
        points.append(point)
        z, p = z_new, p_new
        if float(point["R_rg"]) <= VALIDITY_R_RG:
            status = "reached_validity_surface"
            break
        if outward_count >= 3:
            status = "radial_turn_before_match"
            break
        if not (
            0.0 < float(z[2]) < 5.0
            and 0.0 < float(z[0]) < 25.0
            and 5.0 < float(z[1]) < 25.0
        ):
            status = "state_bounds"
            break
        if len(points) >= 12:
            recent = np.asarray([row["R_rg"] for row in points[-10:]], dtype=float)
            if np.ptp(recent) < 1.0e-5 and abs(float(point["p_R"])) < 1.0e-5:
                status = "radial_stagnation_before_match"
                break
        if diagnostics["nfev"] < 20 and arc < arc_initial:
            arc = min(arc_initial, 1.25 * arc)
    return {
        "label": label,
        "status": status,
        "accepted_steps": len(steps),
        "minimum_R_rg": float(min(row["R_rg"] for row in points)),
        "final_R_rg": float(points[-1]["R_rg"]),
        "final_p_R": float(points[-1]["p_R"]),
        "crossed_scale_validity": any(row["L_u_over_H"] < 1.0 for row in points),
        "crossed_vertical_validity": any(
            row["t_layer_over_t_dyn"] < 1.0 for row in points
        ),
        "points": points,
        "steps": steps,
    }


def _inner_match_state(params, lambda0: float) -> dict[str, Any]:
    z, p, p_mid, ds = classification._load_phase(classification.EXIT_ANCHOR)
    samples: list[tuple[np.ndarray, np.ndarray]] = []
    for pos in range(ds.size):
        z_mid = 0.5 * (z[pos] + z[pos + 1]) + ds[pos] / 8.0 * (
            p[pos] - p[pos + 1]
        )
        samples.extend([(z[pos], p[pos]), (z_mid, p_mid[pos])])
    samples.append((z[-1], p[-1]))
    samples.sort(key=lambda item: float(item[0][3]))
    target_x = math.log(VALIDITY_R_RG * params.r_g)
    x = np.asarray([item[0][3] for item in samples], dtype=float)
    z_target = np.asarray(
        [np.interp(target_x, x, [item[0][col] for item in samples]) for col in range(4)],
        dtype=float,
    )
    nearest = int(np.argmin(np.abs(x - target_x)))
    p_target = np.asarray(samples[nearest][1], dtype=float)
    return _trajectory_point(z_target, p_target, params, lambda0)


def _flux_signature(point: dict[str, Any], params, lambda0: float) -> dict[str, float]:
    z = np.asarray(point["z"], dtype=float)
    p = np.asarray(point["p"], dtype=float)
    F = max(float(z[2]), 1.0e-300)
    dlogF = float(p[2] / (F * p[3]))
    local = model._local_params_with_point_mdot(
        params, float(z[3]), math.log(F * params.Mdot_g_s), dlogF
    )
    state = algebraic_state(float(z[3]), float(z[0]), float(z[1]), lambda0, local)
    y = np.asarray(z[:2], dtype=float)
    g = np.asarray(p[:2], dtype=float) / float(p[3])
    wind = model._safe_wind_prime(float(z[3]), y, g, lambda0, local)
    if not np.isfinite(wind):
        wind = 0.0
    source = float(stream_source_prime(float(z[3]), local))
    ledger = algebraic_flux_ledger(
        float(z[3]),
        state,
        local,
        mdot=F * params.Mdot_g_s,
        mdot_prime=float(wind - source),
        wind_prime=float(wind),
        stream_prime=source,
        closure="representation",
    )
    return {
        "F": F,
        "angular_flux_scaled": float(
            ledger.angular_flux / max(params.Mdot_g_s * state.l_K, 1.0e-300)
        ),
        "advected_internal_energy_scaled": float(F * state.e / C**2),
    }


def _match_at_surface(
    trajectory: dict[str, Any], inner: dict[str, Any], params, lambda0: float
) -> dict[str, Any] | None:
    if trajectory["status"] != "reached_validity_surface":
        return None
    points = trajectory["points"]
    outer = points[-1]
    inner_flux = _flux_signature(inner, params, lambda0)
    outer_flux = _flux_signature(outer, params, lambda0)
    state_delta = np.asarray(outer["z"][:3], dtype=float) - np.asarray(
        inner["z"][:3], dtype=float
    )
    flux_delta = {
        key: float(outer_flux[key] - inner_flux[key]) for key in inner_flux
    }
    state_max = float(np.max(np.abs(state_delta)))
    flux_max = float(max(abs(value) for value in flux_delta.values()))
    if state_max <= 1.0e-3 and flux_max <= 1.0e-4:
        classification = "connected_current_representation"
    elif state_max <= 5.0e-2 and flux_max <= 5.0e-2:
        classification = "exploratory_near_match"
    else:
        classification = "distinct_sheet_at_validity_surface"
    return {
        "classification": classification,
        "state_delta": state_delta,
        "state_max": state_max,
        "flux_delta": flux_delta,
        "flux_max": flux_max,
        "outer_R_rg": float(outer["R_rg"]),
    }


def _shooting_jacobian_audit(trajectories: list[dict[str, Any]]) -> dict[str, Any]:
    """Estimate the local seed-to-match map around the best R230 stencil."""

    matched = {
        row["label"]: row
        for row in trajectories
        if row.get("match") is not None
    }
    required = {
        "base": "R230_u044_F0045",
        "u": "R230_u045_F0045",
        "T": "R230_u044_Tm0005_F0045",
        "F_minus": "R230_u044_F0042",
        "F_plus": "R230_u044_F0048",
    }
    if any(label not in matched for label in required.values()):
        return {"available": False, "required_labels": required}

    def delta(label: str) -> np.ndarray:
        return np.asarray(matched[label]["match"]["state_delta"], dtype=float)

    base = delta(required["base"])
    jacobian = np.column_stack(
        (
            (delta(required["u"]) - base) / -1.0e-3,
            (delta(required["T"]) - base) / -5.0e-4,
            (delta(required["F_plus"]) - delta(required["F_minus"])) / 6.0e-4,
        )
    )
    _left, singular_values, right = np.linalg.svd(jacobian, full_matrices=False)
    condition = float(
        singular_values[0] / max(singular_values[-1], np.finfo(float).tiny)
    )
    u_direction = jacobian[:, 0]
    T_direction = jacobian[:, 1]
    direction_cosine = float(
        np.dot(u_direction, T_direction)
        / max(np.linalg.norm(u_direction) * np.linalg.norm(T_direction), 1.0e-300)
    )
    return {
        "available": True,
        "base_label": required["base"],
        "input_columns": ["delta_logu_seed", "delta_logT_seed", "delta_F_seed"],
        "output_rows": ["delta_logu_match", "delta_logT_match", "delta_F_match"],
        "jacobian": jacobian,
        "singular_values": singular_values,
        "condition": condition,
        "smallest_right_singular_vector": right[-1],
        "u_T_direction_cosine": direction_cosine,
    }


def _write_note(result: dict[str, Any]) -> None:
    lines = [
        "# Mdot=5 independent outer-manifold search",
        "",
        "The outer atlas is initialized only from the pre-phase global checkpoint. No accepted inner phase state or tangent is used to construct an outer seed.",
        "",
        f"Physical matching surface: `R={VALIDITY_R_RG:.6f} rg`.",
        "",
        "## Seed atlas",
        "",
        f"- Local seed states surveyed: `{result['atlas']['state_count']}`.",
        f"- Accepted tangent roots: `{result['atlas']['accepted_root_count']}`.",
        f"- States with multiple distinct tangent roots: `{result['atlas']['multiple_root_states']}`.",
        "",
        "## Continued trajectories",
        "",
        "| label | start R | steps | minimum R | final R | final p_R | status | scale-validity crossed | match class |",
        "|---|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in result["trajectories"]:
        match = row.get("match")
        lines.append(
            f"| `{row['label']}` | {row['start_R_rg']:.3f} | {row['accepted_steps']} | "
            f"{row['minimum_R_rg']:.3f} | {row['final_R_rg']:.3f} | {row['final_p_R']:.3e} | "
            f"`{row['status']}` | {row['crossed_scale_validity']} | "
            f"`{match['classification'] if match else '-'}` |"
        )
    best = result["decision"].get("best_match")
    if best:
        lines.extend(
            [
                "",
                "## Best conservative match",
                "",
                f"- trajectory: `{best['label']}`",
                f"- state delta `(logu, logT, F)`: `{best['state_delta']}`",
                f"- maximum state mismatch: `{best['state_max']:.6e}`",
                f"- flux delta: `{best['flux_delta']}`",
                f"- maximum flux mismatch: `{best['flux_max']:.6e}`",
            ]
        )
    shooting = result.get("shooting_jacobian", {})
    if shooting.get("available"):
        lines.extend(
            [
                "",
                "## Local shooting-map audit",
                "",
                f"- singular values: `{shooting['singular_values']}`",
                f"- condition number: `{shooting['condition']:.6e}`",
                f"- velocity/temperature direction cosine: `{shooting['u_T_direction_cosine']:.9f}`",
                "",
                "The near-collinearity of the velocity and temperature shooting directions is retained as a geometric diagnostic; a small flux residual alone is not promoted to a strict state connection.",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            result["decision"]["statement"],
            "",
            "This is a topology search under the exact algebraic representation closure. It is not a physical stream/wind angular-momentum certification.",
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
    colors = ("#176B87", "#C0392B", "#7D3C98", "#148F77", "#D68910", "#5D6D7E")

    def panel(box, xkey, ykey, title, logy=False):
        left, top, right, bottom = box
        x0, x1, y0, y1 = left + 82, right - 20, top + 42, bottom - 52
        series = []
        all_x, all_y = [], []
        best_label = (result.get("decision", {}).get("best_match") or {}).get("label")
        plotted = [
            row
            for row in result["trajectory_profiles"]
            if "_" not in row["label"] or row["label"] == best_label
        ]
        for position, row in enumerate(plotted):
            color = colors[position % len(colors)]
            xs = np.asarray([point[xkey] for point in row["points"]], dtype=float)
            ys = np.asarray([point[ykey] for point in row["points"]], dtype=float)
            valid = np.isfinite(xs) & np.isfinite(ys) & ((ys > 0.0) if logy else True)
            xs, ys = xs[valid], ys[valid]
            if logy:
                ys = np.log10(ys)
            series.append((xs, ys, color, row["label"]))
            all_x.extend(xs.tolist())
            all_y.extend(ys.tolist())
        if not all_x:
            return
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
            ytext = f"10^{yv:.1f}" if logy else f"{yv:.4g}"
            draw.text((left + 4, py - 6), ytext, fill="#34495E", font=font)
        draw.line((x0, y1, x1, y1), fill="#2C3E50", width=2)
        draw.line((x0, y0, x0, y1), fill="#2C3E50", width=2)
        legend_y = top + 8
        for xs, ys, color, label in series:
            points = [xy(float(x), float(y)) for x, y in zip(xs, ys)]
            if len(points) > 1:
                draw.line(points, fill=color, width=3)
            draw.line((right - 155, legend_y + 5, right - 135, legend_y + 5), fill=color, width=3)
            draw.text((right - 130, legend_y), label, fill="#34495E", font=font)
            legend_y += 16

    panel(boxes[0], "R_rg", "logu", "Independent outer phase trajectories")
    panel(boxes[1], "R_rg", "L_u_over_H", "Scale validity along outer trajectories", logy=True)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    image.save(FIGURE_PATH)


def main() -> None:
    x_log, params, _context, _aux, phase_seed = classification.global_phase._load_problem()
    lambda0 = float(phase_seed["lambda0"])
    atlas_rows, nominal = _seed_atlas(x_log, params)
    inner = _inner_match_state(params, lambda0)
    trajectory_profiles: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    jobs: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for base_label in CONTINUE_LABELS:
        seed_info = nominal.get(base_label)
        if seed_info is None:
            continue
        atlas = next(
            row
            for row in atlas_rows
            if row["label"] == base_label
            and row["variant"] == "compact_c2"
            and row["perturbation"] == "nominal"
        )
        jobs.append((base_label, atlas, seed_info))
    for base_label in SCOUT_LABELS:
        seed_info = nominal.get(base_label)
        if seed_info is None:
            continue
        for atlas in atlas_rows:
            if (
                atlas["label"] != base_label
                or atlas["variant"] != "compact_c2"
                or atlas["perturbation"] == "nominal"
                or (
                    SCOUT_PERTURBATIONS
                    and atlas["perturbation"] not in SCOUT_PERTURBATIONS
                )
            ):
                continue
            scout_info = {**seed_info, "z": np.asarray(atlas["z"], dtype=float)}
            jobs.append(
                (
                    f"{base_label}_{atlas['perturbation']}",
                    atlas,
                    scout_info,
                )
            )

    for label, atlas, seed_info in jobs:
        if not atlas["roots"]:
            trajectory_rows.append(
                {
                    "label": label,
                    "start_R_rg": float(seed_info["actual_R_rg"]),
                    "status": "no_tangent_root",
                    "accepted_steps": 0,
                    "minimum_R_rg": float(seed_info["actual_R_rg"]),
                    "final_R_rg": float(seed_info["actual_R_rg"]),
                    "final_p_R": math.nan,
                    "crossed_scale_validity": False,
                    "match": None,
                }
            )
            continue
        root = atlas["roots"][0]
        trajectory = _continue_arclength(
            label,
            np.asarray(seed_info["z"], dtype=float),
            np.asarray(root["p_inward"], dtype=float),
            params,
            lambda0,
            ARC_TARGET,
        )
        match = _match_at_surface(trajectory, inner, params, lambda0)
        trajectory_profiles.append(trajectory)
        trajectory_rows.append(
            {
                key: value
                for key, value in trajectory.items()
                if key not in {"points", "steps"}
            }
            | {
                "start_R_rg": float(seed_info["actual_R_rg"]),
                "seed_perturbation": np.asarray(atlas["z"], dtype=float)
                - np.asarray(nominal[atlas["label"]]["z"], dtype=float),
                "match": match,
            }
        )
        print(
            label,
            trajectory["status"],
            trajectory["accepted_steps"],
            f"Rmin={trajectory['minimum_R_rg']:.6f}",
            flush=True,
        )

    matches = [row["match"] for row in trajectory_rows if row.get("match")]
    connected = any(
        row["classification"] == "connected_current_representation" for row in matches
    )
    near = any(row["classification"] == "exploratory_near_match" for row in matches)
    reached = sum(row["status"] == "reached_validity_surface" for row in trajectory_rows)
    matched_rows = [row for row in trajectory_rows if row.get("match")]
    best_row = min(
        matched_rows,
        key=lambda row: (
            max(row["match"]["state_max"] / 1.0e-3, row["match"]["flux_max"] / 1.0e-4),
            row["match"]["state_max"],
        ),
        default=None,
    )
    shooting_jacobian = _shooting_jacobian_audit(trajectory_rows)
    if connected:
        statement = "An independently seeded outer trajectory connects to the inner phase state under the current algebraic representation closure. Physical certification still requires explicit stream/wind angular momentum."
        outcome = "connected_current_representation"
    elif near:
        statement = "An independently seeded trajectory reaches an exploratory near-match at the validity surface, but strict state/flux matching is not achieved."
        outcome = "exploratory_near_match"
    elif reached:
        statement = "Independent outer trajectories reach the validity surface only on distinct phase sheets; no state-and-flux match is found in the surveyed atlas."
        outcome = "distinct_sheet_at_validity_surface"
    else:
        statement = "No independently seeded outer trajectory reaches the physical validity surface in the surveyed atlas. This is `not found in the surveyed manifold`, not global nonexistence."
        outcome = "not_found_in_surveyed_manifold"

    result = {
        "target": {
            "eta_E": 98.125,
            "N": int(params.n_nodes),
            "validity_R_rg": VALIDITY_R_RG,
            "arc_target": ARC_TARGET,
            "max_steps": MAX_STEPS,
        },
        "atlas": {
            "state_count": len(atlas_rows),
            "accepted_root_count": int(sum(row["root_count"] for row in atlas_rows)),
            "multiple_root_states": int(sum(row["root_count"] > 1 for row in atlas_rows)),
        },
        "trajectories": trajectory_rows,
        "shooting_jacobian": shooting_jacobian,
        "decision": {
            "outcome": outcome,
            "connected": connected,
            "reached_validity_surface_count": int(reached),
            "statement": statement,
            "physical_closure_certified": False,
            "best_match": (
                {"label": best_row["label"], **best_row["match"]}
                if best_row is not None
                else None
            ),
        },
    }
    profiles = {
        **result,
        "atlas_rows": atlas_rows,
        "inner_match_state": inner,
        "trajectory_profiles": trajectory_profiles,
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n")
    PROFILE_PATH.write_text(json.dumps(_jsonable(profiles), indent=2, sort_keys=True) + "\n")
    _write_note(profiles)
    _write_figure(profiles)
    print(json.dumps(_jsonable(result["decision"]), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
