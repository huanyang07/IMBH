"""Scan L'Hopital sonic-derivative roots for the two-domain source."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from imri_qpe.layer3_minidisk_1d.transonic_local import (
    differential_residual_scales,
    local_scaled_residual,
    local_unscaled_residual,
    sonic_directional_B,
    sonic_frozen_scaled_directional_B,
    sonic_lhopital_residual_form,
    sonic_lhopital_residual,
    sonic_null_vectors,
    sonic_unscaled_directional_B,
    sonic_unscaled_null_vectors,
    sonic_diagnostics,
)
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot

from run_transonic_outer_slope_calibration_audit import fmt
from run_transonic_two_domain_dynamic_sonic_patch import CHECKPOINT_DIR as DYNAMIC_CHECKPOINT_DIR, load_row
from run_transonic_two_domain_mesh_validation import load_checkpoint, make_params
from run_transonic_two_domain_outer_extension import inner_grid, unpack_two_domain
from run_transonic_two_domain_sonic_refinement_sprint import SOURCE_CHECKPOINT, buffer_inner_grid, make_buffer_params


ROOT = Path(__file__).resolve().parents[1]
TABLE_OUTPUT = ROOT / "outputs" / "tables" / "transonic_sonic_derivative_roots.md"
AUDIT_OUTPUT = ROOT / "outputs" / "tables" / "transonic_sonic_lhopital_audit.md"
DYNAMIC_N64_CHECKPOINT = DYNAMIC_CHECKPOINT_DIR / "Nreg64_0p90277664.npz"
EPS_VALUES = (3.0e-4, 1.0e-4, 3.0e-5, 1.0e-5, 3.0e-6)
FORMS = ("scaled", "frozen_scaled", "raw")
SCAN_HALF_WIDTH = 1000.0
SCAN_POINTS = 4001


def form_null_vectors(logR: float, y: np.ndarray, lambda0: float, params, form: str):
    if form == "raw":
        return sonic_unscaled_null_vectors(logR, y, lambda0, params)
    return sonic_null_vectors(logR, y, lambda0, params)


def form_directional_B(logR: float, y: np.ndarray, g: np.ndarray, lambda0: float, params, eps: float, form: str) -> np.ndarray:
    if form == "scaled":
        return sonic_directional_B(logR, y, g, lambda0, params, eps=eps)
    if form == "frozen_scaled":
        return sonic_frozen_scaled_directional_B(logR, y, g, lambda0, params, eps=eps)
    if form == "raw":
        return sonic_unscaled_directional_B(logR, y, g, lambda0, params, eps=eps)
    raise ValueError(f"unknown L'Hopital form {form!r}")


def form_local_residual(logR: float, y: np.ndarray, g: np.ndarray, lambda0: float, params, form: str) -> np.ndarray:
    if form == "scaled":
        return local_scaled_residual(logR, y, g, lambda0, params)
    if form == "frozen_scaled":
        radial_scale, energy_scale = differential_residual_scales(logR, y, lambda0, params)
        return local_unscaled_residual(logR, y, g, lambda0, params) / np.array([radial_scale, energy_scale])
    if form == "raw":
        return local_unscaled_residual(logR, y, g, lambda0, params)
    raise ValueError(f"unknown L'Hopital form {form!r}")


def raw_lhopital_value(logR: float, y: np.ndarray, g: np.ndarray, lambda0: float, params, eps: float, form: str) -> float:
    nulls = form_null_vectors(logR, y, lambda0, params, form)
    B = form_directional_B(logR, y, g, lambda0, params, eps=eps, form=form)
    return float(np.dot(nulls.left_null, B))


def load_source_state():
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)
    x_source, meta = load_checkpoint(SOURCE_CHECKPOINT)
    params = make_params(
        fiducial,
        float(meta["ratio"]),
        mdot_edd,
        int(meta["n_inner"]),
        int(meta["n_outer"]),
        float(meta["R_far_rg"]),
    )
    return x_source, params, meta


def source_first_slope(x_source: np.ndarray, params) -> np.ndarray:
    logu_i, logT_i, _logu_o, _logT_o, logR_son, _lambda0 = unpack_two_domain(x_source, params)
    logR_i = inner_grid(logR_son, params)
    dx = float(logR_i[1] - logR_i[0])
    return np.array([(float(logu_i[1]) - float(logu_i[0])) / dx, (float(logT_i[1]) - float(logT_i[0])) / dx], dtype=float)


def dynamic_patch_slope(ratio: float, mdot_edd: float) -> np.ndarray | None:
    if not DYNAMIC_N64_CHECKPOINT.exists():
        return None
    x_dynamic, meta = load_row(DYNAMIC_N64_CHECKPOINT)
    params = make_buffer_params(
        FiducialParams(),
        ratio,
        mdot_edd,
        int(meta["n_regular"]),
        int(meta["n_outer"]),
        float(meta["R_far_rg"]),
        float(meta["delta_s"]),
    )
    logu_i, logT_i, _logu_o, _logT_o, logR_son, _lambda0 = unpack_two_domain(x_dynamic, params)  # type: ignore[arg-type]
    logR_i = buffer_inner_grid(logR_son, params)
    dx = float(logR_i[1] - logR_i[0])
    return np.array([(float(logu_i[1]) - float(logu_i[0])) / dx, (float(logT_i[1]) - float(logT_i[0])) / dx], dtype=float)


def find_roots(logR: float, y: np.ndarray, lambda0: float, params, eps: float, a_ref: float, form: str) -> list[dict[str, float | str]]:
    nulls = form_null_vectors(logR, y, lambda0, params, form)
    g_p = np.linalg.lstsq(nulls.matrix, -nulls.rhs, rcond=None)[0]
    r = nulls.right_null / (np.linalg.norm(nulls.right_null) + 1.0e-300)
    a_values = np.linspace(a_ref - SCAN_HALF_WIDTH, a_ref + SCAN_HALF_WIDTH, SCAN_POINTS)

    def raw_from_a(a: float) -> float:
        return raw_lhopital_value(logR, y, g_p + a * r, lambda0, params, eps=eps, form=form)

    raw_values = np.asarray([raw_from_a(float(a)) for a in a_values], dtype=float)
    roots: list[dict[str, float | str]] = []
    for idx in range(len(a_values) - 1):
        left_a = float(a_values[idx])
        right_a = float(a_values[idx + 1])
        left_f = float(raw_values[idx])
        right_f = float(raw_values[idx + 1])
        if not np.isfinite(left_f) or not np.isfinite(right_f):
            continue
        if left_f == 0.0:
            root_a = left_a
        elif left_f * right_f > 0.0:
            continue
        else:
            try:
                root_a = float(brentq(raw_from_a, left_a, right_a, xtol=1.0e-10, rtol=1.0e-10, maxiter=80))
            except ValueError:
                continue
        g = g_p + root_a * r
        roots.append({"kind": "sign", "a": root_a, "g_u": float(g[0]), "g_T": float(g[1]), "L": raw_from_a(root_a)})

    # Include a best local minimum of |L| in case the scan misses a tangential root.
    best_idx = int(np.nanargmin(np.abs(raw_values)))
    left = float(a_values[max(0, best_idx - 4)])
    right = float(a_values[min(len(a_values) - 1, best_idx + 4)])
    try:
        minimum = minimize_scalar(lambda a: abs(raw_from_a(float(a))), bounds=(left, right), method="bounded", options={"xatol": 1.0e-8})
        if minimum.success:
            root_a = float(minimum.x)
            g = g_p + root_a * r
            roots.append({"kind": "minimum", "a": root_a, "g_u": float(g[0]), "g_T": float(g[1]), "L": raw_from_a(root_a)})
    except ValueError:
        pass

    # De-duplicate roots caused by adjacent exact zeros.
    unique: list[dict[str, float | str]] = []
    for root in sorted(roots, key=lambda item: float(item["a"])):
        if unique and abs(float(root["a"]) - float(unique[-1]["a"])) < 1.0e-5:
            if abs(float(root["L"])) < abs(float(unique[-1]["L"])):
                unique[-1] = root
            continue
        unique.append(root)
    return unique


def write_table(rows: list[dict[str, object]]) -> None:
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Sonic Derivative Root Scan",
        "",
        "Generated by `scripts/run_transonic_sonic_derivative_root_scan.py`.",
        "",
        "The scan solves `g = g_p + a r` and finds roots/minima of the L'Hopital condition `l^T B(g)=0` at the source sonic point.",
        "",
        "| form | eps | kind | a | g_u | g_T | L raw | L normalized | local max | dist source | dist dynamic | chosen |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            "| {form} | {eps} | {kind} | {a} | {g_u} | {g_T} | {L} | {L_norm} | {local_max} | {dist_source} | {dist_dynamic} | {chosen} |".format(
                form=row["form"],
                eps=fmt(float(row["eps"])),
                kind=row["kind"],
                a=fmt(float(row["a"])),
                g_u=fmt(float(row["g_u"])),
                g_T=fmt(float(row["g_T"])),
                L=fmt(float(row["L"])),
                L_norm=fmt(float(row["L_norm"])),
                local_max=fmt(float(row["local_max"])),
                dist_source=fmt(float(row["dist_source"])),
                dist_dynamic=fmt(float(row["dist_dynamic"])),
                chosen="yes" if row["chosen"] else "no",
            )
        )
    TABLE_OUTPUT.write_text("\n".join(lines) + "\n")


def audit_rows(logR: float, y: np.ndarray, lambda0: float, params, slopes: dict[str, np.ndarray]) -> list[dict[str, object]]:
    sonic = sonic_diagnostics(logR, y, lambda0, params)
    rows = []
    for form in FORMS:
        for eps in EPS_VALUES:
            for label, g in slopes.items():
                B = form_directional_B(logR, y, g, lambda0, params, eps=eps, form=form)
                nulls = form_null_vectors(logR, y, lambda0, params, form)
                rows.append(
                    {
                        "form": form,
                        "eps": eps,
                        "slope": label,
                        "g_u": float(g[0]),
                        "g_T": float(g[1]),
                        "local_max": float(np.max(np.abs(form_local_residual(logR, y, g, lambda0, params, form=form)))),
                        "L_raw": float(np.dot(nulls.left_null, B)),
                        "L_norm": sonic_lhopital_residual_form(logR, y, g, lambda0, params, eps=eps, form=form),
                        "B_norm": float(np.linalg.norm(B)),
                        "D": float(sonic.D),
                        "C1": float(sonic.C1),
                        "C2": float(sonic.C2),
                        "K": float(sonic.compatibility),
                    }
                )
    return rows


def write_audit_table(rows: list[dict[str, object]]) -> None:
    AUDIT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Sonic L'Hopital Form Audit",
        "",
        "Generated by `scripts/run_transonic_sonic_derivative_root_scan.py`.",
        "",
        "This compares source, dynamic-patch, and selected-root slopes using fully scaled, frozen-scale, and raw L'Hopital derivatives.",
        "",
        "| form | eps | slope | g_u | g_T | local max | L raw | L normalized | B norm | D | C1 | C2 | K |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {form} | {eps} | {slope} | {g_u} | {g_T} | {local_max} | {L_raw} | {L_norm} | {B_norm} | {D} | {C1} | {C2} | {K} |".format(
                form=row["form"],
                eps=fmt(float(row["eps"])),
                slope=row["slope"],
                g_u=fmt(float(row["g_u"])),
                g_T=fmt(float(row["g_T"])),
                local_max=fmt(float(row["local_max"])),
                L_raw=fmt(float(row["L_raw"])),
                L_norm=fmt(float(row["L_norm"])),
                B_norm=fmt(float(row["B_norm"])),
                D=fmt(float(row["D"])),
                C1=fmt(float(row["C1"])),
                C2=fmt(float(row["C2"])),
                K=fmt(float(row["K"])),
            )
        )
    AUDIT_OUTPUT.write_text("\n".join(lines) + "\n")


def main() -> None:
    source_x, params, meta = load_source_state()
    logu_i, logT_i, _logu_o, _logT_o, logR_son, lambda0 = unpack_two_domain(source_x, params)
    y_s = np.array([logu_i[0], logT_i[0]], dtype=float)
    nulls = sonic_null_vectors(logR_son, y_s, lambda0, params.physics)
    g_p = np.linalg.lstsq(nulls.matrix, -nulls.rhs, rcond=None)[0]
    r = nulls.right_null / (np.linalg.norm(nulls.right_null) + 1.0e-300)
    g_source = source_first_slope(source_x, params)
    g_dynamic = dynamic_patch_slope(float(meta["ratio"]), eddington_mdot(FiducialParams().M2_g))
    if g_dynamic is None:
        g_dynamic = g_source
    a_ref = float(np.dot(g_source - g_p, r))

    rows: list[dict[str, object]] = []
    chosen_slopes: dict[str, np.ndarray] = {}
    for form in FORMS:
        form_nulls = form_null_vectors(logR_son, y_s, lambda0, params.physics, form)
        form_gp = np.linalg.lstsq(form_nulls.matrix, -form_nulls.rhs, rcond=None)[0]
        form_r = form_nulls.right_null / (np.linalg.norm(form_nulls.right_null) + 1.0e-300)
        form_a_ref = float(np.dot(g_source - form_gp, form_r))
        for eps in EPS_VALUES:
            roots = find_roots(logR_son, y_s, lambda0, params.physics, eps, form_a_ref, form)
            if not roots:
                continue
            best_index = min(
                range(len(roots)),
                key=lambda idx: float(np.linalg.norm(np.array([roots[idx]["g_u"], roots[idx]["g_T"]], dtype=float) - g_source)),
            )
            for idx, root in enumerate(roots):
                g = np.array([root["g_u"], root["g_T"]], dtype=float)
                if eps == 1.0e-5 and idx == best_index:
                    chosen_slopes[f"{form}_root"] = g
                rows.append(
                    {
                        "form": form,
                        "eps": eps,
                        "kind": root["kind"],
                        "a": float(root["a"]),
                        "g_u": float(g[0]),
                        "g_T": float(g[1]),
                        "L": float(root["L"]),
                        "L_norm": sonic_lhopital_residual_form(logR_son, y_s, g, lambda0, params.physics, eps=eps, form=form),
                        "local_max": float(np.max(np.abs(form_local_residual(logR_son, y_s, g, lambda0, params.physics, form=form)))),
                        "dist_source": float(np.linalg.norm(g - g_source)),
                        "dist_dynamic": float(np.linalg.norm(g - g_dynamic)),
                        "chosen": idx == best_index,
                    }
                )
    write_table(rows)
    slopes = {"source": g_source, "dynamic": g_dynamic, **chosen_slopes}
    write_audit_table(audit_rows(logR_son, y_s, lambda0, params.physics, slopes))
    print(f"source slope g=({g_source[0]:.6g}, {g_source[1]:.6g})")
    print(f"dynamic slope g=({g_dynamic[0]:.6g}, {g_dynamic[1]:.6g})")
    print(f"particular g_p=({g_p[0]:.6g}, {g_p[1]:.6g})")
    print(f"right null r=({r[0]:.6g}, {r[1]:.6g})")
    print(f"wrote {TABLE_OUTPUT}")
    print(f"wrote {AUDIT_OUTPUT}")


if __name__ == "__main__":
    main()
