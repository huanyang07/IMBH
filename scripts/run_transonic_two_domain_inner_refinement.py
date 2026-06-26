"""Staged inner-grid refinement for the pressure-supported two-domain root."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import CubicHermiteSpline, PchipInterpolator

from imri_qpe.layer3_minidisk_1d.transonic_local import local_gradient
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot

from run_transonic_outer_slope_calibration_audit import fmt
from run_transonic_two_domain_mesh_validation import (
    build_final_row,
    combined_profile_arrays,
    compare_to_reference,
    load_checkpoint,
    make_params,
    worst_interval,
)
from run_transonic_two_domain_outer_extension import (
    R_MATCH_RG,
    audit_row,
    inner_grid,
    outer_grid,
    pack_two_domain,
    solve_two_domain,
    state_bounds_two_domain,
    unpack_two_domain,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CHECKPOINT = ROOT / "outputs" / "checkpoints" / "transonic_two_domain_mesh_validation" / "outer_N65_O54_R1e5_0p90277664.npz"
TABLE_OUTPUT = ROOT / "outputs" / "tables" / "transonic_two_domain_inner_refinement.md"
CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints" / "transonic_two_domain_inner_refinement"

N_SEQUENCE = (66, 67, 69, 73, 77, 78, 79, 80, 81, 89, 97, 113, 129)
MAX_NFEV_RELEASE = 900
MAX_NFEV_POLISH = 300
CHAIN_SOURCE_PHYSICAL_LIMIT = 1.0e-4
SCIENCE_RESIDUAL_LIMIT = 5.0e-6


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return value if np.isfinite(value) else str(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def row_json(row: dict[str, object]) -> str:
    return json.dumps({key: json_safe(value) for key, value in row.items()}, sort_keys=True)


def pchip_derivatives(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.asarray(PchipInterpolator(x, y).derivative()(x), dtype=float)


def ode_blended_derivatives(x: np.ndarray, logu: np.ndarray, logT: np.ndarray, lambda0: float, params, blend: float = 0.25) -> np.ndarray:
    pchip = np.column_stack([pchip_derivatives(x, logu), pchip_derivatives(x, logT)])
    values = []
    for idx, (lr, lu, lt) in enumerate(zip(x, logu, logT)):
        try:
            ode = local_gradient(float(lr), np.array([lu, lt], dtype=float), lambda0, params.physics)
            if not np.all(np.isfinite(ode)) or np.max(np.abs(ode)) > 10.0:
                raise RuntimeError("unusable local ODE slope")
            if np.linalg.norm(ode - pchip[idx]) > 1.0:
                ode = 0.35 * ode + 0.65 * pchip[idx]
        except Exception:
            ode = pchip[idx]
        values.append((1.0 - blend) * pchip[idx] + blend * ode)
    return np.asarray(values, dtype=float)


def hermite_excluding_sonic(x_old: np.ndarray, y_old: np.ndarray, x_new: np.ndarray, derivatives: np.ndarray) -> np.ndarray:
    interpolant = CubicHermiteSpline(x_old[1:], y_old[1:], derivatives[1:], extrapolate=True)
    y_new = np.asarray(interpolant(x_new), dtype=float)
    y_new[0] = float(y_old[0])
    return y_new


def remap_candidate(x_old: np.ndarray, old_params, new_params, method: str) -> np.ndarray:
    logu_i, logT_i, logu_o, logT_o, logR_son, lambda0 = unpack_two_domain(x_old, old_params)
    old_logR_i = inner_grid(logR_son, old_params)
    old_logR_o = outer_grid(old_params)
    new_logR_i = inner_grid(logR_son, new_params)
    new_logR_o = outer_grid(new_params)

    if method == "sonic_aware_pchip":
        logu_i_new = np.asarray(PchipInterpolator(old_logR_i[1:], logu_i[1:], extrapolate=True)(new_logR_i), dtype=float)
        logT_i_new = np.asarray(PchipInterpolator(old_logR_i[1:], logT_i[1:], extrapolate=True)(new_logR_i), dtype=float)
        logu_i_new[0] = float(logu_i[0])
        logT_i_new[0] = float(logT_i[0])
    elif method == "sonic_aware_hermite_pchip":
        du = pchip_derivatives(old_logR_i, logu_i)
        dT = pchip_derivatives(old_logR_i, logT_i)
        logu_i_new = hermite_excluding_sonic(old_logR_i, logu_i, new_logR_i, du)
        logT_i_new = hermite_excluding_sonic(old_logR_i, logT_i, new_logR_i, dT)
    elif method == "sonic_aware_hermite_ode_blend":
        derivatives = ode_blended_derivatives(old_logR_i, logu_i, logT_i, lambda0, old_params)
        logu_i_new = hermite_excluding_sonic(old_logR_i, logu_i, new_logR_i, derivatives[:, 0])
        logT_i_new = hermite_excluding_sonic(old_logR_i, logT_i, new_logR_i, derivatives[:, 1])
    else:
        raise ValueError(f"unknown remap method {method!r}")

    logu_o_new = np.asarray(PchipInterpolator(old_logR_o, logu_o, extrapolate=True)(new_logR_o), dtype=float)
    logT_o_new = np.asarray(PchipInterpolator(old_logR_o, logT_o, extrapolate=True)(new_logR_o), dtype=float)
    logu_o_new[0] = float(logu_i_new[-1])
    logT_o_new[0] = float(logT_i_new[-1])
    return pack_two_domain(logu_i_new, logT_i_new, logu_o_new, logT_o_new, logR_son, lambda0)


def choose_seed(x_old: np.ndarray, old_params, new_params) -> tuple[str, np.ndarray, list[dict[str, object]]]:
    candidates = []
    for method in ("sonic_aware_pchip", "sonic_aware_hermite_pchip", "sonic_aware_hermite_ode_blend"):
        seed = remap_candidate(x_old, old_params, new_params, method)
        row = audit_row(f"{method}_seed", seed, new_params)
        row["method"] = method
        candidates.append((method, seed, row))
    method, seed, _row = min(candidates, key=lambda item: float(item[2]["physical_active"]))
    return method, seed, [row for _method, _seed, row in candidates]


def save_checkpoint(label: str, row: dict[str, object]) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {key: value for key, value in row.items() if key != "x"}
    np.savez_compressed(
        CHECKPOINT_DIR / f"{label}_0p90277664.npz",
        x=np.asarray(row["x"], dtype=float),
        row_json=np.array(row_json(payload)),
    )


def stage_row(label: str, stage: str, row: dict[str, object], method: str, source_label: str) -> dict[str, object]:
    return {
        "label": label,
        "stage": stage,
        "method": method,
        "source_label": source_label,
        "physical": row["physical_active"],
        "dominant": row["dominant"],
        "inner_R": row["inner_R"],
        "inner_E": row["inner_E"],
        "outer_R": row["outer_R"],
        "far_closure_omega": row["far_omega"],
        "far_energy": row["far_energy"],
        "Rson_rg": row["Rson_rg"],
        "lambda0": row["lambda0"],
        "int_adv": row["int_adv"],
        "nfev": row["nfev"],
        "success": row["success"],
        "message": row["message"],
    }


def final_row_from_audit(label: str, source_label: str, method: str, row: dict[str, object], params, ref_row: dict[str, object], ref_arrays: dict[str, np.ndarray]) -> dict[str, object]:
    comparison = compare_to_reference(np.asarray(row["x"], dtype=float), params, ref_arrays)
    worst = worst_interval(np.asarray(row["x"], dtype=float), params)
    row.update(
        {
            "label": label,
            "source_label": source_label,
            "method": method,
            "group": "inner_grid",
            "note": "staged sonic-aware inner refinement",
            "delta_Rson_rg": float(row["Rson_rg"] - ref_row["Rson_rg"]),
            "delta_lambda0": float(row["lambda0"] - ref_row["lambda0"]),
            "delta_int_adv": float(row["int_adv"] - ref_row["int_adv"]),
            "max_dlogu_ref": comparison["max_dlogu"],
            "rms_dlogu_ref": comparison["rms_dlogu"],
            "max_dlogT_ref": comparison["max_dlogT"],
            "rms_dlogT_ref": comparison["rms_dlogT"],
            "max_dHR_ref": comparison["max_dHR"],
            "max_dlnOmega_ref": comparison["max_dlnOmega"],
            "worst_domain": worst["domain"],
            "worst_i": int(worst["idx"]),
            "worst_R_mid_rg": float(worst["R_mid_rg"]),
            "worst_component": worst["component"],
            "worst_interval_abs": float(worst["abs_max"]),
            "passes_science_limit": bool(float(row["physical_active"]) <= SCIENCE_RESIDUAL_LIMIT),
        }
    )
    return row


def skipped_row(label: str, source_label: str, source_row: dict[str, object], n_inner: int) -> dict[str, object]:
    return {
        "label": label,
        "source_label": source_label,
        "method": "skipped",
        "group": "inner_grid",
        "note": f"skipped because source physical={float(source_row['physical_active']):.3e} exceeds {CHAIN_SOURCE_PHYSICAL_LIMIT:.1e}",
        "n_inner": n_inner,
        "n_outer": source_row["n_outer"],
        "R_far_rg": source_row["R_far_rg"],
        "physical_active": np.nan,
        "dominant": "skipped",
        "passes_science_limit": False,
    }


def write_table(source_row: dict[str, object], final_rows: list[dict[str, object]], stage_rows: list[dict[str, object]], seed_rows: list[dict[str, object]]) -> None:
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Two-Domain Staged Inner Refinement",
        "",
        "Generated by `scripts/run_transonic_two_domain_inner_refinement.py`.",
        "",
        f"Source checkpoint: `{SOURCE_CHECKPOINT}`. The remap excludes the singular sonic endpoint from the Hermite/PCHIP interpolant, preserves the sonic values exactly, and then performs staged full-release plus polish solves at fixed `N_outer=54`, `R_far=1e5 rg`.",
        "",
        "## Source",
        "",
        "| label | N inner | N outer | physical | dominant | Rson/rg | lambda0 | int adv |",
        "|---|---:|---:|---:|---|---:|---:|---:|",
        "| source | {n_inner} | {n_outer} | {physical} | {dominant} | {Rson_rg} | {lambda0} | {int_adv} |".format(
            n_inner=source_row["n_inner"],
            n_outer=source_row["n_outer"],
            physical=fmt(float(source_row["physical_active"])),
            dominant=source_row["dominant"],
            Rson_rg=fmt(float(source_row["Rson_rg"])),
            lambda0=fmt(float(source_row["lambda0"])),
            int_adv=fmt(float(source_row["int_adv"])),
        ),
        "",
        "## Final Rows",
        "",
        "| label | method | source | N inner | physical | pass | dominant | inner R | inner E | outer R | far closure omega | far energy | Rson/rg | dRson | lambda0 | dlambda | int adv | dint adv | max dlogu | rms dlogu | max dlogT | rms dlogT | max dH/R | max dlnOmega | worst domain | worst R/rg | worst comp | worst interval | nfev | success | note |",
        "|---|---|---|---:|---:|:---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|:---:|---:|---:|:---:|---|",
    ]
    for row in final_rows:
        lines.append(
            "| {label} | {method} | {source_label} | {n_inner} | {physical_active} | {passes_science_limit} | {dominant} | "
            "{inner_R} | {inner_E} | {outer_R} | {far_omega} | {far_energy} | {Rson_rg} | {delta_Rson_rg} | "
            "{lambda0} | {delta_lambda0} | {int_adv} | {delta_int_adv} | {max_dlogu_ref} | {rms_dlogu_ref} | "
            "{max_dlogT_ref} | {rms_dlogT_ref} | {max_dHR_ref} | {max_dlnOmega_ref} | {worst_domain} | "
            "{worst_R_mid_rg} | {worst_component} | {worst_interval_abs} | {nfev} | {success} | {note} |".format(
                label=row["label"],
                method=row["method"],
                source_label=row["source_label"],
                n_inner=row["n_inner"],
                physical_active=fmt(float(row.get("physical_active", np.nan))),
                passes_science_limit="yes" if bool(row.get("passes_science_limit", False)) else "no",
                dominant=row.get("dominant", "nan"),
                inner_R=fmt(float(row.get("inner_R", np.nan))),
                inner_E=fmt(float(row.get("inner_E", np.nan))),
                outer_R=fmt(float(row.get("outer_R", np.nan))),
                far_omega=fmt(float(row.get("far_omega", np.nan))),
                far_energy=fmt(float(row.get("far_energy", np.nan))),
                Rson_rg=fmt(float(row.get("Rson_rg", np.nan))),
                delta_Rson_rg=fmt(float(row.get("delta_Rson_rg", np.nan))),
                lambda0=fmt(float(row.get("lambda0", np.nan))),
                delta_lambda0=fmt(float(row.get("delta_lambda0", np.nan))),
                int_adv=fmt(float(row.get("int_adv", np.nan))),
                delta_int_adv=fmt(float(row.get("delta_int_adv", np.nan))),
                max_dlogu_ref=fmt(float(row.get("max_dlogu_ref", np.nan))),
                rms_dlogu_ref=fmt(float(row.get("rms_dlogu_ref", np.nan))),
                max_dlogT_ref=fmt(float(row.get("max_dlogT_ref", np.nan))),
                rms_dlogT_ref=fmt(float(row.get("rms_dlogT_ref", np.nan))),
                max_dHR_ref=fmt(float(row.get("max_dHR_ref", np.nan))),
                max_dlnOmega_ref=fmt(float(row.get("max_dlnOmega_ref", np.nan))),
                worst_domain=row.get("worst_domain", "nan"),
                worst_R_mid_rg=fmt(float(row.get("worst_R_mid_rg", np.nan))),
                worst_component=row.get("worst_component", "nan"),
                worst_interval_abs=fmt(float(row.get("worst_interval_abs", np.nan))),
                nfev=row.get("nfev", 0),
                success="yes" if bool(row.get("success", False)) else "no",
                note=row.get("note", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Seed Candidates",
            "",
            "| label | N inner | method | physical | dominant | inner R | inner E | Rson/rg | int adv |",
            "|---|---:|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in seed_rows:
        lines.append(
            "| {label} | {n_inner} | {method} | {physical} | {dominant} | {inner_R} | {inner_E} | {Rson_rg} | {int_adv} |".format(
                label=row["label"],
                n_inner=row["n_inner"],
                method=row["method"],
                physical=fmt(float(row["physical_active"])),
                dominant=row["dominant"],
                inner_R=fmt(float(row["inner_R"])),
                inner_E=fmt(float(row["inner_E"])),
                Rson_rg=fmt(float(row["Rson_rg"])),
                int_adv=fmt(float(row["int_adv"])),
            )
        )
    lines.extend(
        [
            "",
            "## Stage Rows",
            "",
            "| label | stage | method | source | physical | dominant | inner R | inner E | outer R | far closure omega | far energy | Rson/rg | lambda0 | int adv | nfev | success | message |",
            "|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|",
        ]
    )
    for row in stage_rows:
        lines.append(
            "| {label} | {stage} | {method} | {source_label} | {physical} | {dominant} | {inner_R} | {inner_E} | "
            "{outer_R} | {far_closure_omega} | {far_energy} | {Rson_rg} | {lambda0} | {int_adv} | {nfev} | "
            "{success} | {message} |".format(
                label=row["label"],
                stage=row["stage"],
                method=row["method"],
                source_label=row["source_label"],
                physical=fmt(float(row["physical"])),
                dominant=row["dominant"],
                inner_R=fmt(float(row["inner_R"])),
                inner_E=fmt(float(row["inner_E"])),
                outer_R=fmt(float(row["outer_R"])),
                far_closure_omega=fmt(float(row["far_closure_omega"])),
                far_energy=fmt(float(row["far_energy"])),
                Rson_rg=fmt(float(row["Rson_rg"])),
                lambda0=fmt(float(row["lambda0"])),
                int_adv=fmt(float(row["int_adv"])),
                nfev=row["nfev"],
                success="yes" if row["success"] else "no",
                message=str(row["message"]).replace("|", "/"),
            )
        )
    TABLE_OUTPUT.write_text("\n".join(lines) + "\n")


def main() -> None:
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)
    x_source, source_meta = load_checkpoint(SOURCE_CHECKPOINT)
    ratio = float(source_meta["ratio"])
    source_params = make_params(
        fiducial,
        ratio,
        mdot_edd,
        int(source_meta["n_inner"]),
        int(source_meta["n_outer"]),
        float(source_meta["R_far_rg"]),
    )
    source_audit = audit_row("source", x_source, source_params)
    source_audit["label"] = "source"
    source_audit["source_label"] = "source_checkpoint"
    source_audit["method"] = "loaded"
    source_audit["group"] = "source"
    source_audit["note"] = "validated outer-grid pressure-supported root"
    ref_arrays = combined_profile_arrays(x_source, source_params)
    ref_row = {
        "Rson_rg": float(source_audit["Rson_rg"]),
        "lambda0": float(source_audit["lambda0"]),
        "int_adv": float(source_audit["int_adv"]),
    }

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    for old_path in CHECKPOINT_DIR.glob("*.npz"):
        old_path.unlink()

    current_x = x_source
    current_params = source_params
    current_label = "source"
    final_rows: list[dict[str, object]] = []
    stage_rows: list[dict[str, object]] = []
    seed_rows: list[dict[str, object]] = []
    write_table(source_audit, final_rows, stage_rows, seed_rows)
    print(
        f"source physical={source_audit['physical_active']:.3e} dominant={source_audit['dominant']} "
        f"N=({source_audit['n_inner']},{source_audit['n_outer']})",
        flush=True,
    )

    for n_inner in N_SEQUENCE:
        if final_rows and float(final_rows[-1]["physical_active"]) > CHAIN_SOURCE_PHYSICAL_LIMIT:
            row = skipped_row(f"N{n_inner}", current_label, final_rows[-1], n_inner)
            final_rows.append(row)
            write_table(source_audit, final_rows, stage_rows, seed_rows)
            print(f"N{n_inner} skipped because source {current_label} failed", flush=True)
            break

        params = make_params(fiducial, ratio, mdot_edd, n_inner, current_params.n_outer, current_params.R_far_rg)
        method, seed, candidates = choose_seed(current_x, current_params, params)
        for candidate in candidates:
            candidate["label"] = f"N{n_inner}"
            seed_rows.append(candidate)
        seed_audit = audit_row(f"N{n_inner}_seed", seed, params)
        stage_rows.append(stage_row(f"N{n_inner}", "seed", seed_audit, method, current_label))
        write_table(source_audit, final_rows, stage_rows, seed_rows)
        print(
            f"N{n_inner} seed method={method} physical={seed_audit['physical_active']:.3e} "
            f"dominant={seed_audit['dominant']}",
            flush=True,
        )

        release = solve_two_domain(seed, params, state_bounds_two_domain(params), MAX_NFEV_RELEASE)
        release_audit = audit_row(f"N{n_inner}_release", release.x, params, release)
        stage_rows.append(stage_row(f"N{n_inner}", "release", release_audit, method, current_label))
        write_table(source_audit, final_rows, stage_rows, seed_rows)
        print(
            f"N{n_inner} release physical={release_audit['physical_active']:.3e} "
            f"dominant={release_audit['dominant']} nfev={release.nfev}",
            flush=True,
        )

        polish = solve_two_domain(release.x, params, state_bounds_two_domain(params), MAX_NFEV_POLISH)
        polish_audit = audit_row(f"N{n_inner}_polish", polish.x, params, polish)
        stage_rows.append(stage_row(f"N{n_inner}", "polish", polish_audit, method, current_label))
        final = final_row_from_audit(f"N{n_inner}", current_label, method, polish_audit, params, ref_row, ref_arrays)
        final_rows.append(final)
        save_checkpoint(f"N{n_inner}", final)
        write_table(source_audit, final_rows, stage_rows, seed_rows)
        print(
            f"N{n_inner} polish physical={final['physical_active']:.3e} dominant={final['dominant']} "
            f"Rson={final['Rson_rg']:.4f} dRson={final['delta_Rson_rg']:.3e} nfev={polish.nfev}",
            flush=True,
        )

        current_x = np.asarray(final["x"], dtype=float)
        current_params = params
        current_label = f"N{n_inner}"

    print(f"wrote {TABLE_OUTPUT}")


if __name__ == "__main__":
    main()
