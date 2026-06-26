"""Mesh validation for the pressure-supported two-domain transonic root."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from imri_qpe.layer3_minidisk_1d import TransonicSlimParams
from imri_qpe.layer3_minidisk_1d.transonic_local import algebraic_state
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot

from run_transonic_outer_slope_calibration_audit import fmt
from run_transonic_two_domain_outer_extension import (
    CHECKPOINT_DIR as SOURCE_CHECKPOINT_DIR,
    R_MATCH_RG,
    TwoDomainParams,
    audit_row,
    inner_grid,
    interval_blocks,
    locked_bounds,
    outer_grid,
    solve_two_domain,
    state_bounds_two_domain,
    unpack_two_domain,
)


ROOT = Path(__file__).resolve().parents[1]
TABLE_OUTPUT = ROOT / "outputs" / "tables" / "transonic_two_domain_mesh_validation.md"
CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints" / "transonic_two_domain_mesh_validation"
REFERENCE_CHECKPOINT = SOURCE_CHECKPOINT_DIR / "Rfar1e5_pressure_supported_polish_0p90277664.npz"

MAX_NFEV_LOCKED = 450
MAX_NFEV_RELEASE = 900
MAX_NFEV_POLISH = 300
CHAIN_SOURCE_PHYSICAL_LIMIT = 1.0e-4


@dataclass(frozen=True)
class ValidationSpec:
    label: str
    group: str
    n_inner: int
    n_outer: int
    R_far_rg: float
    source_label: str
    note: str


SPECS = (
    ValidationSpec("outer_N65_O54_R1e5", "outer_grid", 65, 54, 1.0e5, "reference", "outer-grid refinement"),
    ValidationSpec("outer_N65_O72_R1e5", "outer_grid", 65, 72, 1.0e5, "outer_N65_O54_R1e5", "outer-grid refinement"),
    ValidationSpec("far_N65_O54_R5e4", "far_radius", 65, 54, 5.0e4, "outer_N65_O54_R1e5", "far-radius sensitivity"),
    ValidationSpec("far_N65_O54_R2e5", "far_radius", 65, 54, 2.0e5, "outer_N65_O54_R1e5", "far-radius sensitivity"),
    ValidationSpec("inner_N97_O54_R1e5", "inner_grid", 97, 54, 1.0e5, "outer_N65_O54_R1e5", "inner-grid refinement"),
    ValidationSpec("inner_N129_O54_R1e5", "inner_grid", 129, 54, 1.0e5, "inner_N97_O54_R1e5", "inner-grid refinement"),
)


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


def load_checkpoint(path: Path) -> tuple[np.ndarray, dict[str, object]]:
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data["x"], dtype=float), json.loads(str(data["row_json"].item()))


def make_params(fiducial: FiducialParams, ratio: float, mdot_edd: float, n_inner: int, n_outer: int, R_far_rg: float) -> TwoDomainParams:
    physics = TransonicSlimParams(
        M2_g=fiducial.M2_g,
        Mdot_g_s=ratio * mdot_edd,
        alpha=fiducial.alpha_cool,
        n_nodes=n_inner,
        R_out_rg=R_far_rg,
        residual_tol=1.0e-6,
        max_nfev=MAX_NFEV_RELEASE,
        outer_closure="pressure_supported_thin_energy",
        interval_residual_form="differential",
        integrated_residual_weighting="none",
    )
    return TwoDomainParams(
        physics=physics,
        n_inner=n_inner,
        n_outer=n_outer,
        R_match_rg=R_MATCH_RG,
        R_far_rg=R_far_rg,
        far_closure="pressure_supported",
    )


def interp_extrap(x_source: np.ndarray, y_source: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    x_source = np.asarray(x_source, dtype=float)
    y_source = np.asarray(y_source, dtype=float)
    x_new = np.asarray(x_new, dtype=float)
    y_new = np.interp(x_new, x_source, y_source)
    if len(x_source) < 2:
        return y_new
    left = x_new < x_source[0]
    right = x_new > x_source[-1]
    if np.any(left):
        slope = (y_source[1] - y_source[0]) / (x_source[1] - x_source[0])
        y_new[left] = y_source[0] + slope * (x_new[left] - x_source[0])
    if np.any(right):
        slope = (y_source[-1] - y_source[-2]) / (x_source[-1] - x_source[-2])
        y_new[right] = y_source[-1] + slope * (x_new[right] - x_source[-1])
    return y_new


def remap_state(x_old: np.ndarray, old_params: TwoDomainParams, new_params: TwoDomainParams) -> np.ndarray:
    old_logu_i, old_logT_i, old_logu_o, old_logT_o, logR_son, lambda0 = unpack_two_domain(x_old, old_params)
    old_logR_i = inner_grid(logR_son, old_params)
    old_logR_o = outer_grid(old_params)
    new_logR_i = inner_grid(logR_son, new_params)
    new_logR_o = outer_grid(new_params)
    return np.concatenate(
        [
            interp_extrap(old_logR_i, old_logu_i, new_logR_i),
            interp_extrap(old_logR_i, old_logT_i, new_logR_i),
            interp_extrap(old_logR_o, old_logu_o, new_logR_o),
            interp_extrap(old_logR_o, old_logT_o, new_logR_o),
            np.array([logR_son, lambda0], dtype=float),
        ]
    )


def combined_profile_arrays(x: np.ndarray, params: TwoDomainParams) -> dict[str, np.ndarray]:
    logu_i, logT_i, logu_o, logT_o, logR_son, lambda0 = unpack_two_domain(x, params)
    logR_i = inner_grid(logR_son, params)
    logR_o = outer_grid(params)
    logR = np.concatenate([logR_i, logR_o[1:]])
    logu = np.concatenate([logu_i, logu_o[1:]])
    logT = np.concatenate([logT_i, logT_o[1:]])
    H_over_R = []
    lnOmega = []
    for lr, lu, lt in zip(logR, logu, logT):
        state = algebraic_state(float(lr), float(lu), float(lt), lambda0, params.physics)
        H_over_R.append(float(state.H_over_R))
        lnOmega.append(float(np.log(state.Omega / state.Omega_K)))
    return {
        "logR": logR,
        "logu": logu,
        "logT": logT,
        "H_over_R": np.asarray(H_over_R, dtype=float),
        "lnOmega": np.asarray(lnOmega, dtype=float),
    }


def compare_to_reference(x: np.ndarray, params: TwoDomainParams, ref_arrays: dict[str, np.ndarray]) -> dict[str, float]:
    arrays = combined_profile_arrays(x, params)
    logR = arrays["logR"]
    mask = (logR >= ref_arrays["logR"][0]) & (logR <= ref_arrays["logR"][-1])
    if not np.any(mask):
        return {
            "max_dlogu": np.nan,
            "rms_dlogu": np.nan,
            "max_dlogT": np.nan,
            "rms_dlogT": np.nan,
            "max_dHR": np.nan,
            "max_dlnOmega": np.nan,
        }
    x_eval = logR[mask]
    diffs = {}
    for key in ("logu", "logT", "H_over_R", "lnOmega"):
        ref_values = np.interp(x_eval, ref_arrays["logR"], ref_arrays[key])
        diff = arrays[key][mask] - ref_values
        diffs[key] = diff
    return {
        "max_dlogu": float(np.max(np.abs(diffs["logu"]))),
        "rms_dlogu": float(np.sqrt(np.mean(diffs["logu"] ** 2))),
        "max_dlogT": float(np.max(np.abs(diffs["logT"]))),
        "rms_dlogT": float(np.sqrt(np.mean(diffs["logT"] ** 2))),
        "max_dHR": float(np.max(np.abs(diffs["H_over_R"]))),
        "max_dlnOmega": float(np.max(np.abs(diffs["lnOmega"]))),
    }


def worst_interval(x: np.ndarray, params: TwoDomainParams) -> dict[str, object]:
    logu_i, logT_i, logu_o, logT_o, logR_son, lambda0 = unpack_two_domain(x, params)
    logR_i = inner_grid(logR_son, params)
    logR_o = outer_grid(params)
    rows = []
    for domain, block, logR in (
        ("inner", interval_blocks(logu_i, logT_i, logR_i, lambda0, params), logR_i),
        ("outer", interval_blocks(logu_o, logT_o, logR_o, lambda0, params), logR_o),
    ):
        for idx, residual in enumerate(block):
            component = "R" if abs(float(residual[0])) >= abs(float(residual[1])) else "E"
            rows.append(
                {
                    "domain": domain,
                    "idx": idx,
                    "R_mid_rg": float(np.exp(0.5 * (logR[idx] + logR[idx + 1])) / params.r_g),
                    "component": component,
                    "value": float(residual[0] if component == "R" else residual[1]),
                    "abs_max": float(np.max(np.abs(residual))),
                }
            )
    return max(rows, key=lambda row: row["abs_max"])


def save_checkpoint(label: str, row: dict[str, object]) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {key: value for key, value in row.items() if key != "x"}
    np.savez_compressed(
        CHECKPOINT_DIR / f"{label}_0p90277664.npz",
        x=np.asarray(row["x"], dtype=float),
        row_json=np.array(row_json(payload)),
    )


def build_final_row(
    label: str,
    group: str,
    note: str,
    x: np.ndarray,
    params: TwoDomainParams,
    result,
    ref_row: dict[str, object],
    ref_arrays: dict[str, np.ndarray],
    source_label: str,
) -> dict[str, object]:
    row = audit_row(label, x, params, result)
    comparison = compare_to_reference(x, params, ref_arrays)
    worst = worst_interval(x, params)
    row.update(
        {
            "group": group,
            "note": note,
            "source_label": source_label,
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
        }
    )
    return row


def stage_row(label: str, stage: str, x: np.ndarray, params: TwoDomainParams, result, source_label: str) -> dict[str, object]:
    row = audit_row(f"{label}_{stage}", x, params, result)
    return {
        "label": label,
        "stage": stage,
        "source_label": source_label,
        "selected": row["selected_max"],
        "physical": row["physical_active"],
        "dominant": row["dominant"],
        "inner_R": row["inner_R"],
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


def skipped_final_row(spec: ValidationSpec, source_row: dict[str, object]) -> dict[str, object]:
    return {
        "label": spec.label,
        "group": spec.group,
        "note": f"skipped because source {spec.source_label} physical={float(source_row['physical_active']):.3e} exceeds {CHAIN_SOURCE_PHYSICAL_LIMIT:.1e}",
        "source_label": spec.source_label,
        "far_closure": "pressure_supported",
        "ratio": source_row["ratio"],
        "R_match_rg": R_MATCH_RG,
        "R_far_rg": spec.R_far_rg,
        "n_inner": spec.n_inner,
        "n_outer": spec.n_outer,
        "selected_max": np.nan,
        "physical_active": np.nan,
        "dominant": "skipped",
        "inner_R": np.nan,
        "inner_E": np.nan,
        "outer_R": np.nan,
        "outer_E": np.nan,
        "interface": np.nan,
        "far_omega": np.nan,
        "far_energy": np.nan,
        "D": np.nan,
        "C1": np.nan,
        "C2": np.nan,
        "K": np.nan,
        "Rson_rg": np.nan,
        "lambda0": np.nan,
        "int_adv": np.nan,
        "max_HR": np.nan,
        "far_HR": np.nan,
        "far_thin_omega": np.nan,
        "far_pressure_target": np.nan,
        "far_pressure_residual": np.nan,
        "delta_Rson_rg": np.nan,
        "delta_lambda0": np.nan,
        "delta_int_adv": np.nan,
        "max_dlogu_ref": np.nan,
        "rms_dlogu_ref": np.nan,
        "max_dlogT_ref": np.nan,
        "rms_dlogT_ref": np.nan,
        "max_dHR_ref": np.nan,
        "max_dlnOmega_ref": np.nan,
        "worst_domain": "skipped",
        "worst_i": -1,
        "worst_R_mid_rg": np.nan,
        "worst_component": "skipped",
        "worst_interval_abs": np.nan,
        "nfev": 0,
        "success": False,
    }


def skipped_stage_row(spec: ValidationSpec, source_row: dict[str, object]) -> dict[str, object]:
    return {
        "label": spec.label,
        "stage": "skipped",
        "source_label": spec.source_label,
        "physical": np.nan,
        "dominant": "skipped",
        "inner_R": np.nan,
        "outer_R": np.nan,
        "far_closure_omega": np.nan,
        "far_energy": np.nan,
        "Rson_rg": np.nan,
        "lambda0": np.nan,
        "int_adv": np.nan,
        "nfev": 0,
        "success": False,
        "message": f"source physical={float(source_row['physical_active']):.3e} exceeds {CHAIN_SOURCE_PHYSICAL_LIMIT:.1e}",
    }


def solve_validation_case(
    spec: ValidationSpec,
    seed_x: np.ndarray,
    seed_params: TwoDomainParams,
    target_params: TwoDomainParams,
    ref_row: dict[str, object],
    ref_arrays: dict[str, np.ndarray],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    seed = remap_state(seed_x, seed_params, target_params)
    stages = [stage_row(spec.label, "seed", seed, target_params, None, spec.source_label)]
    print(
        f"{spec.label} seed physical={stages[-1]['physical']:.3e} dominant={stages[-1]['dominant']} "
        f"N=({spec.n_inner},{spec.n_outer}) Rfar={spec.R_far_rg:.3g}",
        flush=True,
    )

    locked = solve_two_domain(seed, target_params, locked_bounds(target_params, seed), MAX_NFEV_LOCKED)
    stages.append(stage_row(spec.label, "locked", locked.x, target_params, locked, spec.source_label))
    print(
        f"{spec.label} locked physical={stages[-1]['physical']:.3e} dominant={stages[-1]['dominant']} nfev={locked.nfev}",
        flush=True,
    )

    released = solve_two_domain(locked.x, target_params, state_bounds_two_domain(target_params), MAX_NFEV_RELEASE)
    stages.append(stage_row(spec.label, "release", released.x, target_params, released, spec.source_label))
    print(
        f"{spec.label} release physical={stages[-1]['physical']:.3e} dominant={stages[-1]['dominant']} nfev={released.nfev}",
        flush=True,
    )

    polished = solve_two_domain(released.x, target_params, state_bounds_two_domain(target_params), MAX_NFEV_POLISH)
    stages.append(stage_row(spec.label, "polish", polished.x, target_params, polished, spec.source_label))
    print(
        f"{spec.label} polish physical={stages[-1]['physical']:.3e} dominant={stages[-1]['dominant']} nfev={polished.nfev}",
        flush=True,
    )

    final = build_final_row(spec.label, spec.group, spec.note, polished.x, target_params, polished, ref_row, ref_arrays, spec.source_label)
    save_checkpoint(spec.label, final)
    return final, stages


def write_table(final_rows: list[dict[str, object]], stage_rows: list[dict[str, object]]) -> None:
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Two-Domain Pressure-Supported Mesh Validation",
        "",
        "Generated by `scripts/run_transonic_two_domain_mesh_validation.py`.",
        "",
        "Reference state is `Rfar1e5_pressure_supported_polish_0p90277664.npz` from the two-domain outer-extension run. All validation rows use the pressure-supported far rotation closure and retain the thin far thermal condition.",
        "",
        "## Final Rows",
        "",
        "| label | group | N inner | N outer | Rfar/rg | physical | dominant | inner R | outer R | far closure omega | far thin omega | pressure target | pressure residual | far energy | Rson/rg | dRson | lambda0 | dlambda | int adv | dint adv | max H/R | far H/R | max dlogu | rms dlogu | max dlogT | rms dlogT | max dH/R | max dlnOmega | worst domain | worst R/rg | worst comp | worst interval | nfev | success | note |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|:---:|---:|---:|:---:|---|",
    ]
    for row in final_rows:
        lines.append(
            "| {label} | {group} | {n_inner} | {n_outer} | {R_far_rg} | {physical_active} | {dominant} | "
            "{inner_R} | {outer_R} | {far_omega} | {far_thin_omega} | {far_pressure_target} | "
            "{far_pressure_residual} | {far_energy} | {Rson_rg} | {delta_Rson_rg} | {lambda0} | "
            "{delta_lambda0} | {int_adv} | {delta_int_adv} | {max_HR} | {far_HR} | {max_dlogu_ref} | "
            "{rms_dlogu_ref} | {max_dlogT_ref} | {rms_dlogT_ref} | {max_dHR_ref} | {max_dlnOmega_ref} | "
            "{worst_domain} | {worst_R_mid_rg} | {worst_component} | {worst_interval_abs} | {nfev} | "
            "{success} | {note} |".format(
                label=row["label"],
                group=row["group"],
                n_inner=row["n_inner"],
                n_outer=row["n_outer"],
                R_far_rg=fmt(float(row["R_far_rg"])),
                physical_active=fmt(float(row["physical_active"])),
                dominant=row["dominant"],
                inner_R=fmt(float(row["inner_R"])),
                outer_R=fmt(float(row["outer_R"])),
                far_omega=fmt(float(row["far_omega"])),
                far_thin_omega=fmt(float(row["far_thin_omega"])),
                far_pressure_target=fmt(float(row["far_pressure_target"])),
                far_pressure_residual=fmt(float(row["far_pressure_residual"])),
                far_energy=fmt(float(row["far_energy"])),
                Rson_rg=fmt(float(row["Rson_rg"])),
                delta_Rson_rg=fmt(float(row["delta_Rson_rg"])),
                lambda0=fmt(float(row["lambda0"])),
                delta_lambda0=fmt(float(row["delta_lambda0"])),
                int_adv=fmt(float(row["int_adv"])),
                delta_int_adv=fmt(float(row["delta_int_adv"])),
                max_HR=fmt(float(row["max_HR"])),
                far_HR=fmt(float(row["far_HR"])),
                max_dlogu_ref=fmt(float(row["max_dlogu_ref"])),
                rms_dlogu_ref=fmt(float(row["rms_dlogu_ref"])),
                max_dlogT_ref=fmt(float(row["max_dlogT_ref"])),
                rms_dlogT_ref=fmt(float(row["rms_dlogT_ref"])),
                max_dHR_ref=fmt(float(row["max_dHR_ref"])),
                max_dlnOmega_ref=fmt(float(row["max_dlnOmega_ref"])),
                worst_domain=row["worst_domain"],
                worst_R_mid_rg=fmt(float(row["worst_R_mid_rg"])),
                worst_component=row["worst_component"],
                worst_interval_abs=fmt(float(row["worst_interval_abs"])),
                nfev=row["nfev"],
                success="yes" if row["success"] else "no",
                note=str(row["note"]),
            )
        )
    lines.extend(
        [
            "",
            "## Stage Rows",
            "",
            "| label | stage | source | physical | dominant | inner R | outer R | far closure omega | far energy | Rson/rg | lambda0 | int adv | nfev | success | message |",
            "|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|",
        ]
    )
    for row in stage_rows:
        lines.append(
            "| {label} | {stage} | {source_label} | {physical} | {dominant} | {inner_R} | {outer_R} | "
            "{far_closure_omega} | {far_energy} | {Rson_rg} | {lambda0} | {int_adv} | {nfev} | {success} | {message} |".format(
                label=row["label"],
                stage=row["stage"],
                source_label=row["source_label"],
                physical=fmt(float(row["physical"])),
                dominant=row["dominant"],
                inner_R=fmt(float(row["inner_R"])),
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
    x_ref, row_ref_source = load_checkpoint(REFERENCE_CHECKPOINT)
    ratio = float(row_ref_source["ratio"])
    ref_params = make_params(
        fiducial,
        ratio,
        mdot_edd,
        int(row_ref_source["n_inner"]),
        int(row_ref_source["n_outer"]),
        float(row_ref_source["R_far_rg"]),
    )
    ref_row = build_final_row(
        "reference",
        "reference",
        "loaded pressure-supported two-domain root",
        x_ref,
        ref_params,
        None,
        {
            "Rson_rg": float(row_ref_source["Rson_rg"]),
            "lambda0": float(row_ref_source["lambda0"]),
            "int_adv": float(row_ref_source["int_adv"]),
        },
        combined_profile_arrays(x_ref, ref_params),
        "source_checkpoint",
    )
    ref_arrays = combined_profile_arrays(x_ref, ref_params)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    for old_path in CHECKPOINT_DIR.glob("*.npz"):
        old_path.unlink()
    save_checkpoint("reference", ref_row)

    states: dict[str, tuple[np.ndarray, TwoDomainParams]] = {"reference": (x_ref, ref_params)}
    final_by_label: dict[str, dict[str, object]] = {"reference": ref_row}
    final_rows = [ref_row]
    stage_rows: list[dict[str, object]] = [stage_row("reference", "loaded", x_ref, ref_params, None, "source_checkpoint")]
    write_table(final_rows, stage_rows)
    print(
        f"reference physical={ref_row['physical_active']:.3e} dominant={ref_row['dominant']} "
        f"N=({ref_row['n_inner']},{ref_row['n_outer']}) Rfar={ref_row['R_far_rg']:.3g}",
        flush=True,
    )

    for spec in SPECS:
        source_final = final_by_label.get(spec.source_label)
        if source_final is not None and float(source_final["physical_active"]) > CHAIN_SOURCE_PHYSICAL_LIMIT:
            final = skipped_final_row(spec, source_final)
            final_rows.append(final)
            stage_rows.append(skipped_stage_row(spec, source_final))
            final_by_label[spec.label] = final
            write_table(final_rows, stage_rows)
            print(
                f"{spec.label} skipped because source {spec.source_label} physical={float(source_final['physical_active']):.3e}",
                flush=True,
            )
            continue
        source_x, source_params = states[spec.source_label]
        target_params = make_params(fiducial, ratio, mdot_edd, spec.n_inner, spec.n_outer, spec.R_far_rg)
        final, stages = solve_validation_case(spec, source_x, source_params, target_params, ref_row, ref_arrays)
        states[spec.label] = (np.asarray(final["x"], dtype=float), target_params)
        final_rows.append(final)
        final_by_label[spec.label] = final
        stage_rows.extend(stages)
        write_table(final_rows, stage_rows)

    print(f"wrote {TABLE_OUTPUT}")


if __name__ == "__main__":
    main()
