"""Point-6 polish/resume and N96 robustness audit for the transonic branch."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    TransonicSlimParams,
    pseudo_arclength_step,
    remap_profile_to_new_sonic_grid,
    solve_square_transonic_polish,
    square_collocation_residual,
    transonic_profile_from_state_vector,
)
from imri_qpe.layer3_minidisk_1d.transonic_collocation import (
    profile_from_state_vector,
    residual_audit_from_state_vector,
)
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints" / "transonic_n64_arclength_adaptive"
CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints" / "transonic_point6_resume"
TABLE_OUTPUT = ROOT / "outputs" / "tables" / "transonic_point6_polish_resume.md"

RESIDUAL_TOL = 3.0e-4
POLISH_TOL = 1.0e-6
R_OUT_RG = 3000.0
SOURCE_SEEDS = (
    "032_arc_x1_0p96532914.npz",
    "033_arc_x1_0p99629052.npz",
)
POLISH_TARGETS = (
    "030_arc_x1_0p90277664.npz",
    "032_arc_x1_0p96532914.npz",
    "033_arc_x1_0p99629052.npz",
)
RESUME_STEPS = (
    ("resume", 1.0),
    ("resume", 0.25),
    ("resume", 0.35),
    ("resume", 0.08),
    ("resume", 0.05),
    ("target_1.2_probe", 1.0),
)
N96_SPOT_LABELS = ("resume_1", "resume_3", "resume_5")
N96_INDEPENDENT_STEPS = (1.0, 0.5, 0.25)


def fmt(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.4g}"


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return value if np.isfinite(value) else str(value)
    return value


def params_for(fiducial: FiducialParams, mdot_edd: float, ratio: float, n_nodes: int, residual_tol: float) -> TransonicSlimParams:
    return TransonicSlimParams(
        M2_g=fiducial.M2_g,
        Mdot_g_s=ratio * mdot_edd,
        alpha=fiducial.alpha_cool,
        n_nodes=n_nodes,
        R_out_rg=R_OUT_RG,
        residual_tol=residual_tol,
        max_nfev=120,
    )


def active_max(audit) -> float:
    return max(
        abs(audit.interval_radial_max),
        abs(audit.interval_energy_max),
        abs(audit.outer_omega),
        abs(audit.outer_energy),
        abs(audit.sonic_D),
        abs(audit.sonic_K),
    )


def active_dominant(audit) -> str:
    values = {
        "interval_R": abs(audit.interval_radial_max),
        "interval_E": abs(audit.interval_energy_max),
        "outer_Omega": abs(audit.outer_omega),
        "outer_E": abs(audit.outer_energy),
        "D": abs(audit.sonic_D),
        "K": abs(audit.sonic_K),
    }
    return max(values, key=values.get)


def row_json(row: dict[str, object]) -> str:
    return json.dumps({key: json_safe(value) for key, value in row.items()}, sort_keys=True)


def checkpoint_name(index: int, label: str, ratio: float) -> str:
    return f"{index:03d}_{label}_{ratio:.8f}".replace(".", "p") + ".npz"


def save_checkpoint(index: int, label: str, record: dict[str, object], row: dict[str, object]) -> Path:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIR / checkpoint_name(index, label, float(record["ratio"]))
    params = record["params"]
    np.savez_compressed(
        path,
        index=np.array(index),
        label=np.array(label),
        ratio=np.array(record["ratio"]),
        n_nodes=np.array(params.n_nodes),
        R_out_rg=np.array(params.R_out_rg),
        residual_tol=np.array(params.residual_tol),
        z=np.asarray(record["z"], dtype=float),
        row_json=np.array(row_json(row)),
    )
    return path


def load_record(path: Path, fiducial: FiducialParams, mdot_edd: float, residual_tol: float = RESIDUAL_TOL) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as data:
        z = np.asarray(data["z"], dtype=float)
        ratio = float(data["ratio"])
    params = params_for(fiducial, mdot_edd, ratio, 64, residual_tol)
    profile = profile_from_state_vector(z, params)
    return {"ratio": ratio, "z": z, "params": params, "profile": profile}


def profile_row(stage: str, label: str, record: dict[str, object], message: str) -> dict[str, object]:
    audit = residual_audit_from_state_vector(record["z"], record["params"])
    profile = record["profile"]
    max_residual = active_max(audit)
    return {
        "stage": stage,
        "label": label,
        "ratio": float(record["ratio"]),
        "target": np.nan,
        "accepted": max_residual <= RESIDUAL_TOL,
        "physical": True,
        "dominant": active_dominant(audit),
        "max_residual": max_residual,
        "D": audit.sonic_D,
        "K": audit.sonic_K,
        "C1": audit.sonic_C1,
        "C2": audit.sonic_C2,
        "max_HR": float(np.max(profile.H_over_R)),
        "int_adv": profile.integrated_advective_fraction,
        "corrector": "source",
        "iterations": 0,
        "nfev": 0,
        "cuts": 0,
        "cond": np.nan,
        "n96_active": np.nan,
        "n96_dHR": np.nan,
        "n96_dxi": np.nan,
        "message": message,
    }


def polish_rows(fiducial: FiducialParams, mdot_edd: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for filename in POLISH_TARGETS:
        record = load_record(SOURCE_CHECKPOINT_DIR / filename, fiducial, mdot_edd, residual_tol=POLISH_TOL)
        before = float(np.max(np.abs(square_collocation_residual(record["z"], record["params"], pivot="K"))))
        result = solve_square_transonic_polish(
            record["params"],
            record["z"],
            pivot="K",
            method="newton",
            max_iter=10,
            max_nfev=500,
            residual_tol=POLISH_TOL,
            jacobian_rel_step=3.0e-5,
        )
        rows.append(
            {
                "stage": "polish",
                "label": filename,
                "ratio": float(record["ratio"]),
                "target": np.nan,
                "accepted": result.final_square_max_residual <= POLISH_TOL,
                "physical": result.result.status.physically_valid,
                "dominant": "square",
                "max_residual": result.final_square_max_residual,
                "D": result.result.residual_audit.sonic_D,
                "K": result.result.residual_audit.sonic_K,
                "C1": result.result.residual_audit.sonic_C1,
                "C2": result.result.residual_audit.sonic_C2,
                "max_HR": float(np.max(result.result.profile.H_over_R)),
                "int_adv": result.result.profile.integrated_advective_fraction,
                "corrector": "fixed_mdot_newton",
                "iterations": result.iterations,
                "nfev": result.result.nfev,
                "cuts": result.line_search_reductions,
                "cond": np.nan,
                "n96_active": np.nan,
                "n96_dHR": np.nan,
                "n96_dxi": np.nan,
                "message": f"square {before:.3e}->{result.final_square_max_residual:.3e}; {result.result.message}",
            }
        )
        print(
            f"polish {filename} ratio={record['ratio']:.6g} square={before:.3e}->{result.final_square_max_residual:.3e}",
            flush=True,
        )
    return rows


def n96_spot_row(label: str, record: dict[str, object], fiducial: FiducialParams, mdot_edd: float) -> dict[str, object]:
    ratio = float(record["ratio"])
    params96 = params_for(fiducial, mdot_edd, ratio, 96, RESIDUAL_TOL)
    z96 = remap_profile_to_new_sonic_grid(record["profile"], params96)
    profile96 = transonic_profile_from_state_vector(z96, params96)
    audit96 = residual_audit_from_state_vector(z96, params96)
    profile64 = record["profile"]
    logR64 = np.log(profile64.R)
    logR96 = np.log(profile96.R)
    HR64_on_96 = np.interp(logR96, logR64, profile64.H_over_R)
    xi64_on_96 = np.interp(logR96, logR64, profile64.xi_eff)
    return {
        "stage": "n96_spot",
        "label": label,
        "ratio": ratio,
        "target": np.nan,
        "accepted": active_max(audit96) <= RESIDUAL_TOL,
        "physical": bool(active_max(audit96) <= RESIDUAL_TOL and np.max(profile96.H_over_R) < 0.25),
        "dominant": active_dominant(audit96),
        "max_residual": active_max(audit96),
        "D": audit96.sonic_D,
        "K": audit96.sonic_K,
        "C1": audit96.sonic_C1,
        "C2": audit96.sonic_C2,
        "max_HR": float(np.max(profile96.H_over_R)),
        "int_adv": profile96.integrated_advective_fraction,
        "corrector": "remap_check",
        "iterations": 0,
        "nfev": 0,
        "cuts": 0,
        "cond": np.nan,
        "n96_active": active_max(audit96),
        "n96_dHR": float(np.max(np.abs(profile96.H_over_R - HR64_on_96))),
        "n96_dxi": float(np.max(np.abs(profile96.xi_eff - xi64_on_96))),
        "message": "N96 remap residual/profile agreement check; not an independently polished N96 solve",
    }


def n96_independent_rows(fiducial: FiducialParams, mdot_edd: float) -> list[dict[str, object]]:
    records = [load_record(SOURCE_CHECKPOINT_DIR / filename, fiducial, mdot_edd) for filename in SOURCE_SEEDS]
    base96 = params_for(fiducial, mdot_edd, 1.0, 96, RESIDUAL_TOL)
    rows: list[dict[str, object]] = []
    for step_multiplier in N96_INDEPENDENT_STEPS:
        result = pseudo_arclength_step(
            base96,
            mdot_edd,
            records[0]["profile"],
            float(records[0]["ratio"]),
            records[1]["profile"],
            float(records[1]["ratio"]),
            step_multiplier=step_multiplier,
            mdot_ratio_bounds=(0.25, 4.0),
            max_nfev=120,
            residual_tol=RESIDUAL_TOL,
            arclength_weight=1.0,
            residual_mode="square",
            sonic_pivot="K",
            metric_mode="blockwise",
            tangent_mode="jacobian",
            corrector_method="bordered_newton",
            bordered_max_iter=80,
        )
        rows.append(
            {
                "stage": "n96_independent",
                "label": f"step_{step_multiplier:g}",
                "ratio": result.mdot_ratio,
                "target": np.nan,
                "accepted": bool(result.status.physically_valid and result.max_residual <= RESIDUAL_TOL),
                "physical": result.status.physically_valid,
                "dominant": active_dominant(result.residual_audit),
                "max_residual": result.max_residual,
                "D": result.residual_audit.sonic_D,
                "K": result.residual_audit.sonic_K,
                "C1": result.residual_audit.sonic_C1,
                "C2": result.residual_audit.sonic_C2,
                "max_HR": float(np.max(result.profile.H_over_R)),
                "int_adv": result.profile.integrated_advective_fraction,
                "corrector": result.corrector_method,
                "iterations": result.corrector_iterations,
                "nfev": result.nfev,
                "cuts": result.line_search_reductions,
                "cond": result.condition_estimate,
                "n96_active": result.max_residual,
                "n96_dHR": np.nan,
                "n96_dxi": np.nan,
                "message": result.message,
            }
        )
        print(
            f"N96 independent step={step_multiplier:g} got={result.mdot_ratio:.8g} "
            f"accepted={rows[-1]['accepted']} max={result.max_residual:.3e} cond={result.condition_estimate:.3e}",
            flush=True,
        )
    return rows


def resume_rows(fiducial: FiducialParams, mdot_edd: float) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    rows: list[dict[str, object]] = []
    records = [load_record(SOURCE_CHECKPOINT_DIR / filename, fiducial, mdot_edd) for filename in SOURCE_SEEDS]
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    for old_path in CHECKPOINT_DIR.glob("*.npz"):
        old_path.unlink()
    for idx, record in enumerate(records):
        row = profile_row("source", SOURCE_SEEDS[idx], record, "loaded source anchor")
        rows.append(row)
        save_checkpoint(idx, "source", record, row)

    base = params_for(fiducial, mdot_edd, 1.0, 64, RESIDUAL_TOL)
    accepted: dict[str, dict[str, object]] = {}
    previous, current = records
    checkpoint_index = len(records)
    resume_index = 0
    for label, step_multiplier in RESUME_STEPS:
        target = 1.2 if label == "target_1.2_probe" else np.nan
        result = pseudo_arclength_step(
            base,
            mdot_edd,
            previous["profile"],
            float(previous["ratio"]),
            current["profile"],
            float(current["ratio"]),
            step_multiplier=step_multiplier,
            mdot_ratio_bounds=(0.25, 4.0),
            max_nfev=120,
            residual_tol=RESIDUAL_TOL,
            arclength_weight=1.0,
            residual_mode="square",
            sonic_pivot="K",
            metric_mode="blockwise",
            tangent_mode="jacobian",
            corrector_method="bordered_newton",
            bordered_max_iter=80,
        )
        accepted_flag = bool(
            result.status.physically_valid
            and result.max_residual <= RESIDUAL_TOL
            and result.mdot_ratio > float(current["ratio"]) * 1.00005
        )
        row = {
            "stage": "resume",
            "label": f"{label}_{resume_index + 1}",
            "ratio": result.mdot_ratio,
            "target": target,
            "accepted": accepted_flag,
            "physical": result.status.physically_valid,
            "dominant": active_dominant(result.residual_audit),
            "max_residual": result.max_residual,
            "D": result.residual_audit.sonic_D,
            "K": result.residual_audit.sonic_K,
            "C1": result.residual_audit.sonic_C1,
            "C2": result.residual_audit.sonic_C2,
            "max_HR": float(np.max(result.profile.H_over_R)),
            "int_adv": result.profile.integrated_advective_fraction,
            "corrector": result.corrector_method,
            "iterations": result.corrector_iterations,
            "nfev": result.nfev,
            "cuts": result.line_search_reductions,
            "cond": result.condition_estimate,
            "n96_active": np.nan,
            "n96_dHR": np.nan,
            "n96_dxi": np.nan,
            "message": result.message,
        }
        rows.append(row)
        print(
            f"{row['label']} step={step_multiplier:g} got={result.mdot_ratio:.8g} "
            f"accepted={accepted_flag} max={result.max_residual:.3e} cond={result.condition_estimate:.3e}",
            flush=True,
        )
        if not accepted_flag:
            break
        resume_index += 1
        record = {"ratio": result.mdot_ratio, "z": result.z, "params": result.params, "profile": result.profile}
        accepted[row["label"]] = record
        save_checkpoint(checkpoint_index, row["label"], record, row)
        checkpoint_index += 1
        previous, current = current, record
    return rows, accepted


def write_table(rows: list[dict[str, object]]) -> None:
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Point-6 Polish/Resume and N96 Robustness Audit",
        "",
        "Generated by `scripts/run_transonic_point6_polish_resume.py`.",
        "",
        f"Active square tolerance is `{RESIDUAL_TOL:g}` for resume/N96 checks; fixed-Mdot polish target is `{POLISH_TOL:g}`.",
        "",
        "| stage | label | target | Mdot/Edd | accepted | physical | dominant | max residual | D | K | C1 | C2 | max H/R | int adv | corrector | iter | cuts | cond | N96 active | N96 dH/R | N96 dxi | nfev | message |",
        "|---|---|---:|---:|:---:|:---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {stage} | {label} | {target} | {ratio} | {accepted} | {physical} | {dominant} | "
            "{max_residual} | {D} | {K} | {C1} | {C2} | {max_HR} | {int_adv} | {corrector} | "
            "{iterations} | {cuts} | {cond} | {n96_active} | {n96_dHR} | {n96_dxi} | {nfev} | {message} |".format(
                stage=row["stage"],
                label=row["label"],
                target=fmt(float(row["target"])),
                ratio=fmt(float(row["ratio"])),
                accepted="yes" if row["accepted"] else "no",
                physical="yes" if row["physical"] else "no",
                dominant=row["dominant"],
                max_residual=fmt(float(row["max_residual"])),
                D=fmt(float(row["D"])),
                K=fmt(float(row["K"])),
                C1=fmt(float(row["C1"])),
                C2=fmt(float(row["C2"])),
                max_HR=fmt(float(row["max_HR"])),
                int_adv=fmt(float(row["int_adv"])),
                corrector=row["corrector"],
                iterations=row["iterations"],
                cuts=row["cuts"],
                cond=fmt(float(row["cond"])),
                n96_active=fmt(float(row["n96_active"])),
                n96_dHR=fmt(float(row["n96_dHR"])),
                n96_dxi=fmt(float(row["n96_dxi"])),
                nfev=row["nfev"],
                message=str(row["message"]).replace("|", "/"),
            )
        )
    TABLE_OUTPUT.write_text("\n".join(lines) + "\n")


def main() -> None:
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)
    rows = polish_rows(fiducial, mdot_edd)
    resume, accepted = resume_rows(fiducial, mdot_edd)
    rows.extend(resume)
    for label in N96_SPOT_LABELS:
        if label in accepted:
            rows.append(n96_spot_row(label, accepted[label], fiducial, mdot_edd))
    rows.extend(n96_independent_rows(fiducial, mdot_edd))
    write_table(rows)
    print(f"wrote {TABLE_OUTPUT}")


if __name__ == "__main__":
    main()
