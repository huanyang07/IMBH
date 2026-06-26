"""Anchored square-residual continuation just above Eddington."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from imri_qpe.layer3_minidisk_1d import TransonicSlimParams, pseudo_arclength_step
from imri_qpe.layer3_minidisk_1d.transonic_collocation import (
    _status_from_profile,
    profile_from_state_vector,
    replace_mdot,
    residual_audit_from_state_vector,
)
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints" / "transonic_n64_arclength_adaptive"
CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints" / "transonic_square_anchor_continuation"
TABLE_OUTPUT = ROOT / "outputs" / "tables" / "transonic_square_anchor_continuation.md"

RESIDUAL_TOL = 3.0e-4
R_OUT_RG = 3000.0
MAX_NFEV = 100
MIN_ACCEPTED_RATIO_STEP = 1.0e-4
ANCHORS = (
    1.004,
    1.008,
    1.012,
    1.013,
    1.014,
    1.015,
    1.016,
    1.018,
    1.020,
    1.024,
    1.028,
    1.032,
    1.036,
    1.040,
    1.050,
    1.060,
    1.080,
    1.100,
)
SOURCE_SEEDS = (
    "032_arc_x1_0p96532914.npz",
    "033_arc_x1_0p99629052.npz",
)


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


def row_to_json(row: dict[str, object]) -> str:
    return json.dumps({key: json_safe(value) for key, value in row.items()}, sort_keys=True)


def dominant_block(audit) -> str:
    values = {
        "interval_R": abs(audit.interval_radial_max),
        "interval_E": abs(audit.interval_energy_max),
        "outer_Omega": abs(audit.outer_omega),
        "outer_E": abs(audit.outer_energy),
        "D": abs(audit.sonic_D),
        "K": abs(audit.sonic_K),
    }
    return max(values, key=values.get)


def accepted(status, max_residual: float) -> bool:
    return bool(
        np.isfinite(max_residual)
        and max_residual <= RESIDUAL_TOL
        and status.optimizer_acceptable
        and status.equations_converged
        and status.sonic_regular
        and status.active_bounds_clear
        and status.positive_state
        and status.outer_thin
    )


def base_params(fiducial: FiducialParams, mdot_edd: float, ratio: float = 1.0) -> TransonicSlimParams:
    return TransonicSlimParams(
        M2_g=fiducial.M2_g,
        Mdot_g_s=ratio * mdot_edd,
        alpha=fiducial.alpha_cool,
        n_nodes=64,
        R_out_rg=R_OUT_RG,
        max_nfev=MAX_NFEV,
        residual_tol=RESIDUAL_TOL,
    )


def load_source_seed(path: Path, fiducial: FiducialParams, mdot_edd: float) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as data:
        z = np.asarray(data["z"], dtype=float)
        ratio = float(data["ratio"])
    params = base_params(fiducial, mdot_edd, ratio)
    profile = profile_from_state_vector(z, params)
    return {"label": "source", "ratio": ratio, "z": z, "params": params, "profile": profile}


def row_from_profile(record: dict[str, object], label: str, target: float = np.nan) -> dict[str, object]:
    profile = record["profile"]
    params = record["params"]
    full_audit = residual_audit_from_state_vector(record["z"], params)
    max_residual = max(
        abs(full_audit.interval_radial_max),
        abs(full_audit.interval_energy_max),
        abs(full_audit.outer_omega),
        abs(full_audit.outer_energy),
        abs(full_audit.sonic_D),
        abs(full_audit.sonic_K),
    )
    status = _status_from_profile(profile, full_audit, params, True, max_residual)
    return {
        "label": label,
        "target": target,
        "ratio": float(record["ratio"]),
        "predicted": np.nan,
        "accepted": accepted(status, max_residual),
        "physical": status.physically_valid,
        "equations": status.equations_converged,
        "sonic": status.sonic_regular,
        "dominant": dominant_block(full_audit),
        "max_HR": float(np.max(profile.H_over_R)),
        "int_adv": profile.integrated_advective_fraction,
        "max_residual": max_residual,
        "interval_R": full_audit.interval_radial_max,
        "interval_E": full_audit.interval_energy_max,
        "outer_Omega": full_audit.outer_omega,
        "outer_E": full_audit.outer_energy,
        "D": full_audit.sonic_D,
        "K": full_audit.sonic_K,
        "C1": full_audit.sonic_C1,
        "C2": full_audit.sonic_C2,
        "arc": np.nan,
        "step_multiplier": np.nan,
        "tangent_method": "source",
        "logu_frac": np.nan,
        "logT_frac": np.nan,
        "logR_son_frac": np.nan,
        "lambda0_frac": np.nan,
        "mu_frac": np.nan,
        "dmu_ds": np.nan,
        "nfev": 0,
        "corrector": "source",
        "iterations": 0,
        "cuts": 0,
        "cond": np.nan,
        "pred_err": np.nan,
        "message": "loaded source anchor",
    }


def row_from_result(result, target: float, step_multiplier: float) -> dict[str, object]:
    profile = result.profile
    audit = result.residual_audit
    status = result.status
    tangent = result.tangent_audit
    return {
        "label": "anchor",
        "target": target,
        "ratio": result.mdot_ratio,
        "predicted": result.predicted_mdot_ratio,
        "accepted": accepted(status, result.max_residual),
        "physical": status.physically_valid,
        "equations": status.equations_converged,
        "sonic": status.sonic_regular,
        "dominant": dominant_block(audit),
        "max_HR": float(np.max(profile.H_over_R)),
        "int_adv": profile.integrated_advective_fraction,
        "max_residual": result.max_residual,
        "interval_R": audit.interval_radial_max,
        "interval_E": audit.interval_energy_max,
        "outer_Omega": audit.outer_omega,
        "outer_E": audit.outer_energy,
        "D": audit.sonic_D,
        "K": audit.sonic_K,
        "C1": audit.sonic_C1,
        "C2": audit.sonic_C2,
        "arc": result.arclength_residual,
        "step_multiplier": step_multiplier,
        "tangent_method": tangent.method,
        "logu_frac": tangent.logu_fraction,
        "logT_frac": tangent.logT_fraction,
        "logR_son_frac": tangent.logR_son_fraction,
        "lambda0_frac": tangent.lambda0_fraction,
        "mu_frac": tangent.mu_fraction,
        "dmu_ds": tangent.dmu_ds,
        "nfev": result.nfev,
        "corrector": result.corrector_method,
        "iterations": result.corrector_iterations,
        "cuts": result.line_search_reductions,
        "cond": result.condition_estimate,
        "pred_err": result.predictor_correction_norm,
        "message": result.message,
    }


def write_table(rows: list[dict[str, object]]) -> None:
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Square Anchored Continuation Above Eddington",
        "",
        "Generated by `scripts/run_transonic_square_anchor_continuation.py`.",
        "",
        f"Accepted rows use `residual_tol={RESIDUAL_TOL:g}`, square sonic residuals, `sonic_pivot=K`, and checkpoints in `{CHECKPOINT_DIR}`.",
        "",
        "The tangent-fraction columns audit the metric-normalized tangent used by the predictor.",
        "",
        "| label | target | Mdot/Mdot_Edd | predicted | accepted | physical | equations | sonic | dominant | max H/R | int adv | max residual | interval R | interval E | outer Omega | outer E | D | K | C1 | C2 | arc residual | step x | tangent | corrector | iter | cuts | cond | pred err | logu frac | logT frac | logRson frac | lambda0 frac | mu frac | dmu/ds | nfev | message |",
        "|---|---:|---:|---:|:---:|:---:|:---:|:---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {target} | {ratio} | {predicted} | {accepted} | {physical} | {equations} | {sonic} | "
            "{dominant} | {max_HR} | {int_adv} | {max_residual} | {interval_R} | {interval_E} | "
            "{outer_Omega} | {outer_E} | {D} | {K} | {C1} | {C2} | {arc} | {step_multiplier} | "
            "{tangent_method} | {corrector} | {iterations} | {cuts} | {cond} | {pred_err} | "
            "{logu_frac} | {logT_frac} | {logR_son_frac} | {lambda0_frac} | "
            "{mu_frac} | {dmu_ds} | {nfev} | {message} |".format(
                label=row["label"],
                target=fmt(float(row["target"])),
                ratio=fmt(float(row["ratio"])),
                predicted=fmt(float(row["predicted"])),
                accepted="yes" if row["accepted"] else "no",
                physical="yes" if row["physical"] else "no",
                equations="yes" if row["equations"] else "no",
                sonic="yes" if row["sonic"] else "no",
                dominant=row["dominant"],
                max_HR=fmt(float(row["max_HR"])),
                int_adv=fmt(float(row["int_adv"])),
                max_residual=fmt(float(row["max_residual"])),
                interval_R=fmt(float(row["interval_R"])),
                interval_E=fmt(float(row["interval_E"])),
                outer_Omega=fmt(float(row["outer_Omega"])),
                outer_E=fmt(float(row["outer_E"])),
                D=fmt(float(row["D"])),
                K=fmt(float(row["K"])),
                C1=fmt(float(row["C1"])),
                C2=fmt(float(row["C2"])),
                arc=fmt(float(row["arc"])),
                step_multiplier=fmt(float(row["step_multiplier"])),
                tangent_method=row["tangent_method"],
                corrector=row["corrector"],
                iterations=row["iterations"],
                cuts=row["cuts"],
                cond=fmt(float(row["cond"])),
                pred_err=fmt(float(row["pred_err"])),
                logu_frac=fmt(float(row["logu_frac"])),
                logT_frac=fmt(float(row["logT_frac"])),
                logR_son_frac=fmt(float(row["logR_son_frac"])),
                lambda0_frac=fmt(float(row["lambda0_frac"])),
                mu_frac=fmt(float(row["mu_frac"])),
                dmu_ds=fmt(float(row["dmu_ds"])),
                nfev=row["nfev"],
                message=str(row["message"]).replace("|", "/"),
            )
        )
    TABLE_OUTPUT.write_text("\n".join(lines) + "\n")


def checkpoint_name(index: int, ratio: float) -> str:
    rate = f"{ratio:.8f}".replace(".", "p")
    return f"{index:03d}_anchor_{rate}.npz"


def save_checkpoint(index: int, record: dict[str, object], row: dict[str, object]) -> Path:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIR / checkpoint_name(index, float(record["ratio"]))
    params = record["params"]
    np.savez_compressed(
        path,
        index=np.array(index),
        label=np.array(row["label"]),
        ratio=np.array(record["ratio"]),
        n_nodes=np.array(params.n_nodes),
        R_out_rg=np.array(params.R_out_rg),
        residual_tol=np.array(params.residual_tol),
        z=np.asarray(record["z"], dtype=float),
        row_json=np.array(row_to_json(row)),
    )
    return path


def step_multiplier_to_target(previous_ratio: float, current_ratio: float, target_ratio: float) -> float:
    denominator = np.log(current_ratio / previous_ratio)
    if denominator <= 0.0:
        raise ValueError("current_ratio must exceed previous_ratio")
    return float(np.clip(np.log(target_ratio / current_ratio) / denominator, 0.05, 0.75))


def main() -> None:
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)
    records = [load_source_seed(SOURCE_CHECKPOINT_DIR / filename, fiducial, mdot_edd) for filename in SOURCE_SEEDS]
    rows = [row_from_profile(records[0], "source"), row_from_profile(records[1], "source")]
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    for old_path in CHECKPOINT_DIR.glob("*.npz"):
        old_path.unlink()
    save_checkpoint(0, records[0], rows[0])
    save_checkpoint(1, records[1], rows[1])

    previous = records[0]
    current = records[1]
    next_index = 2
    for target in ANCHORS:
        if float(current["ratio"]) >= target * 0.999:
            continue
        step_multiplier = step_multiplier_to_target(float(previous["ratio"]), float(current["ratio"]), target)
        result = pseudo_arclength_step(
            base_params(fiducial, mdot_edd),
            mdot_edd,
            previous["profile"],
            float(previous["ratio"]),
            current["profile"],
            float(current["ratio"]),
            step_multiplier=step_multiplier,
            mdot_ratio_bounds=(0.25, 1.5),
            max_nfev=MAX_NFEV,
            residual_tol=RESIDUAL_TOL,
            arclength_weight=1.0,
            residual_mode="square",
            sonic_pivot="K",
            metric_mode="blockwise",
            tangent_mode="jacobian",
        )
        row = row_from_result(result, target, step_multiplier)
        rows.append(row)
        print(
            f"target={target:.4g} got={result.mdot_ratio:.5g} accepted={row['accepted']} "
            f"max={result.max_residual:.3e} D={result.residual_audit.sonic_D:.3e} "
            f"K={result.residual_audit.sonic_K:.3e} corrector={result.corrector_method} nfev={result.nfev}",
            flush=True,
        )
        if row["accepted"] and result.mdot_ratio > float(current["ratio"]) * (1.0 + MIN_ACCEPTED_RATIO_STEP):
            new_record = {
                "label": "anchor",
                "ratio": result.mdot_ratio,
                "z": result.z,
                "params": result.params,
                "profile": result.profile,
            }
            save_checkpoint(next_index, new_record, row)
            next_index += 1
            previous, current = current, new_record
        else:
            break

    write_table(rows)
    print(f"wrote {TABLE_OUTPUT}")


if __name__ == "__main__":
    main()
