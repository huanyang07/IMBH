"""Restartable adaptive pseudo-arclength continuation for the N=64 branch."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    TransonicSlimParams,
    pseudo_arclength_step,
    remap_profile_to_new_sonic_grid,
    solve_low_mdot_transonic_homotopy,
    solve_transonic_outer_branch,
    transonic_profile_from_state_vector,
)
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot


ROOT = Path(__file__).resolve().parents[1]
TABLE_OUTPUT = ROOT / "outputs" / "tables" / "transonic_n64_arclength_adaptive.md"
CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints" / "transonic_n64_arclength_adaptive"

RESIDUAL_TOL = 3.0e-4
R_OUT_RG = 3000.0
TARGET_RATIO = 1.0
MAX_ARC_STEPS = 14
MAX_ARC_FAILURES = 5
MIN_STEP_MULTIPLIER = 0.0625
MIN_TANGENT_RATIO_GAP = 1.5e-2

WARMUP_RATIOS = (1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2)
N24_RADIUS_BRIDGE_RATIOS = (1.0e-2, 2.0e-2, 3.0e-2)
N48_BRIDGE_RATIOS = (5.0e-2, 7.0e-2, 0.1, 0.15, 0.2, 0.3)
N64_SEED_RATIOS = (0.3, 0.35)
SONIC_WEIGHTS = (0.3, 1.0, 3.0)
OUTER_WEIGHTS = (0.3, 0.7)


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


def row_from_json(payload: str) -> dict[str, object]:
    row = json.loads(payload)
    for key in (
        "ratio",
        "predicted_ratio",
        "max_HR",
        "int_adv",
        "max_residual",
        "interval_R",
        "outer_Omega",
        "D",
        "C1",
        "C2",
        "arc",
        "step_multiplier",
    ):
        value = row.get(key)
        if isinstance(value, str):
            row[key] = float(value)
    return row


def dominant_block(audit) -> str:
    values = {
        "interval_R": abs(audit.interval_radial_max),
        "interval_E": abs(audit.interval_energy_max),
        "outer_Omega": abs(audit.outer_omega),
        "outer_E": abs(audit.outer_energy),
        "D": abs(audit.sonic_D),
        "C1": abs(audit.sonic_C1),
        "C2": abs(audit.sonic_C2),
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


def base_params_for(fiducial: FiducialParams, mdot_edd: float, ratio: float) -> TransonicSlimParams:
    return TransonicSlimParams(
        M2_g=fiducial.M2_g,
        Mdot_g_s=ratio * mdot_edd,
        alpha=fiducial.alpha_cool,
        n_nodes=64,
        R_out_rg=R_OUT_RG,
        max_nfev=2400,
        residual_tol=RESIDUAL_TOL,
    )


def solve_fixed(
    *,
    M2_g: float,
    mdot_edd: float,
    alpha: float,
    ratio: float,
    n_nodes: int,
    R_out_rg: float,
    seed_profile,
    sonic_weight_sequence: tuple[float, ...],
    outer_weight_sequence: tuple[float, ...],
    max_nfev: int,
    max_nfev_per_stage: int,
):
    params = TransonicSlimParams(
        M2_g=M2_g,
        Mdot_g_s=ratio * mdot_edd,
        alpha=alpha,
        n_nodes=n_nodes,
        R_out_rg=R_out_rg,
        max_nfev=max_nfev,
        residual_tol=RESIDUAL_TOL,
    )
    guess = remap_profile_to_new_sonic_grid(seed_profile, params) if seed_profile is not None else None
    result = solve_low_mdot_transonic_homotopy(
        params,
        initial_guess=guess,
        max_nfev_per_stage=max_nfev_per_stage,
        final_max_nfev=max_nfev,
        sonic_weight_sequence=sonic_weight_sequence,
        outer_weight_sequence=outer_weight_sequence,
    )
    return params, result


def row_from_fixed(label: str, ratio: float, n_nodes: int, result, step_multiplier: float = np.nan) -> dict[str, object]:
    final = result.final_result
    profile = final.profile
    audit = final.residual_audit
    status = final.status
    return {
        "label": label,
        "N": n_nodes,
        "ratio": ratio,
        "predicted_ratio": np.nan,
        "accepted": accepted(status, final.max_residual),
        "physical": status.physically_valid,
        "equations": status.equations_converged,
        "sonic": status.sonic_regular,
        "dominant": dominant_block(audit),
        "max_HR": float(np.max(profile.H_over_R)),
        "int_adv": profile.integrated_advective_fraction,
        "max_residual": final.max_residual,
        "interval_R": audit.interval_radial_max,
        "outer_Omega": audit.outer_omega,
        "D": audit.sonic_D,
        "C1": audit.sonic_C1,
        "C2": audit.sonic_C2,
        "arc": np.nan,
        "step_multiplier": step_multiplier,
        "nfev": final.nfev,
        "message": final.message,
    }


def row_from_outer(label: str, ratio: float, result, step_multiplier: float = np.nan) -> dict[str, object]:
    profile = result.profile
    audit = result.residual_audit
    status = result.status
    return {
        "label": label,
        "N": len(profile.R),
        "ratio": ratio,
        "predicted_ratio": np.nan,
        "accepted": accepted(status, result.max_residual),
        "physical": status.physically_valid,
        "equations": status.equations_converged,
        "sonic": status.sonic_regular,
        "dominant": dominant_block(audit),
        "max_HR": float(np.max(profile.H_over_R)),
        "int_adv": profile.integrated_advective_fraction,
        "max_residual": result.max_residual,
        "interval_R": audit.interval_radial_max,
        "outer_Omega": audit.outer_omega,
        "D": audit.sonic_D,
        "C1": audit.sonic_C1,
        "C2": audit.sonic_C2,
        "arc": np.nan,
        "step_multiplier": step_multiplier,
        "nfev": result.nfev,
        "message": result.message,
    }


def row_from_arc(result, step_multiplier: float) -> dict[str, object]:
    profile = result.profile
    audit = result.residual_audit
    status = result.status
    return {
        "label": f"arc_x{step_multiplier:g}",
        "N": result.params.n_nodes,
        "ratio": result.mdot_ratio,
        "predicted_ratio": result.predicted_mdot_ratio,
        "accepted": accepted(status, result.max_residual),
        "physical": status.physically_valid,
        "equations": status.equations_converged,
        "sonic": status.sonic_regular,
        "dominant": dominant_block(audit),
        "max_HR": float(np.max(profile.H_over_R)),
        "int_adv": profile.integrated_advective_fraction,
        "max_residual": result.max_residual,
        "interval_R": audit.interval_radial_max,
        "outer_Omega": audit.outer_omega,
        "D": audit.sonic_D,
        "C1": audit.sonic_C1,
        "C2": audit.sonic_C2,
        "arc": result.arclength_residual,
        "step_multiplier": step_multiplier,
        "nfev": result.nfev,
        "message": result.message,
    }


def write_table(rows: list[dict[str, object]], resumed: bool = False) -> None:
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# N=64 Adaptive Pseudo-Arclength Continuation",
        "",
        "Generated by `scripts/run_transonic_n64_arclength_adaptive.py`.",
        "",
        f"Target rate: `{TARGET_RATIO:g}` Eddington. Checkpoints: `{CHECKPOINT_DIR}`.",
        "",
    ]
    if resumed:
        lines.extend(
            [
                "This table was reconstructed from accepted branch checkpoints before appending new rows.",
                "",
            ]
        )
    lines.extend(
        [
            "| label | N | Mdot/Mdot_Edd | predicted | accepted | physical | equations | sonic | dominant | max H/R | int adv frac | max residual | interval R | outer Omega | D | C1 | C2 | arc residual | step x | nfev | message |",
            "|---|---:|---:|---:|:---:|:---:|:---:|:---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| {label} | {N} | {ratio} | {predicted_ratio} | {accepted} | {physical} | "
            "{equations} | {sonic} | {dominant} | {max_HR} | {int_adv} | {max_residual} | "
            "{interval_R} | {outer_Omega} | {D} | {C1} | {C2} | {arc} | {step_multiplier} | "
            "{nfev} | {message} |".format(
                label=row["label"],
                N=row["N"],
                ratio=fmt(float(row["ratio"])),
                predicted_ratio=fmt(float(row["predicted_ratio"])),
                accepted="yes" if row["accepted"] else "no",
                physical="yes" if row["physical"] else "no",
                equations="yes" if row["equations"] else "no",
                sonic="yes" if row["sonic"] else "no",
                dominant=row["dominant"],
                max_HR=fmt(float(row["max_HR"])),
                int_adv=fmt(float(row["int_adv"])),
                max_residual=fmt(float(row["max_residual"])),
                interval_R=fmt(float(row["interval_R"])),
                outer_Omega=fmt(float(row["outer_Omega"])),
                D=fmt(float(row["D"])),
                C1=fmt(float(row["C1"])),
                C2=fmt(float(row["C2"])),
                arc=fmt(float(row["arc"])),
                step_multiplier=fmt(float(row["step_multiplier"])),
                nfev=row["nfev"],
                message=str(row["message"]).replace("|", "/"),
            )
        )
    TABLE_OUTPUT.write_text("\n".join(lines) + "\n")


def append_row(rows: list[dict[str, object]], row: dict[str, object], resumed: bool = False) -> None:
    rows.append(row)
    write_table(rows, resumed=resumed)
    print(
        f"{row['label']} N={row['N']} rate={row['ratio']:.4g} "
        f"accepted={row['accepted']} max={row['max_residual']:.4g} "
        f"outer={row['outer_Omega']:.4g} D={row['D']:.4g} "
        f"step={row['step_multiplier']:.4g} dom={row['dominant']}",
        flush=True,
    )


def checkpoint_name(index: int, label: str, ratio: float) -> str:
    rate = f"{ratio:.8f}".replace(".", "p")
    clean_label = "".join(ch if ch.isalnum() else "_" for ch in label)
    return f"{index:03d}_{clean_label}_{rate}.npz"


def save_checkpoint(index: int, label: str, ratio: float, params: TransonicSlimParams, z: np.ndarray, row: dict[str, object]) -> Path:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIR / checkpoint_name(index, label, ratio)
    np.savez_compressed(
        path,
        index=np.array(index),
        label=np.array(label),
        ratio=np.array(ratio),
        n_nodes=np.array(params.n_nodes),
        R_out_rg=np.array(params.R_out_rg),
        residual_tol=np.array(params.residual_tol),
        z=np.asarray(z, dtype=float),
        row_json=np.array(row_to_json(row)),
    )
    return path


def load_checkpoints(fiducial: FiducialParams, mdot_edd: float) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    if not CHECKPOINT_DIR.exists():
        return records
    for path in sorted(CHECKPOINT_DIR.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            n_nodes = int(data["n_nodes"])
            R_out_rg = float(data["R_out_rg"])
            if n_nodes != 64 or not np.isclose(R_out_rg, R_OUT_RG):
                continue
            index = int(data["index"])
            ratio = float(data["ratio"])
            z = np.asarray(data["z"], dtype=float)
            row = row_from_json(str(data["row_json"].item()))
        params = base_params_for(fiducial, mdot_edd, ratio)
        profile = transonic_profile_from_state_vector(z, params)
        records.append({"index": index, "ratio": ratio, "z": z, "profile": profile, "row": row, "path": path})
    records.sort(key=lambda item: (int(item["index"]), float(item["ratio"])))
    return records


def step_limit_for_ratio(ratio: float) -> float:
    return 1.0


def next_step_multiplier(current_ratio: float, old_step: float, nfev: int) -> float:
    ceiling = step_limit_for_ratio(current_ratio)
    if nfev < 250:
        return min(ceiling, old_step * 1.25)
    if nfev > 650:
        return max(MIN_STEP_MULTIPLIER, min(ceiling, old_step * 0.75))
    return min(ceiling, old_step)


def select_frontier_pair(records: list[dict[str, object]]) -> tuple[dict[str, object], dict[str, object]]:
    current = records[-1]
    current_ratio = float(current["ratio"])
    for candidate in reversed(records[:-1]):
        if current_ratio - float(candidate["ratio"]) >= MIN_TANGENT_RATIO_GAP:
            return candidate, current
    return records[-2], current


def rebuild_seed_branch(fiducial: FiducialParams, mdot_edd: float) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    branch_records: list[dict[str, object]] = []

    warmup_profile = None
    warmup_profiles: dict[float, object] = {}
    for ratio in WARMUP_RATIOS:
        print(f"solving warmup N=18 R_out=300 rate={ratio:g}", flush=True)
        params, result = solve_fixed(
            M2_g=fiducial.M2_g,
            mdot_edd=mdot_edd,
            alpha=fiducial.alpha_cool,
            ratio=ratio,
            n_nodes=18,
            R_out_rg=300.0,
            seed_profile=warmup_profile,
            sonic_weight_sequence=(1.0,),
            outer_weight_sequence=(),
            max_nfev=1000,
            max_nfev_per_stage=600,
        )
        row = row_from_fixed("warmup", ratio, params.n_nodes, result)
        append_row(rows, row)
        if row["accepted"] or (result.final_result.status.sonic_regular and result.final_result.max_residual < 8.0e-4):
            warmup_profile = result.final_result.profile
            warmup_profiles[ratio] = warmup_profile

    radius_profile = warmup_profiles.get(N24_RADIUS_BRIDGE_RATIOS[0], warmup_profile)
    radius_usable = False
    for ratio in N24_RADIUS_BRIDGE_RATIOS:
        print(f"solving radius bridge N=24 R_out=1000 rate={ratio:g}", flush=True)
        params = TransonicSlimParams(
            M2_g=fiducial.M2_g,
            Mdot_g_s=ratio * mdot_edd,
            alpha=fiducial.alpha_cool,
            n_nodes=24,
            R_out_rg=1000.0,
            max_nfev=300,
            residual_tol=RESIDUAL_TOL,
        )
        guess = remap_profile_to_new_sonic_grid(radius_profile, params)
        result = solve_transonic_outer_branch(params, initial_guess=guess)
        row = row_from_outer("radius_bridge", ratio, result)
        append_row(rows, row)
        if row["accepted"] or (result.status.sonic_regular and result.max_residual < 8.0e-4):
            radius_profile = result.profile
            radius_usable = True

    if not radius_usable:
        print("stopping: radius bridge did not produce a usable seed", flush=True)
        return rows, branch_records

    bridge_profile = radius_profile
    for ratio in N48_BRIDGE_RATIOS:
        print(f"solving bridge N=48 R_out={R_OUT_RG:g} rate={ratio:g}", flush=True)
        params = TransonicSlimParams(
            M2_g=fiducial.M2_g,
            Mdot_g_s=ratio * mdot_edd,
            alpha=fiducial.alpha_cool,
            n_nodes=48,
            R_out_rg=R_OUT_RG,
            max_nfev=900,
            residual_tol=RESIDUAL_TOL,
        )
        guess = remap_profile_to_new_sonic_grid(bridge_profile, params)
        result = solve_transonic_outer_branch(params, initial_guess=guess)
        row = row_from_outer("bridge", ratio, result)
        append_row(rows, row)
        if row["accepted"] or (result.status.sonic_regular and result.max_residual < 8.0e-4):
            bridge_profile = result.profile
        else:
            print(f"bridge direct failed at {ratio:g}; trying staged fallback", flush=True)
            _fallback_params, fallback = solve_fixed(
                M2_g=fiducial.M2_g,
                mdot_edd=mdot_edd,
                alpha=fiducial.alpha_cool,
                ratio=ratio,
                n_nodes=48,
                R_out_rg=R_OUT_RG,
                seed_profile=bridge_profile,
                sonic_weight_sequence=SONIC_WEIGHTS,
                outer_weight_sequence=(),
                max_nfev=1400,
                max_nfev_per_stage=650,
            )
            fallback_row = row_from_fixed("bridge_homotopy", ratio, 48, fallback)
            append_row(rows, fallback_row)
            if fallback_row["accepted"] or (
                fallback.final_result.status.sonic_regular and fallback.final_result.max_residual < 8.0e-4
            ):
                bridge_profile = fallback.final_result.profile
            else:
                print(f"stopping: bridge rate {ratio:g} did not produce a usable seed", flush=True)
                return rows, branch_records

    seed_profile = bridge_profile
    for ratio in N64_SEED_RATIOS:
        print(f"solving seed N=64 R_out={R_OUT_RG:g} rate={ratio:g}", flush=True)
        params, result = solve_fixed(
            M2_g=fiducial.M2_g,
            mdot_edd=mdot_edd,
            alpha=fiducial.alpha_cool,
            ratio=ratio,
            n_nodes=64,
            R_out_rg=R_OUT_RG,
            seed_profile=seed_profile,
            sonic_weight_sequence=SONIC_WEIGHTS,
            outer_weight_sequence=OUTER_WEIGHTS,
            max_nfev=2400,
            max_nfev_per_stage=1000,
        )
        row = row_from_fixed("seed", ratio, params.n_nodes, result)
        append_row(rows, row)
        if not row["accepted"]:
            print(f"stopping: seed rate {ratio:g} was not accepted", flush=True)
            return rows, branch_records
        seed_profile = result.final_result.profile
        z = np.asarray(result.stages[-1].z, dtype=float)
        index = len(branch_records)
        path = save_checkpoint(index, "seed", ratio, params, z, row)
        branch_records.append({"index": index, "ratio": ratio, "z": z, "profile": seed_profile, "row": row, "path": path})
        print(f"checkpoint {path}", flush=True)
    return rows, branch_records


def continue_branch(fiducial: FiducialParams, mdot_edd: float, rows: list[dict[str, object]], records: list[dict[str, object]], resumed: bool) -> None:
    if len(records) < 2:
        print("stopping: fewer than two accepted N64 branch checkpoints", flush=True)
        return

    records.sort(key=lambda item: (int(item["index"]), float(item["ratio"])))
    previous, current = select_frontier_pair(records)
    previous_ratio = float(previous["ratio"])
    current_ratio = float(current["ratio"])
    print(
        f"frontier tangent pair: {previous_ratio:.4g}->{current_ratio:.4g} "
        f"(gap={current_ratio - previous_ratio:.4g})",
        flush=True,
    )
    previous_profile = previous["profile"]
    current_profile = current["profile"]
    next_index = int(records[-1]["index"]) + 1
    step_multiplier = step_limit_for_ratio(current_ratio)
    failures = 0

    base_params = base_params_for(fiducial, mdot_edd, current_ratio)
    for _step_index in range(MAX_ARC_STEPS):
        if current_ratio >= TARGET_RATIO:
            break
        print(
            f"arclength from {previous_ratio:.4g}->{current_ratio:.4g} "
            f"with step x={step_multiplier:.4g}",
            flush=True,
        )
        result = pseudo_arclength_step(
            base_params,
            mdot_edd,
            previous_profile,
            previous_ratio,
            current_profile,
            current_ratio,
            step_multiplier=step_multiplier,
            mdot_ratio_bounds=(0.25, 1.35),
            max_nfev=1100,
            residual_tol=RESIDUAL_TOL,
            arclength_weight=1.0,
            residual_mode="square",
            sonic_pivot="K",
            metric_mode="blockwise",
            tangent_mode="jacobian",
        )
        row = row_from_arc(result, step_multiplier)
        append_row(rows, row, resumed=resumed)
        if row["accepted"] and result.mdot_ratio > current_ratio * 1.0005:
            path = save_checkpoint(next_index, row["label"], result.mdot_ratio, result.params, result.z, row)
            print(f"checkpoint {path}", flush=True)
            previous_ratio, previous_profile = current_ratio, current_profile
            current_ratio, current_profile = result.mdot_ratio, result.profile
            current = {"index": next_index, "ratio": current_ratio, "profile": current_profile, "z": result.z, "row": row, "path": path}
            records.append(current)
            next_index += 1
            failures = 0
            step_multiplier = next_step_multiplier(current_ratio, step_multiplier, result.nfev)
        else:
            failures += 1
            step_multiplier *= 0.5
            if failures >= MAX_ARC_FAILURES or step_multiplier < MIN_STEP_MULTIPLIER:
                break


def main() -> None:
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)

    records = load_checkpoints(fiducial, mdot_edd)
    if len(records) >= 2:
        rows = [record["row"] for record in records]
        write_table(rows, resumed=True)
        print(
            f"resuming from {len(records)} checkpoints; frontier={float(records[-1]['ratio']):.4g}",
            flush=True,
        )
        continue_branch(fiducial, mdot_edd, rows, records, resumed=True)
    else:
        rows, records = rebuild_seed_branch(fiducial, mdot_edd)
        continue_branch(fiducial, mdot_edd, rows, records, resumed=False)

    print(f"wrote {TABLE_OUTPUT}")


if __name__ == "__main__":
    main()
