"""Dynamic sonic-patch experiment for two-domain regular refinement."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from imri_qpe.layer3_minidisk_1d.transonic_collocation import _differential_interval_residual_from_unpacked
from imri_qpe.layer3_minidisk_1d.transonic_local import algebraic_state, scaled_differential_matrix, sonic_diagnostics
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot

from run_transonic_outer_slope_calibration_audit import fmt
from run_transonic_two_domain_mesh_validation import combined_profile_arrays, load_checkpoint, make_params
from run_transonic_two_domain_outer_extension import (
    R_MATCH_RG,
    far_boundary_residual,
    integrated_advective_fraction,
    outer_grid,
    pack_two_domain,
    state_bounds_two_domain,
)
from run_transonic_two_domain_sonic_refinement_sprint import (
    CHECKPOINT_DIR as FIXED_PATCH_CHECKPOINT_DIR,
    SOURCE_CHECKPOINT,
    BufferGridParams,
    buffer_inner_grid,
    buffer_row_with_reference,
    buffer_to_buffer_seed,
    make_buffer_params,
    row_json,
    unpack_buffer,
)


ROOT = Path(__file__).resolve().parents[1]
TABLE_OUTPUT = ROOT / "outputs" / "tables" / "transonic_two_domain_dynamic_sonic_patch.md"
CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints" / "transonic_two_domain_dynamic_sonic_patch"
FIXED_BUFFER_CHECKPOINT = FIXED_PATCH_CHECKPOINT_DIR / "delta0p02_Nreg64_defect_preserving_0p90277664.npz"
N_SEQUENCE = (64, 80, 96, 112, 128)
MAX_NFEV_RELEASE = 650
MAX_NFEV_POLISH = 300
SCIENCE_LIMIT = 5.0e-6
STOP_LIMIT = 2.0e-3


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return value if np.isfinite(value) else str(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def dynamic_row_json(row: dict[str, object]) -> str:
    return json.dumps({key: json_safe(value) for key, value in row.items() if key != "x"}, sort_keys=True)


def load_row(path: Path) -> tuple[np.ndarray, dict[str, object]]:
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data["x"], dtype=float), json.loads(str(data["row_json"].item()))


def save_checkpoint(label: str, x: np.ndarray, row: dict[str, object]) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {key: value for key, value in row.items() if key != "x"}
    np.savez_compressed(
        CHECKPOINT_DIR / f"{label}_0p90277664.npz",
        x=np.asarray(x, dtype=float),
        row_json=np.array(row_json(payload)),
    )


def dynamic_patch_residual(logu_i: np.ndarray, logT_i: np.ndarray, logR_son: float, lambda0: float, params: BufferGridParams) -> np.ndarray:
    y_s = np.array([float(logu_i[0]), float(logT_i[0])], dtype=float)
    g_s = np.array(
        [
            (float(logu_i[1]) - float(logu_i[0])) / params.delta_s,
            (float(logT_i[1]) - float(logT_i[0])) / params.delta_s,
        ],
        dtype=float,
    )
    matrix, rhs, _radial_scale, _energy_scale = scaled_differential_matrix(logR_son, y_s, lambda0, params.physics)
    return matrix @ g_s + rhs


def dynamic_residual(x: np.ndarray, params: BufferGridParams) -> np.ndarray:
    rows = []
    try:
        logu_i, logT_i, logu_o, logT_o, logR_son, lambda0 = unpack_buffer(x, params)
        logR_i = buffer_inner_grid(logR_son, params)
        logR_o = outer_grid(params)  # type: ignore[arg-type]
        if np.any(np.diff(logR_i) <= 0.0) or np.any(np.diff(logR_o) <= 0.0):
            raise ValueError("mapped radii must increase")

        rows.append(dynamic_patch_residual(logu_i, logT_i, logR_son, lambda0, params))
        for idx in range(1, params.n_inner - 1):
            rows.append(_differential_interval_residual_from_unpacked(logu_i, logT_i, logR_i, lambda0, params.physics, idx))
        for idx in range(params.n_outer - 1):
            rows.append(_differential_interval_residual_from_unpacked(logu_o, logT_o, logR_o, lambda0, params.physics, idx))
        rows.append(np.array([logu_i[-1] - logu_o[0], logT_i[-1] - logT_o[0]], dtype=float))
        rows.append(far_boundary_residual(logu_o, logT_o, logR_o, lambda0, params))  # type: ignore[arg-type]
        sonic = sonic_diagnostics(logR_son, np.array([logu_i[0], logT_i[0]], dtype=float), lambda0, params.physics)
        rows.append(np.array([sonic.D, sonic.C1, sonic.C2, sonic.compatibility], dtype=float))
        return np.concatenate(rows)
    except Exception:
        return np.full(2 * params.n_inner + 2 * params.n_outer + 4, 1.0e6)


def dynamic_sparsity(params: BufferGridParams):
    n_unknown = 2 * params.n_inner + 2 * params.n_outer + 2
    n_rows = 2 * params.n_inner + 2 * params.n_outer + 4
    pattern = lil_matrix((n_rows, n_unknown), dtype=int)
    ni = params.n_inner
    no = params.n_outer
    iu = 0
    iT = ni
    ou = 2 * ni
    oT = 2 * ni + no
    logR_col = 2 * ni + 2 * no
    lambda_col = logR_col + 1
    row = 0

    for col in (iu, iu + 1, iT, iT + 1, logR_col, lambda_col):
        pattern[row : row + 2, col] = 1
    row += 2

    for idx in range(1, ni - 1):
        columns = (iu + idx, iu + idx + 1, iT + idx, iT + idx + 1, logR_col, lambda_col)
        for col in columns:
            pattern[row : row + 2, col] = 1
        row += 2

    for idx in range(no - 1):
        columns = (ou + idx, ou + idx + 1, oT + idx, oT + idx + 1, lambda_col)
        for col in columns:
            pattern[row : row + 2, col] = 1
        row += 2

    for col in (iu + ni - 1, iT + ni - 1, ou, oT):
        pattern[row : row + 2, col] = 1
    row += 2

    for col in (ou + no - 2, ou + no - 1, oT + no - 2, oT + no - 1, lambda_col):
        pattern[row : row + 2, col] = 1
    row += 2

    for col in (iu, iT, logR_col, lambda_col):
        pattern[row : row + 4, col] = 1
    return pattern.tocsr()


def solve_dynamic(seed: np.ndarray, params: BufferGridParams, max_nfev: int):
    lower, upper = state_bounds_two_domain(params)  # type: ignore[arg-type]
    x0 = np.clip(np.asarray(seed, dtype=float), lower + 1.0e-12, upper - 1.0e-12)
    return least_squares(
        lambda trial: dynamic_residual(trial, params),
        x0,
        jac_sparsity=dynamic_sparsity(params),
        bounds=(lower, upper),
        x_scale="jac",
        diff_step=2.0e-5,
        ftol=1.0e-11,
        xtol=1.0e-11,
        gtol=1.0e-10,
        max_nfev=max_nfev,
    )


def dynamic_audit(label: str, x: np.ndarray, params: BufferGridParams, result=None) -> dict[str, object]:
    logu_i, logT_i, logu_o, logT_o, logR_son, lambda0 = unpack_buffer(x, params)
    logR_i = buffer_inner_grid(logR_son, params)
    logR_o = outer_grid(params)  # type: ignore[arg-type]
    patch = dynamic_patch_residual(logu_i, logT_i, logR_son, lambda0, params)
    ordinary_first = _differential_interval_residual_from_unpacked(logu_i, logT_i, logR_i, lambda0, params.physics, 0)
    regular = np.asarray(
        [
            _differential_interval_residual_from_unpacked(logu_i, logT_i, logR_i, lambda0, params.physics, idx)
            for idx in range(1, params.n_inner - 1)
        ],
        dtype=float,
    )
    outer = np.asarray(
        [
            _differential_interval_residual_from_unpacked(logu_o, logT_o, logR_o, lambda0, params.physics, idx)
            for idx in range(params.n_outer - 1)
        ],
        dtype=float,
    )
    interface = np.array([logu_i[-1] - logu_o[0], logT_i[-1] - logT_o[0]], dtype=float)
    far = far_boundary_residual(logu_o, logT_o, logR_o, lambda0, params)  # type: ignore[arg-type]
    sonic = sonic_diagnostics(logR_son, np.array([logu_i[0], logT_i[0]], dtype=float), lambda0, params.physics)
    combined_logR = np.concatenate([logR_i, logR_o[1:]])
    combined_logu = np.concatenate([logu_i, logu_o[1:]])
    combined_logT = np.concatenate([logT_i, logT_o[1:]])
    H_over_R = [
        algebraic_state(float(lr), float(lu), float(lt), lambda0, params.physics).H_over_R
        for lr, lu, lt in zip(combined_logR, combined_logu, combined_logT)
    ]
    blocks = {
        "patch": float(np.max(np.abs(patch))),
        "regular_R": float(np.max(np.abs(regular[:, 0]))) if len(regular) else 0.0,
        "regular_E": float(np.max(np.abs(regular[:, 1]))) if len(regular) else 0.0,
        "outer_R": float(np.max(np.abs(outer[:, 0]))),
        "outer_E": float(np.max(np.abs(outer[:, 1]))),
        "interface": float(np.max(np.abs(interface))),
        "far_omega": abs(float(far[0])),
        "far_energy": abs(float(far[1])),
        "D": abs(float(sonic.D)),
        "C1": abs(float(sonic.C1)),
        "C2": abs(float(sonic.C2)),
        "K": abs(float(sonic.compatibility)),
    }
    physical = max(blocks.values())
    dominant = max(blocks, key=blocks.get)
    g_s = np.array(
        [
            (float(logu_i[1]) - float(logu_i[0])) / params.delta_s,
            (float(logT_i[1]) - float(logT_i[0])) / params.delta_s,
        ],
        dtype=float,
    )
    return {
        "label": label,
        "ratio": params.physics.mdot_edd_ratio,
        "delta_s": params.delta_s,
        "n_regular": params.n_regular,
        "n_inner": params.n_inner,
        "n_outer": params.n_outer,
        "R_far_rg": params.R_far_rg,
        "selected_max": float(np.max(np.abs(dynamic_residual(x, params)))),
        "physical_active": physical,
        "passes_science": bool(physical <= SCIENCE_LIMIT),
        "dominant": dominant,
        "patch": blocks["patch"],
        "patch_R": float(patch[0]),
        "patch_E": float(patch[1]),
        "regular_R": blocks["regular_R"],
        "regular_E": blocks["regular_E"],
        "ordinary_first_R": float(ordinary_first[0]),
        "ordinary_first_E": float(ordinary_first[1]),
        "outer_R": blocks["outer_R"],
        "outer_E": blocks["outer_E"],
        "interface": blocks["interface"],
        "far_omega": float(far[0]),
        "far_energy": float(far[1]),
        "D": float(sonic.D),
        "C1": float(sonic.C1),
        "C2": float(sonic.C2),
        "K": float(sonic.compatibility),
        "smin_over_smax": float(sonic.smin_over_smax),
        "g_u_patch": float(g_s[0]),
        "g_T_patch": float(g_s[1]),
        "first_dx": float(logR_i[1] - logR_i[0]),
        "second_dx": float(logR_i[2] - logR_i[1]) if len(logR_i) > 2 else np.nan,
        "Rson_rg": float(np.exp(logR_son) / params.r_g),
        "lambda0": float(lambda0),
        "int_adv": integrated_advective_fraction(combined_logu, combined_logT, combined_logR, lambda0, params),  # type: ignore[arg-type]
        "max_HR": float(np.max(H_over_R)),
        "nfev": int(result.nfev) if result is not None else 0,
        "success": bool(result.success) if result is not None else True,
        "message": str(result.message) if result is not None else "seed evaluation",
        "x": np.asarray(x, dtype=float),
    }


def add_reference(row: dict[str, object], ref_row: dict[str, float]) -> dict[str, object]:
    row["delta_Rson_rg"] = float(row["Rson_rg"] - ref_row["Rson_rg"])
    row["delta_lambda0"] = float(row["lambda0"] - ref_row["lambda0"])
    row["delta_int_adv"] = float(row["int_adv"] - ref_row["int_adv"])
    return row


def write_table(rows: list[dict[str, object]]) -> None:
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Two-Domain Dynamic Sonic Patch",
        "",
        "Generated by `scripts/run_transonic_two_domain_dynamic_sonic_patch.py`.",
        "",
        "The sonic patch slope is inferred from `(y_buffer-y_s)/Delta_s` and constrained by the scaled local sonic differential equations `A_s g_s + c_s = 0`. The residual also includes `D,C1,C2,K`, so this replaces the earlier fixed-slope patch.",
        "",
        "| label | stage | N regular | N inner | physical | pass | dominant | patch | patch R | patch E | regular R | regular E | ordinary first R | ordinary first E | outer R | far omega | D | C1 | C2 | K | Rson/rg | dRson | lambda0 | dlambda | int adv | dint adv | g_u patch | g_T patch | nfev | success | message |",
        "|---|---|---:|---:|---:|:---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {stage} | {n_regular} | {n_inner} | {physical_active} | {passes_science} | {dominant} | "
            "{patch} | {patch_R} | {patch_E} | {regular_R} | {regular_E} | {ordinary_first_R} | "
            "{ordinary_first_E} | {outer_R} | {far_omega} | {D} | {C1} | {C2} | {K} | {Rson_rg} | "
            "{delta_Rson_rg} | {lambda0} | {delta_lambda0} | {int_adv} | {delta_int_adv} | {g_u_patch} | "
            "{g_T_patch} | {nfev} | {success} | {message} |".format(
                label=row["label"],
                stage=row["stage"],
                n_regular=row["n_regular"],
                n_inner=row["n_inner"],
                physical_active=fmt(float(row["physical_active"])),
                passes_science="yes" if row["passes_science"] else "no",
                dominant=row["dominant"],
                patch=fmt(float(row["patch"])),
                patch_R=fmt(float(row["patch_R"])),
                patch_E=fmt(float(row["patch_E"])),
                regular_R=fmt(float(row["regular_R"])),
                regular_E=fmt(float(row["regular_E"])),
                ordinary_first_R=fmt(float(row["ordinary_first_R"])),
                ordinary_first_E=fmt(float(row["ordinary_first_E"])),
                outer_R=fmt(float(row["outer_R"])),
                far_omega=fmt(float(row["far_omega"])),
                D=fmt(float(row["D"])),
                C1=fmt(float(row["C1"])),
                C2=fmt(float(row["C2"])),
                K=fmt(float(row["K"])),
                Rson_rg=fmt(float(row["Rson_rg"])),
                delta_Rson_rg=fmt(float(row["delta_Rson_rg"])),
                lambda0=fmt(float(row["lambda0"])),
                delta_lambda0=fmt(float(row["delta_lambda0"])),
                int_adv=fmt(float(row["int_adv"])),
                delta_int_adv=fmt(float(row["delta_int_adv"])),
                g_u_patch=fmt(float(row["g_u_patch"])),
                g_T_patch=fmt(float(row["g_T_patch"])),
                nfev=row["nfev"],
                success="yes" if row["success"] else "no",
                message=str(row["message"]).replace("|", "/"),
            )
        )
    TABLE_OUTPUT.write_text("\n".join(lines) + "\n")


def main() -> None:
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)
    source_x, source_meta = load_checkpoint(SOURCE_CHECKPOINT)
    ratio = float(source_meta["ratio"])
    source_params = make_params(
        fiducial,
        ratio,
        mdot_edd,
        int(source_meta["n_inner"]),
        int(source_meta["n_outer"]),
        float(source_meta["R_far_rg"]),
    )
    _ref_arrays = combined_profile_arrays(source_x, source_params)
    ref_row = {
        "Rson_rg": float(source_meta["Rson_rg"]),
        "lambda0": float(source_meta["lambda0"]),
        "int_adv": float(source_meta["int_adv"]),
    }
    fixed_x, fixed_meta = load_row(FIXED_BUFFER_CHECKPOINT)
    current_params = make_buffer_params(
        fiducial,
        ratio,
        mdot_edd,
        int(fixed_meta["n_regular"]),
        int(fixed_meta["n_outer"]),
        float(fixed_meta["R_far_rg"]),
        float(fixed_meta["delta_s"]),
    )
    current_x = fixed_x
    rows: list[dict[str, object]] = []
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    for n_regular in N_SEQUENCE:
        if n_regular == current_params.n_regular:
            seed = current_x
            params = current_params
        else:
            params = make_buffer_params(fiducial, ratio, mdot_edd, n_regular, current_params.n_outer, current_params.R_far_rg, current_params.delta_s)
            seed, _stats = buffer_to_buffer_seed(current_x, current_params, params, "defect_preserving")
        seed_row = add_reference(dynamic_audit(f"Nreg{n_regular}", seed, params), ref_row)
        seed_row["stage"] = "seed"
        rows.append(seed_row)
        print(
            f"Nreg{n_regular} dynamic seed physical={seed_row['physical_active']:.3e} "
            f"dominant={seed_row['dominant']}",
            flush=True,
        )

        release = solve_dynamic(seed, params, MAX_NFEV_RELEASE)
        release_row = add_reference(dynamic_audit(f"Nreg{n_regular}", release.x, params, release), ref_row)
        release_row["stage"] = "release"
        rows.append(release_row)
        write_table(rows)
        print(
            f"Nreg{n_regular} dynamic release physical={release_row['physical_active']:.3e} "
            f"dominant={release_row['dominant']} nfev={release.nfev}",
            flush=True,
        )

        polish = solve_dynamic(release.x, params, MAX_NFEV_POLISH)
        polish_row = add_reference(dynamic_audit(f"Nreg{n_regular}", polish.x, params, polish), ref_row)
        polish_row["stage"] = "polish"
        rows.append(polish_row)
        write_table(rows)
        print(
            f"Nreg{n_regular} dynamic polish physical={polish_row['physical_active']:.3e} "
            f"dominant={polish_row['dominant']} nfev={polish.nfev}",
            flush=True,
        )
        save_checkpoint(f"Nreg{n_regular}", np.asarray(polish_row["x"], dtype=float), polish_row)
        current_x = np.asarray(polish_row["x"], dtype=float)
        current_params = params
        if float(polish_row["physical_active"]) > STOP_LIMIT:
            print(
                f"stopping after Nreg{n_regular}: physical={polish_row['physical_active']:.3e} exceeds {STOP_LIMIT:.1e}",
                flush=True,
            )
            break

    write_table(rows)
    print(f"wrote {TABLE_OUTPUT}")


if __name__ == "__main__":
    main()
