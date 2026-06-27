"""Sonic micro-domain experiment for two-domain regular refinement."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from imri_qpe.layer3_minidisk_1d.transonic_collocation import (
    _differential_interval_residual_from_unpacked,
    _integrated_interval_residual_from_unpacked,
)
from imri_qpe.layer3_minidisk_1d.transonic_local import algebraic_state, sonic_diagnostics
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot

from run_transonic_outer_slope_calibration_audit import fmt
from run_transonic_two_domain_dynamic_sonic_patch import (
    CHECKPOINT_DIR as DYNAMIC_CHECKPOINT_DIR,
    FIXED_BUFFER_CHECKPOINT,
)
from run_transonic_two_domain_mesh_validation import combined_profile_arrays, load_checkpoint, make_params
from run_transonic_two_domain_outer_extension import (
    R_MATCH_RG,
    far_boundary_residual,
    integrated_advective_fraction,
    outer_grid,
    pack_two_domain,
    state_bounds_two_domain,
    unpack_two_domain,
)
from run_transonic_two_domain_sonic_refinement_sprint import (
    SOURCE_CHECKPOINT,
    BufferGridParams,
    buffer_inner_grid,
    defect_preserving_values,
    make_buffer_params,
    pchip_extrap,
    row_json,
)


ROOT = Path(__file__).resolve().parents[1]
TABLE_OUTPUT = ROOT / "outputs" / "tables" / "transonic_two_domain_sonic_microdomain.md"
CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints" / "transonic_two_domain_sonic_microdomain"
DYNAMIC_N64_CHECKPOINT = DYNAMIC_CHECKPOINT_DIR / "Nreg64_0p90277664.npz"
DEFAULT_N_SEQUENCE = (64, 80, 96, 112, 128)
DEFAULT_N_MICRO = 4
MAX_NFEV_RELEASE = int(os.environ.get("IMBH_MICRO_MAX_NFEV_RELEASE", "900"))
MAX_NFEV_POLISH = int(os.environ.get("IMBH_MICRO_MAX_NFEV_POLISH", "450"))
SCIENCE_LIMIT = 5.0e-6
STOP_LIMIT = 2.0e-3
SOLVE_SONIC_COMPONENTS = ("D", "K")


def parse_n_sequence() -> tuple[int, ...]:
    raw = os.environ.get("IMBH_MICRO_N_SEQUENCE")
    if raw is None or not raw.strip():
        return DEFAULT_N_SEQUENCE
    return tuple(int(piece) for piece in raw.replace(":", ",").split(",") if piece.strip())


@dataclass(frozen=True)
class MicroGridParams:
    physics: object
    n_regular: int
    n_micro: int
    n_outer: int
    R_match_rg: float
    R_far_rg: float
    delta_s: float
    micro_solve_form: str = "differential"
    far_closure: str = "pressure_supported"
    grid_power_outer: float = 1.0

    @property
    def n_inner(self) -> int:
        return self.n_regular + self.n_micro + 1

    @property
    def r_g(self) -> float:
        return self.physics.r_g

    @property
    def R_match(self) -> float:
        return self.R_match_rg * self.r_g

    @property
    def R_far(self) -> float:
        return self.R_far_rg * self.r_g


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return value if np.isfinite(value) else str(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def load_row(path: Path) -> tuple[np.ndarray, dict[str, object]]:
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data["x"], dtype=float), json.loads(str(data["row_json"].item()))


def save_checkpoint(label: str, x: np.ndarray, row: dict[str, object]) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {key: json_safe(value) for key, value in row.items() if key != "x"}
    np.savez_compressed(
        CHECKPOINT_DIR / f"{label}_0p90277664.npz",
        x=np.asarray(x, dtype=float),
        row_json=np.array(row_json(payload)),
    )


def make_micro_params(
    fiducial: FiducialParams,
    ratio: float,
    mdot_edd: float,
    n_regular: int,
    n_micro: int,
    n_outer: int,
    R_far_rg: float,
    delta_s: float,
    micro_solve_form: str = "differential",
) -> MicroGridParams:
    base = make_params(fiducial, ratio, mdot_edd, n_regular + n_micro + 1, n_outer, R_far_rg)
    return MicroGridParams(
        physics=base.physics,
        n_regular=int(n_regular),
        n_micro=int(n_micro),
        n_outer=int(n_outer),
        R_match_rg=R_MATCH_RG,
        R_far_rg=float(R_far_rg),
        delta_s=float(delta_s),
        micro_solve_form=micro_solve_form,
        far_closure="pressure_supported",
    )


def micro_inner_grid(logR_son: float, params: MicroGridParams) -> np.ndarray:
    logR_buffer = logR_son + params.delta_s
    logR_match = np.log(params.R_match)
    if logR_buffer >= logR_match:
        raise ValueError("sonic micro-domain exceeds match radius")
    micro = logR_son + np.linspace(0.0, params.delta_s, params.n_micro + 1)
    regular = np.linspace(logR_buffer, logR_match, params.n_regular + 1)
    return np.concatenate([micro, regular[1:]])


def unpack_micro(x: np.ndarray, params: MicroGridParams):
    return unpack_two_domain(x, params)  # type: ignore[arg-type]


def sonic_component_vector(logR_son: float, y_s: np.ndarray, lambda0: float, params: MicroGridParams, components: tuple[str, ...]) -> np.ndarray:
    sonic = sonic_diagnostics(logR_son, y_s, lambda0, params.physics)
    values = {
        "D": sonic.D,
        "C1": sonic.C1,
        "C2": sonic.C2,
        "K": sonic.compatibility,
    }
    return np.asarray([values[name] for name in components], dtype=float)


def solver_interval_residual(logu: np.ndarray, logT: np.ndarray, logR: np.ndarray, lambda0: float, params: MicroGridParams, idx: int) -> np.ndarray:
    if idx >= params.n_micro or params.micro_solve_form == "differential":
        return _differential_interval_residual_from_unpacked(logu, logT, logR, lambda0, params.physics, idx)
    residual = _integrated_interval_residual_from_unpacked(logu, logT, logR, lambda0, params.physics, idx)
    if params.micro_solve_form == "integrated":
        return residual
    dx = float(logR[idx + 1] - logR[idx])
    if params.micro_solve_form == "integrated_sqrt_dx":
        return residual / np.sqrt(dx)
    if params.micro_solve_form == "integrated_dx":
        return residual / dx
    raise ValueError(f"unknown micro_solve_form {params.micro_solve_form!r}")


def micro_residual(x: np.ndarray, params: MicroGridParams) -> np.ndarray:
    rows = []
    try:
        logu_i, logT_i, logu_o, logT_o, logR_son, lambda0 = unpack_micro(x, params)
        logR_i = micro_inner_grid(logR_son, params)
        logR_o = outer_grid(params)  # type: ignore[arg-type]
        if np.any(np.diff(logR_i) <= 0.0) or np.any(np.diff(logR_o) <= 0.0):
            raise ValueError("mapped radii must increase")

        for idx in range(params.n_inner - 1):
            rows.append(solver_interval_residual(logu_i, logT_i, logR_i, lambda0, params, idx))
        for idx in range(params.n_outer - 1):
            rows.append(_differential_interval_residual_from_unpacked(logu_o, logT_o, logR_o, lambda0, params.physics, idx))
        rows.append(np.array([logu_i[-1] - logu_o[0], logT_i[-1] - logT_o[0]], dtype=float))
        rows.append(far_boundary_residual(logu_o, logT_o, logR_o, lambda0, params))  # type: ignore[arg-type]
        rows.append(
            sonic_component_vector(
                logR_son,
                np.array([logu_i[0], logT_i[0]], dtype=float),
                lambda0,
                params,
                SOLVE_SONIC_COMPONENTS,
            )
        )
        return np.concatenate(rows)
    except Exception:
        return np.full(2 * params.n_inner + 2 * params.n_outer + len(SOLVE_SONIC_COMPONENTS), 1.0e6)


def micro_sparsity(params: MicroGridParams):
    n_unknown = 2 * params.n_inner + 2 * params.n_outer + 2
    n_rows = 2 * params.n_inner + 2 * params.n_outer + len(SOLVE_SONIC_COMPONENTS)
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

    for idx in range(ni - 1):
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
        pattern[row : row + len(SOLVE_SONIC_COMPONENTS), col] = 1
    return pattern.tocsr()


def solve_micro(seed: np.ndarray, params: MicroGridParams, max_nfev: int):
    lower, upper = state_bounds_two_domain(params)  # type: ignore[arg-type]
    x0 = np.clip(np.asarray(seed, dtype=float), lower + 1.0e-12, upper - 1.0e-12)
    return least_squares(
        lambda trial: micro_residual(trial, params),
        x0,
        jac_sparsity=micro_sparsity(params),
        bounds=(lower, upper),
        x_scale="jac",
        diff_step=2.0e-5,
        ftol=1.0e-11,
        xtol=1.0e-11,
        gtol=1.0e-10,
        max_nfev=max_nfev,
    )


def buffer_to_micro_seed(
    x_old: np.ndarray,
    old_params: BufferGridParams,
    target_params: MicroGridParams,
    method: str,
) -> tuple[np.ndarray, dict[str, object]]:
    logu_i, logT_i, logu_o, logT_o, logR_son, lambda0 = unpack_two_domain(x_old, old_params)  # type: ignore[arg-type]
    old_logR_i = buffer_inner_grid(logR_son, old_params)
    old_logR_o = outer_grid(old_params)  # type: ignore[arg-type]
    new_logR_i = micro_inner_grid(logR_son, target_params)
    new_logR_o = outer_grid(target_params)  # type: ignore[arg-type]
    old_y = np.column_stack([logu_i, logT_i])
    if method == "pchip_patch":
        new_y = np.column_stack(
            [
                pchip_extrap(old_logR_i, logu_i, new_logR_i),
                pchip_extrap(old_logR_i, logT_i, new_logR_i),
            ]
        )
        stats: dict[str, object] = {"local_splits": 0, "copied": 0, "local_defect_max": 0.0, "local_defect_median": 0.0, "local_success_fraction": 1.0}
    elif method == "defect_preserving":
        new_y, stats = defect_preserving_values(old_logR_i, old_y, new_logR_i, lambda0, target_params.physics)
    else:
        raise ValueError(f"unknown remap method {method!r}")
    new_y[0] = old_y[0]
    new_y[target_params.n_micro] = old_y[1]
    logu_o_new = pchip_extrap(old_logR_o, logu_o, new_logR_o)
    logT_o_new = pchip_extrap(old_logR_o, logT_o, new_logR_o)
    logu_o_new[0] = float(new_y[-1, 0])
    logT_o_new[0] = float(new_y[-1, 1])
    seed = pack_two_domain(new_y[:, 0], new_y[:, 1], logu_o_new, logT_o_new, logR_son, lambda0)
    return seed, stats


def micro_to_micro_seed(
    x_old: np.ndarray,
    old_params: MicroGridParams,
    new_params: MicroGridParams,
    method: str,
) -> tuple[np.ndarray, dict[str, object]]:
    logu_i, logT_i, logu_o, logT_o, logR_son, lambda0 = unpack_micro(x_old, old_params)
    old_logR_i = micro_inner_grid(logR_son, old_params)
    old_logR_o = outer_grid(old_params)  # type: ignore[arg-type]
    new_logR_i = micro_inner_grid(logR_son, new_params)
    new_logR_o = outer_grid(new_params)  # type: ignore[arg-type]
    old_y = np.column_stack([logu_i, logT_i])
    if method == "pchip_patch":
        new_y = np.column_stack(
            [
                pchip_extrap(old_logR_i, logu_i, new_logR_i),
                pchip_extrap(old_logR_i, logT_i, new_logR_i),
            ]
        )
        stats: dict[str, object] = {"local_splits": 0, "copied": 0, "local_defect_max": 0.0, "local_defect_median": 0.0, "local_success_fraction": 1.0}
    elif method == "defect_preserving":
        new_y, stats = defect_preserving_values(old_logR_i, old_y, new_logR_i, lambda0, new_params.physics)
    else:
        raise ValueError(f"unknown remap method {method!r}")
    new_y[0] = old_y[0]
    new_y[1 : new_params.n_micro + 1] = np.column_stack(
        [
            pchip_extrap(old_logR_i[: old_params.n_micro + 1], logu_i[: old_params.n_micro + 1], new_logR_i[1 : new_params.n_micro + 1]),
            pchip_extrap(old_logR_i[: old_params.n_micro + 1], logT_i[: old_params.n_micro + 1], new_logR_i[1 : new_params.n_micro + 1]),
        ]
    )
    logu_o_new = pchip_extrap(old_logR_o, logu_o, new_logR_o)
    logT_o_new = pchip_extrap(old_logR_o, logT_o, new_logR_o)
    logu_o_new[0] = float(new_y[-1, 0])
    logT_o_new[0] = float(new_y[-1, 1])
    seed = pack_two_domain(new_y[:, 0], new_y[:, 1], logu_o_new, logT_o_new, logR_son, lambda0)
    return seed, stats


def micro_audit(label: str, x: np.ndarray, params: MicroGridParams, result=None) -> dict[str, object]:
    logu_i, logT_i, logu_o, logT_o, logR_son, lambda0 = unpack_micro(x, params)
    logR_i = micro_inner_grid(logR_son, params)
    logR_o = outer_grid(params)  # type: ignore[arg-type]
    inner = np.asarray(
        [
            _differential_interval_residual_from_unpacked(logu_i, logT_i, logR_i, lambda0, params.physics, idx)
            for idx in range(params.n_inner - 1)
        ],
        dtype=float,
    )
    micro = inner[: params.n_micro]
    regular = inner[params.n_micro :]
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
        "micro_R": float(np.max(np.abs(micro[:, 0]))) if len(micro) else 0.0,
        "micro_E": float(np.max(np.abs(micro[:, 1]))) if len(micro) else 0.0,
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
    first_slope = np.array(
        [
            (float(logu_i[1]) - float(logu_i[0])) / float(logR_i[1] - logR_i[0]),
            (float(logT_i[1]) - float(logT_i[0])) / float(logR_i[1] - logR_i[0]),
        ],
        dtype=float,
    )
    return {
        "label": label,
        "ratio": params.physics.mdot_edd_ratio,
        "delta_s": params.delta_s,
        "micro_solve_form": params.micro_solve_form,
        "n_micro": params.n_micro,
        "n_regular": params.n_regular,
        "n_inner": params.n_inner,
        "n_outer": params.n_outer,
        "R_far_rg": params.R_far_rg,
        "selected_max": float(np.max(np.abs(micro_residual(x, params)))),
        "physical_active": physical,
        "passes_science": bool(physical <= SCIENCE_LIMIT),
        "dominant": dominant,
        "micro_R": blocks["micro_R"],
        "micro_E": blocks["micro_E"],
        "regular_R": blocks["regular_R"],
        "regular_E": blocks["regular_E"],
        "first_micro_R": float(micro[0, 0]) if len(micro) else np.nan,
        "first_micro_E": float(micro[0, 1]) if len(micro) else np.nan,
        "last_micro_R": float(micro[-1, 0]) if len(micro) else np.nan,
        "last_micro_E": float(micro[-1, 1]) if len(micro) else np.nan,
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
        "first_dx": float(logR_i[1] - logR_i[0]),
        "regular_dx0": float(logR_i[params.n_micro + 1] - logR_i[params.n_micro]) if params.n_regular else np.nan,
        "Rson_rg": float(np.exp(logR_son) / params.r_g),
        "lambda0": float(lambda0),
        "int_adv": integrated_advective_fraction(combined_logu, combined_logT, combined_logR, lambda0, params),  # type: ignore[arg-type]
        "max_HR": float(np.max(H_over_R)),
        "g_u_first": float(first_slope[0]),
        "g_T_first": float(first_slope[1]),
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
        "# Two-Domain Sonic Micro-Domain",
        "",
        "Generated by `scripts/run_transonic_two_domain_sonic_microdomain.py`.",
        "",
        "The sonic replacement uses several small ODE-collocation intervals from `Rson` to `Rson+Delta_s` instead of a fixed or first-order dynamic patch. The solve uses the square sonic pair `D,K`; `C1,C2` remain audited unused compatibility diagnostics.",
        "",
        "| label | stage | form | N regular | N micro | N inner | physical | pass | dominant | micro R | micro E | regular R | regular E | first micro R | first micro E | outer R | far omega | D | C1 | C2 | K | Rson/rg | dRson | lambda0 | dlambda | int adv | dint adv | first dx | regular dx0 | g_u first | g_T first | nfev | success | message |",
        "|---|---|---|---:|---:|---:|---:|:---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {stage} | {micro_solve_form} | {n_regular} | {n_micro} | {n_inner} | {physical_active} | {passes_science} | "
            "{dominant} | {micro_R} | {micro_E} | {regular_R} | {regular_E} | {first_micro_R} | "
            "{first_micro_E} | {outer_R} | {far_omega} | {D} | {C1} | {C2} | {K} | {Rson_rg} | "
            "{delta_Rson_rg} | {lambda0} | {delta_lambda0} | {int_adv} | {delta_int_adv} | {first_dx} | "
            "{regular_dx0} | {g_u_first} | {g_T_first} | {nfev} | {success} | {message} |".format(
                label=row["label"],
                stage=row["stage"],
                micro_solve_form=row["micro_solve_form"],
                n_regular=row["n_regular"],
                n_micro=row["n_micro"],
                n_inner=row["n_inner"],
                physical_active=fmt(float(row["physical_active"])),
                passes_science="yes" if row["passes_science"] else "no",
                dominant=row["dominant"],
                micro_R=fmt(float(row["micro_R"])),
                micro_E=fmt(float(row["micro_E"])),
                regular_R=fmt(float(row["regular_R"])),
                regular_E=fmt(float(row["regular_E"])),
                first_micro_R=fmt(float(row["first_micro_R"])),
                first_micro_E=fmt(float(row["first_micro_E"])),
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
                first_dx=fmt(float(row["first_dx"])),
                regular_dx0=fmt(float(row["regular_dx0"])),
                g_u_first=fmt(float(row["g_u_first"])),
                g_T_first=fmt(float(row["g_T_first"])),
                nfev=row["nfev"],
                success="yes" if row["success"] else "no",
                message=str(row["message"]).replace("|", "/"),
            )
        )
    TABLE_OUTPUT.write_text("\n".join(lines) + "\n")


def load_buffer_source(fiducial: FiducialParams, ratio: float, mdot_edd: float) -> tuple[np.ndarray, BufferGridParams, str]:
    path = DYNAMIC_N64_CHECKPOINT if DYNAMIC_N64_CHECKPOINT.exists() else FIXED_BUFFER_CHECKPOINT
    source_x, source_meta = load_row(path)
    params = make_buffer_params(
        fiducial,
        ratio,
        mdot_edd,
        int(source_meta["n_regular"]),
        int(source_meta["n_outer"]),
        float(source_meta["R_far_rg"]),
        float(source_meta["delta_s"]),
    )
    return source_x, params, path.name


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

    buffer_x, buffer_params, source_name = load_buffer_source(fiducial, ratio, mdot_edd)
    n_micro = int(os.environ.get("IMBH_N_MICRO", str(DEFAULT_N_MICRO)))
    delta_s = float(os.environ.get("IMBH_MICRO_DELTA_S", str(buffer_params.delta_s)))
    micro_solve_form = os.environ.get("IMBH_MICRO_SOLVE_FORM", "differential")
    n_sequence = parse_n_sequence()
    current_params = make_micro_params(
        fiducial,
        ratio,
        mdot_edd,
        int(buffer_params.n_regular),
        n_micro,
        int(buffer_params.n_outer),
        float(buffer_params.R_far_rg),
        delta_s,
        micro_solve_form,
    )
    current_x, seed_stats = buffer_to_micro_seed(buffer_x, buffer_params, current_params, "defect_preserving")
    rows: list[dict[str, object]] = []
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"source={source_name} n_micro={n_micro} delta_s={delta_s:g} "
        f"form={micro_solve_form} sequence={n_sequence}",
        flush=True,
    )

    for n_regular in n_sequence:
        if n_regular == current_params.n_regular:
            seed = current_x
            params = current_params
        else:
            params = make_micro_params(
                fiducial,
                ratio,
                mdot_edd,
                n_regular,
                current_params.n_micro,
                current_params.n_outer,
                current_params.R_far_rg,
                current_params.delta_s,
                current_params.micro_solve_form,
            )
            seed, seed_stats = micro_to_micro_seed(current_x, current_params, params, "defect_preserving")
        seed_row = add_reference(micro_audit(f"Nreg{n_regular}", seed, params), ref_row)
        seed_row["stage"] = "seed"
        seed_row["seed_local_max"] = float(seed_stats.get("local_defect_max", 0.0))
        rows.append(seed_row)
        print(
            f"Nreg{n_regular} micro seed physical={seed_row['physical_active']:.3e} "
            f"dominant={seed_row['dominant']} local_max={seed_row['seed_local_max']:.3e}",
            flush=True,
        )

        release = solve_micro(seed, params, MAX_NFEV_RELEASE)
        release_row = add_reference(micro_audit(f"Nreg{n_regular}", release.x, params, release), ref_row)
        release_row["stage"] = "release"
        release_row["seed_local_max"] = float(seed_stats.get("local_defect_max", 0.0))
        rows.append(release_row)
        write_table(rows)
        print(
            f"Nreg{n_regular} micro release physical={release_row['physical_active']:.3e} "
            f"dominant={release_row['dominant']} nfev={release.nfev}",
            flush=True,
        )

        polish = solve_micro(release.x, params, MAX_NFEV_POLISH)
        polish_row = add_reference(micro_audit(f"Nreg{n_regular}", polish.x, params, polish), ref_row)
        polish_row["stage"] = "polish"
        polish_row["seed_local_max"] = float(seed_stats.get("local_defect_max", 0.0))
        rows.append(polish_row)
        write_table(rows)
        print(
            f"Nreg{n_regular} micro polish physical={polish_row['physical_active']:.3e} "
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
