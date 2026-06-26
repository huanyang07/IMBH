"""Polish high-rate N=64 transonic checkpoints with the square residual."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    TransonicSlimParams,
    collocation_residual,
    solve_square_transonic_polish,
    square_collocation_residual,
    transonic_profile_from_state_vector,
    unused_sonic_compatibility,
)
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = ROOT / "outputs" / "checkpoints" / "transonic_n64_arclength_adaptive"
TABLE_OUTPUT = ROOT / "outputs" / "tables" / "transonic_square_polish.md"
TARGET_RATIOS = (0.9028, 0.9653, 0.9963)
PIVOTS = ("K", "C1", "C2")


def fmt(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.4g}"


def checkpoint_records() -> list[tuple[float, Path]]:
    records: list[tuple[float, Path]] = []
    for path in sorted(CHECKPOINT_DIR.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            records.append((float(data["ratio"]), path))
    if not records:
        raise RuntimeError(f"no checkpoints found in {CHECKPOINT_DIR}")
    return records


def nearest_checkpoints(targets: tuple[float, ...]) -> list[Path]:
    records = checkpoint_records()
    selected: list[Path] = []
    seen: set[Path] = set()
    for target in targets:
        _ratio, path = min(records, key=lambda item: abs(item[0] - target))
        if path not in seen:
            selected.append(path)
            seen.add(path)
    return selected


def profile_changes(before, after) -> tuple[float, float, float, float]:
    dlogu = float(np.max(np.abs(np.log(after.u) - np.log(before.u))))
    dlogT = float(np.max(np.abs(np.log(after.T) - np.log(before.T))))
    dlogRson = float(abs(np.log(after.sonic_radius / before.sonic_radius)))
    dlambda = float(abs(after.lambda0 - before.lambda0))
    return dlogu, dlogT, dlogRson, dlambda


def main() -> None:
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)
    rows = []
    for path in nearest_checkpoints(TARGET_RATIOS):
        with np.load(path, allow_pickle=False) as data:
            z0 = np.asarray(data["z"], dtype=float)
            ratio = float(data["ratio"])
        params = TransonicSlimParams(
            M2_g=fiducial.M2_g,
            Mdot_g_s=ratio * mdot_edd,
            alpha=fiducial.alpha_cool,
            n_nodes=64,
            R_out_rg=3000.0,
            residual_tol=3.0e-4,
            max_nfev=500,
        )
        for pivot in PIVOTS:
            before_profile = transonic_profile_from_state_vector(z0, params)
            before_square = square_collocation_residual(z0, params, pivot=pivot)
            before_full = collocation_residual(z0, params)
            before_unused = unused_sonic_compatibility(z0, params, pivot=pivot)
            print(
                f"polishing {path.name} ratio={ratio:.4g} pivot={pivot} "
                f"square0={np.max(np.abs(before_square)):.3e} full0={np.max(np.abs(before_full)):.3e}",
                flush=True,
            )
            result = solve_square_transonic_polish(
                params,
                z0,
                pivot=pivot,
                method="newton",
                max_iter=10,
                max_nfev=500,
                residual_tol=1.0e-6,
                jacobian_rel_step=3.0e-5,
            )
            after_profile = result.result.profile
            dlogu, dlogT, dlogRson, dlambda = profile_changes(before_profile, after_profile)
            rows.append(
                {
                    "checkpoint": path.name,
                    "ratio": ratio,
                    "pivot": result.pivot,
                    "method": result.method,
                    "before_square": float(np.max(np.abs(before_square))),
                    "after_square": result.final_square_max_residual,
                    "before_full": float(np.max(np.abs(before_full))),
                    "after_full": result.result.max_residual,
                    "before_unused": float(before_unused),
                    "after_unused": result.unused_compatibility,
                    "physical": result.result.status.physically_valid,
                    "equations": result.result.status.equations_converged,
                    "sonic": result.result.status.sonic_regular,
                    "max_HR_before": float(np.max(before_profile.H_over_R)),
                    "max_HR_after": float(np.max(after_profile.H_over_R)),
                    "int_adv_before": before_profile.integrated_advective_fraction,
                    "int_adv_after": after_profile.integrated_advective_fraction,
                    "dlogu": dlogu,
                    "dlogT": dlogT,
                    "dlogRson": dlogRson,
                    "dlambda": dlambda,
                    "iterations": result.iterations,
                    "reductions": result.line_search_reductions,
                    "step": result.final_step_norm,
                    "damping": result.final_linear_damping,
                    "nfev": result.result.nfev,
                    "njev": result.result.njev,
                    "message": result.result.message,
                }
            )
            print(
                f"{path.name} ratio={ratio:.4g} pivot={result.pivot} "
                f"square {np.max(np.abs(before_square)):.3e}->{result.final_square_max_residual:.3e} "
                f"full {np.max(np.abs(before_full)):.3e}->{result.result.max_residual:.3e} "
                f"iter={result.iterations} reductions={result.line_search_reductions}",
                flush=True,
            )

    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Square Fixed-Mdot Polish Audit",
        "",
        "Generated by `scripts/run_transonic_square_polish.py`.",
        "",
        "Each row loads an accepted N=64 arclength checkpoint and polishes it at fixed accretion rate with the square sonic residual system.",
        "",
        "The `K` rows solve `D=0` plus SVD left-null compatibility; `C1`/`C2` rows use the older adjugate compatibility components. The unused column reports the omitted adjugate compatibility scale.",
        "",
        "| checkpoint | Mdot/Mdot_Edd | pivot | method | physical | equations | sonic | square max before | square max after | full max before | full max after | unused before | unused after | max H/R before | max H/R after | int adv before | int adv after | max dlogu | max dlogT | dlog Rson | d lambda0 | iter | cuts | step inf | damping | nfev | njev | message |",
        "|---|---:|:---:|:---:|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {checkpoint} | {ratio} | {pivot} | {method} | {physical} | {equations} | {sonic} | "
            "{before_square} | {after_square} | {before_full} | {after_full} | "
            "{before_unused} | {after_unused} | {max_HR_before} | {max_HR_after} | "
            "{int_adv_before} | {int_adv_after} | {dlogu} | {dlogT} | {dlogRson} | "
            "{dlambda} | {iterations} | {reductions} | {step} | {damping} | {nfev} | {njev} | {message} |".format(
                checkpoint=row["checkpoint"],
                ratio=fmt(row["ratio"]),
                pivot=row["pivot"],
                method=row["method"],
                physical="yes" if row["physical"] else "no",
                equations="yes" if row["equations"] else "no",
                sonic="yes" if row["sonic"] else "no",
                before_square=fmt(row["before_square"]),
                after_square=fmt(row["after_square"]),
                before_full=fmt(row["before_full"]),
                after_full=fmt(row["after_full"]),
                before_unused=fmt(row["before_unused"]),
                after_unused=fmt(row["after_unused"]),
                max_HR_before=fmt(row["max_HR_before"]),
                max_HR_after=fmt(row["max_HR_after"]),
                int_adv_before=fmt(row["int_adv_before"]),
                int_adv_after=fmt(row["int_adv_after"]),
                dlogu=fmt(row["dlogu"]),
                dlogT=fmt(row["dlogT"]),
                dlogRson=fmt(row["dlogRson"]),
                dlambda=fmt(row["dlambda"]),
                iterations=row["iterations"],
                reductions=row["reductions"],
                step=fmt(row["step"]),
                damping=fmt(row["damping"]),
                nfev=row["nfev"],
                njev=row["njev"],
                message=str(row["message"]).replace("|", "/"),
            )
        )
    TABLE_OUTPUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {TABLE_OUTPUT}")


if __name__ == "__main__":
    main()
