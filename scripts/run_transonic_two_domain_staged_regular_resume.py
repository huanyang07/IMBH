"""Resume staged regular-domain refinement above Nreg96 for diagnostics."""

from __future__ import annotations

import numpy as np

import run_transonic_two_domain_staged_regular_refinement as staged
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot
from run_transonic_two_domain_mesh_validation import combined_profile_arrays, load_checkpoint, make_params
from run_transonic_two_domain_sonic_refinement_sprint import (
    SOURCE_CHECKPOINT,
    buffer_audit,
    make_buffer_params,
    source_first_slope,
)


ROOT = staged.ROOT
RESUME_TABLE = ROOT / "outputs" / "tables" / "transonic_two_domain_staged_regular_refinement_resume.md"
START_CHECKPOINT = staged.CHECKPOINT_DIR / "Nreg96_0p90277664.npz"
RESUME_SEQUENCE = (112, 128)


def main() -> None:
    staged.TABLE_OUTPUT = RESUME_TABLE
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
    ref_arrays = combined_profile_arrays(source_x, source_params)
    ref_row = {
        "Rson_rg": float(source_meta["Rson_rg"]),
        "lambda0": float(source_meta["lambda0"]),
        "int_adv": float(source_meta["int_adv"]),
    }
    g_s = source_first_slope(source_x, source_params)

    current_x, current_meta = staged.load_row(START_CHECKPOINT)
    current_params = make_buffer_params(
        fiducial,
        ratio,
        mdot_edd,
        int(current_meta["n_regular"]),
        int(current_meta["n_outer"]),
        float(current_meta["R_far_rg"]),
        float(current_meta["delta_s"]),
    )
    current_audit = buffer_audit("start_Nreg96", current_x, current_params, g_s)
    rows: list[dict[str, object]] = []

    for target in RESUME_SEQUENCE:
        step_ref = {
            "Rson_rg": float(current_audit["Rson_rg"]),
            "lambda0": float(current_audit["lambda0"]),
            "int_adv": float(current_audit["int_adv"]),
        }
        target_params = make_buffer_params(
            fiducial,
            ratio,
            mdot_edd,
            target,
            current_params.n_outer,
            current_params.R_far_rg,
            current_params.delta_s,
        )
        seed, stats = staged.chain_buffer_to_buffer_seed(current_x, current_params, target_params)
        old_logR = staged.buffer_inner_grid(float(current_x[-2]), current_params)
        new_logR = staged.buffer_inner_grid(float(seed[-2]), target_params)
        aligned = staged.aligned_nodes(old_logR, new_logR)
        row = staged.seed_row(target, stats, seed, target_params, g_s, ref_row, ref_arrays, step_ref)
        rows.append(row)
        print(
            f"resume Nreg{target} seed physical={row['physical_active']:.3e} "
            f"dominant={row['dominant']} local={stats['local_defect_max']:.3e}",
            flush=True,
        )

        stage_seed = seed
        final_row = row
        for stage in staged.STAGES:
            free_cols = staged.stage_free_columns(stage, aligned, target_params)
            result = staged.solve_stage(
                stage_seed,
                target_params,
                g_s,
                stage,
                free_cols,
                step_ref["Rson_rg"],
                step_ref["lambda0"],
            )
            row = staged.stage_row(target, stage, len(set(free_cols)), result.x, target_params, g_s, result, ref_row, ref_arrays, step_ref)
            rows.append(row)
            final_row = row
            stage_seed = np.asarray(result.x, dtype=float)
            staged.write_table(rows)
            print(
                f"resume Nreg{target} {stage.name} physical={row['physical_active']:.3e} "
                f"dominant={row['dominant']} free={row['free_count']} nfev={result.nfev}",
                flush=True,
            )

        if float(final_row["physical_active"]) > staged.FALLBACK_TRIGGER:
            release = staged.solve_buffer(seed, target_params, g_s, staged.FALLBACK_RELEASE.max_nfev)
            row = staged.stage_row(
                target,
                staged.FALLBACK_RELEASE,
                2 * target_params.n_inner + 2 * target_params.n_outer + 2,
                release.x,
                target_params,
                g_s,
                release,
                ref_row,
                ref_arrays,
                step_ref,
            )
            rows.append(row)
            staged.write_table(rows)
            print(
                f"resume Nreg{target} {staged.FALLBACK_RELEASE.name} physical={row['physical_active']:.3e} "
                f"dominant={row['dominant']} nfev={release.nfev}",
                flush=True,
            )
            polish = staged.solve_buffer(release.x, target_params, g_s, staged.FALLBACK_POLISH.max_nfev)
            row = staged.stage_row(
                target,
                staged.FALLBACK_POLISH,
                2 * target_params.n_inner + 2 * target_params.n_outer + 2,
                polish.x,
                target_params,
                g_s,
                polish,
                ref_row,
                ref_arrays,
                step_ref,
            )
            rows.append(row)
            staged.write_table(rows)
            final_row = row
            stage_seed = np.asarray(polish.x, dtype=float)
            print(
                f"resume Nreg{target} {staged.FALLBACK_POLISH.name} physical={row['physical_active']:.3e} "
                f"dominant={row['dominant']} nfev={polish.nfev}",
                flush=True,
            )

        staged.save_checkpoint(f"resume_Nreg{target}", stage_seed, final_row)
        current_x = stage_seed
        current_params = target_params
        current_audit = buffer_audit(f"resume_Nreg{target}", current_x, current_params, g_s)

    staged.write_table(rows)
    print(f"wrote {RESUME_TABLE}")


if __name__ == "__main__":
    main()
