"""Separate stream forcing from initial global-state relaxation at N64."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    GlobalAdaptiveRestart,
    GlobalCellSources,
    GlobalConservativeState,
    evaluate_global_rusanov_profile,
    global_conservative_rhs,
    load_global_adaptive_restart,
    make_global_mechanical_energy_reference,
    recover_global_primitives,
    save_global_adaptive_milestone,
)

from run_global_physical_open_preflight import _canonical_open_evaluation
from run_global_roche_adaptive_preflight import (
    _git_metadata,
    run_adaptive_campaign,
)
from run_global_roche_loading_preflight import _prepared_case


ROOT = Path(__file__).resolve().parents[1]
INNER_RADIUS_RG = 4.5
N_CELLS = 64


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--target-loading-fraction", type=float, default=1.0e-7
    )
    parser.add_argument("--maximum-nfev", type=int, default=600)
    parser.add_argument("--maximum-accepted-steps", type=int, default=32)
    return parser.parse_args()


def _profile(
    context,
    grid,
    state,
    correction,
    provider,
    external_sources,
):
    mass = context.base.inner_params.M2_g
    primitives = recover_global_primitives(
        grid,
        state,
        mass,
        specific_mechanical_energy_correction=correction,
    )
    return evaluate_global_rusanov_profile(
        grid,
        state,
        mass,
        reference_state=state,
        boundary_mode="roche_outer",
        alpha=context.base.alpha,
        stress_boundary_mode="outer_zero_torque",
        include_radiative_cooling=True,
        include_vertical_column_work=True,
        external_sources=external_sources,
        primitives=primitives,
        outer_overflow_provider=provider,
        specific_mechanical_energy_correction=correction,
    )


def _state_from_rhs(state, rhs, dt: float) -> GlobalConservativeState:
    return GlobalConservativeState(
        **{
            name: getattr(state, name) + dt * getattr(rhs, name)
            for name in (
                "mass",
                "radial_momentum",
                "angular_momentum",
                "total_energy",
            )
        }
    ).validated()


def _linearized_controller(
    context,
    grid,
    state,
    correction,
    rhs,
    dt: float,
) -> dict:
    mass = context.base.inner_params.M2_g
    old = recover_global_primitives(
        grid,
        state,
        mass,
        specific_mechanical_energy_correction=correction,
    )
    trial_state = _state_from_rhs(state, rhs, dt)
    new = recover_global_primitives(
        grid,
        trial_state,
        mass,
        specific_mechanical_energy_correction=correction,
    )
    old_h = np.asarray(old.vertical.H, dtype=float) / grid.centers
    new_h = np.asarray(new.vertical.H, dtype=float) / grid.centers
    candidates = {
        "log_surface_density": np.abs(
            np.log(new.surface_density / old.surface_density)
        ),
        "log_temperature": np.abs(np.log(new.temperature / old.temperature)),
        "relative_thickness": np.abs(new_h - old_h)
        / np.maximum(old_h, 1.0e-300),
    }
    variable = max(candidates, key=lambda name: float(np.max(candidates[name])))
    values = candidates[variable]
    index = int(np.argmax(values))
    return {
        "variable": variable,
        "cell_index": index,
        "radius": float(grid.centers[index]),
        "change_metric": float(values[index]),
        "maximum_log_surface_density_change": float(
            np.max(candidates["log_surface_density"])
        ),
        "maximum_log_temperature_change": float(
            np.max(candidates["log_temperature"])
        ),
        "maximum_relative_thickness_change": float(
            np.max(candidates["relative_thickness"])
        ),
    }


def _rhs_summary(rhs, state) -> dict:
    summary = {}
    for name in (
        "mass",
        "radial_momentum",
        "angular_momentum",
        "total_energy",
    ):
        values = np.asarray(getattr(rhs, name), dtype=float)
        storage = np.asarray(getattr(state, name), dtype=float)
        floor = max(float(np.max(np.abs(storage))) * 1.0e-14, 1.0e-300)
        summary[name] = {
            "sum": float(np.sum(values)),
            "l1_norm": float(np.sum(np.abs(values))),
            "linf_norm": float(np.max(np.abs(values))),
            "maximum_storage_relative_rate": float(
                np.max(np.abs(values) / np.maximum(np.abs(storage), floor))
            ),
        }
    return summary


def _instantaneous_decomposition(
    context,
    grid,
    initial,
    correction,
    stream,
    provider,
    horizon: float,
) -> dict:
    zero = GlobalCellSources.zeros(grid.centers.size)
    profile_on = _profile(
        context, grid, initial, correction, provider, stream
    )
    profile_off = _profile(
        context, grid, initial, correction, provider, zero
    )
    rhs_on = global_conservative_rhs(
        profile_on.face_fluxes, profile_on.cell_sources
    )
    rhs_off = global_conservative_rhs(
        profile_off.face_fluxes, profile_off.cell_sources
    )
    rhs_source = GlobalConservativeState(
        **{
            name: np.asarray(
                getattr(rhs_on, name) - getattr(rhs_off, name), dtype=float
            )
            for name in (
                "mass",
                "radial_momentum",
                "angular_momentum",
                "total_energy",
            )
        }
    )
    source_agreement = {}
    for name in (
        "mass",
        "radial_momentum",
        "angular_momentum",
        "total_energy",
    ):
        actual = np.asarray(getattr(rhs_source, name), dtype=float)
        expected = np.asarray(getattr(stream, name), dtype=float)
        source_agreement[name] = float(
            np.max(np.abs(actual - expected))
            / max(float(np.max(np.abs(expected))), 1.0)
        )
    inner_flux_response = {
        name: float(
            getattr(profile_on.face_fluxes, name)[0]
            - getattr(profile_off.face_fluxes, name)[0]
        )
        for name in (
            "mass",
            "radial_momentum",
            "angular_momentum",
            "total_energy",
        )
    }
    return {
        "linearized_horizon_seconds": horizon,
        "source_rhs_agreement_relative": source_agreement,
        "instantaneous_inner_flux_response": inner_flux_response,
        "source_on": {
            "rhs": _rhs_summary(rhs_on, initial),
            "linearized_controller": _linearized_controller(
                context, grid, initial, correction, rhs_on, horizon
            ),
        },
        "source_off": {
            "rhs": _rhs_summary(rhs_off, initial),
            "linearized_controller": _linearized_controller(
                context, grid, initial, correction, rhs_off, horizon
            ),
        },
        "source_only": {
            "rhs": _rhs_summary(rhs_source, initial),
            "linearized_controller": _linearized_controller(
                context, grid, initial, correction, rhs_source, horizon
            ),
        },
    }


def _final_state_summary(
    restart_path: Path,
    context,
    correction,
) -> tuple[object, dict]:
    grid, restart = load_global_adaptive_restart(restart_path)
    primitives = recover_global_primitives(
        grid,
        restart.state,
        context.base.inner_params.M2_g,
        specific_mechanical_energy_correction=correction,
    )
    return restart.state, {
        "disk_mass": float(np.sum(restart.state.mass)),
        "internal_energy": float(
            np.sum(
                restart.state.mass
                * np.asarray(primitives.specific_internal_energy, dtype=float)
            )
        ),
        "total_conserved_energy": float(np.sum(restart.state.total_energy)),
    }


def _state_change_comparison(initial, source_on, source_off) -> dict:
    result = {}
    for name in (
        "mass",
        "radial_momentum",
        "angular_momentum",
        "total_energy",
    ):
        initial_values = np.asarray(getattr(initial, name), dtype=float)
        on_change = np.asarray(getattr(source_on, name), dtype=float) - initial_values
        off_change = (
            np.asarray(getattr(source_off, name), dtype=float) - initial_values
        )
        source_effect = np.asarray(
            getattr(source_on, name) - getattr(source_off, name), dtype=float
        )
        scale = max(float(np.linalg.norm(on_change)), 1.0e-300)
        result[name] = {
            "source_on_change_l2": float(np.linalg.norm(on_change)),
            "source_off_change_l2": float(np.linalg.norm(off_change)),
            "on_minus_off_l2": float(np.linalg.norm(source_effect)),
            "on_minus_off_over_source_on_change": float(
                np.linalg.norm(source_effect) / scale
            ),
            "source_off_over_source_on_change": float(
                np.linalg.norm(off_change) / scale
            ),
        }
    return result


def _accepted_dt_sequence(run: dict) -> list[float]:
    return [
        float(record["dt_used_over_loading_time"])
        for record in run["records"]
        if record["accepted"]
    ]


def main() -> None:
    arguments = _arguments()
    if arguments.target_loading_fraction <= 0.0:
        raise ValueError("target loading fraction must be positive")
    if arguments.maximum_nfev < 1:
        raise ValueError("maximum nfev must be positive")
    context, evaluation = _canonical_open_evaluation()
    grid, initial, correction, stream, stream_rate, provider = _prepared_case(
        context,
        evaluation,
        N_CELLS,
        inner_radius_rg=INNER_RADIUS_RG,
    )
    loading_time = float(np.sum(initial.mass) / stream_rate)
    target_time = arguments.target_loading_fraction * loading_time
    git = _git_metadata()
    mechanical = make_global_mechanical_energy_reference(
        grid,
        correction,
        initial,
        provenance={
            "case": "global-source-on-off-control-initial",
            "source": "corrected accepted-rate plunge mapping",
        },
    )
    initial_restart = GlobalAdaptiveRestart(
        state=initial,
        reference_state=initial,
        mechanical_reference=mechanical,
        elapsed_time=0.0,
        dt_next=target_time,
        accepted_steps=0,
        rejected_attempts=0,
        provenance={
            "case": "global-source-on-off-control-initial",
            "n_cells": N_CELLS,
            "inner_radius_rg": INNER_RADIUS_RG,
            "git": git,
            "target_loading_fraction": arguments.target_loading_fraction,
        },
    )
    initial_milestone = save_global_adaptive_milestone(
        ROOT / "outputs/checkpoints/milestones/global_source_on_off/initial",
        "global-source-on-off-initial",
        grid,
        initial_restart,
        metadata={"role": "shared immutable initial state"},
    )
    instantaneous = _instantaneous_decomposition(
        context,
        grid,
        initial,
        correction,
        stream,
        provider,
        target_time,
    )
    runs = {}
    restart_paths = {}
    for source_enabled, label in ((True, "source_on"), (False, "source_off")):
        restart_path = (
            ROOT
            / "outputs/checkpoints/global_source_on_off"
            / f"{label}_N64.npz"
        )
        restart_paths[label] = restart_path
        runs[label] = run_adaptive_campaign(
            context,
            evaluation,
            n_cells=N_CELLS,
            target_loading_fraction=arguments.target_loading_fraction,
            initial_dt_loading_fraction=arguments.target_loading_fraction,
            restart_path=restart_path,
            maximum_accepted_steps=arguments.maximum_accepted_steps,
            inner_radius_rg=INNER_RADIUS_RG,
            maximum_nfev=arguments.maximum_nfev,
            minimum_dt_loading_fraction=1.0e-10,
            reference_loading_time_seconds=loading_time,
            milestone_directory=(
                ROOT
                / "outputs/checkpoints/milestones/global_source_on_off"
                / label
            ),
            milestone_case=f"global-{label}",
            source_enabled=source_enabled,
        )
        if not runs[label]["target_reached"]:
            raise RuntimeError(f"{label} did not reach the matched target")
    on_state, on_storage = _final_state_summary(
        restart_paths["source_on"], context, correction
    )
    off_state, off_storage = _final_state_summary(
        restart_paths["source_off"], context, correction
    )
    on_dt = _accepted_dt_sequence(runs["source_on"])
    off_dt = _accepted_dt_sequence(runs["source_off"])
    report = {
        "git": git,
        "n_cells": N_CELLS,
        "inner_radius_rg": INNER_RADIUS_RG,
        "target_loading_fraction": arguments.target_loading_fraction,
        "loading_time_seconds": loading_time,
        "exact_common_physical_time_seconds": target_time,
        "initial_milestone": initial_milestone,
        "instantaneous_tendency_decomposition": instantaneous,
        "matched_trajectories": {
            "source_on": {**runs["source_on"], "storage": on_storage},
            "source_off": {**runs["source_off"], "storage": off_storage},
            "accepted_dt_sequence_source_on": on_dt,
            "accepted_dt_sequence_source_off": off_dt,
            "identical_accepted_dt_sequence": on_dt == off_dt,
            "state_change_comparison": _state_change_comparison(
                initial, on_state, off_state
            ),
            "disk_mass_on_minus_off": (
                on_storage["disk_mass"] - off_storage["disk_mass"]
            ),
            "disk_mass_on_minus_off_over_injected_mass": (
                on_storage["disk_mass"] - off_storage["disk_mass"]
            )
            / (stream_rate * target_time),
            "internal_energy_relative_difference": (
                on_storage["internal_energy"]
                / off_storage["internal_energy"]
                - 1.0
            ),
            "inner_flux_differences": {
                name: (
                    runs["source_on"]["final_inner_conserved_fluxes"][name]
                    - runs["source_off"]["final_inner_conserved_fluxes"][name]
                )
                for name in ("mass", "angular_momentum", "total_energy")
            },
            "sonic_radius_difference": (
                runs["source_on"]["sonic_resolution"]["sonic_radius"]
                - runs["source_off"]["sonic_resolution"]["sonic_radius"]
            ),
            "maximum_H_over_R_difference": (
                runs["source_on"]["maximum_H_over_R"]
                - runs["source_off"]["maximum_H_over_R"]
            ),
        },
    }
    output = arguments.output
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
