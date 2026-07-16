"""Run the bounded sonic-gradient and early-evolution WP4 audit."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    continue_transonic_supersonic_plunge,
    evaluate_coupled_open_overflow_residual,
    evaluate_global_rusanov_profile,
    global_fixed_radius_diagnostics,
    recover_global_primitives,
)
from imri_qpe.layer3_minidisk_1d.transonic_local import (
    sonic_derivative_branches,
)

from run_coupled_inner_outer_mesh_certification import _load_source
from run_coupled_open_overflow_continuation import _open_context
from run_global_physical_open_preflight import _canonical_open_evaluation
from run_global_roche_adaptive_preflight import run_adaptive_campaign
from run_global_roche_loading_preflight import _prepared_case


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "results/canonical/coupled_open_overflow_eigenvalue"
INNER_RADIUS_RG = 4.5
FIXED_RADII_RG = (4.5, 4.65, 4.75, 5.0)
EVOLUTION_LOADING_FRACTION = 1.0e-9
VARIANTS = {
    "baseline": {
        "sonic_offset": 1.0e-6,
        "rtol": 1.0e-9,
        "atol": 1.0e-11,
        "maximum_log_step": 5.0e-3,
    },
    "larger_offset": {
        "sonic_offset": 1.0e-5,
        "rtol": 1.0e-9,
        "atol": 1.0e-11,
        "maximum_log_step": 5.0e-3,
    },
    "smaller_offset": {
        "sonic_offset": 1.0e-7,
        "rtol": 1.0e-9,
        "atol": 1.0e-11,
        "maximum_log_step": 5.0e-3,
    },
    "tight": {
        "sonic_offset": 1.0e-7,
        "rtol": 1.0e-11,
        "atol": 1.0e-13,
        "maximum_log_step": 1.0e-3,
    },
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-directory", type=Path)
    parser.add_argument("--skip-evolution", action="store_true")
    parser.add_argument("--maximum-nfev", type=int, default=600)
    return parser.parse_args()


def _accepted_evaluations():
    base, _state = _load_source()
    context_96 = _open_context(base, 1.0)
    with np.load(CANONICAL / "Ninner96_Nouter64.npz") as data:
        state_96 = np.asarray(data["state"], dtype=float)
    evaluation_96 = evaluate_coupled_open_overflow_residual(
        state_96, context_96
    )
    context_144, evaluation_144 = _canonical_open_evaluation()
    return (
        ("Ninner096_Nouter064", context_96, evaluation_96),
        ("Ninner144_Nouter096", context_144, evaluation_144),
    )


def _interpolate(radius: float, nodes, values, *, positive: bool) -> float:
    x = np.log(np.asarray(nodes, dtype=float))
    work = np.asarray(values, dtype=float)
    if positive:
        work = np.log(work)
    result = float(np.interp(np.log(radius), x, work))
    return float(np.exp(result)) if positive else result


def _fixed_radius_stationary(plunge, r_g: float) -> list[dict]:
    rows = []
    for radius_rg in FIXED_RADII_RG:
        radius = radius_rg * r_g
        rows.append(
            {
                "radius_rg": radius_rg,
                "surface_density": _interpolate(
                    radius, plunge.R, plunge.Sigma, positive=True
                ),
                "temperature": _interpolate(
                    radius, plunge.R, plunge.T, positive=True
                ),
                "radial_mach_number": _interpolate(
                    radius,
                    plunge.R,
                    plunge.radial_mach_number,
                    positive=False,
                ),
                "radial_velocity": -_interpolate(
                    radius, plunge.R, plunge.u, positive=True
                ),
                "H_over_R": _interpolate(
                    radius, plunge.R, plunge.H, positive=True
                )
                / radius,
            }
        )
    return rows


def _content_hash(plunge) -> str:
    digest = hashlib.sha256()
    for name in (
        "R",
        "u",
        "T",
        "Sigma",
        "H",
        "Omega",
        "radial_mach_number",
        "selected_sonic_gradient",
        "resolved_outer_gradient",
    ):
        values = np.ascontiguousarray(
            np.asarray(getattr(plunge, name), dtype="<f8")
        )
        digest.update(name.encode("ascii"))
        digest.update(values.tobytes())
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_seed(seed_directory: Path, label: str, variant: str, plunge) -> dict:
    seed_directory.mkdir(parents=True, exist_ok=True)
    content_hash = _content_hash(plunge)
    path = seed_directory / f"{label}_{variant}_{content_hash[:12]}.npz"
    if not path.exists():
        np.savez_compressed(
            path,
            R=plunge.R,
            u=plunge.u,
            T=plunge.T,
            Sigma=plunge.Sigma,
            H=plunge.H,
            rho=plunge.rho,
            P=plunge.P,
            Pi=plunge.Pi,
            e=plunge.e,
            tau=plunge.tau,
            Omega=plunge.Omega,
            Omega_K=plunge.Omega_K,
            l=plunge.l,
            W=plunge.W,
            effective_sound_speed=plunge.effective_sound_speed,
            radial_mach_number=plunge.radial_mach_number,
            incoming_characteristics=plunge.incoming_characteristics,
            selected_sonic_gradient=plunge.selected_sonic_gradient,
            resolved_outer_gradient=plunge.resolved_outer_gradient,
            sonic_gradient_mismatch=plunge.sonic_gradient_mismatch,
            maximum_scaled_differential_residual=(
                plunge.maximum_scaled_differential_residual
            ),
            sonic_offset=plunge.sonic_offset,
        )
    return {
        "path": str(path.relative_to(ROOT)),
        "content_sha256": content_hash,
        "file_sha256": _file_sha256(path),
    }


def _regular_branches(inner_profile, params) -> list[dict]:
    sonic_state = np.log(
        np.asarray([inner_profile.u[0], inner_profile.T[0]], dtype=float)
    )
    outer_dx = float(np.log(inner_profile.R[1] / inner_profile.R[0]))
    outer_gradient = np.asarray(
        [
            np.log(inner_profile.u[1] / inner_profile.u[0]) / outer_dx,
            np.log(inner_profile.T[1] / inner_profile.T[0]) / outer_dx,
        ],
        dtype=float,
    )
    branches = sonic_derivative_branches(
        float(np.log(inner_profile.sonic_radius)),
        sonic_state,
        inner_profile.lambda0,
        params,
        gradient_center=outer_gradient,
        half_width=100.0,
        scan_points=801,
    )
    return [
        {
            "kind": branch.kind,
            "a": branch.a,
            "gradient": branch.gradient.tolist(),
            "distance_to_resolved_outer_gradient": float(
                np.linalg.norm(branch.gradient - outer_gradient)
            ),
            "lhopital_raw": branch.lhopital_raw,
            "lhopital_normalized": branch.lhopital_normalized,
        }
        for branch in branches
    ]


def _initial_global_record(context, evaluation) -> tuple[dict, float]:
    grid, state, correction, stream, stream_rate, provider = _prepared_case(
        context,
        evaluation,
        64,
        inner_radius_rg=INNER_RADIUS_RG,
    )
    mass = context.base.inner_params.M2_g
    primitives = recover_global_primitives(
        grid,
        state,
        mass,
        specific_mechanical_energy_correction=correction,
    )
    profile = evaluate_global_rusanov_profile(
        grid,
        state,
        mass,
        reference_state=state,
        boundary_mode="roche_outer",
        alpha=context.base.alpha,
        stress_boundary_mode="outer_zero_torque",
        include_radiative_cooling=True,
        include_vertical_column_work=True,
        external_sources=stream,
        primitives=primitives,
        outer_overflow_provider=provider,
        specific_mechanical_energy_correction=correction,
    )
    fixed = global_fixed_radius_diagnostics(
        grid,
        primitives,
        profile.face_fluxes,
        mass,
        [
            radius * context.base.inner_params.r_g
            for radius in FIXED_RADII_RG
            if radius > INNER_RADIUS_RG
        ],
    )
    loading_time = float(np.sum(state.mass) / stream_rate)
    return (
        {
            "n_cells": 64,
            "loading_time_seconds": loading_time,
            "inner_conserved_fluxes": {
                "mass": float(profile.face_fluxes.mass[0]),
                "angular_momentum": float(
                    profile.face_fluxes.angular_momentum[0]
                ),
                "total_energy": float(profile.face_fluxes.total_energy[0]),
                "mass_over_supply": float(
                    profile.face_fluxes.mass[0] / stream_rate
                ),
            },
            "fixed_radius_diagnostics": [asdict(item) for item in fixed],
        },
        loading_time,
    )


def _run_early_evolution(
    cases,
    loading_times: dict[str, float],
    reference_loading_time: float,
    maximum_nfev: int,
) -> list[dict]:
    target_time = EVOLUTION_LOADING_FRACTION * reference_loading_time
    runs = []
    for label, context, evaluation in cases:
        target_fraction = target_time / loading_times[label]
        run = run_adaptive_campaign(
            context,
            evaluation,
            n_cells=64,
            target_loading_fraction=target_fraction,
            initial_dt_loading_fraction=target_fraction,
            restart_path=(
                ROOT
                / "outputs/checkpoints/sonic_gradient_audit"
                / f"early_evolution_{label}.npz"
            ),
            maximum_accepted_steps=4,
            inner_radius_rg=INNER_RADIUS_RG,
            maximum_nfev=maximum_nfev,
            minimum_dt_loading_fraction=1.0e-10,
            reference_loading_time_seconds=reference_loading_time,
            milestone_directory=(
                ROOT
                / "outputs/checkpoints/milestones/sonic_gradient_audit"
                / label
            ),
            milestone_case=f"sonic-gradient-audit-{label}",
        )
        if not run["target_reached"]:
            raise RuntimeError(f"{label} did not reach the early audit time")
        runs.append({"source_resolution": label, **run})
    return runs


def _resolution_comparison(rows: list[dict]) -> dict:
    coarse, fine = rows
    coarse_fixed = {
        item["radius_rg"]: item
        for item in coarse["variants"]["baseline"][
            "fixed_radius_stationary"
        ]
    }
    fine_fixed = {
        item["radius_rg"]: item
        for item in fine["variants"]["baseline"][
            "fixed_radius_stationary"
        ]
    }
    return {
        "coarse": coarse["source_resolution"],
        "fine": fine["source_resolution"],
        "sonic_gradient_mismatch_ratio": (
            fine["variants"]["baseline"]["sonic_gradient_mismatch"]
            / coarse["variants"]["baseline"]["sonic_gradient_mismatch"]
        ),
        "fixed_radius_differences": [
            {
                "radius_rg": radius,
                "radial_mach_difference": (
                    coarse_fixed[radius]["radial_mach_number"]
                    - fine_fixed[radius]["radial_mach_number"]
                ),
                "log_surface_density_difference": float(
                    np.log(
                        coarse_fixed[radius]["surface_density"]
                        / fine_fixed[radius]["surface_density"]
                    )
                ),
                "log_temperature_difference": float(
                    np.log(
                        coarse_fixed[radius]["temperature"]
                        / fine_fixed[radius]["temperature"]
                    )
                ),
            }
            for radius in FIXED_RADII_RG
        ],
    }


def main() -> None:
    arguments = _arguments()
    if arguments.maximum_nfev < 1:
        raise ValueError("maximum nfev must be positive")
    seed_directory = arguments.seed_directory
    if seed_directory is None:
        seed_directory = ROOT / "outputs/checkpoints/sonic_gradient_audit/seeds"
    elif not seed_directory.is_absolute():
        seed_directory = ROOT / seed_directory
    cases = _accepted_evaluations()
    rows = []
    loading_times = {}
    seed_manifest = []
    for label, context, evaluation in cases:
        inner = evaluation.base.inner_profile
        params = evaluation.trial_context.inner_params
        variants = {}
        baseline = None
        for variant, options in VARIANTS.items():
            plunge = continue_transonic_supersonic_plunge(
                inner,
                params,
                INNER_RADIUS_RG * params.r_g,
                n_nodes=128,
                **options,
            )
            seed = _save_seed(seed_directory, label, variant, plunge)
            seed_manifest.append(
                {"source_resolution": label, "variant": variant, **seed}
            )
            if variant == "baseline":
                baseline = plunge
            variants[variant] = {
                "options": options,
                "seed": seed,
                "selected_sonic_gradient": (
                    plunge.selected_sonic_gradient.tolist()
                ),
                "resolved_outer_gradient": (
                    plunge.resolved_outer_gradient.tolist()
                ),
                "sonic_gradient_mismatch": plunge.sonic_gradient_mismatch,
                "maximum_scaled_differential_residual": (
                    plunge.maximum_scaled_differential_residual
                ),
                "inner_state": {
                    "surface_density": float(plunge.Sigma[0]),
                    "temperature": float(plunge.T[0]),
                    "radial_velocity": float(-plunge.u[0]),
                    "radial_mach_number": float(
                        plunge.radial_mach_number[0]
                    ),
                    "incoming_characteristics": int(
                        plunge.incoming_characteristics[0]
                    ),
                },
                "fixed_radius_stationary": _fixed_radius_stationary(
                    plunge, params.r_g
                ),
            }
        if baseline is None:
            raise RuntimeError("baseline plunge was not evaluated")
        for variant, record in variants.items():
            if variant == "baseline":
                record["maximum_log_difference_from_baseline"] = {
                    "radial_speed": 0.0,
                    "temperature": 0.0,
                    "surface_density": 0.0,
                }
                continue
            with np.load(ROOT / record["seed"]["path"]) as data:
                record["maximum_log_difference_from_baseline"] = {
                    "radial_speed": float(
                        np.max(np.abs(np.log(data["u"] / baseline.u)))
                    ),
                    "temperature": float(
                        np.max(np.abs(np.log(data["T"] / baseline.T)))
                    ),
                    "surface_density": float(
                        np.max(np.abs(np.log(data["Sigma"] / baseline.Sigma)))
                    ),
                }
        initial_global, loading_time = _initial_global_record(
            context, evaluation
        )
        loading_times[label] = loading_time
        rows.append(
            {
                "source_resolution": label,
                "sonic_radius_rg": float(inner.sonic_radius / params.r_g),
                "regular_sonic_derivative_roots": _regular_branches(
                    inner, params
                ),
                "variants": variants,
                "initial_global_mapping": initial_global,
            }
        )
    reference_loading_time = loading_times[rows[-1]["source_resolution"]]
    evolution = []
    if not arguments.skip_evolution:
        evolution = _run_early_evolution(
            cases,
            loading_times,
            reference_loading_time,
            arguments.maximum_nfev,
        )
    manifest_path = seed_directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(seed_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = {
        "inner_radius_rg": INNER_RADIUS_RG,
        "fixed_radii_rg": list(FIXED_RADII_RG),
        "variants": VARIANTS,
        "seed_manifest": str(manifest_path.relative_to(ROOT)),
        "source_resolutions": rows,
        "resolution_comparison": _resolution_comparison(rows),
        "early_evolution": {
            "skipped": arguments.skip_evolution,
            "reference_loading_fraction": EVOLUTION_LOADING_FRACTION,
            "reference_loading_time_seconds": reference_loading_time,
            "exact_physical_time_seconds": (
                EVOLUTION_LOADING_FRACTION * reference_loading_time
            ),
            "runs": evolution,
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
