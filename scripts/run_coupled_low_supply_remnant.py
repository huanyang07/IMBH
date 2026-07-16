"""Continue the coupled open control to a low-throughput remnant state."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    audit_coupled_open_rank,
    continue_coupled_open_supply,
    evaluate_coupled_open_overflow_residual,
)
from imri_qpe.scales import eddington_luminosity

from run_coupled_inner_outer_mesh_certification import _load_source
from run_coupled_open_overflow_continuation import _open_context


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "results/canonical/coupled_open_overflow_eigenvalue"
DEFAULT_OUTPUT = ROOT / "outputs/tables/coupled_low_supply_remnant.json"
DEFAULT_CHECKPOINT = (
    ROOT / "outputs/checkpoints/coupled_low_supply_remnant/state_Ninner96_Nouter64.npz"
)
SUPPLY_FRACTIONS = (0.5, 0.25, 0.125, 0.075, 0.05)
INNER_THROUGHPUT_LIMIT = 1.0e-2


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--maximum-nfev", type=int, default=200)
    return parser.parse_args()


def load_canonical_open_root():
    """Load and verify the accepted Ninner96/Nouter64 open root."""

    base, _wall_state = _load_source()
    context = _open_context(base, 1.0)
    with np.load(CANONICAL / "Ninner96_Nouter64.npz") as data:
        state = np.asarray(data["state"], dtype=float)
    evaluation = evaluate_coupled_open_overflow_residual(state, context)
    maximum = float(np.max(np.abs(evaluation.residual)))
    if maximum > 1.0e-7:
        raise RuntimeError(
            f"canonical open root no longer closes: {maximum:.3e}"
        )
    return context, state, evaluation


def continue_to_low_supply(*, maximum_nfev: int = 200):
    """Return the fixed-stage low-supply continuation and final evaluation."""

    context, state, initial = load_canonical_open_root()
    result = continue_coupled_open_supply(
        state,
        context,
        SUPPLY_FRACTIONS,
        tolerance=1.0e-7,
        max_nfev=maximum_nfev,
    )
    final = evaluate_coupled_open_overflow_residual(
        result.state,
        result.context,
    )
    return context, initial, result, final


def _rank_summary(audit) -> dict:
    singular = np.asarray(audit.singular_values, dtype=float)
    return {
        "jacobian_shape": list(audit.jacobian_shape),
        "ranks_by_relative_threshold": audit.ranks_by_relative_threshold,
        "condition_estimate": audit.condition_estimate,
        "smallest_six_singular_values": singular[-6:].tolist(),
        "preboundary_nullity": audit.preboundary_nullity,
        "interface_response_rank": audit.interface_response_rank,
        "sonic_rank": audit.sonic_rank,
    }


def main() -> None:
    arguments = _arguments()
    initial_context, initial, result, final = continue_to_low_supply(
        maximum_nfev=arguments.maximum_nfev
    )
    initial_supply = float(initial_context.mass_flux_scale)
    final_supply = float(result.context.mass_flux_scale)
    inner_fraction = float(final.mdot_inner / initial_supply)
    outer_fraction = float(final.mdot_outer / initial_supply)
    rank = audit_coupled_open_rank(result.state, result.context)
    inner = final.base.inner_profile
    outer = final.base.outer_energy_profile
    luminosity = float(
        (
            np.trapezoid(2.0 * np.pi * inner.R * inner.Q_rad, inner.R)
            + np.sum(outer.radiative_loss_rate_cells)
        )
        / eddington_luminosity(result.context.base.inner_params.M2_g)
    )
    full_rank = (
        rank.ranks_by_relative_threshold["1e-10"] == result.state.size
    )
    rank_gate = bool(
        full_rank
        and rank.preboundary_nullity == 2
        and rank.interface_response_rank == 2
        and rank.sonic_rank == 2
    )
    throughput_gate = abs(inner_fraction) <= INNER_THROUGHPUT_LIMIT
    accepted = bool(result.accepted and rank_gate and throughput_gate)
    output = {
        "supply_fractions": list(SUPPLY_FRACTIONS),
        "initial_stream_rate": initial_supply,
        "final_stream_rate": final_supply,
        "final_supply_over_initial": final_supply / initial_supply,
        "stages": [asdict(stage) for stage in result.stages],
        "accepted": accepted,
        "continuation_accepted": result.accepted,
        "inner_throughput_limit_over_initial_supply": (
            INNER_THROUGHPUT_LIMIT
        ),
        "final_inner_mdot_over_initial_supply": inner_fraction,
        "final_inner_mdot_over_stage_supply": float(
            final.mdot_inner / final_supply
        ),
        "final_outer_mdot_over_initial_supply": outer_fraction,
        "final_outer_mdot_over_stage_supply": float(
            final.mdot_outer / final_supply
        ),
        "maximum_residual": float(np.max(np.abs(final.residual))),
        "sonic_radius_rg": float(
            inner.R[0] / result.context.base.inner_params.r_g
        ),
        "composite_luminosity_over_eddington": luminosity,
        "maximum_outer_H_over_R": float(
            np.max(outer.H / result.context.base.outer_grid.centers)
        ),
        "rank_gate": rank_gate,
        "rank_audit": _rank_summary(rank),
    }
    checkpoint = arguments.checkpoint
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        checkpoint,
        state=result.state,
        supply_fraction=SUPPLY_FRACTIONS[-1],
        initial_stream_rate=initial_supply,
        final_stream_rate=final_supply,
        accepted=accepted,
        maximum_residual=output["maximum_residual"],
    )
    output["checkpoint"] = str(checkpoint.relative_to(ROOT))
    report = arguments.output
    if not report.is_absolute():
        report = ROOT / report
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(report)
    if not accepted:
        raise RuntimeError("low-supply remnant failed its declared gates")


if __name__ == "__main__":
    main()
