"""Audit power-primary wind transport and terminal Bernoulli at eta_E=8."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from imri_qpe.constants import C
from imri_qpe.layer3_minidisk_1d import (
    conservative_residual,
    conservative_residual_audit,
    conservative_transport_profile,
    conservative_wind_escape_profile,
)
import run_unified_conservative_block_eta_continuation as continuation


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = ROOT / "outputs/checkpoints/unified_conservative_block_eta"
OUTPUT = ROOT / "outputs/tables/unified_conservative_wind_power_escape_audit.json"
N_VALUES = (426, 512, 640)


def _weighted_fraction(mask: np.ndarray, weights: np.ndarray) -> float:
    total = float(np.sum(weights))
    return 0.0 if total <= 0.0 else float(np.sum(weights[mask]) / total)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    total = float(np.sum(weights))
    return float("nan") if total <= 0.0 else float(np.sum(values * weights) / total)


def run() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for n_nodes in N_VALUES:
        path = CHECKPOINT_DIR / f"mdot5_eta8_block_mass5_N{n_nodes}.npz"
        with np.load(path) as data:
            state = np.asarray(data["x"], dtype=float)
            grid = np.asarray(data["custom_grid_xi"], dtype=float)
        power = continuation._problem(grid, 8.0)
        carried = replace(power, wind_energy_transport_mode="carried")
        residual_power = conservative_residual(state, power)
        residual_carried = conservative_residual(state, carried)
        transport = conservative_transport_profile(state, power)
        escape = conservative_wind_escape_profile(state, power)
        wind_mass = np.maximum(np.asarray(transport["wind_mass"], dtype=float), 0.0)
        active = wind_mass > max(float(np.sum(wind_mass)) * 1.0e-12, 1.0e-300)
        escaping = np.asarray(escape["escaping"], dtype=bool)
        required = np.asarray(escape["required_launch_energy"], dtype=float)
        prescribed = np.asarray(escape["prescribed_launch_energy"], dtype=float)
        margin = np.asarray(escape["terminal_margin"], dtype=float)
        terminal_speed = np.asarray(escape["terminal_speed"], dtype=float)
        energy_split_error = np.asarray(
            transport["wind_energy"]
            - transport["wind_base_energy"]
            - transport["wind_launch_energy"],
            dtype=float,
        )
        energy_scale = np.maximum(
            np.abs(np.asarray(transport["wind_energy"], dtype=float)),
            np.finfo(float).tiny,
        )

        active_required = required[active]
        active_prescribed = prescribed[active]
        positive_required = active_required > 0.0
        ratio = (
            active_prescribed[positive_required] / active_required[positive_required]
            if np.any(positive_required)
            else np.asarray([], dtype=float)
        )
        row = {
            "N": n_nodes,
            "power_audit": {
                "maximum": conservative_residual_audit(state, power).maximum,
                "max_residual_mode_difference": float(
                    np.max(np.abs(residual_power - residual_carried))
                ),
                "max_wind_energy_split_error": float(
                    np.max(np.abs(energy_split_error))
                ),
                "max_relative_wind_energy_split_error": float(
                    np.max(np.abs(energy_split_error) / energy_scale)
                ),
            },
            "wind": {
                "active_intervals": int(np.count_nonzero(active)),
                "wind_over_mdot_inner": float(
                    np.sum(wind_mass) / power.flux_scales.mdot
                ),
                "mass_weighted_escaping_fraction": _weighted_fraction(
                    active & escaping, wind_mass
                ),
                "mass_weighted_bound_fraction": _weighted_fraction(
                    active & ~escaping, wind_mass
                ),
                "mass_weighted_superluminal_equivalent_fraction": _weighted_fraction(
                    active & (terminal_speed > C), wind_mass
                ),
                "mass_weighted_margin_over_c2": _weighted_mean(
                    margin[active] / C**2, wind_mass[active]
                ),
                "mass_weighted_terminal_speed_over_c": _weighted_mean(
                    terminal_speed[active] / C, wind_mass[active]
                ),
                "minimum_margin_over_c2_active": float(np.min(margin[active]) / C**2),
                "maximum_margin_over_c2_active": float(np.max(margin[active]) / C**2),
                "minimum_terminal_speed_over_c_active": float(
                    np.min(terminal_speed[active]) / C
                ),
                "maximum_terminal_speed_over_c_active": float(
                    np.max(terminal_speed[active]) / C
                ),
                "radius_of_maximum_terminal_speed_rg": float(
                    np.asarray(escape["R_mid_rg"], dtype=float)[
                        np.argmax(terminal_speed)
                    ]
                ),
                "minimum_prescribed_over_required_positive": (
                    None if ratio.size == 0 else float(np.min(ratio))
                ),
                "maximum_prescribed_over_required_positive": (
                    None if ratio.size == 0 else float(np.max(ratio))
                ),
            },
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    return rows


if __name__ == "__main__":
    run()
