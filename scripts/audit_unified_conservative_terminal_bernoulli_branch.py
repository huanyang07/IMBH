"""Summarize physical diagnostics along the strict terminal-Bernoulli branch."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from imri_qpe.constants import C
from imri_qpe.layer3_minidisk_1d import (
    conservative_wind_escape_profile,
    reconstruct_conservative_state,
    unpack_conservative_state,
)
from imri_qpe.scales import eddington_luminosity
import run_unified_conservative_terminal_bernoulli_ladder as ladder


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = ROOT / "outputs/checkpoints/unified_conservative_terminal_bernoulli"
OUTPUT = ROOT / "outputs/tables/unified_conservative_terminal_bernoulli_branch_audit.json"
TARGETS = (0.10, 0.08, 0.06, 0.04, 0.035, 0.03, 0.025, 0.0225, 0.02)


def _path(target: float) -> Path:
    label = str(target).replace(".", "p")
    return CHECKPOINT_DIR / f"mdot5_Binf_{label}c2_cap0.3_N426.npz"


def _physics(state: np.ndarray, params) -> dict[str, float]:
    logu, logT, F, j, _epsilon, log_r_son, log_r = unpack_conservative_state(
        state, params
    )
    nodes = [
        reconstruct_conservative_state(
            log_r[idx], logu[idx], logT[idx], F[idx], j[idx], params.disk, params.flux_scales
        )
        for idx in range(log_r.size)
    ]
    viscous = []
    advective = []
    radiative = []
    radii = []
    widths = []
    for idx in range(log_r.size - 1):
        dx = float(log_r[idx + 1] - log_r[idx])
        middle = reconstruct_conservative_state(
            0.5 * float(log_r[idx] + log_r[idx + 1]),
            0.5 * float(logu[idx] + logu[idx + 1]),
            0.5 * float(logT[idx] + logT[idx + 1]),
            0.5 * float(F[idx] + F[idx + 1]),
            0.5 * float(j[idx] + j[idx + 1]),
            params.disk,
            params.flux_scales,
        )
        d_omega = float((nodes[idx + 1].Omega - nodes[idx].Omega) / dx)
        d_e = float((nodes[idx + 1].e - nodes[idx].e) / dx)
        d_rho = float((nodes[idx + 1].rho - nodes[idx].rho) / dx)
        tds = float(d_e - middle.P * d_rho / middle.rho**2)
        q_visc = float(-middle.W * d_omega)
        q_adv = float(-(middle.Sigma * middle.u / middle.R) * tds)
        area = float(2.0 * np.pi * middle.R**2)
        viscous.append(area * q_visc)
        advective.append(area * q_adv)
        radiative.append(area * middle.Q_rad)
        radii.append(middle.R / params.disk.r_g)
        widths.append(dx)

    viscous = np.asarray(viscous)
    advective = np.asarray(advective)
    radiative = np.asarray(radiative)
    radii = np.asarray(radii)
    widths = np.asarray(widths)
    viscous_total = float(np.sum(widths * viscous))
    inner = radii <= 20.0
    inner_viscous = float(np.sum(widths[inner] * viscous[inner]))
    return {
        "F_outer": float(F[-1]),
        "Rson_rg": float(np.exp(log_r_son) / params.disk.r_g),
        "max_H_over_R": float(max(node.H / node.R for node in nodes)),
        "Lrad_over_LEdd": float(
            np.sum(widths * radiative)
            / eddington_luminosity(params.disk.M2_g, kappa=params.disk.kappa)
        ),
        "f_adv_global": float(np.sum(widths * advective) / viscous_total),
        "f_adv_inner_Rle20": float(
            np.sum(widths[inner] * advective[inner]) / inner_viscous
        ),
    }


def run() -> list[dict[str, object]]:
    rows = []
    previous = None
    for target in TARGETS:
        with np.load(_path(target)) as data:
            state = np.asarray(data["x"], dtype=float)
            grid = np.asarray(data["custom_grid_xi"], dtype=float)
            metadata = json.loads(str(np.asarray(data["row_json"]).item()))
        params = ladder._params(grid, target)
        wind = conservative_wind_escape_profile(
            state,
            params,
            target_terminal_bernoulli=target * C**2,
        )
        row = {
            "target_terminal_bernoulli_over_c2": target,
            "maximum_residual": float(max(metadata["final"].values())),
            "wind_over_mdot_inner": float(
                np.sum(wind["wind_mass"]) / params.flux_scales.mdot
            ),
            "cap_active_intervals": int(np.count_nonzero(wind["wind_cap_active"])),
            "state_rms_change_from_previous": (
                None
                if previous is None
                else float(np.sqrt(np.mean((state - previous) ** 2)))
            ),
            "physics": _physics(state, params),
        }
        rows.append(row)
        previous = state
        print(json.dumps(row, sort_keys=True), flush=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    return rows


if __name__ == "__main__":
    run()
