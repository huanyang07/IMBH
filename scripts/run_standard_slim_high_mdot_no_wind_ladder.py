"""High-Mdot no-wind standard slim-disk continuation ladder.

This is a thin configuration wrapper around
``run_standard_slim_adaptive_mdot_ladder.py``.  It keeps the mature predictor,
Newton polish, pressure-supported outer closure, and adaptive-grid machinery,
but writes to high-Mdot-specific output locations.
"""

from __future__ import annotations

import os


DEFAULTS = {
    "IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_ANCHOR": (
        "outputs/checkpoints/slim_benchmark_adaptive_outer_mesh_mdot1_scan/s8_mdot_1_N640.npz"
    ),
    "IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_TABLE": "outputs/tables/slim_benchmark_high_mdot_no_wind_ladder.md",
    "IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_FIGURE": "outputs/figures/slim_benchmark_high_mdot_no_wind_ladder.png",
    "IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_CHECKPOINTS": "outputs/checkpoints/slim_benchmark_high_mdot_no_wind_ladder",
    "IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_DOWN_TARGET": "1.0",
    "IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_UP_TARGET": "10.0",
    "IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_START_STEP": "0.025",
    "IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_MAX_STEP": "0.08",
    "IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_MIN_STEP": "0.0025",
    "IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_GROW": "1.18",
    "IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_SHRINK": "0.5",
    "IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_MAX_ATTEMPTS": "80",
    "IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_ACCEPTANCE_TOL": "1e-5",
    "IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_ANCHOR_TOL": "3e-6",
    "IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_CURRENT_TOL": "5e-6",
    "IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_SCOUT_TOL": "1e-5",
    "IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_SOURCE_NFEV": "1200",
    "IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_POLISH_NFEV": "3200",
    "IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_FALLBACK_LSQ_NFEV": "500",
    "IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_NEWTON_MAX_ITER": "28",
    "IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_NEWTON_MAX_STEP_NORM": "0.16",
    "IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_NEWTON_LINEAR_SOLVER": "regularized_lsmr",
    "IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_NEWTON_BLOCK_JACOBIAN": "1",
    "IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_LSQ_BLOCK_JACOBIAN": "0",
    "IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_OUTER_CLOSURE": "pressure_one_sided",
    "IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_OUTER_SLOPE_REFRESHES": "4",
    "IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_OUTER_SLOPE_REFRESH_REPOLISH": "1",
    "IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_GRID_POWER": "0.6",
    "IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_USE_SECANT": "1",
    "IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_SKIP_FINAL_IF_CURRENT": "1",
    "IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_INJECTION_POLICY": "if_better",
    "IMBH_STANDARD_SLIM_ADAPTIVE_MDOT_VERBOSE_PHASES": "0",
}


for key, value in DEFAULTS.items():
    os.environ.setdefault(key, value)


from run_standard_slim_adaptive_mdot_ladder import main  # noqa: E402


if __name__ == "__main__":
    main()
