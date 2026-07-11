"""Test whether a projected pressure-supported reservoir reaches a fixed point."""

from __future__ import annotations

import json
from pathlib import Path

from run_two_domain_interface_sweep import _solve_one


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/tables/pressure_supported_interface_pilot.json"


def run() -> dict[str, object]:
    configurations = [
        (64, 0.10, 0.04),
        (64, 0.10, 0.08),
        (64, 0.10, 0.16),
        (128, 0.10, 0.04),
        (128, 0.10, 0.08),
        (128, 0.10, 0.16),
        (128, 0.05, 0.08),
        (128, 0.20, 0.08),
    ]
    rows = []
    for resolution, damping, smoothing in configurations:
        try:
            row = _solve_one(
                40.0,
                resolution,
                pressure_supported=True,
                pressure_damping=damping,
                pressure_smoothing_log_width=smoothing,
                pressure_max_iterations=100,
            )
            row["failure"] = None
        except (RuntimeError, ValueError) as error:
            row = {
                "N_reservoir": resolution,
                "target_interface_rg": 40.0,
                "pressure_damping": damping,
                "pressure_smoothing_log_width": smoothing,
                "converged": False,
                "failure": f"{type(error).__name__}: {error}",
            }
        rows.append(row)
    successful = [row for row in rows if row["converged"]]
    successful_resolutions = {
        int(row["N_reservoir"]) for row in successful
    }
    attempted_resolutions = {
        int(row["N_reservoir"]) for row in rows
    }
    if len(successful) == len(rows):
        classification = "SUPPORTED"
    elif successful and max(successful_resolutions) < max(attempted_resolutions):
        classification = "COARSE_GRID_ONLY_NOT_MESH_SUPPORTED"
    else:
        classification = "REJECTED_FIXED_POINT_ITERATION"
    result: dict[str, object] = {
        "target_interface_rg": 40.0,
        "pressure_support_stages": [0.10, 0.25, 0.50, 0.75, 1.0],
        "all_converged": len(successful) == len(rows),
        "converged_count": len(successful),
        "attempt_count": len(rows),
        "classification": classification,
        "rows": rows,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
