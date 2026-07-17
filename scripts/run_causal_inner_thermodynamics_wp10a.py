"""Audit causal gas+radiation characteristics on the WP9 inner profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from imri_qpe.constants import C
from imri_qpe.layer3_minidisk_1d import (
    audit_causal_inner_characteristics,
    gas_radiation_relativistic_sound_speed_squared,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "outputs/tables/global_inner_boundary_architecture_gate.json"
DEFAULT_OUTPUT = ROOT / "outputs/tables/causal_inner_thermodynamics_wp10a.json"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    arguments = _arguments()
    with _absolute(arguments.input).open(encoding="utf-8") as stream:
        source = json.load(stream)

    rows = []
    for old in source["architecture_a"]["excision_rows"]:
        sound_squared = gas_radiation_relativistic_sound_speed_squared(
            old["density"], old["temperature"]
        )
        sound_speed = float(np.sqrt(sound_squared))
        radial_velocity = float(old["radial_speed_over_c"] * C)
        audit = audit_causal_inner_characteristics(
            radial_velocity,
            sound_speed,
        )
        rows.append(
            {
                "radius_rg": float(old["radius_rg"]),
                "radial_speed_over_c": float(old["radial_speed_over_c"]),
                "azimuthal_speed_over_c": float(
                    old["azimuthal_speed_over_c"]
                ),
                "total_speed_over_c": float(old["total_speed_over_c"]),
                "full_velocity_subluminal": bool(
                    old["subluminal_total_speed"]
                ),
                "newtonian_sound_speed_over_c": float(
                    old["sound_speed_over_c"]
                ),
                "relativistic_sound_speed_over_c": sound_speed / C,
                "relativistic_acoustic_mach": audit.radial_mach_number,
                "relativistic_characteristic_speeds_over_c": [
                    value / C for value in audit.characteristic_speeds
                ],
                "incoming_characteristics": audit.incoming_characteristics,
                "causally_outgoing": audit.causally_outgoing,
            }
        )

    causal_rows = [row for row in rows if row["causally_outgoing"]]
    full_state_rows = [
        row
        for row in causal_rows
        if row["full_velocity_subluminal"]
    ]
    all_sound_speeds_subluminal = all(
        row["relativistic_sound_speed_over_c"] < 1.0 for row in rows
    )
    output = {
        "derivative": (
            "a2 = c2 (dP/drho)_s / (c2 + e + P/rho)"
        ),
        "rows": rows,
        "all_sound_speeds_subluminal": all_sound_speeds_subluminal,
        "first_audited_causally_outgoing_radius_rg": (
            None if not causal_rows else causal_rows[0]["radius_rg"]
        ),
        "thermodynamic_prototype_passed": all_sound_speeds_subluminal,
        "radial_characteristic_prototype_passed": bool(causal_rows),
        "full_state_excision_candidate_exists": bool(full_state_rows),
        "production_ready": False,
        "blocking_reason": (
            "the global conservative flux and stationary plunge do not yet "
            "use the same relativistic system"
        ),
    }
    destination = _absolute(arguments.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
