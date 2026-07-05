"""Freeze the current Mdot=5 stream-fed energy-wind anchor diagnostics."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_standard_slim_stream_mass_annulus_scan as scan  # noqa: E402
from imri_qpe.parameters import FiducialParams  # noqa: E402
from imri_qpe.scales import eddington_mdot  # noqa: E402


ANCHORS: tuple[tuple[str, str], ...] = (
    (
        "ewind_0",
        "outputs/checkpoints/high_mdot_stream_m5_compact_N896_050_to080_no_energy_merit/"
        "m5n896fast2_mass_0p8_torque_0p005_mdot_5_N896.npz",
    ),
    (
        "ewind_0p50",
        "outputs/checkpoints/high_mdot_stream_m5_fs080_smooth_energy_wind_chi099_w005_eta0_windjac_05_10_N896/"
        "m5smooth_ewind_eta0_chi099_w005_windjac05_10_mass_0p8_wind_0_heat_0_ewind_0p5_chi_0p99_wfrac_0p005_torque_0p005_mdot_5_N896.npz",
    ),
    (
        "ewind_0p80",
        "outputs/checkpoints/high_mdot_stream_m5_fs080_smooth_energy_wind_chi099_w005_eta0_windjac_05_10_N896/"
        "m5smooth_ewind_eta0_chi099_w005_windjac05_10_mass_0p8_wind_0_heat_0_ewind_0p8_chi_0p99_wfrac_0p005_torque_0p005_mdot_5_N896.npz",
    ),
    (
        "ewind_0p98",
        "outputs/checkpoints/high_mdot_stream_m5_fs080_smooth_energy_wind_chi099_w005_eta0_windjac_098_N896/"
        "m5smooth_ewind_eta0_chi099_w005_windjac098_mass_0p8_wind_0_heat_0_ewind_0p98_chi_0p99_wfrac_0p005_torque_0p005_mdot_5_N896.npz",
    ),
    (
        "ewind_0p997",
        "outputs/checkpoints/high_mdot_stream_m5_fs080_smooth_energy_wind_chi099_w005_eta0_windjac_0997_0999_N896/"
        "m5smooth_ewind_eta0_chi099_w005_windjac0997_0999_mass_0p8_wind_0_heat_0_ewind_0p997_chi_0p99_wfrac_0p005_torque_0p005_mdot_5_N896.npz",
    ),
    (
        "eta_6p20",
        "outputs/checkpoints/high_mdot_stream_m5_fs080_smooth_energy_wind_chi099_w005_eta0_eta_590_620_N896/"
        "m5smooth_ewind_eta0_chi099_w005_eta590_620_mass_0p8_wind_0_heat_0_ewind_0p997970569_chi_0p99_wfrac_0p005_torque_0p005_mdot_5_N896.npz",
    ),
    (
        "eta_6p30",
        "outputs/checkpoints/high_mdot_stream_m5_fs080_smooth_energy_wind_chi099_w005_eta0_eta_630_650_N896/"
        "m5smooth_ewind_eta0_chi099_w005_eta630_650_mass_0p8_wind_0_heat_0_ewind_0p998163695_chi_0p99_wfrac_0p005_torque_0p005_mdot_5_N896.npz",
    ),
    (
        "eta_6p35",
        "outputs/checkpoints/high_mdot_stream_m5_fs080_smooth_energy_wind_chi099_w005_eta0_eta_635_N896/"
        "m5smooth_ewind_eta0_chi099_w005_eta635_mass_0p8_wind_0_heat_0_ewind_0p998253253_chi_0p99_wfrac_0p005_torque_0p005_mdot_5_N896.npz",
    ),
)

JSON_OUTPUT = ROOT / "outputs/tables/m5_energy_wind_anchor_freeze.json"
MD_OUTPUT = ROOT / "outputs/tables/m5_energy_wind_anchor_freeze.md"


def _json_slim(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key not in {"z", "custom_grid_xi"}}


def _dummy_polish(z: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(
        z=np.asarray(z, dtype=float),
        pivot="auto",
        method="diagnostic_freeze",
        result=SimpleNamespace(nfev=0, message="diagnostic freeze from checkpoint"),
        iterations=0,
        newton_audit=(),
    )


def _eta_from_epsilon(epsilon: float) -> float:
    if not 0.0 <= epsilon < 1.0:
        return math.inf
    return float(-math.log1p(-epsilon))


def _format(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        number = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(number):
        return "nan"
    if number == 0.0:
        return "0"
    if abs(number) < 1.0e-3 or abs(number) >= 1.0e4:
        return f"{number:.3e}"
    return f"{number:.6g}"


def freeze_anchor(label: str, rel_path: str, fiducial: FiducialParams, mdot_edd: float) -> dict[str, Any]:
    path = ROOT / rel_path
    if not path.exists():
        raise FileNotFoundError(path)
    z, params = scan.load_anchor(path, fiducial, mdot_edd)
    polish = _dummy_polish(z)
    row = scan.row_for_result(
        branch=label,
        mass_fraction=float(params.stream_source_fraction),
        seed=z,
        z=z,
        params=params,
        polish=polish,
        elapsed_s=0.0,
        extra={
            "freeze_label": label,
            "freeze_checkpoint": rel_path,
            "energy_wind_eta": _eta_from_epsilon(float(params.wind_energy_limited_epsilon)),
        },
    )
    scan.apply_physical_gate(row)
    return row


def write_markdown(rows: list[dict[str, Any]]) -> None:
    columns = [
        "freeze_label",
        "wind_energy_limited_epsilon",
        "energy_wind_eta",
        "N",
        "final_full",
        "partition_physical_E",
        "partition_buffer_E",
        "peak_interval_E_rg",
        "integrated_Qwind_Qvisc",
        "integrated_Qwind_Qrad",
        "f_adv_global",
        "f_adv_inner",
        "f_adv_pos",
        "Lrad_LEdd",
        "max_H_R",
        "Rson_rg",
        "Mdot_outer_over_inner",
        "stream_source_integral_over_inner",
        "wind_sink_integral_over_inner",
        "outer_omega",
        "dominant",
    ]
    lines = [
        "# Mdot=5 Energy-Wind Anchor Freeze",
        "",
        "Generated by `scripts/freeze_mdot5_energy_wind_anchors.py`.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format(row.get(column, "")) for column in columns) + " |")
    lines.append("")
    lines.append("## Checkpoints")
    lines.append("")
    for row in rows:
        lines.append(f"- `{row['freeze_label']}`: `{row['freeze_checkpoint']}`")
    MD_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    MD_OUTPUT.write_text("\n".join(lines) + "\n")


def main() -> None:
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)
    rows = [freeze_anchor(label, rel_path, fiducial, mdot_edd) for label, rel_path in ANCHORS]
    JSON_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUTPUT.write_text(json.dumps(scan.json_safe([_json_slim(row) for row in rows]), indent=2, sort_keys=True) + "\n")
    write_markdown(rows)
    print(f"wrote {scan.relative_root_path(JSON_OUTPUT)}")
    print(f"wrote {scan.relative_root_path(MD_OUTPUT)}")


if __name__ == "__main__":
    main()
