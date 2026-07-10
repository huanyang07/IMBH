"""Audit conservative total-energy equivalence on the canonical no-wind disk."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from imri_qpe.layer3_minidisk_1d import (
    TransonicSlimParams,
    legacy_energy_identity_audit,
    unpack_state,
)
from imri_qpe.parameters import FiducialParams
from imri_qpe.scales import eddington_mdot


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "results/canonical/no_wind_mdot5/state.npz"
OUTPUT = ROOT / "outputs/tables/unified_conservative_energy_identity.json"


def _optional_pair(data, key: str) -> tuple[float, float] | None:
    if key not in data:
        return None
    values = np.asarray(data[key], dtype=float)
    if values.shape != (2,) or not np.all(np.isfinite(values)):
        return None
    return float(values[0]), float(values[1])


def _custom_grid(data) -> tuple[float, ...] | None:
    if "custom_grid_xi" not in data:
        return None
    values = np.asarray(data["custom_grid_xi"], dtype=float)
    if values.shape != (int(data["n_nodes"]),):
        return None
    return tuple(float(value) for value in values)


def load_case() -> tuple[np.ndarray, TransonicSlimParams]:
    fiducial = FiducialParams()
    with np.load(CHECKPOINT) as data:
        params = TransonicSlimParams(
            M2_g=fiducial.M2_g,
            Mdot_g_s=float(data["ratio"]) * eddington_mdot(fiducial.M2_g),
            alpha=0.01,
            mu_stress=0.0,
            stress_factor=1.0,
            R_out_rg=float(data["R_out_rg"]),
            n_nodes=int(data["n_nodes"]),
            grid_power=float(data["grid_power"]),
            custom_grid_xi=_custom_grid(data),
            outer_closure=str(np.asarray(data["outer_closure"]).item()),
            outer_match_log_slopes=_optional_pair(data, "outer_match_log_slopes"),
            residual_tol=1.0e-8,
            max_nfev=1,
        )
        z = np.asarray(data["z"], dtype=float)
    return z, params


def run() -> dict[str, object]:
    z, params = load_case()
    logu, logT, _logR_son, lambda0, logR = unpack_state(z, params)
    rows: list[dict[str, float]] = []
    for idx in range(params.n_nodes - 1):
        dx = float(logR[idx + 1] - logR[idx])
        y_left = np.asarray([logu[idx], logT[idx]], dtype=float)
        y_right = np.asarray([logu[idx + 1], logT[idx + 1]], dtype=float)
        audit = legacy_energy_identity_audit(
            0.5 * float(logR[idx] + logR[idx + 1]),
            0.5 * (y_left + y_right),
            (y_right - y_left) / dx,
            lambda0,
            params,
        )
        rows.append(
            {
                "interval": idx,
                "R_mid_rg": float(np.exp(0.5 * (logR[idx] + logR[idx + 1])) / params.r_g),
                "raw_identity_defect": audit.raw_identity_defect,
                "vertical_work_derivative": audit.vertical_work_derivative,
                "corrected_identity_defect": audit.corrected_identity_defect,
                "normalized_raw_defect": audit.normalized_raw_defect,
                "normalized_corrected_defect": audit.normalized_corrected_defect,
            }
        )
    raw = np.abs(np.asarray([row["normalized_raw_defect"] for row in rows], dtype=float))
    corrected = np.abs(np.asarray([row["normalized_corrected_defect"] for row in rows], dtype=float))
    peak = int(np.argmax(raw))
    result: dict[str, object] = {
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "ratio": float(params.mdot_edd_ratio),
        "N": int(params.n_nodes),
        "raw_max": float(np.max(raw)),
        "raw_p90": float(np.percentile(raw, 90.0)),
        "corrected_max": float(np.max(corrected)),
        "corrected_p90": float(np.percentile(corrected, 90.0)),
        "peak_raw_R_rg": float(rows[peak]["R_mid_rg"]),
        "gate": bool(np.max(corrected) <= 1.0e-10),
        "rows": rows,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    result = run()
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
