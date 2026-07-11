"""Build the compact, checksummed canonical scientific result set."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CANONICAL = RESULTS / "canonical"
MANIFESTS = RESULTS / "manifests"
SOURCE_COMMIT = "0a000767a915880c0710b8f4ec03eb0c64aa168a"
SOURCE_TAG = "pre-cleanup-p0-2026-07-11"
PHASE_SUMMARY_KEYS = (
    "global_flux_phase_dae_segment_n_intervals",
    "global_flux_phase_dae_segment_accepted_exploratory",
    "global_flux_phase_dae_segment_final_p_R_sign_changes",
    "global_flux_phase_dae_segment_final_direct_radial_max",
    "global_flux_phase_dae_segment_final_direct_energy_max",
    "global_flux_phase_dae_segment_final_fprime_max",
    "global_flux_phase_dae_segment_final_kinematic_max",
    "global_flux_phase_dae_segment_final_endpoint_state_mismatch_max",
)

NO_WIND = Path(
    "outputs/checkpoints/slim_benchmark_high_mdot_no_wind_m5_adaptive_outer_mesh_N768_spot/s32_mdot_5_N768.npz"
)
STREAM_NO_WIND = Path(
    "outputs/checkpoints/standard_slim_stream_residual_remesh_fs080_N640_N768_N896_s12/N896_s12_mass_0p8_torque_0p005_mdot_2_N896.npz"
)
PHASE_ENTRY = Path(
    "outputs/checkpoints/m5_eta_phase_dae_simpson_k13_certified_98p125_N164/stage_00_etaE_98p125_N164.npz"
)
PHASE_K12 = Path(
    "outputs/checkpoints/m5_eta_phase_dae_simpson_centered_k12_final_polish_98p125_N164/stage_00_etaE_98p125_N164.npz"
)
PHASE_K14 = Path(
    "outputs/checkpoints/m5_eta_phase_dae_simpson_k14_fromk13_98p125_N164/stage_00_etaE_98p125_N164.npz"
)
PHASE_K12_TABLE = Path(
    "outputs/tables/m5_eta_phase_dae_simpson_centered_k12_final_polish_98p125_N164.json"
)
PHASE_K13_TABLE = Path(
    "outputs/tables/m5_eta_phase_dae_simpson_k13_certified_98p125_N164.json"
)
PHASE_K14_TABLE = Path(
    "outputs/tables/m5_eta_phase_dae_simpson_k14_fromk13_98p125_N164.json"
)
PHASE_EXIT = Path(
    "outputs/checkpoints/m5_eta_phase_dae_exit_refinement_98p125_N164/extend2_f8828125.npz"
)
PHASE_BASE_ANCHOR = Path(
    "outputs/checkpoints/m5_energy_wind_powerlaw_mass_coupled_adaptive_0p015_to_0p03/zeta_0p03_N896.npz"
)
CLASSIFICATION = Path(
    "outputs/tables/m5_eta_phase_critical_classification_98p125_N164.json"
)
GLOBALIZATION = Path(
    "outputs/tables/m5_eta_phase_critical_globalization_98p125_N164.json"
)
ENDPOINT = Path("outputs/tables/m5_eta_endpoint_validity_audit_98p125_N164.json")
ENDPOINT_PROFILES = Path(
    "outputs/tables/m5_eta_endpoint_validity_audit_98p125_N164_profiles.json"
)
ANGULAR = Path("outputs/tables/m5_eta_angular_momentum_ledger_98p125_N164.json")
OUTER = Path("outputs/tables/m5_eta_independent_outer_manifold_98p125_N164.json")
OUTER_PROFILES = Path(
    "outputs/tables/m5_eta_independent_outer_manifold_98p125_N164_profiles.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n")


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _npz_config(source: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config: dict[str, Any] = {}
    summary: dict[str, Any] = {}
    with np.load(ROOT / source, allow_pickle=False) as data:
        for key in data.files:
            value = data[key]
            if value.ndim == 0:
                item = value.item()
                if key == "row_json":
                    try:
                        summary.update(json.loads(str(item)))
                    except json.JSONDecodeError:
                        summary[key] = str(item)
                elif key in {"full", "accepted"}:
                    summary[key] = item
                else:
                    config[key] = item
            elif value.size <= 4:
                config[key] = value.tolist()
    return _jsonable(config), _jsonable(summary)


def _write_key_value_csv(path: Path, values: dict[str, Any]) -> None:
    rows = [
        {"metric": key, "value": json.dumps(_jsonable(value), sort_keys=True)}
        for key, value in sorted(values.items())
    ]
    _write_rows(path, rows)


def _finalize_case(
    directory: Path,
    *,
    sources: list[Path],
    status: str,
    purpose: str,
    establishes: str,
    does_not_establish: str,
    solver_generation_command: str | None = None,
) -> list[dict[str, Any]]:
    payload = sorted(
        path for path in directory.iterdir() if path.is_file() and path.name != "SHA256SUMS.txt"
    )
    provenance = {
        "source_commit": SOURCE_COMMIT,
        "source_tag": SOURCE_TAG,
        "solver_generation_command": (
            solver_generation_command
            if solver_generation_command is not None
            else "See source_paths and the pre-cleanup scientific tag."
        ),
        "canonical_packaging_command": (
            "PYTHONPATH=src python3 scripts/build_canonical_results.py"
        ),
        "source_paths": [path.as_posix() for path in sources],
        "source_sha256": {
            path.as_posix(): _sha256(ROOT / path) for path in sources
        },
        "scientific_status": status,
        "purpose": purpose,
        "establishes": establishes,
        "does_not_establish": does_not_establish,
        "payload_sha256": {path.name: _sha256(path) for path in payload},
    }
    _write_json(directory / "provenance.json", provenance)
    files = sorted(path for path in directory.iterdir() if path.is_file())
    sums = "".join(f"{_sha256(path)}  {path.name}\n" for path in files)
    (directory / "SHA256SUMS.txt").write_text(sums)
    records = []
    for path in sorted(directory.iterdir()):
        if path.is_file():
            records.append(
                {
                    "case": directory.name,
                    "path": path.relative_to(ROOT).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                    "scientific_status": status,
                }
            )
    return records


def _prepare(name: str) -> Path:
    directory = CANONICAL / name
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.iterdir():
        if path.is_file():
            path.unlink()
    return directory


def main() -> None:
    CANONICAL.mkdir(parents=True, exist_ok=True)
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    case = _prepare("no_wind_mdot5")
    shutil.copy2(ROOT / NO_WIND, case / "state.npz")
    config, summary = _npz_config(NO_WIND)
    config.update(
        {
            "alpha": 0.01,
            "mu_stress": 0.0,
            "stress_factor": 1.0,
        }
    )
    _write_json(case / "config.json", config)
    _write_key_value_csv(case / "summary.csv", summary)
    records += _finalize_case(
        case,
        sources=[NO_WIND],
        status="CERTIFIED",
        purpose="N768 standard no-wind slim-disk Mdot/Edd=5 regression anchor.",
        establishes="The standard solver supports a strict high-Mdot no-wind branch.",
        does_not_establish="Stream, heating, or wind physics.",
        solver_generation_command=(
            "PYTHONPATH=src python3 "
            "scripts/run_standard_slim_high_mdot_no_wind_ladder.py"
        ),
    )

    case = _prepare("stream_no_wind_mdot2_fs080")
    shutil.copy2(ROOT / STREAM_NO_WIND, case / "state.npz")
    config, summary = _npz_config(STREAM_NO_WIND)
    _write_json(case / "config.json", config)
    _write_key_value_csv(case / "summary.csv", summary)
    records += _finalize_case(
        case,
        sources=[STREAM_NO_WIND],
        status="SUPPORTED BUT NOT FULLY CERTIFIED",
        purpose="N896 residual-remeshed compact stream-fed no-wind fs=0.80 anchor.",
        establishes="A mesh-supported stream-fed no-wind branch at Mdot_inner/Edd=2.",
        does_not_establish="Naive-remap robustness, stream heating, or mass-loaded wind.",
    )

    case = _prepare("phase_dae_entry_N164")
    shutil.copy2(ROOT / PHASE_ENTRY, case / "state.npz")
    shutil.copy2(ROOT / PHASE_K12, case / "k12_state.npz")
    shutil.copy2(ROOT / PHASE_K14, case / "k14_state.npz")
    shutil.copy2(ROOT / PHASE_EXIT, case / "exit_refinement_endpoint.npz")
    shutil.copy2(ROOT / PHASE_BASE_ANCHOR, case / "base_anchor.npz")
    for label, source in (
        ("k12", PHASE_K12_TABLE),
        ("k13", PHASE_K13_TABLE),
        ("k14", PHASE_K14_TABLE),
    ):
        rows = json.loads((ROOT / source).read_text())
        final = rows[-1]
        _write_json(
            case / f"{label}_summary.json",
            {key: final[key] for key in PHASE_SUMMARY_KEYS if key in final},
        )
    config, summary = _npz_config(PHASE_ENTRY)
    _write_json(case / "config.json", config)
    _write_key_value_csv(case / "summary.csv", summary)
    records += _finalize_case(
        case,
        sources=[
            PHASE_K12,
            PHASE_ENTRY,
            PHASE_K14,
            PHASE_K12_TABLE,
            PHASE_K13_TABLE,
            PHASE_K14_TABLE,
            PHASE_EXIT,
            PHASE_BASE_ANCHOR,
        ],
        status="SUPPORTED BUT NOT FULLY CERTIFIED",
        purpose="N164 global phase-DAE entry and interface regression state.",
        establishes="The phase segment can be inserted with controlled local residuals.",
        does_not_establish="A global far-side connection or physical endpoint.",
    )

    classification = json.loads((ROOT / CLASSIFICATION).read_text())
    endpoint = json.loads((ROOT / ENDPOINT).read_text())
    endpoint_profiles = json.loads((ROOT / ENDPOINT_PROFILES).read_text())
    case = _prepare("phase_endpoint_positive_N164")
    path_rows = endpoint_profiles["physical_path"]
    numeric_keys = [
        key
        for key, value in path_rows[0].items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    np.savez_compressed(
        case / "tail_state_or_downsampled_trajectory.npz",
        **{key: np.asarray([row[key] for row in path_rows], dtype=float) for key in numeric_keys},
    )
    fit_rows = []
    for quantity, values in endpoint["fit_summary"].items():
        fit_rows.append({"quantity": quantity, **values})
    _write_rows(case / "scaling_fits.csv", fit_rows)
    _write_json(
        case / "config.json",
        {
            "target": endpoint["target"],
            "fit_windows": endpoint["fit_windows"],
            "definitions": endpoint["definitions"],
        },
    )
    _write_json(
        case / "summary.json",
        {
            "validity_gates": endpoint["validity_gates"],
            "numerical_gates": endpoint["numerical_gates"],
            "interpretation": endpoint["interpretation"],
            "decision": classification["decision"],
        },
    )
    records += _finalize_case(
        case,
        sources=[ENDPOINT, ENDPOINT_PROFILES, CLASSIFICATION],
        status="SUPPORTED BUT NOT FULLY CERTIFIED",
        purpose="Compact accepted positive phase tail and common-window validity/scaling audit.",
        establishes="A finite-radius mathematical low-u limit and an earlier 1D validity boundary.",
        does_not_establish="A physical steady reservoir or global nonexistence.",
    )

    case = _prepare("phase_endpoint_step_convergence")
    convergence = []
    for row in classification["baseline"]:
        convergence.append({"method": "logu_gauge", **row["summary"]})
    for row in classification["bordered"]:
        convergence.append({"method": "bordered_arclength", **row})
    _write_rows(case / "convergence.csv", convergence)
    _write_json(case / "config.json", classification["target"])
    records += _finalize_case(
        case,
        sources=[CLASSIFICATION],
        status="SUPPORTED BUT NOT FULLY CERTIFIED",
        purpose="Compact step-size and bordered-continuation endpoint comparison.",
        establishes="The positive endpoint estimate is stable across accepted continuation controls.",
        does_not_establish="A stable signed crossing or finite-state fold.",
    )

    case = _prepare("source_shape_comparison")
    _write_rows(case / "comparison.csv", [row["summary"] for row in classification["source_branches"]])
    _write_json(case / "config.json", classification["target"])
    records += _finalize_case(
        case,
        sources=[CLASSIFICATION],
        status="SUPPORTED BUT NOT FULLY CERTIFIED",
        purpose="Compact C2, C4, C-infinity, and wider-source endpoint comparison.",
        establishes="The endpoint radius is weakly sensitive to the tested source smoothing choices.",
        does_not_establish="Independence from every source or wind closure.",
    )

    globalization = json.loads((ROOT / GLOBALIZATION).read_text())
    case = _prepare("global_composite_failure")
    moving = globalization["moving_interface"]
    _write_rows(case / "residual_profile.csv", moving)
    np.savez_compressed(
        case / "interface_and_tail_snapshot.npz",
        interface_R_rg=np.asarray([row["interface_R_rg"] for row in moving]),
        phase_radial=np.asarray([row["phase_radial"] for row in moving]),
        phase_energy=np.asarray([row["phase_energy"] for row in moving]),
        outside_radial=np.asarray([row["outside_radial"] for row in moving]),
        outside_energy=np.asarray([row["outside_energy"] for row in moving]),
        global_FV_mass=np.asarray([row["global_FV_mass"] for row in moving]),
    )
    _write_json(case / "config.json", globalization["target"])
    records += _finalize_case(
        case,
        sources=[GLOBALIZATION],
        status="REJECTED",
        purpose="Compact witness where the local phase block is accurate but the ordinary outer tail fails.",
        establishes="The tested global composite exports large radial/energy defects outside the phase block.",
        does_not_establish="Global nonexistence of every independent outer branch.",
    )

    angular = json.loads((ROOT / ANGULAR).read_text())
    outer = json.loads((ROOT / OUTER).read_text())
    outer_profiles = json.loads((ROOT / OUTER_PROFILES).read_text())
    case = _prepare("p0_validity_ledger_outer_manifold")
    shutil.copy2(ROOT / ENDPOINT, case / "endpoint_validity_summary.json")
    shutil.copy2(ROOT / ANGULAR, case / "angular_ledger_summary.json")
    shutil.copy2(ROOT / OUTER, case / "outer_manifold_summary.json")
    shutil.copy2(ROOT / CLASSIFICATION, case / "phase_critical_classification_summary.json")
    nominal = [
        {
            "label": row["label"],
            "actual_R_rg": row["actual_R_rg"],
            "z": row["z"],
        }
        for row in outer_profiles["atlas_rows"]
        if row["variant"] == "compact_c2" and row["perturbation"] == "nominal"
    ]
    _write_json(
        case / "outer_match_seeds.json",
        {"inner_match_state": outer_profiles["inner_match_state"], "nominal_outer_seeds": nominal},
    )
    figures = [
        Path("outputs/figures/m5_eta_endpoint_validity_audit_98p125_N164.png"),
        Path("outputs/figures/m5_eta_angular_momentum_ledger_98p125_N164.png"),
        Path("outputs/figures/m5_eta_independent_outer_manifold_98p125_N164.png"),
    ]
    for source in figures:
        shutil.copy2(ROOT / source, case / source.name)
    _write_json(
        case / "config.json",
        {"endpoint": endpoint["target"], "angular": angular["target"], "outer": outer["target"]},
    )
    records += _finalize_case(
        case,
        sources=[ENDPOINT, ANGULAR, OUTER, OUTER_PROFILES, CLASSIFICATION, *figures],
        status="DIAGNOSTIC ONLY",
        purpose="P0 validity, angular-ledger, and independent outer-manifold review evidence.",
        establishes="The validity boundary, representation-ledger identity, and best surveyed conservative near-match.",
        does_not_establish="A physical angular closure or a strict global steady connection.",
    )

    _write_rows(MANIFESTS / "canonical_artifacts.csv", records)
    _write_json(
        MANIFESTS / "canonical_summary.json",
        {
            "source_commit": SOURCE_COMMIT,
            "source_tag": SOURCE_TAG,
            "case_count": len({row["case"] for row in records}),
            "file_count": len(records),
            "total_bytes": sum(int(row["bytes"]) for row in records),
            "all_payload_hashes_recorded": True,
        },
    )
    print((MANIFESTS / "canonical_summary.json").read_text(), end="")


if __name__ == "__main__":
    main()
