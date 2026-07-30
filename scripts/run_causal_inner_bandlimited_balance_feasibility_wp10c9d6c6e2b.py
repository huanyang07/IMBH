#!/usr/bin/env python3
"""Evaluate the frozen c6e2a band-limited balance search."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import scipy


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for search_path in (SRC, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import run_causal_inner_bandlimited_balance_manifest_wp10c9d6c6e2a as c6e2a
import run_causal_inner_continuum_lift_wp10c9d6c3 as c3
import run_causal_inner_integral_conditioning_validation_wp10c9d6c6e1 as c6e1
import run_causal_inner_windowed_contract_wp10c9d6c6a2 as c6a2

from imri_qpe.layer3_minidisk_1d.causal_inner_continuum_truncation import (  # noqa: E402
    build_causal_five_field_continuum_background,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_packet_manifest import (  # noqa: E402
    causal_array_sha256,
    causal_characteristic_purity,
    causal_canonical_json_sha256,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_packet_resolution import (  # noqa: E402
    causal_packet_spectrum,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_windowed_contract import (  # noqa: E402
    causal_sine_power_window,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c6e2b"
ANALYZED_BASE_COMMIT = "9f4f4b3a720a404619663206878ffc475228eb3f"
ANALYZED_BASE_PARENT = "5c644d2d5912ceca6c661bec6f5db55c798095a9"
ANALYZED_BASE_TREE = "c7a3aaa0101f19de4f3dcbe764d590c18ff7dfa2"
FROZEN_SEARCH_MANIFEST_SHA256 = (
    "7dba2fd9db6cc093eff8a3307dfe851036fee3fbb243a5100b1f0819d4b44c02"
)
THIS_RUNNER = (
    "scripts/"
    "run_causal_inner_bandlimited_balance_feasibility_wp10c9d6c6e2b.py"
)

LABELS = ("uniform_N128", "uniform_N256", "uniform_N512")
REFERENCE_LABEL = LABELS[0]
PRIMARY_PROJECTION_ORDER = 24
SECONDARY_PROJECTION_ORDER = 12
PRIMARY_CONTINUUM_NODES = 769
SECONDARY_CONTINUUM_NODES = 513
ANGULAR_FIELD = 2

PARENT_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_bandlimited_balance_manifest_wp10c9d6c6e2a"
)
PARENT_CONFIG = PARENT_DIRECTORY / "config.json"
PARENT_SUMMARY = PARENT_DIRECTORY / "summary.json"
PARENT_MANIFEST = PARENT_DIRECTORY / "search_manifest.json"
PARENT_PROVENANCE = PARENT_DIRECTORY / "provenance.json"
CANONICAL_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_bandlimited_balance_feasibility_wp10c9d6c6e2b"
)
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
SELECTED_MANIFEST = CANONICAL_DIRECTORY / "selected_profile_manifest.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

IMPLEMENTATION_SOURCES = (
    THIS_RUNNER,
    "scripts/"
    "run_causal_inner_bandlimited_balance_manifest_wp10c9d6c6e2a.py",
    "scripts/"
    "run_causal_inner_integral_conditioning_validation_wp10c9d6c6e1.py",
    "tests/"
    "test_causal_inner_bandlimited_balance_feasibility_wp10c9d6c6e2b.py",
)


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _git_value(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _identity() -> dict:
    resolved = _git_value("rev-parse", ANALYZED_BASE_COMMIT)
    parent = _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
    tree = _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
    if (
        resolved != ANALYZED_BASE_COMMIT
        or parent != ANALYZED_BASE_PARENT
        or tree != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("WP10c9d6c6e2b analyzed git identity changed")
    return {
        "analyzed_base_commit": resolved,
        "analyzed_base_parent": parent,
        "analyzed_base_tree": tree,
    }


def _source_manifest() -> tuple[dict[str, str], str]:
    hashes = {
        path: _sha256(ROOT / path) for path in IMPLEMENTATION_SOURCES
    }
    digest = hashlib.sha256()
    for path, value in sorted(hashes.items()):
        digest.update(path.encode("utf-8"))
        digest.update(value.encode("ascii"))
    return hashes, digest.hexdigest()


def _refresh_sha256s() -> None:
    entries = []
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if not path.is_file() or path.name == "SHA256SUMS.txt":
            continue
        entries.append(f"{_sha256(path)}  {path.name}")
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "\n".join(entries) + "\n",
        encoding="utf-8",
    )


def _refresh_catalog() -> None:
    rows = []
    for case in sorted(CANONICAL_DIRECTORY.parent.iterdir()):
        provenance_path = case / "provenance.json"
        if not case.is_dir() or not provenance_path.is_file():
            continue
        provenance = json.loads(
            provenance_path.read_text(encoding="utf-8")
        )
        status = provenance.get(
            "scientific_status",
            provenance.get("numerical_status", "DIAGNOSTIC ONLY"),
        )
        for path in sorted(case.iterdir()):
            if path.is_file():
                rows.append(
                    {
                        "case": case.name,
                        "path": str(path.relative_to(ROOT)),
                        "bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                        "scientific_status": status,
                    }
                )
    with CANONICAL_MANIFEST.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "case",
                "path",
                "bytes",
                "sha256",
                "scientific_status",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    summary = json.loads(CANONICAL_SUMMARY.read_text(encoding="utf-8"))
    summary.update(
        {
            "case_count": len({row["case"] for row in rows}),
            "file_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "all_payload_hashes_recorded": True,
            "latest_source_parent_commit": ANALYZED_BASE_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write_json(CANONICAL_SUMMARY, summary)


def _load_manifest() -> tuple[dict, dict]:
    parent = json.loads(PARENT_SUMMARY.read_text(encoding="utf-8"))
    manifest = json.loads(PARENT_MANIFEST.read_text(encoding="utf-8"))
    if (
        parent["classification"]
        != (
            "bandlimited_conditioning_search_frozen_"
            "feasibility_authorized"
        )
        or parent["authorized_next"]
        != "WP10c9d6c6e2b_bandlimited_balance_feasibility"
        or not parent["passed"]
    ):
        raise RuntimeError("c6e2b parent authorization changed")
    stored = manifest.pop("manifest_sha256")
    computed = causal_canonical_json_sha256(manifest)
    manifest["manifest_sha256"] = stored
    if (
        stored != FROZEN_SEARCH_MANIFEST_SHA256
        or computed != FROZEN_SEARCH_MANIFEST_SHA256
        or parent["manifest_sha256"] != FROZEN_SEARCH_MANIFEST_SHA256
    ):
        raise RuntimeError("c6e2a search manifest changed")
    return parent, manifest


def _envelope_evaluator(
    *,
    power: int,
    harmonic: int | None,
    interpolator,
    family: str,
    lower_radius: float,
    upper_radius: float,
):
    lower_log = float(np.log(lower_radius))
    upper_log = float(np.log(upper_radius))
    family_index = interpolator.family_labels.index(family)

    def evaluate(radii: np.ndarray) -> np.ndarray:
        values = np.asarray(radii, dtype=float)
        logs = np.log(values)
        coordinate = (logs - lower_log) / (upper_log - lower_log)
        window = causal_sine_power_window(
            logs,
            lower_log_radius=lower_log,
            upper_log_radius=upper_log,
            power=power,
        )
        if harmonic is not None:
            window = window * np.cos(
                float(harmonic) * np.pi * coordinate
            )
        bases = interpolator.evaluate(values)
        vector = bases[:, :, family_index]
        return (
            float(c6e2a.SEARCH_CONTRACT["amplitude"])
            * window[:, None]
            * vector
        )

    return evaluate


def _evaluate_search(manifest: dict):
    configurations, construction_arrays, construction = (
        c3._build_continuum_configurations()
    )
    interpolator, characteristic_report, characteristic_arrays = (
        c6a2._build_characteristic_interpolator(
            configurations,
            construction_arrays,
        )
    )
    reference_grid = configurations[REFERENCE_LABEL]["context"].grid
    finest = configurations[LABELS[-1]]
    background_profile = c3.SmoothCellAverageProfile(
        knots=np.asarray(
            construction_arrays["continuum_background_knots"],
            dtype=float,
        ),
        coefficients=np.asarray(
            construction_arrays["continuum_background_coefficients"],
            dtype=float,
        ),
        degree=c3.PRIMARY_BACKGROUND_DEGREE,
        gravitational_radius=float(reference_grid.gravitational_radius),
    )
    print("WP10c9d6c6e2b: build continuum references", flush=True)
    primary_background = build_causal_five_field_continuum_background(
        finest["context"],
        background_profile.evaluate,
        node_count=PRIMARY_CONTINUUM_NODES,
    )
    secondary_background = build_causal_five_field_continuum_background(
        finest["context"],
        background_profile.evaluate,
        node_count=SECONDARY_CONTINUUM_NODES,
    )
    lower = float(reference_grid.edges[0])
    upper = float(reference_grid.edges[-1])
    field_scales = np.asarray(
        construction_arrays["continuum_perturbation_field_scales"],
        dtype=float,
    )
    centers = np.asarray(reference_grid.centers, dtype=float)
    measures = np.asarray(reference_grid.cell_measures, dtype=float)
    bases = interpolator.evaluate(centers)
    spacing = float(np.mean(np.diff(np.log(reference_grid.edges))))
    contract = manifest["search_contract"]
    reports = {}
    arrays = dict(characteristic_arrays)
    candidate_pair_reports = {}
    for candidate in manifest["candidate_library"]:
        candidate_id = candidate["candidate_id"]
        reports[candidate_id] = {}
        for family in contract["families"]:
            base = _envelope_evaluator(
                power=int(candidate["base_power"]),
                harmonic=None,
                interpolator=interpolator,
                family=family,
                lower_radius=lower,
                upper_radius=upper,
            )
            modulation = _envelope_evaluator(
                power=int(candidate["base_power"]),
                harmonic=int(candidate["modulation_harmonic"]),
                interpolator=interpolator,
                family=family,
                lower_radius=lower,
                upper_radius=upper,
            )
            _, base_primary = c6e1._continuum_action(
                primary_background,
                base,
                reference_grid.edges,
            )
            _, modulation_primary = c6e1._continuum_action(
                primary_background,
                modulation,
                reference_grid.edges,
            )
            _, base_secondary = c6e1._continuum_action(
                secondary_background,
                base,
                reference_grid.edges,
            )
            _, modulation_secondary = c6e1._continuum_action(
                secondary_background,
                modulation,
                reference_grid.edges,
            )
            if abs(modulation_primary) <= np.finfo(float).tiny:
                coefficient = math.inf
            else:
                coefficient = -base_primary / modulation_primary
            if abs(modulation_secondary) <= np.finfo(float).tiny:
                secondary_coefficient = math.inf
            else:
                secondary_coefficient = (
                    -base_secondary / modulation_secondary
                )
            finite = bool(
                np.isfinite(coefficient)
                and np.isfinite(secondary_coefficient)
            )

            def evaluator(
                radii,
                base=base,
                modulation=modulation,
                coefficient=coefficient,
            ):
                return base(radii) + coefficient * modulation(radii)

            coefficient_defect = (
                abs(coefficient - secondary_coefficient)
                / max(
                    abs(coefficient),
                    abs(secondary_coefficient),
                    np.finfo(float).tiny,
                )
                if finite
                else math.inf
            )
            if finite:
                _, primary_action = c6e1._continuum_action(
                    primary_background,
                    evaluator,
                    reference_grid.edges,
                )
                _, secondary_action = c6e1._continuum_action(
                    secondary_background,
                    evaluator,
                    reference_grid.edges,
                )
                cancellation_scale = max(
                    abs(base_secondary)
                    + abs(coefficient * modulation_secondary),
                    np.finfo(float).tiny,
                )
                secondary_cancellation = (
                    abs(secondary_action) / cancellation_scale
                )
                primary_by_label = {}
                secondary_by_label = {}
                maximum_projection = 0.0
                maximum_endpoint = 0.0
                for label in LABELS:
                    grid = configurations[label]["context"].grid
                    primary_values = c3._project_callable_to_cells(
                        grid,
                        evaluator,
                        quadrature_order=PRIMARY_PROJECTION_ORDER,
                    )
                    secondary_values = c3._project_callable_to_cells(
                        grid,
                        evaluator,
                        quadrature_order=SECONDARY_PROJECTION_ORDER,
                    )
                    primary_by_label[label] = primary_values
                    secondary_by_label[label] = secondary_values
                    maximum_projection = max(
                        maximum_projection,
                        c6e1._relative_defect(
                            primary_values,
                            secondary_values,
                        ),
                    )
                    norms = np.linalg.norm(
                        primary_values / field_scales[None, :],
                        axis=1,
                    )
                    maximum_endpoint = max(
                        maximum_endpoint,
                        max(float(norms[0]), float(norms[-1]))
                        / max(
                            float(np.max(norms)),
                            np.finfo(float).tiny,
                        ),
                    )
                values = primary_by_label[REFERENCE_LABEL]
                spectrum = causal_packet_spectrum(
                    values / field_scales[None, :],
                    spacing,
                    quantile=contract["spectral_energy_quantile"],
                )
                theta = (
                    spectrum.quantile_angular_wavenumber * spacing
                )
                family_index = interpolator.family_labels.index(family)
                purity = causal_characteristic_purity(
                    values,
                    bases,
                    field_scales,
                    measures,
                    selected_family=family_index,
                )
                global_purity = float(
                    purity.family_energy_fractions[family_index]
                )
                passed = bool(
                    abs(coefficient)
                    <= contract["maximum_absolute_coefficient"]
                    and coefficient_defect
                    <= contract[
                        "maximum_coefficient_relative_769_513_difference"
                    ]
                    and secondary_cancellation
                    <= contract[
                        "maximum_secondary_initial_cancellation_ratio"
                    ]
                    and theta <= contract["maximum_theta_99"]
                    and spectrum.nyquist_alias_fraction
                    <= contract["maximum_nyquist_alias_fraction"]
                    and maximum_endpoint
                    <= contract["maximum_endpoint_cell_fraction"]
                    and global_purity
                    >= contract["minimum_global_family_purity"]
                    and purity.minimum_active_cell_selected_fraction
                    >= contract["minimum_active_cell_family_purity"]
                    and maximum_projection
                    <= contract["maximum_projection_defect"]
                )
                for label in LABELS:
                    prefix = f"{candidate_id}__{family}__{label}"
                    arrays[prefix + "__primary_physical"] = (
                        primary_by_label[label]
                    )
                    arrays[prefix + "__secondary_physical"] = (
                        secondary_by_label[label]
                    )
                arrays[f"{candidate_id}__{family}__spectrum_theta"] = (
                    spectrum.angular_wavenumbers * spacing
                )
                arrays[f"{candidate_id}__{family}__spectrum_energy"] = (
                    spectrum.spectral_energy
                )
                projection_hashes = {
                    label: causal_array_sha256(
                        primary_by_label[label]
                    )
                    for label in LABELS
                }
                report = {
                    "coefficient": coefficient,
                    "secondary_coefficient": secondary_coefficient,
                    "coefficient_relative_769_513_difference": (
                        coefficient_defect
                    ),
                    "primary_initial_action": primary_action,
                    "secondary_initial_action": secondary_action,
                    "secondary_initial_cancellation_ratio": (
                        secondary_cancellation
                    ),
                    "theta_99": theta,
                    "nyquist_alias_fraction": (
                        spectrum.nyquist_alias_fraction
                    ),
                    "maximum_endpoint_cell_fraction": maximum_endpoint,
                    "global_family_purity": global_purity,
                    "minimum_active_cell_family_purity": (
                        purity.minimum_active_cell_selected_fraction
                    ),
                    "maximum_projection_defect": maximum_projection,
                    "projection_hashes": projection_hashes,
                    "passed": passed,
                }
            else:
                report = {
                    "coefficient": None,
                    "secondary_coefficient": None,
                    "passed": False,
                    "failure": "nonfinite_balance_coefficient",
                }
            reports[candidate_id][family] = report
        inward = reports[candidate_id]["inward_shear"]
        outward = reports[candidate_id]["outward_shear"]
        if inward.get("coefficient") is None or outward.get(
            "coefficient"
        ) is None:
            cross_defect = math.inf
        else:
            cross_defect = abs(
                inward["coefficient"] - outward["coefficient"]
            ) / max(
                abs(inward["coefficient"]),
                abs(outward["coefficient"]),
                np.finfo(float).tiny,
            )
        pair_passed = bool(
            inward["passed"]
            and outward["passed"]
            and cross_defect
            <= contract["maximum_inward_outward_coefficient_difference"]
        )
        candidate_pair_reports[candidate_id] = {
            "base_power": int(candidate["base_power"]),
            "modulation_harmonic": int(
                candidate["modulation_harmonic"]
            ),
            "maximum_theta_99": max(
                inward.get("theta_99", math.inf),
                outward.get("theta_99", math.inf),
            ),
            "maximum_alias_fraction": max(
                inward.get("nyquist_alias_fraction", math.inf),
                outward.get("nyquist_alias_fraction", math.inf),
            ),
            "inward_outward_coefficient_difference": cross_defect,
            "passed": pair_passed,
        }
    eligible = [
        candidate_id
        for candidate_id, report in candidate_pair_reports.items()
        if report["passed"]
    ]
    eligible.sort(
        key=lambda candidate_id: (
            candidate_pair_reports[candidate_id]["maximum_theta_99"],
            candidate_pair_reports[candidate_id][
                "maximum_alias_fraction"
            ],
            candidate_pair_reports[candidate_id]["base_power"],
            candidate_pair_reports[candidate_id][
                "modulation_harmonic"
            ],
        )
    )
    selected = eligible[0] if eligible else None
    selected_payload = None
    if selected is not None:
        candidate = next(
            item
            for item in manifest["candidate_library"]
            if item["candidate_id"] == selected
        )
        selected_payload = {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "source_search_manifest_sha256": (
                FROZEN_SEARCH_MANIFEST_SHA256
            ),
            "selected_candidate": candidate,
            "selection_key_values": {
                key: candidate_pair_reports[selected][key]
                for key in (
                    "maximum_theta_99",
                    "maximum_alias_fraction",
                    "base_power",
                    "modulation_harmonic",
                )
            },
            "family_profiles": {
                family: reports[selected][family]
                for family in contract["families"]
            },
            "operator_changed": False,
            "propagation_executed": False,
        }
        selected_payload = {
            **selected_payload,
            "selected_profile_manifest_sha256": (
                causal_canonical_json_sha256(selected_payload)
            ),
        }
    return {
        "continuum_construction_passed": construction["passed"],
        "characteristic_field_passed": characteristic_report["passed"],
        "candidate_reports": reports,
        "candidate_pair_reports": candidate_pair_reports,
        "eligible_candidates": eligible,
        "selected_candidate": selected,
        "selected_profile_manifest": selected_payload,
        "passed": bool(
            construction["passed"]
            and characteristic_report["passed"]
            and selected is not None
        ),
    }, arrays


def run() -> dict:
    started = time.perf_counter()
    identity = _identity()
    parent, manifest = _load_manifest()
    report, arrays = _evaluate_search(manifest)
    if report["passed"]:
        classification = (
            "bandlimited_balance_profile_selected_"
            "propagation_authorized"
        )
        authorized_next = (
            "WP10c9d6c6e2c_selected_profile_propagation"
        )
        selected_manifest = report["selected_profile_manifest"]
    else:
        classification = "no_eligible_bandlimited_balance_profile"
        authorized_next = "none"
        selected_manifest = {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "source_search_manifest_sha256": (
                FROZEN_SEARCH_MANIFEST_SHA256
            ),
            "selected_candidate": None,
            "operator_changed": False,
            "propagation_executed": False,
        }
        selected_manifest = {
            **selected_manifest,
            "selected_profile_manifest_sha256": (
                causal_canonical_json_sha256(selected_manifest)
            ),
        }
    source_hashes, source_manifest = _source_manifest()
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DECISIVE_ARRAYS, **arrays)
    _write_json(SELECTED_MANIFEST, selected_manifest)
    _write_json(
        CONFIG_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "search_manifest_sha256": FROZEN_SEARCH_MANIFEST_SHA256,
            "search_contract": manifest["search_contract"],
            "propagation_executed": False,
        },
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        **identity,
        "classification": classification,
        "authorized_next": authorized_next,
        "passed": report["passed"],
        "operator_changed": False,
        "evaluation_executed": True,
        "propagation_executed": False,
        "search_manifest_sha256": FROZEN_SEARCH_MANIFEST_SHA256,
        "selected_profile_manifest_sha256": selected_manifest[
            "selected_profile_manifest_sha256"
        ],
        "feasibility_report": report,
        "parent_classification": parent["classification"],
        "parent_classification_preserved": True,
        "c6c_c6d_c6e0_c6e1_status_preserved": True,
        "implementation_source_hashes": source_hashes,
        "implementation_source_manifest_sha256": source_manifest,
        "decisive_arrays_sha256": _sha256(DECISIVE_ARRAYS),
        "decisive_array_hashes": {
            name: _array_sha256(values)
            for name, values in sorted(arrays.items())
        },
        "embedded_export_discrimination_authorized": False,
        "nonlinear_physical_trajectory_authorized": False,
        "production_operator_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "runtime_seconds": float(time.perf_counter() - started),
    }
    _write_json(SUMMARY_PATH, summary)
    _write_json(
        PROVENANCE_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": (
                "PROFILE SELECTED" if report["passed"] else "NO FEASIBLE PROFILE"
            ),
            "source_parent_commit": ANALYZED_BASE_COMMIT,
            **identity,
            "implementation_base_tree": _git_value(
                "rev-parse",
                "HEAD^{tree}",
            ),
            "working_tree_status": _git_value("status", "--short"),
            "command": (
                "PYTHONPATH=src:scripts python "
                "scripts/"
                "run_causal_inner_bandlimited_balance_feasibility_"
                "wp10c9d6c6e2b.py"
            ),
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "platform": platform.platform(),
            },
            "implementation_source_hashes": source_hashes,
            "implementation_source_manifest_sha256": source_manifest,
            "parent_canonical_hashes": {
                str(path.relative_to(ROOT)): _sha256(path)
                for path in (
                    PARENT_CONFIG,
                    PARENT_SUMMARY,
                    PARENT_MANIFEST,
                    PARENT_PROVENANCE,
                )
            },
        },
    )
    _refresh_sha256s()
    _refresh_catalog()
    print(
        json.dumps(
            {
                "classification": classification,
                "authorized_next": authorized_next,
                "eligible_candidates": report["eligible_candidates"],
                "selected_candidate": report["selected_candidate"],
                "selected_profile_manifest_sha256": summary[
                    "selected_profile_manifest_sha256"
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return summary


if __name__ == "__main__":
    run()
