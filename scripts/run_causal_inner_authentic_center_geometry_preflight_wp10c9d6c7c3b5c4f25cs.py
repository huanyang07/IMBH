#!/usr/bin/env python3
"""Execute the frozen authentic-center forward-sector geometry preflight."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_authentic_center_local_field_overlap_manifest_wp10c9d6c7c3b5c4f25cr as manifest  # noqa: E402
import run_causal_inner_shell_gated_atlas_geometry_preflight_wp10c9d6c7c3b5c4f25ci as prior_geometry  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25cs"
MANIFEST_COMMIT = "57d3cd648bb003cebf9cc81b3ed6e7dd13de6526"
MANIFEST_PARENT = "88c23290d83130177cb6982a07dc8a678a1e05cc"
MANIFEST_TREE = "c27c7c882e14d2be2eda15f34025c61c145b7895"

FULL_CLASSIFICATION = "authentic_center_forward_geometry_valid_to_0p015"
TRAINING_ONLY_CLASSIFICATION = (
    "authentic_center_forward_geometry_valid_to_0p0125_only"
)
FAIL_CLASSIFICATION = "authentic_center_forward_geometry_failed"
FULL_AUTHORIZED_NEXT = (
    "definitions_only_authentic_center_exact_rate_training_manifest"
)
TRAINING_ONLY_AUTHORIZED_NEXT = (
    "definitions_only_authentic_center_narrow_geometry_revision_manifest"
)
FAIL_AUTHORIZED_NEXT = "definitions_only_authentic_center_chart_revision_manifest"

ARTIFACT = (
    "causal_inner_authentic_center_geometry_preflight_"
    "wp10c9d6c7c3b5c4f25cs"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_authentic_center_geometry_preflight_"
    "wp10c9d6c7c3b5c4f25cs.py"
)
THIS_TEST = (
    "tests/test_causal_inner_authentic_center_geometry_preflight_"
    "wp10c9d6c7c3b5c4f25cs.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_AUTHENTIC_CENTER_GEOMETRY_"
    "PREFLIGHT_WP10C9D6C7C3B5C4F25CS_2026-08-19.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
DESIGN_ARRAYS = manifest.CANONICAL_DIRECTORY / "center_local_field_design.npz"

_plain = manifest._plain
_read = manifest._read
_write_json = manifest._write_json
_sha = manifest._sha
_checksums = manifest._checksums
_load_npz = manifest._load_npz


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _validate_manifest(*, require_clean: bool) -> dict:
    if _git("rev-parse", MANIFEST_COMMIT) != MANIFEST_COMMIT:
        raise RuntimeError("authentic-center local-field manifest commit changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^") != MANIFEST_PARENT:
        raise RuntimeError("authentic-center local-field lineage changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^{{tree}}") != MANIFEST_TREE:
        raise RuntimeError("authentic-center local-field tree changed")
    hashes = _checksums(manifest.CANONICAL_DIRECTORY)
    summary = _read(manifest.CANONICAL_DIRECTORY / "summary.json")
    contract = _read(manifest.CANONICAL_DIRECTORY / "contract.json")
    metrics = _read(manifest.CANONICAL_DIRECTORY / "design_metrics.json")
    provenance = _read(manifest.CANONICAL_DIRECTORY / "provenance.json")
    lock = _read(manifest.CANONICAL_DIRECTORY / "parent_lock.json")
    design = _load_npz(DESIGN_ARRAYS)
    if (
        not summary["passed"]
        or not summary["definitions_only"]
        or summary["classification"] != manifest.CLASSIFICATION
        or summary["authorized_next"] != WORK_PACKAGE
        or summary["prospective_geometry_candidate_count"] != 8
        or summary["new_truth_rate_calls"] != 0
        or summary["new_generator_assemblies"] != 0
        or summary["new_nonlinear_roots"] != 0
        or not metrics["passed"]
        or not all(metrics["checks"].values())
        or contract["decision"]["pass_authorizes_only"] != WORK_PACKAGE
        or design["authentic_center_primitive_state"].shape != (112, 5)
        or design["authentic_center_scaled_delta"].shape != (560,)
        or design["authentic_center_absolute_coordinate"].shape != (470,)
        or design["training_directions"].shape != (4, 28)
        or design["holdout_directions"].shape != (4, 28)
    ):
        raise RuntimeError("authentic-center geometry authorization changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"authentic-center manifest source changed: {relative}")
    if _sha(DESIGN_ARRAYS) != _checksums(manifest.CANONICAL_DIRECTORY)[
        "center_local_field_design.npz"
    ]:
        raise RuntimeError("authentic-center geometry design changed")
    for name, expected in manifest.parent.THREAD_ENVIRONMENT.items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"thread environment changed: {name}")
    if require_clean and _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("authentic-center geometry preflight requires a clean tracked tree")
    return {
        "summary": summary,
        "contract": contract,
        "metrics": metrics,
        "hashes": hashes,
        "lock": lock,
        "design": design,
    }


def _candidate_specs(design: dict[str, np.ndarray]) -> list[dict]:
    specs = []
    for role, directions_name, bounds_name in (
        ("training", "training_directions", "training_component_bounds"),
        ("holdout", "holdout_directions", "holdout_component_bounds"),
    ):
        directions = np.asarray(design[directions_name], dtype=float)
        bounds = np.asarray(design[bounds_name], dtype=float)
        for index, (direction, bound) in enumerate(zip(directions, bounds)):
            specs.append(
                {
                    "role": role,
                    "role_index": index,
                    "direction": direction,
                    "component_bound": float(bound),
                }
            )
    return specs


def _prepare_center_components(design: dict[str, np.ndarray]) -> dict:
    chart_tools = prior_geometry.geometry_tools.prior_geometry.chart_tools
    geometry_module = prior_geometry.geometry_tools.prior_geometry
    components = geometry_module._prepare_components()
    center = np.asarray(design["authentic_center_primitive_state"], dtype=float)
    components["state"] = np.array(center, copy=True)
    components["coordinate_target"], components["center_coordinate_factors"] = (
        chart_tools._coordinate_value_with_factors(center, components)
    )
    face = 36 * int(components["data"]["layout"].refinement_ratio)
    components["base_q3"], components["base_q3_factors"] = (
        geometry_module.causal_five_field_exterior_q3(
            components["context"], center, exterior_face_index=face
        )
    )
    return components


def _local_contract(component_bound: float) -> dict:
    return prior_geometry._local_contract(
        prior_geometry.manifest._contract(), float(component_bound)
    )


def _local_coordinate(
    state: np.ndarray,
    local_delta: np.ndarray,
    components: dict,
    memory_basis: np.ndarray,
    departure_basis: np.ndarray,
) -> np.ndarray:
    chart_tools = prior_geometry.geometry_tools.prior_geometry.chart_tools
    physical, _factors = chart_tools._coordinate_value_with_factors(
        state, components
    )
    return np.concatenate(
        (
            physical - components["coordinate_target"],
            memory_basis.T @ local_delta,
            departure_basis.T @ local_delta,
        )
    )


def _aggregate(candidates: list[dict], failures: list[dict]) -> dict:
    aggregate = prior_geometry._aggregate(candidates, failures)
    aggregate.update(
        {
            "maximum_new_scaled_state_load": float(
                max(
                    (item["new_scaled_state_load"] for item in candidates),
                    default=math.inf,
                )
            ),
            "maximum_absolute_old_scaled_state_load": float(
                max(
                    (
                        item["absolute_old_scaled_state_load"]
                        for item in candidates
                    ),
                    default=math.inf,
                )
            ),
            "maximum_local_coordinate_translation_defect": float(
                max(
                    (
                        item["local_coordinate_translation_defect"]
                        for item in candidates
                    ),
                    default=math.inf,
                )
            ),
        }
    )
    return aggregate


def _role_checks(aggregate: dict, component_bound: float) -> dict:
    gates = prior_geometry.manifest._contract()["binding_per_rung_gates"]
    checks = prior_geometry._gate_checks(aggregate, gates, component_bound)
    checks.update(
        {
            "new_scaled_load": aggregate["maximum_new_scaled_state_load"]
            <= component_bound + 1.0e-12,
            "translation": aggregate[
                "maximum_local_coordinate_translation_defect"
            ] <= 1.0e-14,
        }
    )
    return checks


def _execute(frozen: dict) -> tuple[dict, dict[str, np.ndarray]]:
    design = frozen["design"]
    components = _prepare_center_components(design)
    center_coordinate = np.asarray(
        design["authentic_center_absolute_coordinate"], dtype=float
    )
    center_delta = np.asarray(
        design["authentic_center_scaled_delta"], dtype=float
    )
    restriction = np.asarray(
        design["authentic_center_fixed_restriction"], dtype=float
    )
    memory_basis = restriction[
        manifest.PHYSICAL_DIMENSION : (
            manifest.PHYSICAL_DIMENSION + manifest.MEMORY_DIMENSION
        )
    ].T
    departure_basis = restriction[-manifest.DEPARTURE_DIMENSION :].T
    chart_tools = prior_geometry.geometry_tools.prior_geometry.chart_tools
    candidates: list[dict] = []
    failures: list[dict] = []
    role_records: list[dict] = []
    arrays = {
        "candidate_primitive_states": [],
        "candidate_local_scaled_deltas": [],
        "candidate_absolute_scaled_deltas": [],
        "candidate_local_coordinates": [],
        "candidate_absolute_coordinates": [],
        "candidate_directions": [],
        "candidate_component_bounds": [],
        "candidate_role_codes": [],
        "candidate_role_indices": [],
    }
    began = time.perf_counter()
    specs = _candidate_specs(design)
    for role_index, role in enumerate(("training", "holdout")):
        role_candidates: list[dict] = []
        role_failures: list[dict] = []
        role_specs = [spec for spec in specs if spec["role"] == role]
        for spec in role_specs:
            candidate_index = len(candidates) + len(failures)
            try:
                candidate, result_arrays = chart_tools._retract_candidate(
                    components,
                    departure_basis,
                    memory_basis,
                    spec["direction"],
                    1,
                    spec["component_bound"],
                    _local_contract(spec["component_bound"]),
                )
                local_delta = np.asarray(result_arrays["scaled_delta"], dtype=float)
                state = np.asarray(result_arrays["primitive_state"], dtype=float)
                local_coordinate = _local_coordinate(
                    state,
                    local_delta,
                    components,
                    memory_basis,
                    departure_basis,
                )
                absolute_coordinate = center_coordinate + local_coordinate
                restored_local = absolute_coordinate - center_coordinate
                absolute_delta = center_delta + local_delta
                candidate.update(
                    {
                        "candidate_index": candidate_index,
                        "role": role,
                        "role_index": spec["role_index"],
                        "new_scaled_state_load": float(
                            np.max(np.abs(local_delta))
                        ),
                        "absolute_old_scaled_state_load": float(
                            np.max(np.abs(absolute_delta))
                        ),
                        "local_coordinate_infinity_load": float(
                            np.max(np.abs(local_coordinate))
                        ),
                        "local_coordinate_translation_defect": float(
                            np.max(np.abs(restored_local - local_coordinate))
                        ),
                    }
                )
                role_candidates.append(candidate)
                candidates.append(candidate)
                arrays["candidate_primitive_states"].append(state)
                arrays["candidate_local_scaled_deltas"].append(local_delta)
                arrays["candidate_absolute_scaled_deltas"].append(absolute_delta)
                arrays["candidate_local_coordinates"].append(local_coordinate)
                arrays["candidate_absolute_coordinates"].append(
                    absolute_coordinate
                )
                arrays["candidate_directions"].append(spec["direction"])
                arrays["candidate_component_bounds"].append(
                    spec["component_bound"]
                )
                arrays["candidate_role_codes"].append(role_index)
                arrays["candidate_role_indices"].append(spec["role_index"])
                status = "accepted"
            except chart_tools.ChartRetractionFailure as error:
                failure = {
                    "candidate_index": candidate_index,
                    "role": role,
                    "role_index": spec["role_index"],
                    "reason": str(error),
                    "diagnostics": error.diagnostics,
                }
                role_failures.append(failure)
                failures.append(failure)
                status = "failed"
            except (ValueError, FloatingPointError) as error:
                failure = {
                    "candidate_index": candidate_index,
                    "role": role,
                    "role_index": spec["role_index"],
                    "reason": f"{type(error).__name__}: {error}",
                    "diagnostics": {},
                }
                role_failures.append(failure)
                failures.append(failure)
                status = "failed"
            print(
                json.dumps(
                    {
                        "candidate": candidate_index,
                        "role": role,
                        "role_index": spec["role_index"],
                        "component_bound": spec["component_bound"],
                        "status": status,
                        "elapsed_seconds": time.perf_counter() - began,
                    }
                ),
                flush=True,
            )
            if role_failures:
                break
        aggregate = _aggregate(role_candidates, role_failures)
        checks = _role_checks(
            aggregate,
            float(role_specs[0]["component_bound"]),
        )
        role_records.append(
            {
                "role": role,
                "component_bound": float(role_specs[0]["component_bound"]),
                "passed": all(checks.values()),
                "checks": checks,
                **aggregate,
            }
        )
        if not role_records[-1]["passed"]:
            break
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "role_records": role_records,
        "attempted_role_count": len(role_records),
        "passing_role_count": sum(record["passed"] for record in role_records),
        "completed_candidate_count": len(candidates),
        "failed_candidate_count": len(failures),
        "failures": failures,
        "new_continuous_rate_calls": 0,
        "new_complete_generator_assemblies": 0,
        "new_nonlinear_fixed_Q_roots": 0,
        "propagated_states": 0,
        "wall_seconds": time.perf_counter() - began,
    }
    array_payload = {
        name: np.asarray(values, dtype=int if "codes" in name or "indices" in name else float)
        for name, values in arrays.items()
    }
    array_payload.update(
        {
            "authentic_center_primitive_state": design[
                "authentic_center_primitive_state"
            ],
            "authentic_center_scaled_delta": center_delta,
            "authentic_center_absolute_coordinate": center_coordinate,
            "authentic_center_fixed_restriction": restriction,
            "training_directions": design["training_directions"],
            "holdout_directions": design["holdout_directions"],
            "active_departure_basis": design["active_departure_basis"],
        }
    )
    return metrics, array_payload


def _classify(metrics: dict) -> dict:
    budget_passed = (
        metrics["new_continuous_rate_calls"] == 0
        and metrics["new_complete_generator_assemblies"] == 0
        and metrics["new_nonlinear_fixed_Q_roots"] == 0
        and metrics["propagated_states"] == 0
    )
    if metrics["passing_role_count"] == 2 and budget_passed:
        return {
            "passed": True,
            "classification": FULL_CLASSIFICATION,
            "authorized_next": FULL_AUTHORIZED_NEXT,
            "largest_passing_component_bound": manifest.HOLDOUT_COMPONENT_BOUND,
        }
    if metrics["passing_role_count"] == 1 and budget_passed:
        return {
            "passed": True,
            "classification": TRAINING_ONLY_CLASSIFICATION,
            "authorized_next": TRAINING_ONLY_AUTHORIZED_NEXT,
            "largest_passing_component_bound": manifest.TRAINING_COMPONENT_BOUND,
        }
    return {
        "passed": False,
        "classification": FAIL_CLASSIFICATION,
        "authorized_next": FAIL_AUTHORIZED_NEXT,
        "largest_passing_component_bound": 0.0,
    }


def _update_catalog(summary: dict) -> None:
    rows = []
    if CANONICAL_MANIFEST.exists():
        with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    status = "CERTIFIED" if summary["passed"] else "REJECTED"
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "case": ARTIFACT,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": _sha(path),
                    "scientific_status": status,
                }
            )
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["case", "path", "bytes", "sha256", "scientific_status"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    catalog = _read(CANONICAL_SUMMARY)
    catalog.setdefault("artifacts", {})[ARTIFACT] = {
        "path": str(CANONICAL_DIRECTORY.relative_to(ROOT)),
        "classification": summary["classification"],
        "passed": summary["passed"],
    }
    catalog.update(
        {
            "case_count": len({row["case"] for row in rows}),
            "file_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "all_payload_hashes_recorded": True,
            "latest_source_parent_commit": MANIFEST_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write_json(CANONICAL_SUMMARY, catalog)


def _run() -> dict:
    frozen = _validate_manifest(require_clean=True)
    if CANONICAL_DIRECTORY.exists():
        raise RuntimeError("authentic-center geometry preflight already exists")
    metrics, arrays = _execute(frozen)
    decision = _classify(metrics)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": decision["classification"],
        "passed": decision["passed"],
        "attempted_role_count": metrics["attempted_role_count"],
        "passing_role_count": metrics["passing_role_count"],
        "completed_candidate_count": metrics["completed_candidate_count"],
        "failed_candidate_count": metrics["failed_candidate_count"],
        "largest_passing_component_bound": decision[
            "largest_passing_component_bound"
        ],
        "new_truth_rate_calls": 0,
        "new_generator_assemblies": 0,
        "new_nonlinear_roots": 0,
        "propagated_states": 0,
        "geometry_candidate_became_atlas_center": False,
        "physical_microburst_authorized": False,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": decision["authorized_next"],
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write_json(CANONICAL_DIRECTORY / "geometry_metrics.json", metrics)
    _write_npz(CANONICAL_DIRECTORY / "geometry_arrays.npz", arrays)
    _write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    _write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "manifest_commit": MANIFEST_COMMIT,
            "manifest_parent": MANIFEST_PARENT,
            "manifest_tree": MANIFEST_TREE,
            "manifest_hashes": frozen["hashes"],
            "design_arrays_sha256": _sha(DESIGN_ARRAYS),
        },
    )
    source_files = (
        THIS_RUNNER,
        THIS_TEST,
        manifest.THIS_RUNNER,
        manifest.THIS_TEST,
        prior_geometry.THIS_RUNNER,
        prior_geometry.THIS_TEST,
    )
    _write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "CERTIFIED" if summary["passed"] else "REJECTED",
            "execution_commit": _git("rev-parse", "HEAD"),
            "execution_tree": _git("rev-parse", "HEAD^{tree}"),
            "source_hashes": {
                relative: _sha(ROOT / relative) for relative in source_files
            },
            "thread_environment": {
                name: os.environ.get(name)
                for name in manifest.parent.THREAD_ENVIRONMENT
            },
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    )
    names = tuple(sorted(path.name for path in CANONICAL_DIRECTORY.iterdir()))
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{_sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names
        ),
        encoding="utf-8",
    )
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Authentic-center geometry preflight WP10c9d6c7c3b5c4f25cs",
                "",
                "## Classification",
                "",
                f"`{summary['classification']}`",
                "",
                f"Accepted `{summary['completed_candidate_count']}` of eight frozen center-local geometry candidates; failures: `{summary['failed_candidate_count']}`.",
                "",
                f"Largest passing center-relative component bound: `{summary['largest_passing_component_bound']}`.",
                "",
                "The preflight used exact geometric retraction at the authentic warm-6 center but no continuous-rate truth, complete generator, fixed-Q time root, or propagated state.",
                "",
                f"Authorized next artifact: `{summary['authorized_next']}`. No physical microburst, predictive cycle, or reduced slow evolution is authorized.",
                "",
            )
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        raise SystemExit("pass --run")
    summary = _run()
    print(json.dumps(_plain(summary), indent=2, sort_keys=True))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
