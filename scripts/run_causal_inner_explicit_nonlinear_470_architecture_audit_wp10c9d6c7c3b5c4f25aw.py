#!/usr/bin/env python3
"""Audit the explicit nonlinear 470-state conservative IMEX architecture."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
from scipy.linalg import null_space


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_explicit_nonlinear_470_architecture_manifest_wp10c9d6c7c3b5c4f25av as manifest  # noqa: E402
import run_causal_inner_first_conditional_branch_seed_preflight_wp10c9d6c7c3b5c4f25aq as preflight  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25aw"
MANIFEST_COMMIT = "23adb653984e97d1f09dae46ddf71ebc29be9ca9"
MANIFEST_PARENT = "8dba5a5c632f53873eef782c507798fd07232bc8"
MANIFEST_TREE = "1296b6debd80eea5a14282835eda601f8d2f87d8"

PASS_CLASSIFICATION = (
    "explicit_nonlinear_470_architecture_structurally_certified_"
    "exact_geometric_departure_chart_manifest_authorized"
)
FAIL_CLASSIFICATION = (
    "explicit_nonlinear_470_architecture_structural_audit_failed_"
    "reduced_evolution_blocked"
)

ARTIFACT = (
    "causal_inner_explicit_nonlinear_470_architecture_audit_"
    "wp10c9d6c7c3b5c4f25aw"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_explicit_nonlinear_470_architecture_audit_"
    "wp10c9d6c7c3b5c4f25aw.py"
)
THIS_TEST = (
    "tests/test_causal_inner_explicit_nonlinear_470_architecture_audit_"
    "wp10c9d6c7c3b5c4f25aw.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_EXPLICIT_NONLINEAR_470_"
    "ARCHITECTURE_AUDIT_WP10C9D6C7C3B5C4F25AW_2026-08-19.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

HIGH_ORDER_PATH = ROOT / (
    "results/canonical/causal_inner_high_order_hermite_audit_"
    "wp10c9d6c7c3b5c4f25ag/decisive_model.npz"
)
FIBER_DIRECTORY = ROOT / (
    "results/canonical/causal_inner_unstable_exact_conservative_fiber_audit_"
    "wp10c9d6c7c3b5c4f25u"
)
FIBER_PATH = FIBER_DIRECTORY / "decisive_fibers.npz"
HYBRID_AUDIT_DIRECTORY = ROOT / (
    "results/canonical/causal_inner_equilibrium_centered_hybrid_"
    "architecture_audit_wp10c9d6c7c3b5c4f25ao"
)


def _plain(value):
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, int):
        return value
    return value


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _tracked_tree_clean() -> bool:
    return not _git("status", "--short", "--untracked-files=no")


def _checksums(directory: Path) -> dict[str, str]:
    recorded = {}
    for line in (directory / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        expected, name = line.split("  ", 1)
        if _sha(directory / name) != expected:
            raise RuntimeError(f"checksum mismatch: {directory / name}")
        recorded[name] = expected
    return recorded


def _validate_manifest(*, require_clean: bool) -> dict:
    if _git("rev-parse", MANIFEST_COMMIT) != MANIFEST_COMMIT:
        raise RuntimeError("470-architecture manifest commit changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^") != MANIFEST_PARENT:
        raise RuntimeError("470-architecture manifest lineage changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^{{tree}}") != MANIFEST_TREE:
        raise RuntimeError("470-architecture manifest tree changed")
    hashes = _checksums(manifest.ARTIFACT_DIRECTORY)
    summary = _read(manifest.ARTIFACT_DIRECTORY / "summary.json")
    contract = _read(manifest.ARTIFACT_DIRECTORY / "contract.json")
    provenance = _read(manifest.ARTIFACT_DIRECTORY / "provenance.json")
    if (
        not summary["passed"]
        or summary["authorized_next"] != WORK_PACKAGE
        or summary["online_dimension"] != manifest.ONLINE_DIMENSION
        or summary["equilibrium_branch_required"]
        or contract["claim_boundary"]["nonlinear_28D_closure_identified"]
    ):
        raise RuntimeError("470-architecture authorization changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"manifest source changed: {relative}")
    _checksums(manifest.parent.CANONICAL_DIRECTORY)
    _checksums(manifest.stable.CANONICAL_DIRECTORY)
    _checksums(preflight.CANONICAL_DIRECTORY)
    _checksums(FIBER_DIRECTORY)
    if require_clean and not _tracked_tree_clean():
        raise RuntimeError("470-architecture audit requires a clean tracked tree")
    for name, expected in preflight.THREAD_ENVIRONMENT.items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"thread environment changed: {name}")
    return {"summary": summary, "contract": contract, "hashes": hashes}


def _relative(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.linalg.norm(np.asarray(left) - np.asarray(right))
        / max(
            float(np.linalg.norm(left)),
            float(np.linalg.norm(right)),
            np.finfo(float).tiny,
        )
    )


def _online_coordinates():
    with np.load(
        preflight.CANONICAL_DIRECTORY / "preflight_diagnostics.npz",
        allow_pickle=False,
    ) as source:
        base = {name: np.asarray(source[name]) for name in source.files}
    with np.load(HIGH_ORDER_PATH, allow_pickle=False) as source:
        memory_trial = np.asarray(source["primary_hidden_truth_trial"], dtype=float)
    with np.load(FIBER_PATH, allow_pickle=False) as source:
        departure_trial = np.asarray(source["primary_right_basis"], dtype=float)
    physical = base["coordinate_jacobian"]
    gram = physical @ physical.T
    projected_memory = memory_trial - physical.T @ np.linalg.solve(
        gram, physical @ memory_trial
    )
    memory_coordinates, memory_triangular = np.linalg.qr(
        projected_memory, mode="reduced"
    )
    physical_memory = np.vstack((physical, memory_coordinates.T))
    projected_departure = departure_trial - physical_memory.T @ np.linalg.solve(
        physical_memory @ physical_memory.T,
        physical_memory @ departure_trial,
    )
    departure_coordinates, departure_triangular = np.linalg.qr(
        projected_departure, mode="reduced"
    )
    restriction = np.vstack(
        (physical, memory_coordinates.T, departure_coordinates.T)
    )
    lifting = restriction.T @ np.linalg.solve(
        restriction @ restriction.T, np.eye(manifest.ONLINE_DIMENSION)
    )
    hidden = null_space(restriction)
    return {
        "base": base,
        "physical": physical,
        "memory_trial": memory_trial,
        "projected_memory": projected_memory,
        "memory_coordinates": memory_coordinates,
        "memory_triangular": memory_triangular,
        "departure_trial": departure_trial,
        "projected_departure": projected_departure,
        "departure_coordinates": departure_coordinates,
        "departure_triangular": departure_triangular,
        "restriction": restriction,
        "lifting": lifting,
        "hidden": hidden,
    }


def _coordinate_audit() -> tuple[dict, dict[str, np.ndarray]]:
    arrays = _online_coordinates()
    restriction = arrays["restriction"]
    hidden = arrays["hidden"]
    rate = arrays["base"]["fixed_Q_rate"]
    physical = arrays["physical"]
    physical_hidden = null_space(physical)
    physical_memory = restriction[:442]
    memory_hidden = null_space(physical_memory)
    q3 = arrays["base"]["Q3_constraint_rows"]
    q3_projection = q3 @ arrays["lifting"] @ restriction
    projected_departure_q = np.linalg.qr(
        arrays["projected_departure"], mode="reduced"
    )[0]
    departure_cosines = np.linalg.svd(
        arrays["departure_coordinates"].T @ projected_departure_q,
        compute_uv=False,
    )
    metrics = {
        "online_coordinate_rank": int(np.linalg.matrix_rank(restriction)),
        "online_coordinate_condition_number": float(np.linalg.cond(restriction)),
        "stable_memory_projected_rank": int(
            np.linalg.matrix_rank(arrays["projected_memory"])
        ),
        "stable_memory_projected_condition_number": float(
            np.linalg.cond(arrays["projected_memory"])
        ),
        "departure_projected_rank": int(
            np.linalg.matrix_rank(arrays["projected_departure"])
        ),
        "departure_minimum_principal_cosine": float(np.min(departure_cosines)),
        "hidden_remainder_dimension": int(hidden.shape[1]),
        "restriction_lifting_identity_defect": float(
            np.max(
                np.abs(
                    restriction @ arrays["lifting"]
                    - np.eye(manifest.ONLINE_DIMENSION)
                )
            )
        ),
        "restriction_hidden_annihilation_defect": float(
            np.max(np.abs(restriction @ hidden))
        ),
        "Q3_rowspace_relative_defect": _relative(q3_projection, q3),
        "full_rate_norm_per_second": float(np.linalg.norm(rate)),
        "physical_162_hidden_rate_norm_per_second": float(
            np.linalg.norm(physical_hidden.T @ rate)
        ),
        "physical_memory_442_hidden_rate_norm_per_second": float(
            np.linalg.norm(memory_hidden.T @ rate)
        ),
        "online_470_hidden_rate_norm_per_second": float(
            np.linalg.norm(hidden.T @ rate)
        ),
        "online_470_hidden_rate_relative_fraction": float(
            np.linalg.norm(hidden.T @ rate) / np.linalg.norm(rate)
        ),
        "online_470_captured_rate_fraction": float(
            np.linalg.norm(restriction.T @ np.linalg.solve(
                restriction @ restriction.T, restriction @ rate
            ))
            / np.linalg.norm(rate)
        ),
    }
    saved = {
        "online_coordinate_restriction": restriction,
        "online_coordinate_lifting": arrays["lifting"],
        "stable_memory_coordinate_basis": arrays["memory_coordinates"],
        "departure_coordinate_basis": arrays["departure_coordinates"],
        "hidden_stable_remainder_basis": hidden,
        "Q3_rowspace_projection": q3_projection,
        "fixed_Q_rate": rate,
    }
    return metrics, saved


def _timed_median(function, repetitions: int) -> float:
    function()
    samples = []
    for _ in range(repetitions):
        began = time.perf_counter()
        function()
        samples.append(time.perf_counter() - began)
    return float(np.median(samples))


def _cost_and_inheritance_audit() -> dict:
    stable_summary = _read(manifest.stable.CANONICAL_DIRECTORY / "summary.json")
    hybrid_metrics = _read(HYBRID_AUDIT_DIRECTORY / "metrics.json")
    stable_seconds = stable_summary["projected_stable_kernel_cycle_wall_seconds"]
    rng = np.random.default_rng(470)
    departure = rng.normal(size=manifest.NONLINEAR_DEPARTURE_DIMENSION)
    resolved = rng.normal(size=manifest.PHYSICAL_DIMENSION)
    linear = rng.normal(
        size=(manifest.NONLINEAR_DEPARTURE_DIMENSION,) * 2
    )
    left2 = rng.normal(size=(32, manifest.NONLINEAR_DEPARTURE_DIMENSION))
    right2 = rng.normal(size=(32, manifest.NONLINEAR_DEPARTURE_DIMENSION))
    lift2 = rng.normal(size=(manifest.NONLINEAR_DEPARTURE_DIMENSION, 32))
    coupling = rng.normal(
        size=(manifest.NONLINEAR_DEPARTURE_DIMENSION, manifest.PHYSICAL_DIMENSION)
    )

    def nonlinear_kernel():
        features = (left2 @ departure) * (right2 @ departure)
        rate = linear @ departure + lift2 @ features + coupling @ resolved
        return float(rate[0])

    per_step = _timed_median(nonlinear_kernel, 1001)
    additional = per_step * manifest.MAXIMUM_MACROSTEPS
    total = stable_seconds + additional
    return {
        "inherited_stable_kernel_cycle_wall_seconds": stable_seconds,
        "nonlinear_28D_low_rank_kernel_median_wall_seconds_per_step": per_step,
        "projected_additional_nonlinear_cycle_wall_seconds": additional,
        "projected_online_cycle_wall_seconds": total,
        "projected_online_cycle_wall_days": total / 86_400.0,
        "projected_fraction_of_three_day_budget": total
        / manifest.WALL_BUDGET_SECONDS,
        "inherited_maximum_stable_spectral_abscissa_per_second": stable_summary[
            "maximum_stable_spectral_abscissa_per_second"
        ],
        "inherited_stable_energy_amplification_max": hybrid_metrics[
            "legacy_descriptor_method_witness"
        ]["maximum_energy_amplification_factor"],
        "exact_departure_dimension": stable_summary["unstable_bundle_dimension"],
        "online_truth_calls_per_macrostep": 0,
        "nonlinear_coefficients_identified": False,
    }


def _update_catalog(summary: dict) -> None:
    rows = []
    if CANONICAL_MANIFEST.exists():
        with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "case": ARTIFACT,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": _sha(path),
                    "scientific_status": (
                        "CERTIFIED" if summary["passed"] else "REJECTED"
                    ),
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
        raise RuntimeError("470-architecture audit is already canonicalized")
    coordinate, arrays = _coordinate_audit()
    cost = _cost_and_inheritance_audit()
    gates = frozen["contract"]["binding_structural_audit_gates"]
    checks = {
        "online_rank": coordinate["online_coordinate_rank"]
        == gates["online_coordinate_rank_equal"],
        "online_condition": coordinate["online_coordinate_condition_number"]
        <= gates["online_coordinate_condition_number_max"],
        "memory_rank": coordinate["stable_memory_projected_rank"]
        == gates["stable_memory_projected_rank_equal"],
        "memory_condition": coordinate["stable_memory_projected_condition_number"]
        <= gates["stable_memory_projected_condition_number_max"],
        "departure_rank": coordinate["departure_projected_rank"]
        == gates["departure_projected_rank_equal"],
        "hidden_dimension": coordinate["hidden_remainder_dimension"]
        == gates["hidden_remainder_dimension_equal"],
        "Q3_rowspace": coordinate["Q3_rowspace_relative_defect"]
        <= gates["Q3_rowspace_relative_defect_max"],
        "hidden_rate_fraction": coordinate[
            "online_470_hidden_rate_relative_fraction"
        ]
        <= gates["anchor_hidden_rate_relative_fraction_max"],
        "stable_energy": cost["inherited_stable_energy_amplification_max"]
        <= gates["inherited_stable_energy_amplification_max"],
        "online_cost": cost["projected_online_cycle_wall_seconds"]
        <= gates["projected_online_cycle_wall_seconds_max"],
        "online_truth": cost["online_truth_calls_per_macrostep"]
        == gates["online_truth_calls_equal"],
    }
    passed = all(checks.values())
    classification = PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION
    authorized_next = (
        "definitions_only_exact_geometric_28D_departure_chart_and_database_manifest"
        if passed
        else None
    )
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    metrics = {
        "checks": checks,
        "coordinate_geometry": coordinate,
        "cost_and_inheritance": cost,
    }
    _write_json(CANONICAL_DIRECTORY / "metrics.json", metrics)
    np.savez_compressed(
        CANONICAL_DIRECTORY / "online_470_geometry.npz", **arrays
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "selected_architecture": "explicit_nonlinear_conservative_IMEX_470",
        "online_coordinate_rank": coordinate["online_coordinate_rank"],
        "online_coordinate_condition_number": coordinate[
            "online_coordinate_condition_number"
        ],
        "hidden_stable_remainder_dimension": coordinate[
            "hidden_remainder_dimension"
        ],
        "anchor_hidden_rate_relative_fraction": coordinate[
            "online_470_hidden_rate_relative_fraction"
        ],
        "projected_online_cycle_wall_days": cost[
            "projected_online_cycle_wall_days"
        ],
        "nonlinear_28D_closure_identified": False,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": authorized_next,
    }
    _write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    _write_json(
        CANONICAL_DIRECTORY / "parent_lock.json",
        {
            "manifest_commit": MANIFEST_COMMIT,
            "manifest_parent": MANIFEST_PARENT,
            "manifest_tree": MANIFEST_TREE,
            "manifest_package_hashes": _checksums(manifest.ARTIFACT_DIRECTORY),
            "stable_470_package_hashes": _checksums(
                manifest.stable.CANONICAL_DIRECTORY
            ),
            "preflight_package_hashes": _checksums(preflight.CANONICAL_DIRECTORY),
            "fiber_package_hashes": _checksums(FIBER_DIRECTORY),
        },
    )
    source_files = (THIS_RUNNER, THIS_TEST, manifest.THIS_RUNNER, manifest.THIS_TEST)
    _write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "CERTIFIED" if passed else "REJECTED",
            "execution_commit": _git("rev-parse", "HEAD"),
            "execution_tree": _git("rev-parse", "HEAD^{tree}"),
            "tracked_worktree_clean_at_start": True,
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "report": REPORT_RELATIVE,
            "source_hashes": {
                relative: _sha(ROOT / relative) for relative in source_files
            },
            "python": sys.version,
            "platform": platform.platform(),
            "thread_environment": preflight.THREAD_ENVIRONMENT,
        },
    )
    names = tuple(sorted(path.name for path in CANONICAL_DIRECTORY.iterdir()))
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Explicit nonlinear 470-state architecture audit WP10c9d6c7c3b5c4f25aw",
                "",
                "## Classification",
                "",
                f"`{classification}`",
                "",
                "## Structural result",
                "",
                f"The exact 470-coordinate map has rank `{coordinate['online_coordinate_rank']}` and condition number `{coordinate['online_coordinate_condition_number']:.6e}`. It consists of 162 physical, 280 stable-memory, and 28 explicit nonlinear departure coordinates, leaving a 90-dimensional stable remainder.",
                "",
                f"At the primary anchor the unresolved rate fraction falls to `{coordinate['online_470_hidden_rate_relative_fraction']:.6e}`. The Q3 rowspace defect is `{coordinate['Q3_rowspace_relative_defect']:.6e}`.",
                "",
                f"The inherited stable kernel plus a deliberately low-rank 28D nonlinear algebra witness projects to `{cost['projected_online_cycle_wall_seconds']:.6f}` wall seconds (`{cost['projected_online_cycle_wall_days']:.6e}` days) for 100,000 macrosteps. This is algebraic feasibility, not yet a physical cycle certificate.",
                "",
                "## Next gate",
                "",
                f"Authorized next artifact: `{authorized_next}`.",
                "",
                "The next package must replace the failed reaction-lift perturbation with exact Newton retraction on C_phys, then build a physically guarded nonlinear departure-rate database. No branch solve is required. The nonlinear closure, online integrator, and predictive cycle remain unestablished.",
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
    print(json.dumps(_plain(_run()), indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
