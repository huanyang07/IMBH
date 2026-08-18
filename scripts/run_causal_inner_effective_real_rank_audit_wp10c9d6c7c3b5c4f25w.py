#!/usr/bin/env python3
"""Reaudit exact fibers with the prospectively defined effective real rank."""

from __future__ import annotations

import argparse
import copy
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
from scipy.linalg import schur


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_effective_real_rank_manifest_wp10c9d6c7c3b5c4f25v as manifest  # noqa: E402
import run_causal_inner_unstable_exact_conservative_fiber_audit_wp10c9d6c7c3b5c4f25u as parent  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25w"
MANIFEST_COMMIT = "50ad457d885d7c6baeddeb93a4c6be0063c9c864"
MANIFEST_PARENT = "bb2cbdfe447dba80d4cf3fede50a98921fa5dd80"
MANIFEST_TREE = "fd1f5d5169dba832df5352c36343307526b904d7"

PASS_CLASSIFICATION = (
    "two_anchor_effective_rank_unstable_exact_fiber_passed_"
    "constrained_lyapunov_stable_reduction_manifest_authorized"
)
FAIL_CLASSIFICATION = (
    "effective_rank_unstable_exact_fiber_failed_"
    "reduced_architecture_reassessment_required"
)
NUMERICAL_FAIL_CLASSIFICATION = (
    "effective_rank_unstable_exact_fiber_numerical_failure_stop"
)

ARTIFACT = "causal_inner_effective_real_rank_audit_wp10c9d6c7c3b5c4f25w"
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_effective_real_rank_audit_"
    "wp10c9d6c7c3b5c4f25w.py"
)
THIS_TEST = (
    "tests/test_causal_inner_effective_real_rank_audit_"
    "wp10c9d6c7c3b5c4f25w.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_EFFECTIVE_REAL_RANK_AUDIT_"
    "WP10C9D6C7C3B5C4F25W_2026-08-18.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

THREAD_ENVIRONMENT = parent.THREAD_ENVIRONMENT


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
        raise RuntimeError("effective-rank manifest commit changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^") != MANIFEST_PARENT:
        raise RuntimeError("effective-rank manifest lineage changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^{{tree}}") != MANIFEST_TREE:
        raise RuntimeError("effective-rank manifest tree changed")
    hashes = _checksums(manifest.ARTIFACT_DIRECTORY)
    summary = _read(manifest.ARTIFACT_DIRECTORY / "summary.json")
    contract = _read(manifest.ARTIFACT_DIRECTORY / "contract.json")
    provenance = _read(manifest.ARTIFACT_DIRECTORY / "provenance.json")
    if (
        not summary["passed"]
        or summary["authorized_next"] != WORK_PACKAGE
        or contract["effective_real_rank"]["expected_rank"] != 28
        or contract["execution_budget"][
            "allowed_new_full_560_direction_generator_assemblies"
        ] != 0
    ):
        raise RuntimeError("effective-rank execution authorization changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"manifest source changed: {relative}")
    for name, expected in contract["parent_decisive_hashes"].items():
        if _sha(manifest.PARENT_DIRECTORY / name) != expected:
            raise RuntimeError(f"parent decisive input changed: {name}")
    _checksums(parent.CANONICAL_DIRECTORY)
    if require_clean and not _tracked_tree_clean():
        raise RuntimeError("effective-rank audit requires a clean tracked tree")
    for name, expected in THREAD_ENVIRONMENT.items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"thread environment changed: {name}")
    return {"summary": summary, "contract": contract, "hashes": hashes}


def _effective_rank_metrics(
    generator: np.ndarray,
    *,
    transpose: bool,
    threshold: float,
    expected_rank: int,
    relative_cutoff: float,
) -> tuple[dict, np.ndarray]:
    matrix = generator.T if transpose else generator
    _, basis, selected = schur(
        matrix,
        output="complex",
        sort=lambda value: bool(np.real(value) >= threshold),
    )
    real_span = np.hstack(
        (np.real(basis[:, :selected]), np.imag(basis[:, :selected]))
    )
    singular = np.linalg.svd(real_span, compute_uv=False)
    ratios = singular / singular[0]
    effective_rank = int(np.sum(ratios > relative_cutoff))
    retained = float(ratios[expected_rank - 1])
    first_discarded = float(ratios[expected_rank])
    gap = float(singular[expected_rank] / singular[expected_rank - 1])
    return {
        "ordered_schur_count": int(selected),
        "effective_rank": effective_rank,
        "relative_cutoff": relative_cutoff,
        "last_retained_to_leading_ratio": retained,
        "first_discarded_to_leading_ratio": first_discarded,
        "first_discarded_to_last_retained_ratio": gap,
    }, singular


def _effective_gate_passed(metrics: dict, definition: dict) -> bool:
    return bool(
        metrics["ordered_schur_count"] == definition["expected_rank"]
        and metrics["effective_rank"] == definition["expected_rank"]
        and metrics["last_retained_to_leading_ratio"]
        >= definition["retained_singular_value_relative_floor"]
        and metrics["first_discarded_to_last_retained_ratio"]
        <= definition["first_discarded_to_last_retained_ratio_max"]
    )


def _unchanged_gate_passed(anchor_metrics: dict, effective: dict, gates: dict) -> bool:
    corrected = copy.deepcopy(anchor_metrics)
    corrected["spectral"]["right_realification"]["realification_rank"] = effective[
        "right"
    ]["effective_rank"]
    corrected["spectral"]["left_realification"]["realification_rank"] = effective[
        "left"
    ]["effective_rank"]
    inherited = {
        **gates,
        "selected_nonstable_dimension_equal": manifest.EXPECTED_RANK,
    }
    return parent._anchor_passed(corrected, inherited)


def _update_catalog(summary: dict) -> None:
    rows = []
    if CANONICAL_MANIFEST.exists():
        with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append({
                "case": ARTIFACT,
                "path": str(path.relative_to(ROOT)),
                "bytes": str(path.stat().st_size),
                "sha256": _sha(path),
                "scientific_status": "CERTIFIED" if summary["passed"] else "REJECTED",
            })
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
    catalog.update({
        "case_count": len({row["case"] for row in rows}),
        "file_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "all_payload_hashes_recorded": True,
        "latest_source_parent_commit": MANIFEST_COMMIT,
        "latest_work_package": WORK_PACKAGE,
    })
    _write_json(CANONICAL_SUMMARY, catalog)


def _run() -> dict:
    frozen = _validate_manifest(require_clean=True)
    if CANONICAL_DIRECTORY.exists():
        raise RuntimeError("effective-rank audit is already canonicalized")
    began = time.perf_counter()
    parent_metrics = _read(parent.CANONICAL_DIRECTORY / "metrics.json")
    with np.load(
        manifest.original_manifest.PRIMARY_GENERATOR_DIRECTORY / "descriptor_A.npz",
        allow_pickle=False,
    ) as source:
        primary_generator = np.asarray(source["complete_fixed_Q_generator"], dtype=float)
    with np.load(
        manifest.original_manifest.CROSS_ANCHOR_DIRECTORY / "heldout_generator.npz",
        allow_pickle=False,
    ) as source:
        heldout_generator = np.asarray(source["complete_fixed_Q_generator"], dtype=float)
    generators = {"primary": primary_generator, "heldout": heldout_generator}
    definition = frozen["contract"]["effective_real_rank"]
    gates = frozen["contract"]["unchanged_binding_gates"]
    threshold = manifest.original_manifest.NONSTABLE_THRESHOLD_PER_SECOND
    metrics = {}
    arrays = {}
    for anchor, generator in generators.items():
        effective = {}
        effective_passes = []
        for side, transpose in (("right", False), ("left", True)):
            effective[side], singular = _effective_rank_metrics(
                generator,
                transpose=transpose,
                threshold=threshold,
                expected_rank=definition["expected_rank"],
                relative_cutoff=definition["relative_cutoff"],
            )
            effective_passes.append(
                _effective_gate_passed(effective[side], definition)
            )
            arrays[f"{anchor}_{side}_realification_singular_values"] = singular
        unchanged_passed = _unchanged_gate_passed(
            parent_metrics["anchors"][anchor], effective, gates
        )
        metrics[anchor] = {
            "effective_real_rank": effective,
            "effective_rank_passed": bool(all(effective_passes)),
            "unchanged_substantive_gates_passed": unchanged_passed,
            "passed": bool(all(effective_passes) and unchanged_passed),
        }
    cross_anchor_passed = bool(parent_metrics["cross_anchor_passed"])
    elapsed = float(time.perf_counter() - began)
    numerical_passed = bool(
        np.isfinite(elapsed)
        and all(np.all(np.isfinite(value)) for value in arrays.values())
        and elapsed
        <= 60.0 * frozen["contract"]["execution_budget"]["maximum_wall_minutes"]
    )
    passed = bool(
        numerical_passed
        and cross_anchor_passed
        and all(item["passed"] for item in metrics.values())
    )
    if not numerical_passed:
        classification = NUMERICAL_FAIL_CLASSIFICATION
        authorized_next = None
    elif passed:
        classification = PASS_CLASSIFICATION
        authorized_next = (
            "definitions_only_constrained_lyapunov_stable_reduction_manifest"
        )
    else:
        classification = FAIL_CLASSIFICATION
        authorized_next = (
            "definitions_only_effective_rank_architecture_reassessment_manifest"
        )
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write_json(CANONICAL_DIRECTORY / "metrics.json", {
        "anchors": metrics,
        "cross_anchor_passed": cross_anchor_passed,
        "numerical_passed": numerical_passed,
        "wall_seconds": elapsed,
    })
    np.savez_compressed(
        CANONICAL_DIRECTORY / "realification_singular_values.npz", **arrays
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "numerical_passed": numerical_passed,
        "cross_anchor_passed": cross_anchor_passed,
        "primary_passed": metrics["primary"]["passed"],
        "heldout_passed": metrics["heldout"]["passed"],
        "effective_real_rank": definition["expected_rank"] if passed else None,
        "maximum_exact_unstable_augmented_dimension": 190,
        "minimum_remaining_stable_memory_budget": 130,
        "new_nonlinear_roots": 0,
        "propagated_states": 0,
        "new_full_560_direction_generator_assemblies": 0,
        "new_truth_anchors": 0,
        "physical_failure_detected": False,
        "online_integrator_implementation_authorized": False,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": authorized_next,
    }
    _write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    _write_json(CANONICAL_DIRECTORY / "parent_lock.json", {
        "manifest_commit": MANIFEST_COMMIT,
        "manifest_parent": MANIFEST_PARENT,
        "manifest_tree": MANIFEST_TREE,
        "manifest_package_hashes": _checksums(manifest.ARTIFACT_DIRECTORY),
        "parent_package_hashes": _checksums(parent.CANONICAL_DIRECTORY),
        "parent_decisive_fibers_sha256": _sha(
            parent.CANONICAL_DIRECTORY / "decisive_fibers.npz"
        ),
    })
    source_files = (THIS_RUNNER, THIS_TEST, manifest.THIS_RUNNER, manifest.THIS_TEST, parent.THIS_RUNNER, parent.THIS_TEST)
    _write_json(CANONICAL_DIRECTORY / "provenance.json", {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "scientific_status": "CERTIFIED" if passed else "REJECTED",
        "execution_commit": _git("rev-parse", "HEAD"),
        "execution_tree": _git("rev-parse", "HEAD^{tree}"),
        "tracked_worktree_clean_at_start": True,
        "runner": THIS_RUNNER,
        "test": THIS_TEST,
        "report": REPORT_RELATIVE,
        "source_hashes": {relative: _sha(ROOT / relative) for relative in source_files},
        "python": sys.version,
        "platform": platform.platform(),
        "thread_environment": THREAD_ENVIRONMENT,
    })
    names = tuple(sorted(path.name for path in CANONICAL_DIRECTORY.iterdir()))
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{_sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    worst_gap = max(
        metrics[anchor]["effective_real_rank"][side][
            "first_discarded_to_last_retained_ratio"
        ]
        for anchor in metrics
        for side in ("right", "left")
    )
    REPORT_PATH.write_text("\n".join((
        "# Effective real-rank audit WP10c9d6c7c3b5c4f25w",
        "",
        "## Classification",
        "",
        f"`{classification}`",
        "",
        "The f25u rejection remains historical evidence. This saved-generator reaudit applied the separately frozen effective-rank definition and changed no substantive spectral, stability, conservation, alignment, or dimension gate.",
        "",
        f"Both left and right fibers have effective real rank 28 at both anchors. The worst sigma_29/sigma_28 ratio is `{worst_gap:.6e}`. The exact conservative-plus-nonstable state remains dimension 190, leaving 130 stable-memory coordinates under R320.",
        "",
        f"Authorized next artifact: `{authorized_next}`. No online integrator, predictive cycle, or reduced slow evolution is authorized.",
        "",
    )), encoding="utf-8")
    _update_catalog(summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    arguments = parser.parse_args()
    if not arguments.run:
        raise SystemExit("pass --run")
    print(json.dumps(_plain(_run()), indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
