#!/usr/bin/env python3
"""Execute the active-8 full-tensor database geometry extension."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_active8_tensor_database_extension_manifest_wp10c9d6c7c3b5c4f25bn as manifest  # noqa: E402
import run_causal_inner_expanded_departure_chart_preflight_wp10c9d6c7c3b5c4f25bc as high_chart  # noqa: E402
import run_causal_inner_exact_geometric_departure_chart_preflight_wp10c9d6c7c3b5c4f25ay as chart_tools  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25bo"
MANIFEST_COMMIT = "cd39ab8d11e59a33afc5c6eea86b8285f0cb32de"
MANIFEST_PARENT = "dc6d68b242194bf311c86f162951a7b8a1c3ad0f"
MANIFEST_TREE = "8dc5e454533174e9d96cb5dc78cb27396dbf1c91"
PASS_CLASSIFICATION = "active8_tensor_geometry_extension_passed"
FAIL_CLASSIFICATION = "active8_tensor_geometry_extension_failed"
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25bp"

ARTIFACT = (
    "causal_inner_active8_tensor_geometry_extension_"
    "wp10c9d6c7c3b5c4f25bo"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
SCRATCH_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_active8_tensor_geometry_extension_"
    "wp10c9d6c7c3b5c4f25bo.py"
)
THIS_TEST = (
    "tests/test_causal_inner_active8_tensor_geometry_extension_"
    "wp10c9d6c7c3b5c4f25bo.py"
)
ENGINE_RUNNER = (
    "scripts/run_causal_inner_active8_mixed_geometry_preflight_"
    "wp10c9d6c7c3b5c4f25bk.py"
)
ENGINE_TEST = (
    "tests/test_causal_inner_active8_mixed_geometry_preflight_"
    "wp10c9d6c7c3b5c4f25bk.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ACTIVE8_TENSOR_GEOMETRY_"
    "EXTENSION_WP10C9D6C7C3B5C4F25BO_2026-08-19.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
DESIGN_PATH = manifest.CANONICAL_DIRECTORY / "extension_design.npz"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


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


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: np.asarray(source[name]) for name in source.files}


def _validate_manifest(*, require_clean: bool) -> dict:
    if _git("rev-parse", MANIFEST_COMMIT) != MANIFEST_COMMIT:
        raise RuntimeError("tensor-geometry manifest commit changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^") != MANIFEST_PARENT:
        raise RuntimeError("tensor-geometry manifest lineage changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^{{tree}}") != MANIFEST_TREE:
        raise RuntimeError("tensor-geometry manifest tree changed")
    hashes = _checksums(manifest.CANONICAL_DIRECTORY)
    summary = _read(manifest.CANONICAL_DIRECTORY / "summary.json")
    contract = _read(manifest.CANONICAL_DIRECTORY / "contract.json")
    provenance = _read(manifest.CANONICAL_DIRECTORY / "provenance.json")
    if (
        not summary["passed"]
        or summary["authorized_next"] != WORK_PACKAGE
        or summary["planned_candidate_count"] != manifest.PLANNED_CANDIDATES
        or not contract["definitions_only"]
        or contract["mathematical_architecture"][
            "online_truth_calls_per_macrostep"
        ]
        != 0
        or contract["claim_boundary"]["trajectory_authorized"]
    ):
        raise RuntimeError("tensor-geometry execution authorization changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"manifest source changed: {relative}")
    source_path = (
        manifest.parent.CANONICAL_DIRECTORY / "tensor_architecture_design.npz"
    )
    expected_source_hash = _read(
        manifest.CANONICAL_DIRECTORY / "parent_lock.json"
    )["decisive_input_hashes"]["tensor_architecture_design"]
    if _sha(source_path) != expected_source_hash:
        raise RuntimeError("source tensor-architecture design changed")
    # The extension design is a strict subset copy of the diagnosed source.
    source_design = _load_npz(source_path)
    extension = _load_npz(DESIGN_PATH)
    for name, values in extension.items():
        if name not in source_design or not np.array_equal(values, source_design[name]):
            raise RuntimeError("frozen extension design changed")
    if require_clean and _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("tensor-geometry extension requires a clean tracked tree")
    for name, expected in chart_tools.coordinate_tools.THREAD_ENVIRONMENT.items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"thread environment changed: {name}")
    return {"summary": summary, "contract": contract, "hashes": hashes}


def _candidate_specifications() -> list[dict]:
    design = _load_npz(DESIGN_PATH)
    groups = (
        (
            "training",
            manifest.REVEALED_TRAINING_DIRECTIONS,
            design["additional_training_directions_active8"],
            manifest.HIGH_COMPONENT_BOUND,
            "0p01",
        ),
        (
            "tuning_high",
            manifest.TOTAL_TRAINING_DIRECTIONS,
            design["new_tuning_directions_active8"],
            manifest.HIGH_COMPONENT_BOUND,
            "0p01",
        ),
        (
            "holdout",
            manifest.TOTAL_TRAINING_DIRECTIONS + manifest.NEW_TUNING_DIRECTIONS,
            design["new_holdout_directions_active8"],
            manifest.HIGH_COMPONENT_BOUND,
            "0p01",
        ),
        (
            "tuning_low",
            manifest.TOTAL_TRAINING_DIRECTIONS,
            design["new_tuning_directions_active8"],
            manifest.LOW_COMPONENT_BOUND,
            "0p005",
        ),
    )
    specifications = []
    for split, offset, directions, bound, amplitude in groups:
        for local_index in range(directions.shape[1]):
            specifications.append(
                {
                    "split": split,
                    "split_direction_index": local_index,
                    "global_direction_index": offset + local_index,
                    "active_direction": np.asarray(
                        directions[:, local_index], dtype=float
                    ),
                    "component_bound": float(bound),
                    "amplitude_label": amplitude,
                }
            )
    return specifications


def _retraction_contract() -> dict:
    contract = manifest._contract()
    contract["binding_preflight_gates"] = contract["binding_geometry_gates"]
    contract["exact_geometric_retraction"] = high_chart.manifest._contract()[
        "exact_geometric_retraction"
    ]
    return contract


def _fresh_engine():
    path = ROOT / ENGINE_RUNNER
    spec = importlib.util.spec_from_file_location(
        "_active8_tensor_geometry_engine", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load certified geometry engine")
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    replacements = {
        "manifest": manifest,
        "WORK_PACKAGE": WORK_PACKAGE,
        "MANIFEST_COMMIT": MANIFEST_COMMIT,
        "MANIFEST_PARENT": MANIFEST_PARENT,
        "MANIFEST_TREE": MANIFEST_TREE,
        "PASS_CLASSIFICATION": PASS_CLASSIFICATION,
        "FAIL_CLASSIFICATION": FAIL_CLASSIFICATION,
        "ARTIFACT": ARTIFACT,
        "CANONICAL_DIRECTORY": CANONICAL_DIRECTORY,
        "SCRATCH_DIRECTORY": SCRATCH_DIRECTORY,
        "THIS_RUNNER": THIS_RUNNER,
        "THIS_TEST": THIS_TEST,
        "REPORT_RELATIVE": REPORT_RELATIVE,
        "REPORT_PATH": REPORT_PATH,
        "CANONICAL_MANIFEST": CANONICAL_MANIFEST,
        "CANONICAL_SUMMARY": CANONICAL_SUMMARY,
        "DESIGN_PATH": DESIGN_PATH,
        "_validate_manifest": _validate_manifest,
        "_candidate_specifications": _candidate_specifications,
        "_retraction_contract": _retraction_contract,
    }
    for name, value in replacements.items():
        setattr(engine, name, value)
    return engine


def _gate_checks(metrics: dict, gates: dict) -> dict:
    return _fresh_engine()._gate_checks(metrics, gates)


def _run() -> dict:
    frozen = _validate_manifest(require_clean=True)
    if CANONICAL_DIRECTORY.exists():
        raise RuntimeError("tensor-geometry extension is already canonicalized")
    engine = _fresh_engine()
    metrics, arrays = engine._execute()
    checks = engine._gate_checks(metrics, frozen["contract"]["binding_geometry_gates"])
    passed = all(checks.values())
    classification = PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION
    authorized_next = AUTHORIZED_NEXT if passed else None
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    engine._write_json(
        CANONICAL_DIRECTORY / "metrics.json", {"checks": checks, **metrics}
    )
    engine._write_npz(CANONICAL_DIRECTORY / "tensor_geometry_database.npz", arrays)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "completed_candidate_count": metrics["completed_candidate_count"],
        "failed_candidate_count": metrics["failed_candidate_count"],
        "maximum_coordinate_residual_infinity": metrics[
            "maximum_coordinate_residual_infinity"
        ],
        "maximum_normalized_Q3_defect": metrics["maximum_normalized_Q3_defect"],
        "maximum_departure_transverse_fraction": metrics[
            "maximum_departure_transverse_fraction"
        ],
        "nonbase_continuous_rate_evaluations": 0,
        "predictive_cycle_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": authorized_next,
    }
    engine._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    engine._write_json(
        CANONICAL_DIRECTORY / "parent_lock.json",
        {
            "manifest_commit": MANIFEST_COMMIT,
            "manifest_parent": MANIFEST_PARENT,
            "manifest_tree": MANIFEST_TREE,
            "manifest_hashes": _checksums(manifest.CANONICAL_DIRECTORY),
        },
    )
    source_files = (
        THIS_RUNNER,
        THIS_TEST,
        ENGINE_RUNNER,
        ENGINE_TEST,
        manifest.THIS_RUNNER,
        manifest.THIS_TEST,
    )
    engine._write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "CERTIFIED" if passed else "REJECTED",
            "execution_commit": _git("rev-parse", "HEAD"),
            "execution_tree": _git("rev-parse", "HEAD^{tree}"),
            "tracked_worktree_clean_at_start": True,
            "resumed_from_candidate_count": metrics["resumed_candidate_count"],
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "certified_geometry_engine": ENGINE_RUNNER,
            "report": REPORT_RELATIVE,
            "source_hashes": {
                relative: _sha(ROOT / relative) for relative in source_files
            },
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "thread_environment": chart_tools.coordinate_tools.THREAD_ENVIRONMENT,
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
                "# Active-8 tensor geometry extension WP10c9d6c7c3b5c4f25bo",
                "",
                "## Classification",
                "",
                f"`{classification}`",
                "",
                f"Completed `{metrics['completed_candidate_count']}` of `{manifest.PLANNED_CANDIDATES}` planned signed exact retractions; failures: `{metrics['failed_candidate_count']}`.",
                "",
                f"Maximum C_phys closure is `{metrics['maximum_coordinate_residual_infinity']:.6e}`; maximum normalized Q3 defect is `{metrics['maximum_normalized_Q3_defect']:.6e}`; maximum transverse departure fraction is `{metrics['maximum_departure_transverse_fraction']:.6e}`.",
                "",
                f"Authorized next work package: `{authorized_next}`. No nonbase rate was evaluated and no state was propagated.",
                "",
            )
        ),
        encoding="utf-8",
    )
    engine._update_catalog(summary)
    if SCRATCH_DIRECTORY.exists():
        shutil.rmtree(SCRATCH_DIRECTORY)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        raise SystemExit("pass --run")
    print(json.dumps(engine_plain(_run()), indent=2, sort_keys=True), flush=True)
    return 0


def engine_plain(value):
    return _fresh_engine()._plain(value)


if __name__ == "__main__":
    raise SystemExit(main())
