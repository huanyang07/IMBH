#!/usr/bin/env python3
"""Execute the frozen projective-kernel independent geometry set."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_active8_projective_kernel_validation_manifest_wp10c9d6c7c3b5c4f25br as manifest  # noqa: E402
import run_causal_inner_active8_tensor_geometry_extension_wp10c9d6c7c3b5c4f25bo as base  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25bs"
MANIFEST_COMMIT = "378e16717e7402d9bdd478cfe76341f68e661dac"
MANIFEST_PARENT = "79a191a6f30537b860edb553e2c1724b9f3fe03f"
MANIFEST_TREE = "4b51077561b42838f7fa8a0e346846b30f7acf04"
PASS_CLASSIFICATION = "active8_projective_kernel_geometry_passed"
FAIL_CLASSIFICATION = "active8_projective_kernel_geometry_failed"
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25bt"

ARTIFACT = (
    "causal_inner_active8_projective_kernel_geometry_"
    "wp10c9d6c7c3b5c4f25bs"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
SCRATCH_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_active8_projective_kernel_geometry_"
    "wp10c9d6c7c3b5c4f25bs.py"
)
THIS_TEST = (
    "tests/test_causal_inner_active8_projective_kernel_geometry_"
    "wp10c9d6c7c3b5c4f25bs.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_ACTIVE8_PROJECTIVE_KERNEL_"
    "GEOMETRY_WP10C9D6C7C3B5C4F25BS_2026-08-19.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
DESIGN_PATH = manifest.CANONICAL_DIRECTORY / "validation_design.npz"
ENGINE_RUNNER = base.ENGINE_RUNNER
ENGINE_TEST = base.ENGINE_TEST
chart_tools = base.chart_tools


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return base._sha(path)


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _checksums(directory: Path) -> dict[str, str]:
    return base._checksums(directory)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    return base._load_npz(path)


def _validate_manifest(*, require_clean: bool) -> dict:
    if _git("rev-parse", MANIFEST_COMMIT) != MANIFEST_COMMIT:
        raise RuntimeError("projective-kernel manifest commit changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^") != MANIFEST_PARENT:
        raise RuntimeError("projective-kernel manifest lineage changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^{{tree}}") != MANIFEST_TREE:
        raise RuntimeError("projective-kernel manifest tree changed")
    hashes = _checksums(manifest.CANONICAL_DIRECTORY)
    summary = _read(manifest.CANONICAL_DIRECTORY / "summary.json")
    contract = _read(manifest.CANONICAL_DIRECTORY / "contract.json")
    provenance = _read(manifest.CANONICAL_DIRECTORY / "provenance.json")
    design_metrics = _read(
        manifest.CANONICAL_DIRECTORY / "design_metrics.json"
    )
    if (
        not summary["passed"]
        or summary["classification"] != manifest.CLASSIFICATION
        or summary["authorized_next"] != WORK_PACKAGE
        or not summary["definitions_only"]
        or summary["new_truth_evaluations"] != 0
        or not summary["design_checks_passed"]
        or not all(design_metrics["checks"].values())
        or not contract["leakage_control"][
            "architecture_and_hyperparameters_frozen_before_new_geometry"
        ]
    ):
        raise RuntimeError("projective-kernel geometry authorization changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"manifest source changed: {relative}")
    if require_clean and _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("projective-kernel geometry requires a clean tree")
    for name, expected in chart_tools.coordinate_tools.THREAD_ENVIRONMENT.items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"thread environment changed: {name}")
    return {"summary": summary, "contract": contract, "hashes": hashes}


def _candidate_specifications() -> list[dict]:
    design = _load_npz(DESIGN_PATH)
    groups = (
        (
            "holdout_high",
            design["new_holdout_directions_active8"],
            manifest.HIGH_COMPONENT_BOUND,
            "0p01",
        ),
        (
            "holdout_low",
            design["new_radial_directions_active8"],
            manifest.LOW_COMPONENT_BOUND,
            "0p005",
        ),
    )
    specifications = []
    for split, directions, bound, amplitude in groups:
        for local_index in range(directions.shape[1]):
            specifications.append(
                {
                    "split": split,
                    "split_direction_index": local_index,
                    "global_direction_index": (
                        manifest.REVEALED_HIGH_DIRECTION_COUNT + local_index
                    ),
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
    contract["exact_geometric_retraction"] = (
        base.high_chart.manifest._contract()["exact_geometric_retraction"]
    )
    return contract


def _fresh_engine():
    engine = base._fresh_engine()
    manifest_adapter = SimpleNamespace(
        **{
            name: getattr(manifest, name)
            for name in dir(manifest)
            if not name.startswith("__")
        },
        ARTIFACT_DIRECTORY=manifest.CANONICAL_DIRECTORY,
    )
    replacements = {
        "manifest": manifest_adapter,
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
        raise RuntimeError("projective-kernel geometry already canonicalized")
    engine = _fresh_engine()
    metrics, arrays = engine._execute()
    checks = engine._gate_checks(
        metrics, frozen["contract"]["binding_geometry_gates"]
    )
    passed = all(checks.values())
    classification = PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION
    authorized_next = AUTHORIZED_NEXT if passed else None
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    engine._write_json(
        CANONICAL_DIRECTORY / "metrics.json", {"checks": checks, **metrics}
    )
    engine._write_npz(
        CANONICAL_DIRECTORY / "projective_kernel_geometry_database.npz", arrays
    )
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
                "# Active-8 projective-kernel geometry WP10c9d6c7c3b5c4f25bs",
                "",
                "## Classification",
                "",
                f"`{classification}`",
                "",
                f"Completed `{metrics['completed_candidate_count']}` of `{manifest.PLANNED_CANDIDATES}` planned signed exact retractions; failures: `{metrics['failed_candidate_count']}`.",
                "",
                f"Maximum C_phys closure `{metrics['maximum_coordinate_residual_infinity']:.6e}`; maximum normalized Q3 defect `{metrics['maximum_normalized_Q3_defect']:.6e}`; maximum transverse fraction `{metrics['maximum_departure_transverse_fraction']:.6e}`.",
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
    print(json.dumps(_fresh_engine()._plain(_run()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
