#!/usr/bin/env python3
"""Execute the high-order square-root Hermite confirmation."""

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


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_high_order_hermite_manifest_wp10c9d6c7c3b5c4f25af as manifest  # noqa: E402
import run_causal_inner_relative_hermite_resolvent_audit_wp10c9d6c7c3b5c4f25ae as hermite_tools  # noqa: E402
import run_causal_inner_saved_R32_memory_selection_audit_wp10c9d6c7c3b5c4f25m as memory_tools  # noqa: E402
import run_causal_inner_square_root_transfer_seeded_audit_wp10c9d6c7c3b5c4f25aa as square_tools  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25ag"
MANIFEST_COMMIT = "4f420b2d6d3516152ee1593e77aeb7d2911b82f1"
MANIFEST_PARENT = "ec4302ee6fec94dbe2b9af18592d48fbee9b6a10"
MANIFEST_TREE = "ea77a7b3687e6263cd2848051bdd8bd534c8fc9a"

PASS_CLASSIFICATION = (
    "two_anchor_higher_order_square_root_Hermite_reduction_passed_"
    "parametric_online_architecture_manifest_authorized"
)
CAP_FAIL_CLASSIFICATION = (
    "higher_order_Hermite_failed_through_R510_"
    "direct_stable_rational_realization_reassessment_required"
)
NUMERICAL_FAIL_CLASSIFICATION = "higher_order_Hermite_numerical_failure_stop"

ARTIFACT = "causal_inner_high_order_hermite_audit_wp10c9d6c7c3b5c4f25ag"
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_high_order_hermite_audit_"
    "wp10c9d6c7c3b5c4f25ag.py"
)
THIS_TEST = (
    "tests/test_causal_inner_high_order_hermite_audit_"
    "wp10c9d6c7c3b5c4f25ag.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_HIGH_ORDER_HERMITE_"
    "AUDIT_WP10C9D6C7C3B5C4F25AG_2026-08-18.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

THREAD_ENVIRONMENT = {
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


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
    for line in (directory / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        if _sha(directory / name) != expected:
            raise RuntimeError(f"checksum mismatch: {directory / name}")
        recorded[name] = expected
    return recorded


def _validate_manifest(*, require_clean: bool) -> dict:
    if _git("rev-parse", MANIFEST_COMMIT) != MANIFEST_COMMIT:
        raise RuntimeError("high-order Hermite manifest commit changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^") != MANIFEST_PARENT:
        raise RuntimeError("high-order Hermite manifest lineage changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^{{tree}}") != MANIFEST_TREE:
        raise RuntimeError("high-order Hermite manifest tree changed")
    hashes = _checksums(manifest.ARTIFACT_DIRECTORY)
    summary = _read(manifest.ARTIFACT_DIRECTORY / "summary.json")
    contract = _read(manifest.ARTIFACT_DIRECTORY / "contract.json")
    provenance = _read(manifest.ARTIFACT_DIRECTORY / "provenance.json")
    split = contract["frequency_split"]
    if (
        not summary["passed"]
        or summary["authorized_next"] != WORK_PACKAGE
        or summary["non_DC_validation_information_used_in_basis"]
        or contract["execution_budget"]["candidate_hidden_orders"]
        != list(manifest.HIDDEN_ORDERS)
        or split["non_DC_validation_frequencies_evaluated_before_freeze"]
        or split["validation_information_may_influence_basis"]
    ):
        raise RuntimeError("high-order Hermite execution authorization changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"manifest source changed: {relative}")
    for name, expected in contract["parent_decisive_hashes"].items():
        if _sha(manifest.PARENT_DIRECTORY / name) != expected:
            raise RuntimeError(f"parent input changed: {name}")
    saved_paths = {
        "primary_generator": manifest.PRIMARY_GENERATOR_DIRECTORY
        / "descriptor_A.npz",
        "primary_output": manifest.PRIMARY_GENERATOR_DIRECTORY / "projection.npz",
        "heldout_generator_and_output": manifest.CROSS_ANCHOR_DIRECTORY
        / "heldout_generator.npz",
        "R32_projection": manifest.R32_DIRECTORY / "R32_projection_promotion.npz",
        "frequency_ladder": manifest.R32_DIRECTORY / "R32_transfer.npz",
    }
    for name, path in saved_paths.items():
        if _sha(path) != contract["saved_input_hashes"][name]:
            raise RuntimeError(f"saved input changed: {name}")
    if require_clean and not _tracked_tree_clean():
        raise RuntimeError("high-order Hermite audit requires a clean tracked tree")
    for name, expected in THREAD_ENVIRONMENT.items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"thread environment changed: {name}")
    return {"summary": summary, "contract": contract, "hashes": hashes}


def _interpolate_interval(left: float, right: float, fraction: float) -> float:
    if left == 0.0:
        return fraction * right
    return left * (right / left) ** fraction


def _refined_frequency_grids(
    parent_frequencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    parent = np.asarray(parent_frequencies, dtype=float)
    if (
        parent.ndim != 1
        or parent.size != 33
        or parent[0] != 0.0
        or np.any(np.diff(parent) <= 0.0)
    ):
        raise ValueError("parent frequency ladder changed")
    training = [float(parent[0])]
    validation = [0.0]
    for left, right in zip(parent[:-1], parent[1:], strict=True):
        training.extend(
            _interpolate_interval(float(left), float(right), fraction)
            for fraction in (0.25, 0.5, 0.75, 1.0)
        )
        validation.extend(
            _interpolate_interval(float(left), float(right), fraction)
            for fraction in (0.125, 0.375, 0.625, 0.875)
        )
    training_array = np.asarray(training)
    validation_array = np.asarray(validation)
    if (
        training_array.size != 129
        or validation_array.size != 129
        or np.intersect1d(training_array[1:], validation_array[1:]).size
    ):
        raise RuntimeError("prospective frequency split is not disjoint")
    return training_array, validation_array


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
        raise RuntimeError("high-order Hermite audit is already canonicalized")
    began = time.perf_counter()
    with np.load(
        manifest.PRIMARY_GENERATOR_DIRECTORY / "descriptor_A.npz",
        allow_pickle=False,
    ) as source:
        primary_generator = np.asarray(source["complete_fixed_Q_generator"])
    with np.load(
        manifest.PRIMARY_GENERATOR_DIRECTORY / "projection.npz", allow_pickle=False
    ) as source:
        primary_output = np.asarray(source["output_map"])
    with np.load(
        manifest.CROSS_ANCHOR_DIRECTORY / "heldout_generator.npz",
        allow_pickle=False,
    ) as source:
        heldout_generator = np.asarray(source["complete_fixed_Q_generator"])
        heldout_output = np.asarray(source["output_map"])
    with np.load(
        manifest.R32_DIRECTORY / "R32_projection_promotion.npz", allow_pickle=False
    ) as source:
        restriction = np.asarray(source["resolved_restriction"])
    with np.load(
        manifest.R32_DIRECTORY / "R32_transfer.npz", allow_pickle=False
    ) as source:
        parent_frequencies = np.asarray(source["angular_frequencies_per_second"])
    training_frequencies, validation_frequencies = _refined_frequency_grids(
        parent_frequencies
    )
    with np.load(
        manifest.FIBER_DIRECTORY / "decisive_fibers.npz", allow_pickle=False
    ) as source:
        fiber = {name: np.asarray(source[name]) for name in source.files}

    generators = {"primary": primary_generator, "heldout": heldout_generator}
    outputs = {"primary": primary_output, "heldout": heldout_output}
    gates = frozen["contract"]["binding_gates"]
    systems = {}
    prepared = {}
    hidden_vectors = {}
    base_metrics = {}
    snapshot_metrics = {}
    base_passed = True
    for anchor in ("primary", "heldout"):
        systems[anchor], base_metrics[anchor] = (
            square_tools._square_root_stable_system(
                generators[anchor],
                outputs[anchor],
                restriction,
                fiber[f"{anchor}_right_basis"],
                fiber[f"{anchor}_left_dual_transpose"],
                fiber[f"{anchor}_unstable_operator"],
            )
        )
        prepared[anchor] = hermite_tools._prepare_reference(
            systems[anchor], training_frequencies, validation_frequencies
        )
        hidden_vectors[anchor], _, snapshot_metrics[anchor] = (
            hermite_tools._relative_hermite_basis(
                systems[anchor]["hidden_operator"],
                prepared[anchor]["forcing"],
                prepared[anchor]["observation"],
                training_frequencies,
                manifest.PHYSICAL_DIMENSION,
            )
        )
        base_metrics[anchor]["passed"] = hermite_tools._base_pass(
            base_metrics[anchor], snapshot_metrics[anchor], gates
        )
        base_passed &= base_metrics[anchor]["passed"]

    candidate_metrics = []
    error_arrays = {
        "training_angular_frequencies_per_second": training_frequencies,
        "validation_angular_frequencies_per_second": validation_frequencies,
    }
    selected = None
    selected_arrays = None
    best = None
    best_arrays = None
    all_candidate_numerical = True
    for order in manifest.HIDDEN_ORDERS:
        item = {
            "hidden_order": order,
            "online_dimension": (
                manifest.PHYSICAL_DIMENSION
                + manifest.EXACT_NONSTABLE_DIMENSION
                + order
            ),
        }
        model_arrays = {}
        for anchor in ("primary", "heldout"):
            item[anchor], arrays = hermite_tools._candidate(
                systems[anchor],
                prepared[anchor],
                hidden_vectors[anchor],
                order,
                training_frequencies,
                validation_frequencies,
                gates,
            )
            all_candidate_numerical &= item[anchor]["numerical_passed"]
            for name, value in arrays.items():
                if name.endswith("_errors"):
                    error_arrays[f"Z{order}_{anchor}_{name}"] = value
                else:
                    model_arrays[f"{anchor}_{name}"] = value
        primary_q = np.linalg.qr(
            model_arrays["primary_hidden_truth_trial"], mode="reduced"
        )[0]
        heldout_q = np.linalg.qr(
            model_arrays["heldout_hidden_truth_trial"], mode="reduced"
        )[0]
        cosines = np.linalg.svd(primary_q.T @ heldout_q, compute_uv=False)
        item["cross_anchor_hidden_principal_cosine_min"] = float(np.min(cosines))
        item["cross_anchor_hidden_largest_principal_angle_degrees"] = float(
            np.degrees(np.arccos(np.clip(np.min(cosines), -1.0, 1.0)))
        )
        item["cross_anchor_passed"] = bool(
            item["cross_anchor_hidden_principal_cosine_min"]
            >= gates["cross_anchor_hidden_principal_cosine_min"]
        )
        item["joint_passed"] = bool(
            base_passed
            and item["cross_anchor_passed"]
            and item["primary"]["passed"]
            and item["heldout"]["passed"]
        )
        item["maximum_gate_ratio"] = hermite_tools._candidate_score(item, gates)
        candidate_metrics.append(item)
        if best is None or item["maximum_gate_ratio"] < best["maximum_gate_ratio"]:
            best = item
            best_arrays = model_arrays
        if item["joint_passed"]:
            selected = item
            selected_arrays = model_arrays
            break

    elapsed = float(time.perf_counter() - began)
    numerical_passed = bool(
        base_passed
        and all_candidate_numerical
        and np.isfinite(elapsed)
        and elapsed
        <= 3600.0 * frozen["contract"]["execution_budget"]["maximum_wall_hours"]
    )
    passed = bool(numerical_passed and selected is not None)
    if not numerical_passed:
        classification = NUMERICAL_FAIL_CLASSIFICATION
        authorized_next = None
    elif passed:
        classification = PASS_CLASSIFICATION
        authorized_next = (
            "definitions_only_stable_parametric_online_architecture_manifest"
        )
    else:
        classification = CAP_FAIL_CLASSIFICATION
        authorized_next = None

    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write_json(
        CANONICAL_DIRECTORY / "metrics.json",
        {
            "base_metrics": base_metrics,
            "snapshot_metrics": snapshot_metrics,
            "candidate_metrics": candidate_metrics,
            "selected": selected,
            "best": best,
            "numerical_passed": numerical_passed,
            "wall_seconds": elapsed,
        },
    )
    np.savez_compressed(
        CANONICAL_DIRECTORY / "candidate_errors.npz", **error_arrays
    )
    np.savez_compressed(
        CANONICAL_DIRECTORY / "decisive_model.npz",
        **(
            selected_arrays
            if selected_arrays is not None
            else (best_arrays if best_arrays is not None else {})
        ),
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "numerical_passed": numerical_passed,
        "base_architecture_passed": base_passed,
        "training_frequency_count": int(training_frequencies.size),
        "validation_frequency_count_including_shared_DC": int(
            validation_frequencies.size
        ),
        "non_DC_validation_information_used_in_basis": False,
        "selected_hidden_order": (
            None if selected is None else selected["hidden_order"]
        ),
        "selected_online_dimension": (
            None if selected is None else selected["online_dimension"]
        ),
        "selected_maximum_gate_ratio": (
            None if selected is None else selected["maximum_gate_ratio"]
        ),
        "best_hidden_order": None if best is None else best["hidden_order"],
        "best_maximum_gate_ratio": (
            None if best is None else best["maximum_gate_ratio"]
        ),
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
    _write_json(
        CANONICAL_DIRECTORY / "parent_lock.json",
        {
            "manifest_commit": MANIFEST_COMMIT,
            "manifest_parent": MANIFEST_PARENT,
            "manifest_tree": MANIFEST_TREE,
            "manifest_package_hashes": _checksums(manifest.ARTIFACT_DIRECTORY),
            "parent_package_hashes": _checksums(manifest.PARENT_DIRECTORY),
            "fiber_package_hashes": _checksums(manifest.FIBER_DIRECTORY),
        },
    )
    source_files = (
        THIS_RUNNER,
        THIS_TEST,
        manifest.THIS_RUNNER,
        manifest.THIS_TEST,
        hermite_tools.THIS_RUNNER,
        hermite_tools.THIS_TEST,
        square_tools.THIS_RUNNER,
        memory_tools.THIS_RUNNER,
    )
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
            "thread_environment": THREAD_ENVIRONMENT,
        },
    )
    names = tuple(sorted(path.name for path in CANONICAL_DIRECTORY.iterdir()))
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{_sha(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names
        ),
        encoding="utf-8",
    )
    if selected is None:
        detail = (
            f"No hidden order through 320 passed. The best order was "
            f"`{best['hidden_order'] if best else None}` with maximum gate ratio "
            f"`{best['maximum_gate_ratio'] if best else None}`."
        )
    else:
        detail = (
            f"Selected hidden order `{selected['hidden_order']}` and total online "
            f"dimension `{selected['online_dimension']}` with maximum gate ratio "
            f"`{selected['maximum_gate_ratio']:.6e}`."
        )
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# High-order square-root Hermite audit WP10c9d6c7c3b5c4f25ag",
                "",
                "## Classification",
                "",
                f"`{classification}`",
                "",
                "This saved-generator confirmation used the prospectively frozen 129-point training grid and the previously unseen 128-frequency eighth-point validation set plus shared DC.",
                "",
                detail,
                "",
                f"Authorized next artifact: `{authorized_next}`. No online integrator, predictive cycle, or reduced slow evolution is authorized.",
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
