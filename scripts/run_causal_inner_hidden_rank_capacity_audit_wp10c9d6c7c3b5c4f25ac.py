#!/usr/bin/env python3
"""Execute the exact-hidden pointwise rank-capacity audit."""

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

import run_causal_inner_hidden_rank_capacity_manifest_wp10c9d6c7c3b5c4f25ab as manifest  # noqa: E402
import run_causal_inner_saved_R32_memory_selection_audit_wp10c9d6c7c3b5c4f25m as memory_tools  # noqa: E402
import run_causal_inner_square_root_transfer_seeded_audit_wp10c9d6c7c3b5c4f25aa as square_tools  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25ac"
MANIFEST_COMMIT = "4ea831cd7e223f85f831f4d9780e37f4782a49cf"
MANIFEST_PARENT = "4b9ec5b4db30ea0e026d5f3c51efdf48a977d133"
MANIFEST_TREE = "76cf44f41d9e8d0e99232fc63aecdd8947e92f43"

PASS_CLASSIFICATION = (
    "two_anchor_R130_pointwise_transfer_capacity_not_ruled_out_"
    "direct_structure_preserving_basis_manifest_authorized"
)
IMPOSSIBLE_CLASSIFICATION = (
    "R320_pointwise_transfer_rank_capacity_impossible_"
    "dimension_or_resolved_coordinates_must_change"
)
MARGINAL_CLASSIFICATION = (
    "R320_pointwise_transfer_rank_capacity_marginal_"
    "no_basis_search_authorized"
)
NUMERICAL_FAIL_CLASSIFICATION = "hidden_rank_capacity_numerical_failure_stop"

ARTIFACT = (
    "causal_inner_hidden_rank_capacity_audit_"
    "wp10c9d6c7c3b5c4f25ac"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = (
    "scripts/run_causal_inner_hidden_rank_capacity_audit_"
    "wp10c9d6c7c3b5c4f25ac.py"
)
THIS_TEST = (
    "tests/test_causal_inner_hidden_rank_capacity_audit_"
    "wp10c9d6c7c3b5c4f25ac.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_HIDDEN_RANK_CAPACITY_"
    "AUDIT_WP10C9D6C7C3B5C4F25AC_2026-08-18.md"
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
        raise RuntimeError("rank-capacity manifest commit changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^") != MANIFEST_PARENT:
        raise RuntimeError("rank-capacity manifest lineage changed")
    if _git("rev-parse", f"{MANIFEST_COMMIT}^{{tree}}") != MANIFEST_TREE:
        raise RuntimeError("rank-capacity manifest tree changed")
    hashes = _checksums(manifest.ARTIFACT_DIRECTORY)
    summary = _read(manifest.ARTIFACT_DIRECTORY / "summary.json")
    contract = _read(manifest.ARTIFACT_DIRECTORY / "contract.json")
    provenance = _read(manifest.ARTIFACT_DIRECTORY / "provenance.json")
    if (
        not summary["passed"]
        or summary["authorized_next"] != WORK_PACKAGE
        or contract["exact_lower_bound"]["target_hidden_order"]
        != manifest.TARGET_HIDDEN_ORDER
        or contract["execution_budget"]["allowed_reduced_model_promotions"] != 0
        or not contract["authority"]["preserve_f25aa_rejection"]
    ):
        raise RuntimeError("rank-capacity execution authorization changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"manifest source changed: {relative}")
    for name, expected in contract["parent_decisive_hashes"].items():
        if _sha(manifest.PARENT_DIRECTORY / name) != expected:
            raise RuntimeError(f"parent input changed: {name}")
    for name, expected in contract["fiber_decisive_hashes"].items():
        if _sha(manifest.FIBER_DIRECTORY / name) != expected:
            raise RuntimeError(f"fiber input changed: {name}")
    saved_paths = {
        "primary_generator": manifest.PRIMARY_GENERATOR_DIRECTORY / "descriptor_A.npz",
        "primary_output": manifest.PRIMARY_GENERATOR_DIRECTORY / "projection.npz",
        "heldout_generator_and_output": manifest.CROSS_ANCHOR_DIRECTORY / "heldout_generator.npz",
        "R32_projection": manifest.R32_DIRECTORY / "R32_projection_promotion.npz",
        "frequency_ladder": manifest.R32_DIRECTORY / "R32_transfer.npz",
    }
    for name, path in saved_paths.items():
        if _sha(path) != contract["saved_input_hashes"][name]:
            raise RuntimeError(f"saved input changed: {name}")
    if require_clean and not _tracked_tree_clean():
        raise RuntimeError("rank-capacity audit requires a clean tracked tree")
    for name, expected in THREAD_ENVIRONMENT.items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"thread environment changed: {name}")
    return {"summary": summary, "contract": contract, "hashes": hashes}


def _tail_relative_errors(
    singular_values: np.ndarray,
    dynamic_norms: np.ndarray,
    total_norms: np.ndarray,
    order: int,
) -> tuple[np.ndarray, np.ndarray]:
    if singular_values.ndim != 2:
        raise ValueError("singular values must be frequency by channel rank")
    if order < 0:
        raise ValueError("order must be nonnegative")
    if order >= singular_values.shape[1]:
        tails = np.zeros(singular_values.shape[0])
    else:
        tails = np.sqrt(np.sum(singular_values[:, order:] ** 2, axis=1))
    tiny = np.finfo(float).tiny
    return (
        tails / np.maximum(dynamic_norms, tiny),
        tails / np.maximum(total_norms, tiny),
    )


def _error_metrics(dynamic: np.ndarray, total: np.ndarray) -> dict:
    return {
        "maximum_normalized_dynamic_transfer_relative_error": float(np.max(dynamic)),
        "RMS_normalized_dynamic_transfer_relative_error": float(np.sqrt(np.mean(dynamic * dynamic))),
        "DC_normalized_dynamic_transfer_relative_error": float(dynamic[0]),
        "maximum_normalized_total_transfer_relative_error": float(np.max(total)),
        "RMS_normalized_total_transfer_relative_error": float(np.sqrt(np.mean(total * total))),
        "DC_normalized_total_transfer_relative_error": float(total[0]),
    }


def _block_pass(metrics: dict, gates: dict, fraction: float) -> bool:
    return bool(
        all(
            metrics[f"{prefix}_{name.removesuffix('_max')}"]
            <= fraction * maximum
            for prefix in ("training", "heldout")
            for name, maximum in gates.items()
        )
    )


def _spectral_payload(
    response: np.ndarray,
    direct: np.ndarray,
    row_slice: slice,
) -> dict[str, np.ndarray]:
    dynamic = response[:, row_slice] - direct[None, row_slice, :]
    singular_values = np.asarray(
        [np.linalg.svd(item, compute_uv=False) for item in dynamic]
    )
    return {
        "singular_values": singular_values,
        "dynamic_norms": np.linalg.norm(dynamic, axis=(1, 2)),
        "total_norms": np.linalg.norm(response[:, row_slice], axis=(1, 2)),
    }


def _candidate(
    spectra: dict,
    order: int,
    gates: dict,
    fraction: float,
) -> tuple[dict, dict[str, np.ndarray]]:
    blocks = {}
    arrays = {}
    for block in ("resolved_self_energy", "conservative_face_flux"):
        combined = {}
        for frequency_set in ("training", "heldout"):
            payload = spectra[frequency_set][block]
            dynamic, total = _tail_relative_errors(
                payload["singular_values"],
                payload["dynamic_norms"],
                payload["total_norms"],
                order,
            )
            combined.update({
                f"{frequency_set}_{name}": value
                for name, value in _error_metrics(dynamic, total).items()
            })
            arrays[f"{block}_{frequency_set}_dynamic_lower_bound"] = dynamic
            arrays[f"{block}_{frequency_set}_total_lower_bound"] = total
        blocks[block] = combined
    safety_passed = bool(
        all(_block_pass(blocks[block], gates[block], fraction) for block in blocks)
    )
    original_gate_passed = bool(
        all(_block_pass(blocks[block], gates[block], 1.0) for block in blocks)
    )
    ratios = []
    for block in blocks:
        for prefix in ("training", "heldout"):
            for name, maximum in gates[block].items():
                ratios.append(
                    blocks[block][f"{prefix}_{name.removesuffix('_max')}"]
                    / (fraction * maximum)
                )
    return {
        "hidden_order": order,
        "online_dimension": (
            manifest.PHYSICAL_DIMENSION
            + manifest.EXACT_NONSTABLE_DIMENSION
            + order
        ),
        "blocks": blocks,
        "safety_margin_passed": safety_passed,
        "original_gate_passed": original_gate_passed,
        "maximum_safety_gate_ratio": float(max(ratios)),
    }, arrays


def _minimum_order(
    spectra: dict,
    gates: dict,
    fraction: float,
) -> int | None:
    maximum_rank = max(
        payload["singular_values"].shape[1]
        for values in spectra.values()
        for payload in values.values()
    )
    for order in range(maximum_rank + 1):
        metrics, _ = _candidate(spectra, order, gates, fraction)
        if metrics["safety_margin_passed"]:
            return order
    return None


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
        writer = csv.DictWriter(handle, fieldnames=["case", "path", "bytes", "sha256", "scientific_status"], lineterminator="\n")
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
        raise RuntimeError("rank-capacity audit is already canonicalized")
    began = time.perf_counter()
    with np.load(manifest.PRIMARY_GENERATOR_DIRECTORY / "descriptor_A.npz", allow_pickle=False) as source:
        primary_generator = np.asarray(source["complete_fixed_Q_generator"], dtype=float)
    with np.load(manifest.PRIMARY_GENERATOR_DIRECTORY / "projection.npz", allow_pickle=False) as source:
        primary_output = np.asarray(source["output_map"], dtype=float)
    with np.load(manifest.CROSS_ANCHOR_DIRECTORY / "heldout_generator.npz", allow_pickle=False) as source:
        heldout_generator = np.asarray(source["complete_fixed_Q_generator"], dtype=float)
        heldout_output = np.asarray(source["output_map"], dtype=float)
    with np.load(manifest.R32_DIRECTORY / "R32_projection_promotion.npz", allow_pickle=False) as source:
        restriction = np.asarray(source["resolved_restriction"], dtype=float)
    with np.load(manifest.R32_DIRECTORY / "R32_transfer.npz", allow_pickle=False) as source:
        frequencies = np.asarray(source["angular_frequencies_per_second"], dtype=float)
    heldout_frequencies = memory_tools._heldout_frequencies(frequencies)
    with np.load(manifest.FIBER_DIRECTORY / "decisive_fibers.npz", allow_pickle=False) as source:
        fiber = {name: np.asarray(source[name]) for name in source.files}
    generators = {"primary": primary_generator, "heldout": heldout_generator}
    outputs = {"primary": primary_output, "heldout": heldout_output}
    gates = frozen["contract"]["binding_gates"]
    fraction = gates["capacity_safety_fraction_of_transfer_gate_max"]
    anchor_metrics = {}
    all_arrays = {
        "training_angular_frequencies_per_second": frequencies,
        "heldout_angular_frequencies_per_second": heldout_frequencies,
    }
    target_safety_passed = True
    target_original_passed = True
    numerical_passed = True
    for anchor in ("primary", "heldout"):
        system, base = square_tools._square_root_stable_system(
            generators[anchor],
            outputs[anchor],
            restriction,
            fiber[f"{anchor}_right_basis"],
            fiber[f"{anchor}_left_dual_transpose"],
            fiber[f"{anchor}_unstable_operator"],
        )
        forcing, observation, direct, _, _ = memory_tools._normalize_system(
            system["hidden_forcing"],
            system["combined_observation"],
            system["combined_direct"],
        )
        responses = {}
        maximum_residual = 0.0
        for label, values in (("training", frequencies), ("heldout", heldout_frequencies)):
            responses[label], residual = memory_tools._frequency_response(
                system["hidden_operator"], forcing, observation, direct, values
            )
            maximum_residual = max(maximum_residual, residual)
        spectra = {}
        for label in ("training", "heldout"):
            spectra[label] = {
                "resolved_self_energy": _spectral_payload(
                    responses[label], direct, slice(0, manifest.PHYSICAL_DIMENSION)
                ),
                "conservative_face_flux": _spectral_payload(
                    responses[label], direct, slice(manifest.PHYSICAL_DIMENSION, None)
                ),
            }
            for block, payload in spectra[label].items():
                for name, values in payload.items():
                    all_arrays[f"{anchor}_{label}_{block}_{name}"] = values
        candidates = []
        for order in manifest.HIDDEN_ORDERS:
            candidate, arrays = _candidate(spectra, order, gates, fraction)
            candidates.append(candidate)
            for name, values in arrays.items():
                all_arrays[f"{anchor}_R{order}_{name}"] = values
        target = next(
            item for item in candidates
            if item["hidden_order"] == manifest.TARGET_HIDDEN_ORDER
        )
        minimum_safety = _minimum_order(spectra, gates, fraction)
        minimum_original = _minimum_order(spectra, gates, 1.0)
        anchor_metrics[anchor] = {
            "base_square_root_metrics": base,
            "maximum_frequency_solve_relative_residual": maximum_residual,
            "minimum_pointwise_order_for_original_gates": minimum_original,
            "minimum_pointwise_order_for_safety_margin": minimum_safety,
            "candidate_metrics": candidates,
            "target": target,
        }
        target_safety_passed &= target["safety_margin_passed"]
        target_original_passed &= target["original_gate_passed"]
        numerical_passed &= bool(
            maximum_residual
            <= gates["maximum_frequency_solve_relative_residual_max"]
            and base["full_coordinate_reconstruction_relative_defect"] <= 5.0e-9
        )
    elapsed = float(time.perf_counter() - began)
    numerical_passed = bool(
        numerical_passed
        and np.isfinite(elapsed)
        and elapsed <= 3600.0 * frozen["contract"]["execution_budget"]["maximum_wall_hours"]
    )
    passed = bool(numerical_passed and target_safety_passed)
    if not numerical_passed:
        classification = NUMERICAL_FAIL_CLASSIFICATION
        authorized_next = None
    elif target_safety_passed:
        classification = PASS_CLASSIFICATION
        authorized_next = "definitions_only_direct_relative_resolvent_basis_manifest"
    elif not target_original_passed:
        classification = IMPOSSIBLE_CLASSIFICATION
        authorized_next = None
    else:
        classification = MARGINAL_CLASSIFICATION
        authorized_next = None
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _write_json(CANONICAL_DIRECTORY / "metrics.json", {
        "anchors": anchor_metrics,
        "target_hidden_order": manifest.TARGET_HIDDEN_ORDER,
        "capacity_safety_fraction": fraction,
        "target_safety_passed": target_safety_passed,
        "target_original_passed": target_original_passed,
        "numerical_passed": numerical_passed,
        "wall_seconds": elapsed,
    })
    np.savez_compressed(CANONICAL_DIRECTORY / "pointwise_rank_bounds.npz", **all_arrays)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": classification,
        "passed": passed,
        "numerical_passed": numerical_passed,
        "target_hidden_order": manifest.TARGET_HIDDEN_ORDER,
        "target_online_dimension": manifest.PHYSICAL_DIMENSION + manifest.EXACT_NONSTABLE_DIMENSION + manifest.TARGET_HIDDEN_ORDER,
        "target_safety_passed": target_safety_passed,
        "target_original_gate_passed": target_original_passed,
        "coherent_dynamic_realizability_certified": False,
        "structure_preserving_basis_certified": False,
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
        "parent_package_hashes": _checksums(manifest.PARENT_DIRECTORY),
        "fiber_package_hashes": _checksums(manifest.FIBER_DIRECTORY),
    })
    source_files = (
        THIS_RUNNER,
        THIS_TEST,
        manifest.THIS_RUNNER,
        manifest.THIS_TEST,
        square_tools.THIS_RUNNER,
        memory_tools.THIS_RUNNER,
    )
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
    mins = ", ".join(
        f"{anchor}: original {values['minimum_pointwise_order_for_original_gates']}, safety {values['minimum_pointwise_order_for_safety_margin']}"
        for anchor, values in anchor_metrics.items()
    )
    REPORT_PATH.write_text("\n".join((
        "# Hidden rank-capacity audit WP10c9d6c7c3b5c4f25ac",
        "",
        "## Classification",
        "",
        f"`{classification}`",
        "",
        "The exact pointwise Eckart-Young transfer tails were evaluated at both saved anchors without constructing or promoting a reduced dynamical model.",
        "",
        f"Minimum pointwise orders ({mins}). Hidden order 130 safety-margin pass: `{target_safety_passed}`.",
        "",
        "A pass means only that R320 is not ruled out by pointwise transfer rank. Coherent realization, stability-preserving basis selection, online integration, and predictive evolution remain uncertified.",
        "",
        f"Authorized next artifact: `{authorized_next}`.",
        "",
    )), encoding="utf-8")
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
