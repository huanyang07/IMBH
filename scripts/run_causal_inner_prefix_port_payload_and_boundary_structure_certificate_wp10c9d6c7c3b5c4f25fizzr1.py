#!/usr/bin/env python3
"""Build the 913 native prefix ports and certify 11-field boundaries."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_equilibrium_column_thermodynamic_potential_implementation_wp10c9d6c7c3b5c4f25fizzc as witnesses  # noqa: E402
import run_causal_inner_prefix_anchor_batch_build_and_physical_boundary_lift_manifest_wp10c9d6c7c3b5c4f25fizzr as manifest  # noqa: E402
from imri_qpe.constants import C  # noqa: E402
from imri_qpe.layer3_minidisk_1d.causal_inner_bounded_ap_trajectory import source_nullity  # noqa: E402
from imri_qpe.layer3_minidisk_1d.causal_inner_entropy_characteristic_boundary import (  # noqa: E402
    audit_outward_entropy_characteristic_boundary,
    build_outward_entropy_characteristic_boundary,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_eleven_field_convex import (  # noqa: E402
    audit_full_shear_rest_frame,
    full_shear_rest_frame,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_full_port_atlas import (  # noqa: E402
    audit_full_port_atlas_anchor,
    build_full_port_atlas_anchor,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_physical_entropy_congruence import (  # noqa: E402
    audit_corrected_physical_port_atlas,
    audit_physical_entropy_congruence,
    build_corrected_physical_port_atlas,
    build_physical_entropy_congruence,
)


WORK_PACKAGE = manifest.AUTHORIZED_NEXT
PASS_CLASSIFICATION = (
    "prefix_913_port_payloads_and_eleven_field_boundary_structure_certified_"
    "outer_cycle_loading_missing"
)
FAIL_CLASSIFICATION = "prefix_port_payload_or_boundary_structure_failed"
AUTHORIZED_NEXT = manifest.PASS_NEXT
ARTIFACT = (
    "causal_inner_prefix_port_payload_and_boundary_structure_certificate_"
    "wp10c9d6c7c3b5c4f25fizzr1"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_PREFIX_PORT_PAYLOAD_AND_BOUNDARY_"
    "STRUCTURE_CERTIFICATE_WP10C9D6C7C3B5C4F25FIZZR1_2026-08-27.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_prefix_port_payload_and_boundary_structure_"
    "certificate_wp10c9d6c7c3b5c4f25fizzr1.py"
)
THIS_TEST = (
    "tests/test_causal_inner_prefix_port_payload_and_boundary_structure_"
    "certificate_wp10c9d6c7c3b5c4f25fizzr1.py"
)
BOUNDARY_SOURCE = (
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_entropy_characteristic_boundary.py"
)
BOUNDARY_TEST = "tests/test_causal_inner_entropy_characteristic_boundary.py"
PARENT_SHA256 = "d7f058bb8a84fe3805d521f93ade40e962948d0f1b1e34b5cccc680a7d1b60dc"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"


def _u():
    return manifest._u()


def _validate_parent(require_clean: bool = False):
    utility = _u()
    if utility._sha256(manifest.CANONICAL_DIRECTORY / "SHA256SUMS.txt") != PARENT_SHA256:
        raise RuntimeError("prefix port/boundary manifest changed")
    hashes = utility._validate_checksums(manifest.CANONICAL_DIRECTORY)
    summary = utility._read_json(manifest.CANONICAL_DIRECTORY / "summary.json")
    contract = utility._read_json(
        manifest.CANONICAL_DIRECTORY / "batch_and_boundary_contract.json"
    )
    if (
        not summary["passed"]
        or not summary["definitions_only"]
        or summary["candidate_anchor_count"] != 913
        or summary["authorized_next"] != WORK_PACKAGE
        or summary["complete_cycle_execution_authorized"]
        or contract["prefix_port_batch"]["slow_forcing_b_included"]
        or contract["boundary_lift"]["outer"]["cycle_wide_loading_complete"]
    ):
        raise RuntimeError("prefix port/boundary contract classification changed")
    if require_clean and utility._git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("prefix port certificate needs a clean tracked tree")
    return hashes, contract


def _physical_context():
    source = (
        witnesses.frozen_audit.parent.parent.parent.boundary_diagnostic.manifest
        .parent.engine.execution.source
    )
    return source._initial_inputs()["base"]["configuration"]["context"]


def _build_port(context, cell: int, chart5: np.ndarray):
    radius = float(context.grid.centers[cell])
    old, chart7 = witnesses.frozen_audit._chart7_at_equilibrium(
        context, radius, np.asarray(chart5, dtype=float)
    )
    chart7 = np.asarray(chart7, dtype=float)
    height = float(np.exp(chart7[5]))
    sigma = float(np.exp(chart7[0]))
    congruence = build_physical_entropy_congruence(
        old.geometry,
        proper_half_thickness=height,
        density=sigma / (2.0 * height),
        temperature=float(np.exp(chart7[3])),
        radial_velocity_over_c=float(chart7[1]),
        azimuthal_velocity_over_c=float(chart7[2]),
        primitive_step=3.0e-4,
    )
    sound = float(old.thermodynamics.sound_speed)
    alpha = float((old.closure.viscous_signal_speed_over_c * C / sound) ** 2)
    omega = float(
        np.sqrt(old.thermodynamics.integrated_pressure / (sigma * height**2))
    )
    anchor = build_full_port_atlas_anchor(
        sound_speed=congruence.sound_speed_over_c * C,
        temperature=float(np.exp(chart7[3])),
        proper_half_thickness=height,
        proper_vertical_frequency=omega,
        alpha=alpha,
        shear_relaxation_time=float(old.closure.relaxation_time),
        transport_speed_over_c=float(chart7[1]),
    )
    corrected = build_corrected_physical_port_atlas(anchor, congruence, old.geometry)
    frame = full_shear_rest_frame(
        old.geometry,
        radial_velocity_over_c=float(chart7[1]),
        azimuthal_velocity_over_c=float(chart7[2]),
        vertical_velocity_over_c=float(chart7[6]),
    )
    return {
        "radius": radius,
        "chart7": chart7,
        "height_over_radius": height / radius,
        "scattering_optical_depth": 0.5 * float(context.kappa) * sigma,
        "congruence": congruence,
        "congruence_audit": audit_physical_entropy_congruence(congruence),
        "anchor": anchor,
        "anchor_audit": audit_full_port_atlas_anchor(anchor),
        "corrected": corrected,
        "corrected_audit": audit_corrected_physical_port_atlas(
            corrected, anchor, congruence, old.geometry
        ),
        "frame_audit": audit_full_shear_rest_frame(
            frame, old_specific_stress=float(chart5[4])
        ),
    }


def _maximum_dataclass_value(values, name: str) -> float:
    return float(max(float(getattr(value, name)) for value in values))


def _certificate():
    began = time.perf_counter()
    _, contract = _validate_parent()
    q2 = manifest.parent
    with np.load(
        q2.CANONICAL_DIRECTORY / "lift_and_coverage_arrays.npz",
        allow_pickle=False,
    ) as payload:
        inputs = {name: np.array(payload[name], copy=True) for name in payload.files}
    cells = np.asarray(inputs["selected_cell_indices"], dtype=int)
    charts5 = np.asarray(inputs["selected_charts5"], dtype=float)
    context_began = time.perf_counter()
    context = _physical_context()
    context_wall = time.perf_counter() - context_began
    if len(context.grid.centers) != 112:
        raise RuntimeError("physical context is not the frozen 112-cell grid")

    build_began = time.perf_counter()
    ports = [_build_port(context, int(cell), chart) for cell, chart in zip(cells, charts5, strict=True)]
    build_wall = time.perf_counter() - build_began

    congruence_audits = [port["congruence_audit"] for port in ports]
    anchor_audits = [port["anchor_audit"] for port in ports]
    corrected_audits = [port["corrected_audit"] for port in ports]
    frame_audits = [port["frame_audit"] for port in ports]
    radial = np.asarray([port["corrected"].radial_matrix for port in ports])
    source = np.asarray([port["corrected"].source_matrix for port in ports])
    speeds = np.asarray([port["corrected"].coordinate_speeds_over_c for port in ports])
    nullities = np.asarray([source_nullity(value) for value in source], dtype=int)

    boundary_records = []
    for cell, normal in ((0, -1.0), (111, 1.0)):
        for position in np.flatnonzero(cells == cell):
            boundary = build_outward_entropy_characteristic_boundary(
                radial[position], outward_normal=normal
            )
            boundary_records.append(
                (cell, int(position), boundary, audit_outward_entropy_characteristic_boundary(boundary))
            )
    inner = [record for record in boundary_records if record[0] == 0]
    outer = [record for record in boundary_records if record[0] == 111]
    boundary_audits = [record[3] for record in boundary_records]

    exterior_chart5 = np.asarray(context.outer_boundary_frozen_exterior_chart, dtype=float)
    exterior_radius = float(context.grid.edges[-1])
    _exterior_old, exterior_chart7 = witnesses.frozen_audit._chart7_at_equilibrium(
        context, exterior_radius, exterior_chart5
    )
    exterior_lift11 = q2._lift_anchor_states(exterior_chart5[None, :])[0]

    frame_maximum = max(max(asdict(audit).values()) for audit in frame_audits)
    boundary_symmetry = _maximum_dataclass_value(
        boundary_audits, "outward_matrix_symmetry_defect"
    )
    boundary_projector = _maximum_dataclass_value(
        boundary_audits, "projector_idempotence_defect"
    )
    boundary_reconstruction = _maximum_dataclass_value(
        boundary_audits, "characteristic_reconstruction_defect"
    )
    boundary_penalty_minimum = min(
        audit.penalty_minimum_eigenvalue for audit in boundary_audits
    )
    maximum_speed = float(np.max(np.abs(speeds)))
    heights = np.asarray([port["height_over_radius"] for port in ports])
    optical_depths = np.asarray([port["scattering_optical_depth"] for port in ports])
    gates = contract["boundary_lift"]["gates"]
    passed = bool(
        len(ports) == 913
        and np.array_equal(np.bincount(cells, minlength=112), inputs["cell_anchor_counts"])
        and all(audit.passed for audit in congruence_audits)
        and all(audit.passed for audit in anchor_audits)
        and all(audit.passed for audit in corrected_audits)
        and all(audit.passed for audit in frame_audits)
        and np.min(nullities) == np.max(nullities) == 4
        and np.all(np.isfinite(radial))
        and np.all(np.isfinite(source))
        and np.all(heights > 0.0)
        and np.max(heights) < 1.0
        and np.min(optical_depths) > 1.0
        and build_wall <= 3600.0 * contract["prefix_port_batch"]["maximum_build_wall_hours"]
        and len(inner) == contract["boundary_lift"]["inner"]["expected_candidate_anchors"]
        and len(outer) == contract["boundary_lift"]["outer"]["expected_candidate_anchors"]
        and all(record[2].incoming_count == 0 for record in inner)
        and all(record[2].incoming_count == 11 for record in outer)
        and all(audit.passed for audit in boundary_audits)
        and boundary_symmetry <= gates["maximum_symmetry_defect"]
        and boundary_projector <= gates["maximum_projector_idempotence_defect"]
        and boundary_penalty_minimum >= gates["minimum_penalty_eigenvalue"]
        and boundary_reconstruction <= gates["maximum_characteristic_reconstruction_defect"]
        and maximum_speed <= gates["maximum_absolute_speed_over_c"]
        and exterior_chart5.shape == (5,)
        and np.all(np.isfinite(exterior_chart5))
    )

    metrics = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION,
        "passed": passed,
        "native_radial_cells": 112,
        "global_state_dimension": 1232,
        "candidate_anchor_count": len(ports),
        "passing_congruence_count": sum(audit.passed for audit in congruence_audits),
        "passing_abstract_port_count": sum(audit.passed for audit in anchor_audits),
        "passing_corrected_port_count": sum(audit.passed for audit in corrected_audits),
        "passing_full_shear_frame_count": sum(audit.passed for audit in frame_audits),
        "minimum_source_nullity": int(np.min(nullities)),
        "maximum_source_nullity": int(np.max(nullities)),
        "minimum_scaled_entropy_eigenvalue_ratio": float(
            min(audit.scaled_entropy_minimum_eigenvalue_ratio for audit in congruence_audits)
        ),
        "maximum_congruence_symmetry_relative_defect": _maximum_dataclass_value(
            congruence_audits, "whitened_symmetry_relative_defect"
        ),
        "maximum_valencia_spectrum_absolute_defect": _maximum_dataclass_value(
            congruence_audits, "valencia_spectrum_absolute_defect"
        ),
        "maximum_corrected_core_reconstruction_defect": _maximum_dataclass_value(
            corrected_audits, "zero_shear_core_reconstruction_defect"
        ),
        "maximum_full_shear_frame_defect": float(frame_maximum),
        "maximum_absolute_speed_over_c": maximum_speed,
        "minimum_height_over_radius": float(np.min(heights)),
        "maximum_height_over_radius": float(np.max(heights)),
        "minimum_scattering_optical_depth": float(np.min(optical_depths)),
        "maximum_scattering_optical_depth": float(np.max(optical_depths)),
        "context_initialization_wall_seconds": context_wall,
        "anchor_payload_build_wall_seconds": build_wall,
        "total_certificate_wall_seconds": time.perf_counter() - began,
        "new_truth_calls": 0,
        "slow_forcing_b_included": False,
        "inner_boundary_anchor_count": len(inner),
        "outer_boundary_anchor_count": len(outer),
        "inner_incoming_counts": sorted({record[2].incoming_count for record in inner}),
        "outer_incoming_counts": sorted({record[2].incoming_count for record in outer}),
        "maximum_boundary_symmetry_defect": boundary_symmetry,
        "maximum_boundary_projector_idempotence_defect": boundary_projector,
        "minimum_boundary_penalty_eigenvalue": float(boundary_penalty_minimum),
        "maximum_boundary_characteristic_reconstruction_defect": boundary_reconstruction,
        "all_boundary_audits_passed": all(audit.passed for audit in boundary_audits),
        "old_frozen_exterior_candidate_lifted": True,
        "outer_cycle_loading_complete": False,
        "prefix_port_payloads_built": passed,
        "eleven_field_boundary_structure_certified": passed,
        "cycle_wide_inputs_complete": False,
        "events_and_resets_certified": False,
        "complete_cycle_execution_authorized": False,
        "complete_cycle_steps": 0,
        "authorized_next": AUTHORIZED_NEXT if passed else None,
    }
    boundary_positions = np.asarray([record[1] for record in boundary_records], dtype=int)
    arrays = {
        "selected_cell_indices": cells,
        "selected_profile_indices": inputs["selected_profile_indices"],
        "selected_source_kinds": inputs["selected_source_kinds"],
        "selected_source_indices": inputs["selected_source_indices"],
        "selected_charts5": charts5,
        "selected_charts7": np.asarray([port["chart7"] for port in ports]),
        "selected_anchor_local_states11": inputs["selected_anchor_local_states11"],
        "selected_radii_cm": np.asarray([port["radius"] for port in ports]),
        "cell_cover_maximum_trust_fractions": inputs["cell_maximum_trust_fractions"][cells],
        "scaled_entropy_square_roots": np.asarray(
            [port["congruence"].scaled_entropy_square_root for port in ports]
        ),
        "scaled_entropy_inverse_square_roots": np.asarray(
            [port["congruence"].scaled_entropy_inverse_square_root for port in ports]
        ),
        "conserved_scales4": np.asarray(
            [port["congruence"].conserved_scales for port in ports]
        ),
        "core_orientations4x4": np.asarray(
            [port["corrected"].core_orientation for port in ports]
        ),
        "corrected_radial_matrices11x11": radial,
        "source_matrices11x11": source,
        "coordinate_speeds_over_c": speeds,
        "source_nullities": nullities,
        "height_over_radius": heights,
        "scattering_optical_depths": optical_depths,
        "boundary_cell_indices": np.asarray([record[0] for record in boundary_records]),
        "boundary_anchor_positions": boundary_positions,
        "boundary_outward_normals": np.asarray(
            [record[2].outward_normal for record in boundary_records]
        ),
        "boundary_characteristic_speeds": np.asarray(
            [record[2].characteristic_speeds for record in boundary_records]
        ),
        "boundary_incoming_projectors": np.asarray(
            [record[2].incoming_projector for record in boundary_records]
        ),
        "boundary_incoming_penalties": np.asarray(
            [record[2].incoming_penalty for record in boundary_records]
        ),
        "outer_frozen_exterior_candidate_chart5": exterior_chart5,
        "outer_frozen_exterior_candidate_chart7": np.asarray(exterior_chart7),
        "outer_frozen_exterior_candidate_anchor_local_state11": exterior_lift11,
        "outer_boundary_radius_cm": np.asarray(exterior_radius),
    }
    return metrics, arrays


def _update(summary):
    utility = _u()
    rows = list(csv.DictReader(CANONICAL_MANIFEST.open(newline="", encoding="utf-8")))
    rows = [row for row in rows if row.get("case") != ARTIFACT]
    status = "SUPPORTED" if summary["passed"] else "REJECTED"
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "case": ARTIFACT,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": str(path.stat().st_size),
                    "sha256": utility._sha256(path),
                    "scientific_status": status,
                }
            )
    with CANONICAL_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("case", "path", "bytes", "sha256", "scientific_status"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    catalog = utility._read_json(CANONICAL_SUMMARY)
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
            "latest_source_parent_commit": utility._git("rev-parse", "HEAD"),
            "latest_work_package": WORK_PACKAGE,
        }
    )
    utility._write_json(CANONICAL_SUMMARY, catalog)


def _canonicalize(metrics, arrays):
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("prefix port/boundary certificate exists")
    hashes, _ = _validate_parent(require_clean=True)
    utility = _u()
    CANONICAL_DIRECTORY.mkdir(parents=True)
    utility._write_json(CANONICAL_DIRECTORY / "port_and_boundary_metrics.json", metrics)
    np.savez_compressed(CANONICAL_DIRECTORY / "prefix_port_payloads.npz", **arrays)
    summary = {
        "schema_version": 1,
        "work_package": WORK_PACKAGE,
        "classification": metrics["classification"],
        "passed": metrics["passed"],
        "candidate_anchor_count": metrics["candidate_anchor_count"],
        "prefix_port_payloads_built": metrics["prefix_port_payloads_built"],
        "eleven_field_boundary_structure_certified": metrics[
            "eleven_field_boundary_structure_certified"
        ],
        "outer_cycle_loading_complete": False,
        "full_slow_forcing_complete": False,
        "cycle_wide_inputs_complete": False,
        "events_and_resets_certified": False,
        "complete_cycle_execution_authorized": False,
        "complete_cycle_steps": 0,
        "authorized_next": metrics["authorized_next"],
    }
    utility._write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    utility._write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "manifest_artifact": manifest.ARTIFACT,
            "manifest_checksum_manifest_sha256": PARENT_SHA256,
            "manifest_hashes": hashes,
            "prefix_cover_artifact": manifest.parent.ARTIFACT,
            "prefix_cover_checksum_manifest_sha256": utility._sha256(
                manifest.parent.CANONICAL_DIRECTORY / "SHA256SUMS.txt"
            ),
        },
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "# Native prefix-port payload and 11-field boundary certificate\n\n"
        f"Classification: `{metrics['classification']}`.\n\n"
        f"All `{metrics['candidate_anchor_count']}` native prefix anchors pass the physical "
        "entropy congruence, abstract full-port, corrected physical-port, full-STF frame, "
        "causality, and dissipative-source audits. The batch build took "
        f"`{metrics['anchor_payload_build_wall_seconds']:.3f}` s after one "
        f"`{metrics['context_initialization_wall_seconds']:.3f}` s context initialization. "
        f"The maximum speed is `{metrics['maximum_absolute_speed_over_c']:.9f} c`.\n\n"
        "In outward-normal entropy variables, all 17 inner-edge anchors have zero incoming "
        "modes and therefore preserve pure excision; both outer-edge anchors have eleven "
        "incoming modes and require prescribed physical loading. The incoming projector and "
        "positive-semidefinite penalty pass reconstruction, symmetry, and idempotence gates.\n\n"
        "The old five-field frozen exterior chart is preserved only as a lifted prefix "
        "candidate. It does not provide cycle-wide outer loading. Slow forcing, events, and "
        "resets remain absent, and no complete-cycle step is authorized.\n",
        encoding="utf-8",
    )
    sources = (THIS_RUNNER, THIS_TEST, BOUNDARY_SOURCE, BOUNDARY_TEST, REPORT_RELATIVE)
    utility._write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "implementation_commit": utility._git("rev-parse", "HEAD"),
            "source_hashes": {source: utility._sha256(ROOT / source) for source in sources},
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "thread_environment": {
                name: os.environ.get(name, "")
                for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
            },
        },
    )
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{utility._sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names
        ),
        encoding="utf-8",
    )
    _update(summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    arguments = parser.parse_args()
    if not arguments.run:
        parser.error("choose --run")
    metrics, arrays = _certificate()
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    return 0 if _canonicalize(metrics, arrays)["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
