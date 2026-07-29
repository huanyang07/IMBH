#!/usr/bin/env python3
"""Run the operator-neutral WP10c9d6c5 truncation and phase audit.

The c4 prospective validation is frozen without amendment.  This package
uses a high-order continuum collocation action to resolve the initial
linearized DAE truncation by physical block and fixed physical boundary
band.  It separately measures whether the short-time conservative-export
error is a transport-time shift and propagates predeclared width/support
controls.  No physical or numerical operator is changed.
"""

from __future__ import annotations

import argparse
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
from scipy.interpolate import BSpline, make_interp_spline
from scipy.sparse.linalg import expm_multiply


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for search_path in (SRC, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import run_causal_inner_continuum_lift_wp10c9d6c3 as wp10c9d6c3
import run_causal_inner_prospective_uniform_validation_wp10c9d6c4 as wp10c9d6c4

from imri_qpe.layer3_minidisk_1d.causal_inner_continuum_truncation import (
    CONTINUUM_DAE_BLOCK_NAMES,
    build_causal_five_field_continuum_background,
    causal_five_field_discrete_dae_truncation,
    linearize_causal_five_field_continuum_reference,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_geometry import (
    kerr_schild_column_geometry,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c5"
ANALYZED_BASE_COMMIT = "c082f62f62f9c5c9f28e61c7f25f4d353a5f7a09"
ANALYZED_BASE_PARENT = "3d973aa28c68d242bd33c88efe2226e2e48eb281"
ANALYZED_BASE_TREE = "aaa11a689a57fb63e27f45effec610032a9aa224"
THIS_RUNNER = (
    "scripts/run_causal_inner_local_truncation_wp10c9d6c5.py"
)

MESHES = tuple(wp10c9d6c4.MESHES)
LABELS = tuple(wp10c9d6c4.LABELS)
REFERENCE_LABEL = wp10c9d6c4.REFERENCE_LABEL
FINE_LABELS = LABELS[1:]
CONSERVATIVE_FIELDS = np.asarray((0, 2, 3), dtype=int)
OBSERVABLE_NAMES = tuple(wp10c9d6c4.OBSERVABLE_NAMES)

PRIMARY_CONTINUUM_NODES = 769
SECONDARY_CONTINUUM_NODES = 513
CONTINUUM_COMPARISON_SAMPLES = 257
PROJECTION_ORDER = wp10c9d6c4.PRIMARY_PROJECTION_ORDER

# These diagnostic profiles are fixed before c5 propagation.
BOUNDARY_PROFILE_DEFINITIONS = {
    "boundary_band_outgoing_original": {
        "center_over_rg": 1.84,
        "log_width": 0.065,
        "amplitude": 0.015,
        "role": "frozen_c4_binding_profile",
    },
    "boundary_band_outgoing_wider": {
        "center_over_rg": 1.84,
        "log_width": 0.130,
        "amplitude": 0.015,
        "role": "same_center_width_control",
    },
    "boundary_band_outgoing_shifted": {
        "center_over_rg": 1.98,
        "log_width": 0.065,
        "amplitude": 0.015,
        "role": "same_width_support_control",
    },
    "boundary_band_outgoing_shifted_wider": {
        "center_over_rg": 1.98,
        "log_width": 0.130,
        "amplitude": 0.015,
        "role": "width_and_support_control",
    },
}
BOUNDARY_PROFILES = tuple(BOUNDARY_PROFILE_DEFINITIONS)
PASSING_C4_CONTROLS = (
    "heldout_mid_inner",
    "heldout_broad_outer_inner",
    "heldout_two_lobe_mixed",
)
TRUNCATION_PROFILES = PASSING_C4_CONTROLS + BOUNDARY_PROFILES
HISTORICAL_REPRESENTATIONS = (
    "historical_common_quintic",
    "historical_common_septic",
    "historical_common_one_sided_trace",
)
PROPAGATED_PROFILES = BOUNDARY_PROFILES + HISTORICAL_REPRESENTATIONS

# Common boundaries of the first three N64 cells.  They are discovered from
# the frozen grid and recorded rather than entered as rounded radii.
BOUNDARY_BAND_COARSE_EDGE_INDICES = (1, 2, 3)

MAXIMUM_CONTINUUM_LEDGER_DEFECT = 2.0e-9
MAXIMUM_DISCRETE_LEDGER_DEFECT = 2.0e-10
MAXIMUM_CONTINUUM_REFERENCE_RELATIVE_DIFFERENCE = 2.0e-4
MINIMUM_CLEAN_TRUNCATION_ORDER = 0.75
MINIMUM_PHASE_EXPLAINED_FRACTION = 0.80
MINIMUM_PHASE_SHIFT_ORDER = 0.75
MINIMUM_GROUP_TARGET_FRACTION = 0.70
MAXIMUM_GROUP_RESIDUAL_RATIO = 0.60
MINIMUM_GROUP_DIRECTION_COSINE = 0.90
MAXIMUM_HISTORICAL_REPRESENTATION_TO_FINE_RATIO = 0.10
MINIMUM_ERROR_COSINE = wp10c9d6c4.MINIMUM_ERROR_COSINE

PARENT_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_prospective_uniform_validation_wp10c9d6c4"
)
PARENT_CONFIG = PARENT_DIRECTORY / "config.json"
PARENT_SUMMARY = PARENT_DIRECTORY / "summary.json"
PARENT_ARRAYS = PARENT_DIRECTORY / "decisive_arrays.npz"
PARENT_PROVENANCE = PARENT_DIRECTORY / "provenance.json"

CANONICAL_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_local_truncation_wp10c9d6c5"
)
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
CANONICAL_MANIFEST = (
    ROOT / "results/manifests/canonical_artifacts.csv"
)
CANONICAL_SUMMARY = (
    ROOT / "results/manifests/canonical_summary.json"
)

IMPLEMENTATION_SOURCES = (
    THIS_RUNNER,
    "scripts/run_causal_inner_prospective_uniform_validation_wp10c9d6c4.py",
    "scripts/run_causal_inner_continuum_lift_wp10c9d6c3.py",
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_continuum_truncation.py",
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_monolithic_tangent.py",
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_radial_linear_tangent.py",
    "tests/test_causal_inner_local_truncation_wp10c9d6c5.py",
)

TRUNCATION_BLOCK_NAMES = (
    "mapped_temporal",
    "responsive_height_temporal",
    "mapped_storage_rate",
    "responsive_height_storage_rate",
    "inner_shared_face",
    "conservative_transport_remainder",
    "candidate_shear_principal",
    "candidate_height_principal",
    "candidate_local_stress_relaxation",
    "candidate_geometry",
    "candidate_cooling",
    "candidate_stream",
    "candidate_lower_height_work",
)

ATTRIBUTION_GROUPS = {
    "boundary": (
        "inner_shared_face",
        "conservative_transport_remainder",
    ),
    "mapped_storage": (
        "mapped_temporal",
        "mapped_storage_rate",
    ),
    "responsive_height_storage": (
        "responsive_height_temporal",
        "responsive_height_storage_rate",
    ),
    "principal_path": (
        "candidate_shear_principal",
        "candidate_height_principal",
    ),
    "local_stress_relaxation": (
        "candidate_local_stress_relaxation",
    ),
    "lower_sources": (
        "candidate_geometry",
        "candidate_cooling",
        "candidate_stream",
        "candidate_lower_height_work",
    ),
}


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


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def _refresh_sha256s(directory: Path) -> None:
    entries = []
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.name == "SHA256SUMS.txt":
            continue
        entries.append(f"{_sha256(path)}  {path.name}")
    (directory / "SHA256SUMS.txt").write_text(
        "\n".join(entries) + "\n",
        encoding="utf-8",
    )


def _refresh_canonical_catalog() -> None:
    """Rebuild the compact catalog for every committed canonical case."""
    rows: list[dict[str, str | int]] = []
    for case in sorted(CANONICAL_DIRECTORY.parent.iterdir()):
        if not case.is_dir():
            continue
        provenance_path = case / "provenance.json"
        if not provenance_path.is_file():
            continue
        provenance = json.loads(
            provenance_path.read_text(encoding="utf-8")
        )
        status = provenance.get(
            "scientific_status",
            provenance.get("numerical_status", "DIAGNOSTIC ONLY"),
        )
        for path in sorted(case.iterdir()):
            if not path.is_file():
                continue
            rows.append(
                {
                    "case": case.name,
                    "path": _relative(path),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                    "scientific_status": status,
                }
            )

    CANONICAL_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
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

    catalog_summary = json.loads(
        CANONICAL_SUMMARY.read_text(encoding="utf-8")
    )
    catalog_summary.update(
        {
            "case_count": len({str(row["case"]) for row in rows}),
            "file_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "all_payload_hashes_recorded": True,
            "latest_source_parent_commit": ANALYZED_BASE_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write_json(CANONICAL_SUMMARY, catalog_summary)


def _source_manifest() -> tuple[dict[str, str], str]:
    hashes = {
        path: _sha256(ROOT / path)
        for path in IMPLEMENTATION_SOURCES
        if (ROOT / path).exists()
    }
    digest = hashlib.sha256()
    for path, value in sorted(hashes.items()):
        digest.update(path.encode("utf-8"))
        digest.update(value.encode("ascii"))
    return hashes, digest.hexdigest()


def _git_value(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _validate_analyzed_git_identity() -> dict:
    resolved = _git_value("rev-parse", ANALYZED_BASE_COMMIT)
    parent = _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
    tree = _git_value("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
    if (
        resolved != ANALYZED_BASE_COMMIT
        or parent != ANALYZED_BASE_PARENT
        or tree != ANALYZED_BASE_TREE
    ):
        raise RuntimeError("WP10c9d6c5 analyzed git identity changed")
    return {
        "analyzed_base_commit": resolved,
        "analyzed_base_parent_commit": parent,
        "analyzed_base_tree_sha": tree,
    }


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: np.asarray(source[name]) for name in source.files}


def _load_parent() -> tuple[dict, dict[str, np.ndarray]]:
    summary = json.loads(PARENT_SUMMARY.read_text(encoding="utf-8"))
    if not (
        summary["work_package"] == "WP10c9d6c4"
        and summary["classification"]
        == "prospective_heldout_uniform_validation_failed"
        and summary["method_passed"]
        and summary["lift_uncertainty_passed"]
        and summary["smooth_profile_local_truncation_audit_authorized"]
        and not summary["direct_operator_redesign_authorized"]
        and not summary["reduced_slow_evolution_authorized"]
    ):
        raise RuntimeError("WP10c9d6c4 binding classification changed")
    return summary, _load_npz(PARENT_ARRAYS)


def _fixed_definition_hash() -> str:
    payload = json.dumps(
        {
            "boundary_profiles": BOUNDARY_PROFILE_DEFINITIONS,
            "continuum_nodes": (
                PRIMARY_CONTINUUM_NODES,
                SECONDARY_CONTINUUM_NODES,
            ),
            "band_indices": BOUNDARY_BAND_COARSE_EDGE_INDICES,
            "groups": ATTRIBUTION_GROUPS,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _weighted_cell_moment_matrix(
    grid,
    cell_count: int,
    polynomial_degree: int,
) -> np.ndarray:
    """Map a boundary polynomial to proper-measure first-cell averages."""

    count = int(cell_count)
    degree = int(polynomial_degree)
    nodes, weights = np.polynomial.legendre.leggauss(PROJECTION_ORDER)
    origin = float(np.log(grid.edges[0]))
    result = np.empty((count, degree + 1), dtype=float)
    for cell in range(count):
        lower = float(np.log(grid.edges[cell]))
        upper = float(np.log(grid.edges[cell + 1]))
        midpoint = 0.5 * (lower + upper)
        half_width = 0.5 * (upper - lower)
        log_radii = midpoint + half_width * nodes
        radii = np.exp(log_radii)
        raw = np.asarray(
            [
                half_width
                * weight
                * radius
                * kerr_schild_column_geometry(
                    float(radius),
                    grid.gravitational_radius,
                ).face_measure
                for radius, weight in zip(
                    radii,
                    weights,
                    strict=True,
                )
            ],
            dtype=float,
        )
        raw *= float(grid.cell_measures[cell]) / float(np.sum(raw))
        powers = np.column_stack(
            [
                (log_radii - origin) ** power
                for power in range(degree + 1)
            ]
        )
        result[cell] = raw @ powers / float(grid.cell_measures[cell])
    return result


def _one_sided_boundary_trace(grid, cell_averages: np.ndarray) -> np.ndarray:
    """Infer the inner trace from five one-sided finite-volume averages."""

    values = np.asarray(cell_averages, dtype=float)
    moment = _weighted_cell_moment_matrix(grid, 5, 3)
    coefficients = np.linalg.lstsq(
        moment,
        values[:5],
        rcond=None,
    )[0]
    return np.asarray(coefficients[0], dtype=float)


def _boundary_profile(
    definition: dict,
    radii: np.ndarray,
    *,
    gravitational_radius: float,
    outer_radius: float,
    outgoing_vector: np.ndarray,
) -> np.ndarray:
    values = np.asarray(radii, dtype=float)
    radii_over_rg = values / float(gravitational_radius)
    envelope = np.exp(
        -0.5
        * (
            np.log(
                radii_over_rg / float(definition["center_over_rg"])
            )
            / float(definition["log_width"])
        )
        ** 2
    )
    cutoff = wp10c9d6c3._smooth_cutoff(
        np.log(radii_over_rg),
        float(np.log(outer_radius / gravitational_radius)),
    )
    return (
        float(definition["amplitude"])
        * cutoff[:, None]
        * envelope[:, None]
        * np.asarray(outgoing_vector, dtype=float)[None, :]
    )


def _build_profiles_and_configurations(
    parent_arrays: dict[str, np.ndarray],
) -> tuple[dict, dict[str, object], dict[str, np.ndarray], dict]:
    """Replay c4 and add only the predeclared c5 diagnostic lifts."""

    configurations, decisive, c4_construction = (
        wp10c9d6c4._build_configurations(
            _load_npz(wp10c9d6c4.PARENT_ARRAYS)
        )
    )
    if not c4_construction["passed"]:
        raise RuntimeError("frozen c4 construction failed")
    reference = configurations[REFERENCE_LABEL]
    grid = reference["context"].grid
    background = wp10c9d6c3.SmoothCellAverageProfile(
        knots=np.asarray(
            decisive["continuum_background_knots"],
            dtype=float,
        ),
        coefficients=np.asarray(
            decisive["continuum_background_coefficients"],
            dtype=float,
        ),
        degree=wp10c9d6c3.PRIMARY_BACKGROUND_DEGREE,
        gravitational_radius=float(grid.gravitational_radius),
    )
    outgoing_vector = np.asarray(
        decisive["first_cell_outgoing_physical_vector"],
        dtype=float,
    )
    historical_target = np.asarray(
        decisive["historical_common_target_cell_averages"],
        dtype=float,
    )
    historical_profiles, historical_report = (
        wp10c9d6c4._historical_profile_fit(
            grid,
            historical_target,
        )
    )
    one_sided_trace = _one_sided_boundary_trace(
        grid,
        historical_target,
    )
    one_sided_profile, one_sided_fit = (
        wp10c9d6c3._fit_cell_average_profile(
            grid,
            historical_target,
            one_sided_trace,
            np.zeros(5, dtype=float),
            degree=int(
                wp10c9d6c4.HISTORICAL_PROFILE_DEFINITION[
                    "primary_degree"
                ]
            ),
            quadrature_order=PROJECTION_ORDER,
        )
    )
    evaluators: dict[str, object] = {}
    for name in PASSING_C4_CONTROLS:
        evaluators[name] = (
            lambda radii, profile_name=name: (
                wp10c9d6c4._analytic_profile(
                    profile_name,
                    radii,
                    gravitational_radius=grid.gravitational_radius,
                    field_scales=np.asarray(
                        decisive["continuum_perturbation_field_scales"],
                        dtype=float,
                    ),
                    outer_radius=float(grid.edges[-1]),
                    outgoing_vector=outgoing_vector,
                )
            )
        )
    for name, definition in BOUNDARY_PROFILE_DEFINITIONS.items():
        evaluators[name] = (
            lambda radii, declared=definition: _boundary_profile(
                declared,
                radii,
                gravitational_radius=grid.gravitational_radius,
                outer_radius=float(grid.edges[-1]),
                outgoing_vector=outgoing_vector,
            )
        )
    evaluators["historical_common_quintic"] = historical_profiles[
        int(
            wp10c9d6c4.HISTORICAL_PROFILE_DEFINITION["primary_degree"]
        )
    ].evaluate
    evaluators["historical_common_septic"] = historical_profiles[
        int(
            wp10c9d6c4.HISTORICAL_PROFILE_DEFINITION[
                "independent_degree"
            ]
        )
    ].evaluate
    evaluators["historical_common_one_sided_trace"] = (
        one_sided_profile.evaluate
    )

    maximum_original_lift_defect = 0.0
    for label in LABELS:
        configuration = configurations[label]
        local_grid = configuration["context"].grid
        columns = np.asarray(
            configuration["primitive_column_scales"],
            dtype=float,
        ).reshape(-1, 5)
        directions = dict(configuration["initial_directions"])
        physical_directions = dict(configuration["physical_directions"])
        for name in PROPAGATED_PROFILES:
            physical = wp10c9d6c3._project_callable_to_cells(
                local_grid,
                evaluators[name],
                quadrature_order=PROJECTION_ORDER,
            )
            directions[name] = (physical / columns).ravel()
            physical_directions[name] = physical
            decisive[f"{label}__{name}__physical_direction"] = physical
        frozen = np.asarray(
            configuration["physical_directions"][
                "heldout_first_cell_outgoing"
            ],
            dtype=float,
        )
        current = np.asarray(
            physical_directions["boundary_band_outgoing_original"],
            dtype=float,
        )
        maximum_original_lift_defect = max(
            maximum_original_lift_defect,
            wp10c9d6c3._relative_difference(frozen, current),
        )
        configurations[label] = {
            **configuration,
            "initial_directions": directions,
            "physical_directions": physical_directions,
        }

    original_boundary_amplitude = float(
        np.linalg.norm(
            evaluators["boundary_band_outgoing_original"](
                np.asarray([grid.edges[0]], dtype=float)
            )[0]
        )
        / max(
            np.linalg.norm(
                evaluators["boundary_band_outgoing_original"](
                    np.asarray(
                        [
                            BOUNDARY_PROFILE_DEFINITIONS[
                                "boundary_band_outgoing_original"
                            ]["center_over_rg"]
                            * grid.gravitational_radius
                        ],
                        dtype=float,
                    )
                )[0]
            ),
            np.finfo(float).tiny,
        )
    )
    decisive["historical_one_sided_inner_trace"] = one_sided_trace
    decisive["historical_first_cell_average_anchor"] = np.asarray(
        historical_report["inner_anchor"],
        dtype=float,
    )
    decisive["historical_one_sided_trace_fit_knots"] = (
        one_sided_profile.knots
    )
    decisive["historical_one_sided_trace_fit_coefficients"] = (
        one_sided_profile.coefficients
    )
    report = {
        "c4_construction_preserved": True,
        "c4_profile_definition_sha256": (
            c4_construction["profile_definition_sha256"]
        ),
        "c5_profile_definition_sha256": _fixed_definition_hash(),
        "maximum_original_profile_lift_defect": (
            maximum_original_lift_defect
        ),
        "boundary_band_original_amplitude_at_excision_relative_to_peak": (
            original_boundary_amplitude
        ),
        "historical_primary_fit": historical_report,
        "historical_one_sided_trace": one_sided_trace,
        "historical_first_cell_average_anchor": (
            historical_report["inner_anchor"]
        ),
        "historical_one_sided_fit": one_sided_fit,
        "passed": bool(
            maximum_original_lift_defect <= 1.0e-13
            and historical_report["passed"]
            and one_sided_fit["maximum_boundary_defect"] <= 1.0e-12
        ),
    }
    # The direct parent arrays are used later for frozen-history parity.
    decisive["parent_fixed_physical_observable_scales"] = np.asarray(
        parent_arrays["fixed_physical_observable_scales"],
        dtype=float,
    )
    return configurations, evaluators, decisive, report


def _batch_propagate(
    configurations: dict,
    tangents: dict,
    observable_maps: dict,
    decisive: dict[str, np.ndarray],
) -> tuple[dict, dict[str, float]]:
    """Propagate all c5 diagnostic profiles in one Krylov call per grid."""

    histories = {name: {} for name in PROPAGATED_PROFILES}
    restart_defects: dict[str, float] = {}
    for label in LABELS:
        print(f"WP10c9d6c5: batch propagate {label}", flush=True)
        configuration = configurations[label]
        tangent = tangents[label]
        times = np.asarray(configuration["times"], dtype=float)
        initial = np.column_stack(
            [
                np.asarray(
                    configuration["initial_directions"][name],
                    dtype=float,
                )
                for name in PROPAGATED_PROFILES
            ]
        )
        generator = np.asarray(tangent.scaled_generator_per_s, dtype=float)
        trace = float(np.trace(generator))
        states = np.asarray(
            expm_multiply(
                generator,
                initial,
                start=float(times[0]),
                stop=float(times[-1]),
                num=int(times.size),
                endpoint=True,
                traceA=trace,
            ),
            dtype=float,
        )
        half = np.asarray(
            expm_multiply(
                generator * (0.5 * float(times[-1])),
                initial,
                traceA=0.5 * float(times[-1]) * trace,
            ),
            dtype=float,
        )
        restarted = np.asarray(
            expm_multiply(
                generator * (0.5 * float(times[-1])),
                half,
                traceA=0.5 * float(times[-1]) * trace,
            ),
            dtype=float,
        )
        observable = np.asarray(observable_maps[label], dtype=float)
        signals = np.einsum("tnp,on->tpo", states, observable)
        for index, name in enumerate(PROPAGATED_PROFILES):
            profile_signals = signals[:, index, :]
            histories[name][label] = {
                "times": times,
                "signals": profile_signals,
                "final_scaled_state": states[-1, :, index],
            }
            endpoint_scale = max(
                float(np.linalg.norm(initial[:, index])),
                float(np.linalg.norm(states[-1, :, index])),
                np.finfo(float).tiny,
            )
            restart_defects[f"{name}__{label}"] = float(
                np.linalg.norm(
                    restarted[:, index] - states[-1, :, index]
                )
                / endpoint_scale
            )
            prefix = f"{name}__{label}__"
            decisive[prefix + "times"] = times
            decisive[prefix + "signals"] = profile_signals
            decisive[prefix + "cumulative"] = (
                wp10c9d6c3.wp10c9d6c._cumulative(
                    times,
                    profile_signals,
                )
            )
            decisive[prefix + "final_scaled_state"] = (
                states[-1, :, index]
            )
    return histories, restart_defects


def _frozen_history_parity(
    histories: dict,
    parent_arrays: dict[str, np.ndarray],
) -> dict:
    defects = {}
    for label in LABELS:
        current = histories["boundary_band_outgoing_original"][label][
            "signals"
        ]
        frozen = np.asarray(
            parent_arrays[
                "heldout_first_cell_outgoing"
                f"__primary__{label}__signals"
            ],
            dtype=float,
        )
        defects[label] = wp10c9d6c3._relative_difference(
            current,
            frozen,
        )
    return {
        "defects": defects,
        "maximum_defect": max(defects.values()),
        "passed": bool(max(defects.values()) <= 2.0e-12),
    }


def _continuum_observables(reference) -> np.ndarray:
    fields = CONSERVATIVE_FIELDS
    edges = np.asarray(
        [
            reference.background.radii[0],
            reference.background.radii[-1],
        ],
        dtype=float,
    )
    integrated = reference.integrate_blocks(edges)
    inner, outer = reference.evaluate_face_flux_jvp(edges)
    stationary = sum(
        (
            integrated[name][0]
            for name in CONTINUUM_DAE_BLOCK_NAMES[4:]
        ),
        start=np.zeros(5, dtype=float),
    )
    cooling = integrated["candidate_cooling"][0]
    height = integrated["candidate_lower_height_work"][0]
    return np.concatenate(
        (
            inner[fields],
            outer[fields],
            -stationary[fields],
            -cooling[fields[1:]],
            -height[fields[1:]],
        )
    )


def _continuum_reference_report(
    primary,
    secondary,
    physical_scales: np.ndarray,
) -> dict:
    radii = np.geomspace(
        float(primary.background.radii[0]),
        float(primary.background.radii[-1]),
        CONTINUUM_COMPARISON_SAMPLES,
    )
    primary_rate = primary.evaluate_rate(radii)
    secondary_rate = secondary.evaluate_rate(radii)
    rate_scale = max(
        float(np.linalg.norm(primary_rate)),
        float(np.linalg.norm(secondary_rate)),
        np.finfo(float).tiny,
    )
    rate_defect = float(
        np.linalg.norm(primary_rate - secondary_rate) / rate_scale
    )
    primary_export = _continuum_observables(primary)
    secondary_export = _continuum_observables(secondary)
    export_defect = float(
        np.max(
            np.abs(primary_export - secondary_export)
            / np.asarray(physical_scales, dtype=float)
        )
    )
    maximum_ledger = max(
        primary.maximum_pointwise_ledger_relative_defect,
        secondary.maximum_pointwise_ledger_relative_defect,
    )
    return {
        "rate_relative_difference": rate_defect,
        "export_fixed_physical_difference": export_defect,
        "maximum_pointwise_ledger_relative_defect": maximum_ledger,
        "passed": bool(
            rate_defect
            <= MAXIMUM_CONTINUUM_REFERENCE_RELATIVE_DIFFERENCE
            and export_defect
            <= MAXIMUM_CONTINUUM_REFERENCE_RELATIVE_DIFFERENCE
            and maximum_ledger <= MAXIMUM_CONTINUUM_LEDGER_DEFECT
        ),
    }


def _project_continuum_rate(grid, reference) -> np.ndarray:
    return wp10c9d6c3._project_callable_to_cells(
        grid,
        reference.evaluate_rate,
        quadrature_order=PROJECTION_ORDER,
    )


def _split_truncation_blocks(truncation) -> dict[str, np.ndarray]:
    blocks = {
        name: np.asarray(values, dtype=float)
        for name, values in truncation.block_rows.items()
        if name != "candidate_conservative_transport"
    }
    inner = np.zeros_like(truncation.total_rows)
    inner[0] = -(
        truncation.discrete_face_flux_jvp[0]
        - truncation.continuum_face_flux_jvp[0]
    )
    blocks["inner_shared_face"] = inner
    blocks["conservative_transport_remainder"] = (
        truncation.block_rows["candidate_conservative_transport"]
        - inner
    )
    return {
        name: blocks[name]
        for name in TRUNCATION_BLOCK_NAMES
    }


def _build_truncation_ledgers(
    configurations: dict,
    evaluators: dict[str, object],
    tangents: dict,
    physical_scales: np.ndarray,
    decisive: dict[str, np.ndarray],
) -> tuple[dict, dict, dict]:
    """Build continuum references and every cellwise truncation ledger."""

    finest = configurations[LABELS[-1]]
    background_profile = wp10c9d6c3.SmoothCellAverageProfile(
        knots=np.asarray(
            decisive["continuum_background_knots"],
            dtype=float,
        ),
        coefficients=np.asarray(
            decisive["continuum_background_coefficients"],
            dtype=float,
        ),
        degree=wp10c9d6c3.PRIMARY_BACKGROUND_DEGREE,
        gravitational_radius=float(
            finest["context"].grid.gravitational_radius
        ),
    )
    print("WP10c9d6c5: build primary continuum background", flush=True)
    primary_background = build_causal_five_field_continuum_background(
        finest["context"],
        background_profile.evaluate,
        node_count=PRIMARY_CONTINUUM_NODES,
    )
    print("WP10c9d6c5: build secondary continuum background", flush=True)
    secondary_background = build_causal_five_field_continuum_background(
        finest["context"],
        background_profile.evaluate,
        node_count=SECONDARY_CONTINUUM_NODES,
    )
    references = {}
    reference_reports = {}
    ledgers = {name: {} for name in TRUNCATION_PROFILES}
    maximum_discrete_ledger = 0.0
    maximum_continuum_ledger = 0.0
    maximum_truncation_ledger = 0.0
    for name in TRUNCATION_PROFILES:
        print(f"WP10c9d6c5: continuum action {name}", flush=True)
        primary = linearize_causal_five_field_continuum_reference(
            primary_background,
            evaluators[name],
        )
        secondary = linearize_causal_five_field_continuum_reference(
            secondary_background,
            evaluators[name],
        )
        references[name] = primary
        reference_reports[name] = _continuum_reference_report(
            primary,
            secondary,
            physical_scales,
        )
        decisive[f"{name}__continuum_observables"] = (
            _continuum_observables(primary)
        )
        decisive[f"{name}__continuum_rate_nodes"] = (
            primary.perturbation_rate_per_s
        )
        for label in LABELS:
            configuration = configurations[label]
            grid = configuration["context"].grid
            continuum_rows = primary.integrate_blocks(grid.edges)
            continuum_rate = _project_continuum_rate(grid, primary)
            continuum_faces = primary.evaluate_face_flux_jvp(grid.edges)
            if name in BOUNDARY_PROFILES:
                direction_name = name
            else:
                direction_name = name
            truncation = causal_five_field_discrete_dae_truncation(
                tangents[label],
                configuration["initial_directions"][direction_name],
                continuum_rate,
                continuum_rows,
                continuum_faces,
            )
            split = _split_truncation_blocks(truncation)
            ledgers[name][label] = {
                "truncation": truncation,
                "split_blocks": split,
            }
            prefix = f"{name}__{label}__"
            decisive[prefix + "total_truncation_rows"] = (
                truncation.total_rows
            )
            decisive[prefix + "mass_solved_scaled_rate_error"] = (
                truncation.mass_solved_scaled_rate_error
            )
            decisive[prefix + "discrete_face_flux_jvp"] = (
                truncation.discrete_face_flux_jvp
            )
            decisive[prefix + "continuum_face_flux_jvp"] = (
                truncation.continuum_face_flux_jvp
            )
            for block_name, values in split.items():
                decisive[
                    prefix + "truncation_block__" + block_name
                ] = values
            maximum_discrete_ledger = max(
                maximum_discrete_ledger,
                truncation.maximum_discrete_ledger_relative_defect,
            )
            maximum_continuum_ledger = max(
                maximum_continuum_ledger,
                truncation.maximum_continuum_ledger_relative_defect,
            )
            maximum_truncation_ledger = max(
                maximum_truncation_ledger,
                truncation.maximum_truncation_ledger_relative_defect,
            )
    ledger_report = {
        "maximum_discrete_ledger_relative_defect": (
            maximum_discrete_ledger
        ),
        "maximum_continuum_ledger_relative_defect": (
            maximum_continuum_ledger
        ),
        "maximum_truncation_ledger_relative_defect": (
            maximum_truncation_ledger
        ),
        "passed": bool(
            maximum_discrete_ledger <= MAXIMUM_DISCRETE_LEDGER_DEFECT
            and maximum_continuum_ledger
            <= MAXIMUM_CONTINUUM_LEDGER_DEFECT
            and maximum_truncation_ledger
            <= MAXIMUM_DISCRETE_LEDGER_DEFECT
        ),
    }
    return ledgers, reference_reports, ledger_report


def _band_endpoints(configurations: dict) -> tuple[float, ...]:
    grid = configurations[LABELS[0]]["context"].grid
    return tuple(
        float(grid.edges[index])
        for index in BOUNDARY_BAND_COARSE_EDGE_INDICES
    )


def _band_row_count(grid, endpoint: float) -> int:
    matches = np.flatnonzero(
        np.isclose(
            np.asarray(grid.edges, dtype=float),
            float(endpoint),
            rtol=2.0e-12,
            atol=0.0,
        )
    )
    if matches.size != 1:
        raise RuntimeError("fixed boundary band is not a common grid face")
    return int(matches[0])


def _scaled_conservative_vector(
    values: np.ndarray,
    physical_scales: np.ndarray,
) -> np.ndarray:
    return (
        np.asarray(values, dtype=float)[CONSERVATIVE_FIELDS]
        / np.asarray(physical_scales, dtype=float)[:3]
    )


def _safe_order(coarse: float, fine: float) -> float:
    if coarse <= np.finfo(float).tiny and fine <= np.finfo(float).tiny:
        return math.inf
    if coarse <= 0.0 or fine <= 0.0:
        return -math.inf
    return float(np.log2(coarse / fine))


def _safe_cosine(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=float).ravel()
    right = np.asarray(second, dtype=float).ravel()
    scale = float(np.linalg.norm(left) * np.linalg.norm(right))
    if scale <= np.finfo(float).tiny:
        return 1.0
    return float(np.dot(left, right) / scale)


def _band_attribution(
    ledgers: dict,
    configurations: dict,
    physical_scales: np.ndarray,
    field_scales: np.ndarray,
    decisive: dict[str, np.ndarray],
) -> tuple[dict, dict]:
    """Resolve truncation contraction and signed block attribution by band."""

    endpoints = _band_endpoints(configurations)
    report = {}
    stable_candidates: dict[str, dict] = {}
    for profile in TRUNCATION_PROFILES:
        profile_report = {}
        for band_index, endpoint in enumerate(endpoints, start=1):
            vectors = {}
            block_vectors = {}
            label_reports = {}
            for label in LABELS:
                grid = configurations[label]["context"].grid
                count = _band_row_count(grid, endpoint)
                split = ledgers[profile][label]["split_blocks"]
                total = np.sum(
                    ledgers[profile][label]["truncation"].total_rows[
                        :count
                    ],
                    axis=0,
                )
                vector = _scaled_conservative_vector(
                    total,
                    physical_scales,
                )
                vectors[label] = vector
                blocks = {
                    name: _scaled_conservative_vector(
                        np.sum(values[:count], axis=0),
                        physical_scales,
                    )
                    for name, values in split.items()
                }
                block_vectors[label] = blocks
                physical_direction = np.asarray(
                    configurations[label]["physical_directions"][profile],
                    dtype=float,
                )
                profile_scale = max(
                    float(
                        np.linalg.norm(
                            physical_direction
                            / np.asarray(field_scales, dtype=float)[
                                None, :
                            ]
                        )
                        / math.sqrt(physical_direction.shape[0])
                    ),
                    np.finfo(float).tiny,
                )
                target_norm_sq = max(
                    float(np.dot(vector, vector)),
                    np.finfo(float).tiny,
                )
                group_reports = {}
                for group_name, names in ATTRIBUTION_GROUPS.items():
                    group = sum(
                        (blocks[name] for name in names),
                        start=np.zeros(3, dtype=float),
                    )
                    alpha = float(np.dot(vector, group) / target_norm_sq)
                    residual = float(
                        np.linalg.norm(vector - group)
                        / max(
                            np.linalg.norm(vector),
                            np.finfo(float).tiny,
                        )
                    )
                    group_reports[group_name] = {
                        "target_aligned_fraction": alpha,
                        "fixed_coefficient_residual_ratio": residual,
                        "scaled_norm": float(np.linalg.norm(group)),
                        "profile_normalized_scaled_norm": float(
                            np.linalg.norm(group) / profile_scale
                        ),
                    }
                ordered = tuple(TRUNCATION_BLOCK_NAMES)
                matrix = np.asarray([blocks[name] for name in ordered])
                gram = matrix @ matrix.T
                label_reports[label] = {
                    "row_count": count,
                    "outer_radius_over_rg": float(
                        endpoint / grid.gravitational_radius
                    ),
                    "total_scaled_vector": vector,
                    "total_scaled_norm": float(np.linalg.norm(vector)),
                    "profile_scaled_rms_amplitude": profile_scale,
                    "block_names": ordered,
                    "gram_matrix": gram,
                    "groups": group_reports,
                }
                prefix = (
                    f"{profile}__band_{band_index}__{label}__"
                )
                decisive[prefix + "total_scaled_vector"] = vector
                decisive[prefix + "block_scaled_vectors"] = matrix
                decisive[prefix + "block_gram_matrix"] = gram

            norms = {
                label: float(np.linalg.norm(vectors[label]))
                for label in LABELS
            }
            pair_orders = {
                f"{LABELS[index]}_to_{LABELS[index + 1]}": (
                    _safe_order(
                        norms[LABELS[index]],
                        norms[LABELS[index + 1]],
                    )
                )
                for index in range(len(LABELS) - 1)
            }
            fine_direction_cosine = _safe_cosine(
                vectors[LABELS[-2]],
                vectors[LABELS[-1]],
            )
            group_stability = {}
            for group_name in ATTRIBUTION_GROUPS:
                medium = label_reports[LABELS[-2]]["groups"][group_name]
                fine = label_reports[LABELS[-1]]["groups"][group_name]
                medium_vector = sum(
                    (
                        block_vectors[LABELS[-2]][name]
                        for name in ATTRIBUTION_GROUPS[group_name]
                    ),
                    start=np.zeros(3, dtype=float),
                )
                fine_vector = sum(
                    (
                        block_vectors[LABELS[-1]][name]
                        for name in ATTRIBUTION_GROUPS[group_name]
                    ),
                    start=np.zeros(3, dtype=float),
                )
                cosine = _safe_cosine(medium_vector, fine_vector)
                stable = bool(
                    medium["target_aligned_fraction"]
                    >= MINIMUM_GROUP_TARGET_FRACTION
                    and fine["target_aligned_fraction"]
                    >= MINIMUM_GROUP_TARGET_FRACTION
                    and medium["fixed_coefficient_residual_ratio"]
                    <= MAXIMUM_GROUP_RESIDUAL_RATIO
                    and fine["fixed_coefficient_residual_ratio"]
                    <= MAXIMUM_GROUP_RESIDUAL_RATIO
                    and cosine >= MINIMUM_GROUP_DIRECTION_COSINE
                )
                group_stability[group_name] = {
                    "medium_fine_direction_cosine": cosine,
                    "stable": stable,
                }
                if (
                    profile == "boundary_band_outgoing_original"
                    and stable
                ):
                    stable_candidates.setdefault(
                        group_name,
                        {"bands": []},
                    )["bands"].append(band_index)
            fine_orders = (
                pair_orders[
                    f"{LABELS[-3]}_to_{LABELS[-2]}"
                ],
                pair_orders[
                    f"{LABELS[-2]}_to_{LABELS[-1]}"
                ],
            )
            profile_report[f"band_{band_index}"] = {
                "outer_radius_over_rg": float(
                    endpoint
                    / configurations[LABELS[0]][
                        "context"
                    ].grid.gravitational_radius
                ),
                "labels": label_reports,
                "total_norms": norms,
                "pair_orders": pair_orders,
                "minimum_fine_pair_order": min(fine_orders),
                "fine_direction_cosine": fine_direction_cosine,
                "cleanly_contracting": bool(
                    min(fine_orders) >= MINIMUM_CLEAN_TRUNCATION_ORDER
                ),
                "group_stability": group_stability,
            }
        report[profile] = profile_report

    for group_name, candidate in stable_candidates.items():
        candidate["persistent_two_bands"] = bool(
            any(
                second == first + 1
                for first, second in zip(
                    candidate["bands"][:-1],
                    candidate["bands"][1:],
                    strict=True,
                )
            )
        )
    original = report["boundary_band_outgoing_original"]
    local_clean = bool(
        all(
            item["cleanly_contracting"]
            for item in original.values()
        )
    )
    return {
        "band_outer_radii_over_rg": [
            float(
                endpoint
                / configurations[LABELS[0]][
                    "context"
                ].grid.gravitational_radius
            )
            for endpoint in endpoints
        ],
        "profile_reports": report,
        "original_profile_local_truncation_cleanly_contracts": local_clean,
        "stable_group_candidates": stable_candidates,
    }, stable_candidates


def _initial_export_map_report(
    ledgers: dict,
    configurations: dict,
    tangents: dict,
    observable_maps: dict,
    evaluators: dict[str, object],
    physical_scales: np.ndarray,
    decisive: dict[str, np.ndarray],
) -> dict:
    """Compare every t=0 discrete export with the continuum export map."""

    del evaluators
    report = {}
    for profile in TRUNCATION_PROFILES:
        continuum = np.asarray(
            decisive[f"{profile}__continuum_observables"],
            dtype=float,
        )
        errors = {}
        values = {}
        for label in LABELS:
            direction = np.asarray(
                configurations[label]["initial_directions"][profile],
                dtype=float,
            )
            discrete = np.asarray(
                observable_maps[label] @ direction,
                dtype=float,
            )
            error = (discrete - continuum) / physical_scales
            errors[label] = error
            values[label] = discrete
            decisive[f"{profile}__{label}__initial_export_error"] = (
                error
            )
            # Retain a direct face-specific parity target.  The prefix
            # reconstruction is not used to define this quantity.
            truncation = ledgers[profile][label]["truncation"]
            decisive[
                f"{profile}__{label}__inner_face_flux_error"
            ] = (
                truncation.discrete_face_flux_jvp[0]
                - truncation.continuum_face_flux_jvp[0]
            )
        norms = {
            label: float(np.linalg.norm(errors[label]))
            for label in LABELS
        }
        pair_orders = {
            f"{LABELS[index]}_to_{LABELS[index + 1]}": (
                _safe_order(
                    norms[LABELS[index]],
                    norms[LABELS[index + 1]],
                )
            )
            for index in range(len(LABELS) - 1)
        }
        report[profile] = {
            "continuum_observables": continuum,
            "discrete_observables": values,
            "fixed_physical_error_norms": norms,
            "pair_orders": pair_orders,
            "fine_error_direction_cosine": _safe_cosine(
                errors[LABELS[-2]],
                errors[LABELS[-1]],
            ),
            "minimum_fine_pair_order": min(
                pair_orders[
                    f"{LABELS[-3]}_to_{LABELS[-2]}"
                ],
                pair_orders[
                    f"{LABELS[-2]}_to_{LABELS[-1]}"
                ],
            ),
        }
    del tangents
    return report


def _time_weights(times: np.ndarray) -> np.ndarray:
    values = np.asarray(times, dtype=float)
    weights = np.zeros(values.size, dtype=float)
    intervals = np.diff(values)
    weights[:-1] += 0.5 * intervals
    weights[1:] += 0.5 * intervals
    return weights


def _weighted_inner(
    first: np.ndarray,
    second: np.ndarray,
    weights: np.ndarray,
) -> float:
    return float(
        np.sum(
            np.asarray(weights, dtype=float)[:, None]
            * np.asarray(first, dtype=float)
            * np.asarray(second, dtype=float)
        )
    )


def _event_times(times: np.ndarray, scaled_flux: np.ndarray) -> dict:
    magnitude = np.linalg.norm(np.asarray(scaled_flux, dtype=float), axis=1)
    peak_index = int(np.argmax(magnitude))
    weights = _time_weights(times)
    weighted = weights * magnitude
    total = float(np.sum(weighted))
    centroid = float(
        np.sum(weighted * times) / max(total, np.finfo(float).tiny)
    )
    cumulative = np.zeros_like(magnitude)
    cumulative[1:] = np.cumsum(
        0.5
        * np.diff(times)
        * (magnitude[1:] + magnitude[:-1])
    )
    half_index = int(
        np.searchsorted(
            cumulative,
            0.5 * float(cumulative[-1]),
            side="left",
        )
    )
    significant = np.flatnonzero(
        magnitude >= 1.0e-3 * max(float(np.max(magnitude)), 1.0e-300)
    )
    final_index = int(significant[-1]) if significant.size else 0
    return {
        "peak_time_s": float(times[peak_index]),
        "centroid_time_s": centroid,
        "half_exit_time_s": float(times[min(half_index, times.size - 1)]),
        "final_significant_flux_time_s": float(times[final_index]),
        "peak_index": peak_index,
    }


def _phase_fit(
    times: np.ndarray,
    coarse: np.ndarray,
    fine: np.ndarray,
) -> dict:
    weights = _time_weights(times)
    error = np.asarray(fine, dtype=float) - np.asarray(coarse, dtype=float)
    derivative = np.asarray(
        make_interp_spline(
            np.asarray(times, dtype=float),
            np.asarray(fine, dtype=float),
            k=3,
            axis=0,
        ).derivative()(times),
        dtype=float,
    )
    denominator = max(
        _weighted_inner(derivative, derivative, weights),
        np.finfo(float).tiny,
    )
    shift = _weighted_inner(error, derivative, weights) / denominator
    phase = shift * derivative
    residual = error - phase
    error_energy = max(
        _weighted_inner(error, error, weights),
        np.finfo(float).tiny,
    )
    residual_energy = _weighted_inner(residual, residual, weights)
    return {
        "time_shift_s": float(shift),
        "explained_energy_fraction": float(
            1.0 - residual_energy / error_energy
        ),
        "error": error,
        "phase_component": phase,
        "amplitude_residual": residual,
    }


def _error_cosine_with_mask(
    times: np.ndarray,
    coarse_medium: np.ndarray,
    medium_fine: np.ndarray,
    mask: np.ndarray,
) -> float:
    selected_times = np.asarray(times, dtype=float)[mask]
    if selected_times.size < 2:
        return 1.0
    weights = _time_weights(selected_times)
    first = np.asarray(coarse_medium, dtype=float)[mask]
    second = np.asarray(medium_fine, dtype=float)[mask]
    numerator = _weighted_inner(first, second, weights)
    denominator = math.sqrt(
        max(
            _weighted_inner(first, first, weights)
            * _weighted_inner(second, second, weights),
            np.finfo(float).tiny,
        )
    )
    return float(numerator / denominator)


def _phase_report(
    histories: dict,
    physical_scales: np.ndarray,
    profile_reports: dict,
) -> dict:
    """Separate transport-time and residual-amplitude refinement errors."""

    result = {}
    inner_scales = np.asarray(physical_scales[:3], dtype=float)
    for profile in BOUNDARY_PROFILES:
        fine_histories = {
            label: histories[profile][label]
            for label in FINE_LABELS
        }
        times = np.asarray(
            fine_histories[FINE_LABELS[0]]["times"],
            dtype=float,
        )
        fluxes = {
            label: (
                np.asarray(
                    fine_histories[label]["signals"],
                    dtype=float,
                )[:, :3]
                / inner_scales[None, :]
            )
            for label in FINE_LABELS
        }
        coarse_medium = _phase_fit(
            times,
            fluxes[FINE_LABELS[0]],
            fluxes[FINE_LABELS[1]],
        )
        medium_fine = _phase_fit(
            times,
            fluxes[FINE_LABELS[1]],
            fluxes[FINE_LABELS[2]],
        )
        event_reports = {
            label: _event_times(times, fluxes[label])
            for label in FINE_LABELS
        }
        event_orders = {}
        for event in (
            "peak_time_s",
            "centroid_time_s",
            "half_exit_time_s",
            "final_significant_flux_time_s",
        ):
            coarse_difference = abs(
                event_reports[FINE_LABELS[0]][event]
                - event_reports[FINE_LABELS[1]][event]
            )
            fine_difference = abs(
                event_reports[FINE_LABELS[1]][event]
                - event_reports[FINE_LABELS[2]][event]
            )
            event_orders[event] = _safe_order(
                coarse_difference,
                fine_difference,
            )
        shift_order = _safe_order(
            abs(coarse_medium["time_shift_s"]),
            abs(medium_fine["time_shift_s"]),
        )
        all_mask = np.ones(times.size, dtype=bool)
        without_initial = np.arange(times.size) > 0
        raw_cosine = _error_cosine_with_mask(
            times,
            coarse_medium["error"],
            medium_fine["error"],
            all_mask,
        )
        without_initial_cosine = _error_cosine_with_mask(
            times,
            coarse_medium["error"],
            medium_fine["error"],
            without_initial,
        )
        residual_cosine = _error_cosine_with_mask(
            times,
            coarse_medium["amplitude_residual"],
            medium_fine["amplitude_residual"],
            all_mask,
        )
        cumulative_passed = bool(
            profile_reports[profile]["historical"]["primary_fine"][
                "cumulative"
            ]["passed"]
        )
        phase_selected = bool(
            coarse_medium["explained_energy_fraction"]
            >= MINIMUM_PHASE_EXPLAINED_FRACTION
            and medium_fine["explained_energy_fraction"]
            >= MINIMUM_PHASE_EXPLAINED_FRACTION
            and shift_order >= MINIMUM_PHASE_SHIFT_ORDER
            and cumulative_passed
        )
        result[profile] = {
            "event_times": event_reports,
            "event_orders": event_orders,
            "coarse_medium": {
                key: value
                for key, value in coarse_medium.items()
                if key not in {"error", "phase_component", "amplitude_residual"}
            },
            "medium_fine": {
                key: value
                for key, value in medium_fine.items()
                if key not in {"error", "phase_component", "amplitude_residual"}
            },
            "phase_shift_order": shift_order,
            "weighted_error_cosine": raw_cosine,
            "weighted_error_cosine_excluding_t0": (
                without_initial_cosine
            ),
            "phase_removed_error_cosine": residual_cosine,
            "cumulative_export_passed": cumulative_passed,
            "phase_crossover_selected": phase_selected,
        }
    return result


def _historical_representation_report(
    histories: dict,
    physical_scales: np.ndarray,
    parent_arrays: dict[str, np.ndarray],
) -> dict:
    """Measure continuum-fit uncertainty against the binding fine error."""

    label = LABELS[-1]
    primary = np.asarray(
        histories["historical_common_quintic"][label]["signals"],
        dtype=float,
    )
    alternatives = {
        name: np.asarray(histories[name][label]["signals"], dtype=float)
        for name in HISTORICAL_REPRESENTATIONS[1:]
    }
    frozen_medium = np.asarray(
        parent_arrays[
            "historical_common_smooth_fit"
            f"__primary__{LABELS[-2]}__signals"
        ],
        dtype=float,
    )
    frozen_fine = np.asarray(
        parent_arrays[
            "historical_common_smooth_fit"
            f"__primary__{LABELS[-1]}__signals"
        ],
        dtype=float,
    )
    fine_spatial = float(
        np.max(
            np.abs(frozen_fine - frozen_medium)
            / physical_scales[None, :]
        )
    )
    comparisons = {}
    for name, values in alternatives.items():
        difference = float(
            np.max(
                np.abs(values - primary)
                / physical_scales[None, :]
            )
        )
        comparisons[name] = {
            "maximum_fixed_physical_difference": difference,
            "difference_to_fine_spatial_ratio": (
                difference / max(fine_spatial, np.finfo(float).tiny)
            ),
        }
    maximum_ratio = max(
        item["difference_to_fine_spatial_ratio"]
        for item in comparisons.values()
    )
    return {
        "fine_spatial_maximum_fixed_physical_difference": fine_spatial,
        "comparisons": comparisons,
        "maximum_representation_to_fine_spatial_ratio": maximum_ratio,
        "binding_attribution_eligible": bool(
            maximum_ratio
            <= MAXIMUM_HISTORICAL_REPRESENTATION_TO_FINE_RATIO
        ),
    }


def _mechanism_selection(
    attribution: dict,
    phase: dict,
    profile_reports: dict,
) -> dict:
    """Select at most one next hypothesis without calling association causal."""

    local_clean = bool(
        attribution[
            "original_profile_local_truncation_cleanly_contracts"
        ]
    )
    candidates = attribution["stable_group_candidates"]
    persistent = [
        name
        for name, report in candidates.items()
        if report["persistent_two_bands"]
    ]
    control_ratios = {}
    original_reports = attribution["profile_reports"][
        "boundary_band_outgoing_original"
    ]
    for group_name in persistent:
        ratios = []
        for band_name, original_band in original_reports.items():
            if not original_band["group_stability"][group_name]["stable"]:
                continue
            original_effect = original_band["labels"][LABELS[-1]][
                "groups"
            ][group_name]["profile_normalized_scaled_norm"]
            controls = [
                attribution["profile_reports"][profile][band_name][
                    "labels"
                ][LABELS[-1]]["groups"][group_name][
                    "profile_normalized_scaled_norm"
                ]
                for profile in PASSING_C4_CONTROLS
            ]
            ratios.append(
                original_effect
                / max(float(np.median(controls)), np.finfo(float).tiny)
            )
        control_ratios[group_name] = min(ratios) if ratios else 0.0
    selected_groups = [
        name
        for name in persistent
        if control_ratios.get(name, 0.0) >= 2.0
    ]
    original_phase = phase["boundary_band_outgoing_original"]
    phase_selected = bool(original_phase["phase_crossover_selected"])
    instantaneous_pass = {
        name: bool(
            profile_reports[name]["historical"]["primary_fine"][
                "instantaneous"
            ]["passed"]
        )
        for name in BOUNDARY_PROFILES
    }
    width_crossover = bool(
        not instantaneous_pass["boundary_band_outgoing_original"]
        and instantaneous_pass["boundary_band_outgoing_wider"]
        and not instantaneous_pass["boundary_band_outgoing_shifted"]
        and instantaneous_pass[
            "boundary_band_outgoing_shifted_wider"
        ]
    )

    if not local_clean and len(selected_groups) == 1:
        group = selected_groups[0]
        if group == "boundary":
            mechanism = "boundary_half_cell_hypothesis"
            authorized_next = "single_boundary_half_cell_intervention"
        elif group in {
            "mapped_storage",
            "responsive_height_storage",
        }:
            mechanism = "fixed_band_space_storage_hypothesis"
            authorized_next = "single_space_storage_intervention"
        elif group in {"principal_path", "local_stress_relaxation"}:
            mechanism = "principal_or_stress_hypothesis"
            authorized_next = "single_principal_source_audit"
        else:
            mechanism = "lower_source_hypothesis"
            authorized_next = "single_lower_source_audit"
    elif local_clean and width_crossover:
        mechanism = "narrow_profile_preasymptotic_width_crossover"
        authorized_next = "prospective_transport_packet_validation"
    elif local_clean and phase_selected:
        mechanism = "transport_phase_crossover_no_redesign"
        authorized_next = "prospective_transport_packet_validation"
    elif local_clean:
        mechanism = "clean_local_truncation_no_stable_phase_mechanism"
        authorized_next = "prospective_transport_packet_validation"
    else:
        mechanism = "no_stable_local_or_phase_mechanism"
        authorized_next = "reconsider_strict_direction_or_architecture"
    return {
        "local_truncation_cleanly_contracts": local_clean,
        "persistent_group_candidates": persistent,
        "control_separation_ratios": control_ratios,
        "evidence_selected_groups": selected_groups,
        "phase_crossover_selected": phase_selected,
        "instantaneous_width_support_passes": instantaneous_pass,
        "narrow_profile_width_crossover_selected": width_crossover,
        "selected_mechanism": mechanism,
        "authorized_next": authorized_next,
        "causality_claimed": False,
    }


def _environment() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }


def run() -> dict:
    started = time.perf_counter()
    identity = _validate_analyzed_git_identity()
    parent_summary, parent_arrays = _load_parent()
    configurations, evaluators, decisive, construction = (
        _build_profiles_and_configurations(parent_arrays)
    )
    if not construction["passed"]:
        raise RuntimeError("WP10c9d6c5 profile construction failed")

    tangents, observable_maps, method_reports, baselines = (
        wp10c9d6c3._build_tangents(configurations, decisive)
    )
    method_passed = bool(
        all(report["passed"] for report in method_reports.values())
    )
    if not method_passed:
        raise RuntimeError("WP10c9d6c5 inherited tangent method failed")

    histories, restart_defects = _batch_propagate(
        configurations,
        tangents,
        observable_maps,
        decisive,
    )
    frozen_parity = _frozen_history_parity(histories, parent_arrays)
    physical_scales = np.asarray(
        parent_arrays["fixed_physical_observable_scales"],
        dtype=float,
    )
    profile_reports = {}
    for name in BOUNDARY_PROFILES:
        report, arrays = wp10c9d6c3._profile_report(
            {
                "primary": histories[name],
                "projection_order_12": histories[name],
            },
            physical_scales,
        )
        profile_reports[name] = report
        for array_name, values in arrays.items():
            decisive[f"{name}__{array_name}"] = values

    ledgers, continuum_reports, ledger_report = (
        _build_truncation_ledgers(
            configurations,
            evaluators,
            tangents,
            physical_scales,
            decisive,
        )
    )
    field_scales = np.asarray(
        decisive["continuum_perturbation_field_scales"],
        dtype=float,
    )
    attribution, _stable_candidates = _band_attribution(
        ledgers,
        configurations,
        physical_scales,
        field_scales,
        decisive,
    )
    initial_exports = _initial_export_map_report(
        ledgers,
        configurations,
        tangents,
        observable_maps,
        evaluators,
        physical_scales,
        decisive,
    )
    phase = _phase_report(
        histories,
        physical_scales,
        profile_reports,
    )
    historical = _historical_representation_report(
        histories,
        physical_scales,
        parent_arrays,
    )
    mechanism = _mechanism_selection(
        attribution,
        phase,
        profile_reports,
    )

    maximum_restart = max(restart_defects.values())
    continuum_passed = bool(
        all(report["passed"] for report in continuum_reports.values())
    )
    audit_passed = bool(
        construction["passed"]
        and method_passed
        and frozen_parity["passed"]
        and ledger_report["passed"]
        and continuum_passed
        and maximum_restart <= wp10c9d6c4.MAXIMUM_RESTART_DEFECT
    )
    if not audit_passed:
        classification = "local_truncation_audit_method_unresolved"
        authorized_next = "none"
    else:
        selected = mechanism["selected_mechanism"]
        if selected == "boundary_half_cell_hypothesis":
            classification = (
                "boundary_half_cell_hypothesis_selected_not_causal"
            )
        elif selected == "fixed_band_space_storage_hypothesis":
            classification = (
                "fixed_band_space_storage_hypothesis_selected_not_causal"
            )
        elif selected in {
            "principal_or_stress_hypothesis",
            "lower_source_hypothesis",
        }:
            classification = selected + "_selected_not_causal"
        elif selected == "narrow_profile_preasymptotic_width_crossover":
            classification = (
                "narrow_profile_preasymptotic_width_crossover_no_redesign"
            )
        elif selected == "transport_phase_crossover_no_redesign":
            classification = (
                "boundary_packet_phase_crossover_no_redesign"
            )
        elif selected == (
            "clean_local_truncation_no_stable_phase_mechanism"
        ):
            classification = (
                "local_dae_truncation_convergent_strict_direction_unresolved"
            )
        else:
            classification = "no_stable_local_or_phase_mechanism"
        authorized_next = mechanism["authorized_next"]

    source_hashes, source_manifest = _source_manifest()
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DECISIVE_ARRAYS, **decisive)
    config = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "meshes": MESHES,
        "labels": LABELS,
        "passing_c4_controls": PASSING_C4_CONTROLS,
        "boundary_profile_definitions": BOUNDARY_PROFILE_DEFINITIONS,
        "historical_representations": HISTORICAL_REPRESENTATIONS,
        "profile_definition_sha256": _fixed_definition_hash(),
        "primary_continuum_nodes": PRIMARY_CONTINUUM_NODES,
        "secondary_continuum_nodes": SECONDARY_CONTINUUM_NODES,
        "projection_order": PROJECTION_ORDER,
        "boundary_band_coarse_edge_indices": (
            BOUNDARY_BAND_COARSE_EDGE_INDICES
        ),
        "truncation_block_names": TRUNCATION_BLOCK_NAMES,
        "attribution_groups": ATTRIBUTION_GROUPS,
        "gates": {
            "maximum_continuum_ledger_defect": (
                MAXIMUM_CONTINUUM_LEDGER_DEFECT
            ),
            "maximum_discrete_ledger_defect": (
                MAXIMUM_DISCRETE_LEDGER_DEFECT
            ),
            "maximum_continuum_reference_relative_difference": (
                MAXIMUM_CONTINUUM_REFERENCE_RELATIVE_DIFFERENCE
            ),
            "minimum_clean_truncation_order": (
                MINIMUM_CLEAN_TRUNCATION_ORDER
            ),
            "minimum_phase_explained_fraction": (
                MINIMUM_PHASE_EXPLAINED_FRACTION
            ),
            "minimum_phase_shift_order": MINIMUM_PHASE_SHIFT_ORDER,
            "minimum_group_target_fraction": (
                MINIMUM_GROUP_TARGET_FRACTION
            ),
            "maximum_group_residual_ratio": (
                MAXIMUM_GROUP_RESIDUAL_RATIO
            ),
            "minimum_group_direction_cosine": (
                MINIMUM_GROUP_DIRECTION_COSINE
            ),
            "maximum_historical_representation_to_fine_ratio": (
                MAXIMUM_HISTORICAL_REPRESENTATION_TO_FINE_RATIO
            ),
        },
        "operator_change": False,
        "production_defaults_changed": False,
    }
    _write_json(CONFIG_PATH, config)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        **identity,
        "classification": classification,
        "authorized_next": authorized_next,
        "passed": audit_passed,
        "audit_executed": True,
        "parent_wp10c9d6c4_classification_preserved": True,
        "parent_classification": parent_summary["classification"],
        "construction": construction,
        "method_reports": method_reports,
        "baseline_observables": baselines,
        "frozen_history_parity": frozen_parity,
        "restart_defects": restart_defects,
        "maximum_restart_defect": maximum_restart,
        "continuum_reference_reports": continuum_reports,
        "ledger_report": ledger_report,
        "boundary_profile_reports": profile_reports,
        "band_attribution": attribution,
        "initial_export_map_report": initial_exports,
        "phase_amplitude_report": phase,
        "historical_representation_report": historical,
        "mechanism_selection": mechanism,
        "historical_calibration_binding_attribution_eligible": (
            historical["binding_attribution_eligible"]
        ),
        "direct_operator_redesign_authorized": bool(
            authorized_next
            in {
                "single_boundary_half_cell_intervention",
                "single_space_storage_intervention",
            }
        ),
        "embedded_export_discrimination_authorized": False,
        "nonlinear_physical_trajectory_authorized": False,
        "production_operator_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "operator_changed": False,
        "decisive_arrays_path": _relative(DECISIVE_ARRAYS),
        "decisive_arrays_sha256": _sha256(DECISIVE_ARRAYS),
        "decisive_array_hashes": {
            name: _array_sha256(values)
            for name, values in decisive.items()
        },
        "implementation_source_hashes": source_hashes,
        "implementation_source_manifest_sha256": source_manifest,
        "environment": _environment(),
        "runtime_seconds": time.perf_counter() - started,
    }
    _write_json(SUMMARY_PATH, summary)
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "scientific_status": (
            "DIAGNOSTIC ONLY" if audit_passed else "REJECTED"
        ),
        "source_parent_commit": ANALYZED_BASE_COMMIT,
        **identity,
        "generation_command": (
            "PYTHONPATH=src:scripts python3 "
            "scripts/run_causal_inner_local_truncation_wp10c9d6c5.py"
        ),
        "profile_definition_sha256": _fixed_definition_hash(),
        "implementation_source_hashes": source_hashes,
        "implementation_source_manifest_sha256": source_manifest,
        "parent_canonical_hashes": {
            _relative(PARENT_CONFIG): _sha256(PARENT_CONFIG),
            _relative(PARENT_SUMMARY): _sha256(PARENT_SUMMARY),
            _relative(PARENT_ARRAYS): _sha256(PARENT_ARRAYS),
            _relative(PARENT_PROVENANCE): _sha256(PARENT_PROVENANCE),
        },
        "establishes": (
            "the initial continuum-versus-discrete linearized DAE "
            "truncation ledger, fixed-band block attribution, boundary "
            "packet phase/amplitude split, and prospective width/support "
            "controls for the unchanged c4 uniform operator"
        ),
        "does_not_establish": (
            "causality of an associated block, embedded convergence, "
            "nonlinear convergence, production eligibility, fixed-Q "
            "closure, or reduced slow-time evolution"
        ),
    }
    _write_json(PROVENANCE_PATH, provenance)
    _refresh_sha256s(CANONICAL_DIRECTORY)
    _refresh_canonical_catalog()
    print(
        f"WP10c9d6c5: classification={classification}",
        flush=True,
    )
    return summary


def refresh_metadata_only() -> dict:
    """Refresh provenance hashes without recomputing scientific arrays."""
    required = (
        CONFIG_PATH,
        DECISIVE_ARRAYS,
        PROVENANCE_PATH,
        SUMMARY_PATH,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "metadata refresh requires the completed canonical package: "
            + ", ".join(missing)
        )

    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    provenance = json.loads(
        PROVENANCE_PATH.read_text(encoding="utf-8")
    )
    source_hashes, source_manifest = _source_manifest()
    decisive = _load_npz(DECISIVE_ARRAYS)

    summary["decisive_arrays_sha256"] = _sha256(DECISIVE_ARRAYS)
    summary["decisive_array_hashes"] = {
        name: _array_sha256(values)
        for name, values in decisive.items()
    }
    summary["implementation_source_hashes"] = source_hashes
    summary["implementation_source_manifest_sha256"] = source_manifest
    summary["metadata_refresh_command"] = (
        "PYTHONPATH=src:scripts python3 "
        "scripts/run_causal_inner_local_truncation_wp10c9d6c5.py "
        "--refresh-metadata-only"
    )
    provenance["implementation_source_hashes"] = source_hashes
    provenance["implementation_source_manifest_sha256"] = source_manifest
    provenance["scientific_status"] = (
        "DIAGNOSTIC ONLY" if summary["passed"] else "REJECTED"
    )
    provenance["metadata_refresh_command"] = summary[
        "metadata_refresh_command"
    ]

    _write_json(SUMMARY_PATH, summary)
    _write_json(PROVENANCE_PATH, provenance)
    _refresh_sha256s(CANONICAL_DIRECTORY)
    _refresh_canonical_catalog()
    print(
        "WP10c9d6c5: refreshed canonical provenance metadata only",
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh-metadata-only",
        action="store_true",
        help=(
            "refresh source and decisive-array hashes in an already "
            "completed canonical package without rerunning the audit"
        ),
    )
    arguments = parser.parse_args()
    if arguments.refresh_metadata_only:
        refresh_metadata_only()
    else:
        run()


if __name__ == "__main__":
    main()
