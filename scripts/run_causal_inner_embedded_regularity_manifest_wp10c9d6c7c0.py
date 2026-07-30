#!/usr/bin/env python3
"""Freeze prospective endpoint/interface regularity controls.

WP10c9d6c7b rejected the complete embedded profile class because the C2
zero-extended ``sin^3`` shear profiles failed only the instantaneous
refinement-error-direction gate at the coupling face.  This definitions-only
package freezes two independent controls before any new propagation:

* C3 ``sin^4`` shear endpoints remain active at the coupling surface;
* C2 ``sin^3`` shear profiles end at parent face 45, leaving an exact
  three-parent-cell zero buffer to coupling face 48.

The existing C2 ``sin^3`` failures and C4 ``sin^5`` passes remain immutable
historical controls.  This script changes no operator and propagates no state.
"""

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

import run_causal_inner_continuum_lift_wp10c9d6c3 as c3
import run_causal_inner_embedded_manifest_wp10c9d6c7a as c7a
import run_causal_inner_frozen_hardening_wp10c9d5a as wp10c9d5a
import run_causal_inner_integral_conditioning_validation_wp10c9d6c6e1 as c6e1
import run_causal_inner_windowed_contract_wp10c9d6c6a2 as c6a2

from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    make_kerr_schild_column_grid_from_edges,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_embedded_patch import (  # noqa: E402
    make_causal_embedded_patch_layout,
    restrict_causal_embedded_patch_cell_averages,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_packet_manifest import (  # noqa: E402
    causal_array_sha256,
    causal_canonical_json_sha256,
    causal_characteristic_purity,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_packet_resolution import (  # noqa: E402
    causal_packet_spectrum,
)
from imri_qpe.layer3_minidisk_1d.causal_inner_radial_linear_tangent import (
    _frozen_quadratic_reconstruction_weights,
)  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c0"
ANALYZED_BASE_COMMIT = "fad76852220d7c304fff9016ff99ada64d404eff"
ANALYZED_BASE_PARENT = "82c3a9e5a326fedeccfafd8e8a4a9704935c64a3"
ANALYZED_BASE_TREE = "fa62d1f6e32446b0c7d39b98b1ff6505bcbf64fe"
THIS_RUNNER = (
    "scripts/"
    "run_causal_inner_embedded_regularity_manifest_wp10c9d6c7c0.py"
)

PRIMARY_PROJECTION_ORDER = 24
SECONDARY_PROJECTION_ORDER = 12
SUPPORT_BUFFER_PARENT_FACE = 45
COUPLING_PARENT_FACE = 48
MINIMUM_ZERO_BUFFER_PARENT_CELLS = 3
MINIMUM_ACTIVE_COUPLING_TRACE_FRACTION = 1.0e-10
MAXIMUM_BUFFERED_COUPLING_TRACE_FRACTION = 1.0e-15
MAXIMUM_RECONSTRUCTION_WEIGHT_DEFECT = 1.0e-12
MAXIMUM_RESTRICTION_DEFECT = 2.0e-12
MAXIMUM_EXTERIOR_NORM = 0.0

UNIFORM_LABELS = tuple(c6e1.LABELS)
EMBEDDED_LABELS = tuple(
    c7a.LAYOUTS[ratio] for ratio in c7a.REFINEMENT_RATIOS
)
REFINEMENT_RATIOS = tuple(c7a.REFINEMENT_RATIOS)

PROFILE_DEFINITIONS = {
    "p4__inward_shear": {
        "family": "inward_shear",
        "window_power": 4,
        "support_upper_parent_face": COUPLING_PARENT_FACE,
        "zero_extension_regularity": "C3",
        "role": "prospective_active_C3_endpoint",
        "coupling_trace_expectation": "active",
    },
    "p4__outward_shear": {
        "family": "outward_shear",
        "window_power": 4,
        "support_upper_parent_face": COUPLING_PARENT_FACE,
        "zero_extension_regularity": "C3",
        "role": "prospective_active_C3_endpoint",
        "coupling_trace_expectation": "active",
    },
    "p3_buffer45__inward_shear": {
        "family": "inward_shear",
        "window_power": 3,
        "support_upper_parent_face": SUPPORT_BUFFER_PARENT_FACE,
        "zero_extension_regularity": "C2",
        "role": "prospective_C2_exact_zero_buffer",
        "coupling_trace_expectation": "inactive",
    },
    "p3_buffer45__outward_shear": {
        "family": "outward_shear",
        "window_power": 3,
        "support_upper_parent_face": SUPPORT_BUFFER_PARENT_FACE,
        "zero_extension_regularity": "C2",
        "role": "prospective_C2_exact_zero_buffer",
        "coupling_trace_expectation": "inactive",
    },
}
PROFILE_NAMES = tuple(PROFILE_DEFINITIONS)

C7A_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_embedded_manifest_wp10c9d6c7a"
)
C7A_MANIFEST = C7A_DIRECTORY / "embedded_manifest.json"
C7A_SUMMARY = C7A_DIRECTORY / "summary.json"
C7A_ARRAYS = C7A_DIRECTORY / "decisive_arrays.npz"

C7B_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_embedded_validation_wp10c9d6c7b"
)
C7B_SUMMARY = C7B_DIRECTORY / "summary.json"
C7B_ARRAYS = C7B_DIRECTORY / "decisive_arrays.npz"

E1_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_integral_conditioning_validation_wp10c9d6c6e1"
)
E1_CONFIG = E1_DIRECTORY / "config.json"

C0E_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_cross_grid_analytic_tangent_wp10c9d5c0e"
)
C0E_CONTEXTS = C0E_DIRECTORY / "replay_contexts.json"
C0E_INPUTS = C0E_DIRECTORY / "replay_inputs.npz"

CANONICAL_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_embedded_regularity_manifest_wp10c9d6c7c0"
)
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
MANIFEST_PATH = CANONICAL_DIRECTORY / "regularity_manifest.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

IMPLEMENTATION_SOURCES = (
    THIS_RUNNER,
    "scripts/run_causal_inner_embedded_manifest_wp10c9d6c7a.py",
    "scripts/run_causal_inner_embedded_validation_wp10c9d6c7b.py",
    "scripts/run_causal_inner_windowed_contract_wp10c9d6c6a2.py",
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_embedded_patch.py",
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_packet_manifest.py",
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_packet_resolution.py",
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_radial_linear_tangent.py",
    "tests/"
    "test_causal_inner_embedded_regularity_manifest_wp10c9d6c7c0.py",
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


def _write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(
            _plain(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        raise RuntimeError("WP10c9d6c7c0 analyzed git identity changed")
    return {
        "analyzed_base_commit": resolved,
        "analyzed_base_parent": parent,
        "analyzed_base_tree": tree,
    }


def _source_manifest() -> tuple[dict[str, str], str]:
    hashes = {
        path: _sha256(ROOT / path)
        for path in IMPLEMENTATION_SOURCES
        if (ROOT / path).is_file()
    }
    digest = hashlib.sha256()
    for path, value in sorted(hashes.items()):
        digest.update(path.encode("utf-8"))
        digest.update(value.encode("ascii"))
    return hashes, digest.hexdigest()


def _array_hashes(arrays: dict[str, np.ndarray]) -> dict[str, str]:
    return {
        name: causal_array_sha256(values)
        for name, values in sorted(arrays.items())
    }


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {
            name: np.array(source[name], copy=True)
            for name in source.files
        }


def _relative_defect(
    values: np.ndarray,
    reference: np.ndarray,
) -> float:
    left = np.asarray(values, dtype=float)
    right = np.asarray(reference, dtype=float)
    scale = max(
        float(np.linalg.norm(left)),
        float(np.linalg.norm(right)),
        np.finfo(float).tiny,
    )
    return float(np.linalg.norm(left - right) / scale)


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
            "case_count": len({str(row["case"]) for row in rows}),
            "file_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "all_payload_hashes_recorded": True,
            "latest_source_parent_commit": ANALYZED_BASE_COMMIT,
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write_json(CANONICAL_SUMMARY, summary)


def _parent_hashes() -> dict[str, str]:
    paths = (
        C7A_MANIFEST,
        C7A_SUMMARY,
        C7A_ARRAYS,
        C7B_SUMMARY,
        C7B_ARRAYS,
        E1_CONFIG,
        C0E_CONTEXTS,
        C0E_INPUTS,
    )
    return {
        str(path.relative_to(ROOT)): _sha256(path)
        for path in paths
    }


def _profile_variants() -> list[dict]:
    variants = []
    for profile_name, definition in PROFILE_DEFINITIONS.items():
        for amplitude in (0.5, 1.0):
            for sign, sign_value in (("minus", -1), ("plus", 1)):
                variants.append(
                    {
                        "profile_id": (
                            f"{profile_name}::a{amplitude:.2f}::{sign}"
                        ),
                        "base_profile": profile_name,
                        "amplitude_factor": amplitude,
                        "sign": sign_value,
                        "binding": True,
                        "role": definition["role"],
                    }
                )
    return variants


def _support_endpoint_fraction(
    values: np.ndarray,
    field_scales: np.ndarray,
    support_face: int,
) -> float:
    normalized = np.asarray(values, dtype=float) / field_scales[None, :]
    norms = np.linalg.norm(normalized, axis=1)
    peak = max(float(np.max(norms)), np.finfo(float).tiny)
    indices = (0, int(support_face) - 1)
    return float(max(float(norms[index]) for index in indices) / peak)


def _build() -> tuple[dict, dict[str, np.ndarray]]:
    c7a_manifest = json.loads(C7A_MANIFEST.read_text(encoding="utf-8"))
    c7a_summary = json.loads(C7A_SUMMARY.read_text(encoding="utf-8"))
    c7b_summary = json.loads(C7B_SUMMARY.read_text(encoding="utf-8"))
    eligibility = json.loads(
        E1_CONFIG.read_text(encoding="utf-8")
    )["eligibility_contract"]
    replay_contexts = json.loads(
        C0E_CONTEXTS.read_text(encoding="utf-8")
    )
    replay_arrays = _load_npz(C0E_INPUTS)
    c7a_arrays = _load_npz(C7A_ARRAYS)

    if (
        c7a_summary["classification"]
        != "embedded_layout_and_profile_manifest_frozen_"
        "propagation_authorized"
        or c7a_manifest["manifest_sha256"]
        != c7a_summary["manifest_sha256"]
    ):
        raise RuntimeError("WP10c9d6c7a manifest identity changed")
    if (
        c7b_summary["classification"]
        != "prospective_embedded_profile_validation_failed"
        or c7b_summary["passed"]
    ):
        raise RuntimeError("WP10c9d6c7b rejection changed")

    configurations, construction_arrays, construction_report = (
        c3._build_continuum_configurations()
    )
    interpolator, characteristic_report, characteristic_arrays = (
        c6a2._build_characteristic_interpolator(
            configurations,
            construction_arrays,
        )
    )
    if not characteristic_report["passed"]:
        raise RuntimeError("tracked characteristic field is not certified")

    field_scales = np.asarray(
        construction_arrays["continuum_perturbation_field_scales"],
        dtype=float,
    )
    inherited_field_scales = np.asarray(
        c7a_arrays["field_scales"],
        dtype=float,
    )
    field_scale_defect = _relative_defect(
        field_scales,
        inherited_field_scales,
    )
    if field_scale_defect > 2.0e-12:
        raise RuntimeError("frozen physical field scales changed")

    parent_edges = np.asarray(
        c7a_arrays["parent_grid_edges"],
        dtype=float,
    )
    parent_grid = make_kerr_schild_column_grid_from_edges(
        parent_edges,
        float(
            configurations[
                "uniform_N128"
            ]["context"].grid.gravitational_radius
        ),
    )
    lower_radius = float(parent_edges[0])
    coupling_radius = float(parent_edges[COUPLING_PARENT_FACE])
    buffer_radius = float(parent_edges[SUPPORT_BUFFER_PARENT_FACE])
    evaluators = {}
    for profile_name, definition in PROFILE_DEFINITIONS.items():
        support_face = int(definition["support_upper_parent_face"])
        upper_radius = (
            coupling_radius
            if support_face == COUPLING_PARENT_FACE
            else buffer_radius
        )
        evaluators[profile_name] = c6a2._probe_evaluator(
            {
                "window_power": int(definition["window_power"]),
                "family": str(definition["family"]),
                "mixed_coefficients": None,
            },
            interpolator,
            lower_radius=lower_radius,
            upper_radius=upper_radius,
        )

    arrays: dict[str, np.ndarray] = {
        "field_scales": field_scales,
        "parent_grid_edges": parent_edges,
        "support_parent_face_indices": np.asarray(
            [
                PROFILE_DEFINITIONS[name][
                    "support_upper_parent_face"
                ]
                for name in PROFILE_NAMES
            ],
            dtype=np.int64,
        ),
        "characteristic_field_radii": np.asarray(
            characteristic_arrays["characteristic_field_radii"],
            dtype=float,
        ),
        "characteristic_field_physical_vectors": np.asarray(
            characteristic_arrays[
                "characteristic_field_physical_vectors"
            ],
            dtype=float,
        ),
    }
    uniform_values: dict[
        tuple[str, str, str], np.ndarray
    ] = {}
    projection_reports: dict[str, dict] = {
        name: {} for name in PROFILE_NAMES
    }
    maximum_projection_defect = 0.0
    for label in UNIFORM_LABELS:
        grid = configurations[label]["context"].grid
        ratio = int(grid.centers.size // configurations["uniform_N128"][
            "context"
        ].grid.centers.size)
        for profile_name, evaluator in evaluators.items():
            primary = c3._project_callable_to_cells(
                grid,
                evaluator,
                quadrature_order=PRIMARY_PROJECTION_ORDER,
            )
            secondary = c3._project_callable_to_cells(
                grid,
                evaluator,
                quadrature_order=SECONDARY_PROJECTION_ORDER,
            )
            defect = _relative_defect(primary, secondary)
            maximum_projection_defect = max(
                maximum_projection_defect,
                defect,
            )
            support_face = int(
                PROFILE_DEFINITIONS[profile_name][
                    "support_upper_parent_face"
                ]
            ) * ratio
            outside_norm = float(np.linalg.norm(primary[support_face:]))
            endpoint_fraction = _support_endpoint_fraction(
                primary,
                field_scales,
                support_face,
            )
            uniform_values[(profile_name, label, "primary")] = primary
            uniform_values[(profile_name, label, "secondary")] = secondary
            arrays[
                f"{profile_name}__{label}__primary_physical"
            ] = primary
            arrays[
                f"{profile_name}__{label}__secondary_physical"
            ] = secondary
            projection_reports[profile_name][label] = {
                "projection_defect": defect,
                "support_face_index": support_face,
                "outside_support_norm": outside_norm,
                "support_endpoint_cell_fraction": endpoint_fraction,
                "primary_array_sha256": causal_array_sha256(primary),
                "secondary_array_sha256": causal_array_sha256(secondary),
            }

    n128 = configurations["uniform_N128"]
    n128_grid = n128["context"].grid
    spacing = float(np.mean(np.diff(np.log(n128_grid.edges))))
    bases = interpolator.evaluate(
        np.asarray(n128_grid.centers, dtype=float)
    )
    eligibility_reports = {}
    maximum_theta = 0.0
    maximum_alias = 0.0
    maximum_endpoint = 0.0
    minimum_global_purity = 1.0
    minimum_active_purity = 1.0
    for profile_name, definition in PROFILE_DEFINITIONS.items():
        primary = uniform_values[
            (profile_name, "uniform_N128", "primary")
        ]
        spectrum = causal_packet_spectrum(
            primary / field_scales[None, :],
            spacing,
            quantile=float(eligibility["spectral_energy_quantile"]),
        )
        theta = float(
            spectrum.quantile_angular_wavenumber * spacing
        )
        family_index = interpolator.family_labels.index(
            str(definition["family"])
        )
        purity = causal_characteristic_purity(
            primary,
            bases,
            field_scales,
            np.asarray(n128_grid.cell_measures, dtype=float),
            selected_family=family_index,
        )
        global_purity = float(
            purity.family_energy_fractions[family_index]
        )
        active_purity = float(
            purity.minimum_active_cell_selected_fraction
        )
        endpoint = max(
            float(
                projection_reports[profile_name][label][
                    "support_endpoint_cell_fraction"
                ]
            )
            for label in UNIFORM_LABELS
        )
        projection_defect = max(
            float(
                projection_reports[profile_name][label][
                    "projection_defect"
                ]
            )
            for label in UNIFORM_LABELS
        )
        outside_norm = max(
            float(
                projection_reports[profile_name][label][
                    "outside_support_norm"
                ]
            )
            for label in UNIFORM_LABELS
        )
        report = {
            "family": definition["family"],
            "role": definition["role"],
            "theta_99": theta,
            "nyquist_alias_fraction": float(
                spectrum.nyquist_alias_fraction
            ),
            "selected_global_family_fraction": global_purity,
            "minimum_active_cell_family_fraction": active_purity,
            "maximum_projection_defect": projection_defect,
            "maximum_support_endpoint_cell_fraction": endpoint,
            "maximum_outside_support_norm": outside_norm,
        }
        report["passed"] = bool(
            theta <= float(eligibility["maximum_theta_99"])
            and float(spectrum.nyquist_alias_fraction)
            <= float(eligibility["maximum_nyquist_alias_fraction"])
            and global_purity
            >= float(eligibility["minimum_global_family_purity"])
            and active_purity
            >= float(eligibility["minimum_active_cell_family_purity"])
            and projection_defect
            <= float(eligibility["maximum_projection_replay_defect"])
            and endpoint
            <= float(eligibility["maximum_endpoint_cell_fraction"])
            and outside_norm <= MAXIMUM_EXTERIOR_NORM
        )
        eligibility_reports[profile_name] = report
        maximum_theta = max(maximum_theta, theta)
        maximum_alias = max(
            maximum_alias,
            float(spectrum.nyquist_alias_fraction),
        )
        maximum_endpoint = max(maximum_endpoint, endpoint)
        minimum_global_purity = min(
            minimum_global_purity,
            global_purity,
        )
        minimum_active_purity = min(
            minimum_active_purity,
            active_purity,
        )

    reference_restrictions: dict[
        tuple[str, str], np.ndarray
    ] = {}
    layout_reports = {}
    maximum_restriction = 0.0
    maximum_embedded_exterior = 0.0
    maximum_reconstruction_defect = 0.0
    minimum_active_trace = float("inf")
    maximum_buffered_trace = 0.0
    minimum_buffer_parent_cells = float("inf")
    for ratio, embedded_label, uniform_label in zip(
        REFINEMENT_RATIOS,
        EMBEDDED_LABELS,
        UNIFORM_LABELS,
        strict=True,
    ):
        edges = np.asarray(
            c7a_arrays[f"{embedded_label}__grid_edges"],
            dtype=float,
        )
        base = np.asarray(
            c7a_arrays[
                f"{embedded_label}__spliced_base_primitives"
            ],
            dtype=float,
        )
        layout = make_causal_embedded_patch_layout(
            parent_grid,
            COUPLING_PARENT_FACE,
            ratio,
        )
        if not np.array_equal(edges, layout.grid.edges):
            raise RuntimeError(f"{embedded_label} grid replay changed")
        context = wp10c9d5a._context_from_payload(
            replay_contexts["contexts"][embedded_label],
            replay_arrays,
        )
        left_weights, right_weights, reconstruction_defect = (
            _frozen_quadratic_reconstruction_weights(context, base)
        )
        maximum_reconstruction_defect = max(
            maximum_reconstruction_defect,
            reconstruction_defect,
        )
        active_cells = int(layout.coupling_face_index)
        profile_reports = {}
        for profile_name, definition in PROFILE_DEFINITIONS.items():
            primary_inner = uniform_values[
                (profile_name, uniform_label, "primary")
            ]
            secondary_inner = uniform_values[
                (profile_name, uniform_label, "secondary")
            ]
            exterior_cells = int(layout.n_cells - active_cells)
            primary = np.concatenate(
                (
                    primary_inner,
                    np.zeros((exterior_cells, 5), dtype=float),
                ),
                axis=0,
            )
            secondary = np.concatenate(
                (
                    secondary_inner,
                    np.zeros((exterior_cells, 5), dtype=float),
                ),
                axis=0,
            )
            for kind, values in (
                ("primary", primary),
                ("secondary", secondary),
            ):
                arrays[
                    f"{profile_name}__{embedded_label}__"
                    f"{kind}_physical"
                ] = values
                restricted = (
                    restrict_causal_embedded_patch_cell_averages(
                        values,
                        layout,
                    )
                )
                key = (profile_name, kind)
                if key not in reference_restrictions:
                    reference_restrictions[key] = np.array(
                        restricted,
                        copy=True,
                    )
                restriction = _relative_defect(
                    restricted,
                    reference_restrictions[key],
                )
                maximum_restriction = max(
                    maximum_restriction,
                    restriction,
                )
            exterior_norm = float(np.linalg.norm(primary[active_cells:]))
            maximum_embedded_exterior = max(
                maximum_embedded_exterior,
                exterior_norm,
            )
            left_trace = left_weights[active_cells] @ primary
            right_trace = right_weights[active_cells] @ primary
            normalized_cells = primary / field_scales[None, :]
            profile_peak = max(
                float(
                    np.max(
                        np.linalg.norm(normalized_cells, axis=1)
                    )
                ),
                np.finfo(float).tiny,
            )
            trace_fraction = float(
                max(
                    np.linalg.norm(left_trace / field_scales),
                    np.linalg.norm(right_trace / field_scales),
                )
                / profile_peak
            )
            support_face = int(
                definition["support_upper_parent_face"]
            ) * ratio
            buffer_cells = active_cells - support_face
            buffer_norm = float(
                np.linalg.norm(primary[support_face:active_cells])
            )
            expectation = str(
                definition["coupling_trace_expectation"]
            )
            if expectation == "active":
                trace_passed = bool(
                    trace_fraction
                    >= MINIMUM_ACTIVE_COUPLING_TRACE_FRACTION
                )
                minimum_active_trace = min(
                    minimum_active_trace,
                    trace_fraction,
                )
            else:
                trace_passed = bool(
                    trace_fraction
                    <= MAXIMUM_BUFFERED_COUPLING_TRACE_FRACTION
                    and buffer_cells
                    >= MINIMUM_ZERO_BUFFER_PARENT_CELLS * ratio
                    and buffer_norm == 0.0
                )
                maximum_buffered_trace = max(
                    maximum_buffered_trace,
                    trace_fraction,
                )
                minimum_buffer_parent_cells = min(
                    minimum_buffer_parent_cells,
                    buffer_cells / ratio,
                )
            profile_reports[profile_name] = {
                "primary_array_sha256": causal_array_sha256(primary),
                "secondary_array_sha256": causal_array_sha256(
                    secondary
                ),
                "restriction_to_parent_defect": _relative_defect(
                    restrict_causal_embedded_patch_cell_averages(
                        primary,
                        layout,
                    ),
                    reference_restrictions[(profile_name, "primary")],
                ),
                "exterior_norm": exterior_norm,
                "support_face_index": support_face,
                "zero_buffer_cell_count": buffer_cells,
                "zero_buffer_norm": buffer_norm,
                "coupling_left_trace_fraction": float(
                    np.linalg.norm(left_trace / field_scales)
                    / profile_peak
                ),
                "coupling_right_trace_fraction": float(
                    np.linalg.norm(right_trace / field_scales)
                    / profile_peak
                ),
                "maximum_coupling_trace_fraction": trace_fraction,
                "coupling_trace_expectation": expectation,
                "coupling_trace_expectation_passed": trace_passed,
            }
        layout_reports[embedded_label] = {
            "refinement_ratio": ratio,
            "n_cells": int(layout.n_cells),
            "active_cell_count": active_cells,
            "coupling_face_index": active_cells,
            "reconstruction_weight_defect": reconstruction_defect,
            "profile_reports": profile_reports,
        }

    all_uniform_eligible = all(
        report["passed"] for report in eligibility_reports.values()
    )
    all_trace_expectations_pass = all(
        report["coupling_trace_expectation_passed"]
        for layout_report in layout_reports.values()
        for report in layout_report["profile_reports"].values()
    )
    passed = bool(
        construction_report["passed"]
        and characteristic_report["passed"]
        and all_uniform_eligible
        and maximum_projection_defect
        <= float(eligibility["maximum_projection_replay_defect"])
        and maximum_restriction <= MAXIMUM_RESTRICTION_DEFECT
        and maximum_embedded_exterior <= MAXIMUM_EXTERIOR_NORM
        and maximum_reconstruction_defect
        <= MAXIMUM_RECONSTRUCTION_WEIGHT_DEFECT
        and all_trace_expectations_pass
    )

    historical_controls = {}
    packet_reports = c7b_summary["historical_direct_contract_report"][
        "packet_reports"
    ]
    for profile_name in (
        "p3__inward_shear",
        "p3__outward_shear",
        "p5__inward_shear",
        "p5__outward_shear",
    ):
        report = packet_reports[f"{profile_name}::a1.00::plus"]
        historical_controls[profile_name] = {
            "passed": bool(report["passed"]),
            "instantaneous_refinement_error_cosine": float(
                report["instantaneous_exports"][
                    "refinement_error_cosine"
                ]
            ),
            "instantaneous_rms_order": float(
                report["instantaneous_exports"][
                    "observed_rms_order"
                ]
            ),
            "cumulative_passed": bool(
                report["cumulative_exports"]["passed"]
            ),
        }

    variants = _profile_variants()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "parent_commit": ANALYZED_BASE_COMMIT,
        "operator_changed": False,
        "propagation_executed": False,
        "c7b_rejection_preserved": True,
        "c7a_layout_preserved": True,
        "profile_definitions": PROFILE_DEFINITIONS,
        "profile_variants": variants,
        "uniform_eligibility_contract": eligibility,
        "embedded_definition_contract": {
            "parent_coupling_face": COUPLING_PARENT_FACE,
            "buffer_support_parent_face": SUPPORT_BUFFER_PARENT_FACE,
            "minimum_zero_buffer_parent_cells": (
                MINIMUM_ZERO_BUFFER_PARENT_CELLS
            ),
            "minimum_active_coupling_trace_fraction": (
                MINIMUM_ACTIVE_COUPLING_TRACE_FRACTION
            ),
            "maximum_buffered_coupling_trace_fraction": (
                MAXIMUM_BUFFERED_COUPLING_TRACE_FRACTION
            ),
            "maximum_reconstruction_weight_defect": (
                MAXIMUM_RECONSTRUCTION_WEIGHT_DEFECT
            ),
            "maximum_restriction_defect": (
                MAXIMUM_RESTRICTION_DEFECT
            ),
            "maximum_exterior_norm": MAXIMUM_EXTERIOR_NORM,
            "no_profile_taper_shift_or_fit_after_manifest": True,
        },
        "future_propagation_contract": {
            "phase_1": (
                "new_profiles_must_pass_the_unchanged_uniform_state_and_"
                "13_export_contract_before_embedded_propagation"
            ),
            "phase_2": (
                "only_uniform_passes_may_run_on_the_unchanged_three_"
                "embedded_layouts"
            ),
            "physical_export_gates_inherited_from_c7a": True,
            "direct_state_gates_inherited_from_f1": True,
            "minimum_characteristic_energy_relative_activity": 1.0e-8,
            "absolute_reflection_threshold": None,
            "exact_shared_flux_and_prefix_ledgers_required": True,
            "all_sign_and_amplitude_variants_binding": True,
            "no_threshold_or_profile_change_after_this_manifest": True,
        },
        "decision_contract": {
            "p4_and_buffered_p3_pass_embedded": (
                "endpoint_interface_regularity_crossover_no_operator_"
                "redesign_certify_only_declared_smooth_or_buffered_class"
            ),
            "p4_fails_buffered_p3_passes": (
                "active_endpoint_coupling_stencil_hypothesis_selected_"
                "local_truncation_audit_only"
            ),
            "p4_passes_buffered_p3_fails": (
                "short_support_or_global_p3_preasymptotic_hypothesis_"
                "selected_no_interface_redesign"
            ),
            "p4_and_buffered_p3_fail": (
                "no_regularized_embedded_class_selected_stop_before_"
                "operator_change"
            ),
        },
        "historical_controls": historical_controls,
        "uniform_eligibility_reports": eligibility_reports,
        "layout_reports": layout_reports,
        "measured_extrema": {
            "field_scale_defect": field_scale_defect,
            "maximum_projection_defect": maximum_projection_defect,
            "maximum_theta_99": maximum_theta,
            "maximum_nyquist_alias_fraction": maximum_alias,
            "maximum_support_endpoint_cell_fraction": maximum_endpoint,
            "minimum_global_family_purity": minimum_global_purity,
            "minimum_active_cell_family_purity": minimum_active_purity,
            "maximum_restriction_defect": maximum_restriction,
            "maximum_embedded_exterior_norm": (
                maximum_embedded_exterior
            ),
            "maximum_reconstruction_weight_defect": (
                maximum_reconstruction_defect
            ),
            "minimum_active_coupling_trace_fraction": (
                minimum_active_trace
            ),
            "maximum_buffered_coupling_trace_fraction": (
                maximum_buffered_trace
            ),
            "minimum_zero_buffer_parent_cells": (
                minimum_buffer_parent_cells
            ),
        },
        "all_uniform_profiles_eligible": all_uniform_eligible,
        "all_coupling_trace_expectations_passed": (
            all_trace_expectations_pass
        ),
        "passed": passed,
    }
    manifest = {
        **payload,
        "manifest_sha256": causal_canonical_json_sha256(payload),
    }
    return manifest, arrays


def _config(manifest: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "operator_change": False,
        "propagation_executed": False,
        "profile_definitions": manifest["profile_definitions"],
        "profile_variants": manifest["profile_variants"],
        "uniform_eligibility_contract": (
            manifest["uniform_eligibility_contract"]
        ),
        "embedded_definition_contract": (
            manifest["embedded_definition_contract"]
        ),
        "future_propagation_contract": (
            manifest["future_propagation_contract"]
        ),
        "decision_contract": manifest["decision_contract"],
    }


def run() -> dict:
    started = time.perf_counter()
    identity = _validate_analyzed_git_identity()
    manifest, arrays = _build()
    source_hashes, source_manifest = _source_manifest()
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_json(CONFIG_PATH, _config(manifest))
    _write_json(MANIFEST_PATH, manifest)
    np.savez_compressed(DECISIVE_ARRAYS, **arrays)
    passed = bool(manifest["passed"])
    classification = (
        "endpoint_interface_regularity_manifest_frozen_"
        "uniform_control_preflight_authorized"
        if passed
        else "endpoint_interface_regularity_controls_ineligible"
    )
    authorized_next = (
        "WP10c9d6c7c1_uniform_then_embedded_regularity_discrimination"
        if passed
        else None
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        **identity,
        "classification": classification,
        "authorized_next": authorized_next,
        "passed": passed,
        "operator_changed": False,
        "propagation_executed": False,
        "c7b_rejection_preserved": True,
        "c7a_layout_preserved": True,
        "manifest_sha256": manifest["manifest_sha256"],
        "profile_count": len(PROFILE_NAMES),
        "profile_variant_count": len(manifest["profile_variants"]),
        "uniform_eligibility_reports": (
            manifest["uniform_eligibility_reports"]
        ),
        "layout_reports": manifest["layout_reports"],
        "historical_controls": manifest["historical_controls"],
        "measured_extrema": manifest["measured_extrema"],
        "all_uniform_profiles_eligible": (
            manifest["all_uniform_profiles_eligible"]
        ),
        "all_coupling_trace_expectations_passed": (
            manifest["all_coupling_trace_expectations_passed"]
        ),
        "implementation_source_hashes": source_hashes,
        "implementation_source_manifest_sha256": source_manifest,
        "decisive_arrays_path": str(
            DECISIVE_ARRAYS.relative_to(ROOT)
        ),
        "decisive_arrays_sha256": _sha256(DECISIVE_ARRAYS),
        "decisive_array_hashes": _array_hashes(arrays),
        "uniform_control_propagation_authorized": passed,
        "embedded_control_propagation_authorized": False,
        "bounded_nonlinear_common_mode_authorized": False,
        "nonlinear_physical_trajectory_authorized": False,
        "production_operator_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "runtime_seconds": float(time.perf_counter() - started),
    }
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "scientific_status": (
            "DEFINITIONS FROZEN; UNIFORM CONTROL PREFLIGHT AUTHORIZED"
            if passed
            else "DEFINITIONS INELIGIBLE"
        ),
        "source_parent_commit": ANALYZED_BASE_COMMIT,
        **identity,
        "implementation_base_tree": _git_value("rev-parse", "HEAD^{tree}"),
        "working_tree_status": _git_value("status", "--short"),
        "command": (
            "PYTHONPATH=src:scripts python scripts/"
            "run_causal_inner_embedded_regularity_manifest_"
            "wp10c9d6c7c0.py"
        ),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
        },
        "implementation_source_hashes": source_hashes,
        "implementation_source_manifest_sha256": source_manifest,
        "parent_canonical_hashes": _parent_hashes(),
    }
    _write_json(SUMMARY_PATH, summary)
    _write_json(PROVENANCE_PATH, provenance)
    _refresh_sha256s(CANONICAL_DIRECTORY)
    _refresh_canonical_catalog()
    print(
        json.dumps(
            {
                "classification": classification,
                "authorized_next": authorized_next,
                "manifest_sha256": manifest["manifest_sha256"],
                "all_uniform_profiles_eligible": manifest[
                    "all_uniform_profiles_eligible"
                ],
                "all_coupling_trace_expectations_passed": manifest[
                    "all_coupling_trace_expectations_passed"
                ],
                "measured_extrema": manifest["measured_extrema"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return summary


if __name__ == "__main__":
    run()
