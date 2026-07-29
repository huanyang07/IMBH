#!/usr/bin/env python3
"""Run WP10c9d5c1 extended non-tautological localization.

The audit uses the certified analytic frozen-subspace generator and searches
through the last common face whose reconstruction halo remains inside the
embedded inner patch.  Direct face-flux actions are the export targets.  The
target outer face is excluded from all explanatory groups.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for search_path in (SRC, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import run_causal_inner_analytic_tangent_physical_sensitivity_wp10c9d5c0f as wp10c9d5c0f
import run_causal_inner_cross_grid_analytic_tangent_wp10c9d5c0e as wp10c9d5c0e
import run_causal_inner_cross_grid_hardening_wp10c9d5c0 as wp10c9d5c0
import run_causal_inner_derivative_repair_wp10c9d5c0a as wp10c9d5c0a
import run_causal_inner_dynamic_localization_wp10c9d5b as wp10c9d5b

from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    causal_five_field_frozen_analytic_tangent,
    causal_five_field_radial_analytic_tangent,
    causal_five_field_reduced_storage_matrices,
    causal_radial_first_consecutive_recovery,
    causal_radial_history_convergence,
    causal_radial_prefix_face_fluxes,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d5c1"
ANALYZED_BASE_COMMIT = "f409244f0f9b487b918d4e93f49e8bcf41049af1"
ANALYZED_BASE_PARENT = "e5fd93352aea3dc920e528bb566b60fa7a3c8b0c"
ANALYZED_BASE_TREE = "be11d8cdf3825c50f857ba7fc198ca212e0a16a2"
THIS_RUNNER = (
    "scripts/run_causal_inner_extended_localization_wp10c9d5c1.py"
)

LABELS = tuple(wp10c9d5c0e.LABELS)
N_FIELDS = 5
CONSERVATIVE_FIELDS = np.asarray((0, 2, 3), dtype=int)
FIELD_NAMES = ("mass", "angular_momentum", "killing_energy")
BLOCK_NAMES = tuple(wp10c9d5b.BLOCK_NAMES)
PATH_QUADRATURE_ORDER = 6
RECONSTRUCTION_HALO_CELLS = 3
STRIDE_AUDITS = (1, 2, 4)
DIRECT_PARITY_RADII_OVER_RG = (3.0, 5.0, 8.0)
DIRECT_PARITY_TIME_FRACTIONS = (0.0, 0.5, 1.0)

MINIMUM_RECOVERY_ORDER = 0.75
MAXIMUM_FINE_NORMALIZED_DIFFERENCE = 0.05
MINIMUM_HISTORY_COSINE = 0.90
MINIMUM_ERROR_COSINE = 0.90
MINIMUM_RELATIVE_ACTIVITY = 1.0e-8
REQUIRED_CONSECUTIVE_SURFACES = 2
MAXIMUM_ANALYTIC_GENERATOR_REPLAY_DEFECT = 1.0e-12
MAXIMUM_PREFIX_FACE_PARITY_DEFECT = 1.0e-12
MAXIMUM_MOVING_PROJECTOR_FACE_PARITY_DEFECT = 2.0e-6
MAXIMUM_CONTROL_VOLUME_CLOSURE_DEFECT = 1.0e-10
MAXIMUM_STRIDE_DEFECT = 5.0e-3

MINIMUM_GROUP_TARGET_ALIGNED_FRACTION = 0.80
MAXIMUM_GROUP_FIXED_COEFFICIENT_RESIDUAL = 0.45
MINIMUM_GROUP_SUBSPACE_COSINE = 0.90

EXPLANATORY_TERMS = (
    "inner_shared_face",
    "shear_principal",
    "height_principal",
    "local_stress_relaxation",
    "geometry",
    "cooling",
    "stream",
    "lower_height_work",
    "mapped_storage_rate",
    "responsive_height_storage_rate",
    "production_anchor_storage_derivative",
)
GROUPS = {
    "inner_boundary": ("inner_shared_face",),
    "mapped_anchor_storage": (
        "mapped_storage_rate",
        "production_anchor_storage_derivative",
    ),
    "height_space_storage": (
        "height_principal",
        "responsive_height_storage_rate",
        "lower_height_work",
    ),
    "stress_principal_relaxation": (
        "shear_principal",
        "local_stress_relaxation",
    ),
    "lower_sources": ("geometry", "cooling", "stream"),
}

PARENT_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_analytic_tangent_physical_sensitivity_wp10c9d5c0f"
)
PARENT_SUMMARY = PARENT_DIRECTORY / "summary.json"
PARENT_DECISIVE_ARRAYS = PARENT_DIRECTORY / "decisive_arrays.npz"
C0E_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_cross_grid_analytic_tangent_wp10c9d5c0e"
)
C0E_SUMMARY = C0E_DIRECTORY / "summary.json"
C0E_DECISIVE_ARRAYS = C0E_DIRECTORY / "decisive_arrays.npz"

CANONICAL_DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_extended_localization_wp10c9d5c1"
)
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"

IMPLEMENTATION_SOURCES = (
    THIS_RUNNER,
    "src/imri_qpe/layer3_minidisk_1d/"
    "causal_inner_radial_linear_tangent.py",
    "tests/test_causal_inner_extended_localization_wp10c9d5c1.py",
    "tests/test_causal_inner_radial_linear_tangent.py",
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


def _relative_difference(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=float)
    right = np.asarray(second, dtype=float)
    scale = max(
        float(np.linalg.norm(left)),
        float(np.linalg.norm(right)),
        np.finfo(float).tiny,
    )
    return float(np.linalg.norm(left - right) / scale)


def _refresh_sha256s(directory: Path) -> None:
    entries = []
    for path in sorted(directory.iterdir()):
        if path.name == "SHA256SUMS.txt" or not path.is_file():
            continue
        entries.append(f"{_sha256(path)}  {path.name}")
    (directory / "SHA256SUMS.txt").write_text(
        "\n".join(entries) + "\n",
        encoding="utf-8",
    )


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _validate_analyzed_git_identity() -> dict:
    commit = _git("rev-parse", ANALYZED_BASE_COMMIT)
    parent = _git("rev-parse", f"{ANALYZED_BASE_COMMIT}^")
    tree = _git("rev-parse", f"{ANALYZED_BASE_COMMIT}^{{tree}}")
    if (commit, parent, tree) != (
        ANALYZED_BASE_COMMIT,
        ANALYZED_BASE_PARENT,
        ANALYZED_BASE_TREE,
    ):
        raise RuntimeError("WP10c9d5c1 analyzed Git identity changed")
    return {
        "analyzed_base_commit": commit,
        "analyzed_base_parent_commit": parent,
        "analyzed_base_tree_sha": tree,
    }


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


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {
            name: np.asarray(source[name])
            for name in source.files
        }


def _load_inputs() -> tuple[dict, dict, dict, dict[str, np.ndarray]]:
    parent = json.loads(PARENT_SUMMARY.read_text(encoding="utf-8"))
    if (
        not parent["passed"]
        or not parent["derivative_choice_physical_sensitivity_passed"]
        or not parent["wp10c9d5c1_extended_localization_authorized"]
        or not parent["parent_wp10c9d5_candidate_remains_rejected"]
        or parent["self_consistent_tangent_authorized"]
    ):
        raise RuntimeError("WP10c9d5c0f binding classification changed")
    parent_arrays = _load_npz(PARENT_DECISIVE_ARRAYS)
    if _sha256(PARENT_DECISIVE_ARRAYS) != parent[
        "decisive_arrays_sha256"
    ]:
        raise RuntimeError("WP10c9d5c0f decisive arrays changed")
    c0e = json.loads(C0E_SUMMARY.read_text(encoding="utf-8"))
    if (
        not c0e["passed"]
        or not c0e["cross_grid_analytic_tangent_certified"]
    ):
        raise RuntimeError("WP10c9d5c0e certification changed")
    c0e_arrays = _load_npz(C0E_DECISIVE_ARRAYS)
    replay_payload, replay_arrays = wp10c9d5c0e._load_replay_inputs()
    configurations = wp10c9d5c0e._configurations(
        replay_payload,
        replay_arrays,
    )
    return parent, configurations, parent_arrays, c0e_arrays


def _cumulative(times: np.ndarray, values: np.ndarray) -> np.ndarray:
    time = np.asarray(times, dtype=float)
    data = np.asarray(values, dtype=float)
    result = np.zeros_like(data)
    shape = (time.size - 1,) + (1,) * (data.ndim - 1)
    increments = (
        0.5
        * np.diff(time).reshape(shape)
        * (data[:-1] + data[1:])
    )
    result[1:] = np.cumsum(increments, axis=0)
    return result


def _physical_rows(
    matrix: np.ndarray,
    values: np.ndarray,
    row_scales: np.ndarray,
    n_cells: int,
) -> np.ndarray:
    return (
        np.asarray(
            [np.asarray(matrix) @ vector for vector in values],
            dtype=float,
        )
        * row_scales[None, :]
    ).reshape(values.shape[0], n_cells, N_FIELDS)


def _common_faces(configurations: dict) -> tuple[np.ndarray, dict]:
    factors = dict(zip(LABELS, (1, 2, 4), strict=True))
    coarse = configurations[LABELS[0]]
    last = min(
        (
            int(configuration["active_cells"])
            - RECONSTRUCTION_HALO_CELLS
        )
        // factors[label]
        for label, configuration in configurations.items()
    )
    coarse_faces = np.arange(last + 1, dtype=int)
    maps = {
        label: factors[label] * coarse_faces
        for label in LABELS
    }
    coarse_edges = (
        np.asarray(coarse["context"].grid.edges, dtype=float)
        / float(coarse["context"].grid.gravitational_radius)
    )
    for label, indices in maps.items():
        configuration = configurations[label]
        if int(indices[-1]) + RECONSTRUCTION_HALO_CELLS > int(
            configuration["active_cells"]
        ):
            raise RuntimeError("common face exceeds certified inner halo")
        edges = (
            np.asarray(configuration["context"].grid.edges, dtype=float)
            / float(configuration["context"].grid.gravitational_radius)
        )
        if not np.allclose(
            edges[indices],
            coarse_edges[coarse_faces],
            rtol=5.0e-14,
            atol=0.0,
        ):
            raise RuntimeError("embedded grids do not share c1 faces")
    return coarse_edges[coarse_faces], maps


def _metrics_payload(metrics) -> dict:
    return wp10c9d5c0._metrics_payload(metrics)


def _convergence(
    histories: dict[str, np.ndarray],
    *,
    scales: np.ndarray,
) -> object | None:
    try:
        return causal_radial_history_convergence(
            *(histories[label] for label in LABELS),
            minimum_order=MINIMUM_RECOVERY_ORDER,
            maximum_fine_normalized_difference=(
                MAXIMUM_FINE_NORMALIZED_DIFFERENCE
            ),
            minimum_fine_signed_cosine=MINIMUM_HISTORY_COSINE,
            minimum_relative_activity=MINIMUM_RELATIVE_ACTIVITY,
            component_reference_scales=scales,
            minimum_error_cosine=MINIMUM_ERROR_COSINE,
        )
    except ValueError as error:
        if str(error) != "history convergence has no significant component":
            raise
        return None


def _optional_metrics_payload(metrics) -> dict:
    if metrics is None:
        return {
            "active": False,
            "passed": False,
            "reason": "no component exceeds the fixed physical activity gate",
        }
    return {
        "active": True,
        **_metrics_payload(metrics),
    }


def _prefix(values: np.ndarray, face: int) -> np.ndarray:
    return np.sum(
        np.asarray(values, dtype=float)[
            :, : int(face), :
        ][:, :, CONSERVATIVE_FIELDS],
        axis=1,
    )


def _balance_histories(history: dict, face: int) -> dict[str, np.ndarray]:
    return {
        "inner_shared_face": -history["direct_face_fluxes"][:, 0],
        "shear_principal": _prefix(
            history["shear_principal_rows"],
            face,
        ),
        "height_principal": _prefix(
            history["height_principal_rows"],
            face,
        ),
        "local_stress_relaxation": _prefix(
            history["local_stress_relaxation_rows"],
            face,
        ),
        "geometry": _prefix(history["geometry_rows"], face),
        "cooling": _prefix(history["cooling_rows"], face),
        "stream": _prefix(history["stream_rows"], face),
        "lower_height_work": _prefix(
            history["lower_height_work_rows"],
            face,
        ),
        "mapped_storage_rate": _prefix(
            history["mapped_storage_rate"],
            face,
        ),
        "responsive_height_storage_rate": _prefix(
            history["vertical_storage_rate"],
            face,
        ),
        "production_anchor_storage_derivative": _prefix(
            history["production_anchor_storage_derivative"],
            face,
        ),
    }


def _normalized_vector(
    values: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    return (
        np.asarray(values, dtype=float)
        / np.asarray(scales, dtype=float)[None, :]
    ).ravel()


def _subspace_cosine(
    first: dict[str, np.ndarray],
    second: dict[str, np.ndarray],
    names: tuple[str, ...],
    scales: np.ndarray,
) -> float:
    matrices = []
    for blocks in (first, second):
        matrix = np.column_stack(
            tuple(
                _normalized_vector(blocks[name], scales)
                for name in names
            )
        )
        left, singular, _right = np.linalg.svd(
            matrix,
            full_matrices=False,
        )
        threshold = (
            np.finfo(float).eps
            * max(matrix.shape)
            * max(float(singular[0]), np.finfo(float).tiny)
        )
        rank = int(np.sum(singular > threshold))
        matrices.append(left[:, :rank])
    if matrices[0].shape[1] == 0 and matrices[1].shape[1] == 0:
        return 1.0
    if matrices[0].shape[1] != matrices[1].shape[1]:
        return 0.0
    cosines = np.linalg.svd(
        matrices[0].T @ matrices[1],
        compute_uv=False,
    )
    return float(np.clip(np.min(cosines), 0.0, 1.0))


def _pair_attribution(
    target: np.ndarray,
    terms: dict[str, np.ndarray],
    scales: np.ndarray,
) -> tuple[dict, np.ndarray]:
    target_vector = _normalized_vector(target, scales)
    target_activity = float(
        np.max(
            np.abs(np.asarray(target, dtype=float))
            / np.asarray(scales, dtype=float)[None, :]
        )
    )
    target_norm_squared = max(
        float(np.dot(target_vector, target_vector)),
        np.finfo(float).tiny,
    )
    term_vectors = np.column_stack(
        tuple(
            _normalized_vector(terms[name], scales)
            for name in EXPLANATORY_TERMS
        )
    )
    gram = term_vectors.T @ term_vectors
    groups = {}
    for group_name, names in GROUPS.items():
        values = sum(
            (terms[name] for name in names),
            start=np.zeros_like(target),
        )
        vector = _normalized_vector(values, scales)
        alpha = -float(np.dot(target_vector, vector)) / target_norm_squared
        rho = float(
            np.linalg.norm(target_vector + vector)
            / math.sqrt(target_norm_squared)
        )
        groups[group_name] = {
            "target_aligned_fraction": alpha,
            "fixed_coefficient_residual": rho,
        }
    complete = sum(
        (terms[name] for name in EXPLANATORY_TERMS),
        start=np.zeros_like(target),
    )
    complete_vector = _normalized_vector(complete, scales)
    scaled_closure = float(
        np.linalg.norm(target_vector + complete_vector)
    )
    active = bool(target_activity > MINIMUM_RELATIVE_ACTIVITY)
    relative_closure = (
        float(
            scaled_closure
            / max(
                float(np.linalg.norm(target_vector)),
                float(np.linalg.norm(complete_vector)),
                np.finfo(float).tiny,
            )
        )
        if active
        else None
    )
    return {
        "active": active,
        "maximum_target_activity": target_activity,
        "target_norm": math.sqrt(target_norm_squared),
        "groups": groups,
        # This is an absolute residual in the predeclared fixed physical
        # M/J/E normalization.  A relative residual is ill-defined once the
        # refinement target falls below the activity gate; using it as a
        # method gate would turn harmless roundoff at inactive outer faces
        # into an apparent O(1) ledger failure.
        "complete_explanatory_closure_defect": scaled_closure,
        "complete_explanatory_relative_closure_defect": (
            relative_closure
        ),
    }, gram


def _surface_attribution(
    histories: dict,
    face_map: dict[str, int],
    times: np.ndarray,
    face_scales: np.ndarray,
) -> tuple[dict, dict[str, np.ndarray]]:
    balances = {
        label: _balance_histories(histories[label], face_map[label])
        for label in LABELS
    }
    targets = {
        label: histories[label]["direct_face_fluxes"][
            :,
            face_map[label],
            :,
        ]
        for label in LABELS
    }
    pairs = {
        "coarse_medium": (LABELS[0], LABELS[1]),
        "medium_fine": (LABELS[1], LABELS[2]),
    }
    reports = {"instantaneous": {}, "cumulative": {}}
    arrays = {}
    pair_terms = {"instantaneous": {}, "cumulative": {}}
    duration = max(float(times[-1]), np.finfo(float).tiny)
    for pair_name, (left, right) in pairs.items():
        instantaneous_terms = {
            name: balances[right][name] - balances[left][name]
            for name in EXPLANATORY_TERMS
        }
        cumulative_terms = {
            name: _cumulative(times, values)
            for name, values in instantaneous_terms.items()
        }
        target = targets[right] - targets[left]
        cumulative_target = _cumulative(times, target)
        pair_terms["instantaneous"][pair_name] = instantaneous_terms
        pair_terms["cumulative"][pair_name] = cumulative_terms
        for kind, values, terms, scales in (
            (
                "instantaneous",
                target,
                instantaneous_terms,
                face_scales,
            ),
            (
                "cumulative",
                cumulative_target,
                cumulative_terms,
                face_scales * duration,
            ),
        ):
            report, gram = _pair_attribution(values, terms, scales)
            reports[kind][pair_name] = report
            arrays[f"{kind}__{pair_name}__gram"] = gram
            arrays[f"{kind}__{pair_name}__target"] = values
            for name, term in terms.items():
                arrays[f"{kind}__{pair_name}__term__{name}"] = term

    group_reports = {}
    for group_name, names in GROUPS.items():
        kind_reports = {}
        group_passed = True
        for kind, scales in (
            ("instantaneous", face_scales),
            ("cumulative", face_scales * duration),
        ):
            coarse_medium = reports[kind]["coarse_medium"]["groups"][
                group_name
            ]
            medium_fine = reports[kind]["medium_fine"]["groups"][
                group_name
            ]
            subspace_cosine = _subspace_cosine(
                pair_terms[kind]["coarse_medium"],
                pair_terms[kind]["medium_fine"],
                names,
                scales,
            )
            kind_passed = bool(
                reports[kind]["coarse_medium"]["active"]
                and reports[kind]["medium_fine"]["active"]
                and
                all(
                    report["target_aligned_fraction"]
                    >= MINIMUM_GROUP_TARGET_ALIGNED_FRACTION
                    and report["fixed_coefficient_residual"]
                    <= MAXIMUM_GROUP_FIXED_COEFFICIENT_RESIDUAL
                    for report in (coarse_medium, medium_fine)
                )
                and subspace_cosine >= MINIMUM_GROUP_SUBSPACE_COSINE
            )
            group_passed = bool(group_passed and kind_passed)
            kind_reports[kind] = {
                "coarse_medium": coarse_medium,
                "medium_fine": medium_fine,
                "subspace_cosine": subspace_cosine,
                "passed": kind_passed,
            }
        group_reports[group_name] = {
            "terms": names,
            "kinds": kind_reports,
            "passed": group_passed,
        }
    maximum_closure = max(
        reports[kind][pair]["complete_explanatory_closure_defect"]
        for kind in reports
        for pair in pairs
    )
    return {
        "pairs": reports,
        "groups": group_reports,
        "maximum_complete_explanatory_closure_defect": maximum_closure,
    }, arrays


def _history_for_grid(
    configuration: dict,
    parent_arrays: dict[str, np.ndarray],
    c0e_arrays: dict[str, np.ndarray],
) -> tuple[dict, dict]:
    label = configuration["label"]
    native = configuration["candidate_native"]
    analytic = causal_five_field_radial_analytic_tangent(
        configuration["context"],
        configuration["base_primitives"],
        primitive_column_scales=native["primitive_column_scales"],
        conservation_row_scales=native["conservation_row_scales"],
        path_quadrature_order=PATH_QUADRATURE_ORDER,
    )
    frozen = causal_five_field_frozen_analytic_tangent(
        analytic,
        native["production_generator"],
        native["descriptor"],
        configuration["anchor_storage_derivative"],
    )
    stored_generator = wp10c9d5c0e._unpack_sparse(
        f"{label}__analytic_candidate_generator",
        c0e_arrays,
    ).toarray()
    generator_replay = _relative_difference(
        frozen.candidate_scaled_generator_per_s,
        stored_generator,
    )
    scaled_state = np.asarray(
        parent_arrays[
            f"common_mode__analytic_frozen_subspace__{label}__scaled_state"
        ],
        dtype=float,
    )
    times = np.asarray(
        parent_arrays[
            f"common_mode__analytic_frozen_subspace__{label}__times"
        ],
        dtype=float,
    )
    scaled_rate = np.asarray(
        [
            frozen.candidate_scaled_generator_per_s @ state
            for state in scaled_state
        ],
        dtype=float,
    )
    rows = np.asarray(
        native["conservation_row_scales"],
        dtype=float,
    )
    n_cells = int(configuration["base_primitives"].shape[0])
    block_rows = {
        name: _physical_rows(
            analytic.block_scaled_jacobians[f"candidate_{name}"],
            scaled_state,
            rows,
            n_cells,
        )
        for name in BLOCK_NAMES
    }
    storage = causal_five_field_reduced_storage_matrices(
        configuration["context"],
        np.asarray(configuration["base_primitives"], dtype=float).ravel(),
        primitive_column_scales=native["primitive_column_scales"],
        conservation_row_scales=native["conservation_row_scales"],
    )
    mapped = np.asarray(
        storage["conserved_descriptor_reduced_scaled_matrix"],
        dtype=float,
    )
    vertical = np.asarray(
        storage["vertical_descriptor_reduced_scaled_matrix"],
        dtype=float,
    )
    mapped_rate = _physical_rows(
        mapped,
        scaled_rate,
        rows,
        n_cells,
    )
    vertical_rate = _physical_rows(
        vertical,
        scaled_rate,
        rows,
        n_cells,
    )
    anchor = _physical_rows(
        configuration["anchor_storage_derivative"],
        scaled_state,
        rows,
        n_cells,
    )
    direct_face_fluxes = np.einsum(
        "fkd,td->tfk",
        analytic.shared_face_flux_scaled_jacobians[
            :,
            CONSERVATIVE_FIELDS,
            :,
        ],
        scaled_state,
    )
    prefix_face_fluxes = causal_radial_prefix_face_fluxes(
        direct_face_fluxes[:, 0],
        block_rows["conservative_transport"][
            :,
            :,
            CONSERVATIVE_FIELDS,
        ],
    )
    prefix_parity = _relative_difference(
        direct_face_fluxes,
        prefix_face_fluxes,
    )
    stationary = sum(
        block_rows.values(),
        start=np.zeros_like(block_rows[BLOCK_NAMES[0]]),
    )
    full_balance = mapped_rate + vertical_rate + anchor + stationary
    balance_scale = max(
        float(np.linalg.norm(mapped_rate + vertical_rate + anchor)),
        float(np.linalg.norm(stationary)),
        np.finfo(float).tiny,
    )
    closure = float(np.linalg.norm(full_balance) / balance_scale)
    history = {
        "times": times,
        "scaled_state": scaled_state,
        "scaled_rate": scaled_rate,
        "direct_face_fluxes": direct_face_fluxes,
        "prefix_face_fluxes": prefix_face_fluxes,
        "mapped_storage_rate": mapped_rate,
        "vertical_storage_rate": vertical_rate,
        "production_anchor_storage_derivative": anchor,
        **{f"{name}_rows": values for name, values in block_rows.items()},
    }
    return history, {
        "generator_replay_defect": generator_replay,
        "prefix_face_parity_defect": prefix_parity,
        "control_volume_closure_defect": closure,
        "analytic": analytic,
    }


def _moving_projector_face_parity(
    configurations: dict,
    histories: dict,
    common_radii: np.ndarray,
    face_maps: dict,
) -> tuple[dict, dict[str, np.ndarray]]:
    selected_surfaces = sorted(
        {
            *(
                int(np.argmin(np.abs(common_radii - radius)))
                for radius in DIRECT_PARITY_RADII_OVER_RG
            ),
            int(common_radii.size - 1),
        }
    )
    reports = {}
    arrays = {
        "selected_surface_indices": np.asarray(
            selected_surfaces,
            dtype=int,
        ),
        "selected_radii_over_rg": common_radii[selected_surfaces],
    }
    maximum = 0.0
    for label in LABELS:
        history = histories[label]
        times = np.asarray(history["times"], dtype=float)
        time_indices = np.asarray(
            [
                int(round(fraction * (times.size - 1)))
                for fraction in DIRECT_PARITY_TIME_FRACTIONS
            ],
            dtype=int,
        )
        faces = np.asarray(
            face_maps[label][selected_surfaces],
            dtype=int,
        )
        defects = []
        for time_index in time_indices:
            direction = history["scaled_state"][time_index]
            direct = wp10c9d5c0a._direct_selected_face_actions(
                configurations[label],
                direction,
                faces,
            )[OUTPUT_REFERENCE_ORDER]
            analytic = history["direct_face_fluxes"][
                time_index,
                faces,
                :,
            ].ravel()
            defects.append(_relative_difference(direct, analytic))
        maximum = max(maximum, max(defects))
        reports[label] = {
            "face_indices": faces,
            "time_indices": time_indices,
            "relative_defects": defects,
            "maximum_relative_defect": max(defects),
            "passed": bool(
                max(defects)
                <= MAXIMUM_MOVING_PROJECTOR_FACE_PARITY_DEFECT
            ),
        }
        arrays[f"{label}__relative_defects"] = np.asarray(
            defects,
            dtype=float,
        )
    return {
        "configurations": reports,
        "maximum_relative_defect": maximum,
        "passed": bool(
            maximum <= MAXIMUM_MOVING_PROJECTOR_FACE_PARITY_DEFECT
        ),
    }, arrays


OUTPUT_REFERENCE_ORDER = 6


def _recovery_for_stride(
    histories: dict,
    common_radii: np.ndarray,
    face_maps: dict,
    face_scales: np.ndarray,
    stride: int,
) -> dict:
    surfaces = []
    passes = []
    sampled_times = np.asarray(
        histories[LABELS[0]]["times"],
        dtype=float,
    )[::stride]
    duration = max(float(sampled_times[-1]), np.finfo(float).tiny)
    for surface, radius in enumerate(common_radii):
        instant = {
            label: histories[label]["direct_face_fluxes"][
                ::stride,
                face_maps[label][surface],
                :,
            ]
            for label in LABELS
        }
        cumulative = {
            label: _cumulative(sampled_times, values)
            for label, values in instant.items()
        }
        instant_metrics = _convergence(
            instant,
            scales=face_scales,
        )
        cumulative_metrics = _convergence(
            cumulative,
            scales=face_scales * duration,
        )
        passed = bool(
            instant_metrics is not None
            and cumulative_metrics is not None
            and instant_metrics.passed
            and cumulative_metrics.passed
        )
        passes.append(passed)
        surfaces.append(
            {
                "surface": surface,
                "radius_over_rg": float(radius),
                "instantaneous": _optional_metrics_payload(
                    instant_metrics
                ),
                "cumulative": _optional_metrics_payload(
                    cumulative_metrics
                ),
                "passed": passed,
            }
        )
    recovery_index = causal_radial_first_consecutive_recovery(
        np.asarray(passes, dtype=bool),
        required_consecutive=REQUIRED_CONSECUTIVE_SURFACES,
    )
    return {
        "stride": stride,
        "surface_reports": surfaces,
        "surface_passes": passes,
        "recovery_surface_index": recovery_index,
        "recovery_radius_over_rg": (
            None
            if recovery_index is None
            else float(common_radii[recovery_index])
        ),
    }


def _stride_report(
    histories: dict,
    common_radii: np.ndarray,
    face_maps: dict,
    face_scales: np.ndarray,
) -> tuple[dict, dict[int, dict]]:
    reports = {
        stride: _recovery_for_stride(
            histories,
            common_radii,
            face_maps,
            face_scales,
            stride,
        )
        for stride in STRIDE_AUDITS
    }
    maximum_endpoint_defect = 0.0
    for label in LABELS:
        times = np.asarray(histories[label]["times"], dtype=float)
        faces = histories[label]["direct_face_fluxes"][
            :,
            face_maps[label],
            :,
        ]
        reference = _cumulative(times, faces)[-1]
        duration = max(float(times[-1]), np.finfo(float).tiny)
        for stride in STRIDE_AUDITS[1:]:
            sampled = np.arange(0, times.size, stride, dtype=int)
            if sampled[-1] != times.size - 1:
                sampled = np.append(sampled, times.size - 1)
            endpoint = _cumulative(
                times[sampled],
                faces[sampled],
            )[-1]
            defect = float(
                np.max(
                    np.abs(endpoint - reference)
                    / (face_scales[None, :] * duration)
                )
            )
            maximum_endpoint_defect = max(
                maximum_endpoint_defect,
                defect,
            )
    indices = [
        reports[stride]["recovery_surface_index"]
        for stride in STRIDE_AUDITS
    ]
    stable = bool(
        all(index is None for index in indices)
        or (
            all(index is not None for index in indices)
            and max(int(index) for index in indices)
            - min(int(index) for index in indices)
            <= 1
        )
    )
    return {
        "maximum_cumulative_endpoint_defect": maximum_endpoint_defect,
        "recovery_location_stable": stable,
        "recovery_indices": dict(zip(STRIDE_AUDITS, indices, strict=True)),
        "passed": bool(
            maximum_endpoint_defect <= MAXIMUM_STRIDE_DEFECT and stable
        ),
    }, reports


def _stable_group(
    surface_reports: list[dict],
) -> tuple[str | None, int | None]:
    for start in range(1, len(surface_reports) - 1):
        for group_name in GROUPS:
            if (
                surface_reports[start]["groups"][group_name]["passed"]
                and surface_reports[start + 1]["groups"][group_name][
                    "passed"
                ]
            ):
                return group_name, start
    return None, None


def run() -> dict:
    started = time.perf_counter()
    identity = _validate_analyzed_git_identity()
    parent, configurations, parent_arrays, c0e_arrays = _load_inputs()
    common_radii, face_maps = _common_faces(configurations)
    face_scales = wp10c9d5c0._fixed_face_scales(configurations)
    decisive: dict[str, np.ndarray] = {
        "common_face_radii_over_rg": common_radii,
        "fixed_face_scales": face_scales,
    }
    histories = {}
    grid_reports = {}
    method_passed = True
    for label in LABELS:
        print(f"WP10c9d5c1: assemble {label}", flush=True)
        history, report = _history_for_grid(
            configurations[label],
            parent_arrays,
            c0e_arrays,
        )
        analytic = report.pop("analytic")
        histories[label] = history
        report["passed"] = bool(
            report["generator_replay_defect"]
            <= MAXIMUM_ANALYTIC_GENERATOR_REPLAY_DEFECT
            and report["prefix_face_parity_defect"]
            <= MAXIMUM_PREFIX_FACE_PARITY_DEFECT
            and report["control_volume_closure_defect"]
            <= MAXIMUM_CONTROL_VOLUME_CLOSURE_DEFECT
        )
        method_passed = bool(method_passed and report["passed"])
        grid_reports[label] = report
        decisive[f"{label}__times"] = history["times"]
        decisive[f"{label}__direct_face_fluxes"] = (
            history["direct_face_fluxes"][:, face_maps[label], :]
        )
        decisive[f"{label}__prefix_face_fluxes"] = (
            history["prefix_face_fluxes"][:, face_maps[label], :]
        )
        decisive[f"{label}__scaled_state"] = history["scaled_state"]
        decisive[f"{label}__scaled_rate"] = history["scaled_rate"]
        decisive[f"{label}__shared_face_flux_scaled_jacobians"] = (
            analytic.shared_face_flux_scaled_jacobians[
                face_maps[label],
                :,
                :,
            ]
        )

    moving_parity, moving_arrays = _moving_projector_face_parity(
        configurations,
        histories,
        common_radii,
        face_maps,
    )
    method_passed = bool(method_passed and moving_parity["passed"])
    for name, values in moving_arrays.items():
        decisive[f"moving_projector_parity__{name}"] = values

    stride_report, recovery_reports = _stride_report(
        histories,
        common_radii,
        face_maps,
        face_scales,
    )
    method_passed = bool(method_passed and stride_report["passed"])
    primary_recovery = recovery_reports[1]
    attribution_reports = []
    maximum_attribution_closure = 0.0
    times = histories[LABELS[0]]["times"]
    for surface, radius in enumerate(common_radii):
        face_map = {
            label: int(face_maps[label][surface])
            for label in LABELS
        }
        report, arrays = _surface_attribution(
            histories,
            face_map,
            times,
            face_scales,
        )
        report = {
            "surface": surface,
            "radius_over_rg": float(radius),
            "face_indices": face_map,
            **report,
        }
        attribution_reports.append(report)
        maximum_attribution_closure = max(
            maximum_attribution_closure,
            report["maximum_complete_explanatory_closure_defect"],
        )
        for name, values in arrays.items():
            decisive[f"surface_{surface}__{name}"] = values
    method_passed = bool(
        method_passed
        and maximum_attribution_closure
        <= MAXIMUM_CONTROL_VOLUME_CLOSURE_DEFECT
    )

    recovery_index = primary_recovery["recovery_surface_index"]
    recovery_radius = primary_recovery["recovery_radius_over_rg"]
    selected_group, group_start = _stable_group(attribution_reports)
    if not method_passed:
        branch = "METHOD_GATE_FAILED"
        authorized_next = "none"
    elif recovery_index is not None:
        branch = "A_recovery_before_coupling"
        authorized_next = "conservative_extraction_surface_audit"
    elif selected_group == "inner_boundary":
        branch = "B_stable_boundary_contribution"
        authorized_next = "outgoing_source_balanced_half_cell_audit"
    elif selected_group == "mapped_anchor_storage":
        branch = "C_stable_storage_anchor_contribution"
        authorized_next = "self_consistent_candidate_tangent_audit"
    elif selected_group in {
        "height_space_storage",
        "stress_principal_relaxation",
        "lower_sources",
    }:
        branch = "E_stable_principal_or_lower_source_contribution"
        authorized_next = "targeted_source_path_consistency_audit"
    else:
        branch = "D_no_recovery_or_stable_non_target_mechanism"
        authorized_next = "monolithic_conservative_space_storage_dae"

    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DECISIVE_ARRAYS, **decisive)
    source_hashes, source_manifest = _source_manifest()
    gates = {
        "minimum_recovery_order": MINIMUM_RECOVERY_ORDER,
        "maximum_fine_normalized_difference": (
            MAXIMUM_FINE_NORMALIZED_DIFFERENCE
        ),
        "minimum_history_cosine": MINIMUM_HISTORY_COSINE,
        "minimum_error_cosine": MINIMUM_ERROR_COSINE,
        "minimum_relative_activity": MINIMUM_RELATIVE_ACTIVITY,
        "required_consecutive_surfaces": REQUIRED_CONSECUTIVE_SURFACES,
        "maximum_analytic_generator_replay_defect": (
            MAXIMUM_ANALYTIC_GENERATOR_REPLAY_DEFECT
        ),
        "maximum_prefix_face_parity_defect": (
            MAXIMUM_PREFIX_FACE_PARITY_DEFECT
        ),
        "maximum_moving_projector_face_parity_defect": (
            MAXIMUM_MOVING_PROJECTOR_FACE_PARITY_DEFECT
        ),
        "maximum_control_volume_closure_defect": (
            MAXIMUM_CONTROL_VOLUME_CLOSURE_DEFECT
        ),
        "maximum_stride_defect": MAXIMUM_STRIDE_DEFECT,
        "minimum_group_target_aligned_fraction": (
            MINIMUM_GROUP_TARGET_ALIGNED_FRACTION
        ),
        "maximum_group_fixed_coefficient_residual": (
            MAXIMUM_GROUP_FIXED_COEFFICIENT_RESIDUAL
        ),
        "minimum_group_subspace_cosine": (
            MINIMUM_GROUP_SUBSPACE_COSINE
        ),
    }
    config = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "labels": LABELS,
        "path_quadrature_order": PATH_QUADRATURE_ORDER,
        "reconstruction_halo_cells": RECONSTRUCTION_HALO_CELLS,
        "stride_audits": STRIDE_AUDITS,
        "explanatory_terms": EXPLANATORY_TERMS,
        "groups": GROUPS,
        "target_definition": (
            "direct analytic outer-face refinement difference"
        ),
        "target_excluded_from_explanatory_groups": True,
        "gates": gates,
    }
    _write_json(CONFIG_PATH, config)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        **identity,
        "classification": branch,
        "authorized_next_work": authorized_next,
        "method_passed": method_passed,
        "parent_wp10c9d5c0f_summary_path": _relative(PARENT_SUMMARY),
        "parent_wp10c9d5c0f_summary_sha256": _sha256(PARENT_SUMMARY),
        "parent_wp10c9d5_candidate_remains_rejected": True,
        "parent_wp10c9d5b_branch_d_preserved_as_input": True,
        "analytic_tangent_physical_sensitivity_remains_passed": bool(
            parent["passed"]
        ),
        "last_common_face_radius_over_rg": float(common_radii[-1]),
        "common_surface_count": int(common_radii.size),
        "grid_method_reports": grid_reports,
        "moving_projector_face_parity": moving_parity,
        "stride_report": stride_report,
        "recovery_reports_by_stride": {
            str(stride): report
            for stride, report in recovery_reports.items()
        },
        "recovery_surface_index": recovery_index,
        "recovery_radius_over_rg": recovery_radius,
        "attribution_target": (
            "direct outer-face refinement error; target excluded from all "
            "groups"
        ),
        "attribution_reports": attribution_reports,
        "maximum_attribution_closure_defect": (
            maximum_attribution_closure
        ),
        "stable_explanatory_group": selected_group,
        "stable_group_start_surface": group_start,
        "conservative_extraction_surface_audit_authorized": bool(
            method_passed and branch == "A_recovery_before_coupling"
        ),
        "boundary_half_cell_audit_authorized": bool(
            method_passed and branch == "B_stable_boundary_contribution"
        ),
        "self_consistent_tangent_audit_authorized": bool(
            method_passed
            and branch == "C_stable_storage_anchor_contribution"
        ),
        "targeted_source_path_audit_authorized": bool(
            method_passed
            and branch
            == "E_stable_principal_or_lower_source_contribution"
        ),
        "monolithic_replacement_authorized": bool(
            method_passed
            and branch == "D_no_recovery_or_stable_non_target_mechanism"
        ),
        "frozen_candidate_recertification_authorized": False,
        "production_operator_authorized": False,
        "nonlinear_candidate_authorized": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "decisive_arrays_path": _relative(DECISIVE_ARRAYS),
        "decisive_arrays_sha256": _sha256(DECISIVE_ARRAYS),
        "decisive_array_hashes": {
            name: _array_sha256(values)
            for name, values in decisive.items()
        },
        "implementation_source_hashes": source_hashes,
        "implementation_source_manifest_sha256": source_manifest,
        "environment": wp10c9d5c0._environment(),
        "runtime_seconds": time.perf_counter() - started,
    }
    _write_json(SUMMARY_PATH, summary)
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        **identity,
        "generation_command": (
            "PYTHONPATH=src python3 "
            "scripts/run_causal_inner_extended_localization_wp10c9d5c1.py"
        ),
        "source_parent_commit": ANALYZED_BASE_COMMIT,
        "parent_canonical_hashes": {
            _relative(path): _sha256(path)
            for path in (
                PARENT_SUMMARY,
                PARENT_DECISIVE_ARRAYS,
                C0E_SUMMARY,
                C0E_DECISIVE_ARRAYS,
            )
        },
        "implementation_source_hashes": source_hashes,
        "implementation_source_manifest_sha256": source_manifest,
        "scientific_status": (
            "CERTIFIED" if method_passed else "REJECTED"
        ),
        "authorization_status": authorized_next,
        "establishes": (
            "Whether direct M/J/E exports recover before coupling and "
            "whether one proper subset of non-target control-volume terms "
            "stably predicts the outer-export refinement error."
        ),
        "does_not_establish": (
            "Causality of an associated group, a repaired operator, "
            "nonlinear convergence, fixed-Q closure, or reduced evolution."
        ),
    }
    _write_json(PROVENANCE_PATH, provenance)
    _refresh_sha256s(CANONICAL_DIRECTORY)
    return summary


def main() -> None:
    print(json.dumps(_plain(run()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
