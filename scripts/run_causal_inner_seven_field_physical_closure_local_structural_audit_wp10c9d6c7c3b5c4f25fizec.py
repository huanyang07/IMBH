#!/usr/bin/env python3
"""Execute the fail-fast physical seven-field entropy-structure audit.

The prospectively frozen Stage-2 contract requires the nonlinear state and
flux maps to share an exact entropy potential.  Exactness implies that the
entropy-flux one-form ``w(q)^T dF(q)`` is closed.  This runner evaluates that
necessary condition at the committed primary-cell witness before any
eigenvalue campaign or trajectory is allowed.  A failed closure test is a
binding derivation failure, not a numerical trajectory result.
"""

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

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_seven_field_physical_closure_local_audit_manifest_wp10c9d6c7c3b5c4f25fizeb as parent  # noqa: E402
import run_causal_inner_tangent_phase_lap_stage2_hyperbolicity_boundary_diagnostic_wp10c9d6c7c3b5c4f25fizdb as boundary_diagnostic  # noqa: E402
from imri_qpe.constants import C  # noqa: E402
from imri_qpe.layer3_minidisk_1d.causal_inner_seven_field_physical import (  # noqa: E402
    seven_field_physical_state,
)


SCHEMA_VERSION = 1
WORK_PACKAGE = parent.AUTHORIZED_NEXT
CLASSIFICATION = "seven_field_physical_closure_entropy_failed"
AUTHORIZED_NEXT = (
    "definitions_only_WP10c9d6c7c3b5c4f25fized_"
    "generalized_Maxwell_Cattaneo_architecture_manifest"
)
ARTIFACT = (
    "causal_inner_seven_field_physical_closure_local_structural_audit_"
    "wp10c9d6c7c3b5c4f25fizec"
)
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_SEVEN_FIELD_PHYSICAL_CLOSURE_"
    "LOCAL_STRUCTURAL_AUDIT_WP10C9D6C7C3B5C4F25FIZEC_2026-08-26.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
THIS_RUNNER = (
    "scripts/run_causal_inner_seven_field_physical_closure_local_"
    "structural_audit_wp10c9d6c7c3b5c4f25fizec.py"
)
THIS_TEST = (
    "tests/test_causal_inner_seven_field_physical_closure_local_"
    "structural_audit_wp10c9d6c7c3b5c4f25fizec.py"
)
PHYSICAL_SOURCE = (
    "src/imri_qpe/layer3_minidisk_1d/causal_inner_seven_field_physical.py"
)
PHYSICAL_TEST = "tests/test_causal_inner_seven_field_physical.py"
CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"

REPRESENTATIVE_PROFILE = "primary_20ms_base_charts5"
REPRESENTATIVE_CELL = 36
STEP_FACTORS = (2.0, 1.0, 0.5)


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _validate_parent(*, require_clean: bool) -> dict:
    hashes = parent._validate_checksums(parent.CANONICAL_DIRECTORY)
    summary = _read_json(parent.CANONICAL_DIRECTORY / "summary.json")
    contract = _read_json(
        parent.CANONICAL_DIRECTORY / "local_audit_manifest.json"
    )
    provenance = _read_json(parent.CANONICAL_DIRECTORY / "provenance.json")
    if (
        summary["classification"] != parent.CLASSIFICATION
        or not summary["passed"]
        or not summary["definitions_only"]
        or not summary["local_structural_audit_authorized"]
        or summary["authorized_next"] != WORK_PACKAGE
        or contract["authorized_next"] != WORK_PACKAGE
        or not contract["binding_local_audit_gates"]["fail_closed"]
        or not contract["physical_entropy_extension"][
            "no_post_hoc_matrix_symmetrization"
        ]
    ):
        raise RuntimeError("Stage-2 local-audit authorization changed")
    for relative, expected in provenance["source_hashes"].items():
        if _sha256(ROOT / relative) != expected:
            raise RuntimeError(f"Stage-2 source changed: {relative}")
    if require_clean and _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("structural audit requires a clean tracked tree")
    return {
        "hashes": hashes,
        "summary": summary,
        "contract": contract,
        "provenance": provenance,
    }


def _sixth_order_centered_jacobian(
    function,
    chart: np.ndarray,
    steps: np.ndarray,
):
    center = np.asarray(function(chart), dtype=float)
    jacobian = np.empty((center.size, chart.size), dtype=float)
    for column, step in enumerate(steps):
        direction = np.zeros_like(chart)
        direction[column] = step
        jacobian[:, column] = (
            -np.asarray(function(chart - 3.0 * direction))
            + 9.0 * np.asarray(function(chart - 2.0 * direction))
            - 45.0 * np.asarray(function(chart - direction))
            + 45.0 * np.asarray(function(chart + direction))
            - 9.0 * np.asarray(function(chart + 2.0 * direction))
            + np.asarray(function(chart + 3.0 * direction))
        ) / (60.0 * step)
    return center, jacobian


def _base_steps(chart: np.ndarray, stress_scale: float) -> np.ndarray:
    return np.asarray(
        [
            2.0e-5,
            2.0e-6,
            2.0e-6,
            2.0e-5,
            2.0e-4 * max(abs(float(chart[4])), stress_scale),
            2.0e-5,
            2.0e-6,
        ],
        dtype=float,
    )


def _entropy_flux_one_form(
    evaluator,
    chart: np.ndarray,
    steps: np.ndarray,
    *,
    state_scales: np.ndarray,
    entropy_scale: float,
) -> tuple[np.ndarray, float]:
    _state, state_jacobian = _sixth_order_centered_jacobian(
        lambda values: evaluator(values).conserved / state_scales,
        chart,
        steps,
    )
    _flux, flux_jacobian = _sixth_order_centered_jacobian(
        lambda values: evaluator(values).flux_over_c / state_scales,
        chart,
        steps,
    )
    _entropy, entropy_gradient = _sixth_order_centered_jacobian(
        lambda values: np.atleast_1d(
            evaluator(values).mathematical_entropy / entropy_scale
        ),
        chart,
        steps,
    )
    condition = float(np.linalg.cond(state_jacobian))
    entropy_variables = np.linalg.solve(
        state_jacobian.T,
        entropy_gradient.ravel(),
    )
    return entropy_variables @ flux_jacobian, condition


def _audit() -> tuple[dict, dict[str, np.ndarray]]:
    validated = _validate_parent(require_clean=False)
    envelope_path = parent.CANONICAL_DIRECTORY / "audit_envelope.npz"
    with np.load(envelope_path, allow_pickle=False) as archive:
        chart5 = np.asarray(
            archive[REPRESENTATIVE_PROFILE][REPRESENTATIVE_CELL],
            dtype=float,
        )

    inputs = boundary_diagnostic.manifest.parent.engine.execution.source._initial_inputs()
    context = inputs["base"]["configuration"]["context"]
    radius = float(context.grid.centers[REPRESENTATIVE_CELL])
    old_state = boundary_diagnostic.radial._cell_state(context, radius, chart5)
    omega = float(context.vertical_frequency.frequency(radius))
    chart7 = np.concatenate(
        (
            chart5,
            [np.log(old_state.thermodynamics.proper_half_thickness), 0.0],
        )
    )

    def evaluator(values):
        return seven_field_physical_state(
            old_state.geometry,
            values,
            proper_vertical_frequency=omega,
            alpha=context.alpha,
            stress_factor=context.stress_factor,
        )

    base = evaluator(chart7)
    state_scales = np.maximum(np.abs(base.conserved), 1.0)
    rest_mass = float(base.conserved[0])
    state_scales[4] = max(
        abs(float(base.conserved[4])),
        rest_mass * base.calibration.equilibrium_specific_stress,
        np.finfo(float).tiny,
    )
    state_scales[5] = max(
        abs(float(base.conserved[5])),
        np.finfo(float).tiny,
    )
    state_scales[6] = max(
        rest_mass * C * parent.VERTICAL_VELOCITY_OVER_C_STENCIL[-1],
        np.finfo(float).tiny,
    )
    entropy_scale = max(abs(base.mathematical_entropy), 1.0)
    base_steps = _base_steps(
        chart7,
        base.calibration.equilibrium_specific_stress,
    )
    rows = []
    curls = []
    one_forms = []
    derivatives = []
    conditions = []
    for factor in STEP_FACTORS:
        steps = factor * base_steps

        def one_form(values):
            value, _condition = _entropy_flux_one_form(
                evaluator,
                values,
                steps,
                state_scales=state_scales,
                entropy_scale=entropy_scale,
            )
            return value

        value, derivative = _sixth_order_centered_jacobian(
            one_form,
            chart7,
            2.0 * steps,
        )
        _unused, condition = _entropy_flux_one_form(
            evaluator,
            chart7,
            steps,
            state_scales=state_scales,
            entropy_scale=entropy_scale,
        )
        curl = derivative - derivative.T
        relative = float(
            np.linalg.norm(curl)
            / max(np.linalg.norm(derivative), np.finfo(float).tiny)
        )
        maximum = np.unravel_index(np.argmax(np.abs(curl)), curl.shape)
        rows.append(
            {
                "step_factor": factor,
                "relative_entropy_flux_curl_defect": relative,
                "maximum_absolute_curl_component": float(
                    abs(curl[maximum])
                ),
                "maximum_curl_pair": list(maximum),
                "state_jacobian_condition": condition,
            }
        )
        curls.append(curl)
        one_forms.append(value)
        derivatives.append(derivative)
        conditions.append(condition)

    threshold = validated["contract"]["binding_local_audit_gates"][
        "A1_relative_symmetry_defect_max"
    ]
    minimum_defect = min(
        item["relative_entropy_flux_curl_defect"] for item in rows
    )
    stable_obstruction = bool(
        min(item["relative_entropy_flux_curl_defect"] for item in rows) > 1.0e-1
        and max(item["relative_entropy_flux_curl_defect"] for item in rows)
        / minimum_defect
        < 1.1
        and all(item["maximum_curl_pair"] == [3, 4] for item in rows)
    )
    entropy_passed = bool(
        all(
            item["relative_entropy_flux_curl_defect"] <= threshold
            for item in rows
        )
    )
    if entropy_passed or not stable_obstruction:
        raise RuntimeError(
            "the prospectively expected entropy obstruction did not reproduce"
        )

    calibration = base.calibration
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "audit_completed": True,
        "scientific_passed": False,
        "entropy_integrability_passed": False,
        "fail_fast_before_eigenvalue_campaign": True,
        "representative_profile": REPRESENTATIVE_PROFILE,
        "representative_cell": REPRESENTATIVE_CELL,
        "representative_radius_cm": radius,
        "proper_vertical_frequency_per_second": omega,
        "step_ladder": rows,
        "binding_relative_symmetry_gate": threshold,
        "minimum_relative_entropy_flux_curl_defect": minimum_defect,
        "stable_order_unity_obstruction": stable_obstruction,
        "dominant_obstruction_coordinates": [
            "log_temperature",
            "specific_shear_stress",
        ],
        "reservoir_calibration": {
            "viscous_signal_speed_over_c": (
                calibration.viscous_signal_speed_over_c
            ),
            "reservoir_coefficient": calibration.reservoir_coefficient,
            "extended_specific_enthalpy_over_c2": (
                calibration.extended_specific_enthalpy_over_c2
            ),
            "calibration_relative_defect": abs(
                calibration.reservoir_coefficient
                - calibration.extended_specific_enthalpy_over_c2
                * calibration.viscous_signal_speed_over_c**2
            )
            / calibration.reservoir_coefficient,
        },
        "failure_scope": (
            "candidate nonlinear D*chi conservative/Godunov realization; "
            "not the Stage-1 local normal form and not a trajectory"
        ),
        "likely_missing_structure": (
            "thermodynamic nonlinear terms associated with the state-dependent "
            "shear modulus, or a conjugate shear-strain variable"
        ),
        "new_trajectory_steps": 0,
        "new_nonlinear_roots": 0,
        "failed_candidate_propagated": False,
        "authorized_next": AUTHORIZED_NEXT,
    }
    arrays = {
        "representative_chart5": chart5,
        "representative_chart7": chart7,
        "base_conserved7": base.conserved,
        "base_flux7_over_c": base.flux_over_c,
        "primitive_steps7": base_steps,
        "state_scales7": state_scales,
        "entropy_flux_one_forms": np.stack(one_forms),
        "entropy_flux_one_form_derivatives": np.stack(derivatives),
        "entropy_flux_curls": np.stack(curls),
        "state_jacobian_conditions": np.asarray(conditions),
    }
    return metrics, arrays


def _source_hashes() -> dict[str, str]:
    paths = (
        THIS_RUNNER,
        THIS_TEST,
        PHYSICAL_SOURCE,
        PHYSICAL_TEST,
        parent.THIS_RUNNER,
        parent.THIS_TEST,
        REPORT_RELATIVE,
    )
    return {path: _sha256(ROOT / path) for path in paths}


def _report(metrics: dict) -> str:
    rows = metrics["step_ladder"]
    ladder = "\n".join(
        f"- `{row['step_factor']:g} h`: curl defect "
        f"`{row['relative_entropy_flux_curl_defect']:.16e}`, maximum pair "
        f"`{row['maximum_curl_pair']}`."
        for row in rows
    )
    return "\n".join(
        (
            "# Seven-field physical closure local structural audit",
            "",
            f"Classification: `{CLASSIFICATION}`.",
            "",
            "The first binding nonlinear entropy test fails. No eigenvalue campaign, spatial discretization, nonlinear root, or trajectory was executed.",
            "",
            "## Result",
            "",
            "For an exact Godunov closure, the entropy-flux one-form `w(q)^T dF(q)` must be closed. At the prospectively selected primary 20 ms cell 36 witness, independent sixth-order-centered derivative ladders give:",
            "",
            ladder,
            "",
            f"The frozen symmetry tolerance is `{metrics['binding_relative_symmetry_gate']:.1e}`. The minimum observed defect is `{metrics['minimum_relative_entropy_flux_curl_defect']:.16e}`. The dominant, step-stable curl component couples log temperature to specific shear stress.",
            "",
            "## Interpretation",
            "",
            "The positive state-local shear reservoir and alpha signal calibration close algebraically, but inserting the resulting state-dependent modulus into a `D chi` balance does not produce the nonlinear thermodynamic terms required for a common entropy potential. Post-hoc matrix symmetrization is forbidden by the Stage-2 contract, so the candidate is rejected fail-fast.",
            "",
            "This does not reject the Stage-1 symmetric local normal form, finite-inertia height dynamics, or a generalized Israel--Stewart/Maxwell--Cattaneo model. It rejects this particular conservative Godunov realization with `R_pi=D chi` and the proposed state-dependent reservoir.",
            "",
            "## Decision",
            "",
            f"Authorized next: `{AUTHORIZED_NEXT}` only. The next definitions package must replace the incompatible stress coordinate/closure with a full nonlinear transient model (or a conjugate shear-strain formulation), freeze exact causality and strong-hyperbolicity gates, and retain the stopped-trajectory boundary. Complete-cycle execution remains unauthorized.",
            "",
        )
    )


def _update_catalog(summary: dict) -> None:
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
                    "sha256": _sha256(path),
                    "scientific_status": "FAILED",
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
    catalog = _read_json(CANONICAL_SUMMARY)
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
            "latest_source_parent_commit": _git("rev-parse", "HEAD"),
            "latest_work_package": WORK_PACKAGE,
        }
    )
    _write_json(CANONICAL_SUMMARY, catalog)


def _execute() -> dict:
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists():
        raise RuntimeError("Stage-3 structural audit is already frozen")
    parent_data = _validate_parent(require_clean=True)
    metrics, arrays = _audit()
    CANONICAL_DIRECTORY.mkdir(parents=True)
    _write_json(CANONICAL_DIRECTORY / "audit_metrics.json", metrics)
    with (CANONICAL_DIRECTORY / "audit_arrays.npz").open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    _write_json(
        CANONICAL_DIRECTORY / "input_lock.json",
        {
            "parent_artifact": parent.ARTIFACT,
            "parent_classification": parent.CLASSIFICATION,
            "parent_hashes": parent_data["hashes"],
            "audit_envelope_sha256": parent_data["hashes"]["audit_envelope.npz"],
            "canonical_sources_only": True,
        },
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": CLASSIFICATION,
        "passed": False,
        "audit_completed": True,
        "scientific_passed": False,
        "entropy_integrability_passed": False,
        "stable_order_unity_obstruction": True,
        "new_trajectory_steps": 0,
        "failed_candidate_propagated": False,
        "seven_field_spatial_discretization_authorized": False,
        "seven_field_trajectory_authorized": False,
        "fixed_Q_invariant_object_authorized": False,
        "slow_flux_atlas_authorized": False,
        "complete_cycle_execution_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "corrective_architecture_manifest_authorized": True,
        "authorized_next": AUTHORIZED_NEXT,
    }
    _write_json(CANONICAL_DIRECTORY / "summary.json", summary)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(_report(metrics), encoding="utf-8")
    _write_json(
        CANONICAL_DIRECTORY / "provenance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "scientific_status": "FAILED",
            "implementation_commit": _git("rev-parse", "HEAD"),
            "implementation_tree": _git("rev-parse", "HEAD^{tree}"),
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "report": REPORT_RELATIVE,
            "source_hashes": _source_hashes(),
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "thread_environment": {
                name: os.environ.get(name, "")
                for name in (
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                )
            },
        },
    )
    names = sorted(path.name for path in CANONICAL_DIRECTORY.iterdir())
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(
            f"{_sha256(CANONICAL_DIRECTORY / name)}  {name}\n"
            for name in names
        ),
        encoding="utf-8",
    )
    _update_catalog(summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--probe", action="store_true")
    arguments = parser.parse_args()
    if arguments.probe:
        metrics, _arrays = _audit()
        print(json.dumps(_plain(metrics), indent=2, sort_keys=True))
        return 0
    if arguments.execute:
        print(json.dumps(_execute(), indent=2, sort_keys=True))
        return 0
    parser.error("choose --probe or --execute")


if __name__ == "__main__":
    raise SystemExit(main())
