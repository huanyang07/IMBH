#!/usr/bin/env python3
"""Execute the cost-bounded middle continuation from 2 to 5 ms."""

from __future__ import annotations

import json
import math
import platform
from pathlib import Path
import sys

import numpy as np
import scipy


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_causal_inner_nonlinear_middle_1ms_continuation_wp10c9d6c7c3b5c3h2b1 as h2b1  # noqa: E402
import run_causal_inner_nonlinear_middle_2ms_continuation_wp10c9d6c7c3b5c3h2c1 as h2c1  # noqa: E402
import run_causal_inner_nonlinear_middle_5ms_completion_manifest_wp10c9d6c7c3b5c3h2d0 as h2d0  # noqa: E402


SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c3h2d1"
ANALYZED_BASE_COMMIT = "ccc98c1333b0e2a2af7460cfc9143b4c35c04bb4"
TARGET_MICROSECONDS = tuple(h2d0.TARGET_MICROSECONDS)
SAMPLED_INDICES = {0, 3, 7}

ARTIFACT = (
    "causal_inner_nonlinear_middle_5ms_completion_"
    "wp10c9d6c7c3b5c3h2d1"
)
THIS_RUNNER = (
    "scripts/run_causal_inner_nonlinear_middle_5ms_completion_"
    "wp10c9d6c7c3b5c3h2d1.py"
)
THIS_TEST = (
    "tests/test_causal_inner_nonlinear_middle_5ms_completion_"
    "wp10c9d6c7c3b5c3h2d1.py"
)
REPORT_RELATIVE = (
    "docs/reports/current/CODEX_CAUSAL_INNER_NONLINEAR_MIDDLE_5MS_"
    "COMPLETION_WP10C9D6C7C3B5C3H2D1_2026-08-06.md"
)
REPORT_PATH = ROOT / REPORT_RELATIVE
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
CONFIG_PATH = CANONICAL_DIRECTORY / "config.json"
SUMMARY_PATH = CANONICAL_DIRECTORY / "summary.json"
PROVENANCE_PATH = CANONICAL_DIRECTORY / "provenance.json"
DECISIVE_ARRAYS = CANONICAL_DIRECTORY / "decisive_arrays.npz"
CHECKPOINT_DIRECTORY = ROOT / "outputs/checkpoints" / ARTIFACT
PROGRESS_PATH = CHECKPOINT_DIRECTORY / "progress.json"
BASE_PATH = CHECKPOINT_DIRECTORY / "base.npz"
TANGENT_PATH = CHECKPOINT_DIRECTORY / "tangent.npz"
ANCHOR_PATH = CHECKPOINT_DIRECTORY / "anchor.npz"


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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _source_identity() -> dict[str, str]:
    return {
        path: h2c1._sha256(ROOT / path)
        for path in (
            THIS_RUNNER,
            THIS_TEST,
            h2b1.CONTROLLER_RELATIVE,
            h2b1.MODULE_RELATIVE,
        )
        if (ROOT / path).exists()
    }


def _validate_parent() -> None:
    manifest = _read_json(h2d0.SUMMARY_PATH)
    parent = _read_json(h2c1.SUMMARY_PATH)
    full_commit = h2c1._git_value("rev-parse", ANALYZED_BASE_COMMIT)
    if (
        not manifest["passed"]
        or not manifest["middle_5ms_propagation_authorized"]
        or not parent["passed"]
        or not parent["middle_5ms_completion_manifest_authorized"]
    ):
        raise RuntimeError("h2d1 authorization changed")
    if full_commit != ANALYZED_BASE_COMMIT:
        raise RuntimeError("h2d1 analyzed identity changed")


def _load_parent_arrays() -> dict[str, np.ndarray]:
    with np.load(h2c1.DECISIVE_ARRAYS, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _patch_shared_modules() -> None:
    for module in (h2b1, h2c1):
        module.WORK_PACKAGE = WORK_PACKAGE
        module.ARTIFACT = ARTIFACT
        module.CHECKPOINT_DIRECTORY = CHECKPOINT_DIRECTORY
        module.PROGRESS_PATH = PROGRESS_PATH
        module.BASE_PATH = BASE_PATH
        module.TANGENT_PATH = TANGENT_PATH
        module.ANCHOR_PATH = ANCHOR_PATH
        module._source_identity = _source_identity
    h2b1.TARGET_MICROSECONDS = TARGET_MICROSECONDS
    h2b1.START_SECONDS = 2.0e-3
    h2b1.STOP_SECONDS = 5.0e-3
    h2b1._anchor_sample_indices = lambda _base: set(SAMPLED_INDICES)
    h2b1._tangent_audit_indices = lambda _base: set(SAMPLED_INDICES)
    h2c1._load_parent_arrays = _load_parent_arrays
    h2c1.CANONICAL_DIRECTORY = CANONICAL_DIRECTORY
    h2c1.CANONICAL_MANIFEST = ROOT / "results/manifests/canonical_artifacts.csv"
    h2c1.CANONICAL_SUMMARY = ROOT / "results/manifests/canonical_summary.json"
    h2c1.ANALYZED_BASE_COMMIT = h2c1._git_value("rev-parse", ANALYZED_BASE_COMMIT)


def main() -> int:
    _validate_parent()
    _patch_shared_modules()
    progress = h2c1._seed_checkpoints()
    configuration = h2b1._configuration()
    print("h2d1: build frozen nonlinear tangent", flush=True)
    frozen_tangent, setup_seconds = h2b1._build_frozen_tangent(configuration)
    with np.load(h2c1.DECISIVE_ARRAYS, allow_pickle=False) as parent:
        field_scales = np.asarray(parent["tangent__field_scales"], dtype=float)
        export_scales = np.asarray(parent["tangent__export_scales"], dtype=float)
    contract, _strict = h2b1.h2a2.h2.g._controller_contracts()
    base_report, base = h2b1._run_base_targets(
        progress, configuration, frozen_tangent, field_scales, export_scales, contract
    )
    tangent_report, tangent = h2b1._run_tangent(progress, configuration, base)
    anchor_report, anchor = h2b1._run_anchor(
        progress,
        configuration,
        frozen_tangent,
        base,
        tangent,
        field_scales,
        export_scales,
        contract,
    )
    replays = {
        "base": h2b1._serialized_last_step_replay(
            "base",
            configuration,
            frozen_tangent,
            base["accepted_states"],
            base["accepted_primitive_histories"],
            base["accepted_mapped_histories"],
            base["accepted_height_histories"],
            base["accepted_previous_timesteps"],
            base["accepted_timesteps"],
            base["accepted_times"],
            None,
        ),
        "anchor": h2b1._serialized_last_step_replay(
            "anchor",
            configuration,
            frozen_tangent,
            anchor["anchor_states"],
            anchor["anchor_primitive_histories"],
            anchor["anchor_mapped_histories"],
            anchor["anchor_height_histories"],
            anchor["anchor_previous_timesteps"],
            base["accepted_timesteps"],
            base["accepted_times"],
            anchor["anchor_predictors"][-1],
        ),
    }
    replay_passed = all(
        item["checkpoint_roundtrip_bitwise"]
        and item["last_step_replay_bitwise"]
        and item["maximum_scaled_residual"] <= 1.0e-10
        for item in replays.values()
    )
    surrogate_passed = bool(
        anchor_report["state"]["discrepancy_fraction_of_observable_response"]
        <= 0.01
        and anchor_report["instantaneous_Tier_I"][
            "discrepancy_fraction_of_observable_response"
        ]
        <= 0.01
        and anchor_report["cumulative_Tier_I"][
            "discrepancy_fraction_of_observable_response"
        ]
        <= 0.01
    )
    passed = bool(
        base_report["passed"]
        and tangent_report["passed"]
        and anchor_report["passed"]
        and replay_passed
        and surrogate_passed
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "work_package": WORK_PACKAGE,
        "classification": (
            "middle_5ms_completion_passed_spatial_analysis_and_fine_manifest_authorized"
            if passed
            else "middle_5ms_completion_failed_fine_work_blocked"
        ),
        "passed": passed,
        "setup_wall_seconds": setup_seconds,
        "base": base_report,
        "tangent": tangent_report,
        "anchor": anchor_report,
        "serialized_replays": replays,
        "surrogate_relative_gate_passed": surrogate_passed,
        "middle_5ms_spatial_analysis_authorized": passed,
        "fine_manifest_authorized": passed,
        "fine_cost_bounded_propagation_authorized": False,
        "middle_fine_5ms_spatial_certificate_issued": False,
        "fixed_q_micro_solver_authorized": False,
        "reduced_slow_evolution_authorized": False,
        "authorized_next": (
            "WP10c9d6c7c3b5c3h2e0_middle_spatial_analysis_and_fine_manifest"
            if passed
            else None
        ),
    }
    combined = {
        **{f"base__{key}": value for key, value in base.items()},
        **{f"tangent__{key}": value for key, value in tangent.items()},
        **{f"anchor__{key}": value for key, value in anchor.items()},
    }
    CANONICAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _write_json(
        CONFIG_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "analyzed_base_commit": h2c1._git_value("rev-parse", ANALYZED_BASE_COMMIT),
            "layout": h2b1.MIDDLE_LAYOUT,
            "profiles": h2b1.PROFILES,
            "generic_profile": h2b1.GENERIC_PROFILE,
            "coupling_face": h2b1.COUPLING_FACE,
            "target_microseconds": TARGET_MICROSECONDS,
            "sampled_indices": sorted(SAMPLED_INDICES),
            "controller": contract,
            "surrogate_gates": h2b1.h2a3.h2a1.GATES,
        },
    )
    np.savez_compressed(DECISIVE_ARRAYS, **combined)
    _write_json(SUMMARY_PATH, summary)
    _write_json(
        PROVENANCE_PATH,
        {
            "schema_version": SCHEMA_VERSION,
            "work_package": WORK_PACKAGE,
            "analyzed_base_commit": h2c1._git_value("rev-parse", ANALYZED_BASE_COMMIT),
            "working_head": h2c1._git_value("rev-parse", "HEAD"),
            "runner": THIS_RUNNER,
            "test": THIS_TEST,
            "report": REPORT_RELATIVE,
            "input_hashes": {
                "completion_manifest": h2c1._sha256(h2d0.MANIFEST_PATH),
                "parent_summary": h2c1._sha256(h2c1.SUMMARY_PATH),
                "parent_arrays": h2c1._sha256(h2c1.DECISIVE_ARRAYS),
            },
            "implementation_source_hashes": _source_identity(),
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            },
            "command": f"PYTHONPATH=src:scripts python {THIS_RUNNER}",
        },
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            (
                "# Middle 5 ms completion WP10c9d6c7c3b5c3h2d1",
                "",
                "## Classification",
                "",
                f"`{summary['classification']}`",
                "",
                f"The middle base, generic nonlinear anchor, and five-profile tangent reached 5 ms in `{base_report['accepted_steps']}` new accepted steps with `{base_report['rejected_attempts']}` rejected attempts.",
                "",
                f"The tangent discrepancy is `{anchor_report['state']['discrepancy_fraction_of_observable_response']:.6e}` of the state response and `{anchor_report['instantaneous_Tier_I']['discrepancy_fraction_of_observable_response']:.6e}` of the instantaneous Tier-I response. Base and anchor last-step replays are bitwise.",
                "",
                "A pass authorizes only spatial analysis and a fresh fine campaign manifest. Fixed-Q experiments and reduced slow evolution remain blocked.",
                "",
            )
        ),
        encoding="utf-8",
    )
    names = ("config.json", "decisive_arrays.npz", "provenance.json", "summary.json")
    (CANONICAL_DIRECTORY / "SHA256SUMS.txt").write_text(
        "".join(f"{h2c1._sha256(CANONICAL_DIRECTORY / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    h2c1._update_catalog(summary)
    print(json.dumps(_plain(summary), indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
