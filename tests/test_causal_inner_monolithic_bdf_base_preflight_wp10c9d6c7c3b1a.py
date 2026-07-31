from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = (
    ROOT
    / "results/canonical/"
    "causal_inner_monolithic_bdf_base_preflight_wp10c9d6c7c3b1a"
)


def _json(name: str) -> dict:
    return json.loads((DIRECTORY / name).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_c3b1a_classification_and_authorization() -> None:
    summary = _json("summary.json")
    assert summary["passed"]
    assert (
        summary["classification"]
        == "monolithic_bdf_base_method_preflight_certified_"
        "full_profile_variant_preflight_authorized"
    )
    assert (
        summary["authorized_next"]
        == "WP10c9d6c7c3b1b_full_profile_variant_method_preflight"
    )
    assert summary["full_profile_variant_method_preflight_authorized"]
    assert not summary["long_nonlinear_physical_ladder_authorized"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]


def test_c3b1a_method_gates_and_replay_pass() -> None:
    summary = _json("summary.json")
    config = _json("config.json")
    gates = config["gates"]
    assert (
        summary["maximum_scaled_residual"]
        <= gates["maximum_scaled_residual"]
    )
    assert (
        summary["maximum_discrete_ledger_defect"]
        <= gates["maximum_discrete_ledger_defect"]
    )
    assert (
        summary["maximum_dense_colored_jacobian_defect"]
        <= gates["maximum_dense_colored_jacobian_defect"]
    )
    assert (
        summary["maximum_independent_jvp_defect"]
        <= gates["maximum_independent_jvp_defect"]
    )
    assert summary["all_restart_roundtrips_bitwise"]
    assert summary["all_split_replays_bitwise"]
    assert len(summary["base_layout_reports"]) == 3
    assert all(row["passed"] for row in summary["base_layout_reports"])


def test_c3b1a_complete_storage_history_is_committed() -> None:
    with np.load(DIRECTORY / "decisive_arrays.npz", allow_pickle=False) as data:
        for label in (
            "N128_exterior_N128_inner_c48",
            "N128_exterior_N256_inner_c48",
            "N128_exterior_N512_inner_c48",
        ):
            assert data[f"{label}__states"].shape[0] == 5
            assert data[f"{label}__scaled_residuals"].shape == (4,)
            assert data[f"{label}__mapped_endpoint_path_closures"].shape == (
                4,
            )


def test_c3b1a_checksums_close() -> None:
    lines = (
        DIRECTORY / "SHA256SUMS.txt"
    ).read_text(encoding="utf-8").splitlines()
    for line in lines:
        digest, name = line.split("  ", 1)
        assert _sha256(DIRECTORY / name) == digest
