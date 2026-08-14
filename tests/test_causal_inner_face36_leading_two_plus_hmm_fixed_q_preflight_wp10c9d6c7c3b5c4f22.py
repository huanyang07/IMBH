import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "results/canonical"
    / "causal_inner_face36_leading_two_plus_hmm_fixed_q_preflight_"
    "wp10c9d6c7c3b5c4f22"
)


def _read(name: str) -> dict:
    return json.loads((ARTIFACT / name).read_text(encoding="utf-8"))


def test_c4f22_passes_both_endpoint_preflights_without_a_trajectory():
    summary = _read("summary.json")
    assert summary["passed"]
    assert summary["analysis_only"]
    assert not summary["trajectory_executed"]
    assert not summary["physical_operator_changed"]
    assert summary["middle"]["passed"]
    assert summary["fine"]["passed"]


def test_c4f22_uses_ledger_reaction_and_passes_KKT_gates():
    summary = _read("summary.json")
    gates = summary["gates"]
    for layout in ("middle", "fine"):
        metrics = summary[layout]
        assert metrics["DQ_M_inverse_BQ_identity_defect"] <= gates[
            "maximum_DQ_M_inverse_BQ_identity_defect"
        ]
        assert metrics["KKT_linear_solve_relative_defect"] <= gates[
            "maximum_KKT_linear_solve_relative_defect"
        ]
        assert metrics["reaction_ledger_relative_defect"] <= gates[
            "maximum_reaction_ledger_relative_defect"
        ]
        assert metrics["reaction_support_relative_defect"] <= gates[
            "maximum_reaction_support_relative_defect"
        ]
        assert metrics["projected_block_solve_relative_defect"] <= gates[
            "maximum_projected_block_solve_relative_defect"
        ]


def test_c4f22_stable_a2_duals_and_face36_maps_pass():
    summary = _read("summary.json")
    gates = summary["gates"]
    for layout in ("middle", "fine"):
        metrics = summary[layout]
        assert metrics["a2_dual_biorthogonality_defect"] <= gates[
            "maximum_a2_dual_biorthogonality_defect"
        ]
        assert metrics["a2_dual_reaction_annihilation_defect"] <= gates[
            "maximum_a2_dual_reaction_annihilation_defect"
        ]
        assert metrics["maximum_face36_five_point_JVP_relative_defect"] <= gates[
            "maximum_face36_directional_JVP_relative_defect"
        ]
        assert metrics["incoming_excision_characteristics"] == 0


def test_c4f22_stores_twenty_four_equal_Q_lifts_and_transient_diagnostics():
    with np.load(ARTIFACT / "decisive_arrays.npz", allow_pickle=False) as arrays:
        assert arrays["middle_equal_Q_lifts"].shape == (560, 24)
        assert arrays["fine_equal_Q_lifts"].shape == (1040, 24)
        assert arrays["middle_state_singular_values"].shape == (4, 24)
        assert arrays["fine_state_singular_values"].shape == (4, 24)
        assert arrays["middle_face36_output_singular_values"].shape == (4, 3)
        assert arrays["fine_face36_output_singular_values"].shape == (4, 3)
        assert arrays["middle_a2_singular_values"].shape == (4, 2)
        assert arrays["fine_a2_singular_values"].shape == (4, 2)


def test_c4f22_does_not_overclaim_frozen_local_diagnostics():
    summary = _read("summary.json")
    assert summary["fixed_Q_KKT_algebra_certified"]
    assert summary["frozen_projected_local_tangent_certified"]
    assert not summary["state_dependent_constrained_tangent_certified"]
    assert not summary["guard_mixing_or_decay_claimed"]
    assert not summary["fixed_Q_micro_solver_authorized"]
    assert not summary["nonlinear_retained_mode_pilot_authorized"]
    assert summary["one_Q_nonlinear_pilot_manifest_authorized"]
    assert not summary["fifty_ms_propagation_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]
    assert summary["raw_face48_export_rejection_preserved"]


def test_c4f22_hashes_and_source_provenance_are_self_consistent():
    entries = {}
    for line in (ARTIFACT / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        digest, name = line.split("  ", maxsplit=1)
        entries[name] = digest
    for name, digest in entries.items():
        assert hashlib.sha256((ARTIFACT / name).read_bytes()).hexdigest() == digest

    provenance = _read("provenance.json")
    for relative, digest in provenance["source_hashes"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest
