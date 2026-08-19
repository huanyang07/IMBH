from __future__ import annotations

import run_causal_inner_equilibrium_centered_hybrid_architecture_audit_wp10c9d6c7c3b5c4f25ao as f25ao


def test_frozen_manifest_is_locked():
    frozen = f25ao._validate_manifest(require_clean=False)
    assert frozen["summary"]["passed"]
    assert frozen["summary"]["authorized_next"] == f25ao.WORK_PACKAGE


def test_conditional_branch_dimensions_and_coordinate_ranks():
    metrics = f25ao._dimension_and_coordinate_audit()
    assert metrics["all_dimension_identities_exact"]
    assert metrics["both_resolved_coordinate_maps_full_rank"]
    assert metrics["conditional_branch_unknown_count"] == 560
    assert metrics["conditional_branch_equation_count"] == 560


def test_finite_volume_fluxes_telescope_globally():
    metrics, _ = f25ao._finite_volume_audit()
    assert metrics["incidence_rank"] == 32
    assert metrics["global_telescoping_relative_defect"] <= 1.0e-14


def test_actual_minimum_norm_reset_geometry_preserves_constraints():
    metrics, _ = f25ao._reset_geometry_audit()
    for anchor in ("primary", "heldout"):
        assert metrics[anchor]["constraint_normal_identity_defect"] <= 1.0e-12
        assert metrics[anchor]["projector_constraint_defect"] <= 1.0e-12
        assert metrics[anchor]["reset_constraint_relative_defect"] <= 1.0e-12
        assert metrics[anchor]["minimum_norm_jump_norm"] <= metrics[anchor]["augmented_jump_norm"]


def test_inherited_stable_descriptor_is_energy_contracting_at_macrostep():
    metrics, _ = f25ao._descriptor_energy_audit()
    assert metrics["maximum_energy_amplification_factor"] <= 1.0
    assert metrics["maximum_spectral_abscissa_per_second"] < 0.0
    assert metrics["legacy_coefficients_are_promoted_to_branch_closure"] is False
