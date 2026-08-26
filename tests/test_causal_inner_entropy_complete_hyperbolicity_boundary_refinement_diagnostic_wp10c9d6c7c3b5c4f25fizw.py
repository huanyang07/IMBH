import numpy as np

import run_causal_inner_entropy_complete_hyperbolicity_boundary_refinement_diagnostic_wp10c9d6c7c3b5c4f25fizw as diagnostic


def test_manifest_authorizes_exactly_three_nonpropagating_truth_probes():
    validated = diagnostic._validate_manifest(require_clean=False)
    scope = validated["contract"]["diagnostic_scope"]
    assert scope["nonpropagating"]
    assert scope["maximum_new_truth_operator_calls"] == 3


def test_imaginary_ratio_uses_the_declared_relative_scale():
    values = np.asarray([2.0 + 2.0e-9j, -1.0, 0.0])
    assert np.isclose(diagnostic._imaginary_ratio(values), 1.0e-9)


def test_outcomes_separate_overshoot_from_persistent_boundary():
    assert diagnostic.OVERSHOOT_CLASSIFICATION != diagnostic.BOUNDARY_CLASSIFICATION
    assert diagnostic.BOUNDARY_CLASSIFICATION != diagnostic.METHOD_CLASSIFICATION


def test_only_overshoot_branch_names_a_recovery_manifest():
    assert "event_aware_hyperbolicity_retry" in diagnostic.AUTHORIZED_NEXT
