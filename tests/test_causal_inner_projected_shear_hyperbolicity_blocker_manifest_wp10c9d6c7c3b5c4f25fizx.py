import run_causal_inner_projected_shear_hyperbolicity_blocker_manifest_wp10c9d6c7c3b5c4f25fizx as manifest


def test_refined_boundary_is_preserved():
    validated = manifest._validate_parent(require_clean=False)
    assert not validated["summary"]["passed"]
    assert validated["metrics"]["accepted_endpoint_hyperbolic"]


def test_derivative_ladder_spans_factor_sixty_four():
    factors = manifest._contract()["independent_derivative_ladder"][
        "derivative_step_factors"
    ]
    assert factors[-1] / factors[0] == 64.0


def test_selected_extension_has_five_shear_amplitudes_and_eleven_fields():
    architecture = manifest._contract()["selected_next_architecture_if_certified"]
    assert "five independent" in architecture["dissipative_extension"]
    assert architecture["total_local_field_count"] == 11
    assert not architecture["project_to_one_Rphi_amplitude"]


def test_cycle_remains_blocked():
    assert not manifest._contract()["claim_boundary"][
        "complete_cycle_execution_authorized"
    ]
