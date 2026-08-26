import run_causal_inner_entropy_complete_hyperbolicity_boundary_refinement_manifest_wp10c9d6c7c3b5c4f25fizv as manifest


def test_parent_rejection_is_hash_validated_and_preserved():
    validated = manifest._validate_parent(require_clean=False)
    assert not validated["summary"]["passed"]
    assert validated["metrics"]["physical_failure"]


def test_probe_ladder_is_bounded_and_nonpropagating():
    scope = manifest._contract()["diagnostic_scope"]
    assert scope["nonpropagating"]
    assert scope["probe_timestep_seconds"] == (5.0e-4, 2.5e-4, 1.25e-4)
    assert scope["maximum_new_truth_operator_calls"] == 3


def test_refined_truth_probes_are_binding():
    gates = manifest._contract()["binding_gates"]
    assert gates["coarse_0p5ms_nonreal_face_reproduced"]
    assert gates["refined_0p25ms_full_truth_operator_passes"]
    assert gates["refined_0p125ms_full_truth_operator_passes"]


def test_cycle_execution_remains_unauthorized():
    assert not manifest._contract()["claim_boundary"][
        "complete_cycle_execution_authorized"
    ]
