import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = (
    ROOT
    / "scripts"
    / "run_causal_inner_face36_fixed_q_authentic_history_ladder_"
    "wp10c9d6c7c3b5c4f24e1.py"
)


def _module():
    specification = importlib.util.spec_from_file_location("c4f24e1", RUNNER)
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    specification.loader.exec_module(module)
    return module


def test_c4f24e1_freezes_both_states_before_refining() -> None:
    module = _module()
    assert module.CASE_ORDER == (
        "primary_coarse",
        "heldout_coarse",
        "primary_middle",
        "heldout_middle",
        "primary_fine",
        "heldout_fine",
    )
    assert module.TIMESTEPS == (1.0e-7, 5.0e-8, 2.5e-8)


def test_c4f24e1_preserves_binding_method_gates() -> None:
    module = _module()
    gates = module.GATES
    assert gates["maximum_scaled_residual"] == 1.0e-10
    assert gates["maximum_Q3_relative_defect"] == 1.0e-12
    assert gates["maximum_storage_parity_relative_defect"] == 1.0e-9
    assert gates["minimum_path_reconstruction_factor"] == 1.0 - 1.0e-12
    assert gates["maximum_complete_Jacobian_assemblies"] == 1
    assert gates["minimum_state_rate_convergence_order"] == 0.9
    assert gates["minimum_reaction_action_convergence_order"] == 0.9


def test_c4f24e1_keeps_reduction_blocked_until_finalize_passes() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert '"fixed_Q_micro_solver_authorized": False' in source
    assert '"reduced_slow_evolution_authorized": False' in source
    assert "causal_five_field_fixed_q_bdf_restart" in source
    assert "causal_five_field_fixed_q_bdf_restarts_equal" in source
