import hashlib,json,numpy as np
import run_causal_inner_fine_complement_exact_jvp_audit_wp10c9d6c7c3b5c4f5 as c4f5
def _read(p):return json.loads(p.read_text())
def test_method_passes_but_adjacent_observability_blocks_promotion():
    s=_read(c4f5.SUMMARY_PATH);assert not s["passed"] and s["method_gates_passed"]
    assert s["analytic_FD_relative_defect"]<1e-9 and s["FD_step_plateau_relative_change"]<1e-8
    assert s["transition_JVP_fraction_of_actual_middle_fine_difference"]>100
    assert s["maximum_adjacent_face_fraction_of_transition_JVP"]>0.10
    assert s["classification"]=="fine_complement_observable_across_adjacent_faces_localization_required"
def test_reduction_remains_blocked_and_response_preserved():
    s=_read(c4f5.SUMMARY_PATH);assert s["response_certificate_preserved"] and not s["physical_failure_detected"]
    assert not s["absolute_closure_fit_authorized"] and not s["observable_memory_propagation_authorized"]
    assert not s["fixed_Q_micro_solver_authorized"] and not s["reduced_slow_evolution_authorized"]
def test_arrays_and_hashes():
    with np.load(c4f5.DECISIVE_ARRAYS) as a: assert a["analytic_JVP"].shape==(4,3,3) and a["FD_JVP"].shape==(4,3,3,3)
    for line in (c4f5.CANONICAL_DIRECTORY/"SHA256SUMS.txt").read_text().splitlines():
        expected,name=line.split("  ",1);assert hashlib.sha256((c4f5.CANONICAL_DIRECTORY/name).read_bytes()).hexdigest()==expected
