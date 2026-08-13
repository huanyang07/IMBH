import hashlib, json
import run_causal_inner_fine_complement_jvp_manifest_wp10c9d6c7c3b5c4f4 as c4f4

def _read(p): return json.loads(p.read_text(encoding="utf-8"))
def test_manifest_is_analysis_only_and_prospective():
    m=c4f4._manifest(); assert m["definitions_only"] and not m["new_trajectory"]
    assert m["direction_is_diagnostic_not_a_physical_lift"]
    assert m["finite_difference_audit"]["maximum_analytic_FD_relative_defect"]==1e-6
    assert m["gates"]["maximum_JVP_fraction_of_actual_middle_fine_transition_difference"]==0.10
    assert not m["observable_memory_propagation_authorized"]
def test_canonical_summary_and_hashes():
    s=_read(c4f4.SUMMARY_PATH);assert s["passed"] and s["definitions_only"]
    assert not s["fixed_Q_micro_solver_authorized"] and not s["reduced_slow_evolution_authorized"]
    for line in (c4f4.CANONICAL_DIRECTORY/"SHA256SUMS.txt").read_text().splitlines():
        expected,name=line.split("  ",1);assert hashlib.sha256((c4f4.CANONICAL_DIRECTORY/name).read_bytes()).hexdigest()==expected
