import hashlib,json,pytest
import run_causal_inner_equilibrium_selected_metric_stencil_rerun_manifest_wp10c9d6c7c3b5c4f25fizzc5 as target
def test_selected_factor_and_gates_are_frozen():
 target._validate();c=target._contract();assert c["selected_step_factor"]==.5;assert c["same_47_witnesses"];assert c["unchanged_gates"]["sixth_order"]==2e-5
@pytest.mark.skipif(not target.CANONICAL_DIRECTORY.exists(),reason="not frozen")
def test_canonical_closes():
 s=json.loads((target.CANONICAL_DIRECTORY/"summary.json").read_text());assert s["passed"] and s["definitions_only"];assert not s["equilibrium_physical_potential_certified"]
 for line in (target.CANONICAL_DIRECTORY/"SHA256SUMS.txt").read_text().splitlines():e,n=line.split("  ",1);assert hashlib.sha256((target.CANONICAL_DIRECTORY/n).read_bytes()).hexdigest()==e
