import hashlib,json,pytest
import run_causal_inner_equilibrium_selected_metric_stencil_full_rerun_wp10c9d6c7c3b5c4f25fizzc6 as target
def test_manifest_authorizes_factor_half_full_rerun():target._validate();assert target.manifest._contract()["selected_step_factor"]==.5
@pytest.mark.skipif(not target.CANONICAL_DIRECTORY.exists(),reason="not run")
def test_canonical_equilibrium_pass_stops_before_height():
 s=json.loads((target.CANONICAL_DIRECTORY/"summary.json").read_text());assert s["passed"] and s["equilibrium_physical_potential_certified"];assert not s["dynamic_height_potential_certified"];assert not s["complete_cycle_execution_authorized"]
 for line in (target.CANONICAL_DIRECTORY/"SHA256SUMS.txt").read_text().splitlines():e,n=line.split("  ",1);assert hashlib.sha256((target.CANONICAL_DIRECTORY/n).read_bytes()).hexdigest()==e
