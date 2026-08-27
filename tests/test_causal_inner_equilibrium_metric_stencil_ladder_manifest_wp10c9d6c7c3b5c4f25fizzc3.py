import hashlib,json,pytest
import run_causal_inner_equilibrium_metric_stencil_ladder_manifest_wp10c9d6c7c3b5c4f25fizzc3 as target

def test_parent_failed_only_stencil_gate():
    v=target._validate_parent();assert not v["summary"]["passed"];assert v["metrics"]["maximum_physical_current_relative_defect"]<=1e-10
def test_ladder_is_prospective_and_gate_unchanged():
    c=target._contract();assert c["diagnostic"]["step_factors"]==target.FACTORS;assert c["diagnostic"]["passing_factor_gate"]==2e-5;assert c["preserved_rejection"]["retroactive_pass_forbidden"]
@pytest.mark.skipif(not target.CANONICAL_DIRECTORY.exists(),reason="not frozen")
def test_canonical_closes():
    s=json.loads((target.CANONICAL_DIRECTORY/"summary.json").read_text());assert s["passed"] and s["definitions_only"];assert not s["equilibrium_physical_potential_certified"]
    for line in (target.CANONICAL_DIRECTORY/"SHA256SUMS.txt").read_text().splitlines():e,n=line.split("  ",1);assert hashlib.sha256((target.CANONICAL_DIRECTORY/n).read_bytes()).hexdigest()==e
