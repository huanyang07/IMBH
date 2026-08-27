import hashlib,json,pytest
import run_causal_inner_equilibrium_metric_stencil_ladder_diagnostic_wp10c9d6c7c3b5c4f25fizzc4 as target
def test_manifest_authorizes_exact_ladder():
    v=target._validate();assert tuple(v["contract"]["diagnostic"]["step_factors"])==target.manifest.FACTORS
def test_contiguous_runs_are_deterministic():assert target._runs((False,True,True,True,False,True))==[(1,4),(5,6)]
@pytest.mark.skipif(not target.CANONICAL_DIRECTORY.exists(),reason="not run")
def test_canonical_closes_without_certifying_equilibrium():
    s=json.loads((target.CANONICAL_DIRECTORY/"summary.json").read_text());assert s["original_rejection_preserved"];assert not s["equilibrium_physical_potential_certified"]
    for line in (target.CANONICAL_DIRECTORY/"SHA256SUMS.txt").read_text().splitlines():e,n=line.split("  ",1);assert hashlib.sha256((target.CANONICAL_DIRECTORY/n).read_bytes()).hexdigest()==e
