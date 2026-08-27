import hashlib,json,pytest
import run_causal_inner_conditioned_discrete_gradient_flux_manifest_wp10c9d6c7c3b5c4f25fizzi2 as target
def test_contract_preserves_rejection_and_freezes_discrete_gradient():
 target._validate_parent();c=target._contract();assert c["preserved"]["straight_entropy_path_rejected"];assert not c["repair"]["path_integration"];assert c["repair"]["symmetry"].startswith("f_DG");assert c["kernel"]["trajectory_steps"]==0
@pytest.mark.skipif(not target.CANONICAL_DIRECTORY.exists(),reason="manifest not frozen")
def test_canonical_manifest_is_definitions_only():
 s=json.loads((target.CANONICAL_DIRECTORY/"summary.json").read_text());assert s["passed"] and s["definitions_only"] and s["straight_entropy_path_rejection_preserved"] and not s["trajectory_authorized"]
 for line in (target.CANONICAL_DIRECTORY/"SHA256SUMS.txt").read_text().splitlines():e,n=line.split("  ",1);assert hashlib.sha256((target.CANONICAL_DIRECTORY/n).read_bytes()).hexdigest()==e
