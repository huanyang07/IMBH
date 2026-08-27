import hashlib,json,pytest
import run_causal_inner_compensated_discrete_gradient_flux_manifest_wp10c9d6c7c3b5c4f25fizzi4 as target
def test_contract_changes_representation_not_mathematics():
 target._validate_parent();c=target._contract();assert c["preserved"]["ordinary_double_discrete_gradient_rejected"];assert c["preserved"]["discrete_gradient_formula_unchanged"];assert c["preserved"]["Tadmor_gate_unchanged"]==2e-12;assert c["kernel"]["trajectory_steps"]==0
@pytest.mark.skipif(not target.CANONICAL_DIRECTORY.exists(),reason="manifest not frozen")
def test_manifest_is_definitions_only():
 s=json.loads((target.CANONICAL_DIRECTORY/"summary.json").read_text());assert s["passed"] and s["definitions_only"] and s["prior_rejections_preserved"] and not s["trajectory_authorized"]
 for line in (target.CANONICAL_DIRECTORY/"SHA256SUMS.txt").read_text().splitlines():e,n=line.split("  ",1);assert hashlib.sha256((target.CANONICAL_DIRECTORY/n).read_bytes()).hexdigest()==e
