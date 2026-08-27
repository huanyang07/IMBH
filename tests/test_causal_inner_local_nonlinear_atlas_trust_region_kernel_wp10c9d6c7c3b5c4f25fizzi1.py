import hashlib,json,pytest
import run_causal_inner_local_nonlinear_atlas_trust_region_kernel_wp10c9d6c7c3b5c4f25fizzi1 as target
def test_manifest_authorizes_only_local_trust_kernel():
    _,c=target._validate_parent();assert c["authorized_next"]==target.WORK_PACKAGE;assert c["kernel"]["deterministic_endpoint_pairs_per_anchor"]==8;assert c["kernel"]["trajectory_steps"]==0
@pytest.mark.skipif(not target.CANONICAL_DIRECTORY.exists(),reason="kernel not run")
def test_canonical_trust_kernel_is_fail_closed():
    s=json.loads((target.CANONICAL_DIRECTORY/"summary.json").read_text());m=json.loads((target.CANONICAL_DIRECTORY/"kernel_metrics.json").read_text());assert s["passed"]==m["passed"] and s["local_nonlinear_atlas_trust_region_certified"]==m["passed"] and not s["trajectory_authorized"];assert m["endpoint_pair_count"]==376
    if not m["passed"]:
        assert m["classification"]==target.FAIL_CLASSIFICATION
        assert m["passing_endpoint_pair_count"]<m["endpoint_pair_count"]
        assert m["maximum_quadrature_refinement_relative_defect"]>2e-9
    for line in (target.CANONICAL_DIRECTORY/"SHA256SUMS.txt").read_text().splitlines():e,n=line.split("  ",1);assert hashlib.sha256((target.CANONICAL_DIRECTORY/n).read_bytes()).hexdigest()==e
