import hashlib,json,pytest
import run_causal_inner_conservative_entropy_projection_microstep_manifest_wp10c9d6c7c3b5c4f25fizzj0 as target
def test_projection_is_zero_sum_and_supersedes_before_execution():
 target._validate_parent();c=target._contract();assert c["pre_execution_change"];assert c["entropy_projection"]["zero_sum_direction"].startswith("z_i=-M");assert c["spatial_step"]["proposal"]=="explicit midpoint RK2";assert not c["kernel"]["trajectory_authorized"]
@pytest.mark.skipif(not target.CANONICAL_DIRECTORY.exists(),reason="manifest not frozen")
def test_manifest_is_definitions_only():
 s=json.loads((target.CANONICAL_DIRECTORY/"summary.json").read_text());assert s["passed"] and s["prior_microstep_manifest_superseded_before_execution"] and not s["trajectory_authorized"]
 for line in (target.CANONICAL_DIRECTORY/"SHA256SUMS.txt").read_text().splitlines():e,n=line.split("  ",1);assert hashlib.sha256((target.CANONICAL_DIRECTORY/n).read_bytes()).hexdigest()==e
