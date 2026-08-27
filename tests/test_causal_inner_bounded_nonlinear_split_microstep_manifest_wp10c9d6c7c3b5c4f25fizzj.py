import hashlib,json,pytest
import run_causal_inner_bounded_nonlinear_split_microstep_manifest_wp10c9d6c7c3b5c4f25fizzj as target
def test_contract_freezes_relaxed_conservative_microstep():
 target._validate_parent();c=target._contract();assert c["spatial_step"]["proposal"]=="explicit midpoint RK2";assert c["spatial_step"]["entropy_relaxation"].startswith("one scalar gamma");assert c["recovery"]["accepted_state_only"];assert not c["kernel"]["trajectory_authorized"]
@pytest.mark.skipif(not target.CANONICAL_DIRECTORY.exists(),reason="manifest not frozen")
def test_manifest_is_definitions_only():
 s=json.loads((target.CANONICAL_DIRECTORY/"summary.json").read_text());assert s["passed"] and s["definitions_only"] and not s["bounded_nonlinear_microstep_certified"] and not s["trajectory_authorized"]
 for line in (target.CANONICAL_DIRECTORY/"SHA256SUMS.txt").read_text().splitlines():e,n=line.split("  ",1);assert hashlib.sha256((target.CANONICAL_DIRECTORY/n).read_bytes()).hexdigest()==e
