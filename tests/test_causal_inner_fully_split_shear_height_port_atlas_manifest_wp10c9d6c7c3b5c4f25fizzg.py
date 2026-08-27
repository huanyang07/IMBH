import hashlib,json,pytest
import run_causal_inner_fully_split_shear_height_port_atlas_manifest_wp10c9d6c7c3b5c4f25fizzg as target
def test_manifest_freezes_complete_eleven_field_port_atlas():
 target._validate_parent();c=target._contract();assert c["architecture"]["fields"].endswith("= 11");assert c["architecture"]["full_tensor_no_projection"];assert c["normalized_local_principal_form"]["coordinate_matrix"].startswith("A_r=");assert not c["claim_boundary"]["cycle_execution_authorized"]
@pytest.mark.skipif(not target.CANONICAL_DIRECTORY.exists(),reason="manifest not frozen")
def test_canonical_manifest_is_definitions_only():
 s=json.loads((target.CANONICAL_DIRECTORY/"summary.json").read_text());assert s["passed"] and s["definitions_only"] and s["prior_rejections_preserved"];assert not s["fully_split_port_atlas_kernel_certified"]
 for line in (target.CANONICAL_DIRECTORY/"SHA256SUMS.txt").read_text().splitlines():e,n=line.split("  ",1);assert hashlib.sha256((target.CANONICAL_DIRECTORY/n).read_bytes()).hexdigest()==e
