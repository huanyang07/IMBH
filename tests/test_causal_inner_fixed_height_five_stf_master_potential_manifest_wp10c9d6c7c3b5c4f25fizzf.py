import hashlib, json, pytest
import run_causal_inner_fixed_height_five_stf_master_potential_manifest_wp10c9d6c7c3b5c4f25fizzf as target

def test_contract_freezes_restricted_linear_stress_question():
    target._validate_parent(); contract = target._contract(); assert contract["diagnostic"]["same_47_witnesses"]; assert contract["necessary_physical_condition"]["all_five_components_required"]; assert contract["decision"]["fail"]["authorized_next"] == target.FAILURE_NEXT; assert not contract["claim_boundary"]["complete_cycle_execution_authorized"]

@pytest.mark.skipif(not target.CANONICAL_DIRECTORY.exists(), reason="manifest not frozen")
def test_canonical_manifest_is_definitions_only():
    summary = json.loads((target.CANONICAL_DIRECTORY / "summary.json").read_text()); assert summary["passed"] and summary["definitions_only"]; assert summary["split_height_port_kernel_preserved"]; assert not summary["physical_five_STF_potential_certified"]
    for line in (target.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text().splitlines(): expected, name = line.split("  ", 1); assert hashlib.sha256((target.CANONICAL_DIRECTORY / name).read_bytes()).hexdigest() == expected
