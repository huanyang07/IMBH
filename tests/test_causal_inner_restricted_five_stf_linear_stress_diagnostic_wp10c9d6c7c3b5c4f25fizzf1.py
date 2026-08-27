import hashlib, json, pytest
import run_causal_inner_restricted_five_stf_linear_stress_diagnostic_wp10c9d6c7c3b5c4f25fizzf1 as target

def test_manifest_authorizes_only_restricted_stress_diagnostic():
    _, contract = target._validate_parent(); assert contract["authorized_next"] == target.WORK_PACKAGE; assert contract["diagnostic"]["trajectory_steps"] == 0

@pytest.mark.skipif(not target.CANONICAL_DIRECTORY.exists(), reason="diagnostic not run")
def test_canonical_no_go_is_failure_aware():
    summary = json.loads((target.CANONICAL_DIRECTORY / "summary.json").read_text()); metrics = json.loads((target.CANONICAL_DIRECTORY / "diagnostic_metrics.json").read_text()); assert summary["audit_completed"] and not summary["passed"]; assert summary["fully_split_port_atlas_manifest_authorized"]; assert summary["fixed_height_equilibrium_potential_preserved"] and summary["split_height_port_kernel_preserved"]; assert metrics["coefficient_independent_no_go"]; assert metrics["viable_witness_count"] == 0
    for line in (target.CANONICAL_DIRECTORY / "SHA256SUMS.txt").read_text().splitlines(): expected, name = line.split("  ", 1); assert hashlib.sha256((target.CANONICAL_DIRECTORY / name).read_bytes()).hexdigest() == expected
