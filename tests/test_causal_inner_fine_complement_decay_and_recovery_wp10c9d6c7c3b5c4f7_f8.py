import json
import run_causal_inner_fine_complement_spatial_decay_audit_wp10c9d6c7c3b5c4f7 as f7
import run_causal_inner_recovered_coupling_control_volume_manifest_wp10c9d6c7c3b5c4f8 as f8
def _r(p):return json.loads(p.read_text())
def test_decay_selects_face_36_prospectively():
 s=_r(f7.SUMMARY_PATH);assert s['passed'] and s['selected_recovery_parent_face']==36
 assert s['JVP_fraction_of_transition']['36']<.1 and s['JVP_fraction_of_transition']['40']>.1
 assert all(s['face_metrics'][str(face)]['passed'] for face in range(32,48) if str(face) in s['face_metrics'])
def test_guard_buffer_manifest_preserves_physics_and_stops():
 m=f8._manifest();assert m['guard_buffer_remains_explicitly_evolved']
 assert m['recovery_flux_is_not_relabelled_horizon_or_original_coupling_flux']
 assert m['candidate_architectures']['control_volume_identity']['mapped_storage_and_nonexact_height_history_both_required']
 assert m['candidate_architectures']['control_volume_identity']['direct_BDF_identity_may_not_be_used_as_independent_convergence_evidence']
 s=_r(f8.SUMMARY_PATH);assert s['passed'] and s['definitions_only'] and not s['memory_propagation_authorized'] and not s['reduced_slow_evolution_authorized']
