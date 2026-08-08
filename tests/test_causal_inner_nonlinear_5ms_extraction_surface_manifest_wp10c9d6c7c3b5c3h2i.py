from __future__ import annotations

import json

import run_causal_inner_nonlinear_5ms_extraction_surface_manifest_wp10c9d6c7c3b5c3h2i as manifest


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_extraction_surface_is_fixed_and_common() -> None:
    payload = _read(manifest.MANIFEST_PATH)
    surface = payload["extraction_surface"]
    assert surface["radius_rg"] == manifest.EXTRACTION_RADIUS_RG
    assert tuple(surface["layout_face_indices"].values()) == (1, 2, 4)
    assert surface["same_physical_surface_required"]


def test_partition_semantics_do_not_relabel_horizon_flux() -> None:
    payload = _read(manifest.MANIFEST_PATH)
    partition = payload["domain_partition"]
    assert partition["slow_exterior_consumes_extraction_surface_flux"]
    assert partition["inner_buffer_storage_and_sources_remain_explicit"]
    assert partition["extraction_flux_is_not_relabeled_as_instantaneous_horizon_flux"]
    assert partition["pointwise_horizon_flux_convergence_not_claimed"]


def test_manifest_preserves_stops_and_authorizes_only_audit() -> None:
    summary = _read(manifest.SUMMARY_PATH)
    assert summary["passed"]
    assert summary["definitions_only"]
    assert summary["raw_inner_face_rejection_preserved"]
    assert summary["extraction_partition_certificate_authorized"]
    assert not summary["new_propagation_authorized"]
    assert not summary["fourth_duration_rung_manifest_authorized"]
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]
    assert summary["authorized_next"] == (
        "WP10c9d6c7c3b5c3h2i1_conservative_extraction_surface_certificate"
    )
