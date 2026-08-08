from __future__ import annotations

import json

import numpy as np

import run_causal_inner_nonlinear_5ms_extraction_surface_certificate_wp10c9d6c7c3b5c3h2i1 as certificate


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_certificate_preserves_domain_partition_semantics() -> None:
    summary = _read(certificate.SUMMARY_PATH)
    assert summary["raw_inner_face_spatial_convergence_certified"] is False
    assert summary["pointwise_horizon_flux_convergence_certified"] is False
    assert not summary["fixed_q_micro_solver_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]


def test_certificate_uses_fixed_common_surface() -> None:
    summary = _read(certificate.SUMMARY_PATH)
    analysis = summary["analysis"]
    assert analysis["extraction_radius_rg"] == certificate.h2i.EXTRACTION_RADIUS_RG
    assert tuple(analysis["layout_extraction_face_indices"].values()) == (1, 2, 4)


def test_binding_channels_and_ledgers_determine_classification() -> None:
    summary = _read(certificate.SUMMARY_PATH)
    analysis = summary["analysis"]
    expected = bool(
        analysis["state_channel_inherited_from_h2f_passed"]
        and analysis["instantaneous_exterior_partition"]["binding_channel_passed"]
        and analysis["cumulative_exterior_partition"]["binding_channel_passed"]
        and analysis["ledger_audits"]["passed"]
    )
    assert summary["passed"] is expected
    assert analysis["extraction_partition_spatial_certificate_passed"] is expected
    assert summary["fourth_duration_rung_manifest_authorized"] is expected


def test_decisive_arrays_have_all_layout_histories() -> None:
    with np.load(certificate.DECISIVE_ARRAYS, allow_pickle=False) as payload:
        times = payload["times_seconds"]
        for layout in certificate.LAYOUTS:
            assert payload[f"{layout}__exterior_response"].shape == (times.size, 13)
            assert payload[f"{layout}__cumulative_exterior_response"].shape == (times.size, 13)
