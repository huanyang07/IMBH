from __future__ import annotations

from imri_qpe.layer3_minidisk_1d import OverlapGateConfig

from scripts.run_inner_outer_overlap_audit import _audit_tier


def test_canonical_overlap_is_threshold_sensitive() -> None:
    strict = _audit_tier(OverlapGateConfig(max_radial_pressure_fraction=0.05))
    sensitivity = _audit_tier(
        OverlapGateConfig(max_radial_pressure_fraction=0.10)
    )
    assert strict["common_transonic_wall_bands_rg"] == []
    assert strict["common_transonic_open_bands_rg"] == []
    wall_band = sensitivity["common_transonic_wall_bands_rg"][0]
    open_band = sensitivity["common_transonic_open_bands_rg"][0]
    assert 29.0 < wall_band[0] < 30.0
    assert 59.0 < wall_band[1] < 60.0
    assert 24.0 < open_band[0] < 25.0
    assert 59.0 < open_band[1] < 60.0
