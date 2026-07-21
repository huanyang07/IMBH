from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_causal_tangent_certification_wp10c8j as wp10c8j


def test_locked_contract_matches_wp10c8i() -> None:
    contract = wp10c8j._locked_contract()

    assert contract["resolutions"] == (64, 128)
    assert contract["anchors"] == wp10c8j.LOCKED_ANCHORS
    assert contract["finite_time_horizons_seconds"] == (0.0, 0.01, 0.025)
    assert contract["maximum_screening_gate_fraction"] == 0.25


def test_locked_contract_rejects_changed_wp10c8i_gate(monkeypatch) -> None:
    monkeypatch.setattr(
        wp10c8j.wp10c8i,
        "SCREENING_MAXIMUM_GATE_FRACTION",
        0.5,
    )

    with pytest.raises(RuntimeError, match="contract changed"):
        wp10c8j._locked_contract()


def test_authorization_requires_canonical_wp10c8i_evidence() -> None:
    evidence, digest = wp10c8j._validate_authorization()

    assert evidence["decision"] == wp10c8j.WP10C8I_REQUIRED_DECISION
    assert evidence["next_authorization"] == (
        wp10c8j.WP10C8I_REQUIRED_NEXT_AUTHORIZATION
    )
    assert len(digest) == 64


def _write_operator_fixture(
    path: Path,
    *,
    state_sha256: str,
    shell_edges: np.ndarray,
    operator_contract: dict | None = None,
) -> None:
    metadata = {
        "schema_version": wp10c8j.wp10c8i.CACHE_SCHEMA_VERSION,
        "work_package": "WP10c8i",
        "base_commit": wp10c8j.WP10C8I_BASE_COMMIT,
        "n_cells": 1,
        "anchor_label": "t_0",
        "state_vector_sha256": state_sha256,
        "shell_edges_rg": shell_edges.tolist(),
        "operator_contract": operator_contract or {},
    }
    np.savez_compressed(
        path,
        dynamic=np.eye(5),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def test_authorized_operator_loader_accepts_canonical_artifact(
    monkeypatch,
    tmp_path,
) -> None:
    vector = np.zeros(20)
    state_sha256 = wp10c8j.wp10c8i._array_sha256(vector)
    shell_edges = np.asarray([1.0, 2.0])
    cache = tmp_path / "operator.npz"
    _write_operator_fixture(
        cache,
        state_sha256=state_sha256,
        shell_edges=shell_edges,
    )
    evidence = tmp_path / "evidence.json"
    evidence.write_text(
        json.dumps(
            {
                "operator_provenance": {
                    "1": {
                        "t_0": {
                            "path": str(cache),
                            "sha256": wp10c8j._sha256(cache),
                            "state_vector_sha256": state_sha256,
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(wp10c8j, "WP10C8I_OUTPUT", evidence)
    monkeypatch.setattr(wp10c8j, "_relative", lambda path: str(path))

    arrays, _metadata, provenance = (
            wp10c8j._authorized_wp10c8i_operator_cache(
                {"state": SimpleNamespace(n_cells=1), "context": object()},
                vector,
                "t_0",
                "construction",
                shell_edges,
            )
    )

    assert np.array_equal(arrays["dynamic"], np.eye(5))
    assert provenance["canonical_hash_matched"]
    assert not provenance["explicit_rebuild_performed"]
    assert provenance["canonical_path_left_unmodified"]


def test_authorized_operator_loader_rejects_unhashed_canonical_replacement(
    monkeypatch,
    tmp_path,
) -> None:
    vector = np.zeros(20)
    state_sha256 = wp10c8j.wp10c8i._array_sha256(vector)
    shell_edges = np.asarray([1.0, 2.0])
    cache = tmp_path / "operator.npz"
    _write_operator_fixture(
        cache,
        state_sha256=state_sha256,
        shell_edges=shell_edges,
    )
    evidence = tmp_path / "evidence.json"
    evidence.write_text(
        json.dumps(
            {
                "operator_provenance": {
                    "1": {
                        "t_0": {
                            "path": str(cache),
                            "sha256": "not-the-current-hash",
                            "state_vector_sha256": state_sha256,
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(wp10c8j, "WP10C8I_OUTPUT", evidence)
    monkeypatch.setattr(
        wp10c8j,
        "_operator_source_path",
        lambda *_args: tmp_path / "absent-versioned-source.npz",
    )

    with pytest.raises(RuntimeError, match="immutable hash"):
        wp10c8j._authorized_wp10c8i_operator_cache(
            {"state": SimpleNamespace(n_cells=1), "context": object()},
            vector,
            "t_0",
            "construction",
            shell_edges,
        )


def test_versioned_operator_source_rejects_array_mutation(
    monkeypatch,
    tmp_path,
) -> None:
    vector = np.zeros(20)
    state_sha256 = wp10c8j.wp10c8i._array_sha256(vector)
    shell_edges = np.asarray([1.0, 2.0])
    contract = {"physics": "fixed", "code_sha256": {"source.py": "abc"}}
    arrays = {"dynamic": np.eye(5)}
    metadata = {
        "schema_version": wp10c8j.wp10c8i.CACHE_SCHEMA_VERSION,
        "work_package": "WP10c8i",
        "base_commit": wp10c8j.WP10C8I_BASE_COMMIT,
        "n_cells": 1,
        "anchor_label": "t_0",
        "state_vector_sha256": state_sha256,
        "shell_edges_rg": shell_edges.tolist(),
        "operator_contract": contract,
        "operator_contract_sha256": wp10c8j.wp10c8i._text_sha256(
            json.dumps(contract, sort_keys=True)
        ),
        "wp10c8j_operator_source": {
            "schema_version": 1,
            "work_package": wp10c8j.WORK_PACKAGE,
            "base_commit": wp10c8j.BASE_COMMIT,
            "array_sha256": wp10c8j._operator_array_hashes(arrays),
        },
    }
    versioned = tmp_path / "versioned.npz"
    np.savez_compressed(
        versioned,
        dynamic=2.0 * np.eye(5),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    canonical = tmp_path / "changed-canonical.npz"
    _write_operator_fixture(
        canonical,
        state_sha256=state_sha256,
        shell_edges=shell_edges,
    )
    evidence = tmp_path / "evidence.json"
    evidence.write_text(
        json.dumps(
            {
                "operator_provenance": {
                    "1": {
                        "t_0": {
                            "path": str(canonical),
                            "sha256": "immutable-canonical-hash",
                            "state_vector_sha256": state_sha256,
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(wp10c8j, "WP10C8I_OUTPUT", evidence)
    monkeypatch.setattr(
        wp10c8j,
        "_operator_source_path",
        lambda *_args: versioned,
    )
    monkeypatch.setattr(
        wp10c8j.wp10c8i,
        "_operator_contract",
        lambda *_args: contract,
    )

    with pytest.raises(RuntimeError, match="not authorized"):
        wp10c8j._authorized_wp10c8i_operator_cache(
            {"state": SimpleNamespace(n_cells=1), "context": object()},
            vector,
            "t_0",
            "construction",
            shell_edges,
        )


def test_matrix_comparison_reports_controlling_cell_and_field() -> None:
    reference = np.eye(10)
    candidate = reference.copy()
    candidate[7, 2] += 1.0e-2
    directions = {"azimuthal": np.eye(10)[2]}

    row = wp10c8j._matrix_comparison(
        candidate,
        reference,
        directions,
        row_kind="primitive",
    )

    assert not row["passed"]
    assert row["controlling_matrix_row"] == {
        "flat_index": 7,
        "cell_index": 1,
        "component": "azimuthal_three_velocity_over_c",
    }
    assert row["controlling_matrix_column"]["component"] == (
        "azimuthal_three_velocity_over_c"
    )
    assert row["controlling_jvp_direction"] == "azimuthal"


def test_scan_variants_requires_every_neighboring_comparison() -> None:
    identity = np.eye(5)
    variants = {
        "1e-06": identity,
        "2e-06": identity,
        "4e-06": 1.1 * identity,
    }
    row = wp10c8j._scan_variants(
        variants,
        base_key="2e-06",
        directions={"one": np.ones(5) / np.sqrt(5.0)},
        row_kind="primitive",
    )

    assert not row["passed"]
    assert row["comparisons"]["1e-06_versus_2e-06"]["passed"]
    assert not row["comparisons"]["4e-06_versus_2e-06"]["passed"]


def test_separated_scan_varies_blocks_independently(monkeypatch) -> None:
    size = 5
    identity = np.eye(size)
    calls = {"stationary": [], "mass": []}

    monkeypatch.setattr(
        wp10c8j,
        "unpack_causal_five_field_state",
        lambda _vector, _n: SimpleNamespace(primitives=np.zeros((1, 5))),
    )
    monkeypatch.setattr(
        wp10c8j,
        "_normalized_directions",
        lambda *_args: {"one": np.eye(size)[0]},
    )

    def stationary(_context, _vector, *, finite_difference_step):
        calls["stationary"].append(finite_difference_step)
        return {"stationary_reduced_scaled_jacobian": -2.0 * identity}

    def mass(
        _context,
        _primitives,
        *,
        primitive_column_scales,
        conservation_row_scales,
        finite_difference_step,
        storage_quadrature_order,
        storage_directional_step,
    ):
        del primitive_column_scales, conservation_row_scales
        del storage_quadrature_order, storage_directional_step
        calls["mass"].append(finite_difference_step)
        return {
            "descriptor_reduced_scaled_matrix": identity,
            "conserved_descriptor_reduced_scaled_matrix": 0.75 * identity,
            "vertical_descriptor_reduced_scaled_matrix": 0.25 * identity,
            "maximum_scaled_component_reconstruction_defect": 0.0,
        }

    monkeypatch.setattr(
        wp10c8j,
        "causal_five_field_reduced_stationary_jacobian",
        stationary,
    )
    monkeypatch.setattr(
        wp10c8j,
        "causal_five_field_reduced_storage_matrices",
        mass,
    )
    monkeypatch.setattr(
        wp10c8j,
        "_direct_action_storage_rate_result",
        lambda *_args, **_kwargs: {
            "storage_rate_derivative_scaled_matrix": 0.5 * identity,
            "conserved_storage_rate_derivative_scaled_matrix": (
                0.4 * identity
            ),
            "vertical_storage_rate_derivative_scaled_matrix": (
                0.1 * identity
            ),
            "maximum_scaled_component_reconstruction_defect": 0.0,
        },
    )
    monkeypatch.setattr(
        wp10c8j,
        "_repaired_generator_jvp_contract",
        lambda *_args, **_kwargs: ({}, {"passed": True}),
    )
    monkeypatch.setattr(
        wp10c8j,
        "_fresh_independent_vector_field_jvp_contract",
        lambda *_args, **_kwargs: ({}, {"passed": True}),
    )
    monkeypatch.setattr(
        wp10c8j,
        "_fresh_base_vector_field",
        lambda *_args, **_kwargs: (
            np.zeros(size),
            {"scaled_primitive_rate_per_s": np.ones(size)},
            ({}, np.ones(1), {}),
            {"passed": True},
        ),
    )
    initial = {"state": SimpleNamespace(n_cells=1), "context": object()}
    operator_arrays = {
        "primitive_column_scales": np.ones(size),
        "conservation_row_scales": np.ones(size),
        "scaled_primitive_rate": np.ones(size),
        "dynamic": 1.5 * identity,
        "storage_rate_derivative": 0.5 * identity,
        **{
            f"generator_inner_storage_fd_dynamic_{step:.0e}": 1.5 * identity
            for step in wp10c8j.INNER_DIFFERENCE_STEPS
        },
        **{
            f"generator_outer_storage_rate_fd_dynamic_{step:.0e}": (
                1.5 * identity
            )
            for step in wp10c8j.LEGACY_OUTER_DIFFERENCE_STEPS
        },
        **{
            f"generator_vertical_action_fd_dynamic_{step:.0e}": (
                1.5 * identity
            )
            for step in wp10c8j.LEGACY_VERTICAL_ACTION_DIFFERENCE_STEPS
        },
    }
    arrays, metadata = wp10c8j._separated_tangent_scan(
        initial,
        np.zeros(20),
        operator_arrays,
        {},
    )

    assert metadata["passed"]
    assert calls["stationary"] == list(wp10c8j.INNER_DIFFERENCE_STEPS)
    assert calls["mass"] == list(wp10c8j.INNER_DIFFERENCE_STEPS)
    assert "generator_stationary_jacobian_1e-06" in arrays
    assert (
        f"block_direct_total_dm_outer_"
        f"{wp10c8j.OUTER_DIFFERENCE_STEPS[0]:.0e}"
    ) in arrays
    assert metadata["binding_step_scans_passed"]


def _branch_arrays(n_cells: int) -> dict[str, np.ndarray]:
    size = 5 * n_cells
    return {
        "dynamic": -np.eye(size),
        "state_weights": np.ones(size),
        "production_rusanov_kink_generator_left_factors": np.empty(
            (size, 0)
        ),
        "production_rusanov_kink_generator_right_factors": np.empty(
            (size, 0)
        ),
    }


def test_branch_certificate_does_not_turn_absent_anchor_kinks_into_pass(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        wp10c8j.wp10c8i,
        "_response_stack",
        lambda *_args: (
            np.eye(10),
            np.ones(10),
            tuple(f"row_{index}" for index in range(10)),
            {},
        ),
    )
    monkeypatch.setattr(
        wp10c8j.wp10c8i,
        "_rusanov_kink_instantaneous_output_deltas",
        lambda *_args: np.empty((0, 10, 10)),
    )

    row = wp10c8j._branch_certification(
        _branch_arrays(2),
        {"levels": ({"name": "level_0"},)},
        nonlinear_remainder_rate=0.0,
        nonlinear_output_remainder_bounds={
            ("level_0", horizon): np.zeros(10)
            for horizon in wp10c8j.LOCKED_FINITE_TIME_HORIZONS_SECONDS
        },
        nonlinear_remainder_certified=True,
        certified_neighborhood_radius=1.0,
    )

    # A finite neighborhood still needs coverage of the one interior face;
    # no anchor kink is not a coverage proof.
    assert row["consequential_branch_count"] == 0
    assert row["interior_face_count"] == 1
    assert not row["passed"]
    assert all(
        not result["binding"]
        for result in row["levels"]["level_0"].values()
    )


def test_branch_certificate_passes_trivial_zero_face_case(monkeypatch) -> None:
    monkeypatch.setattr(
        wp10c8j.wp10c8i,
        "_response_stack",
        lambda *_args: (
            np.eye(5),
            np.ones(5),
            tuple(f"row_{index}" for index in range(5)),
            {},
        ),
    )
    monkeypatch.setattr(
        wp10c8j.wp10c8i,
        "_rusanov_kink_instantaneous_output_deltas",
        lambda *_args: np.empty((0, 5, 5)),
    )

    row = wp10c8j._branch_certification(
        _branch_arrays(1),
        {"levels": ({"name": "level_0"},)},
        nonlinear_remainder_rate=0.0,
        nonlinear_output_remainder_bounds={
            ("level_0", horizon): np.zeros(5)
            for horizon in wp10c8j.LOCKED_FINITE_TIME_HORIZONS_SECONDS
        },
        nonlinear_remainder_certified=True,
        certified_neighborhood_radius=1.0,
    )

    assert row["interior_face_count"] == 0
    assert row["passed"]


def test_campaign_repeat_is_strictly_conditional() -> None:
    failed = {
        "n64_t_0": {
            "separated_tangent": {"passed": False},
            "rusanov_finite_neighborhood": {
                "all_rows_binding": False,
                "passed": False,
            },
        }
    }
    tangent_only = {
        "n64_t_0": {
            "separated_tangent": {"passed": True},
            "rusanov_finite_neighborhood": {
                "all_rows_binding": False,
                "passed": False,
            },
        }
    }
    passed = {
        "n64_t_0": {
            "separated_tangent": {"passed": True},
            "rusanov_finite_neighborhood": {
                "all_rows_binding": True,
                "passed": True,
            },
        }
    }

    assert wp10c8j._campaign_decision(
        failed,
        complete_selection=True,
    )[0] == "wp10c8j_smooth_tangent_failed_rusanov_certificate_absent"
    assert wp10c8j._campaign_decision(
        tangent_only,
        complete_selection=True,
    )[0] == "wp10c8j_smooth_tangent_certified_rusanov_certificate_absent"
    assert wp10c8j._campaign_decision(
        passed,
        complete_selection=True,
    )[0] == "wp10c8j_tangent_and_finite_branch_contract_passed"
    assert wp10c8j._campaign_decision(
        passed,
        complete_selection=False,
    )[0] == "wp10c8j_partial_certification_completed"
    assert wp10c8j._campaign_decision(
        passed,
        complete_selection=True,
    )[1].endswith("_in_a_separate_package")


def _passing_cached_numerical_metadata() -> dict:
    return {
        "storage_audit": {
            "passed": True,
            "complete_vector_one_form_present": True,
            "maximum_relative_storage_action_defect": 1.0e-7,
            "maximum_relative_historical_storage_action_defect": 1.0e-7,
            "maximum_scaled_descriptor_component_reconstruction_defect": (
                1.0e-13
            ),
            "maximum_scaled_storage_rate_component_reconstruction_defect": (
                1.0e-13
            ),
            "maximum_scaled_generator_factorization_defect": 1.0e-12,
        },
        "tangent_differentiability_audit": {"passed": True},
        "generator_stability_audit": {
            "production_vector_field_jvp": {
                "scope": "independent_nonlinear_vector_field",
                "passed": True,
                "direction_names": ("thermal",),
                "directions": {"thermal": {"passed": True}},
            }
        },
    }


@pytest.mark.parametrize(
    ("section", "key", "bad_value"),
    (
        ("storage_audit", "passed", False),
        ("storage_audit", "complete_vector_one_form_present", False),
        (
            "storage_audit",
            "maximum_relative_storage_action_defect",
            1.0,
        ),
        (
            "storage_audit",
            "maximum_relative_historical_storage_action_defect",
            1.0,
        ),
        (
            "storage_audit",
            "maximum_scaled_descriptor_component_reconstruction_defect",
            1.0,
        ),
        (
            "storage_audit",
            "maximum_scaled_storage_rate_component_reconstruction_defect",
            1.0,
        ),
        (
            "storage_audit",
            "maximum_scaled_generator_factorization_defect",
            1.0,
        ),
        ("tangent_differentiability_audit", "passed", False),
        ("production_vector_field_jvp", "direction_names", ()),
    ),
)
def test_cached_operator_contract_cannot_pass_false_cached_evidence(
    section,
    key,
    bad_value,
) -> None:
    metadata = copy.deepcopy(_passing_cached_numerical_metadata())
    if section == "production_vector_field_jvp":
        metadata["generator_stability_audit"][section][key] = bad_value
    else:
        metadata[section][key] = bad_value

    row = wp10c8j._cached_operator_numerical_contract(metadata)

    assert not row["passed"]


def test_non_full_anchor_requires_repaired_base_tangent(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        wp10c8j,
        "_authorized_wp10c8i_operator_cache",
        lambda *_args, **_kwargs: (
            {"dynamic": np.eye(5)},
            _passing_cached_numerical_metadata(),
            {"path": "operator.npz", "sha256": "abc"},
        ),
    )
    monkeypatch.setattr(
        wp10c8j,
        "_separated_tangent_scan",
        lambda *_args: pytest.fail(
            "separated scan must remain locked to t_0 and t_0p10"
        ),
    )
    monkeypatch.setattr(
        wp10c8j,
        "_base_repaired_tangent",
        lambda *_args, **_kwargs: (
            {"repaired_dynamic": np.eye(5)},
            {"passed": True, "separated_scan_evaluated": False},
        ),
    )
    monkeypatch.setattr(
        wp10c8j,
        "_branch_certification",
        lambda *_args: {"passed": True, "all_rows_binding": True},
    )
    monkeypatch.setattr(
        wp10c8j.wp10c8i,
        "_operator_contract",
        lambda *_args: {"contract": "fixed"},
    )
    monkeypatch.setattr(
        wp10c8j,
        "_cache_path",
        lambda *_args: tmp_path / "cache.npz",
    )
    monkeypatch.setattr(wp10c8j, "_relative", lambda path: str(path))
    initial = {"state": SimpleNamespace(n_cells=1), "context": object()}

    _arrays, metadata, _provenance = wp10c8j._build_certification_cache(
        initial,
        np.zeros(20),
        "t_0p025",
        "construction",
        np.asarray([1.0, 2.0]),
    )

    assert metadata["separated_tangent"]["passed"]
    assert not metadata["separated_tangent"]["separated_scan_evaluated"]
    assert metadata["passed"]


def test_full_anchor_requires_cached_contract_even_when_scan_passes(
    monkeypatch,
    tmp_path,
) -> None:
    metadata = _passing_cached_numerical_metadata()
    metadata["storage_audit"]["passed"] = False
    monkeypatch.setattr(
        wp10c8j,
        "_authorized_wp10c8i_operator_cache",
        lambda *_args, **_kwargs: (
            {"dynamic": np.eye(5)},
            metadata,
            {"path": "operator.npz", "sha256": "abc"},
        ),
    )
    monkeypatch.setattr(
        wp10c8j,
        "_separated_tangent_scan",
        lambda *_args: (
            {"repaired_dynamic": np.eye(5)},
            {"passed": True},
        ),
    )
    monkeypatch.setattr(
        wp10c8j,
        "_branch_certification",
        lambda *_args: {"passed": True, "all_rows_binding": True},
    )
    monkeypatch.setattr(
        wp10c8j.wp10c8i,
        "_operator_contract",
        lambda *_args: {"contract": "fixed"},
    )
    monkeypatch.setattr(
        wp10c8j,
        "_cache_path",
        lambda *_args: tmp_path / "cache.npz",
    )
    monkeypatch.setattr(wp10c8j, "_relative", lambda path: str(path))

    _arrays, result, _provenance = wp10c8j._build_certification_cache(
        {"state": SimpleNamespace(n_cells=1), "context": object()},
        np.zeros(20),
        "t_0",
        "construction",
        np.asarray([1.0, 2.0]),
    )

    assert not result["separated_tangent"]["passed"]
    assert not result["passed"]
    assert not result["separated_tangent"][
        "cached_operator_numerical_contract"
    ]["checks"]["storage_audit_passed"]


def test_runner_does_not_own_wp10c8i_repeat_subprocess() -> None:
    assert not hasattr(wp10c8j, "_run_unchanged_wp10c8i")
    assert not hasattr(wp10c8j, "_validate_moment_repeat")


def _mock_fresh_jvp_audit_result(
    direction: np.ndarray,
    *,
    jump: float = 0.0,
    numerical_passed: bool = True,
) -> tuple[dict[str, np.ndarray], dict]:
    values = np.asarray(direction, dtype=float)
    arrays = {
        "independent_vector_field_jvp_direction": values,
        "independent_vector_field_jvp_direct": values,
        "independent_vector_field_jvp_forward": values,
        "independent_vector_field_jvp_backward": values,
    }
    for sample in ("base", "plus", "minus"):
        arrays[
            "independent_vector_field_jvp_"
            f"{sample}_rusanov_relative_margins"
        ] = np.asarray([0.0])
        arrays[
            "independent_vector_field_jvp_"
            f"{sample}_rusanov_scaled_relative_jumps"
        ] = np.asarray([jump])
    defect = {"passed": numerical_passed}
    return arrays, {
        "plus_minus_reconstruction_differentiable": True,
        "plus_minus_outer_active_set_unchanged": True,
        "rusanov_controls_unchanged": True,
        "minimum_required_rusanov_control_relative_margin": 1.0e-6,
        "central_jvp_defect": defect,
        "forward_jvp_defect": defect,
        "backward_jvp_defect": defect,
    }


def _fresh_jvp_fixture() -> tuple[dict, np.ndarray, dict[str, np.ndarray]]:
    initial = {"state": SimpleNamespace(n_cells=1), "context": object()}
    operator_arrays = {
        "primitive_column_scales": np.ones(5),
        "conservation_row_scales": np.ones(5),
    }
    return initial, np.zeros(20), operator_arrays


def test_fresh_strict_jvp_checks_every_direction_at_multiple_steps(
    monkeypatch,
) -> None:
    initial, vector, operator_arrays = _fresh_jvp_fixture()
    monkeypatch.setattr(
        wp10c8j,
        "unpack_causal_five_field_state",
        lambda _vector, _n: SimpleNamespace(primitives=np.zeros((1, 5))),
    )
    calls = []

    def audit(
        _initial,
        _vector,
        _evolving,
        direction,
        *,
        direction_name,
        centered_difference_step,
        **_kwargs,
    ):
        calls.append((direction_name, centered_difference_step))
        return _mock_fresh_jvp_audit_result(direction)

    monkeypatch.setattr(
        wp10c8j.wp10c8i,
        "_independent_vector_field_jvp_audit",
        audit,
    )
    directions = {
        "thermal": np.eye(5)[0],
        "radial": np.eye(5)[1],
    }

    _arrays, row = wp10c8j._fresh_independent_vector_field_jvp_contract(
        initial,
        vector,
        np.eye(5),
        operator_arrays,
        directions,
        base_vector_field={"scaled_primitive_rate_per_s": np.zeros(5)},
        base_branch_state=({}, np.ones(1), {}),
    )

    assert calls == [
        (name, step)
        for name in directions
        for step in wp10c8j.INDEPENDENT_VECTOR_FIELD_JVP_STEPS
    ]
    assert wp10c8j.INDEPENDENT_VECTOR_FIELD_JVP_STEPS == (
        5.0e-4,
        1.0e-3,
        3.0e-3,
    )
    assert wp10c8j.BASE_INDEPENDENT_VECTOR_FIELD_JVP_STEP == 1.0e-3
    assert row["centered_difference_steps"] == (
        wp10c8j.INDEPENDENT_VECTOR_FIELD_JVP_STEPS
    )
    assert row["binding_smooth_direction_names"] == tuple(directions)
    assert row["passed"]


def test_fresh_strict_jvp_reserves_nonzero_tied_rusanov_jump_but_allows_zero(
    monkeypatch,
) -> None:
    initial, vector, operator_arrays = _fresh_jvp_fixture()
    monkeypatch.setattr(
        wp10c8j,
        "unpack_causal_five_field_state",
        lambda _vector, _n: SimpleNamespace(primitives=np.zeros((1, 5))),
    )

    def audit(
        _initial,
        _vector,
        _evolving,
        direction,
        *,
        direction_name,
        **_kwargs,
    ):
        jump = 0.0 if direction_name == "exact_zero" else 1.0e-300
        return _mock_fresh_jvp_audit_result(direction, jump=jump)

    monkeypatch.setattr(
        wp10c8j.wp10c8i,
        "_independent_vector_field_jvp_audit",
        audit,
    )

    _arrays, row = wp10c8j._fresh_independent_vector_field_jvp_contract(
        initial,
        vector,
        np.eye(5),
        operator_arrays,
        {
            "exact_zero": np.eye(5)[0],
            "small_nonzero": np.eye(5)[1],
        },
        base_vector_field={"scaled_primitive_rate_per_s": np.zeros(5)},
        base_branch_state=({}, np.ones(1), {}),
    )

    assert row["binding_smooth_direction_names"] == ("exact_zero",)
    assert row["rusanov_reserved_direction_names"] == ("small_nonzero",)
    assert row["directions"]["exact_zero"][
        "rusanov_smooth_or_exact_zero_at_every_step"
    ]
    assert not row["directions"]["small_nonzero"][
        "rusanov_smooth_or_exact_zero_at_every_step"
    ]
    assert row["passed"]


def test_fresh_strict_jvp_numerical_failure_is_binding(monkeypatch) -> None:
    initial, vector, operator_arrays = _fresh_jvp_fixture()
    monkeypatch.setattr(
        wp10c8j,
        "unpack_causal_five_field_state",
        lambda _vector, _n: SimpleNamespace(primitives=np.zeros((1, 5))),
    )
    monkeypatch.setattr(
        wp10c8j.wp10c8i,
        "_independent_vector_field_jvp_audit",
        lambda *_args, **_kwargs: _mock_fresh_jvp_audit_result(
            np.eye(5)[0],
            numerical_passed=False,
        ),
    )

    _arrays, row = wp10c8j._fresh_independent_vector_field_jvp_contract(
        initial,
        vector,
        np.eye(5),
        operator_arrays,
        {"bad_smooth_direction": np.eye(5)[0]},
        base_vector_field={"scaled_primitive_rate_per_s": np.zeros(5)},
        base_branch_state=({}, np.ones(1), {}),
    )

    assert row["binding_smooth_direction_names"] == (
        "bad_smooth_direction",
    )
    assert not row["directions"]["bad_smooth_direction"][
        "numerical_jvp_passed_at_every_step"
    ]
    assert not row["passed"]


def _patch_separated_scan_contract_fixture(
    monkeypatch,
    *,
    fresh_jvp_passed: bool = True,
    cached_jvp_passed: bool = True,
    selected_component_defect: float = 0.0,
    diagnostic_component_defect: float = 0.0,
) -> tuple[dict, np.ndarray, dict[str, np.ndarray]]:
    size = 5
    identity = np.eye(size)
    monkeypatch.setattr(
        wp10c8j,
        "unpack_causal_five_field_state",
        lambda _vector, _n: SimpleNamespace(primitives=np.zeros((1, 5))),
    )
    monkeypatch.setattr(
        wp10c8j,
        "_normalized_directions",
        lambda *_args: {"one": np.eye(size)[0]},
    )
    monkeypatch.setattr(
        wp10c8j,
        "_fresh_base_vector_field",
        lambda *_args, **_kwargs: (
            np.zeros(size),
            {"scaled_primitive_rate_per_s": np.ones(size)},
            ({}, np.ones(1), {}),
            {"passed": True},
        ),
    )
    monkeypatch.setattr(
        wp10c8j,
        "causal_five_field_reduced_stationary_jacobian",
        lambda *_args, **_kwargs: {
            "stationary_reduced_scaled_jacobian": -2.0 * identity
        },
    )
    monkeypatch.setattr(
        wp10c8j,
        "causal_five_field_reduced_storage_matrices",
        lambda *_args, **_kwargs: {
            "descriptor_reduced_scaled_matrix": identity,
            "conserved_descriptor_reduced_scaled_matrix": 0.75 * identity,
            "vertical_descriptor_reduced_scaled_matrix": 0.25 * identity,
            "maximum_scaled_component_reconstruction_defect": 0.0,
        },
    )

    def storage_rate(
        *_args,
        outer_step=wp10c8j.BASE_OUTER_DIFFERENCE_STEP,
        action_step=wp10c8j.BASE_VERTICAL_ACTION_DIFFERENCE_STEP,
        **_kwargs,
    ):
        selected = bool(
            outer_step == wp10c8j.BASE_OUTER_DIFFERENCE_STEP
            and action_step == wp10c8j.BASE_VERTICAL_ACTION_DIFFERENCE_STEP
        )
        return {
            "storage_rate_derivative_scaled_matrix": 0.5 * identity,
            "conserved_storage_rate_derivative_scaled_matrix": (
                0.4 * identity
            ),
            "vertical_storage_rate_derivative_scaled_matrix": (
                0.1 * identity
            ),
            "maximum_scaled_component_reconstruction_defect": (
                selected_component_defect
                if selected
                else diagnostic_component_defect
            ),
        }

    monkeypatch.setattr(
        wp10c8j,
        "_direct_action_storage_rate_result",
        storage_rate,
    )
    monkeypatch.setattr(
        wp10c8j,
        "_repaired_generator_jvp_contract",
        lambda *_args, **_kwargs: ({}, {"passed": cached_jvp_passed}),
    )
    monkeypatch.setattr(
        wp10c8j,
        "_fresh_independent_vector_field_jvp_contract",
        lambda *_args, **_kwargs: ({}, {"passed": fresh_jvp_passed}),
    )
    initial = {"state": SimpleNamespace(n_cells=1), "context": object()}
    operator_arrays = {
        "primitive_column_scales": np.ones(size),
        "conservation_row_scales": np.ones(size),
        "scaled_primitive_rate": np.ones(size),
        "dynamic": 1.5 * identity,
        "storage_rate_derivative": 0.5 * identity,
        **{
            f"generator_inner_storage_fd_dynamic_{step:.0e}": (
                1.5 * identity
            )
            for step in wp10c8j.INNER_DIFFERENCE_STEPS
        },
        **{
            f"generator_outer_storage_rate_fd_dynamic_{step:.0e}": (
                1.5 * identity
            )
            for step in wp10c8j.LEGACY_OUTER_DIFFERENCE_STEPS
        },
        **{
            f"generator_vertical_action_fd_dynamic_{step:.0e}": (
                1.5 * identity
            )
            for step in wp10c8j.LEGACY_VERTICAL_ACTION_DIFFERENCE_STEPS
        },
    }
    return initial, np.zeros(20), operator_arrays


def test_separated_scan_fresh_jvp_is_binding_but_cached_jvp_is_diagnostic(
    monkeypatch,
) -> None:
    initial, vector, operator_arrays = _patch_separated_scan_contract_fixture(
        monkeypatch,
        fresh_jvp_passed=True,
        cached_jvp_passed=False,
    )

    _arrays, cached_failed = wp10c8j._separated_tangent_scan(
        initial,
        vector,
        operator_arrays,
        {},
    )
    assert not cached_failed["repaired_independent_vector_field_jvp"][
        "passed"
    ]
    assert cached_failed["passed"]

    monkeypatch.setattr(
        wp10c8j,
        "_fresh_independent_vector_field_jvp_contract",
        lambda *_args, **_kwargs: ({}, {"passed": False}),
    )
    _arrays, fresh_failed = wp10c8j._separated_tangent_scan(
        initial,
        vector,
        operator_arrays,
        {},
    )
    assert not fresh_failed[
        "fresh_full_direction_independent_vector_field_jvp"
    ]["passed"]
    assert not fresh_failed["passed"]


def test_direct_ladder_binds_generator_frobenius_not_raw_named_jvp(
    monkeypatch,
) -> None:
    initial, vector, operator_arrays = _patch_separated_scan_contract_fixture(
        monkeypatch,
    )

    def scan(
        _variants,
        *,
        base_key,
        directions,
        row_kind,
    ):
        del directions, row_kind
        direct_or_legacy = base_key != (
            f"{wp10c8j.BASE_INNER_DIFFERENCE_STEP:.0e}"
        )
        return {
            "passed": not direct_or_legacy,
            "comparisons": {
                "neighbor": {
                    "passed": not direct_or_legacy,
                    "frobenius_passed": True,
                    "deterministic_physical_jvp_passed": (
                        not direct_or_legacy
                    ),
                }
            },
        }

    monkeypatch.setattr(wp10c8j, "_scan_variants", scan)

    _arrays, row = wp10c8j._separated_tangent_scan(
        initial,
        vector,
        operator_arrays,
        {},
    )

    assert not row["block_scans"]["direct_total_dm_outer"]["passed"]
    assert not row["generator_scans"]["direct_dm_action"]["passed"]
    assert row["direct_full_matrix_step_scans_passed"]
    assert row["binding_step_scans_passed"]
    assert row["passed"]


def _mock_step_scan_result(
    *,
    passed: bool,
    frobenius_passed: bool,
) -> dict:
    return {
        "passed": passed,
        "comparisons": {
            "neighbor": {
                "passed": passed,
                "frobenius_passed": frobenius_passed,
                "deterministic_physical_jvp_passed": passed,
            }
        },
    }


def test_raw_dm_frobenius_failure_is_diagnostic_when_generator_passes(
    monkeypatch,
) -> None:
    initial, vector, operator_arrays = _patch_separated_scan_contract_fixture(
        monkeypatch,
    )
    inner_key = f"{wp10c8j.BASE_INNER_DIFFERENCE_STEP:.0e}"

    def scan(
        variants,
        *,
        base_key,
        directions,
        row_kind,
    ):
        del directions, row_kind
        if base_key == inner_key:
            return _mock_step_scan_result(
                passed=True,
                frobenius_passed=True,
            )
        base = np.asarray(variants[base_key], dtype=float)
        is_raw_dm = np.isclose(np.trace(base) / base.shape[0], 0.5)
        return _mock_step_scan_result(
            passed=not is_raw_dm,
            frobenius_passed=not is_raw_dm,
        )

    monkeypatch.setattr(wp10c8j, "_scan_variants", scan)

    _arrays, row = wp10c8j._separated_tangent_scan(
        initial,
        vector,
        operator_arrays,
        {},
    )

    assert not row["direct_raw_dm_full_matrix_step_scans_passed"]
    assert row["direct_generator_full_matrix_step_scans_passed"]
    assert row["direct_full_matrix_step_scans_passed"]
    assert row["binding_step_scans_passed"]
    assert row["passed"]


def test_generator_frobenius_failure_is_binding_even_when_raw_dm_passes(
    monkeypatch,
) -> None:
    initial, vector, operator_arrays = _patch_separated_scan_contract_fixture(
        monkeypatch,
    )
    inner_key = f"{wp10c8j.BASE_INNER_DIFFERENCE_STEP:.0e}"

    def scan(
        variants,
        *,
        base_key,
        directions,
        row_kind,
    ):
        del directions, row_kind
        if base_key == inner_key:
            return _mock_step_scan_result(
                passed=True,
                frobenius_passed=True,
            )
        base = np.asarray(variants[base_key], dtype=float)
        is_generator = np.isclose(np.trace(base) / base.shape[0], 1.5)
        return _mock_step_scan_result(
            passed=not is_generator,
            frobenius_passed=not is_generator,
        )

    monkeypatch.setattr(wp10c8j, "_scan_variants", scan)

    _arrays, row = wp10c8j._separated_tangent_scan(
        initial,
        vector,
        operator_arrays,
        {},
    )

    assert row["direct_raw_dm_full_matrix_step_scans_passed"]
    assert not row["direct_generator_full_matrix_step_scans_passed"]
    assert not row["direct_full_matrix_step_scans_passed"]
    assert not row["binding_step_scans_passed"]
    assert not row["passed"]


@pytest.mark.parametrize(
    ("selected_defect", "diagnostic_defect", "expected_passed"),
    (
        (0.0, 1.0, True),
        (1.0, 0.0, False),
    ),
)
def test_only_selected_base_component_reconstruction_defect_is_binding(
    monkeypatch,
    selected_defect,
    diagnostic_defect,
    expected_passed,
) -> None:
    initial, vector, operator_arrays = _patch_separated_scan_contract_fixture(
        monkeypatch,
        selected_component_defect=selected_defect,
        diagnostic_component_defect=diagnostic_defect,
    )

    _arrays, row = wp10c8j._separated_tangent_scan(
        initial,
        vector,
        operator_arrays,
        {},
    )

    assert row["selected_storage_rate_component_reconstruction_defect"] == (
        selected_defect
    )
    assert row["maximum_storage_rate_component_reconstruction_defect"] == (
        max(selected_defect, diagnostic_defect)
    )
    assert row["storage_rate_component_reconstruction_passed"] is (
        selected_defect
        <= wp10c8j.MAXIMUM_STORAGE_COMPONENT_RECONSTRUCTION_DEFECT
    )
    assert row["passed"] is expected_passed


def test_base_repaired_tangent_uses_fresh_all_direction_contract(
    monkeypatch,
) -> None:
    size = 5
    identity = np.eye(size)
    initial = {"state": SimpleNamespace(n_cells=1), "context": object()}
    vector = np.zeros(20)
    declared = {
        "thermal": np.eye(size)[0],
        "outer_transport": np.eye(size)[1],
    }
    monkeypatch.setattr(
        wp10c8j,
        "_normalized_directions",
        lambda *_args: declared,
    )
    monkeypatch.setattr(
        wp10c8j,
        "_fresh_base_vector_field",
        lambda *_args, **_kwargs: (
            np.zeros(size),
            {"scaled_primitive_rate_per_s": np.ones(size)},
            ({}, np.ones(1), {}),
            {"passed": True},
        ),
    )
    monkeypatch.setattr(
        wp10c8j,
        "_direct_action_storage_rate_result",
        lambda *_args, **_kwargs: {
            "storage_rate_derivative_scaled_matrix": 0.5 * identity,
            "conserved_storage_rate_derivative_scaled_matrix": (
                0.4 * identity
            ),
            "vertical_storage_rate_derivative_scaled_matrix": (
                0.1 * identity
            ),
            "maximum_scaled_component_reconstruction_defect": 0.0,
        },
    )
    monkeypatch.setattr(
        wp10c8j,
        "causal_five_field_assemble_evolving_tangent",
        lambda *_args, **_kwargs: {
            "evolving_scaled_generator_per_s": identity,
            "maximum_scaled_generator_factorization_defect": 0.0,
        },
    )
    monkeypatch.setattr(
        wp10c8j,
        "_repaired_generator_jvp_contract",
        lambda *_args, **_kwargs: ({}, {"passed": False}),
    )
    seen = {}

    def fresh(
        _initial,
        _vector,
        _generator,
        _operator_arrays,
        directions,
        **_kwargs,
    ):
        seen["names"] = tuple(directions)
        return {}, {"passed": True}

    monkeypatch.setattr(
        wp10c8j,
        "_fresh_independent_vector_field_jvp_contract",
        fresh,
    )
    operator_arrays = {
        "primitive_column_scales": np.ones(size),
        "direct_vector_storage_descriptor": identity,
        "stationary_jacobian": -1.5 * identity,
    }

    _arrays, row = wp10c8j._base_repaired_tangent(
        initial,
        vector,
        operator_arrays,
        {},
    )

    assert seen["names"] == tuple(declared)
    assert not row["repaired_independent_vector_field_jvp"]["passed"]
    assert row["fresh_full_direction_independent_vector_field_jvp"][
        "passed"
    ]
    assert row["passed"]


def _valid_nonfull_certification_cache_payload(
    n_cells: int = 1,
) -> tuple[dict[str, np.ndarray], dict]:
    width = 5 * n_cells
    directions = {
        name: {"passed": True}
        for name in wp10c8j.LOCKED_PHYSICAL_DIRECTION_NAMES
    }
    fresh = {
        "directions": directions,
        "centered_difference_steps": (
            wp10c8j.INDEPENDENT_VECTOR_FIELD_JVP_STEPS
        ),
        "passed": True,
    }
    tangent = {
        "separated_scan_evaluated": False,
        "storage_rate_component_reconstruction_passed": True,
        "generator_factorization_passed": True,
        "fresh_base_vector_field_rate_contract": {"passed": True},
        "fresh_full_direction_independent_vector_field_jvp": fresh,
        "passed": True,
    }
    branch = {
        "levels": {
            "level": {
                "0": {"binding": False, "passed": False},
            }
        },
        "all_rows_binding": False,
        "passed": False,
    }
    metadata = {
        "separated_tangent": tangent,
        "rusanov_finite_neighborhood": branch,
        "passed": False,
    }
    arrays = {
        "repaired_dynamic": np.eye(width),
        "repaired_storage_rate_derivative": np.eye(width),
        "repaired_conserved_storage_rate_derivative": np.eye(width),
        "repaired_vertical_storage_rate_derivative": np.eye(width),
    }
    for direction_name in wp10c8j.LOCKED_PHYSICAL_DIRECTION_NAMES:
        for step in wp10c8j.INDEPENDENT_VECTOR_FIELD_JVP_STEPS:
            prefix = (
                f"fresh_{direction_name}_step_{step:.0e}_"
                "independent_vector_field_jvp_"
            )
            for suffix in (
                "direction",
                "direct",
                "forward",
                "backward",
                "predicted",
                "base_rate",
            ):
                arrays[prefix + suffix] = np.zeros(width)
    return arrays, metadata


def test_certification_cache_rejects_metadata_only_payload() -> None:
    _arrays, metadata = _valid_nonfull_certification_cache_payload()

    with pytest.raises(RuntimeError, match="absent or malformed"):
        wp10c8j._validate_certification_cache_payload(
            {},
            metadata,
            n_cells=1,
            label="t_0p025",
        )


def test_certification_cache_rejects_inconsistent_decision() -> None:
    arrays, metadata = _valid_nonfull_certification_cache_payload()
    metadata["passed"] = True

    with pytest.raises(RuntimeError, match="top-level decision"):
        wp10c8j._validate_certification_cache_payload(
            arrays,
            metadata,
            n_cells=1,
            label="t_0p025",
        )


def test_certification_cache_accepts_complete_consistent_payload() -> None:
    arrays, metadata = _valid_nonfull_certification_cache_payload()

    wp10c8j._validate_certification_cache_payload(
        arrays,
        metadata,
        n_cells=1,
        label="t_0p025",
    )
