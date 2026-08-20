from __future__ import annotations

import numpy as np

import run_causal_inner_fixed_q_fast_attractor_screen_wp10c9d6c7c3b5c4f25cb as f25cb


def test_frozen_manifest_authorizes_exact_screen():
    frozen = f25cb._validate_manifest(require_clean=False)
    assert frozen["summary"]["passed"]
    assert frozen["summary"]["authorized_next"] == f25cb.WORK_PACKAGE
    assert frozen["contract"]["mathematical_split"][
        "z_is_the_only_block_authorized_for_linear_elimination"
    ]


def test_fast_model_has_frozen_dimensions_and_elimination_closes_memory_rows():
    model = f25cb.FastAttractorModel()
    assert model.Azz.shape == (280, 280)
    assert model.linear_fast_matrix.shape == (308, 308)
    departure = f25cb.manifest._search_design()[1]
    memory = model.eliminated_memory(departure)
    fast_rate = model.full_fast_rate(memory, departure)
    assert memory.shape == (280,)
    assert np.linalg.norm(fast_rate[:280]) <= 1.0e-8


def test_naive_split_is_rejected_but_memory_split_is_hurwitz():
    metrics = f25cb._structure_metrics(f25cb.FastAttractorModel())
    assert metrics["stable_memory_spectral_abscissa_per_second"] <= -1.0
    assert metrics["stable_memory_nonnegative_eigenvalue_count"] == 0
    assert metrics["stable_memory_condition_number"] <= 1.0e6
    assert metrics["naive_eliminated_block_shape"] == [346, 346]
    assert metrics["naive_eliminated_block_coordinate_selection"].startswith(
        "cell_major_mapped_components_1_and_4"
    )
    assert metrics["naive_eliminated_block_nonnegative_eigenvalue_count"] > 0


def test_canonical_screen_if_present():
    if not f25cb.CANONICAL_DIRECTORY.exists():
        return
    f25cb._checksums(f25cb.CANONICAL_DIRECTORY)
    summary = f25cb._read(f25cb.CANONICAL_DIRECTORY / "summary.json")
    metrics = f25cb._read(f25cb.CANONICAL_DIRECTORY / "screen_metrics.json")
    assert summary["passed"]
    assert summary["classification"] in {
        f25cb.STABLE_CLASSIFICATION,
        f25cb.NONCLOSURE_CLASSIFICATION,
        f25cb.UNSTABLE_CLASSIFICATION,
    }
    assert summary["new_truth_calls"] == 0
    assert not summary["predictive_cycle_authorized"]
    assert not summary["reduced_slow_evolution_authorized"]
    assert all(metrics["decision"]["structure_checks"].values())


def test_no_screen_branch_directly_authorizes_cycle_evolution():
    frozen = f25cb._validate_manifest(require_clean=False)
    decision = frozen["contract"]["decision"]
    assert not decision["physical_microburst_authorized"]
    assert not decision["predictive_cycle_authorized"]
    assert not decision["reduced_slow_evolution_authorized"]
