"""Fail-closed schema and audits for cycle-wide physical AP input bundles.

Passing these checks establishes numerical and bookkeeping consistency only.
The metadata flag ``physical_model_complete`` and the absence of
``synthetic_fixture`` are separately required before a bundle can be used as
physical cycle data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


Array = np.ndarray
TWO_PI = 2.0 * np.pi


def _array(payload: Mapping[str, object], name: str, *, ndim: int, dtype=float) -> Array:
    if name not in payload:
        raise ValueError(f"missing bundle array: {name}")
    value = np.asarray(payload[name], dtype=dtype)
    if value.ndim != ndim or (dtype is not str and np.any(~np.isfinite(value))):
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    return value


def _relative(left: Array, right: Array) -> float:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    return float(
        np.linalg.norm(a - b)
        / max(np.linalg.norm(a), np.linalg.norm(b), np.finfo(float).tiny)
    )


def _maximum_sample_relative(left: Array, right: Array) -> float:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    if a.shape != b.shape or a.ndim < 1:
        raise ValueError("sample-relative arrays disagree")
    flattened_a = a.reshape((-1, a.shape[-1]))
    flattened_b = b.reshape((-1, b.shape[-1]))
    numerator = np.linalg.norm(flattened_a - flattened_b, axis=1)
    denominator = np.maximum(
        np.maximum(np.linalg.norm(flattened_a, axis=1), np.linalg.norm(flattened_b, axis=1)),
        np.finfo(float).tiny,
    )
    return float(np.max(numerator / denominator))


def _indices(metadata: Mapping[str, object], name: str, upper: int) -> Array:
    values = np.asarray(metadata.get(name, ()), dtype=int)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError(f"{name} must be a nonempty index vector")
    if np.any(values < 0) or np.any(values >= int(upper)) or len(np.unique(values)) != len(values):
        raise ValueError(f"{name} contains invalid or duplicate indices")
    return values


def _disjoint(training: Array, heldout: Array, name: str) -> None:
    if np.intersect1d(training, heldout).size:
        raise ValueError(f"{name} training and heldout indices overlap")


@dataclass(frozen=True)
class CyclePhysicalInputBundleAudit:
    physical_model_complete: bool
    synthetic_fixture: bool
    phase_count: int
    invariant_count: int
    mode_count: int
    branch_anchor_count: int
    event_count: int
    integrated_period_relative_defect: float
    maximum_periodic_endpoint_relative_defect: float
    maximum_forcing_ledger_relative_defect: float
    maximum_branch_invariant_relative_defect: float
    maximum_branch_radial_symmetry_defect: float
    maximum_branch_source_positive_eigenvalue: float
    minimum_branch_spectral_gap_per_second: float
    maximum_event_reset_ledger_relative_defect: float
    maximum_event_constitutive_null_relative_defect: float
    checkpoint_roundtrip_bitwise: bool

    @property
    def structurally_passed(self) -> bool:
        return bool(
            self.phase_count >= 5
            and self.invariant_count >= 2
            and self.mode_count >= 1
            and self.branch_anchor_count >= 2
            and self.event_count >= 2
            and self.integrated_period_relative_defect <= 2.0e-3
            and self.maximum_periodic_endpoint_relative_defect <= 2.0e-10
            and self.maximum_forcing_ledger_relative_defect <= 2.0e-12
            and self.maximum_branch_invariant_relative_defect <= 2.0e-12
            and self.maximum_branch_radial_symmetry_defect <= 2.0e-12
            and self.maximum_branch_source_positive_eigenvalue <= 2.0e-12
            and self.minimum_branch_spectral_gap_per_second > 0.0
            and self.maximum_event_reset_ledger_relative_defect <= 2.0e-12
            and self.maximum_event_constitutive_null_relative_defect <= 2.0e-12
            and self.checkpoint_roundtrip_bitwise
        )

    @property
    def physically_usable(self) -> bool:
        return bool(
            self.structurally_passed
            and self.physical_model_complete
            and not self.synthetic_fixture
        )


def validate_cycle_physical_input_bundle(
    metadata: Mapping[str, object],
    driver: Mapping[str, object],
    branch: Mapping[str, object],
    events: Mapping[str, object],
    heldout: Mapping[str, object],
    *,
    conservation_map,
    require_physical: bool = True,
    checkpoint_roundtrip_bitwise: bool = True,
) -> CyclePhysicalInputBundleAudit:
    """Validate one complete acquisition bundle or raise ``ValueError``."""

    required_metadata = (
        "schema_version",
        "physical_model_id",
        "physical_model_complete",
        "synthetic_fixture",
        "period_seconds",
        "unit_system",
        "source_citations",
        "source_code_commit",
        "split_frozen_before_fit",
    )
    missing_metadata = [name for name in required_metadata if name not in metadata]
    if missing_metadata:
        raise ValueError(f"missing bundle metadata: {missing_metadata}")
    if int(metadata["schema_version"]) != 1:
        raise ValueError("unsupported physical-input schema")
    if not str(metadata["physical_model_id"]).strip():
        raise ValueError("physical_model_id is empty")
    period = float(metadata["period_seconds"])
    if not np.isfinite(period) or period <= 0.0:
        raise ValueError("period_seconds must be positive and finite")
    if str(metadata["unit_system"]).lower() != "cgs":
        raise ValueError("physical input bundle must declare cgs units")
    citations = metadata["source_citations"]
    if not isinstance(citations, (list, tuple)) or not citations or any(
        not str(value).strip() for value in citations
    ):
        raise ValueError("physical input bundle needs source citations")
    if len(str(metadata["source_code_commit"]).strip()) < 7:
        raise ValueError("physical input source commit is not identifiable")
    if metadata["split_frozen_before_fit"] is not True:
        raise ValueError("training/heldout split was not prospectively frozen")
    physical_complete = bool(metadata["physical_model_complete"])
    synthetic_fixture = bool(metadata["synthetic_fixture"])
    if require_physical and (not physical_complete or synthetic_fixture):
        raise ValueError("bundle is structural/synthetic rather than complete physical data")

    conservation = np.asarray(conservation_map, dtype=float)
    if conservation.shape != (4, 1232) or np.any(~np.isfinite(conservation)):
        raise ValueError("conservation_map must be finite with shape (4,1232)")
    if np.linalg.matrix_rank(conservation) != 4:
        raise ValueError("conservation_map is rank deficient")

    phase = _array(driver, "phase_nodes", ndim=1)
    rates = _array(driver, "phase_rate_per_second", ndim=1)
    invariants = _array(driver, "retained_invariant_nodes4", ndim=2)
    modes = _array(driver, "mode_labels", ndim=1, dtype=str)
    forcing = _array(driver, "slow_forcing1232_per_second", ndim=4)
    distributed = _array(driver, "distributed_source_ledger_rate4", ndim=4)
    boundary = _array(driver, "boundary_ledger_rate4", ndim=4)
    incoming = _array(driver, "outer_incoming_characteristics11", ndim=4)
    n_phase, n_q, n_mode = len(phase), len(invariants), len(modes)
    if n_phase < 5 or n_q < 2 or n_mode < 1:
        raise ValueError("driver bundle has insufficient phase/state/mode coverage")
    if phase[0] != 0.0 or phase[-1] != TWO_PI or np.any(np.diff(phase) <= 0.0):
        raise ValueError("phase nodes must increase exactly from zero to two pi")
    if rates.shape != (n_phase,) or np.any(rates <= 0.0):
        raise ValueError("phase rate must be positive at every phase node")
    if invariants.shape != (n_q, 4) or len(np.unique(invariants, axis=0)) != n_q:
        raise ValueError("retained invariant nodes are malformed or duplicated")
    if any(not value.strip() for value in modes) or len(np.unique(modes)) != n_mode:
        raise ValueError("mode labels must be nonempty and unique")
    common = (n_phase, n_q, n_mode)
    if forcing.shape != common + (1232,):
        raise ValueError("slow forcing has the wrong shape")
    if distributed.shape != common + (4,) or boundary.shape != common + (4,):
        raise ValueError("driver ledger components have the wrong shape")
    if incoming.shape != common + (11,):
        raise ValueError("outer incoming characteristic loading has the wrong shape")
    integrated_period = float(np.trapezoid(1.0 / rates, phase))
    period_defect = abs(integrated_period - period) / period
    periodic_defects = (
        _relative(rates[0], rates[-1]),
        _relative(forcing[0], forcing[-1]),
        _relative(distributed[0], distributed[-1]),
        _relative(boundary[0], boundary[-1]),
        _relative(incoming[0], incoming[-1]),
    )
    maximum_periodic = max(periodic_defects)
    realized_ledgers = np.einsum("as,pqms->pqma", conservation, forcing)
    ledger_defect = _maximum_sample_relative(realized_ledgers, distributed + boundary)
    if period_defect > 2.0e-3:
        raise ValueError("phase law does not integrate to the declared period")
    if maximum_periodic > 2.0e-10:
        raise ValueError("driver payload does not close periodically")
    if ledger_defect > 2.0e-12:
        raise ValueError("slow forcing does not close its physical ledger")

    phase_training = _indices(metadata, "training_phase_indices", n_phase)
    phase_heldout = _indices(metadata, "heldout_phase_indices", n_phase)
    q_training = _indices(metadata, "training_invariant_indices", n_q)
    q_heldout = _indices(metadata, "heldout_invariant_indices", n_q)
    _disjoint(phase_training, phase_heldout, "phase")
    _disjoint(q_training, q_heldout, "invariant")
    if not {0, n_phase - 1}.issubset(set(phase_training.tolist())):
        raise ValueError("periodic endpoint nodes must be training anchors")

    states = _array(branch, "anchor_states1232", ndim=2)
    anchor_phase = _array(branch, "anchor_phase", ndim=1)
    anchor_invariants = _array(branch, "anchor_invariants4", ndim=2)
    anchor_modes = _array(branch, "anchor_mode_index", ndim=1, dtype=int)
    radial = _array(branch, "radial_matrices112x11x11", ndim=4)
    source = _array(branch, "source_matrices112x11x11", ndim=4)
    anchor_forcing = _array(branch, "forcing1232_per_second", ndim=2)
    trust = _array(branch, "trust_radii", ndim=1)
    gaps = _array(branch, "stable_spectral_gaps_per_second", ndim=1)
    guard_margins = _array(branch, "guard_margins", ndim=2)
    tangents = _array(branch, "pseudo_arclength_tangents", ndim=2)
    n_anchor = len(states)
    expected_anchor = {
        "states": (n_anchor, 1232),
        "phase": (n_anchor,),
        "invariants": (n_anchor, 4),
        "modes": (n_anchor,),
        "radial": (n_anchor, 112, 11, 11),
        "source": (n_anchor, 112, 11, 11),
        "forcing": (n_anchor, 1232),
        "trust": (n_anchor,),
        "gaps": (n_anchor,),
        "tangents": (n_anchor, 1237),
    }
    actual_anchor = {
        "states": states.shape,
        "phase": anchor_phase.shape,
        "invariants": anchor_invariants.shape,
        "modes": anchor_modes.shape,
        "radial": radial.shape,
        "source": source.shape,
        "forcing": anchor_forcing.shape,
        "trust": trust.shape,
        "gaps": gaps.shape,
        "tangents": tangents.shape,
    }
    if n_anchor < 2 or actual_anchor != expected_anchor or guard_margins.shape[0] != n_anchor:
        raise ValueError("branch payload shapes disagree")
    if np.any(anchor_phase < 0.0) or np.any(anchor_phase > TWO_PI):
        raise ValueError("branch phase lies outside the cycle")
    if np.any(anchor_modes < 0) or np.any(anchor_modes >= n_mode):
        raise ValueError("branch mode index is invalid")
    if np.any(trust <= 0.0) or np.any(gaps <= 0.0):
        raise ValueError("branch trust radii and spectral gaps must be positive")
    if np.any(np.linalg.norm(tangents, axis=1) <= 0.0):
        raise ValueError("pseudo-arclength tangents must be nonzero")
    branch_realized = states @ conservation.T
    branch_closure = _maximum_sample_relative(branch_realized, anchor_invariants)
    radial_symmetry = float(np.max(np.abs(radial - np.swapaxes(radial, -1, -2))))
    source_symmetric = 0.5 * (source + np.swapaxes(source, -1, -2))
    source_positive = float(np.max(np.linalg.eigvalsh(source_symmetric)))
    source_nullities = np.count_nonzero(
        np.linalg.svd(source, compute_uv=False)
        <= 1.0e-11 * np.maximum(np.max(np.linalg.svd(source, compute_uv=False), axis=-1), 1.0)[..., None],
        axis=-1,
    )
    if branch_closure > 2.0e-12:
        raise ValueError("branch anchors do not close retained invariants")
    if radial_symmetry > 2.0e-12 or source_positive > 2.0e-12:
        raise ValueError("branch port/source structure is inadmissible")
    if np.any(source_nullities != 4):
        raise ValueError("branch source nullity is not four")
    branch_training = _indices(metadata, "training_branch_anchor_indices", n_anchor)
    branch_heldout = _indices(metadata, "heldout_branch_anchor_indices", n_anchor)
    _disjoint(branch_training, branch_heldout, "branch anchor")

    pre = _array(events, "pre_states1232", ndim=2)
    post = _array(events, "post_states1232", ndim=2)
    pre_q = _array(events, "pre_invariants4", ndim=2)
    event_phase = _array(events, "phase", ndim=1)
    source_mode = _array(events, "source_mode_index", ndim=1, dtype=int)
    destination_mode = _array(events, "destination_mode_index", ndim=1, dtype=int)
    duration = _array(events, "duration_seconds", ndim=1)
    impulses = _array(events, "integrated_ledger_impulse4", ndim=2)
    constitutive = _array(events, "ledger_null_constitutive_jump1232", ndim=2)
    guards = _array(events, "guard_value_and_direction", ndim=2)
    n_event = len(pre)
    if n_event < 2 or any(
        shape != expected
        for shape, expected in (
            (pre.shape, (n_event, 1232)),
            (post.shape, (n_event, 1232)),
            (pre_q.shape, (n_event, 4)),
            (event_phase.shape, (n_event,)),
            (source_mode.shape, (n_event,)),
            (destination_mode.shape, (n_event,)),
            (duration.shape, (n_event,)),
            (impulses.shape, (n_event, 4)),
            (constitutive.shape, (n_event, 1232)),
            (guards.shape, (n_event, 2)),
        )
    ):
        raise ValueError("event payload shapes disagree")
    if np.any(event_phase < 0.0) or np.any(event_phase > TWO_PI) or np.any(duration <= 0.0):
        raise ValueError("event phase/duration is invalid")
    if (
        np.any(source_mode < 0)
        or np.any(source_mode >= n_mode)
        or np.any(destination_mode < 0)
        or np.any(destination_mode >= n_mode)
        or np.any(source_mode == destination_mode)
    ):
        raise ValueError("event source/destination modes are invalid")
    if np.max(np.abs(guards[:, 0])) > 2.0e-10 or np.min(np.abs(guards[:, 1])) <= 1.0e-8:
        raise ValueError("event guards are not localized transverse crossings")
    pre_closure = _maximum_sample_relative(pre @ conservation.T, pre_q)
    reset_closure = _maximum_sample_relative((post - pre) @ conservation.T, impulses)
    constitutive_scale = np.maximum(np.linalg.norm(constitutive, axis=1), np.finfo(float).tiny)
    constitutive_null = float(
        np.max(np.linalg.norm(constitutive @ conservation.T, axis=1) / constitutive_scale)
    )
    if pre_closure > 2.0e-12 or reset_closure > 2.0e-12 or constitutive_null > 2.0e-12:
        raise ValueError("event reset or constitutive ledger does not close")
    event_training = _indices(metadata, "training_event_indices", n_event)
    event_heldout = _indices(metadata, "heldout_event_indices", n_event)
    _disjoint(event_training, event_heldout, "event")

    withheld_branch = _array(heldout, "withheld_branch_anchor_indices", ndim=1, dtype=int)
    withheld_events = _array(heldout, "withheld_event_indices", ndim=1, dtype=int)
    phase_windows = _array(heldout, "withheld_phase_windows", ndim=2)
    sequence_modes = _array(heldout, "sequence_mode_indices", ndim=1, dtype=int)
    sequence_ledgers = _array(heldout, "sequence_ledger_increments4", ndim=2)
    spatial_cells = int(np.asarray(heldout.get("spatial_truth_grid_cells", -1)).item())
    if (
        not np.array_equal(np.sort(withheld_branch), np.sort(branch_heldout))
        or not np.array_equal(np.sort(withheld_events), np.sort(event_heldout))
        or phase_windows.ndim != 2
        or phase_windows.shape[1] != 2
        or len(phase_windows) < 2
        or np.any(phase_windows[:, 0] < 0.0)
        or np.any(phase_windows[:, 1] > TWO_PI)
        or np.any(phase_windows[:, 1] <= phase_windows[:, 0])
        or len(sequence_modes) < 2
        or sequence_ledgers.shape != (len(sequence_modes) - 1, 4)
        or np.any(sequence_modes < 0)
        or np.any(sequence_modes >= n_mode)
        or spatial_cells < 112
    ):
        raise ValueError("heldout truth inventory is incomplete or inconsistent")

    audit = CyclePhysicalInputBundleAudit(
        physical_complete,
        synthetic_fixture,
        n_phase,
        n_q,
        n_mode,
        n_anchor,
        n_event,
        period_defect,
        maximum_periodic,
        ledger_defect,
        branch_closure,
        radial_symmetry,
        max(source_positive, 0.0),
        float(np.min(gaps)),
        reset_closure,
        constitutive_null,
        bool(checkpoint_roundtrip_bitwise),
    )
    if not audit.structurally_passed:
        raise ValueError("cycle physical input bundle failed structural gates")
    return audit


def save_cycle_physical_input_bundle(
    directory,
    metadata: Mapping[str, object],
    driver: Mapping[str, object],
    branch: Mapping[str, object],
    events: Mapping[str, object],
    heldout: Mapping[str, object],
) -> None:
    target = Path(directory)
    if target.exists():
        raise FileExistsError("cycle physical input bundle directory already exists")
    target.mkdir(parents=True)
    (target / "metadata.json").write_text(
        json.dumps(dict(metadata), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for name, payload in (
        ("driver.npz", driver),
        ("branch.npz", branch),
        ("events.npz", events),
        ("heldout_truth.npz", heldout),
    ):
        np.savez_compressed(target / name, **payload)


def load_cycle_physical_input_bundle(directory):
    source = Path(directory)
    expected = ("metadata.json", "driver.npz", "branch.npz", "events.npz", "heldout_truth.npz")
    if any(not (source / name).is_file() for name in expected):
        raise FileNotFoundError("cycle physical input bundle is incomplete")
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    payloads = []
    for name in expected[1:]:
        with np.load(source / name, allow_pickle=False) as payload:
            payloads.append({key: np.array(payload[key], copy=True) for key in payload.files})
    return metadata, *payloads


__all__ = [
    "CyclePhysicalInputBundleAudit",
    "load_cycle_physical_input_bundle",
    "save_cycle_physical_input_bundle",
    "validate_cycle_physical_input_bundle",
]
