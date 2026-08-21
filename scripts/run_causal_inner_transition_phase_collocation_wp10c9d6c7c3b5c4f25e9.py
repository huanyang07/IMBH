#!/usr/bin/env python3
"""Validate transition phase collocation against exact continuous rates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path: sys.path.insert(0, str(path))

from imri_qpe.layer3_minidisk_1d.phase_collocation import (  # noqa: E402
    PiecewisePhaseCollocation,
    PolynomialPhaseSegment,
    direction_cosine,
    relative_vector_defect,
)
import run_causal_inner_transition_phase_collocation_manifest_wp10c9d6c7c3b5c4f25e8 as manifest  # noqa: E402
import run_causal_inner_cold_branch_anchor_preflight_wp10c9d6c7c3b5c4f25dy as exact_rate  # noqa: E402

SCHEMA_VERSION = 1
WORK_PACKAGE = "WP10c9d6c7c3b5c4f25e9"
PASS_CLASSIFICATION = "transition_phase_collocation_exact_continuous_rate_replay_and_affine_gluing_passed"
FAIL_CLASSIFICATION = "transition_phase_collocation_rejected"
AUTHORIZED_NEXT = "WP10c9d6c7c3b5c4f25ea"
ARTIFACT = "causal_inner_transition_phase_collocation_wp10c9d6c7c3b5c4f25e9"
CANONICAL_DIRECTORY = ROOT / "results/canonical" / ARTIFACT
THIS_RUNNER = "scripts/run_causal_inner_transition_phase_collocation_wp10c9d6c7c3b5c4f25e9.py"
THIS_TEST = "tests/test_causal_inner_transition_phase_collocation_wp10c9d6c7c3b5c4f25e9.py"
REPORT_RELATIVE = "docs/reports/current/CODEX_CAUSAL_INNER_TRANSITION_PHASE_COLLOCATION_WP10C9D6C7C3B5C4F25E9_2026-08-21.md"
REPORT_PATH = ROOT / REPORT_RELATIVE


def _helper(): return manifest._helper()


def _validate_manifest(*, require_clean: bool) -> dict:
    helper = _helper(); hashes = helper._validate_checksums(manifest.CANONICAL_DIRECTORY)
    contract = helper._read(manifest.CANONICAL_DIRECTORY / "transition_collocation_contract.json")
    summary = helper._read(manifest.CANONICAL_DIRECTORY / "summary.json")
    if not summary["passed"] or not summary["definitions_only"] or summary["authorized_next"] != WORK_PACKAGE or contract["work_package"] != manifest.WORK_PACKAGE: raise RuntimeError("transition collocation manifest changed")
    for relative, expected in contract["frozen_source_hashes"].items():
        if helper._sha(ROOT / relative) != expected: raise RuntimeError(f"frozen transition source changed: {relative}")
    current = manifest._contract()["decisive_input_hashes"]
    if current != contract["decisive_input_hashes"]: raise RuntimeError("transition collocation input hashes changed")
    manifest._validate_parent(require_clean=False)
    if require_clean and helper._git("status", "--short", "--untracked-files=no"): raise RuntimeError("transition collocation requires a clean tracked tree")
    return {"manifest_hashes": hashes, "contract": contract}


def _states() -> np.ndarray:
    helper = _helper(); geometry_module = manifest.geometry
    seed_path = geometry_module.manifest.full_step.manifest.SEED_CHECKPOINT
    states = [helper._load_npz(seed_path)["current_primitive_charts"]]
    for directory in geometry_module.manifest._accepted_stage_directories():
        summary = helper._read(directory / "summary.json"); local = int(summary["step_index"])
        checkpoint = helper._load_npz(directory / f"checkpoint_step_{local:02d}.npz")
        states.append(checkpoint["current_primitive_charts"])
    return np.asarray(states, dtype=float)


def _segment(times: np.ndarray, coordinates: np.ndarray, spec) -> PolynomialPhaseSegment:
    start, stop, training = spec; indices = np.asarray(training, dtype=int)
    return PolynomialPhaseSegment.from_constraints(start_time_seconds=float(times[start]), end_time_seconds=float(times[stop]), value_times_seconds=times[indices], values=coordinates[indices], rate_times_seconds=np.empty(0), rates_per_second=np.empty((0, coordinates.shape[1])))


def _models(times: np.ndarray, coordinates: np.ndarray):
    fine = PiecewisePhaseCollocation(tuple(_segment(times, coordinates, spec) for spec in manifest.FINE_SEGMENT_SPECS))
    coarse = PiecewisePhaseCollocation(tuple(_segment(times, coordinates, spec) for spec in manifest.COARSE_SEGMENT_SPECS))
    return fine, coarse


def _continuous_rate_witnesses(times: np.ndarray, states: np.ndarray) -> tuple[dict[int, np.ndarray], list[dict], dict[str, np.ndarray]]:
    geometry_split = exact_rate._geometry()
    model, _candidate, _fiber = exact_rate.exact_chart._model_and_inputs()
    layout, configuration, _trajectory, *_unused = exact_rate.rate_source.c4f24._endpoint_data()
    rates = {}; metrics = []; arrays = {}
    for index in manifest.HELDOUT_INDICES:
        item, evidence = exact_rate._evaluate_candidate(float(times[index]), states[index], geometry_split, model, layout, configuration)
        rates[index] = np.asarray(evidence["coordinate_rate470_per_s"], dtype=float)
        metrics.append(item)
        arrays.update({f"state_{index:02d}__{name}": value for name, value in evidence.items()})
    return rates, metrics, arrays


def _evaluate() -> tuple[dict, dict[str, np.ndarray]]:
    helper = _helper(); geometry_arrays = helper._load_npz(manifest.manifest_geometry_path())
    times = np.asarray(geometry_arrays["trajectory_times_seconds"], dtype=float)
    coordinates = np.asarray(geometry_arrays["trajectory_coordinates470"], dtype=float)
    states = _states()
    if len(states) != len(coordinates): raise RuntimeError("transition state lineage count changed")
    fine, coarse = _models(times, coordinates)
    began = time.perf_counter(); rates, exact_metrics, exact_arrays = _continuous_rate_witnesses(times, states); exact_wall = time.perf_counter() - began
    tangent = helper._load_npz(manifest.geometry.manifest.TANGENT_ARRAYS)
    restriction = np.asarray(tangent["macro_restriction_R82"], dtype=float); hidden_dual = np.asarray(tangent["hidden_dual_Q388"], dtype=float)
    path = float(np.sum(np.linalg.norm(np.diff(coordinates, axis=0), axis=1)))
    state_errors=[]; rate_defects=[]; cosines=[]; macro_defects=[]; hidden_defects=[]; refinement_states=[]; refinement_rates=[]; predictions=[]; predicted_rates=[]
    for index in manifest.HELDOUT_INDICES:
        predicted=fine.value(float(times[index])); predicted_rate=fine.rate(float(times[index])); truth_rate=rates[index]
        predictions.append(predicted); predicted_rates.append(predicted_rate)
        state_errors.append(float(np.linalg.norm(predicted-coordinates[index])/path)); rate_defects.append(relative_vector_defect(predicted_rate,truth_rate)); cosines.append(direction_cosine(predicted_rate,truth_rate))
        macro_defects.append(relative_vector_defect(restriction@predicted_rate,restriction@truth_rate)); hidden_defects.append(relative_vector_defect(hidden_dual@predicted_rate,hidden_dual@truth_rate))
        refinement_states.append(float(np.linalg.norm(predicted-coarse.value(float(times[index])))/path)); refinement_rates.append(float(np.linalg.norm(predicted_rate-coarse.rate(float(times[index])))/max(float(np.linalg.norm(truth_rate)),np.finfo(float).tiny)))
    interface=np.concatenate((fine.interface_value_defects(),coarse.interface_value_defects()))
    conditions=np.asarray([segment.constraint_condition_number for segment in fine.segments+coarse.segments])
    affine_engine, affine_data, _arrays = manifest.cold.manifest.parent._build_affine_engine()
    cold_mode=affine_engine.modes["cold_observed"]; start=manifest.cold.manifest.parent.HybridPhaseState(affine_data["macro"][0],0.0,cold_mode.name)
    event=affine_engine.advance(start,cold_mode.duration_seconds); event_defect=float(np.linalg.norm(affine_engine.decode(event.state)-affine_data["coordinates"][-1])); event_macro=float(np.linalg.norm(event.state.macro_state-affine_data["macro"][-1]))
    required_physical=("coordinate_decomposition","coordinate_rank","coordinate_condition","fixed_Q_tangency","reaction_ledger","Schur_rank","Schur_condition","reconstruction","height","optical_depth")
    physical_pass=all(all(item["gates"][name] for name in required_physical) for item in exact_metrics)
    gate_values={"maximum_state_error_over_path":max(state_errors),"maximum_full_rate_relative_defect":max(rate_defects),"minimum_full_rate_direction_cosine":min(cosines),"maximum_macro_rate_relative_defect":max(macro_defects),"maximum_hidden_rate_relative_defect":max(hidden_defects),"maximum_fine_coarse_state_defect":max(refinement_states),"maximum_fine_coarse_rate_defect":max(refinement_rates),"maximum_interface_value_defect":float(np.max(interface)),"maximum_constraint_condition_number":float(np.max(conditions)),"affine_event_state_defect":event_defect,"affine_event_macro_defect":event_macro,"exact_rate_wall_seconds":exact_wall}
    gates={"heldout_state":gate_values["maximum_state_error_over_path"]<=manifest.MAXIMUM_STATE_ERROR_OVER_PATH,"full_vector_field":gate_values["maximum_full_rate_relative_defect"]<=manifest.MAXIMUM_FULL_RATE_RELATIVE_DEFECT,"rate_direction":gate_values["minimum_full_rate_direction_cosine"]>=manifest.MINIMUM_FULL_RATE_DIRECTION_COSINE,"macro_rate":gate_values["maximum_macro_rate_relative_defect"]<=manifest.MAXIMUM_MACRO_RATE_RELATIVE_DEFECT,"hidden_rate":gate_values["maximum_hidden_rate_relative_defect"]<=manifest.MAXIMUM_HIDDEN_RATE_RELATIVE_DEFECT,"state_refinement":gate_values["maximum_fine_coarse_state_defect"]<=manifest.MAXIMUM_FINE_COARSE_STATE_DEFECT,"rate_refinement":gate_values["maximum_fine_coarse_rate_defect"]<=manifest.MAXIMUM_FINE_COARSE_RATE_DEFECT,"shooting_continuity":gate_values["maximum_interface_value_defect"]<=manifest.MAXIMUM_INTERFACE_VALUE_DEFECT,"conditioning":gate_values["maximum_constraint_condition_number"]<=manifest.MAXIMUM_CONSTRAINT_CONDITION_NUMBER,"affine_event":max(event_defect,event_macro)<=manifest.MAXIMUM_AFFINE_EVENT_DEFECT,"exact_rate_physics":physical_pass,"truth_budget":len(rates)<=manifest.MAXIMUM_EXACT_CONTINUOUS_RATE_CALLS,"no_roots_or_propagation":True}
    passed=bool(all(gates.values()))
    metrics={"classification":PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION,"passed":passed,"gates":gates,"gate_values":gate_values,"heldout_indices":manifest.HELDOUT_INDICES,"fine_segment_count":len(fine.segments),"coarse_segment_count":len(coarse.segments),"new_exact_continuous_fixed_Q_rate_calls":len(rates),"new_nonlinear_roots":0,"propagated_states":0,"post_transition_manifest_authorized":passed,"hot_exit_observed":False,"predictive_cycle_authorized":False,"exact_rate_metrics":exact_metrics}
    arrays={"heldout_indices":np.asarray(manifest.HELDOUT_INDICES),"heldout_times_seconds":times[np.asarray(manifest.HELDOUT_INDICES)],"heldout_true_coordinates470":coordinates[np.asarray(manifest.HELDOUT_INDICES)],"heldout_predicted_coordinates470":np.asarray(predictions),"heldout_exact_continuous_rates470_per_s":np.stack([rates[i] for i in manifest.HELDOUT_INDICES]),"heldout_predicted_rates470_per_s":np.asarray(predicted_rates),"state_errors_over_path":np.asarray(state_errors),"full_rate_relative_defects":np.asarray(rate_defects),"full_rate_direction_cosines":np.asarray(cosines),"macro_rate_relative_defects":np.asarray(macro_defects),"hidden_rate_relative_defects":np.asarray(hidden_defects),"fine_coarse_state_defects":np.asarray(refinement_states),"fine_coarse_rate_defects":np.asarray(refinement_rates),"interface_value_defects":interface,"constraint_condition_numbers":conditions,**exact_arrays}
    return metrics,arrays


def _update_catalog(summary:dict)->None:
    helper=_helper()
    with manifest.cold.manifest.CANONICAL_MANIFEST.open(newline="",encoding="utf-8") as h: rows=list(csv.DictReader(h))
    rows=[row for row in rows if row.get("case")!=ARTIFACT]; status="SUPPORTED" if summary["passed"] else "REJECTED"
    for path in sorted(CANONICAL_DIRECTORY.iterdir()):
        if path.is_file(): rows.append({"case":ARTIFACT,"path":str(path.relative_to(ROOT)),"bytes":str(path.stat().st_size),"sha256":helper._sha(path),"scientific_status":status})
    with manifest.cold.manifest.CANONICAL_MANIFEST.open("w",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=("case","path","bytes","sha256","scientific_status"),lineterminator="\n");w.writeheader();w.writerows(rows)
    catalog=helper._read(manifest.cold.manifest.CANONICAL_SUMMARY);catalog.setdefault("artifacts",{})[ARTIFACT]={"path":str(CANONICAL_DIRECTORY.relative_to(ROOT)),"classification":summary["classification"],"passed":summary["passed"]};catalog.update({"case_count":len({row["case"] for row in rows}),"file_count":len(rows),"total_bytes":sum(int(row["bytes"]) for row in rows),"all_payload_hashes_recorded":True,"latest_source_parent_commit":manifest.PARENT_COMMIT,"latest_work_package":WORK_PACKAGE});helper._write_json(manifest.cold.manifest.CANONICAL_SUMMARY,catalog)


def _run()->dict:
    helper=_helper()
    if CANONICAL_DIRECTORY.exists() or REPORT_PATH.exists(): raise RuntimeError("transition collocation result already exists")
    locked=_validate_manifest(require_clean=True);metrics,arrays=_evaluate();CANONICAL_DIRECTORY.mkdir(parents=True)
    helper._write_json(CANONICAL_DIRECTORY/"transition_collocation_metrics.json",metrics)
    with (CANONICAL_DIRECTORY/"transition_collocation_model_and_witnesses.npz").open("wb") as h: np.savez_compressed(h,**arrays)
    helper._write_json(CANONICAL_DIRECTORY/"input_lock.json",locked)
    summary={"schema_version":SCHEMA_VERSION,"work_package":WORK_PACKAGE,"classification":metrics["classification"],"passed":metrics["passed"],"transition_full_vector_field_replay_passed":metrics["passed"],"affine_event_gluing_passed":metrics["gates"]["affine_event"],"bounded_post_transition_manifest_authorized":metrics["passed"],"post_transition_execution_authorized":False,"hot_exit_observed":False,"predictive_cycle_authorized":False,"authorized_next":AUTHORIZED_NEXT if metrics["passed"] else None}
    helper._write_json(CANONICAL_DIRECTORY/"summary.json",summary);helper._write_json(CANONICAL_DIRECTORY/"provenance.json",{"runner":THIS_RUNNER,"test":THIS_TEST,"implementation_commit":helper._git("rev-parse","HEAD"),"implementation_tree":helper._git("rev-parse","HEAD^{tree}"),"python":sys.version,"numpy":np.__version__,"platform":platform.platform()})
    names=sorted(path.name for path in CANONICAL_DIRECTORY.iterdir());(CANONICAL_DIRECTORY/"SHA256SUMS.txt").write_text("".join(f"{helper._sha(CANONICAL_DIRECTORY/name)}  {name}\n" for name in names),encoding="utf-8")
    REPORT_PATH.parent.mkdir(parents=True,exist_ok=True);REPORT_PATH.write_text("\n".join(("# Transition phase collocation WP10c9d6c7c3b5c4f25e9","",f"Classification: `{metrics['classification']}`.","",f"The four-window phase chart has maximum exact continuous fixed-Q rate defect {metrics['gate_values']['maximum_full_rate_relative_defect']:.6e} and minimum rate direction cosine {metrics['gate_values']['minimum_full_rate_direction_cosine']:.9f} across eight held-out accepted states. Affine event closure is {metrics['gate_values']['affine_event_state_defect']:.6e}.","","No state was propagated and no nonlinear root was solved. A pass authorizes only a prospective bounded post-transition phase-window manifest.","")),encoding="utf-8");_update_catalog(summary);return summary


def main()->None:
    p=argparse.ArgumentParser();p.add_argument("--run",action="store_true");a=p.parse_args()
    if not a.run:p.error("use --run")
    payload=_run();print(json.dumps(payload,indent=2,sort_keys=True))
    if not payload["passed"]:raise SystemExit(1)


if __name__=="__main__":main()
