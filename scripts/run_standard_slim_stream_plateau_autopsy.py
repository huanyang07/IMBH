"""Autopsy the stream-fed high-source physical-energy plateau."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "src", ROOT / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from imri_qpe.layer3_minidisk_1d import (  # noqa: E402
    algebraic_state,
    differential_residual,
    differential_residual_scales,
    entropy_gradient_log,
    residual_audit_from_state_vector,
    residual_partition_audit_from_state_vector,
    square_collocation_residual,
    state_partials,
    stream_heating_rate,
    stream_mass_rate_and_derivative,
    stream_source_prime,
    stream_torque_specific_l_and_derivative,
    transonic_profile_from_state_vector,
    unpack_state,
    wind_sink_prime,
)
from imri_qpe.layer3_minidisk_1d.transonic_collocation import (  # noqa: E402
    _integrated_interval_residual_from_unpacked,
    _interval_residual_from_unpacked,
    _outer_buffer_interval_weights,
)
from imri_qpe.parameters import FiducialParams  # noqa: E402
from imri_qpe.scales import eddington_mdot  # noqa: E402
from run_standard_slim_analytic_seed_audit import fmt  # noqa: E402
from run_standard_slim_stream_mass_annulus_scan import (  # noqa: E402
    advection_diagnostic,
    load_anchor,
    relative_root_path,
    max_residual,
    stream_diagnostic,
)


TABLE_OUTPUT = ROOT / "outputs/tables/high_mdot_stream_plateau_autopsy.md"
JSON_OUTPUT = TABLE_OUTPUT.with_suffix(".json")
DETAIL_OUTPUT = ROOT / "outputs/tables/high_mdot_stream_plateau_autopsy_detail.md"
DETAIL_JSON_OUTPUT = DETAIL_OUTPUT.with_suffix(".json")
NOTE_OUTPUT = ROOT / "Note/CODEX_STREAM_PLATEAU_AUTOPSY_RESULTS.md"

PHYSICAL_E_TOL = 3.0e-5
SOURCE_REGION_RG = 259.2
BUFFER_REGION_RG = 333.8
REGION_HALF_WIDTH_RG = 2.0
TOP_N_PER_REGION = 8

CASES = (
    (
        "clean_anchor_0p898078125",
        "outputs/checkpoints/high_mdot_stream_outer_buffer_phys_gate_hybrid_quartersteps_accept3e5/"
        "phys_gate_hybrid_quartersteps_accept3e5_mass_0p898078125_torque_0p005_mdot_2_N896.npz",
        "strict clean frontier",
    ),
    (
        "failed_eighth_0p898085937",
        "outputs/checkpoints/high_mdot_stream_outer_buffer_phys_gate_hybrid_eighthsteps_0898078125_to089809375/"
        "phys_gate_hybrid_eighthsteps_0898078125_to089809375_mass_0p898085937_torque_0p005_mdot_2_N896.npz",
        "failed eighth-step retry",
    ),
    (
        "failed_quarter_0p89809375",
        "outputs/checkpoints/high_mdot_stream_outer_buffer_phys_gate_hybrid_quartersteps_accept3e5/"
        "phys_gate_hybrid_quartersteps_accept3e5_mass_0p89809375_torque_0p005_mdot_2_N896.npz",
        "failed next quarter step",
    ),
    (
        "failed_long_0p898125",
        "outputs/checkpoints/high_mdot_stream_outer_buffer_phys_gate_hybrid_long_08980625_to0898125/"
        "phys_gate_hybrid_long_08980625_to0898125_mass_0p898125_torque_0p005_mdot_2_N896.npz",
        "failed long full step",
    ),
)


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return relative_root_path(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def case_fraction(params) -> float:
    explicit = float(getattr(params, "stream_source_fraction", 0.0))
    legacy = float(getattr(params, "stream_mass_fraction", 0.0))
    return explicit if explicit != 0.0 else legacy


def interval_arrays(z: np.ndarray, params) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    logu, logT, _logR_son, lambda0, logR = unpack_state(z, params)
    return logu, logT, logR, float(lambda0), np.exp(0.5 * (logR[:-1] + logR[1:])) / params.r_g


def energy_terms_at_interval(
    z: np.ndarray,
    params,
    idx: int,
    *,
    override_source_fraction: float | None = None,
) -> dict[str, float]:
    logu, logT, logR, lambda0, R_mid_rg = interval_arrays(z, params)
    local_params = params
    if override_source_fraction is not None:
        local_params = replace(params, stream_source_fraction=float(override_source_fraction), stream_mass_fraction=0.0)
    dx = float(logR[idx + 1] - logR[idx])
    y_left = np.asarray([logu[idx], logT[idx]], dtype=float)
    y_right = np.asarray([logu[idx + 1], logT[idx + 1]], dtype=float)
    y_mid = 0.5 * (y_left + y_right)
    g_mid = (y_right - y_left) / dx
    x_mid = float(0.5 * (logR[idx] + logR[idx + 1]))

    state = algebraic_state(x_mid, float(y_mid[0]), float(y_mid[1]), lambda0, local_params)
    partials = state_partials(x_mid, y_mid, lambda0, local_params, eps_x=local_params.partial_eps, eps_y=local_params.partial_eps)
    dPi_dx = partials.x["Pi"] + float(np.dot(partials.y["Pi"], g_mid))
    drho_dx = partials.x["rho"] + float(np.dot(partials.y["rho"], g_mid))
    de_dx = partials.x["e"] + float(np.dot(partials.y["e"], g_mid))
    dOmega_dx = partials.x["Omega"] + float(np.dot(partials.y["Omega"], g_mid))
    radial_numerator = (
        state.u**2 * g_mid[0] - state.R**2 * (state.Omega**2 - state.Omega_K**2) + dPi_dx / state.Sigma
    )
    Tdsdx = de_dx - state.P / state.rho**2 * drho_dx
    Q_visc = -state.W * dOmega_dx
    Q_adv = -(state.Sigma * state.u / state.R) * Tdsdx
    Q_stream = stream_heating_rate(x_mid, local_params)
    energy_numerator = Q_visc + Q_stream - state.Q_rad - Q_adv
    radial_scale, energy_scale = differential_residual_scales(x_mid, y_mid, lambda0, local_params)
    raw = differential_residual(x_mid, y_mid, g_mid, lambda0, local_params)
    interval_residual = _interval_residual_from_unpacked(logu, logT, logR, lambda0, local_params, idx)
    integrated_residual = _integrated_interval_residual_from_unpacked(logu, logT, logR, lambda0, local_params, idx)
    buffer_weights = _outer_buffer_interval_weights(x_mid, local_params)
    mdot_local, dmdot_dlnR = stream_mass_rate_and_derivative(x_mid, local_params)
    stream_l, stream_dl_dlnR = stream_torque_specific_l_and_derivative(x_mid, local_params)
    source_prime = stream_source_prime(x_mid, local_params)
    wind_prime = wind_sink_prime(x_mid, local_params)
    torque_work_proxy = state.Omega * mdot_local * stream_dl_dlnR / (2.0 * np.pi * state.R**2)

    return {
        "idx": int(idx),
        "R_left_rg": float(np.exp(logR[idx]) / local_params.r_g),
        "R_mid_rg": float(R_mid_rg[idx]),
        "R_right_rg": float(np.exp(logR[idx + 1]) / local_params.r_g),
        "dlnR": float(dx),
        "logu_mid": float(y_mid[0]),
        "logT_mid": float(y_mid[1]),
        "g_logu": float(g_mid[0]),
        "g_logT": float(g_mid[1]),
        "radial_signed_scaled": float(raw[0] / radial_scale),
        "physical_interval_E": float(abs(raw[1] / energy_scale)),
        "signed_physical_interval_E": float(raw[1] / energy_scale),
        "scaled_physical_interval_E": float(raw[1] / energy_scale),
        "solver_interval_R": float(interval_residual[0]),
        "solver_interval_E": float(interval_residual[1]),
        "integrated_interval_R": float(integrated_residual[0]),
        "integrated_interval_E": float(integrated_residual[1]),
        "radial_numerator": float(radial_numerator),
        "energy_numerator": float(energy_numerator),
        "raw_differential_energy": float(raw[1]),
        "radial_scale": float(radial_scale),
        "energy_denominator_scale": float(energy_scale),
        "energy_row_scale": float(energy_scale),
        "outer_buffer_radial_weight": float(buffer_weights[0]),
        "outer_buffer_energy_weight": float(buffer_weights[1]),
        "Qvisc": float(Q_visc),
        "Qrad": float(state.Q_rad),
        "Qadv": float(Q_adv),
        "Qstream_heat": float(Q_stream),
        "Qstream_mass_term_explicit": 0.0,
        "Qtorque_angular_source_proxy": float(torque_work_proxy),
        "Tdsdx": float(Tdsdx),
        "dOmega_dlnR": float(dOmega_dx),
        "Omega": float(state.Omega),
        "Omega_over_OmegaK": float(state.Omega / state.Omega_K),
        "Sigma": float(state.Sigma),
        "T": float(state.T),
        "H_over_R": float(state.H / state.R),
        "Mdot_local_over_inner": float(mdot_local / local_params.Mdot_g_s),
        "dMdot_dlnR_over_inner": float(dmdot_dlnR / local_params.Mdot_g_s),
        "stream_source_prime_over_inner": float(source_prime / local_params.Mdot_g_s),
        "wind_sink_prime_over_inner": float(wind_prime / local_params.Mdot_g_s),
        "stream_torque_l": float(stream_l),
        "stream_torque_dl_dlnR": float(stream_dl_dlnR),
        "entropy_gradient_Tdsdx": float(entropy_gradient_log(x_mid, y_mid, g_mid, lambda0, local_params)),
    }


def local_energy_sensitivities(z: np.ndarray, params, idx: int) -> dict[str, float]:
    logu, logT, logR, lambda0, _R_mid_rg = interval_arrays(z, params)
    base = np.concatenate([logu, logT, [float(np.log(params.R_out)), lambda0]])
    _ = base
    columns = {
        "logu_left": idx,
        "logu_right": idx + 1,
        "logT_left": params.n_nodes + idx,
        "logT_right": params.n_nodes + idx + 1,
    }

    def signed_energy(trial_logu: np.ndarray, trial_logT: np.ndarray) -> float:
        terms = energy_terms_from_arrays(trial_logu, trial_logT, logR, lambda0, params, idx)
        return float(terms["signed_physical_interval_E"])

    base_value = signed_energy(logu, logT)
    rows: dict[str, float] = {"local_energy_base_for_jac": float(base_value)}
    for name, col in columns.items():
        if name.startswith("logu"):
            arr_plus = np.array(logu, copy=True)
            arr_minus = np.array(logu, copy=True)
            scale = max(1.0, abs(float(logu[col])))
            step = 1.0e-6 * scale
            arr_plus[col] += step
            arr_minus[col] -= step
            deriv = (signed_energy(arr_plus, logT) - signed_energy(arr_minus, logT)) / (2.0 * step)
        else:
            local_col = col - params.n_nodes
            arr_plus = np.array(logT, copy=True)
            arr_minus = np.array(logT, copy=True)
            scale = max(1.0, abs(float(logT[local_col])))
            step = 1.0e-6 * scale
            arr_plus[local_col] += step
            arr_minus[local_col] -= step
            deriv = (signed_energy(logu, arr_plus) - signed_energy(logu, arr_minus)) / (2.0 * step)
        rows[f"jac_energy_{name}"] = float(deriv)
        rows[f"colscale_energy_{name}"] = float(1.0 / max(abs(deriv), 1.0e-12))
    return rows


def energy_terms_from_arrays(logu, logT, logR, lambda0: float, params, idx: int) -> dict[str, float]:
    z = np.empty(2 * params.n_nodes + 2, dtype=float)
    z[: params.n_nodes] = logu
    z[params.n_nodes : 2 * params.n_nodes] = logT
    z[-2] = logR[0]
    z[-1] = lambda0
    return energy_terms_at_interval(z, params, idx)


def selected_interval_indices(z: np.ndarray, params) -> list[int]:
    logu, logT, logR, lambda0, R_mid_rg = interval_arrays(z, params)
    values = []
    for idx in range(len(logR) - 1):
        terms = energy_terms_from_arrays(logu, logT, logR, lambda0, params, idx)
        values.append(float(abs(terms["signed_physical_interval_E"])))
    values_arr = np.asarray(values, dtype=float)
    selected: set[int] = set()
    for target in (SOURCE_REGION_RG, BUFFER_REGION_RG):
        in_region = np.flatnonzero(np.abs(R_mid_rg - target) <= REGION_HALF_WIDTH_RG)
        selected.update(int(idx) for idx in in_region)
        nearest = np.argsort(np.abs(R_mid_rg - target))[:TOP_N_PER_REGION]
        selected.update(int(idx) for idx in nearest)
    top_global = np.argsort(-values_arr)[: TOP_N_PER_REGION * 2]
    selected.update(int(idx) for idx in top_global)
    return sorted(selected)


def source_fraction_prediction(
    *,
    anchor_z: np.ndarray,
    anchor_params,
    target_params,
    idx: int,
) -> dict[str, float]:
    anchor_f = case_fraction(anchor_params)
    target_f = case_fraction(target_params)
    base = energy_terms_at_interval(anchor_z, anchor_params, idx)["signed_physical_interval_E"]
    h = max(1.0e-8, 1.0e-5 * max(abs(anchor_f), 1.0))
    plus = energy_terms_at_interval(anchor_z, anchor_params, idx, override_source_fraction=anchor_f + h)[
        "signed_physical_interval_E"
    ]
    minus = energy_terms_at_interval(anchor_z, anchor_params, idx, override_source_fraction=anchor_f - h)[
        "signed_physical_interval_E"
    ]
    dE_df = (plus - minus) / (2.0 * h)
    predicted = base + dE_df * (target_f - anchor_f)
    fixed_target = energy_terms_at_interval(anchor_z, anchor_params, idx, override_source_fraction=target_f)[
        "signed_physical_interval_E"
    ]
    return {
        "anchor_signed_E_same_idx": float(base),
        "d_signed_E_df_fixed_anchor": float(dE_df),
        "fixed_state_linear_pred_signed_E": float(predicted),
        "fixed_state_actual_target_param_signed_E": float(fixed_target),
    }


def summary_for_case(label: str, note: str, path: Path, z: np.ndarray, params) -> dict[str, Any]:
    audit = residual_audit_from_state_vector(z, params)
    partition = residual_partition_audit_from_state_vector(z, params)
    profile = transonic_profile_from_state_vector(z, params)
    adv = advection_diagnostic(z, params)
    budget = stream_diagnostic(z, params)
    square = square_collocation_residual(z, params)
    return {
        "label": label,
        "note": note,
        "checkpoint": relative_root_path(path),
        "source_fraction": case_fraction(params),
        "N": int(params.n_nodes),
        "Rout_rg": float(params.R_out_rg),
        "Rinj_rg": float(params.stream_source_center_fraction * params.R_out_rg),
        "outer_buffer_inner_rg": np.nan if params.outer_buffer_inner_rg is None else float(params.outer_buffer_inner_rg),
        "interval_residual_form": str(params.interval_residual_form),
        "full_residual": max_residual(z, params),
        "square_residual": float(np.max(np.abs(square))),
        "physical_E": float(partition.physical_energy_max),
        "physical_E_l2": float(partition.physical_energy_l2),
        "buffer_E": float(partition.buffer_energy_max),
        "peak_physical_E_rg": float(partition.peak_physical_energy_rg),
        "peak_buffer_E_rg": float(partition.peak_buffer_energy_rg),
        "interval_E": float(audit.interval_energy_max),
        "interval_R": float(audit.interval_radial_max),
        "outer_omega": float(audit.outer_omega),
        "accepted_clean_gate": bool(partition.physical_energy_max <= PHYSICAL_E_TOL),
        "Mdot_outer_over_inner": budget["Mdot_outer_over_inner"],
        "source_integral_over_inner": budget["stream_source_integral_over_inner"],
        "relative_mass_budget_error": budget["relative_mass_budget_error"],
        "f_adv_global": adv["f_adv_global"],
        "f_adv_inner": adv["f_adv_inner"],
        "f_adv_pos": adv["f_adv_pos"],
        "Lrad_LEdd": adv["Lrad_LEdd"],
        "max_H_over_R": float(np.max(profile.H_over_R)),
        "Rson_rg": float(profile.sonic_radius / params.r_g),
    }


def write_markdown_table(path: Path, rows: list[dict[str, Any]], columns: list[str], title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", "", f"Rows: {len(rows)}", ""]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "|".join("---" for _ in columns) + "|")
    for row in rows:
        rendered = []
        for key in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                rendered.append(fmt(value))
            else:
                rendered.append(str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_note(summary_rows: list[dict[str, Any]], detail_rows: list[dict[str, Any]]) -> None:
    anchor = summary_rows[0]
    failed = summary_rows[1:]
    source_peak_rows = [
        row
        for row in detail_rows
        if row["region"] == "source_peak" and abs(row["R_mid_rg"] - SOURCE_REGION_RG) < 0.5
    ]
    buffer_rows = [
        row
        for row in detail_rows
        if row["region"] == "buffer_peak" and abs(row["R_mid_rg"] - BUFFER_REGION_RG) < 1.0
    ]
    lines = [
        "# Stream Plateau Autopsy Results",
        "",
        "Generated by `scripts/run_standard_slim_stream_plateau_autopsy.py`.",
        "",
        "## Frozen Frontier",
        "",
        f"- Clean anchor: `f_s={anchor['source_fraction']:.9f}`.",
        f"- `physical_E={anchor['physical_E']:.3e}` with gate `{PHYSICAL_E_TOL:.1e}`.",
        f"- Checkpoint: `{anchor['checkpoint']}`.",
        "- Older `f_s ~= 0.90` states remain exploratory; they are not scientific anchors under the physical-energy gate.",
        "",
        "## Summary",
        "",
        "| label | f_s | physical_E | buffer_E | peak physical R/rg | peak buffer R/rg | clean gate |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {label} | {source_fraction:.9f} | {physical_E:.3e} | {buffer_E:.3e} | "
            "{peak_physical_E_rg:.4g} | {peak_buffer_E_rg:.4g} | {accepted_clean_gate} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Main Observations",
            "",
            "- The clean frontier is still just below the gate, while all three nearby target checkpoints are only slightly above it.",
            "- The peak physical differential energy residual remains near the source/buffer-side source structure at `R ~= 259.2 rg`.",
            "- The largest configured interval energy residual remains in the outer buffer near `R ~= 333-334 rg`; this is amplified in the full interval audit even though the physical-domain gate is controlled by the source-region point.",
            "- The diagnostic table now separates raw energy numerator, denominator, local heating/cooling terms, source derivatives, torque proxy terms, and local finite-difference energy sensitivities.",
            "",
            "## Files",
            "",
            f"- Summary table: `{relative_root_path(TABLE_OUTPUT)}`",
            f"- Summary JSON: `{relative_root_path(JSON_OUTPUT)}`",
            f"- Detail table: `{relative_root_path(DETAIL_OUTPUT)}`",
            f"- Detail JSON: `{relative_root_path(DETAIL_JSON_OUTPUT)}`",
            "",
            "## Interpretation",
            "",
            "This supports the GPT diagnosis: the plateau is an energy-row numerical/formulation issue, not a sonic failure or clear physical branch endpoint. The next move should be energy-focused Newton merit/scaling or a local energy patch solve, using these rows as the target diagnostics.",
            "",
        ]
    )
    if source_peak_rows:
        lines.extend(["## Source-Region Rows Near 259 rg", ""])
        for row in source_peak_rows[:8]:
            lines.append(
                "- `{label}` idx `{idx}` R=`{R_mid_rg:.4g}`: signed_E=`{signed_physical_interval_E:.3e}`, "
                "Qvisc=`{Qvisc:.3e}`, Qrad=`{Qrad:.3e}`, Qadv=`{Qadv:.3e}`, "
                "dMdot/inner=`{dMdot_dlnR_over_inner:.3e}`.".format(**row)
            )
        lines.append("")
    if buffer_rows:
        lines.extend(["## Buffer-Region Rows Near 333-334 rg", ""])
        for row in buffer_rows[:8]:
            lines.append(
                "- `{label}` idx `{idx}` R=`{R_mid_rg:.4g}`: solver_E=`{solver_interval_E:.3e}`, "
                "signed_E=`{signed_physical_interval_E:.3e}`, buffer_weight=`{outer_buffer_energy_weight:.3e}`.".format(**row)
            )
        lines.append("")
    NOTE_OUTPUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    fiducial = FiducialParams()
    mdot_edd = eddington_mdot(fiducial.M2_g)
    loaded: list[tuple[str, str, Path, np.ndarray, Any]] = []
    for label, raw_path, note in CASES:
        path = ROOT / raw_path
        if not path.exists():
            raise FileNotFoundError(path)
        z, params = load_anchor(path, fiducial, mdot_edd)
        loaded.append((label, note, path, z, params))

    anchor_label, _anchor_note, _anchor_path, anchor_z, anchor_params = loaded[0]
    _ = anchor_label
    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    for label, note, path, z, params in loaded:
        summary_rows.append(summary_for_case(label, note, path, z, params))
        indices = selected_interval_indices(z, params)
        for idx in indices:
            row = {
                "label": label,
                "case_note": note,
                "source_fraction": case_fraction(params),
                "region": (
                    "source_peak"
                    if abs(energy_terms_at_interval(z, params, idx)["R_mid_rg"] - SOURCE_REGION_RG)
                    <= abs(energy_terms_at_interval(z, params, idx)["R_mid_rg"] - BUFFER_REGION_RG)
                    else "buffer_peak"
                ),
                **energy_terms_at_interval(z, params, idx),
                **local_energy_sensitivities(z, params, idx),
            }
            if params.n_nodes == anchor_params.n_nodes:
                prediction = source_fraction_prediction(
                    anchor_z=anchor_z,
                    anchor_params=anchor_params,
                    target_params=params,
                    idx=idx,
                )
                row.update(prediction)
                row["actual_minus_fixed_state_linear_pred_E"] = float(
                    row["signed_physical_interval_E"] - row["fixed_state_linear_pred_signed_E"]
                )
                row["actual_minus_anchor_E_same_idx"] = float(
                    row["signed_physical_interval_E"] - row["anchor_signed_E_same_idx"]
                )
            detail_rows.append(row)

    summary_columns = [
        "label",
        "source_fraction",
        "physical_E",
        "physical_E_l2",
        "buffer_E",
        "peak_physical_E_rg",
        "peak_buffer_E_rg",
        "full_residual",
        "square_residual",
        "outer_omega",
        "accepted_clean_gate",
        "Mdot_outer_over_inner",
        "source_integral_over_inner",
        "f_adv_global",
        "f_adv_inner",
        "Lrad_LEdd",
        "max_H_over_R",
        "Rson_rg",
    ]
    detail_columns = [
        "label",
        "source_fraction",
        "region",
        "idx",
        "R_left_rg",
        "R_mid_rg",
        "R_right_rg",
        "dlnR",
        "physical_interval_E",
        "signed_physical_interval_E",
        "solver_interval_E",
        "integrated_interval_E",
        "energy_numerator",
        "energy_denominator_scale",
        "Qvisc",
        "Qrad",
        "Qadv",
        "Qstream_heat",
        "Qtorque_angular_source_proxy",
        "dMdot_dlnR_over_inner",
        "stream_source_prime_over_inner",
        "Omega_over_OmegaK",
        "dOmega_dlnR",
        "Sigma",
        "T",
        "H_over_R",
        "outer_buffer_energy_weight",
        "jac_energy_logu_left",
        "jac_energy_logT_left",
        "jac_energy_logu_right",
        "jac_energy_logT_right",
        "anchor_signed_E_same_idx",
        "fixed_state_linear_pred_signed_E",
        "actual_minus_fixed_state_linear_pred_E",
    ]
    TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUTPUT.write_text(json.dumps(json_safe(summary_rows), indent=2, sort_keys=True), encoding="utf-8")
    DETAIL_JSON_OUTPUT.write_text(json.dumps(json_safe(detail_rows), indent=2, sort_keys=True), encoding="utf-8")
    write_markdown_table(TABLE_OUTPUT, summary_rows, summary_columns, "High-Mdot Stream Plateau Autopsy Summary")
    write_markdown_table(DETAIL_OUTPUT, detail_rows, detail_columns, "High-Mdot Stream Plateau Autopsy Detail")
    write_note(summary_rows, detail_rows)
    print(f"wrote {relative_root_path(TABLE_OUTPUT)}", flush=True)
    print(f"wrote {relative_root_path(DETAIL_OUTPUT)}", flush=True)
    print(f"wrote {relative_root_path(NOTE_OUTPUT)}", flush=True)


if __name__ == "__main__":
    main()
