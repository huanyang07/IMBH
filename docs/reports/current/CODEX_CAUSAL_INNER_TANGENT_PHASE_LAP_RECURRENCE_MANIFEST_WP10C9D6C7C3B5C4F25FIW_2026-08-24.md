# Tangent-phase-lap and state-recurrence acquisition manifest

Classification: `bounded_tangent_phase_lap_recurrence_acquisition_selected_stage1_required`.

The 16-point prospective phase holdout passed. Its exact local phase increments span `4.471378e-02` to `4.947195e-02` rad, implying `141` accepted 0.25 ms segments for a conservative 2*pi estimate.

The acquisition is split into three 48-endpoint tranches. Only stage 1 is authorized now. Measured-cost projections are `5.47` hours for stage 1 and `16.40` hours for the full 144-endpoint acquisition.

A phase lap is only a candidate. A cycle additionally requires metric state return/path length <= 0.10, tangent cosine >= 0.99, a positive transverse registered-section crossing near 2*pi, all physical gates, and exact restart/replay. Any candidate must then undergo exact registered-section refinement and periodic multiple shooting/collocation.

The expensive tangent acquisition is an offline one-time task. A future online reduced solver may use only precomputed averaged slow drift over certified periodic Q anchors; it may not call the truth integrator, fixed-Q reactions, nonlinear roots, or micro-BDF steps.

Authorized next artifact: `WP10c9d6c7c3b5c4f25fix_tangent_phase_lap_recurrence_stage1_execution`. Complete-cycle execution and reduced slow evolution remain unauthorized.
