# Face-36 one-Q nonlinear-pilot manifest

Classification: `one_Q_leading_two_plus_HMM_nonlinear_pilot_manifest_frozen_state_dependent_constrained_step_preflight_authorized`.

This package is definitions-only. It freezes the exact state-dependent constraint, augmented BDF residual/JVP, finite equal-Q lift, cost, and fail-fast contracts for the first one-Q pilot.

The next package may implement and audit the augmented constrained backward-Euler/BDF2 step at the committed middle 20 ms endpoint. It may not advance a constrained trajectory. The JVP must include derivatives of `M`, `R`, `DQ3`, `B_Q`, multiplier coupling, and both storage-history channels; the frozen `P G` operator from c4f22 is reference-only.

Strong frozen transient amplification prevents any assumption that the guard rapidly decays. If the step/JVP preflight later passes, a separate execution manifest may authorize one middle constrained base, a 24-RHS block tangent, and at most two prospectively selected nonlinear anchors. Fine and 50 ms runs remain conditional.
