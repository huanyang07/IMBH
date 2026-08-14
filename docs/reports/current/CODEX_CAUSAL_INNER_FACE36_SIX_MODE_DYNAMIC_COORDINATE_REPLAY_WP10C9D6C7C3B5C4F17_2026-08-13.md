# Face-36 six-mode middle dynamic-coordinate replay

Classification: `face36_six_mode_middle_dynamic_coordinate_preflight_rejected_fine_blocked_numerical_audit_recovery_manifest_authorized`.

The analysis-only middle replay completed all 39 committed 5--20 ms steps in 1.865 hours. Fine was not started because the frozen middle fail-fast gate did not pass.

The complete BDF step JVP (`5.85e-11`), block solve (`3.31e-16`), Q3 leakage (`0.004143` versus `0.10`), component closure (`6.56e-17`), initial Q3 lift (`3.41e-15`), dual biorthogonality (`1.35e-11`), and outgoing excision all pass.

Two numerical-audit gates fail: the normal-equation Petrov dual has normalized slow-lift annihilation `3.84e-10 > 1e-10`, and the selected face-36 directional finite-difference check gives `1.37e-7 > 1e-8`. This is not a physical truth-model failure and the saved dynamic histories are not interpreted before those audits are recovered.

The next authorized work is analysis-only: reconstruct the dual with stable reduced-QR/thin-SVD algebra and establish a predeclared central/five-point face-36 JVP step plateau on all six saved directions. No tolerance may be relaxed, middle propagation must not be repeated, and fine remains blocked.
