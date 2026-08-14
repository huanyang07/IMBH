# Face-36 Q+a reaction and coordinate preflight

Classification: `face36_two_mode_coordinate_preflight_rejected_six_mode_manifest_authorized`.

The physical macro-only reaction map passes on every committed middle/fine state at 5, 10, 16, and 20 ms. The maximum normalized reaction identity defect is `2.447e-12`, the KKT solve defect is `1.334e-15`, the reaction-ledger defect is `3.819e-16`, and the maximum raw Schur condition number is `36616.2`.

The 5 ms state lifts and descriptor-weighted Petrov duals also pass their endpoint algebraic gates. Their maximum Q3 defect is `7.624e-16` and maximum biorthogonality defect is `4.435e-13`.

The two-mode coordinate is nevertheless rejected. Its worst aggregate output-weighted RMS error is `0.070173`, but its worst significant-direction error is `1.043772`, above the frozen `0.25` gate. A prospective consensus dimension of `6` is the first to pass both output gates; its worst significant-direction error is `0.033078`.

The committed memory result contains output histories but not intermediate state-direction histories. Modal output kernels are therefore reported only as diagnostics and are not relabelled as descriptor-dual state-amplitude histories. No fixed-Q or nonlinear pilot is authorized. The next package may only freeze a six-mode coordinate manifest and a cost-bounded dynamic-coordinate propagation contract.
