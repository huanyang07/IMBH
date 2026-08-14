# Face-36 leading-two plus HMM fixed-Q preflight

Classification: `leading_two_plus_HMM_fixed_Q_constraint_preflight_passed_one_Q_nonlinear_manifest_authorized`.

No trajectory was advanced. At the committed middle and fine 20 ms states, the macro constraint is imposed with the ledger-derived reaction projection `P = I - L (DQ L)^-1 DQ`; no Euclidean primitive projection is used.

## Binding results

| layout | DQ M^-1 BQ | KKT | a2 biorth | a2 reaction | block | face-36 five-point | incoming |
|---|---:|---:|---:|---:|---:|---:|---:|
| middle | 1.307e-12 | 7.204e-16 | 5.040e-14 | 4.180e-13 | 0.000e+00 | 1.209e-09 | 0 |
| fine | 2.447e-12 | 1.201e-15 | 1.654e-14 | 1.757e-13 | 0.000e+00 | 1.345e-09 | 0 |

The 24-direction finite-time screens use the frozen Galerkin generator on the screened lift span. They are local diagnostics. They do not prove guard mixing, attraction, or decay. In particular, `P G` omits the state derivative of the reaction projection; the next definitions-only nonlinear-pilot manifest must require the complete state-dependent constrained residual and its JVP before any microburst.

A pass authorizes only that definitions-only one-Q pilot manifest. The fixed-Q micro-solver, nonlinear pilot propagation, 50 ms run, and reduced slow evolution remain blocked.
