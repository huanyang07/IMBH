# Face-36 fixed-Q second-state and BDF2 preflight

## Classification

`fixed_Q_second_state_Jacobian_and_exact_BDF2_roots_passed_but_synthetic_history_limit_orders_failed`

This analysis-only package changes no physical operator and advances no
trajectory.  It repeats the complete state-dependent Jacobian audit at the
committed middle-layout `16 ms` state and extends the exact constrained solver
to BDF2.  The second-state derivative and every exact constrained BDF2 root
pass.  The prospectively retained rate and multiplier order gates do not pass,
so a one-Q execution manifest, fixed-Q microsolver, and reduced evolution remain
blocked.

## Positive results

The complete residual was independently differentiated at the existing
variable-step BDF2 endpoint.  The measured relative defects are:

| audit | defect | gate |
|---|---:|---:|
| monolithic BDF2 JVP | `6.17225485e-9` | `1e-8` |
| full augmented nonzero-multiplier JVP | `7.66602374e-11` | `1e-8` |
| raw reaction action JVP | `1.36710570e-9` | `1e-8` |
| augmented block closure | `0` | diagnostic |
| reaction ledger | `1.30987011e-16` | `1e-12` |

The BDF implementation now accepts a complete previous primitive, mapped
storage, and responsive-height history.  Its direct current-interval storage
rate is combined with the stored previous increment using the exact BDF2
coefficients, avoiding subtraction of endpoint storage values divided by a
tiny timestep.

Five independent exact constrained BDF2 roots were solved.  All pass the
unchanged `1e-10` residual and `1e-12` Q3 gates:

| `dt` (s) | residual | current Q3 defect | history Q3 defect | rate defect | multiplier defect |
|---:|---:|---:|---:|---:|---:|
| `8e-9` | `8.7793e-11` | `2.8727e-16` | `1.4363e-16` | `1.23212e-3` | `1.91607e-5` |
| `4e-9` | `2.2789e-11` | `1.4363e-16` | `1.9935e-14` | `5.83417e-4` | `4.68856e-5` |
| `2e-9` | `6.1572e-13` | `2.3522e-16` | `1.7548e-14` | `4.56042e-4` | `2.78598e-5` |
| `1e-9` | `5.4988e-13` | `1.4039e-16` | `1.4039e-16` | `2.56100e-4` | `1.36837e-5` |
| `5e-10` | `3.8644e-13` | `1.4363e-16` | `1.6285e-14` | `2.23473e-4` | `1.42797e-5` |

These roots prove that the repaired state-dependent augmented operator can
solve exact equal-Q BDF2 equations at a second committed state.  They also
exclude the earlier missing reaction derivative as the cause of the remaining
limit failure.

## Binding failure

The binding consistency ladder is `8e-9`, `4e-9`, and `2e-9 s`.  Its rate
orders are

`1.07854, 0.35536`,

and its multiplier orders are

`-1.29100, 0.75096`.

The unchanged minimum order is `0.9`.  The smaller `2e-9`, `1e-9`, and
`5e-10 s` diagnostic sequence also loses rate order and reaches a multiplier
floor.  Moving the gate or choosing only a favorable pair would hide this
behavior and is not permitted.

The failure is localized to the synthetic previous-state construction.  The
history is formed by taking one backward continuous-KKT tangent increment and
then projecting it onto exact Q3 with the state-local reaction coordinates.
The Q3 manifold is strongly curved in those coordinates: the `8e-9 s` history
requires seven nonlinear corrections, whereas the smallest histories reach a
numerical floor.  Each individual history and BDF2 root is exact, but this
family is not a smooth enough discrete history ladder for a binding
discrete-to-continuous order measurement.

This is not a physical fixed-Q rejection.  It is not a Jacobian rejection and
does not invalidate the already certified Q3 map, reaction ledger, equal-Q
lifts, continuous KKT algebra, or exact backward-Euler limit.

## Next authorized package

Only a definitions-only constrained startup/history preflight is authorized.
It must freeze an execution-shaped sequence:

1. solve one exact constrained BDF1 startup from the committed state;
2. construct the complete primitive, mapped-storage, and responsive-height
   history from that accepted root;
3. solve the following exact constrained BDF2 step without projecting a
   synthetic previous state;
4. require the existing residual, Q3, reaction-ledger, admissibility,
   causality, and restart gates;
5. repeat at the `16 ms` state and one held-out committed state;
6. use the bordered sparse block as a preconditioner and the certified
   reaction JVP matrix-free; do not require repeated dense exact refreshes in
   execution;
7. authorize a one-Q execution manifest only if the BDF1-to-BDF2 chain and a
   serialized replay pass.

No 50 ms trajectory, fixed-Q microburst, reduced closure fit, or slow
evolution is authorized by this result.
