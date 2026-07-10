# IMBH project-review P0 results

Date: 2026-07-10

Frozen target:

- `Mdot_inner/Mdot_Edd = 5`
- `Rout = 335 rg`
- `Rinj = 240 rg`
- `f_s = 0.80`
- compact C2 source
- local-Mdot wind formulation
- `eta_E = 98.125`
- `N = 164`

## Work completed

1. Endpoint model-validity, integrability, transport, self-gravity, timescale, and common-window exponent audit.
2. Explicit angular-momentum ledger with representation, local-disk, and Keplerian source/wind closures.
3. Independent outer-to-inner phase-manifold atlas with six nominal outer starts, source-shape tangent scans, signed arclength, gauge switching, and local shooting scouts.
4. State and conserved-flux matching at the first physical model-validity boundary.
5. Regression coverage and full-suite verification.

## Main numerical findings

### Endpoint and validity

- The accepted positive phase branch approaches a formal finite-radius low-velocity limit at `R*=225.52125 rg`.
- The first 1D-model validity failure occurs earlier, at `R=223.236427 rg`, where `L_u/H=0.6253`.
- The phase homogeneous residual remains below `3.079e-5`; the homogeneous mass residual remains below `1.069e-6`.
- The annulus-mass exponent is positive in every common-window fit: `0.4354` to `0.5590`. The formal surface-density divergence is locally integrable.
- The formal endpoint is therefore a mathematical continuation outside the resolved radial-vertical scale-separation regime, not a certified physical steady reservoir.

### Angular momentum

- The algebraic representation ledger closes pointwise to machine precision and in finite volume to `9.754e-6` over the phase segment.
- Assigning source and wind material the full local disk angular momentum while retaining the existing explicit torque leaves a phase point defect of `0.4366` and a phase FV defect of `5.383e-4`.
- The representation closure is a correct identity for the present equations, but it is not an independently specified physical stream/wind closure.

### Independent outer manifold

- The atlas contains 150 independently constructed local states and 146 accepted tangent roots; no multiple local tangent roots were found.
- Four nominal outer starts terminate before the validity surface by step exhaustion, solver failure, stagnation, or a radial turn.
- Nominal starts near `235` and `230 rg` reach the validity surface on distinct sheets.
- A focused `R=230 rg` shooting scan gives the best conservative near-match:

```text
delta(logu, logT, F) = (1.77058e-3, 1.77045e-3, 1.04194e-5)
maximum state mismatch = 1.77058e-3
maximum flux mismatch  = 1.04194e-5
```

- This passes the conservative-flux gate (`1e-4`) but misses the strict state gate (`1e-3`). It is not promoted to a connected branch.
- The local shooting-map singular values are `(28.5641, 0.99645, 3.26797e-5)`, giving condition number `8.74e5`. Velocity and temperature shooting directions have cosine `0.999999993`.
- The residual mismatch is consequently a nearly tangent fixed-`lambda0` sheet separation, not evidence that a simple unconverged seed correction remains.

## Scientific classification

Directly supported:

- the current positive phase branch reaches a closure-dependent singular/stagnating limit;
- the current 1D assumptions fail before that formal limit;
- an independent outer sheet approaches, but does not strictly connect at the first validity boundary under the frozen `lambda0` and representation closure;
- the present algebraic angular ledger is conservative as a representation identity.

Not established:

- global nonexistence of every far-side steady branch;
- a physical steady stagnation reservoir;
- a physical stream/wind angular-momentum closure;
- a globally connected physical Mdot=5 stream-fed wind branch.

## Recommended next move

Do not lower `eta_E` or continue tuning the Lobatto/outer shooting weights.

1. Specify physical `l_s(R)`, `l_w(R)`, and `tau_ext(R)` without double counting the stream torque.
2. Promote mass, angular momentum, and energy fluxes to a unified conservative production formulation.
3. Repeat the validity-surface manifold match under that physical closure, allowing the global eigenvalue to participate only through a proper bordered continuation problem.
4. If no steady connection survives those gates, move to the signed-flux/time-dependent conservative disk formulation; the current endpoint then becomes an initial-condition/benchmark diagnostic rather than a steady solution claim.
5. Perform the separately documented non-destructive repository cleanup before the major formulation redesign.

## Verification

```text
182 passed, 4 subtests passed
```

## Detailed reports

- `Note/CODEX_MDOT5_ENDPOINT_VALIDITY_AND_EXPONENT_AUDIT_RESULTS.md`
- `Note/CODEX_MDOT5_ANGULAR_MOMENTUM_LEDGER_RESULTS.md`
- `Note/CODEX_MDOT5_INDEPENDENT_OUTER_MANIFOLD_RESULTS.md`
- `Note/CODEX_IMBH_REPOSITORY_CLEANUP_SPEC_2026-07-10.md`
