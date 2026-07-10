# Project Status

- Updated: 2026-07-11
- Pre-cleanup scientific tag: `pre-cleanup-p0-2026-07-11`
- Legacy phase classification tag: `legacy-steady-positive-flux-dae-2026-07-10`

This is the canonical project handoff. Status labels mean:

- **CERTIFIED:** passes the stated numerical and physical gates for its scope.
- **SUPPORTED BUT NOT FULLY CERTIFIED:** strong numerical evidence with an
  identified unresolved robustness or closure condition.
- **DIAGNOSTIC ONLY:** useful mathematical or numerical evidence that must not
  be promoted to a physical branch claim.
- **REJECTED:** tested formulation or composite fails its acceptance gates.
- **PLANNED:** not implemented or not yet evaluated.

## Result Matrix

| Result | Status | Decisive evidence | Limitation |
|---|---|---|---|
| Standard no-wind slim disk through `Mdot/Edd=5` | **CERTIFIED** | N768 accepted canonical state; high-rate ladder and mesh checks | Does not include stream, heating, or wind |
| Compact stream-fed no-wind `Mdot_inner/Edd=2`, `f_s=0.80` | **SUPPORTED BUT NOT FULLY CERTIFIED** | N896 residual-remeshed canonical state and N640/768/896 diagnostics | Relies on residual-aware remeshing; naive remaps fail |
| N164 global phase-DAE entry at `Mdot_inner/Edd=5`, `eta_E=98.125` | **SUPPORTED BUT NOT FULLY CERTIFIED** | Local phase radial/energy/FV equations can be solved accurately | Global far-side attachment is unresolved |
| Formal low-velocity endpoint near `225.52125 rg` | **DIAGNOSTIC ONLY** | Two step sizes, bordered continuation, source-shape scans, homogeneous residual audit | `L_u/H<1` first at `223.23643 rg`; endpoint is outside 1D validity |
| Local annulus-mass integrability | **SUPPORTED BUT NOT FULLY CERTIFIED** | Common-window exponent gives positive mass power `0.435-0.559` | Applies to the mathematical asymptote under current equations |
| Existing global phase-plus-ordinary-tail composite | **REJECTED** | Phase rows remain small while outside radial/energy defects become large | Rejection is not global nonexistence |
| Independent outer-manifold connection | **DIAGNOSTIC ONLY** | Best flux mismatch `1.04e-5`; best state mismatch `1.77e-3` | Misses strict `1e-3` state gate; shooting map condition `8.74e5` |
| Algebraic angular representation ledger | **SUPPORTED BUT NOT FULLY CERTIFIED** | Point closure at machine precision; phase FV floor `9.75e-6` | Representation identity, not physical `l_s`, `l_w`, `tau_ext` closure |
| Physical mass-loaded-wind steady branch | **PLANNED** | Requires conservative production equations and physical angular closure | Current result must not be labeled a recovered hot/wind branch |
| Signed-flux/time-dependent conservative disk | **PLANNED** | Next formulation if a physically closed steady connection fails | Not implemented |

## Frozen Target Under Review

```text
Mdot_inner/Mdot_Edd = 5
Rout                 = 335 rg
Rinj                 = 240 rg
stream fraction      = 0.80
source shape         = compact C2
wind formulation     = local Mdot
eta_E                = 98.125
N                    = 164
```

## Most Important Findings

1. The standard no-wind high-rate benchmark is solid; the present obstruction
   is not failure of the underlying slim-disk solver.
2. A phase-space DAE representation is required in the stiff source/transition
   layer; ordinary `ln R` polynomial derivatives are incompatible there.
3. The accepted positive phase branch approaches a closure-dependent formal
   low-velocity singular limit, but the 1D radial/vertical separation fails
   before the limit.
4. Independent outer branches reach the physical validity boundary, including
   a conservative near-match, but no strict state-and-flux connection has been
   certified at fixed `lambda0`.
5. The existing angular ledger closes only as an algebraic representation. A
   physical stream/wind model must specify `l_s(R)`, `l_w(R)`, and
   `tau_ext(R)` without double counting.

## Claims That Are Not Allowed Yet

- “A physical advective/hot mass-loaded-wind branch has been recovered.”
- “The branch ends physically at `225.52125 rg`.”
- “No global far-side steady solution exists.”
- “The present stream torque and local angular momentum prescription is a
  conservative physical closure.”

## Next Scientific Work

1. Define physical stream and wind angular-momentum transport.
2. Promote mass, angular momentum, and energy fluxes to one conservative
   production formulation.
3. Repeat validity-surface matching with a bordered global eigenvalue solve.
4. If the physically closed steady problem still has no connection, implement
   the signed-flux/time-dependent conservative model.
5. Keep `eta_E` frozen until the closure and connection gates pass.

## Review Entry Points

- Equations: [`MODEL_EQUATIONS.md`](MODEL_EQUATIONS.md)
- Reproduction and archive recovery: [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)
- Compact evidence: [`../results/README.md`](../results/README.md)
- P0 synthesis: `reports/current/CODEX_IMBH_PROJECT_REVIEW_P0_RESULTS_2026-07-10.md`
- Detailed current reports: `reports/current/`
- Historical development sequence: [`history/MILESTONES.md`](history/MILESTONES.md)
