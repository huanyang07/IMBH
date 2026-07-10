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
| Unified conservative mass/angular/energy formulation | **SUPPORTED BUT NOT FULLY CERTIFIED** | No-wind `Mdot/Edd=5` regression and physical compact-stream roots pass raw ledgers | Stream-fed wind tests currently start from `Mdot_inner/Edd=2` |
| Physical mass-loaded-wind steady branch at `Mdot_inner/Edd=2` | **SUPPORTED BUT NOT FULLY CERTIFIED** | Exploratory roots through `epsilon_w=0.54`; scouts through `0.90` | Wind loss is `<0.5%`; this is not a strong hot branch |
| Unified compact-stream branch at `Mdot_inner/Edd=5`, `Rout=335 rg` | **SUPPORTED BUT NOT FULLY CERTIFIED** | `f_s=0.05,0.10,0.30` pass at `N=192,256,384`; conservative mass budget closes | Outer compatibility convergence is only exploratory |
| Unified Mdot=5 energy-limited wind | **DIAGNOSTIC ONLY** | `epsilon_w=0.20` passes at `eta_E=98.125`; eta scouts reach `8` | Wind loss reaches only `1.66%`; no hot transition; low-eta source-band defects remain |
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
5. A new unified conservative solver explicitly specifies `l_s`, `B_s`,
   `l_w`, and `B_w`, and separates external torque from external power. Its
   first stream-fed wind branch is physical but only weakly mass loaded.

## Claims That Are Not Allowed Yet

- “A strong advective/hot mass-loaded-wind branch has been recovered.”
- “The branch ends physically at `225.52125 rg`.”
- “No global far-side steady solution exists.”
- “The present stream torque and local angular momentum prescription is a
  conservative physical closure.”

## Next Scientific Work

1. Build a conservative source-band refinement for the Mdot=5 low-launch-energy
   wind branch, preserving exact source support and sonic nodes.
2. Certify selected `eta_E=20,10,8` checkpoints before continuing toward
   order-unity launch energy.
3. Test wind lever arm and stream heating separately after the low-eta branch
   is stable.
4. Repeat validity-surface matching with a bordered global eigenvalue solve if
   a mesh-stable critical layer appears.
5. If the physically closed steady problem still has no connection, implement
   the signed-flux/time-dependent conservative model.

## Review Entry Points

- Equations: [`MODEL_EQUATIONS.md`](MODEL_EQUATIONS.md)
- Reproduction and archive recovery: [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)
- Compact evidence: [`../results/README.md`](../results/README.md)
- P0 synthesis: `reports/current/CODEX_IMBH_PROJECT_REVIEW_P0_RESULTS_2026-07-10.md`
- Detailed current reports: `reports/current/`
- Historical development sequence: [`history/MILESTONES.md`](history/MILESTONES.md)
- Cleanup verification: `reports/current/CODEX_REPOSITORY_CLEANUP_RESULTS_2026-07-11.md`
- Unified conservative transport: `reports/current/CODEX_UNIFIED_CONSERVATIVE_TRANSPORT_RESULTS_2026-07-11.md`
