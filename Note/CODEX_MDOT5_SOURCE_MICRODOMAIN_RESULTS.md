# Mdot=5 eta_E=90 Source Micro-Domain Results

Date: 2026-07-06

## Context

Commit `ba243eb` left the Mdot_inner/Edd=5, Rout=335 rg, Rinj=240 rg,
f_s=0.80, eta_E=90 local-Mdot branch midpoint-strict but not
source-band-collocation strict:

- midpoint/base residual: `~6.53e-6`
- split/source-band audit: `O(1e-1)`
- rectangular source-band rows could reduce the hidden defect to `~1.9e-2`
  but did not remove it.

GPT suggested replacing rectangular rows as production equations with a true
source-annulus micro-domain.

## Implementation Added

In `scripts/run_mdot5_local_mdot_eta_continuation.py`:

- Added opt-in source micro-domain grid construction:
  - `IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_MICRO_DOMAIN=1`
  - preserves all old nodes and adds source-support subnodes;
  - source support edges and Rinj are real nodes at the seed state;
  - `IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_MICRO_NODES=N`.
- Added ODE-slope Hermite remapping with bounded overshoot fallback.
- Added finite-volume source-band mass rows:
  - `IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_FINITE_VOLUME_MASS=1`
  - analytic integral for the compact/tanh stream source;
  - Simpson integral for the wind mass sink.
- Added a local source-band Hermite-Simpson corrector, but the first tests show
  endpoint ODE slopes are too stiff for this seed.
- Added audit-only mode for rectangular quarter-point rows:
  - `IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_EXTRA_AUDIT_ONLY=1`.

## Key Results

All runs started from:

`outputs/checkpoints/m5_local_mdot_eta90_N168_localjac_innerweight20/stage_00_etaE_90_N168.npz`

### Source Micro-Domain With Hermite-Simpson Local Corrector

Initial replacement-grid attempt with N_source=16 failed because ODE-slope
Hermite midpoints overshot into nonphysical thermodynamic states near
`R~248 rg`.

After bounded Hermite fallback, the run was finite but still bad:

- `m5_local_mdot_eta90_micro_Nsrc16_seed_v2`
- N = 176
- final production residual: `1.406e2`
- source-band audit: `2.247e2`

Conclusion: endpoint ODE-slope Hermite-Simpson is not yet a usable first
corrector for this source band.

### Node-Preserving Micro-Grid + FV Mass + Band Pre-Corrector

The grid was changed to preserve all old nodes and add micro-domain nodes.

| run | N | source-band nodes | final | source audit | interval mass | interval R | interval E |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `m5_local_mdot_eta90_micro_Nsrc16_bandfv_seed` | 185 | 25 | `1.368e0` | `1.138e2` | `1.368e0` | `5.91e-1` | `8.15e-2` |
| `m5_local_mdot_eta90_micro_Nsrc32_bandfv_seed` | 201 | 41 | `3.803e-1` | `3.941e-1` | `3.803e-1` | `3.08e-2` | `2.17e-2` |
| `m5_local_mdot_eta90_micro_Nsrc64_bandfv_seed` | 233 | 73 | `9.058e-1` | `5.445e0` | `1.811e-1` | `2.31e-1` | `9.058e-1` |

N_source=32 is the best of these probes. N_source=64 does not monotonically
improve, so this is not just a resolution problem.

### Global Release From N_source=32

Run:

`m5_local_mdot_eta90_micro_Nsrc32_bandfv_global`

- initial/final production residual: `3.803e-1`
- source-band audit: `3.941e-1`
- no improvement; optimizer stopped by `xtol` after 10 evaluations.

Conclusion: the N_source=32 band-corrected point is locally stuck under the
current parameterization/Jacobian.

### Rectangular Rows As Diagnostic On Micro-Grid

Run:

`m5_local_mdot_eta90_micro_Nsrc16_rectrows_probe`

- start from the N_source=16 bandFV global checkpoint;
- turn rectangular source-band rows back on as production rows;
- final augmented residual: `6.447e-2`
- source-band audit: `4.253e-2`
- interval mass residual: `6.447e-2`
- midpoint interval R/E: `3.26e-2`, `3.21e-2`
- max function evaluations reached.

Conclusion: overdetermined rows can still push the hidden defect downward, but
not to certification. They remain a diagnostic/constraint aid rather than a
production formulation.

## Current Interpretation

The new implementation confirms GPT's main diagnosis: the eta_E=90 checkpoint
is not representation-robust in the source annulus.

The source micro-domain helps, especially with N_source=32 and finite-volume
mass rows, but it does not yet remove the residual floor. The remaining problem
appears to be a combination of:

- endpoint/source-edge mass residual stiffness;
- intra-cell energy oscillation still visible in quarter-point audits;
- poor conditioning of endpoint ODE-slope Hermite-Simpson rows;
- insufficient source-band variable parameterization, not physical branch death.

## Follow-Up: Multipoint Source-Domain Corrector

Implemented after the first micro-domain tests:

- `IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_DOMAIN_CORRECT=1`
- multipoint differential rows inside the source domain, default fractions
  `0.25,0.5,0.75`;
- finite-volume mass row per interval;
- optional halo intervals outside the formal source support;
- no endpoint `-A^{-1}c` Hermite-Simpson rows.

This is still a local source-domain corrector, but it is closer to the intended
formulation: the hidden quarter-point residuals become part of the local
source-domain solve without making the whole global problem rectangular.

All follow-up runs start from the best previous N_source=32 finite-volume
band-corrected seed:

`outputs/checkpoints/m5_local_mdot_eta90_micro_Nsrc32_bandfv_seed/stage_00_etaE_90_N201.npz`

| run | halo | final | source audit | interval mass | peak mass R | interval R | interval E |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `m5_local_mdot_eta90_Nsrc32_source_domain_qm_seed` | 0 | `3.803e-1` | `2.625e-2` | `3.803e-1` | `219.903 rg` | `1.764e-2` | `2.560e-2` |
| `m5_local_mdot_eta90_Nsrc32_source_domain_qm_halo2_seed` | 2 | `1.033e-1` | `2.826e-2` | `1.033e-1` | `261.781 rg` | `1.799e-2` | `2.591e-2` |
| `m5_local_mdot_eta90_Nsrc32_source_domain_qm_halo4_seed` | 4 | `3.929e-2` | `3.088e-2` | `3.929e-2` | `271.746 rg` | `1.934e-2` | `2.604e-2` |
| `m5_local_mdot_eta90_Nsrc32_source_domain_qm_halo8_seed` | 8 | `1.150e-1` | `2.648e-2` | `1.150e-1` | `261.781 rg` | `1.749e-2` | `2.577e-2` |
| `m5_local_mdot_eta90_Nsrc32_source_domain_qm_halo4_edges_seed` | 4, edges free | `1.966e-1` | `2.768e-2` | `1.966e-1` | `210.807 rg` | `2.426e-2` | `2.475e-2` |

Global release from the halo-4 seed did not improve it:

- `m5_local_mdot_eta90_Nsrc32_source_domain_halo4_global`
- final remains `3.929e-2`;
- optimizer stopped by `xtol` after 9 evaluations.

### Interpretation of Follow-Up

The multipoint source-domain corrector is a real improvement over the previous
N_source=32 finite-volume seed:

- source audit improves from `3.941e-1` to `~3e-2`;
- production residual improves from `3.803e-1` to `3.929e-2` with halo 4.

But it still does not certify eta_E=90. The dominant residual becomes the
finite-volume mass row at the edge of the local correction window. Increasing
the halo first helps, then worsens; freeing the source-domain edges also
worsens. This looks like a coupling/window-boundary problem rather than a
physical endpoint.

Updated best current point:

- `m5_local_mdot_eta90_Nsrc32_source_domain_qm_halo4_seed`
- N = 201
- production residual: `3.929e-2`
- source-band audit: `3.088e-2`
- interval mass residual: `3.929e-2`
- midpoint interval R/E: `1.934e-2`, `2.604e-2`

## Suggested Next Step

Do not lower eta_E below 90 yet.

The next numerical move should be a dedicated source-domain formulation rather
than more global polishing:

1. Keep the multipoint source-domain rows; they successfully remove most of
   the hidden source-band energy audit.
2. Replace the hard local correction window with a source-plus-buffer block,
   or add explicit interface compatibility rows, because the mass residual is
   now living at the halo boundary.
3. Add analytic/local Jacobian blocks for the finite-volume mass/source-domain
   rows before wider scans.
4. Consider using integrated mass increments as variables inside the source
   block, instead of free logMdot at each node; the current logMdot
   parameterization lets the residual move to the window edge.
5. Keep rectangular quarter-point rows as an audit and optional diagnostic
   penalty, not as the definition of success.

Acceptance criteria remain unchanged:

- production residual <= `1e-5`;
- source-band audit <= `1e-5` preferred, <= `3e-5` exploratory;
- mass residual <= `3e-6`;
- stable Mdot_outer/Mdot_inner, Lrad, Rson, and f_adv;
- no unresolved single-cell source-band energy wall.

## Follow-Up: Source-Plus-Buffer With Integrated Mass Increments

Implemented next in `scripts/run_mdot5_local_mdot_eta_continuation.py`:

- `IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BUFFER_CORRECT=1`
- builds a buffered source window around the compact source annulus;
- keeps source support inherited from the micro-domain grid;
- adds auxiliary interval-integrated mass variables `DeltaM_i` during the
  local reduced solve;
- replaces one direct source-buffer mass row by two conservative rows:
  - `DeltaM_i - integral_i(Mwind_prime - Mstream_prime) = 0`;
  - `Mdot_{i+1} - Mdot_i - DeltaM_i = 0`;
- keeps state residual rows sampled inside each source-buffer interval;
- supports midpoint or Simpson wind quadrature for the reduced corrector:
  - `IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BUFFER_MASS_QUADRATURE=midpoint|simpson`;
- uses sparse local finite-difference Jacobian support for the reduced block;
- leaves the production residual and source-band audit unchanged.

All runs in this section start from the previous best halo-4 source-domain
checkpoint:

`outputs/checkpoints/m5_local_mdot_eta90_Nsrc32_source_domain_qm_halo4_seed/stage_00_etaE_90_N201.npz`

### Source-Buffer Results

| run | fractions | quadrature | edges | final | source audit | interval mass | jump row | alpha | nfev |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `m5_local_mdot_eta90_source_buffer_mid_from_halo4_seed_eta90` | `0.5` | midpoint | frozen | `3.889e-2` | `3.025e-2` | `3.889e-2` | `2.225e-2` | `0.125` | 60 |
| `m5_local_mdot_eta90_source_buffer_mid_edges_from_halo4_seed_eta90` | `0.5` | midpoint | free+anchored | `3.891e-2` | `3.025e-2` | `3.891e-2` | `2.221e-2` | `0.125` | 60 |
| `m5_local_mdot_eta90_source_buffer_qm_midmass_from_halo4_seed_eta90` | `0.25,0.5,0.75` | midpoint | frozen | `3.929e-2` | `3.088e-2` | `3.929e-2` | `2.363e-2` | `0` | 50 |
| `m5_local_mdot_eta90_source_buffer_mid_exactmass_from_halo4_seed_eta90` | `0.5` | Simpson | frozen | `3.922e-2` | `3.906e-2` | `3.922e-2` | `2.303e-2` | `0.25` | 25 |

Additional diagnostic:

- direct finite-volume `Mdot` reconstruction across the source-buffer window
  can reduce the mass row in isolation, but it drives the energy/source-band
  defect to `O(1)`.
- Therefore the remaining problem cannot be fixed by a standalone mass-budget
  reconstruction; state and mass must move consistently.

### Interpretation of Source-Buffer Test

The integrated-mass-increment source-buffer formulation is implemented and
diagnostic, but it is not yet a solution.

What improved:

- the midpoint source-buffer solve gives a small but real reduction:
  `3.929e-2 -> 3.889e-2`;
- the source audit also nudges down:
  `3.088e-2 -> 3.025e-2`;
- the internal jump row improves:
  `2.363e-2 -> 2.225e-2`.

What did not improve enough:

- the same outer source-buffer mass/interface wall remains near
  `R~272 rg`;
- releasing the buffer edges does not help;
- midpoint vs Simpson mass quadrature is not the main limiter;
- adding quarter/mid/three-quarter state samples prevents accepting a mass
  improvement because the production mass residual worsens;
- the full source-buffer step wants to trade mass residual against the
  quarter-point source-band audit, so the line search damps it heavily.

Updated conclusion:

The current obstruction is sharper than before. It is not just missing
integrated `DeltaM_i` variables. It is a coupled source-annulus compatibility
problem: the solver can reduce either the mass jump or the intra-cell
source-band residual, but the present nodal state representation cannot reduce
both simultaneously to certification accuracy.

Recommended next numerical move:

1. Promote true subcell/source-band state degrees of freedom, not only
   interval mass increments.
2. Try a real collocation element inside the source annulus with midpoint
   state variables or Lobatto/Hermite-Simpson internal state unknowns.
3. Keep finite-volume mass as conservative rows, but couple it to the same
   internal state variables used by the radial/energy rows.
4. Add analytic Jacobian pieces for:
   - `Mdot_{i+1} - Mdot_i - DeltaM_i`;
   - `DeltaM_i - integral_i(source)`;
   - local wind integral sensitivities only after the state representation is
     upgraded.
5. Do not lower eta_E below 90 yet.

## Follow-Up: Real Source-Element Internal Nodes

Implemented an opt-in source-element refinement in
`scripts/run_mdot5_local_mdot_eta_continuation.py`:

- `IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_ELEMENT_REFINE=1`
- splits each selected source-plus-buffer interval into real collocation
  subintervals;
- preserves all old nodes and inserts internal source-element nodes;
- default test used `SOURCE_ELEMENT_SUBDIVISIONS=2`, so one internal node is
  inserted per selected interval;
- for the current N201 halo-4 checkpoint this gives N251:
  - old N: `201`
  - new N: `251`
  - inserted nodes: `50`
- later local source-domain/source-buffer/block correctors operate on these
  real internal nodes, not disposable auxiliary midpoint variables.

Also fixed a numerical-efficiency bug:

- when source-band extra rows are audit-only, global polishing now keeps the
  sparse Jacobian path instead of falling back to dense finite differences.

### Source-Element Results

All runs start from:

`outputs/checkpoints/m5_local_mdot_eta90_Nsrc32_source_domain_qm_halo4_seed/stage_00_etaE_90_N201.npz`

Reference N201 result:

- production residual: `3.929e-2`
- source-band audit: `3.088e-2`
- interval mass: `3.929e-2`

Raw source-element refinement:

- run: `m5_local_mdot_eta90_source_element_refine2_seed_eta90`
- N: `251`
- production residual: `1.528e-1`
- source-band audit: `1.162e-1`
- interpretation: linear insertion of internal nodes creates a large artificial
  source-edge/source-buffer mass defect near `R~221 rg`.

Mass-budget seeding was tested and rejected:

- midpoint and Simpson finite-volume mass-budget seeds both worsened the
  energy/source-band residual to `O(1)`;
- standalone `Mdot` redistribution is not a safe seed.

Targeted mass-block repair:

- run: `m5_local_mdot_eta90_source_element_refine2_blockmass_relaxed_eta90`
- relaxed the block-guard for this seed-repair step;
- production residual recovered from `1.528e-1` to `3.929e-2`;
- source-band audit remained poor at `1.162e-1`.

Source-domain repair on the N251 grid:

- run: `m5_local_mdot_eta90_source_element_refine2_blockmass_then_domain_eta90`
- production residual: `3.929e-2`
- source-band audit: `3.403e-2`

Source-buffer tie-break polish:

- run: `m5_local_mdot_eta90_source_element_refine2_block_domain_buffer_tiebreak_eta90`
- production residual: `3.929e-2`
- source-band audit: `3.338e-2`
- accepted a damped audit-improving step with unchanged production residual.

Sparse global polish plus source-domain repair:

| run | N | production | source audit | interval R | interval E |
| --- | ---: | ---: | ---: | ---: | ---: |
| old N201 halo-4 | 201 | `3.929e-2` | `3.088e-2` | `1.934e-2` | `2.604e-2` |
| N251 after source-buffer tie-break | 251 | `3.929e-2` | `3.338e-2` | `2.744e-2` | `2.426e-2` |
| N251 sparse global polish | 251 | `3.767e-2` | `7.498e-2` | `2.694e-2` | `2.415e-2` |
| N251 global + domain repair | 251 | `3.752e-2` | `4.217e-2` | `2.696e-2` | `2.426e-2` |
| N251 global + domain repair x2 | 251 | `3.752e-2` | `3.612e-2` | `2.696e-2` | `2.428e-2` |

A third source-domain repair pass did not improve further.

### Interpretation

This sprint successfully implemented real internal source-element nodes and
showed that the refined N251 grid can be stabilized. The sparse global polish
also gives a modest production-residual improvement:

- old N201 production residual: `3.929e-2`;
- best N251 production residual: `3.752e-2`.

But the source-band audit is still worse than the old N201 result:

- old N201 source audit: `3.088e-2`;
- best repaired N251 source audit: `3.612e-2`.

So this is not yet a certified improvement. The current evidence says:

- real internal nodes help make the formulation more honest;
- they do not by themselves remove the `O(3e-2)` source-annulus defect;
- the remaining wall is still a coupled energy/mass/source-edge compatibility
  problem, not just lack of subcell resolution.

Recommended next move:

1. Do not lower eta_E below 90 yet.
2. Add production source-band extra rows with a sparse/local Jacobian, instead
   of audit-only rows, so the global solve cannot trade production mass against
   hidden source-band energy.
3. Alternatively implement a true rectangular/penalty continuation:
   gradually increase source-band extra-row weight while preserving the
   finite-volume mass rows.
4. Keep the N251 source-element checkpoint as an exploratory state, not a
   certified branch point.
