# ADR 0015: Fresh-loading inner-boundary architecture gate

## Status

Accepted as a stop decision. Neither existing candidate is selected for
production fresh-loading evolution.

## Context

The global fresh low-mass startup closes its equations and ledgers on N64 and
N96, but the first-cell velocity changes sign under refinement. The inner
characteristic count therefore changes from one to three. A fixed
one-incoming-mode boundary is not mesh invariant.

WP8 authorized a comparison of exactly two alternatives:

1. a one-domain inner excision inside a permanently acoustic-supersonic
   plunge; and
2. an evolving outer reservoir coupled to a quasi-steady transonic inner
   response.

## Exact counts

### A. One-domain causal excision

For N finite-volume cells the differential state is

```text
cell mass, radial momentum, angular momentum, total energy     4 N
```

A monolithic backward-Euler step has `4 N` unknown states and `4 N`
conservation rows. At an inward, acoustic-supersonic inner edge all four
radial characteristics point out of the domain, so the boundary supplies zero
rows and uses the one-sided physical flux.

This count is valid only while

```text
incoming inner characteristics = 0
|v_R| < c
c_eff < c
```

on every accepted mesh and state.

### B. Quasi-steady transonic response

The implemented flux-primary hybrid has

```text
inner transonic state                              2 Ni + 2
outer differential primitives                     3 No
face mass fluxes                                   No + 1
face angular fluxes                                No + 1
interface total-energy flux                        1
```

for exactly

\[
2N_i+5N_o+5
\]

unknowns and rows. The rows are the inner core, three outer conservation
blocks, common stress, radial force, two primitive interface conditions, two
flux-extraction conditions, and one open-edge torque condition. Its outer
storage rank is `3 No`.

## Evidence

The low-throughput `0.025 Mdot_Edd` profile has a stationary critical point at
`5.996987 rg`, but its Euler acoustic Mach number there is only `-0.05786`.
It is not an acoustic sonic point of the global four-equation system.

Continuing the same accepted local equations inward gives one incoming
acoustic characteristic at every audited radius from `4.5 rg` through
`2.0001 rg`. At the deepest point the inward speed is `0.862 c`, while the
nonrelativistic gas-radiation sound speed exceeds `c` by more than two orders
of magnitude. Moving the excision closer to the Paczynski-Wiita singularity
therefore does not produce a physically admissible zero-incoming boundary.

Architecture B is ADR 0012, not a new fallback. It accepts coarse repeated
steps, but the `24/16` refinement rejects its third subcycled step at
`1.0466e-7`, above the fixed `1e-7` gate. Primitive elimination moves the
defect into the inner core. All declared interface remedies are closed.

## Decision

Do not select either existing architecture:

- A fails the physical characteristic gate for the low-rate branch.
- B fails the refined repeated-step gate and must not be reopened as another
  splice-conditioning campaign.

The low-rate stationary critical point must not be called an acoustic sonic
point when setting the global time-dependent boundary.

## Consequences

1. No N128 fresh startup, longer evolution, tide, or wind run is authorized.
2. Do not move the edge closer to `2 rg`, cap the sound speed, force the
   velocity sign, or alter the incoming-mode count inside the present model.
3. Do not reopen residual weighting, primitive elimination, or interface
   preconditioning for ADR 0012.
4. The next physics work package must supply a causal inner model whose
   time-dependent characteristics and stationary critical condition are the
   same system. A relativistic or otherwise explicitly causal inner closure is
   required before another production trajectory.
5. The existing high-throughput causal-plunge runs remain valid for their
   stated initial-state scope; this decision concerns fresh low-throughput
   loading.
