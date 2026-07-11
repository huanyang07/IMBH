# Unified Terminal-Bernoulli Wind Results

Date: 2026-07-11

## Scope

This work replaces the artificial eta launch multiplier as the physical wind
control with a prescribed terminal Bernoulli energy:

```text
B_wind = B_infinity,
E_launch = B_infinity - B_disk - Omega (l_w-l).
```

The target remains the unified conservative `Mdot_inner/Edd=5`, `Rout=335 rg`,
compact-stream, `f_s=0.30`, `epsilon_w=0.20` branch. Eta remains available as
a numerical regression mode.

## Implementation

`PhysicalTransportClosure` now provides:

```text
wind_launch_mode = eta | terminal_bernoulli
wind_terminal_bernoulli
wind_mass_loading_cap_per_log_radius
```

Both the carried-energy ledger and the wind mass law call the same launch
energy function. A terminal target is rejected if its required launch energy
is non-positive; no hidden energy floor is used.

The optional mass cap enforces

```text
dMdot_wind/dlnR <= cap * Mdot_local.
```

If active, the effective launch power is reduced consistently with the capped
mass loss. The unused allocated power is not silently counted as wind energy.

## N426 Continuation

The eta=8 root seeds a terminal-energy ladder. All accepted roots use the same
`3e-5` raw residual gate and cap `0.3` per logarithmic radius.

| B_infinity/c2 | maximum | nfev | wind/Mdot_inner | cap cells |
|---:|---:|---:|---:|---:|
| 0.1000 | `2.064e-5` | 51 | `0.01735` | 0 |
| 0.0800 | `2.002e-5` | 41 | `0.02125` | 0 |
| 0.0600 | `1.867e-5` | 60 | `0.02800` | 0 |
| 0.0400 | `1.718e-5` | 60 | `0.03909` | 0 |
| 0.0350 | `1.594e-5` | 68 | `0.04423` | 0 |
| 0.0300 | `1.510e-5` | 77 | `0.05046` | 0 |
| 0.0250 | `1.416e-5` | 80 | `0.05818` | 0 |
| 0.0225 | `1.343e-5` | 86 | `0.06322` | 0 |
| 0.0200 | `1.327e-5` | 97 | `0.06885` | 0 |

The large direct jump from `0.04` to `0.02 c2` failed at `7.66e-5`. A retry
from `0.025` reached `4.78e-5`. Inserting `0.0225` produced the accepted
`0.02` root. This is a continuation/corrector limitation, not cap activation or
loss of positive launch energy.

A global tangent experiment was also rejected: LSMR reached 10,000 iterations
with residual norm `0.467`, and the tangent predictor exported flux improvement
into radial residual. A future tangent should use better scaling and a
parameter-aware analytic/block derivative.

## Mesh Validation

Nested refinement preserves all old nodes and repolishes the production
equations:

| N | maximum | wind/Mdot_inner | cap cells |
|---:|---:|---:|---:|
| 426 | `1.327e-5` | `0.068850` | 0 |
| 512 | `1.266e-5` | `0.068749` | 0 |
| 640 | `1.264e-5` | `0.068729` | 0 |

Thus the `B_infinity=0.02 c2` root is mesh supported. The configured mass cap
does not create this branch.

## Thermal Topology

| B_infinity/c2 | f_adv global | f_adv R<=20rg | max H/R | Lrad/LEdd | Rson/rg |
|---:|---:|---:|---:|---:|---:|
| 0.1000 | `0.4004` | `0.4306` | `0.2916` | `1.2626` | `4.4511` |
| 0.0600 | `0.4015` | `0.4333` | `0.2913` | `1.2626` | `4.4511` |
| 0.0400 | `0.4078` | `0.4460` | `0.2909` | `1.2642` | `4.4483` |
| 0.0300 | `0.4081` | `0.4446` | `0.2909` | `1.2646` | `4.4483` |
| 0.0200 | `0.4138` | `0.4535` | `0.2910` | `1.2669` | `4.4456` |

The physical terminal closure increases wind loading from `1.7%` to `6.9%`,
but it does not produce a new hot/advective topology. Disk thickness,
luminosity, and sonic radius are nearly unchanged; advection rises only
modestly.

## Conclusion

This is the first mesh-supported branch in the project where wind escape is
specified by a physical terminal Bernoulli condition rather than an eta
multiplier. It is a stronger mass-loaded wind than the eta=8 branch, but it is
still not the sought distinct hot branch.

Pushing `B_infinity` closer to zero is no longer the highest-value move. The
accepted branch already shows that stronger wind loading alone leaves the disk
topology largely unchanged, while corrector cost grows rapidly. The next
physical experiment should prescribe absolute stream supply and allow the
inner rate, overflow, or accumulation to emerge.

## Reproduction

```text
scripts/run_unified_conservative_terminal_bernoulli_ladder.py
scripts/audit_unified_conservative_terminal_bernoulli_branch.py
scripts/run_unified_conservative_terminal_bernoulli_mesh_validation.py
```

Verification:

```text
209 passed, 4 subtests passed
```
