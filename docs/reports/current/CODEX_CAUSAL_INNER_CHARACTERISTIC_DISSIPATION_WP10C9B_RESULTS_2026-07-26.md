# WP10c9b — Family-resolved characteristic dissipation

## Verdict

WP10c9b is a binding negative result for the tested full five-family matrix
dissipation and a positive certification of the missing coordinate principal
basis:

```text
characteristic_matrix_rejected_bdf_noise_and_inward_shear_damping_unresolved
```

The audit-only candidate replaces the scalar maximum-speed Rusanov penalty by

```text
R |Lambda| L Delta Q_descriptor
```

while retaining the production central physical flux, one shared face flux,
the exact embedded fine/coarse coupling, responsive-height storage, and the
unchanged physical boundary maps. Production remains
`interior_dissipation_mode="scalar_rusanov"` by default.

The coordinate eigensystem is real, complete, continuous, causally outgoing
at excision, and closes its eigenpairs and left/right biorthogonality near
machine precision. The candidate also passes constant-state, equal-speed,
non-negative dissipation, shared-flux, storage, dense/colored Jacobian, and
smooth characteristic-dissipation order checks.

It is nevertheless rejected for two independent reasons:

1. the inward causal-shear packet still has nonconvergent damping,
   `p_damping = -0.02646 < 0.75`;
2. a nonlinear synthetic BDF step stagnates at a scaled residual
   `7.65e-9`, above the unchanged `1e-10` tolerance, before a replayable
   split history exists.

Four other pure families pass. Because the inward-shear result is binding,
the WP10c8z common mode is not rerun and bounded nonlinear patch truth is not
authorized.

The main scientific correction to WP10c9a is therefore:

```text
scalar max-speed Rusanov dissipation is not the sole cause of the shear
damping failure.
```

The next target must include the causal-shear nonconservative principal
product in a path-consistent interface coupling; another conservative
dissipation matrix alone is not justified.

## Complete coordinate principal basis

The principal pencil is assembled in

```text
(ln Sigma, beta_R, beta_phi, ln T, chi)
```

as

```text
A(p) p_ct + B(p) p_R = lower-order terms.
```

`A` contains:

- the full mapped five-field Killing/stress storage derivative;
- all four responsive-height temporal-storage components.

`B` contains:

- the derivative of the complete physical face flux;
- the vertical-work spatial principal contribution;
- the resolved causal-shear gradient source.

Across every active inner face:

| Grid | Maximum ideal-cone speed defect | Maximum eigenpair defect | Maximum biorthogonality defect | Maximum scaled condition | Minimum cross-face continuity |
|---|---:|---:|---:|---:|---:|
| N128 inner | `2.49576e-3` | `4.38e-15` | `8.77e-14` | `2.80e3` | `0.999996` |
| N256 inner | `2.49696e-3` | `5.84e-15` | `9.59e-14` | `2.75e3` | `0.999979` |
| N512 inner | `2.49716e-3` | `5.16e-15` | `1.29e-13` | `2.73e3` | `0.999984` |

Every eigenvalue is real and every tested excision face has zero incoming
physical characteristics.

The roughly `2.5e-3 c` difference from the ideal Valencia cone is resolved
and stable under refinement. It comes from using the complete implemented
background-stress/descriptor pencil rather than the stress-free analytic cone.
WP10c9b reports both. It does not overwrite the implemented eigenvalues with
the ideal values.

## Audit-only matrix dissipation

The candidate evaluates one symmetric face state, constructs the descriptor
basis, and applies the path-integrated mapped plus responsive-height storage
jump:

```text
F_star =
    0.5 (F_L + F_R)
    - 0.5 R |Lambda| L Delta Q_descriptor.
```

The row-equilibrated basis is used internally so the very different physical
units of mass, radial momentum, angular momentum, Killing energy, and stress
do not corrupt the characteristic projection.

Method identities are:

```text
constant-state dissipative flux defect = 0
equal-speed scalar reduction defect    = 2.22e-13
minimum characteristic quadratic form = 0
shared embedded-patch flux defect      = 0
maximum storage-action defect          = 2.15e-9
dense/colored Jacobian defect          = 0
omitted sparse-Jacobian entry defect   = 0
```

The zero minimum quadratic value comes from exactly zero reconstructed jumps;
all nonzero sampled jumps have non-negative characteristic dissipation.

The three candidate generators close:

| Grid | Generator factorization defect | Storage-action defect | Build wall time |
|---|---:|---:|---:|
| N128 inner | `3.87e-15` | `2.15e-9` | `243.9 s` |
| N256 inner | `5.33e-15` | `1.07e-9` | `427.2 s` |
| N512 inner | `4.44e-15` | `7.80e-10` | `776.0 s` |

## Nonlinear BDF contract failure

The audit candidate is deterministic and its colored Jacobian matches every
dense column on the small-grid contract. It does not meet the existing
nonlinear step tolerance.

The first synthetic BDF step reaches:

```text
maximum scaled residual             = 7.65361e-9
maximum scaled algebraic residual   = 7.65361e-9
discrete-ledger relative defect     = 4.24257e-11
maximum Newton iterations           = 9
binding residual tolerance          = 1.0e-10
completed steps                     = 0
```

This is not a conservation failure. It is a nonlinear/Jacobian noise or
conditioning floor introduced by differentiating a numerically assembled,
state-dependent eigensystem. The tolerance is not relaxed, so bitwise BDF2
split/replay is not certified.

This alone rejects promotion. It is not the reason the pure shear packet
fails, because the packet ladder uses the already factorized frozen-linear
candidate generators.

## Pure-family packet ladder

All five packets use the exact WP10c9a continuum support and the same
N128/N256/N512 live-coupled grids. No slow-coordinate projection or rematched
anchor is introduced.

| Family | State-history order | Rate-history order | Phase order | Damping order | Smooth dissipation order | Fine minimum cosine | Pass |
|---|---:|---:|---:|---:|---:|---:|---|
| inward acoustic | `1.7976` | `1.2019` | `3.2050` | `2.4969` | `2.6941` | `0.99853` | yes |
| inward shear | `1.5853` | `1.0151` | `1.4474` | `-0.02646` | `2.7004` | `0.99863` | **no** |
| material/contact | `1.8305` | `1.2768` | `1.5049` | `1.3730` | `2.6943` | `0.99891` | yes |
| outward shear | `1.6408` | `1.0428` | `1.8388` | `2.2489` | `2.7004` | `0.99882` | yes |
| outward acoustic | `1.7249` | `1.1229` | `2.8423` | `1.9531` | `2.6956` | `0.99853` | yes |

Every packet stays far from the live coupling:

```text
maximum coupling signal fraction <= 1.21e-18.
```

The inward-shear same-time direction remains excellent. Its failure is again
a fine amplitude/damping error, not a family swap or phase reversal.

Relative to scalar Rusanov, its damping order changes only from

```text
-0.07156 -> -0.02646.
```

That small change is nowhere near the `0.75` gate. A full family-resolved
penalty therefore does not solve the binding spatial error.

## Main problems and solutions

### Problem 1 — The causal-shear wave is not a purely conservative Riemann family

The fifth equation obtains part of its spatial principal coupling from the
resolved rest-frame shear gradient in the relaxation source. That
nonconservative product is presently discretized as a cell source, separately
from the shared central/Rusanov face flux.

Changing only

```text
scalar |lambda|max I -> R |Lambda| L
```

repairs the family-dependent conservative penalty but does not make the
central flux and shear-gradient source one path-consistent Riemann operator.
The failed inward-shear damping ladder is direct evidence that the split
remains controlling.

#### Solution

Derive a path-conservative shear fluctuation:

```text
P_shear =
    integral_0^1 B_shear(Psi(s)) dPsi/ds ds
```

and combine it with the physical flux jump and the two shear-family projectors
in one interface solve. Mass, angular momentum, and Killing-energy fluxes must
remain one shared telescoping vector. The nonconservative fifth-row
fluctuation must be reported separately rather than disguised as a conserved
flux.

### Problem 2 — A numerically differentiated full eigensystem is not a robust nonlinear flux kernel

The full coordinate basis is an excellent audit object, but repeated
finite-difference map derivatives plus a generalized eigensolve introduce a
nonlinear residual floor near `1e-8`.

#### Solution

For the next shear-specific candidate, derive a closed-form or
automatic-differentiation-compatible two-family projector. It must be smooth
under state perturbations and must pass the unchanged `1e-10` nonlinear
tolerance before any live generator or BDF2 replay is accepted.

### Problem 3 — The scalar-Rusanov term audit was a localization, not a causal proof

WP10c9a correctly found Rusanov to be the largest forcing-side refinement
defect. WP10c9b shows that removing its scalar over-damping does not restore
shear damping convergence.

#### Solution

Retain the WP10c9a term decomposition, but now split the shear family into:

- conservative central stress transport;
- family penalty;
- nonconservative shear path contribution;
- stress relaxation;
- mapped and responsive-height descriptor response.

The next candidate is accepted only if the complete split contracts.

### Problem 4 — The mixed common mode remains untested under the candidate

The locked gate requires every pure packet to pass first. Inward shear does
not, so rerunning the common mode would confound a known linear family error
with nonlinear family coupling.

#### Solution

Keep the common mode blocked until both inward and outward shear packets pass
the three-grid phase, damping, smooth-order, Jacobian, storage, and BDF
contracts.

## Locked next plan: WP10c9c

### Phase 1 — Freeze WP10c9b

Freeze and hash:

- the complete coordinate principal bases on all active faces;
- all three matrix-dissipation generators;
- the five pure packet histories;
- the nonlinear BDF failure diagnostics;
- the exact WP10c9a parent evidence.

Do not promote the matrix flux or relax the BDF tolerance.

### Phase 2 — Extract the causal-shear subsystem

At the N256/N512 active faces:

1. construct the inward/outward shear spectral projectors;
2. project the full temporal descriptor, central flux, family penalty, and
   shear-gradient source into the two-family subspace;
3. verify that the projected small-jump system reproduces the full coordinate
   shear eigenpairs;
4. localize the N256/N512 damping defect by face, time, and projected term.

Stop if the shear subspace is not smooth and mesh aligned.

### Phase 3 — Derive one path-conservative shear coupling

For a declared admissible face path `Psi(s)`:

```text
Delta F_path =
    F_R - F_L
    + integral B_shear(Psi) dPsi.
```

Build left/right fluctuations from this total path jump and the two physical
shear speeds. Requirements:

- one shared conservative `M/J/E_K` face flux;
- exact equal-and-opposite conservative coupling;
- a separately declared fifth-row nonconservative fluctuation;
- constant-state and path-reversal identities;
- small-jump agreement with the complete principal pencil;
- no incoming excision mode;
- non-negative shear-family dissipation;
- no fitted viscosity or transient-specific coefficient.

Use a closed-form or AD-compatible two-family basis. Do not reuse the
finite-difference/eigensolve kernel as a nonlinear production candidate.

### Phase 4 — Method, Jacobian, and BDF certification

Before a packet history, require:

```text
constant-state path defect             = 0
path-reversal identity defect          <= 1e-12
small-jump principal closure           <= 1e-10
smooth shear state/rate order          >= 1.8
shared conservative flux defect        <= 1e-12
storage-action defect                  <= 2e-5
dense/colored Jacobian parity          <= 1e-10
nonlinear residual                     <= 1e-10
BDF2 split/replay                      = bitwise
incoming excision characteristics      = 0
```

Stop before generator construction if any gate fails.

### Phase 5 — Shear-first packet ladder

Run only:

1. inward shear;
2. outward shear.

Use the unchanged N128/N256/N512 hybrid grids and packet definitions. Require:

```text
centroid phase order >= 0.75
damping order        >= 0.75
same-time cosine     >= 0.90
smooth order         >= 1.8
```

The inward-shear damping result is binding.

### Phase 6 — Conditional all-family and common-mode ladders

Only if both shear families pass:

1. rerun acoustic/contact packets to prove no regression;
2. rerun the exact WP10c8z common mode;
3. authorize one bounded nonlinear embedded-patch truth experiment only if
   the common mode contracts.

If the shear path candidate still fails, redesign the complete
near-horizon path-conservative finite-volume operator. Do not authorize
N1024 brute-force refinement.

### Hard stops

WP10c9c must not:

- change production by default;
- loosen the `1e-10` nonlinear contract;
- run fixed-`Q` averaging;
- select an initial-slip map or reduced coordinate;
- launch a production embedded patch;
- run loading-time macrosteps;
- add tide, wind, hot-state, or cycle physics.

## Machine evidence

```text
outputs/tables/causal_inner_characteristic_dissipation_audit_wp10c9b.json
outputs/tables/causal_inner_characteristic_dissipation_audit_wp10c9b_arrays.npz
```

Runner:

```text
scripts/run_causal_inner_characteristic_dissipation_audit_wp10c9b.py
```

Core audit implementation:

```text
src/imri_qpe/layer3_minidisk_1d/causal_inner_characteristic_dissipation.py
```

Focused tests:

```text
tests/test_causal_inner_characteristic_dissipation.py
tests/test_causal_inner_characteristic_dissipation_audit_wp10c9b.py
```

## Verification

```text
WP10c9a/WP10c9b focused tests       8 passed
shared causal-subsystem tests       89 passed
full repository suite               769 passed, 4 subtests passed
repository hygiene                  passed for 749 tracked files
git diff --check                    passed
```

Machine-evidence hashes:

```text
JSON   d569f80d303c3297164960bb1d774990db82babab6d4b97f5b4ab4a8b23db7b2
arrays ac74b019a56f0901e86fd2dbf3018641cde1f51dddeb19cbbaec22109ffe60f3
```
