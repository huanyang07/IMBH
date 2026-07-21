# WP10c8i Storage-Consistent Moment-Sufficiency Audit

Date: 2026-07-21

Base commit under test:
`3e204d173a71f5c2ad02228e7c673601a7316e11`

## Decision

WP10c8i implements the requested storage-consistent operator audit, but it
does not establish either sufficiency or insufficiency of any candidate
moment set:

```text
decision                         wp10c8i_moment_sufficiency_inconclusive
meshes                           N64, N128
construction anchors             0, 0.025, 0.05, 0.125 s
held-out anchors                 0.075, 0.10 s
frozen-linear horizons           0, 0.01, 0.025 s
new full-DAE trajectory          no
nonlinear lift/healing burst     no
reduced nonlinear evolution      no
online-cost gate                 not evaluated
```

The complete responsive-height storage and candidate-coordinate machinery
is now explicit and tested. The binding finite-time moment decision remains
inconclusive because the generator finite-difference-consistency and exact finite-branch
contracts do not both pass at every anchor. Raw null-space gains are retained
as conditional diagnostics only; they are not used to add or reject a moment
family.

## Scope

The package reuses:

- the certified N64/N128 no-tide states through `0.125 s`;
- the existing mesh-coincident five-shell layout;
- the production spatial operator and exact linked stream source;
- the full five-field primitive descriptor;
- the existing scientific gates for cooling, exterior cooling, inner
  accretion, and the complete thickness profile.

It adds no new physics and runs no new nonlinear truth trajectory. The
operator audit is offline and may call the full N64/N128 residual and
Jacobian. Such calls remain forbidden in an ordinary reduced evaluation.

## Complete Vector Storage Contract

Responsive-height storage is retained as the vector one-form

\[
\boldsymbol\omega_H(x)[dx]
\propto
\left(0,u_R,u_\phi,-u_t,0\right)d\ln H .
\]

Its mass and stress-storage components vanish, while its radial-momentum,
angular-momentum, and Killing-energy components generally do not. The audit
therefore pulls back the complete vector one-form into the primitive
descriptor. It does not invent an instantaneous effective energy state.

The cumulative object

\[
\boldsymbol W_H(t)
=
\int_{t_0}^{t}\boldsymbol\omega_H(x(t'))[\dot x(t')]\,dt'
\]

is continuum notation for the declared discrete path quadrature. It remains
a path ledger and is not used as a shell coordinate or a newly exact state
variable.

Writing the descriptor as

\[
M(x)\dot x+R(x)=0,
\]

linearization along an evolving anchor \(x_a(t)\) gives

\[
M_a\,\delta\dot x
+
\left[
DR_a
+
\mathcal K_a
\right]\delta x
=0 .
\]

Here

\[
\mathcal K_a v=(DM_a[v])\dot x_a .
\]

The audited frozen generator includes the state-dependent
\((DM[v])\dot x_a\) action. Endpoint and increment operators are,
respectively,

\[
Oe^{Lh},
\qquad
O(e^{Lh}-I).
\]

These are exact actions of the selected frozen linearized generator, not
exact nonlinear finite-time responses.

## Incremental Candidate Coordinates

The five cumulative levels are:

| Level | Coordinates | Count |
|---|---|---:|
| 0 | Five-shell instantaneous \(M/J/E_K\) | 15 |
| 1 | Level 0 plus shell mean \(\ln T\) | 20 |
| 2 | Level 1 plus shell radial momentum | 25 |
| 3 | Level 2 plus shell causal-stress storage | 30 |
| 4 | Level 3 plus targeted first \(\ln T/\ln\Sigma\) shape moments | 34 |

The 15 shell conserved coordinates are an invertible recharting of three
global totals plus zero-sum shell redistributions. Killing energy is
instantaneous. Responsive-height work remains solely in the descriptor and
path ledger.

The unresolved complement is denoted \(\eta\), but WP10c8i does not assume
that it is fast, Markovian, or algebraically slaved. Before closure, the
projected identity still contains \(\eta\), \(\dot\eta\), moving-fiber terms,
and potentially memory.

The mesh-coincident shell edges, in \(r_g\), are

\[
(1.8,\ 6.127035383335207,\ 60.294232385347335,
205.23605290892192,\ 284.52106300239655,\ 335.0).
\]

They are the unchanged five-shell layout from WP10c8h: the nominal interior
targets were \(6,60,200,280\,r_g\), with the physical domain endpoints added.
The two targeted shape bands are \(6\)–\(60\,r_g\) and
\(200\)–\(280\,r_g\).

## Null-Space Audit

For each candidate coordinate matrix \(C\), the package constructs a
weighted-orthonormal basis of

\[
\ker C
\]

using scaled SVD rank decisions. It then evaluates gate-normalized
finite-time responses using Krylov exponential actions followed by an
explicit null-space SVD.

The response stack includes:

- the original scientific observables;
- integrated rest mass, angular momentum, and Killing energy;
- the complete native \(\log(H/R)\) profile and its independent 129-point
  common-radius cross-mesh reconstruction;
- macro-interface mass, angular-momentum, and Killing-energy fluxes;
- the window-scaled coarse-rate operator \(0.025\,C L\).

For every output, pointwise admissibility boxes give a rigorous lower and
upper gain bound. The three-way rule is:

- upper bound at or below `0.25`: screening pass;
- lower bound above `0.25`: proven screening failure;
- otherwise: inconclusive.

The cross-mesh-normalized input metric is a continuum-\(L_2\) metric with
primitive pointwise amplitude boxes

```text
log surface density            0.01
radial three-velocity / c      0.002
azimuthal three-velocity / c   0.002
log temperature                0.01
specific causal stress         1% of the maximum absolute equilibrium
                               target stress, with a robust median floor
```

The production finite-difference step is `2e-6`. The inner and outer
generator ladders use `1e-6`, `2e-6`, and `4e-6`; the complete vector-storage
action ladder uses `5e-5`, `1e-4`, and `2e-4`. The unchanged maximum
generator FD-consistency relative defect is `5e-3`.

The Rusanov audit uses a declared control-margin threshold of `1e-8`, a
numerical-tie margin of `1e-14`, and a maximum suppressible component-scaled
conserved jump of `1e-4`. Every characteristic-speed candidate within the
declared margin is included rather than selecting one nominal branch.

The corresponding pre-microburst reserve is `0.10`. Neither threshold is
binding unless rank, storage, generator, differentiability, exact Rusanov
branch, and cross-mesh contracts all pass.

## Numerical Contract Result

The per-anchor numerical results and controlling failures are recorded in
the generated JSON. The final tables below are populated from that artifact.

| Mesh | Anchor (s) | Vector storage | Generator FD consistency | Local tangent | Exact finite branch | Consequential kinks |
|---:|---:|:---:|:---:|:---:|:---:|---:|
| N64 | 0 | pass | **fail** (`0.65472`) | pass | **fail** | 12 |
| N64 | 0.025 | pass | pass (local scan) | pass | **fail** | 1 |
| N64 | 0.05 | pass | pass (local scan) | pass | pass | 0 |
| N64 | 0.075 | pass | pass (local scan) | pass | pass | 0 |
| N64 | 0.10 | pass | **fail** (`0.39247`) | pass | pass | 0 |
| N64 | 0.125 | pass | pass (local scan) | pass | pass | 0 |
| N128 | 0 | pass | **fail** (`0.29758`) | pass | **fail** | 27 |
| N128 | 0.025 | pass | pass (local scan) | pass | pass | 0 |
| N128 | 0.05 | pass | pass (local scan) | pass | pass | 0 |
| N128 | 0.075 | pass | pass (local scan) | pass | **fail** | 1 |
| N128 | 0.10 | pass | **fail** (`0.21034`) | pass | pass | 0 |
| N128 | 0.125 | pass | pass (local scan) | pass | pass | 0 |

The parenthesized generator values are the largest deterministic physical
JVP changes between the `1e-6/2e-6` pair in the declared full scan; the gate
is `5e-3`. The corresponding `4e-6/2e-6` defects are
`0.30723/0.19307` on N64 and `0.13434/0.19288` on N128 at
`0/0.10 s`. A “local scan” row passed the all-anchor production JVP checks
but was not one of the two predeclared full separated-FD anchors.

Here “generator stability” in the schema means finite-difference step
consistency of the constructed evolving-anchor generator; it is not spectral
or dynamical stability. Schema-9 compatibility note: the per-anchor JSON field named
`maximum_relative_defect` stores the declared maximum *allowed* defect
`0.005`; it is not the measured defect. The measured values are the nested
`maximum_deterministic_physical_jvp_relative_defect` entries quoted above.

Across all anchors, the complete vector-storage action defect is at most
`2.98e-7`, the direct storage rank is full, the largest direct storage
condition estimate is `4.55e9 < 1e12`, the component reconstruction defect
is below `2.14e-13`, and the generator factorization defect is below
`9.10e-13`. Thus storage succeeds independently of the generator FD-consistency
failure.

## Conditional Moment Results

Because at least one binding numerical contract fails, all moment gains in
this section are conditional diagnostics. They do not prove that a candidate
is sufficient or insufficient.

| Candidate | Count | Maximum condition | Construction lower / upper | Held-out lower / upper | Conditional controller |
|---|---:|---:|---:|---:|---|
| shell \(M/J/E_K\) | 15 | `4.45e3` | `338.949 / 426.315` | `338.400 / 423.466` | interface 4 angular momentum |
| + mean \(\ln T\) | 20 | `4.45e3` | `338.949 / 426.315` | `338.400 / 423.466` | interface 4 angular momentum |
| + radial momentum | 25 | `5.15e3` | `341.271 / 426.315` | `340.663 / 423.466` | interface 4 angular momentum |
| + stress storage | 30 | `5.56e3` | `340.975 / 426.315` | `340.419 / 423.466` | interface 4 angular momentum |
| + targeted shape moments | 34 | `5.56e3` | `340.990 / 426.315` | `340.434 / 423.466` | interface 4 angular momentum |

Every coordinate matrix has full declared row rank. The construction maxima
come from N128 at `0.125 s`; the held-out maxima come from N128 at `0.10 s`.
The lower maxima occur for the `0.025 s` endpoint response and the upper
maxima at the zero-horizon endpoint operator. All raw lower bounds are far
above the `0.25` screen, but this is only evidence about the selected,
uncertified generator.

The overall maximum lower/upper gains agree between N64 and N128 to within
`0.095/0.158`, but the binding 129-point common-radius thickness bounds do
not: the largest lower/upper relative differences range from
`0.640/0.621` for the 15-coordinate level to approximately
`0.898/0.844` for the richest levels. Consequently no candidate passes the
cross-mesh contract, even conditionally.

## Rusanov Nonsmoothness

Every characteristic-speed candidate inside the declared near-tie margin is
enumerated. For each consequential branch the cache stores its face, radius,
control labels, relative margin, component-scaled conserved jump, rank-one
generator difference, and direct output difference.

The generalized-Jacobian Fréchet response is retained only as a diagnostic.
It omits the nonlinear exponential remainder and therefore cannot restore a
binding finite-time decision. Exact absence of consequential kinks—or a
future rigorous finite-branch enclosure—is required.

## Online-Cost Stop

WP10c8i is an offline sufficiency audit, not an online reduced evaluator.
Accordingly:

```text
ordinary reduced evaluation implemented       no
full N128 call in ordinary reduced evaluation no
online cost measured                          no
required future online/full cost              <= 0.10
complete reduced-model contract               false
```

No level may pass the complete model contract until lifting, closure,
Jacobian, ledger, and amortized microburst costs are measured below `10%` of
the full operator, with no routine full-N128 residual or Jacobian call.

## Interpretation

The result is narrower than another closure no-go:

> WP10c8i implemented the requested storage-consistent operator audit, but
> did not establish either sufficiency or insufficiency of any candidate
> moment set because the generator finite-difference-consistency and exact finite-branch
> numerical contracts do not both pass at every anchor.

It would be incorrect to add moments based on nonbinding gains, loosen the
`5e-3` generator FD-consistency gate, or begin nonlinear lifting.

## Next Authorized Package

Within this reduction branch, the only authorized next package is a bounded
tangent-certification package:

1. replace or certify the nested finite-difference action for the evolving
   descriptor, especially \(DM[\delta x,\dot x]\) and the stationary
   Jacobian, using analytic, automatic-differentiation, complex-step, or
   otherwise independently converged derivatives without relaxing the gate;
2. implement a semismooth/generalized derivative or a rigorous finite-branch
   enclosure for every consequential Rusanov near-tie, including finite-time
   branch switching;
3. rerun the same N64/N128 six-anchor numerical contract before adding
   moments or making any reduced-model claim.

A differentiable dissipation envelope is a separate spatial-operator change
and would require complete spatial and temporal recertification.

The following remain blocked:

- finite-amplitude equal-coordinate lifting;
- constrained healing or nonlinear microbursts;
- factor-two/factor-four macro predictions;
- another moment family;
- loading-time evolution;
- tide, wind, hot-state, stability, or cycle claims.

## Reproduction and Artifacts

Primary runner:

`scripts/run_causal_moment_sufficiency_audit_wp10c8i.py`

Canonical evidence:

| Artifact | SHA-256 |
|---|---|
| `outputs/tables/causal_moment_sufficiency_audit_wp10c8i.json` | `3f90dfb94ad013d05f2d27038ee31fa883c2a9c8a3c3af81f16dac4b285615b3` |
| `outputs/tables/causal_moment_sufficiency_audit_wp10c8i_arrays.npz` | `05074951f6ca58828fbb26c32ba898cf711b0c0bf446d88e32dd8de51690b1e0` |

The 12 resumable operator caches live under
`outputs/checkpoints/causal_five_field_wp10c8i/`. Each uses schema 9 and
contains a complete per-file source SHA manifest plus Python, platform,
NumPy, and SciPy provenance. The N64 and N128 operator-contract SHA-256
values are, respectively,
`0748c012dec3472e65c3f589513b9045d3ee650151bd390ace65560d83d86e59`
and
`35daba90d0cb48971458e4b66db9f122daabd94d2c1bd7fae51b334699e93deb`.
The runner snapshots this contract before cache construction, aborts if
source changes during construction, and verifies every cache before final
assembly. The canonical JSON records each cache path, state hash, and
artifact hash.

These runtime JSON, NPZ, and cache files are intentionally gitignored local
artifacts. The committed report records their exact paths and hashes, but
does not make them part of the source tree.

Validation:

```text
affected causal suite              101 passed
complete repository suite          605 passed, 4 subtests passed
repository hygiene                 passed for 699 tracked files
Python bytecode compilation        passed
git diff --check                   passed
```

Commands:

```text
PYTHONPATH=src:scripts python -m pytest -q \
  tests/test_causal_inner_dae.py \
  tests/test_causal_inner_dae_system.py \
  tests/test_causal_inner_spatial_audit.py \
  tests/test_causal_inner_mixed_reduction.py \
  tests/test_causal_inner_moment_audit.py \
  tests/test_causal_moment_sufficiency_audit_wp10c8i.py

PYTHONPATH=src:scripts python -m pytest -q
python scripts/check_repository_hygiene.py
```
