# Causal BDF Method WP10c7a Results

Date: 2026-07-18

## Verdict

The increment-primary BDF1/BDF2 method contract passes.

WP10c7a adds:

- validated constant- and variable-step BDF coefficients;
- the variable-step BDF2 zero-stability ratio bound;
- current/previous finite-increment derivative operators;
- a complete five-field history containing the previous physical increment,
  previous path-integrated vertical Killing-storage increment, previous
  timestep, and temporal-height scheme;
- a five-field BDF1/BDF2 residual that leaves primitive, face-flux,
  characteristic, and boundary rows algebraic at the new endpoint;
- distinct discrete-BDF and physical-interval ledger primitives;
- a separate checksummed BDF restart schema with complete two-step history;
- scalar, index-one DAE, vertical-storage, five-field, rank, and restart
  tests;
- one machine-readable N4 method audit.

Every predeclared method gate passes:

```text
stiff scalar order                  2.031-2.063
index-one DAE order                 2.006-2.013
manufactured vertical order         2.039-2.074
physical interval ledger order      2.929-2.964
BDF1/backward-Euler parity defect   0
declared conserved-history defect   0
vertical-history defect             0
N4 BDF2 Jacobian rank               65/65
BDF restart round trip              bitwise
```

Therefore:

```text
WP10c7a method contract                  certified
WP10c7b fixed-step N16 BDF2             authorized
WP10c7c adaptive N16 BDF2               not yet authorized
WP10c7d matched N32 BDF2                not yet authorized
long evolution and new physics          not authorized
```

No production disk trajectory is run in WP10c7a.

## Increment Form

Let:

```text
h_n       = t_(n+1) - t_n
h_(n-1)   = t_n - t_(n-1)
r         = h_n / h_(n-1)
Delta U_n = U_(n+1) - U_n
Delta U_p = U_n - U_(n-1)
```

The variable-step BDF2 derivative is represented without subtracting large
stored states:

```text
dU/dt at n+1 =
  [a0 Delta U_n + ap Delta U_p] / h_n

a0 = (1 + 2r) / (1 + r)
ap = -r^2 / (1 + r)
```

For equal timesteps:

```text
a0 = 3/2
ap = -1/2
```

BDF1 is the same API with:

```text
a0 = 1
ap = 0
```

The public coefficient object validates itself against the declared formula.
Variable-step BDF2 rejects:

```text
r > 1 + sqrt(2)
```

before a residual is evaluated. Production adaptation will use a tighter
step-ratio policy and fall back to BDF1 after a larger forced change.

## Five-Field Storage

The current full increment remains the Newton unknown. The differential
storage rows use:

```text
a0 * current declared conserved increment
+
ap * previous accepted conserved increment
```

The responsive-height storage uses the identical BDF weights:

```text
a0 * current path-integrated vertical Killing increment
+
ap * previous path-integrated vertical Killing increment
```

The previous vertical increment is fixed history, not reconstructed from an
endpoint pressure-work approximation. The history also records whether the
path is `endpoint` or `path_integrated`; mixing the two schemes is rejected.

At the new endpoint, these remain algebraic:

- conserved-to-primitive recovery;
- numerical interior face fluxes;
- inner causal outflow flux;
- physical Roche boundary flux;
- all characteristic counts and boundary active sets.

The existing backward-Euler evaluator now calls the generic order-one BDF
path. Its complete residual and temporal storage arrays remain bitwise
identical.

## Method Tests

### Variable-Step Polynomial

A nonuniform BDF2 formula with:

```text
r = 0.5714285714285713
```

differentiates a quadratic exactly:

```text
measured derivative   2.8
exact derivative      2.8
defect                0
```

### Stiff Scalar Relaxation

For:

```text
y' = -4 y
```

one backward-Euler startup step followed by equal-step BDF2 gives:

| Subdivisions | Absolute endpoint error |
|---:|---:|
| 40 | `1.165e-4` |
| 80 | `2.788e-5` |
| 160 | `6.822e-6` |

Observed orders:

```text
2.06252
2.03103
```

The one order-one startup step contributes an `O(h^2)` one-time endpoint
error and does not reduce global second-order convergence.

### Index-One DAE

The manufactured DAE is:

```text
x' + x - z = 0
z - x/2     = 0
```

The algebraic equation is solved at every new endpoint. For 20/40/80
subdivisions:

```text
orders                     2.01305, 2.00616
maximum algebraic defect   5.55e-17
```

This verifies that BDF differentiation is applied only to differential
storage while the algebraic constraint remains exact.

### Manufactured Vertical Storage

For a smooth manufactured cumulative vertical storage `V(t)=sin(t)`, the
BDF2 derivative at a common endpoint gives:

```text
orders  2.07394, 2.03945
```

The actual five-field audit separately checks the weighted current and
previous gas-radiation path integrals and finds zero relative defect.

## Dual Ledgers

WP10c7a intentionally distinguishes two statements.

### Discrete BDF Ledger

The solved method must satisfy:

```text
a0 Delta U_n + ap Delta U_p + h_n R(U_(n+1)) = 0
```

For the manufactured BDF2 relaxation:

```text
discrete closure defect   1.11e-16
```

This is the machine-precision equation-solving audit.

### Physical Interval Ledger

The actual physical interval uses:

```text
U_(n+1) - U_n
+
h_n [R(U_n) + R(U_(n+1))] / 2
```

Its local defects converge at third order:

```text
2.92895
2.96420
```

Accumulated over a fixed horizon, this supplies a separately second-order
physical budget. A machine-zero BDF residual is not substituted for that
physical cumulative audit.

WP10c7b must preserve component-separated boundary, endogenous-source, exact
stream-source, conserved-storage, and vertical-storage entries in both
ledgers.

## Five-Field Audit

The N4 gas-radiation, causal-stress, stream, and Roche context gives:

```text
BDF1/backward-Euler parity defect              0
declared conserved-history relative defect     0
vertical-history relative defect               0
```

At an equal-step `0.1 s` method-rank point:

```text
Jacobian dimensions     65 x 65
numerical rank          65
condition estimate      7.23e10
```

At very small timesteps, differential storage singular values grow as
`1/h`, so a fixed relative raw-SVD threshold can hide finite algebraic
singular values. WP10c7b must use the repository's equilibrated sparse solve
and report scaled/equilibrated conditioning; it must not reinterpret that
mixed-scale raw threshold as physical rank loss.

No N16 disk root or timestep is claimed here.

## Complete Restart

The BDF restart stores:

- current full `15N+5` state;
- previous full physical increment;
- previous vertical Killing-storage increment;
- previous timestep;
- next requested timestep and order;
- temporal-height scheme;
- elapsed time;
- accepted and rejected counters;
- provenance and schema;
- grid and state/history checksum.

The method checkpoint is:

```text
outputs/checkpoints/causal_five_field_wp10c7a/
  causal_wp10c7a_N004_method_restart.npz

SHA-256
44fa429efcb871f4fa05eb5d36edf4c975ab8ba104328db23140378db38bee53
```

Its round trip is bitwise.

## Machine Evidence

The machine-readable audit is:

```text
outputs/tables/causal_bdf_method_audit_wp10c7a.json

SHA-256
50ecf95de73292ea3d52fbfc3bb995b52859da5622005ff76d3fd79ba3feaba3
```

Runtime artifacts remain ignored under the artifact policy.

## Classification

WP10c7a establishes:

```text
increment-form BDF1 coefficients             certified
variable-step BDF2 coefficients              certified
zero-stability ratio guard                   certified
generic discrete/physical ledger primitives  certified
five-field conserved history                 certified
five-field vertical history                  certified
algebraic endpoint treatment                 certified
complete BDF restart schema                  certified
fixed-step N16 BDF2 trajectory               not yet run
adaptive BDF2 controller                     not implemented
matched N32 BDF2                             not run
```

It establishes no physical relaxation, stable or unstable branch, hot state,
limit cycle, tide response, or wind solution.

## Locked WP10c7b

The next atomic package is fixed-step N16 BDF2 certification.

Keep unchanged:

```text
mesh                         N16 only
initial state                accepted WP10c5q checkpoint
physics                      exact circularized stream, no tide, no wind
target duration              1.537457597966907e-2 s
selected reference           WP10c6e S512 backward Euler
reference uncertainty        raw S256-to-S512 endpoint difference
observable gates             unchanged v1 schema
```

Implement:

1. one backward-Euler startup step;
2. fixed equal-step BDF2 thereafter;
3. the existing equilibrated sparse Newton backend;
4. discrete BDF component ledgers at nonlinear tolerance;
5. trapezoidal physical cumulative ledgers with second-order convergence;
6. exact restart history at an interior fixed-step checkpoint;
7. work telemetry matching the backward-Euler references.

Run the predeclared ladder:

```text
8, 16, 32, 64 subdivisions
```

Use 128 only if the 32/64 result does not resolve a declared gate.

For every non-negligible observable require:

```text
1.7 <= observed order <= 2.3
```

and:

```text
|BDF2 endpoint - S512 endpoint|
+
|S512 endpoint - S256 endpoint|
<= immutable observable gate
```

The first BDF2 disk package must also pass every nonlinear, algebraic,
physical-ledger, causal, optical-depth, Roche, and emergency-change gate.

Do not add adaptivity, Jacobian reuse, a variable-order controller, N32,
N64/N128 production, long timescale, tide, wind, stability, hot-state, or
cycle work in WP10c7b.

## Verification

Before the atomic commit:

```text
new BDF method tests                     10 passed
causal DAE/evolution regression tests    40 passed
machine method audit                     passed
full repository suite                    515 passed, 4 subtests passed
repository hygiene                       passed for 655 staged files
```

## Reproduction

```text
PYTHONPATH=src python3 \
  scripts/run_causal_bdf_method_audit_wp10c7a.py
```
