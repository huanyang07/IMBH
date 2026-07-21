# WP10c8j Evolving-Tangent and Rusanov Certification

Date: 2026-07-21

Base commit under test:
`6233914eab6d9b719b90602243e59c7f09de525d`

## Decision

```text
decision                         wp10c8j_smooth_tangent_failed_rusanov_certificate_absent
next authorization               repair_direct_storage/vector-field tangent and supply finite-neighborhood Rusanov evidence
new full-DAE trajectory           no
new nonlinear microburst          no
moment families changed           no
production Rusanov flux changed   no
unchanged WP10c8i repeat launched no
reduced evolution authorized      no
```

WP10c8j is a bounded negative certification result. It removes the nested
finite-difference construction from the storage-rate derivative, exposes the
smooth tangent as independently audited blocks, and implements the strict
finite-neighborhood Rusanov contract. The repaired matrices are substantially
more consistent than the WP10c8i tangent, but the locked smooth contract does
not pass at every tested anchor and no binding Rusanov neighborhood evidence
exists. The unchanged WP10c8i moment audit therefore remains blocked.

The campaign stopped once failures at locked N64 and N128 anchors made a full
unchanged pass impossible. It completed:

- N64 full scans at `t=0` and `0.10 s`;
- N64 base scans at `t=0.025` and `0.05 s`;
- the N128 full scan at `t=0.10 s`.

No result is claimed for the unrun anchors.

## Scope and implementation

The package reuses without changing:

- the certified no-tide truth states;
- the five-shell layout and five incremental coordinate levels;
- the production spatial operator and exact Rusanov maximum;
- the stream source, physical gates, response gates, and moment gates.

The implementation adds public helpers for:

1. the Schur-reduced stationary primitive Jacobian;
2. mapped-conserved and complete responsive-height storage matrices;
3. the storage-rate derivative `DM[., p_dot]`;
4. assembly and factorization of the evolving generator.

For

\[
M(p)\dot p+R(p)=0,
\]

the generator is

\[
L=-M^{-1}\left(DR+DM[\cdot]\dot p\right).
\]

The new `direct_action` backend applies the full nonlinear storage action to
the fresh physical rate and differentiates that action once in the outer
state coordinate. It avoids differentiating an already finite-differenced
mass matrix. The old nested construction remains diagnostic only.

Responsive-height storage retains its complete vector-one-form semantics. It
acts in radial momentum, angular momentum, and Killing energy; its mass and
stress-storage components vanish. No instantaneous effective-energy state is
introduced, and cumulative height work remains a path ledger.

The cache schema is version 3. It rejects metadata-only caches, malformed or
missing scientific arrays, a changed secant ladder, and inconsistent tangent,
branch, or top-level decisions. Missing ignored WP10c8i operator caches are
rebuilt only into versioned WP10c8j-owned paths; canonical evidence is never
overwritten.

## Numerical contract

| Block | Steps | Selected step |
|---|---|---:|
| stationary Jacobian and storage matrix | `1e-6`, `2e-6`, `4e-6` | `2e-6` |
| direct storage-rate outer state | `7e-6`, `8e-6`, `9e-6` | `8e-6` |
| vertical storage-rate action | `3.2e-3`, `6.4e-3`, `1.28e-2` | `6.4e-3` |
| nonlinear vector-field secant | `5e-4`, `1e-3`, `3e-3` | `1e-3` |

An explicitly retained pilot showed that `3e-4` was below the reproducible
secant floor at N64, `t=0.10 s`. Independent `5e-4` probes passed the two
controlling pilot directions, so `5e-4` was locked before schema-3 campaign
caches were created. The later failures reported below are not removed by
this calibration: at N64, `t=0.05 s`, the controlling directions also fail at
`1e-3` and `3e-3`.

Binding smooth checks include:

- stationary-Jacobian and complete-storage block stability;
- full-matrix local step stability of the assembled generator, gated at
  `5e-3`;
- fresh production-vector-field centered JVPs, gated at `1e-2`;
- one-sided JVPs, gated at `2e-2`;
- centered-secant step stability, gated at `1e-2`;
- generator factorization below `1e-8`;
- storage component reconstruction below `1e-10`;
- fresh base-rate agreement below `1e-12`.

Raw un-inverted `DM` differences are diagnostic because their fixed
conservation-row scaling is ill-conditioned. The assembled generator and
fresh nonlinear secants are binding. Cached WP10c8i JVPs cannot veto or rescue
the new contract.

## Smooth-tangent results

| Mesh | Time (s) | Smooth result | Binding smooth directions | Rusanov-reserved directions | Controlling result |
|---:|---:|:---:|---:|---:|---|
| N64 | 0 | fail | 0 | 9 | every direction reserved; no nonempty smooth binding set |
| N64 | 0.025 | fail | 0 | 9 | every direction reserved; one consequential cached branch |
| N64 | 0.05 | fail | 6 | 3 | outer thermal/density JVPs fail at all three steps |
| N64 | 0.10 | pass | 6 | 3 | maximum centered defect `7.032e-3` |
| N128 | 0.10 | fail | 6 | 3 | outer-density `5e-4` centered defect `1.020884e-2` |

At N64, `t=0.05 s`, the centered relative-infinity defects are:

| Direction | `5e-4` | `1e-3` | `3e-3` | Gate |
|---|---:|---:|---:|---:|
| density redistribution, `20-200 rg` | `1.2741e-2` | `1.0557e-2` | `1.0649e-2` | `1e-2` |
| thermal redistribution, `60-200 rg` | `2.0712e-2` | `1.8749e-2` | `1.9350e-2` | `1e-2` |

This is the decisive smooth failure. It persists across the resolved secant
plateau and cannot be classified as only a smallest-step cancellation effect.

At N128, `t=0.10 s`, the density redistribution at `5e-4` also gives:

```text
centered relative-infinity defect  1.020883943e-2 > 1e-2
backward relative-infinity defect  2.278124989e-2 > 2e-2
forward relative-infinity defect   1.885828038e-2 < 2e-2
```

Its `1e-3` and `3e-3` probes pass, so this N128 failure is narrow. It is not
used alone: the N64 `t=0.05 s` plateau failure already closes the unchanged
campaign.

The strongest positive result is that the matrix construction itself is now
well behaved. At the matched N64/N128 `t=0.10 s` full scans:

| Gate | Limit | N64 | N128 |
|---|---:|---:|---:|
| generator factorization | `1e-8` | `3.638e-12` | `6.821e-13` |
| selected storage-rate reconstruction | `1e-10` | `2.133e-11` | `2.607e-11` |
| direct-DM outer assembled-generator stability | `5e-3` | `1.633e-3` | `1.593e-4` |
| direct-DM action assembled-generator stability | `5e-3` | `3.515e-4` | `2.381e-4` |

Thus WP10c8j isolates the remaining defect to the independent nonlinear
production-vector-field response, particularly outer thermodynamic
directions, rather than to generator factorization or storage decomposition.

## Rusanov finite-neighborhood result

The helper now requires:

1. exactly one candidate-coverage record for every interior face;
2. certified speed-gap variation throughout a finite state neighborhood;
3. explicit `(face, candidate)` identity for each branch factor;
4. no omitted possible winner unless its sampled effect is bitwise zero or a
   positive bound is included in the remainder;
5. certified nonlinear vector-field and output remainders;
6. trajectory containment within the certified neighborhood.

The production call supplies none of the missing finite-neighborhood inputs:

```text
candidate coverage count       0
certified neighborhood radius  absent
nonlinear remainder            absent
all rows binding               false
```

The N64 `t=0` and `0.025 s` caches contain `12` and `1` consequential cached
branches, respectively. Even anchors with no cached consequential factor are
not certified, because all-face candidate coverage and a uniform nonlinear
remainder are still required. Sampled Fréchet information is not promoted to
a finite-neighborhood theorem.

## Interpretation

WP10c8j materially improves the audit infrastructure and removes the original
nested-derivative ambiguity, but it does not authorize the reduced system.
Specifically:

- no moment set is promoted or rejected using WP10c8i's conditional gains;
- no unchanged WP10c8i repeat is launched;
- no equal-coordinate lift, healing test, microburst, or macrostep is
  authorized;
- the online-cost gate remains unevaluated;
- long-duration evolution and new physics remain blocked.

The next package should not change moments or scientific gates. It should:

1. isolate the outer thermal/density mismatch by storage, central flux,
   fixed-branch Rusanov, source, and boundary blocks;
2. replace the remaining finite-differenced storage descriptor used by the
   independent vector-field evaluator with analytic, automatic-differentiated,
   complex-step-safe, or otherwise independently step-converged derivatives;
3. certify all-face Rusanov candidate/gap coverage, uniform nonlinear
   remainders, and neighborhood containment;
4. rerun the failed N64 `0/0.025/0.05 s` and N128 `0.10 s` anchors before any
   complete six-anchor campaign;
5. repeat WP10c8i unchanged only if both smooth and branch contracts pass.

## Artifacts

Runtime artifacts remain ignored; hashes make the decisive evidence
reproducible.

| Artifact | SHA-256 |
|---|---|
| `outputs/tables/wp10c8j_partial_n64_t0_final.json` | `2d2bb331a5ef5c06079451379ec6947b8ff38baab160ef83d0eb9e9eb444f5ef` |
| `outputs/tables/wp10c8j_partial_n64_t0_final_arrays.npz` | `ff99d30dbaaf04c1b2c4d8f39bdfb8d0d7a93eb2afb7abd0d0628a3b42e65d8e` |
| `outputs/tables/wp10c8j_partial_n64_base_completed.json` | `98622b54a15bc2d8f2aa0cc43337e12c2ecd0371c6efa3de1ebeed9d8e1b6fc1` |
| `outputs/tables/wp10c8j_partial_n64_base_completed_arrays.npz` | `dc09e740cff2356f40d41f672ceea08b9c0293fcc534e2b60241cd97f8c70e11` |
| `outputs/tables/wp10c8j_partial_n64_t010_v4.json` | `fb9b53286e2c8ed21b014697d13da7ddaf1911b69b5dcb834ec06cc314f9032f` |
| `outputs/tables/wp10c8j_partial_n64_t010_v4_arrays.npz` | `a02a746751e3a9acc0c0d14d43df874418d33412f3835f6a0404b3c063f3cac5` |
| `outputs/tables/wp10c8j_partial_n128_t010_final.json` | `46465c9e730f4c4d8c1c6bc97f74362436e0e7200d3c2346d121b8731c287a52` |
| `outputs/tables/wp10c8j_partial_n128_t010_final_arrays.npz` | `6ffaec62ac09b8332c4fec2b1452801db4a179719dea936b5077451487068ead` |

Primary runner:
`scripts/run_causal_tangent_certification_wp10c8j.py`

## Validation

- focused WP10c8j and affected storage tests: `70 passed`;
- complete causal test suite: `261 passed`;
- repository-wide suite: `662 passed, 4 subtests passed`;
- changed Python modules and runners compile successfully;
- `git diff --check` passes;
- repository hygiene passes after the documented tracked-tree policy increment
  from `<700` to `<800`; the staged WP10c8j tree contains `709` reviewed
  files and no tracked generated output.
