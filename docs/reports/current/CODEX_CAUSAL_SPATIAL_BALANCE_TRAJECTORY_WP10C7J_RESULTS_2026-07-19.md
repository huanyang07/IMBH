# Causal Spatial Balance Trajectory WP10c7j Results

Date: 2026-07-19

## Verdict

WP10c7j passes its bounded N32/N64 spatial trajectory contract.

The general high-order operator certified by WP10c7i was applied without a
baseline-specific correction:

```text
spatial reconstruction                          quadratic_admissible
physical boundary trace                         plm_one_sided
cell rate scheme                                arithmetic_face
cell source quadrature                          gauss_legendre_4_local_rates
cell storage quadrature                         gauss_legendre_4
```

All N32/N64 S32/S64 fixed-BDF2 campaigns reach the exact
`1.537457597966907e-2 s` target. The endpoint N32/N64
`Delta log(H/R)` difference is:

```text
raw spatial difference                          0.00152768674
WP10c7i tangent projection                       0.00152798791
measured/projected ratio                         0.999802900
N32 temporal uncertainty                         0.000141547783
N64 temporal uncertainty                         0.000147558736
spatial plus both temporal uncertainties         0.00181679326
locked spatial gate                              0.005
conservative gate utilization                    0.363359
```

The old smooth-PLM WP10c7h endpoint difference was `0.0446191`. The selected
operator reduces that bounded trajectory discrepancy by `29.207x`.

This is a numerical certification over a short bounded horizon. It is not a
disk-relaxation, thermal-state, stability, hot-branch, or cycle result.

## Fresh Initial States

WP10c7j does not reuse a smooth-PLM checkpoint. It samples the exact
WP10c7i source-compatible continuum independently on N32 and N64, rebuilds
the selected primitive/storage/face maps, computes a fresh DAE-consistent
tangent predictor, and takes one BDF1 startup step in each campaign.

```text
N32 initial SHA256
71eb92170b2da456b8ec83060b657d467526f12cf311e6d6de999dbba1ba21e9

N64 initial SHA256
783eb300a1db51d6d23481a0780fd231db4747f47575bba31e68dc730414abce

N32 inner-throughput/stream ratio                 0.9999999999999981
N64 inner-throughput/stream ratio                 0.9999999999999980
source restriction defect                         1.72942e-16
```

All initial state gates pass. The maximum scaled consistency defects are
`4.44e-15` at N32 and `9.99e-15` at N64. No initial, intermediate snapshot,
or final state activates admissibility rescaling.

## Fixed BDF2 Campaigns

Every campaign uses one BDF1 startup step followed by equal-step BDF2.
Modified Newton reuses one colored finite-difference Jacobian within each
step. No rejected attempt is permitted.

| Mesh/rung | Steps | BDF1/BDF2 | Max residual | Physical ledger | Jacobians | Function evaluations |
|---|---:|---:|---:|---:|---:|---:|
| N32 S32 | `32` | `1/31` | `7.896e-12` | `1.995e-4` | `32` | `1664` |
| N32 S64 | `64` | `1/63` | `9.220e-12` | `4.994e-5` | `64` | `3328` |
| N64 S32 | `32` | `1/31` | `9.576e-12` | `2.014e-4` | `32` | `1664` |
| N64 S64 | `64` | `1/63` | `9.302e-12` | `5.042e-5` | `64` | `3330` |

All 192 steps pass:

- nonlinear residual;
- algebraic map;
- discrete BDF ledger;
- cumulative physical ledger;
- causal characteristic;
- optical-depth;
- Roche-channel;
- state-change;
- positivity and thickness gates.

Every final checkpoint and snapshot sidecar reloads bitwise under the full
five-option spatial provenance.

## Common-Time Contract

Full states are saved at `T/8`, `T/4`, `T/2`, and `T`. At every time, N64
cell responses are restricted conservatively onto the exact nested N32
Kerr-Schild control volumes.

| Time | Raw spatial | N32 temporal | N64 temporal | Conservative total |
|---|---:|---:|---:|---:|
| `T/8` | `1.90980e-4` | `1.55455e-4` | `1.63328e-4` | `5.09763e-4` |
| `T/4` | `3.81995e-4` | `1.55411e-4` | `1.63082e-4` | `7.00487e-4` |
| `T/2` | `7.64011e-4` | `1.50454e-4` | `1.57509e-4` | `1.07197e-3` |
| `T` | `1.52769e-3` | `1.41548e-4` | `1.47559e-4` | `1.81679e-3` |

The maximum temporal uncertainty is `1.63328e-4`, below the locked
`2.5e-4` target. Both the raw and conservative spatial errors pass `0.005`
at every snapshot.

The spatial response grows nearly linearly over this bounded interval. There
is no earlier mesh disagreement hidden by endpoint cancellation.

## Endpoint Profiles

At S64, the endpoint response differences are:

| Response | Full maximum | Peak radius | `15-60 rg` maximum |
|---|---:|---:|---:|
| `Delta log(H/R)` | `0.00152769` | `16.3242 rg` | `0.00152769` |
| `Delta log T` | `0.000358629` | `16.3242 rg` | `0.000358629` |
| `Delta log Pi` | `0.00296027` | `16.3242 rg` | `0.00296027` |
| `Delta log e` | `0.00305634` | `16.3242 rg` | `0.00305634` |
| `Delta log Sigma` | `0.000166786` | `5.20379 rg` | `0.000101512` |
| `Delta beta_R` | `0.000117175` | `1.95316 rg` | `0.0000356757` |
| `Delta specific stress` | `5.86952e-7` | `1.95316 rg` | `2.12874e-9` |

The former first-cell thermodynamic boundary maximum is absent. The
controlling thickness, temperature, pressure, and energy differences
coincide at `16.3242 rg`.

The stored term-response audit shows that the remaining mesh difference is
transport dominated. The prescribed stream response remains exact, the
flux-primary closure is negligible, and vertical/cooling source differences
are far below the transport contribution. The compact arrays retain every
five-field term profile at all four common times.

## Interpretation

WP10c7j validates the central WP10c7i inference:

1. smooth PLM removed much of the old face-transport error but left
   inconsistent physical-boundary, storage, and local-source treatment;
2. quadratic admissible face traces remove the boundary-order defect;
3. measure-weighted storage and four-point source integration must be used
   together;
4. shear and `d log(H)/d tau` must follow the same reconstructed path as the
   thermodynamic source state;
5. a constant reference-state residual correction is neither required nor
   retained.

The measured endpoint agrees with the initial tangent projection to
`1.97e-4` relative. That agreement, combined with the monotone snapshot
sequence, shows that the bounded trajectory is still in the locally linear
response regime.

## Evidence

Runtime artifacts remain ignored by repository policy.

```text
outputs/tables/causal_spatial_balance_trajectory_wp10c7j.json
SHA256 d6b8f9062eb1a32cf7d1b2d508ae04b70685fd64602d4e50972170f0cc4d04ab

outputs/tables/causal_spatial_balance_trajectory_wp10c7j_arrays.npz
SHA256 91d722c5c4a398f3e19b6a4e20c0915d8cc87cf4360f4ad8e7fe928ee7db90d2
```

Fixed checkpoint hashes:

```text
N32 S32  7e3386a4c330a9e03a7b3a2d6b9f81a3da2d1b2ce148fa6285dd7ac271463575
N32 S64  c3a5e443f5b9af50a991d7507f1c1a0e08cfe2136b62e376854246963855f
N64 S32  dfcbc339a8cc96279f62a77f2315d53f4621c278cab034e89156e03705468049
N64 S64  88a4bcf33e8a37a1935e3ff161eefcfa76455abb87d06576446b06df63ddac6d
```

Snapshot-sidecar hashes:

```text
N32 S32  1b39f9a6e69e3ef899dae2b09dd0920696e91017c8c96cbf8ee572246963855f
N32 S64  e8912836b70b57364b5bc470db47e050587f4316cdf2fdd5f4cf783289f934d1
N64 S32  765fcb76d29051fa521de824f73fe46525447f5be5eecfde38488289afc4610e
N64 S64  69ec98550139b7ca483c047420ef3921aae9c8f0184388816a7c12853b320e7a
```

## Verification

```text
WP10c7j preflight                             passed
N32/N64 S32/S64 fixed trajectories           passed
all checkpoint and snapshot reloads          bitwise
focused BDF evolution tests                  7 passed
focused causal evolution/DAE/spatial tests  60 passed
complete repository suite                    550 passed
complete repository subtests                 4 passed
Python byte compilation                      passed
git diff whitespace check                    passed
```

## Authorization

WP10c7j authorizes one matched adaptive-BDF2 confirmation package:

1. use the identical selected operator and fresh source-compatible N32/N64
   states;
2. reach the exact WP10c7j target on both meshes;
3. compare adaptive endpoints and common snapshots with the fixed-S64
   references, retaining the S32/S64 reference uncertainties;
4. preserve the conservative `0.005` N32/N64 spatial budget;
5. preserve all source, state, nonlinear, ledger, restart, and limiter
   diagnostics;
6. report Jacobian and function-evaluation work relative to fixed S64.

Only after matched adaptive confirmation may no-tide duration extend
geometrically toward the characteristic, stress-relaxation, cooling, and
thermal clocks. N128, tide, wind, stability, hot-state, and cycle work remain
closed.
