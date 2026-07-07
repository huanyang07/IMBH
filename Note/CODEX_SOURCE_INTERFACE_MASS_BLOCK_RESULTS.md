# Source-Interface Mass Block Results

Date: 2026-07-06

## Purpose

This sprint implemented the first part of the true source-plus-buffer
formulation suggested in `CODEX_SOURCE_ELEMENT_CONSISTENCY_AUDIT_RESULTS.md`:

```text
duplicated local source-buffer states
explicit source-block interface compatibility rows
per-interval DeltaM variables
finite-volume mass integral and jump rows
```

The goal was deliberately limited.  I did not yet promote finite-volume energy
or angular momentum to production rows.  This pass asks whether mass/interface
compatibility alone can remove the compact source-annulus defect.

## Code Changes

Updated:

```text
scripts/run_mdot5_local_mdot_eta_continuation.py
```

Added opt-in source-interface controls:

```text
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_INTERFACE_CORRECT
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_INTERFACE_HALO_INTERVALS
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_INTERFACE_MAX_NFEV
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_INTERFACE_WRITE_EDGES
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_INTERFACE_HS_STATE_ROWS
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_INTERFACE_POLY_STATE_ROWS
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_INTERFACE_MASS_QUADRATURE
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_INTERFACE_EDGE_STATE_WEIGHT
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_INTERFACE_EDGE_MDOT_WEIGHT
```

Default behavior remains unchanged because the new corrector is off unless
`SOURCE_INTERFACE_CORRECT=1`.

### Local variables

For a selected source-plus-buffer block, the local least-squares problem uses:

```text
logu_i, logT_i, logMdot_i for every local block node
DeltaM_e for every local source-buffer interval
```

The local block states are duplicated relative to the global vector.  By
default, only interior nodes are written back:

```text
SOURCE_INTERFACE_WRITE_EDGES=0
```

This mimics explicit source-block interfaces inside the current global-vector
architecture.

### Rows

Mass/interface-only rows are now the default for this corrector:

```text
DeltaM_e - int(Mwind_prime - Mstream_prime) dlnR = 0
Mdot_R - Mdot_L - DeltaM_e = 0
interface logu, logT, logMdot compatibility at both block edges
```

The differential state rows are staged behind toggles:

```text
SOURCE_INTERFACE_HS_STATE_ROWS=1
SOURCE_INTERFACE_POLY_STATE_ROWS=1
```

This lets us test mass/interface mechanics separately before coupling the
harder state/energy representation.

## Runs

### 1. Weak source-only, state rows enabled

Output:

```text
outputs/tables/m5_source_interface_weak_source_fs0010_N164_seed.json
```

Setup:

```text
Mdot_inner/Edd = 5
Rout = 400 rg
compact_c2 source at R = 240 rg
f_s = 0.01
wind off
heating off
torque off
SOURCE_INTERFACE_HS_STATE_ROWS=1
SOURCE_INTERFACE_POLY_STATE_ROWS=1
```

Result:

```text
source_interface_applied = false
selected                 = 4.055e-2
state                    = 4.055e-2
FV mass                  = 3.113e-5
interface                = 0
global final_full        = 9.483e-5
```

Interpretation:

The weak-source checkpoint has good mass rows, but the stricter
Hermite/polynomial state rows see an O(4e-2) defect.  This confirms the
state/energy representation is stricter than the existing midpoint production
scheme.

### 2. Weak source-only, mass/interface only

Output:

```text
outputs/tables/m5_source_interface_massonly_weak_source_fs0010_N164_seed.json
```

Result:

```text
source_interface_applied = true
alpha                    = 1.953e-3
selected                 = 3.113e-5 -> 3.107e-5
FV mass                  = 3.113e-5 -> 3.107e-5
interface                = 0 -> 2.94e-11
global final_full        = 9.483e-5
source extra             = 7.638e-4 -> 7.621e-4
```

Interpretation:

The mass/interface machinery behaves sensibly at weak source amplitude.  A full
local mass correction would reduce FV mass to `1.5e-7`, but it worsens global
mass/source-band rows, so the hierarchical guard only accepts a tiny step.

### 3. eta_E = 100 compact source checkpoint, mass/interface only

Output:

```text
outputs/tables/m5_source_interface_massonly_eta100_N164_seed.json
```

Setup:

```text
Mdot_inner/Edd = 5
Rout = 335 rg
Rinj = 240 rg
f_s = 0.80
eta_E = 100
compact source
torque_delta_l_fraction = 0.005
```

Result:

```text
production final_full = 9.056e-6
source_interface_applied = false
selected/FV mass          = 1.208e-2
source extra              = 1.594e-1
source_element poly_E     = 1.341e-1
source_element FV_E       = 2.178e-2
```

The rejected full local step would reduce the local interface FV-mass row to
`3.5e-4`, but it would produce:

```text
global full      ~ 5.59e2
source extra     ~ 1.32e3
```

Interpretation:

The eta_E=100 hidden defect is not fixed by mass/interface compatibility alone.
Trying to repair mass locally without coupled state/energy compatibility moves
the solution into a completely incompatible basin.

### 4. eta_E = 90 compact source checkpoint, mass/interface only

Output:

```text
outputs/tables/m5_source_interface_massonly_eta90_N201_seed_fvmass.json
```

Important bookkeeping note:

The eta_E=90 checkpoint must be evaluated with

```text
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_BAND_FINITE_VOLUME_MASS=1
```

to reproduce the previously reported `final_full ~ 3.93e-2`.  Without that,
the old differential mass row gives `final_full ~ 7.28`.

Result with FV source-band mass enabled:

```text
production final_full     = 3.929e-2
source_interface_applied  = false
selected/FV mass          = 2.363e-2
source extra              = 3.088e-2
source_element poly_E     = 2.554e-1
source_element FV_E       = 3.801e-2
```

The rejected full local mass step would reduce local FV mass to `2.97e-4`, but
would produce:

```text
global full      ~ 4.35e2
source extra     ~ 6.81e2
```

Interpretation:

This is the same pattern as eta_E=100.  The mass/interface-only correction
direction is not compatible with the state/energy/source-band residuals.

## Verification

```text
PYTHONPYCACHEPREFIX=/tmp/imbh_pycache python -m py_compile scripts/run_mdot5_local_mdot_eta_continuation.py
PYTHONPATH=src PYTHONPYCACHEPREFIX=/tmp/imbh_pycache python -m pytest tests/test_transonic_local.py tests/test_transonic_collocation.py -q
```

Result:

```text
73 passed, 2 subtests passed
```

## Findings

1. The new source-interface mass block is numerically wired correctly and
   behaves on weak source-only tests.
2. At high source fraction, the local FV mass direction exists but is not a
   physically/global-compatible correction direction.
3. The compact source-annulus defect is therefore not a pure mass-interface
   problem.
4. The state/energy representation must be solved simultaneously with mass and
   interface compatibility.
5. The eta_E=90 residual bookkeeping depends on the finite-volume source-band
   mass row; this should be made explicit in future reports.

## Next Plan

The next implementation should not be another mass-only correction.  It should
promote the source-interface block into a coupled mixed formulation:

1. Use the duplicated source-interface local states already implemented here.
2. Enable state rows in a staged way, but make them element-local and
   self-consistent rather than merely linear global-stencil rows.
3. Add a differential/FV energy identity audit inside the same local block:
   compare integrated polynomial energy residual against the FV energy
   numerator before using FV energy as a hard row.
4. Add local energy rows with a homotopy:
   `gamma_E = 0.03, 0.1, 0.3, 1`.
5. Only after mass, state, and energy can improve together should angular
   momentum flux rows be added.
6. Do not lower `eta_E` below 90 until the source-interface block reduces
   `poly_E`, `FV_E`, `FV_M`, source extra, and production full together.
