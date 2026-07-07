# Source-Interface Energy Audit and DeltaE Rows

Date: 2026-07-06

## Purpose

This note follows `CODEX_SOURCE_INTERFACE_MASS_BLOCK_RESULTS.md`.

The previous source-interface mass block showed that mass/interface
compatibility alone cannot repair the compact source-annulus defect.  This
pass adds the next staged piece:

```text
source-interface FV energy audit
optional per-interval DeltaE variables
optional FV energy integral/balance rows
```

The goal is still diagnostic.  The new energy rows are disabled by default.

## Code Changes

Updated:

```text
scripts/run_mdot5_local_mdot_eta_continuation.py
```

Added opt-in controls:

```text
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_INTERFACE_ENERGY_AUDIT
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_INTERFACE_FV_ENERGY_ROWS
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_INTERFACE_ENERGY_WEIGHT
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_INTERFACE_ENERGY_INTEGRAL_WEIGHT
IMBH_MDOT5_LOCAL_MDOT_ETA_SOURCE_INTERFACE_ENERGY_BALANCE_WEIGHT
```

When FV energy rows are enabled, each source-interface interval receives a
local `DeltaE_e` variable.  The rows are:

```text
(DeltaE_e - integral 2*pi*R^2*Qnet dlnR) / energy_denominator = 0
DeltaE_e / energy_denominator = 0
```

Here:

```text
Qnet = Qvisc + Qstream - Qrad - Qadv - Qwind
```

This is intentionally conservative.  Since we do not yet have a trusted
advective/enthalpy energy-flux difference for the source block, the second row
is the local steady-balance condition.

## Runs

### 1. eta_E = 100 compact source, energy audit only

Output:

```text
outputs/tables/m5_source_interface_energy_audit_eta100_N164_seed.json
```

Result:

```text
production final_full                  = 9.056e-6
source-interface FV_E max              = 1.023e-3
source-interface scaled diff integral  = 4.260e-5
source-element poly_E max              = 1.341e-1
source-element FV_E max                = 2.178e-2
source-element FV_M max                = 1.207e-2
```

Interpretation:

The source-interface FV energy audit is much smaller than the older
global-stencil source-element energy audit.  This means the large `poly_E`
defect is not reproduced by the current Hermite/source-interface FV energy
normalization.  The source-interface audit and old source-element audit are not
yet equivalent representations.

### 2. eta_E = 100 compact source, FV energy rows enabled

Output:

```text
outputs/tables/m5_source_interface_energy_rows_eta100_N164_gamma1_seed.json
```

Result:

```text
source_interface_applied       = false
initial FV energy row          = 1.023e-3
candidate FV energy row        = 6.028e-5
final FV energy row            = 1.023e-3
production final_full          = 9.056e-6
source extra                   = 1.594e-1
```

The local candidate can reduce the new FV energy row, but it is rejected
because it damages the global/source-band residuals:

```text
alpha=1 trial:
    FV energy row      = 6.028e-5
    FV mass row        = 3.413e-4
    global full        = 2.335e1
    source extra       = 2.557e1
```

Interpretation:

FV energy is not independently repairable.  As with mass, a local correction
exists but is not globally compatible unless state/energy/source-band
compatibility are solved together.

### 3. eta_E = 100 compact source, Hermite-Simpson state + FV energy rows

Output:

```text
outputs/tables/m5_source_interface_hs_energy_rows_eta100_N164_seed.json
```

Result:

```text
source_interface_applied = false
initial selected         = 7.068e2
initial state row        = 7.068e2
initial FV energy        = 1.023e-3
production final_full    = 9.056e-6
```

Interpretation:

The current Hermite-Simpson state row is not usable as a production row with
this scaling/formulation.  It reports an enormous state defect even on the
production-strict eta_E=100 checkpoint.  This is a formulation/normalization
issue, not evidence that the physical branch is bad.

### 4. eta_E = 90 compact source, energy audit only

Output:

```text
outputs/tables/m5_source_interface_energy_audit_eta90_N201_seed_fvmass.json
```

Bookkeeping:

```text
SOURCE_BAND_FINITE_VOLUME_MASS=1
SOURCE_BAND_EXTRA_ROWS=1
SOURCE_BAND_EXTRA_AUDIT_ONLY=1
```

Result:

```text
production final_full                  = 3.929e-2
source-interface FV_E max              = 1.130e-3
source-interface scaled diff integral  = 2.895e-5
source-element poly_E max              = 2.554e-1
source-element FV_E max                = 3.801e-2
source-element FV_M max                = 2.363e-2
```

Interpretation:

The same mismatch exists at eta_E=90: the current source-interface FV energy
audit is small compared with the older source-element polynomial/FV energy
audit.  Therefore the next step should be to reconcile representations, not to
keep adding stronger FV energy rows.

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

1. The new `DeltaE` machinery is wired and can reduce its own local FV energy
   rows, but accepted global-compatible improvement was not found.
2. The source-interface FV energy audit reports `~1e-3`, much smaller than the
   old source-element energy audit (`~2e-2` to `~2.5e-1` depending row type).
3. The current Hermite-Simpson state row is badly scaled or otherwise not
   equivalent to the accepted production residual; it should not be promoted to
   production yet.
4. The real next bottleneck is representation consistency:
   the source-interface FV/Hermite view and the global-stencil polynomial audit
   are not measuring the same defect.

## Next Plan

Before adding angular momentum or lowering eta_E:

1. Reconcile the source-interface and source-element energy representations.
   Use the same interpolation basis, quadrature points, and normalization in
   both audits.
2. Replace the current HS state row with a scaled differential row that matches
   `_scaled_residual_at` at the same collocation points used by the
   source-element audit.
3. Add a source-interface polynomial row mode using the same 5-node or
   element-local polynomial basis as the existing source-element audit.
4. Only after the two audits agree should FV energy rows be used as production
   constraints.
5. Then rerun eta_E=100 and eta_E=90 with a staged source-interface solve:
   mass rows, matched polynomial state/energy rows, then FV energy rows.
