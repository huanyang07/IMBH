# Fixed-Q Same-History Equivalence Policy Manifest WP10c9d6c7c3b5c4f24e14o

## Classification

`same_history_equivalence_policy_manifest_frozen_nonpropagating_certificate_authorized`

This definitions-only package preserves the binding
`bounded_continuation_failed` classification from WP10c9d6c7c3b5c4f24e14l.
It authorizes only a saved-endpoint, nonpropagating certificate of a tighter
same-history control-equivalence policy.

## Frozen policy

Production fixed-`Q` step acceptance is unchanged:

```text
maximum scaled production residual       1e-10
```

For a same-history control comparison only, an already accepted control root
must reach

```text
maximum scaled comparison residual       1e-12
maximum scaled state difference           1e-8
maximum reaction-action relative defect   1e-8
```

If the accepted control root is not yet accurate enough for that comparison,
the policy permits at most one exact complete bordered assembly and one
endpoint correction. The frozen relative line-search factors are
`1, 1/2, ..., 1/128`.

## Safety boundary

The polished control:

- cannot define BDF history;
- cannot define a continuation state;
- cannot advance trajectory time;
- cannot change the production `1e-10` root gate;
- cannot retroactively pass WP10c9d6c7c3b5c4f24e14l.

Multiplier-coordinate equality remains nonbinding. Physical reaction-action
equality is binding because it is the basis-invariant quantity.

## Locked evidence

The manifest checksum-locks the positive WP10c9d6c7c3b5c4f24e14n diagnosis,
the historical WP10c9d6c7c3b5c4f24e14l rejection, the common start
checkpoint, both saved endpoint arrays, and every source and focused test used
by the certificate.

The manifest is frozen from definition commit `9e9848a` with single-threaded
BLAS/OpenMP provenance.

## Next action

Execute only the saved-endpoint nonpropagating policy certificate. A pass may
authorize only a definitions-only primary-evidence aggregation manifest. It
does not authorize another continuation run, held-out work, an operational
timestep search, a physical microburst, fast averaging, or reduced evolution.
