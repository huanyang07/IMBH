# Nonlinear 5 ms inner-face/half-cell audit WP10c9d6c7c3b5c3h2h1

## Classification

`five_ms_inner_export_recovers_on_common_surface_extraction_surface_manifest_authorized`

No state was propagated and no operator was changed. Seven committed late-history targets were evaluated on ten nested common physical faces. The final accepted BDF tangent was decomposed into its exact temporal-storage, shared-face, principal, and lower-source actions.

## Result

- Compact recovery is selected at coarse face `1`, the shared physical
  surface `R=1.87501653 r_g`.  Its full mass/angular-momentum/Killing-energy
  response has RMS/max/min-component orders `0.93648/1.20276/0.91219`,
  refinement-error cosine `0.97535`, and fine fixed-scale difference
  `6.02e-8`.  The next common face also passes, satisfying the prospectively
  frozen two-consecutive-face rule.
- Every one of the ten nested common faces from `1.87501653 r_g` through the
  coupling surface passes.  Orders approach second order away from excision;
  the coupling-face response is below observability and passes only by the
  frozen upper-bound route.
- The analytic final accepted-BDF component closures and linear solves are
  `<=5.28e-17` and `<=2.87e-16`, respectively, with zero incoming excision
  characteristics.
- No near-inner temporal, principal, or lower-source block reaches the frozen
  `0.70` dominance threshold on both grid pairs.  The first common-face flux
  is the largest compensator but contributes only `0.54/0.63` of the mass
  absolute balance on the two pairs.
- The primitive-column path decomposition is nonbinding: its maximum closure
  defect is `9.36e-9`, above the frozen `1e-9` gate.  No primitive field or
  half-cell correction is selected.

The result therefore does not diagnose an incorrect outgoing excision
condition.  It shows that the nonconvergent raw face value is confined inside
the first common control volume.  The conservative horizon-equivalent
exchange can be evaluated at the recovery surface by including the exact
interior temporal-storage and source ledger:

`F_inner = F_recovery + temporal_storage_prefix + source_prefix`.

## Decision

Only `WP10c9d6c7c3b5c3h2i_conservative_extraction_surface_manifest` is
authorized.  It must freeze the `1.87501653 r_g` surface prospectively and
certify the complete horizon-equivalent M/J/E ledger, not merely replace one
reported flux field. The rejected 5 ms certificate, fourth duration rung,
fixed-Q experiments, and reduced slow evolution remain blocked.
