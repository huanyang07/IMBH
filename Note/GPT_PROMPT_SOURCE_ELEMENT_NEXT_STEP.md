# GPT Prompt: Source-Annulus Certification Bottleneck

Please review the latest GitHub state of the IMBH_QPE project, especially:

- `Note/CODEX_MDOT5_SOURCE_MICRODOMAIN_RESULTS.md`
- `scripts/run_mdot5_local_mdot_eta_continuation.py`
- latest output families:
  - `outputs/tables/m5_local_mdot_eta90_source_buffer_*`
  - `outputs/tables/m5_local_mdot_eta90_source_element_refine2_*`
  - matching checkpoints under `outputs/checkpoints/`

Current physical/numerical target:

- Mdot_inner/Edd = 5
- Rout = 335 rg
- Rinj = 240 rg
- f_s = 0.80
- eta_E = 90
- compact source annulus
- local-Mdot, mass-loaded wind formulation

Current accepted diagnosis:

- The eta_E=90 branch is not source-band-collocation certified.
- The old N201 halo-4 source-domain checkpoint has:
  - production residual ~3.929e-2
  - source-band audit ~3.088e-2
  - interval mass residual ~3.929e-2
- Source-plus-buffer integrated DeltaM variables were implemented.
  They slightly improve the residual but do not remove the wall.
- Real source-element internal nodes were implemented.
  The N201 grid can be refined to N251 by splitting source-plus-buffer intervals.
- Raw N251 refinement creates an artificial source-edge mass spike near R~221 rg.
  A mass-centered local block repair removes that artificial spike.
- Sparse global polishing now works again after fixing the audit-only sparse
  Jacobian bookkeeping.

Best current N251 result:

- run: `m5_local_mdot_eta90_source_element_refine2_global_domain2_eta90`
- production residual: ~3.752e-2
- interval mass residual: ~3.752e-2
- source-band audit: ~3.612e-2
- radial source-band audit: ~3.288e-2
- energy source-band audit: ~3.612e-2

Comparison:

- N251 slightly improves production residual relative to old N201:
  - 3.929e-2 -> 3.752e-2
- but N251 worsens source-band audit:
  - 3.088e-2 -> 3.612e-2

Therefore this is not yet a certified improvement.

Important implementation details:

- Source-band extra rows can currently be audit-only.
- Production residual uses finite-volume mass rows in the source band.
- Source-domain corrector samples differential residuals at 0.25, 0.5, 0.75.
- Source-buffer corrector introduces auxiliary interval DeltaM variables:
  - DeltaM_i - integral_i(Mwind_prime - Mstream_prime) = 0
  - Mdot_{i+1} - Mdot_i - DeltaM_i = 0
- The remaining defect is still a coupled mass/energy/source-edge compatibility
  wall, not simply lack of subcell nodes or mass bookkeeping.

Please suggest the next concrete numerical formulation to certify or reject the
eta_E=90 source-annulus branch.

Questions to answer:

1. Should we make the source-band extra rows production rows with sparse/local
   Jacobian, rather than audit-only rows?
2. If yes, what row weighting or penalty-continuation strategy should be used
   so the solver does not simply trade production mass residual against hidden
   source-band energy residual?
3. Should the source annulus be converted to a rectangular/least-squares
   production problem, or should we keep a square formulation by adding true
   internal state/flux variables?
4. What exact unknowns and residual rows would you add next?
5. Should finite-volume energy and angular-momentum balance be introduced in
   the source annulus before trying to lower eta_E below 90?
6. What acceptance criteria should be used before continuing eta_E to 80, 70,
   or 60?

Please give an implementation-level plan that fits the current code structure.
Do not suggest adding wind complexity or lowering eta_E until the eta_E=90
source-annulus representation is certified.
