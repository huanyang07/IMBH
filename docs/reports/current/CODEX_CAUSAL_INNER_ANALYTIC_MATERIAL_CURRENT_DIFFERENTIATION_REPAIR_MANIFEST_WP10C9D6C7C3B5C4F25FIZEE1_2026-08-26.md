# Analytic material-current differentiation repair manifest

Classification: `analytic_material_current_differentiation_repair_manifest_frozen`.

The parent `entropy_complete_projected_strong_hyperbolicity_failed` result remains binding. It is not converted into a pass.

The saved complex split is not stable under stencil refinement. The three material fluxes are exact products `F=v_transport*U`, but the rejected implementation differentiated `U` and `F` independently. This package prospectively replaces only those three flux derivatives by the analytic identity `d(vU)=v dU+U dv` using the same centered stencil.

No eigenvalue is clipped or projected, no matrix is symmetrized, and no tolerance is changed. The first execution is restricted to the saved held-out point at factors 2, 1, and 0.5 and advances no trajectory.

Authorized next: `WP10c9d6c7c3b5c4f25fizee2_saved_advective_degeneracy_repair_certificate` only. A full-envelope retry remains unauthorized until that saved-point certificate passes.
