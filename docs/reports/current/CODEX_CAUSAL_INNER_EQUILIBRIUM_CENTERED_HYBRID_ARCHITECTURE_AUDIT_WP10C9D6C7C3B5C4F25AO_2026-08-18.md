# Equilibrium-centered conservative slow-fast hybrid architecture audit WP10c9d6c7c3b5c4f25ao

## Classification

`equilibrium_centered_conservative_slow_fast_hybrid_architecture_certified_offline_branch_seed_manifest_authorized`

## Selected mathematical architecture

Let `y=(c96, eta66)` be 162 resolved physical observables and let the 398-column basis `H(x)` span `ker D C_phys(x)`. At fixed `y` and branch label, the offline branch state solves

`C_phys(x) - y = 0`,

`H(x)^T W(x) F_Q(x) = 0`.

This is a square 560-equation problem. It does not force the full rate to zero: the branch may move along the resolved slow coordinates. Normal hyperbolicity is evaluated only from the hidden Jacobian at a converged conditional branch root.

Online, the 96 M/J/E coordinates evolve through a conservative finite-volume flux divergence. The remaining 66 resolved coordinates and at most 280 memory states use an equilibrium-centered dissipative descriptor update. A discrete branch label replaces linear macro-propagation of the 28 positive-growth directions.

Fast switches are offline intrinsic fixed-Q orthogonal-collocation boundary-value solves. Their reset maps integrate the conservative flux/source impulse and preserve global Q3 in the absence of an external impulse.

## Structural results

- Both inherited resolved coordinate maps have rank 162. The hidden dimension is 398 and the online continuous upper bound is 442.
- The finite-volume global telescoping defect is `9.434657e-16`.
- The maximum actual minimum-norm reset constraint defect is `1.306077e-15`.
- The largest 5.7888 s stable-descriptor energy amplification factor across primary/midpoint/held-out is `6.858775e-04`.
- A deliberately dense per-step algebra witness plus recomputed stable exponential projects to `1591.758383` wall seconds (`1.842313e-02` days) for 100,000 macrosteps.

The runtime result establishes feasibility of the online algebraic architecture, not the cost of building its offline database.

## Claim boundary and next gate

Authorized next artifact: `definitions_only_first_conditional_fast_branch_seed_manifest`.

No physical conditional branch or transition has yet been found. The old 442-state coefficients remain a transfer/stability/cost witness and are not promoted to an equilibrium-centered physical closure. No predictive cycle or reduced slow evolution is authorized.
