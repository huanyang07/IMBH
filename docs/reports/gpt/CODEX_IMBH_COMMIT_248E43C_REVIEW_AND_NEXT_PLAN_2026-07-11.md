# Review of Commit `248e43c` and Recommended Next Plan

**Project:** IMBH/QPE stream-fed minidisk
**Repository:** `huanyang07/IMBH`
**Commit reviewed:** `248e43cacb64cc9f2a06be0c15309d7b6767d649`
**Commit title:** `Add signed-flux total-energy closure`
**Review date:** 2026-07-11, Asia/Tokyo
**Review type:** Static review of the checked-in equations, report, implementation, tests, and canonical summaries. The production calculations and full test suite were not independently rerun by the reviewer.

---

# 1. Executive verdict

Commit `248e43c` is an important advance, but it should be treated as a **total-energy prototype pending one algebraic identity correction**, not yet as the final energy closure to couple to the inner transonic disk.

The commit does several things well:

- it uses the angularly closed signed-flux reservoir from the parent milestone;
- it introduces one stream state carrying mass, angular momentum, and total energy;
- it defines a column Bernoulli quantity;
- it carries viscous torque work once through `-Omega G`;
- it includes radiative losses and named external power;
- it rejects the fixed-Keplerian reservoir at `Rin=6.1 rg`;
- it preserves the rejection witness and the `Rin=10 rg` numerical controls;
- it correctly recognizes that a physical inner match is mandatory;
- and it reports 234 passing tests plus four subtests.

The main numerical conclusions are credible for the equations as implemented:

- the fixed-transport energy rows can be solved accurately;
- the near-ISCO alpha-viscosity iteration fails in the invalid region;
- moving the numerical edge to `10 rg` removes that immediate iteration failure;
- the total-energy closure lowers the wall solution relative to the earlier internal-energy result;
- and the wall remains warmer/thicker than the open solution.

However, the current implementation appears to mix:

1. an **enthalpy-carrying advective flux**, and
2. the **vertical-work correction appropriate to an internal-energy advective flux**.

That pairing must be derived and tested directly before inner transonic coupling.

The recommended order is therefore:

```text
correct/prove total-energy identity
-> implement prescribed inner Mdot/J/F_E boundary data
-> perform overlap audit
-> couple inner and outer domains
-> calibrate tidal torque and power
-> stability/time evolution
-> wind
```

---

# 2. What commit `248e43c` establishes

## 2.1 Angular closure remains a real improvement

The parent signed-flux model now includes the physical stream angular moment in the steady mass/angular solution. The project reports:

```text
open inner fraction      = 0.170064596
open overflow fraction   = 0.829935404
wall torque fraction     = 0.768986584
unnamed angular defect   ~ roundoff
```

This resolves the principal angular-closure objection to commit `53566fa`.

## 2.2 The total-energy prototype is numerically well behaved away from the invalid inner edge

The new implementation defines

\[
B_{\rm col}
=
\Phi+\frac{v_\phi^2}{2}+\frac{v_R^2}{2}
+e+\frac{\Pi}{\Sigma},
\]

and

\[
F_E=\dot M B_{\rm col}-\Omega G.
\]

At fixed transport, the wall and open energy roots close below the reported normalized residual gates.

## 2.3 The near-ISCO rejection is scientifically correct

At `Rin=6.1 rg`, the high-resolution thermoviscous fixed point fails in the first few cells, where:

- \(d\ln l_K/d\ln R\) is small;
- radial pressure support is large;
- and the fixed-Keplerian diffusion representation is invalid.

This is not evidence against the outer/source solution.

The rejection should remain canonical.

## 2.4 `Rin=10 rg` is only a control

At `N=512`, the reported controls are:

```text
wall:
  H/R max             0.28447
  Lrad/LEdd           0.94785
  tau_es min          25.43
  viscosity mismatch  6.60e-5

open:
  H/R max             0.12340
  Lrad/LEdd           0.34705
  tau_es min          21.61
  viscosity mismatch  2.20e-4
```

The wall radial-pressure fractions are reported as:

```text
12 rg: 0.2231
15 rg: 0.0698
20 rg: 0.0016
30 rg: 0.0514
```

The zero-torque boundary creates a sharp artificial layer, so `10 rg` is not a physical disk interface.

## 2.5 The prior hot-state numbers are superseded quantitatively

The near-ISCO total-energy wall control approaches approximately

```text
H/R   ~ 0.2975
Lrad  ~ 1.173 LEdd
```

rather than the earlier internal-energy values

```text
H/R   ~ 0.3413
Lrad  ~ 1.323 LEdd.
```

The large change correctly triggers the project’s stop condition.

The surviving conclusion is only:

> the ideal wall remains warmer and thicker than the open edge under the current prototype equations.

---

# 3. Critical algebraic issue to resolve before coupling

## 3.1 Current implemented pair

The code uses the enthalpy flux

\[
F_E
=
\dot M
\left(
q+e+\frac{\Pi}{\Sigma}
\right)
-\Omega G,
\]

where

\[
q
=
\Phi+\frac{v_\phi^2}{2}+\frac{v_R^2}{2}.
\]

It then adds the cell work term represented continuously as

\[
W_{\rm current}
=
\dot M
\left(
\frac{d\Pi}{\Sigma}
-
\frac{P}{\rho^2}d\rho
\right).
\]

The manufactured test verifies the numerical integration of this expression. It does not prove that this expression is the correction appropriate to the chosen enthalpy flux.

## 3.2 Continuous derivation

Consider a source-free interval with constant \(\dot M\).

The radial momentum equation and angular-momentum conservation give

\[
dq
=
\Omega\,dl-\frac{d\Pi}{\Sigma},
\]

and

\[
dG=\dot M\,dl.
\]

Therefore

\[
dF_E
=
\dot M
\left[
de
-
\frac{\Pi}{\Sigma^2}d\Sigma
\right]
-
G\,d\Omega.
\]

The entropy form used by the transonic equations contains

\[
\dot M
\left[
de-\frac{P}{\rho^2}d\rho
\right]
-
G\,d\Omega.
\]

Consequently, the work correction paired with the **enthalpy flux** is

\[
W_H
=
\dot M
\left[
\frac{\Pi}{\Sigma^2}d\Sigma
-
\frac{P}{\rho^2}d\rho
\right].
\]

Because the one-zone closure has

\[
\frac{\Pi}{\Sigma}=\frac{P}{\rho},
\qquad
\Sigma=2\rho H,
\]

this is equivalently

\[
W_H
=
\dot M\frac{P}{\rho}\,d\ln H.
\]

By contrast,

\[
W_{\rm current}
=
\dot M
\left(
\frac{d\Pi}{\Sigma}
-
\frac{P}{\rho^2}d\rho
\right)
\]

is the correction that closes the entropy equation when the advective energy flux carries **internal energy but not enthalpy**.

The difference is

\[
W_{\rm current}-W_H
=
\dot M\,d\left(\frac{\Pi}{\Sigma}\right).
\]

Thus the current code appears to use the enthalpy flux and then add an extra derivative of the column pressure enthalpy.

## 3.3 Why the existing identity audit does not settle this

The legacy identity audit evaluates a derivative proportional to

\[
\dot M
\left(
de-\frac{d\Pi}{\Sigma}
\right)
-
G\,d\Omega,
\]

then adds `W_current`.

That closes the legacy entropy equation exactly, but it corresponds to the **internal-energy-flux representation**.

It is not a direct derivative of the new checked-in flux

\[
\dot M
\left(
q+e+\frac{\Pi}{\Sigma}
\right)
-\Omega G.
\]

## 3.4 Required decision

Codex should choose one of two mathematically consistent representations.

### Option A — recommended

Keep the standard enthalpy-carrying flux:

\[
F_E
=
\dot M
\left(
q+e+\frac{\Pi}{\Sigma}
\right)
-\Omega G,
\]

and replace the vertical correction by

\[
W_H
=
\dot M
\left[
\frac{\Pi}{\Sigma^2}d\Sigma
-
\frac{P}{\rho^2}d\rho
\right].
\]

### Option B

Use an internal-energy advective flux:

\[
F_E^{(e)}
=
\dot M(q+e)-\Omega G,
\]

and retain

\[
W_{\rm current}
=
\dot M
\left(
\frac{d\Pi}{\Sigma}
-
\frac{P}{\rho^2}d\rho
\right).
\]

Both can be equivalent if sources and boundary work are treated consistently.

Do not keep the present mixed pairing without a derivation that invalidates the algebra above.

---

# 4. Mandatory WP2b validation

Before any inner coupling, add a work package named something like:

```text
WP2b — total-energy/entropy equivalence correction
```

## 4.1 Pointwise identity test

On a certified no-wind transonic profile, calculate independently:

1. the derivative of the chosen total-energy flux;
2. the selected vertical-work term;
3. the radiative loss;
4. the legacy entropy residual.

Require

\[
\left[
\frac{dF_E}{d\ln R}
+
W_H
-
L_{\rm rad}'
\right]
-
\left[
\dot M T\frac{ds}{d\ln R}
-
G\frac{d\Omega}{d\ln R}
-
L_{\rm rad}'
\right]
\]

to converge to zero.

Do not compare two expressions that share the same pre-simplified algebra.

## 4.2 Finite-volume convergence test

Sample the same smooth transonic profile onto:

```text
N = 64, 128, 256, 512
```

and require the discrete energy compatibility error to converge at the expected order.

## 4.3 Source-bearing manufactured test

Use a smooth prescribed \(\dot M(R)\), \(l_s(R)\), and \(B_s(R)\) with an analytic solution.

Verify simultaneously:

```text
mass
angular momentum
total energy
vertical work
source mixing energy
```

This test is needed because source-bearing cells contain \(d\dot M\neq0\).

## 4.4 Recompute all canonical total-energy states

After choosing the consistent identity, rerun:

```text
near-ISCO N256/N512 failure witness
Rin=10 wall/open N64-N512
fixed-transport roots
thermoviscous fixed points
```

If `H/R`, luminosity, or interface diagnostics shift materially, replace the `248e43c` canonical controls rather than preserving both as equally physical.

---

# 5. Reclassify the global “ledger defect”

The code computes the cell residual array and then defines the global ledger from the same boundary-flux difference and summed cell terms.

Therefore the reported `~1e-16` global defect is primarily a telescoping bookkeeping check.

It is valuable for detecting:

- sign errors;
- omitted cells;
- indexing errors;
- inconsistent summation.

It is not an independent physical validation of the energy closure.

Rename or supplement it with:

```text
total_energy_telescoping_defect
total_energy_equation_residual
total_energy_identity_mismatch
total_energy_manufactured_solution_error
```

The fixed-transport root residual is a stronger numerical gate than the telescoping defect.

---

# 6. Inner coupling should follow a staged interface audit

The repository currently proposes matching only

```text
Mdot, J, F_E
```

near `15 rg`.

Those fluxes are necessary, but they are not sufficient evidence of a physically smooth interface. Different primitive states can carry the same three fluxes.

## 6.1 Add prescribed-flux inner boundary support to the reservoir

The current `Rin=10 rg` control uses a zero-torque inner edge.

The physical inner slim disk will generally provide a nonzero angular flux and nonzero stress at `15-30 rg`.

Add a reservoir boundary type that accepts:

```text
Mdot_inner
J_inner
F_E_inner
```

or the equivalent:

```text
Mdot_inner
G_inner
F_E_inner.
```

The artificial zero-torque boundary should not be used in production overlap tests.

## 6.2 Build a common flux extractor

For both the outer reservoir and inner transonic solver, define exactly the same diagnostics:

\[
\dot M,
\qquad
J=\dot M l-G,
\qquad
F_E=\dot M B-\Omega G.
\]

They must use the same:

- gravitational potential;
- thermodynamic enthalpy convention;
- vertical-work convention;
- stress sign;
- radial velocity sign;
- and energy zero.

## 6.3 Map the overlap region rather than hard-coding `15 rg`

Evaluate candidate radii, initially over approximately

```text
12-60 rg
```

and require a contiguous overlap band satisfying all of:

```text
radial-pressure force fraction
d ln l_K / d ln R
H/R
radial Mach number
scattering and effective optical depth
radial gradient length / H
source terms negligible
inner response time / outer evolution time
```

The present data show that the pressure-force fraction is not monotonic between `15` and `30 rg`, so a single-radius gate is insufficient.

## 6.4 Flux match plus state-consistency gate

Use \((\dot M,J,F_E)\) as the primary conservation constraints.

Also require the two descriptions to agree within stated tolerances over the overlap band in quantities such as:

```text
Sigma
T or entropy
Pi
Omega/Omega_K
H/R
radial velocity
```

These state quantities need not all be imposed as hard boundary conditions. They must, however, agree diagnostically if there is no physical shock or contact layer.

If fluxes match but state profiles do not, the result is not a smooth domain match.

---

# 7. Suggested coupling sequence

## Stage C1 — no-source benchmark

Use the canonical no-wind \(\dot M/\dot M_{\rm Edd}=5\) transonic solution.

1. Extract \((\dot M,J,F_E)\) at candidate interface radii.
2. Run the reservoir with those prescribed inner fluxes and no source.
3. Verify that the outer and inner descriptions possess a common overlap solution.
4. Move the interface through the overlap band and demonstrate interface-location invariance.

This isolates coupling mechanics from stream and tide physics.

## Stage C2 — stream-fed reservoir with ideal wall/open controls

Restore the absolute stream outside the interface.

Compare:

```text
open outer edge
ideal wall
```

but retain the inner transonic flux boundary.

Determine whether the warm/thick wall–open contrast survives removal of the artificial zero-torque inner edge.

## Stage C3 — simultaneous two-domain solve

Treat as unknowns, as required:

```text
inner sonic radius
inner angular eigenvalue
interface radius
outer thermoviscous state
interface fluxes
```

Use a bordered Newton or pseudo-arclength solve.

Do not pass outer primitive variables in one direction and fluxes in the other without iterating to a common solution.

---

# 8. Tidal torque and power need a paired boundary contract

The current ideal wall is a useful limiting control.

It is not a physical binary closure.

At the wall, the stress-energy flux contains

\[
-\Omega_{\rm disk}G.
\]

For a companion pattern rotating at \(\Omega_p\), the external mechanical power is tied to the applied torque by

\[
P_{\rm ext}=\Omega_p T_{\rm disk}.
\]

The differential work associated with \(\Omega_{\rm disk}-\Omega_p\) must be assigned consistently to:

```text
local tidal dissipation
energy transferred to the orbit
waves leaving the modeled domain
```

Add a boundary object that carries mass, angular momentum, and energy together.

Before a full tidal calibration, run a sensitivity bracket at the required wall torque:

```text
current ideal stress-energy wall
pattern-speed power
distributed tidal dissipation
```

If this bracket changes the interface flux or wall temperature materially, the physical tide and inner coupling must be solved together.

---

# 9. Additional validity work

## 9.1 Effective optical depth

The canonical data explicitly state that effective optical depth is unavailable.

Add at least a diagnostic absorption opacity and report

\[
\tau_{\rm eff}
=
\sqrt{\tau_{\rm abs}
\left(\tau_{\rm abs}+\tau_{\rm es}\right)}.
\]

Do not certify an optically thick thermalized overlap using scattering depth alone.

## 9.2 Independent entropy-advection audit

From the total-energy root, calculate

\[
Q_{\rm adv}
=
-\frac{\Sigma u}{R}
T\frac{ds}{d\ln R}
\]

as an audit.

Do not put this term back into the total-energy solver.

Use it to report:

```text
local Qadv/Qvisc
integrated entropy-advection fraction
photon-trapping diagnostics
```

This is the correct way to determine whether the new wall state is actually strongly advective.

## 9.3 Boundary-layer exclusion

Report maxima both:

```text
over the entire grid
over the validity-gated physical domain
```

The current maximum radial-pressure fraction is boundary dominated and should not be mixed with fixed-radius physical diagnostics.

---

# 10. Recommended work packages

## WP0 — Freeze and reclassify `248e43c`

- preserve the commit and canonical controls;
- label the energy implementation as a prototype pending identity proof;
- distinguish root residuals from telescoping defects;
- record the exact implemented flux/work variant in provenance.

## WP1 — Correct/prove the total-energy identity

- derive the continuous equation including variable \(\dot M\);
- select Option A or Option B;
- implement the matching cell work term;
- add direct transonic-equivalence tests;
- add source-bearing manufactured tests;
- rerun the resolution ladder.

## WP2 — Prescribed inner flux boundary and overlap audit

- add `Mdot/J/F_E` inner boundary data;
- remove the zero-torque interface from production tests;
- add a common inner/outer flux extractor;
- map the overlap band;
- require flux and state compatibility.

## WP3 — Coupled no-wind inner–outer solve

- first recover the canonical no-wind transonic benchmark;
- prove interface-radius invariance;
- then restore the absolute stream and ideal wall/open controls.

## WP4 — Physical tidal torque and power

- add a binary-calibrated torque;
- add pattern-speed power;
- close disk-plus-orbit angular momentum and energy;
- continue between open and wall limits.

## WP5 — Stability and coupled time evolution

- test multiple thermal seeds;
- compute the coupled Jacobian/eigenvalues;
- implement mass+total-energy IMEX evolution;
- allow accumulation, fronts, and cycles.

## WP6 — Wind

Only after WP1-WP5 pass:

- port the terminal-Bernoulli wind;
- use one mass/angular/energy wind state;
- impose a vertical/surface-mass launch limit near stagnation;
- search for a genuinely strongly mass-loaded branch.

---

# 11. Provisional acceptance gates

## Energy identity

```text
analytic pointwise identity mismatch       <= 1e-10 relative
discrete smooth-profile convergence        expected order demonstrated
N512 transonic-equivalence residual        <= 1e-6 relative
source mass/J/E normalization              <= 1e-12 relative
no viscous-work double counting
```

## Interface

```text
Mdot mismatch                              <= 1e-8 relative
J mismatch                                 <= 1e-8 relative
F_E mismatch                               <= 1e-7 relative
state mismatch over overlap                stated and mesh converged
result stable under interface movement
no zero-torque boundary layer in overlap
```

## Validity

```text
radial-pressure fraction                   preferably < 0.03 in core overlap
d ln l_K/d ln R                            safely separated from zero
effective optical depth                    above declared diffusion gate
radial Mach number                         below declared reservoir gate
gradient lengths/H                         above declared scale-separation gate
```

## Tidal closure

```text
angular ledger                             <= 1e-8 relative
energy/pattern-power ledger                <= 1e-7 relative
required torque physically compared with binary capacity
```

---

# 12. Stop conditions

Stop and revise the formulation if:

1. the corrected total-energy identity materially changes the wall/open states;
2. no common inner/outer overlap band exists;
3. fluxes match but primitive states remain discontinuous;
4. a physical tidal power prescription changes the wall state qualitatively;
5. effective optical depth fails in the proposed overlap;
6. the warm wall root is thermally or viscously unstable;
7. the coupled solution requires non-Keplerian outer rotation over most of the reservoir;
8. a steady solve suppresses accumulation demanded by the absolute stream supply.

These outcomes are scientific information, not failed numerics.

---

# 13. Immediate Codex prompt

```text
Review commit 248e43c and implement WP1 only.

Before inner transonic coupling, verify the algebraic compatibility of the
checked-in enthalpy flux

F_E = Mdot [q + e + Pi/Sigma] - Omega G

with the checked-in vertical-work term

Mdot [dPi/Sigma - P drho/rho^2].

Derive the source-free identity from radial momentum and angular conservation.
Test the conclusion against the certified no-wind transonic profile without
reusing the pre-simplified legacy identity.

The expected enthalpy-flux correction is

Mdot [Pi/Sigma^2 dSigma - P/rho^2 drho]
= Mdot (P/rho) dlnH.

Alternatively, retain the current work term only with an internal-energy
advective flux. Choose and document one consistent representation.

Add pointwise, finite-volume convergence, and source-bearing manufactured
tests. Recompute the near-ISCO rejection witness and the Rin=10 wall/open
ladders. Do not begin inner coupling, tidal continuation, or wind until this
identity gate passes.
```

---

# 14. Bottom line

Commit `248e43c` correctly moves the project toward a conservative total-energy formulation and correctly rejects the near-ISCO use of the fixed-Keplerian reservoir.

But the next step should **not yet be the inner transonic match**.

First prove that the newly implemented enthalpy flux and vertical-work correction belong to the same mathematical energy representation.

Once that gate passes, the highest-value implementation is a prescribed-flux inner boundary and a genuine overlap audit—not another artificial zero-torque interface.
