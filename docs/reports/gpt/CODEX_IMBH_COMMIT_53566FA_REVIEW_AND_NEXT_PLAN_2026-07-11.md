# Review of Commit `53566fa` and Recommended Next Plan

**Project:** IMBH/QPE stream-fed minidisk
**Repository:** `huanyang07/IMBH`
**Branch:** `codex/repository-cleanup-after-1e7438e`
**Commit reviewed:** `53566fa8c3e4639df1b0dd637ab14494f1a31c36`
**Commit title:** `Add signed-flux thermoviscous stream model`
**Review date:** 2026-07-11, Asia/Tokyo
**Review type:** Static review of the checked-in reports, equations, implementation, runners, and tests. The reviewer did not independently rerun the production calculations.

---

# 1. Executive verdict

Commit `53566fa` is the strongest architectural step in the project so far.

It accomplishes several important goals:

1. It resolves the earlier low-`eta_E` numerical bottleneck sufficiently to certify mesh-supported `eta_E=8` states under the unified positive-flux solver.
2. It introduces a terminal-Bernoulli wind branch with approximately `6.87-6.88%` wind loss at `B_infinity=0.02 c^2`.
3. It adds an independent-surface-density signed-flux disk that admits decretion and finite-density stagnation.
4. It prescribes an absolute stream supply instead of only a fraction of an imposed inner accretion rate.
5. It adds gas+radiation thermal balance and self-consistent alpha viscosity.
6. Under an ideal tidal-wall boundary, it finds a mesh-stable warm/thick numerical fixed point:
   - internal-energy advection metric about `0.548`;
   - maximum `H/R` about `0.341`;
   - luminosity about `1.32 L_Edd`.
7. Under an open zero-torque edge, it finds a cooler accretion/decretion solution:
   - inner accretion fraction about `0.1885`;
   - overflow fraction about `0.8115`;
   - internal-energy advection metric about `0.054`;
   - maximum `H/R` about `0.157`.

This is strong evidence that the outer angular-momentum boundary controls the thermal state.

However, the new state is not yet a physically closed hot slim-disk branch.

The two largest unresolved closures are:

1. the stream angular momentum is audited but is not yet included in the signed-flux transport equation; and
2. the thermal solver conserves an internal-energy ledger, not the full mass/angular/total-energy system.

The recommended next order is therefore:

```text
physical angular-momentum closure
-> total-energy/enthalpy closure
-> inner transonic coupling
-> stability and time evolution
-> wind in the signed model
```

Do not add wind to the thermoviscous signed model before the first three items pass.

---

# 2. Updated scientific status

## 2.1 Results that are numerically supported

### Unified positive-flux wind branch

The previous source-quadrature diagnosis has been corrected:

- compact-source cell mass is integrated from its cumulative primitive;
- constant stream angular and energy moments use the same exact mass increment;
- one interval operator is shared by production and tests;
- high-order transport audits are available;
- an interval-local Jacobian and bordered/direct continuation recover strict roots through `eta_E=8`.

The low-`eta_E` wall was mainly a Jacobian/corrector problem rather than an unresolved source integral.

The terminal-Bernoulli branch is also a useful controlled result:

```text
B_infinity = 0.02 c^2
wind / Mdot_inner = 0.0687-0.0688
no mass-cap activation
```

It confirms that stronger mass loss alone does not create the desired hot topology in the fixed-inner positive-flux model.

### Prescribed-viscosity signed-flux model

The independent-Sigma module correctly demonstrates:

- inward and outward mass flux;
- finite surface density at a zero-flux radius;
- exact integrated mass conservation;
- an implicit viscous step without hidden clipping;
- an absolute source supply;
- distinct ideal-wall and open-overflow solutions.

This is a valuable numerical bridge.

### Thermoviscous signed-flux model

The fixed-point iteration produces reproducible wall/open thermal contrasts across `N=64-512`.

The wall state is a real numerical solution of the current equations. It is the first clear evidence in this project that a hotter/thicker state can emerge from physical absolute feeding plus a strong angular-momentum sink rather than from lowering a wind launch-energy multiplier.

---

## 2.2 Claims that remain too strong

The current state should be described as:

> a mesh-stable thermoviscous **hot-reservoir candidate under an ideal wall and an internal-energy closure**.

It should not yet be described as:

- a physically closed hot slim-disk branch;
- a certified strongly advective solution;
- a physical tidal-truncation solution;
- a stable steady state;
- a QPE limit cycle;
- or a hot mass-loaded-wind solution.

Suggested status split:

```json
{
  "numerical_status": "SUPPORTED BUT NOT FULLY CERTIFIED",
  "physical_status": "DIAGNOSTIC ONLY",
  "claim_scope": "thermoviscous fixed point under fixed-Keplerian angular transport, ideal wall, and internal-energy advection"
}
```

The current quantity called `f_adv` should temporarily be renamed:

```text
f_internal_energy_export
```

or

```text
f_e,adv
```

because it is computed from the net boundary transport of internal energy, not from the complete slim-disk entropy/enthalpy identity.

---

# 3. Highest-priority issue: stream angular momentum is not in the transport solve

## 3.1 What the code currently does

The signed-flux module computes the viscous torque

\[
G=-2\pi R^3\nu\Sigma\frac{d\Omega_K}{dR},
\]

and determines face mass flux from

\[
\dot M=\frac{dG}{dl_K}.
\]

The stream specific angular momentum is then used only in the diagnostic angular ledger:

\[
S_J=S_Ml_s.
\]

It does not alter the mass-flux operator or the steady surface-density solution.

Therefore the current signed model solves the transport problem that would apply if each injected parcel arrived with the local `l_K`, then reports the torque needed to reconcile the actual `l_s`.

That is a useful diagnostic, but it is not a closed physical angular-momentum equation.

## 3.2 Correct steady relation

With inward-positive mass flux, outward-increasing \(x=\ln R\), no wind, and inward angular flux

\[
J=\dot M l-G,
\]

the conservative ledgers are

\[
\frac{d\dot M}{dx}=-S_M,
\]

\[
\frac{dJ}{dx}=-S_Ml_s+\tau_{\rm ext}.
\]

For a fixed nearly Keplerian \(l=l_K(R)\), these imply

\[
\dot M\frac{dl_K}{dx}
=
\frac{dG}{dx}
+
S_M(l_K-l_s)
+
\tau_{\rm ext}.
\]

The current implementation keeps only the \(dG/dx\) term.

With wind included later, the corresponding relation is

\[
\dot M\frac{dl_K}{dx}
=
\frac{dG}{dx}
+
S_M(l_K-l_s)
+
W_M(l_w-l_K)
+
\tau_{\rm ext}.
\]

The finite-volume implementation should preferably solve mass and angular flux ledgers directly rather than rely on a pointwise derived formula.

## 3.3 Quantitative consequence for the current open solution

For zero viscous torque at both radial boundaries, one constant-\(l_s\) source, and no other external torque, global conservation gives

\[
\frac{\dot M_{\rm in}}{\dot M_s}
=
\frac{l_{\rm out}-l_s}{l_{\rm out}-l_{\rm in}}.
\]

Using the checked-in Paczynski-Wiita potential and the stated radii

```text
Rin = 6.1 rg
Rout = 335 rg
l_s = l_K(248.96693 rg)
```

gives approximately

\[
\frac{\dot M_{\rm in}}{\dot M_s}=0.1700646.
\]

The current code reports

\[
0.1885119.
\]

The difference is about `0.01845` of the stream mass rate, or roughly `10.8%` relative to the physically closed value.

For the ideal wall, zero inner torque and constant \(l_s\) require

\[
\frac{G_{\rm out}}{\dot M_s l_s}
=
1-\frac{l_{\rm in}}{l_s}
\simeq0.7689866.
\]

The current code reports `0.75189398`.

The difference,

\[
0.0170926,
\]

is exactly the reported required mixing-torque fraction.

This is an excellent regression test for the next angular closure.

## 3.4 Required implementation

Create a source state used by every equation:

```python
@dataclass(frozen=True)
class StreamInjectionState:
    mass_rate_cells: ndarray
    angular_momentum_rate_cells: ndarray
    total_energy_rate_cells: ndarray
```

Do not pass `source_specific_angular_momentum` separately to the mass solver and duplicate it inside a thermal-closure object.

An accepted signed-flux solution should require:

```text
unmodeled angular-momentum defect = 0
```

within tolerance.

A nonzero residual may be assigned to a named physical term such as:

```text
stream-disk mixing torque
distributed tidal torque
outer-boundary companion torque
wind torque
```

but it must not remain an unnamed diagnostic torque.

---

# 4. The current energy “closure” is not an independent total-energy test

## 4.1 What is currently conserved

The thermal module evolves

\[
E_{{\rm th},i}=\Sigma_i e_i A_i
\]

with internal-energy face flux

\[
F_e=\dot M e_{\rm donor}
\]

and source terms

\[
Q_{\rm visc}A
+
S_M(B_s-E_{\rm orb})
-
Q_{\rm rad}A.
\]

This is a consistent discrete internal-energy equation for its stated closure.

## 4.2 Why the roundoff global defect is limited

The reported global energy defect is calculated from

\[
\sum_i
\left[
(F_{e,i+1/2}-F_{e,i-1/2})
+
Q_i
\right]
\]

minus

\[
F_{e,{\rm out}}-F_{e,{\rm in}}+\sum_iQ_i.
\]

Those expressions telescope by construction.

The roundoff defect is useful for detecting indexing or sign mistakes. It does not independently prove conservation of:

- orbital energy;
- enthalpy and pressure work;
- viscous torque work;
- stream angular-momentum mixing work;
- tidal torque power;
- or energy exchange with the companion.

Rename the current field to something like:

```text
internal_energy_telescoping_defect
```

and reserve `total_energy_budget_defect` for the completed physical ledger.

## 4.3 Missing terms

The full target must account for:

- advected enthalpy, not only internal energy;
- radial pressure/compression work;
- the known one-zone vertical-work correction;
- mechanical torque work \(-\Omega G\);
- source total energy \(S_MB_s\);
- the work associated with \(l_s-l\);
- companion/tidal torque power;
- radiative loss;
- later, wind-carried energy.

Once torque work is represented in a total-energy face flux, do not also add viscous heat independently unless the algebra explicitly shows a non-double-counted internal-energy split.

## 4.4 Recommended total-energy form

Define a column total-energy state

\[
U_E=M_{\rm cell}E_{\rm col},
\]

where \(E_{\rm col}\) is derived consistently from the existing one-zone vertical thermodynamics.

Use a face energy flux of the form

\[
{\cal F}_E
=
\dot M B_{\rm col}
-
\Omega G.
\]

The cell ledger should be

\[
\frac{dU_{E,i}}{dt}
=
{\cal F}_{E,i+1/2}
-
{\cal F}_{E,i-1/2}
+
S_{E,i}
-
L_{{\rm rad},i}
+
P_{{\rm ext},i}
-
W_{E,i}.
\]

The exact definition of \(B_{\rm col}\) must be chosen so that the discrete equation reproduces the existing slim entropy equation plus the already identified vertical-work correction.

This equivalence is an acceptance test, not an optional diagnostic.

---

# 5. The ideal tidal wall needs an energy and torque closure

## 5.1 Why the wall result is informative but not yet physical

The wall enforces

\[
\dot M_{\rm out}=0
\]

while permitting a finite outward angular-momentum flux.

In a steady no-wind solution, mass conservation then forces all supplied mass through the inner boundary. The result does not by itself demonstrate that a real companion can supply the required torque or that the state is dynamically stable.

The wall solution requires an outer torque of order `0.75-0.77` of the supplied stream angular-momentum flux. That is a major part of the global ledger and must be physically checked.

## 5.2 Companion power

Let \(T_{\rm disk}\) be the torque applied to the disk and \(\Omega_p\) the binary pattern speed.

The external mechanical power is

\[
P_{\rm ext}=\Omega_p T_{\rm disk}.
\]

For a disk that loses angular momentum to a more slowly rotating tidal pattern, the positive local dissipation associated with differential rotation is schematically

\[
Q_{\rm tide}
=
(\Omega-\Omega_p)(-T_{\rm disk}),
\]

with signs fixed in the project ADR.

The code must avoid counting both \(-\Omega G\) and a separate tidal heat term inconsistently.

## 5.3 Required boundary comparison

Retain three cases:

1. **Open zero-torque edge** — control case.
2. **Ideal tidal wall** — limiting case.
3. **Finite physical tide** — distributed or boundary torque tied to the binary parameters and accompanied by a consistent power term.

Continue the physical torque amplitude from open to wall and record:

```text
inner accretion fraction
overflow fraction
required companion torque
tidal power/dissipation
H/R
thermal-advection metric
luminosity
stagnation radius
```

The scientifically important question is now:

> At what physically plausible tidal torque does the disk change from a cool overflow solution to a hot inward-processing solution?

---

# 6. Inner-boundary and optical-depth warning

## 6.1 Optical-depth minimum is not converging safely

For the tidal-wall sequence, the reported minimum scattering depth is

```text
N=64   tau_min=16.20
N=128  tau_min=6.33
N=256  tau_min=2.70
N=512  tau_min=1.23
```

For the open sequence it is

```text
30.31, 12.06, 5.19, 2.38.
```

The global thermal metrics converge, but the minimum optical depth falls by roughly a factor of two whenever the resolution doubles.

This is evidence of a boundary-layer quantity tending toward the edge, not evidence of a mesh-converged optically thick inner interface.

A simple power-law extrapolation of the four wall values would put `tau_min` below unity near the next refinement. That extrapolation is only diagnostic, but the direction is unambiguous.

## 6.2 Why the current inner radius is difficult

The outer signed model begins at

\[
R_{\rm in}=6.1\,r_g,
\]

only `0.1 rg` outside the Paczynski-Wiita ISCO at \(6r_g\).

The transport law uses

\[
\dot M=\frac{dG}{dl_K}.
\]

But

\[
\frac{dl_K}{dR}=0
\]

at the pseudo-Newtonian ISCO. The near-ISCO diffusion representation is consequently poorly conditioned and physically outside the intended nearly Keplerian reservoir regime.

## 6.3 Required interface audit

Do not keep the production signed outer core down to `6.1 rg`.

Find an overlap radius or overlap band using:

```text
d ln l_K / d ln R safely away from zero
tau and effective tau safely above the chosen diffusion threshold
H/R within the outer-model range
radial pressure force / gravity below a stated limit
Mach number small
source terms negligible
radial gradient lengths larger than H
inner response time shorter than outer evolution time
```

An initial search over roughly `10-30 rg` is reasonable, but the actual interface must be selected from the audited profiles.

If no overlap band exists, the next model must evolve non-Keplerian angular momentum and radial momentum rather than forcing a hybrid match.

---

# 7. The nearly Keplerian approximation needs a radial-force audit

The wall state reaches

\[
H/R\simeq0.341.
\]

At this thickness, a radial pressure correction of order \((H/R)^2\) can be non-negligible unless the pressure gradient is unusually weak.

Compute

\[
\epsilon_P
=
\frac{\left|\Sigma^{-1}d\Pi/dR\right|}
     {R\Omega_K^2}
\]

throughout the signed model.

Suggested interpretation:

```text
epsilon_P < 0.03    nearly Keplerian approximation comfortable
0.03-0.10           use with explicit uncertainty
epsilon_P > 0.10    fixed Omega_K closure not acceptable for production
```

These are provisional engineering gates and should be justified or adjusted after comparison with the transonic solver.

If the wall state fails, promote the signed model to evolve angular momentum with an algebraic or dynamic radial-force equation.

---

# 8. Immediate code corrections

## 8.1 Recheck the final thermoviscous fixed point

The current fixed-point routine:

1. detects convergence;
2. sets viscosity to the target;
3. recomputes transport and thermal state;
4. returns the old convergence flag and old viscosity-change metric.

It does not recompute

\[
\max\left|
\ln\frac{\alpha H_{\rm final}^2\Omega_K}{\nu_{\rm final}}
\right|
\]

after the final thermal solve.

It can therefore return `converged=True` using a stale closure test.

Required correction:

```text
recompute final transport
recompute final thermal state
recompute target viscosity
recompute final log-viscosity defect
recompute thermal acceptance
set converged only from the final state
```

Add a production test with the same tolerance used by the runner, not a looser post-hoc threshold.

## 8.2 Unify the source state

`SignedThermalClosure.stream_specific_angular_momentum` is currently stored but not used by the thermal profile, while a separate source-angular array is passed to the mass solver.

Replace duplicated parameters with one immutable source-moment object.

## 8.3 Clarify source normalization

The present Gaussian source weights sum exactly to the imposed rate on a uniform log grid.

For future nonuniform/multidomain grids, use cell-integrated source weights from an analytic primitive or fixed high-order cell quadrature. Do not normalize center samples and call the resulting radial moment exact.

## 8.4 Freeze canonical checkpoints

Add compact canonical artifacts for:

```text
signed_flux_ring_N256
absolute_stream_wall_N512
absolute_stream_open_N512
thermoviscous_wall_N512
thermoviscous_open_N512
terminal_bernoulli_B002_N512
```

Each should contain:

```text
state arrays
configuration
source moments
boundary fluxes
mass/angular/energy budgets
validity metrics
Git SHA
environment
status scope
checksums
```

The current commit adds reports and runners but no small canonical state set for the new signed solutions.

---

# 9. Validation gaps

## 9.1 Ring spreading

Machine-precision global conservation is necessary but does not prove that the diffusion coefficient and spatial operator are correct.

Add a quantitative spreading-ring benchmark:

- compare to the analytic Newtonian Green-function solution in the large-radius limit;
- evolve for a visible fraction of a viscous time, not only `2e-4`;
- report `L1`, `L2`, and mass/angular errors;
- demonstrate convergence order;
- then repeat a manufactured Paczynski-Wiita solution.

## 9.2 Angular source test

For zero torques at both boundaries and constant \(l_s\), verify the analytic boundary split:

\[
\dot M_{\rm in}/\dot M_s
=
(l_{\rm out}-l_s)/(l_{\rm out}-l_{\rm in}).
\]

The test must fail if the source angular momentum is only audited rather than dynamically included.

## 9.3 Total-energy manufactured tests

Required cases:

```text
no source/no cooling/closed boundaries
source with known mass, l, and B
nonzero boundary torque and known pattern speed
open outflow carrying mass, angular momentum, and energy
steady no-wind canonical slim state
```

## 9.4 Multiple thermal roots and stability

The current least-squares thermal solve starts from one uniform seed.

Run at least:

```text
Tseed = 1e4, 1e5, 1e6, 1e7, 1e8 K
```

and seed from both wall and open solutions.

Record:

```text
number of distinct roots
thermal Jacobian spectrum
local heating/cooling derivative
dominant global mass-energy eigenvalues
```

A steady fixed point is not a stable physical state until perturbations decay.

---

# 10. Recommended next work packages

## WP0 — Status, provenance, and small correctness patch

No physics redesign yet.

1. Freeze canonical signed-flux checkpoints.
2. Split numerical and physical status.
3. Rename the current advection metric.
4. Rename the telescoping energy defect.
5. Fix the stale final fixed-point convergence check.
6. Add radial-force and fixed-radius optical-depth diagnostics.

## WP1 — Fully conservative angular-momentum closure

This is the immediate scientific priority.

1. Use one source-moment object.
2. Add exact source angular momentum to the finite-volume transport equations.
3. Add explicit external/tidal torque cells or boundary torque.
4. Require zero unmodeled angular defect.
5. Recover the analytic open-boundary split.
6. Recompute the wall torque.
7. Re-run wall/open thermoviscous roots.

**Decision gate:** determine whether the hot wall state survives when \(l_s\) is dynamically included.

## WP2 — Total-energy/enthalpy signed model

1. Derive the column total-energy state.
2. Implement face flux \(\dot M B_{\rm col}-\Omega G\).
3. Add source \(B_s\), radiative loss, and tidal power.
4. Include the vertical-work correction.
5. Avoid double-counting viscous dissipation.
6. Demonstrate equivalence to the certified slim entropy identity.
7. Recompute wall/open roots and all hot metrics.

**Decision gate:** only after WP2 may the project call the result a physical hot-branch candidate.

## WP3 — Physical tidal-torque family

1. Keep open and ideal-wall limits.
2. Add a finite torque tied to binary/Hill parameters.
3. Include the companion pattern speed and power.
4. Continue torque strength between the two limits.
5. Locate any thermal/topological transition.
6. Verify that the required torque is physically plausible.

## WP4 — Inner transonic coupling

1. Scan for a valid overlap band.
2. Terminate the signed reservoir outside the near-ISCO singular region.
3. Match:
   ```text
   Mdot
   angular flux J
   total-energy flux E
   ```
4. Do not overconstrain primitive variables.
5. Reproduce the canonical no-wind inner branch.
6. Reject the hybrid model if no overlap band exists.

## WP5 — Stability and coupled IMEX evolution

Use the same semi-discrete residual for steady and time-dependent calculations.

1. Linearize the coupled mass/angular/energy system.
2. Classify wall/open roots as stable or unstable.
3. Implement backward-Euler validation, then BDF2 or an IMEX scheme.
4. Solve mass, energy, and viscosity consistently within each step.
5. Preserve positivity without hidden clipping.
6. Run source turn-on, perturbation, and long-duration tests.
7. Allow accumulation if no steady state exists.

## WP6 — Wind in the signed total-energy model

Only after WP1-WP5 pass.

1. Port the terminal-Bernoulli wind.
2. Use one wind mass/angular/energy state.
3. Replace a throughput-based cap near stagnation with a vertical/surface-mass launch limit.
4. Permit wind on both accreting and decreting regions where physically justified.
5. Compare no-wind and wind hot states.
6. Search for a genuinely strongly mass-loaded branch.

---

# 11. Quantitative acceptance gates

## Angular momentum

```text
global angular defect                  <= 1e-9 relative initially
unmodeled mixing torque               <= 1e-10 relative
analytic open split                    <= 1e-6 relative
wall torque analytic control           <= 1e-6 relative
```

## Total energy

```text
manufactured total-energy defect       <= 1e-9 relative
steady slim identity equivalence       <= 1e-7 relative
source mass/J/E normalization          <= 1e-12 relative
tidal torque-power ledger              <= 1e-8 relative
```

## Fixed point

```text
final thermal residual                 <= stated production tolerance
final alpha-viscosity closure defect   <= stated production tolerance
same result from multiple viscosity seeds
no stale pre-final convergence flag
```

## Validity

```text
optical depth evaluated at fixed radii and interface
effective optical depth reported
radial pressure-force fraction reported
d ln l_K / d ln R safely away from zero at interface
no production use of the outer diffusion core at 6.1 rg
```

## Stability

```text
multiple thermal seeds tested
dominant eigenvalue reported
time evolution reproduces stable roots
unstable roots evolve without numerical clipping
```

---

# 12. Stop conditions

Stop and revise the model if:

1. Including the physical source angular momentum removes the reported wall/open solutions.
2. The hot state depends on an unmodeled mixing torque.
3. Total-energy closure changes `H/R`, luminosity, or the advection metric substantially.
4. No optically thick, nearly Keplerian overlap band exists.
5. A realistic tidal torque cannot supply the required angular flux.
6. The wall fixed point is thermally or viscously unstable.
7. The desired solution requires a signed flux or accumulation that a steady solve is trying to suppress.
8. Wind strength depends on a cap tied to `|Mdot|` near a stagnation point.
9. The minimum optical depth continues to fall with refinement while the model still uses diffusion cooling to the inner edge.

These are scientific outcomes, not solver failures.

---

# 13. Recommended immediate Codex prompt

```text
Review commit 53566fa and implement WP0-WP1 only.

First freeze compact canonical wall/open signed-flux and thermoviscous states.
Then replace the diagnostic source-angular mismatch with a physically closed
finite-volume angular-momentum equation. Use one immutable stream source
containing cell-integrated mass, angular momentum, and total energy moments.

For the open zero-torque control, recover the analytic constant-l_s split:
Mdot_in/Mdot_s = (l_out-l_s)/(l_out-l_in).
For the current Paczynski-Wiita radii this is approximately 0.1700646.
For the wall control, recover G_out/(Mdot_s l_s) approximately 0.7689866.
No unnamed mixing torque may remain in an accepted physical state.

Do not yet implement wind or the full total-energy redesign.
Add final-state fixed-point convergence checks, radial-pressure diagnostics,
fixed-radius optical-depth diagnostics, tests, and a report comparing the
recomputed wall/open states with commit 53566fa.
```

---

# 14. Bottom line

Commit `53566fa` changes the project’s direction in a good way.

The main question is no longer:

> How low must `eta_E` go to create a hot branch?

It is now:

> Does a physically closed stream-plus-tide angular and energy ledger support the hot tidal-wall candidate, and is that state stable when mass and energy evolve?

The hot wall result is promising precisely because it identifies the important physical control: angular-momentum disposal.

The next task is to make that disposal mechanism explicit and energetically conservative before adding more physics.
