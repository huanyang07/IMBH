# Codex Handoff: Stream-Front `f_s≈0.8086` Assessment and Next Plan

Date prepared: 2026-07-02  
Repository: `https://github.com/huanyang07/IMBH`  
Latest reviewed commit page status: `Add stream-front adaptive continuation results` / `aa547bd` on 2026-07-02  
Primary notes reviewed:

```text
Note/GPT_PROMPT_STREAM_FRONT_0808.md
Note/CODEX_ADAPTIVE_SOURCE_FRONT_0808_RESULTS.md
Note/CODEX_STREAM_BRANCH_OUTER_GRID_SECANT_RESULTS.md
Note/CODEX_HIGH_MDOT_STREAM_BRIDGE_RESULTS.md
Note/CODEX_POINT2_TO5_STREAM_AND_HIGH_MDOT_RUN_RESULTS.md
```

---

## 1. Executive verdict

The current `f_s≈0.8086` stream-front wall is **mostly an outer-tail/source-boundary mesh and closure problem**, with a significant continuation-predictor limitation. It is **not yet good evidence for physical loss of the inner high-`Mdot` branch**.

A sharper statement:

```text
The inner transonic/slim branch still looks alive.
The failure localizes near the source tail / outer boundary.
The current formulation may be becoming physically ill-conditioned near the outer reservoir,
but the evidence does not yet support declaring a true physical branch endpoint.
```

Therefore, Codex should **not** simply continue hand-tuning the outer grid and inching `f_s` upward. The next implementation should convert the current ad hoc continuation into a robustness program:

```text
residual-based remeshing
+ true source-fraction tangent predictor
+ pseudo-arclength fallback
+ soft/Robin outer reservoir closure
+ source-shape / reservoir-interface tests
+ N and mesh convergence checks
```

Wind should **not** be added yet. Wind could hide a no-wind boundary/source-tail pathology and make a numerical relief mechanism look like physical wind regulation.

---

## 2. Current status snapshot

### 2.1 Standard no-wind high-rate benchmark

The standard no-wind slim-disk branch is solid through `Mdot/Edd=5`.

Important anchor at `Mdot/Edd=5`:

```text
N = 768
residual = 2.293e-6
f_adv_global = 0.4534
f_adv_inner(R<20rg) = 0.4666
Lrad/LEdd = 1.541
max H/R = 0.3164
Rson = 4.360 rg
```

Interpretation:

```text
The solver can recover a real hot/advective no-wind slim branch when true inner Mdot is high.
This is the backbone that the finite stream-fed minidisk branch must eventually connect to.
```

### 2.2 Earlier corrected stream-source bookkeeping anchor

The corrected stream-source sign convention is:

```text
dMdot/dlnR = Mdot_wind_prime - Mdot_stream_prime
```

With inward-positive accretion rate:

```text
positive stream source -> Mdot_outer < Mdot_inner
positive wind sink     -> Mdot_outer > Mdot_inner
```

At `Mdot_inner/Edd=1`, `Rout=300 rg`, the corrected source case reached:

```text
source fraction = 0.49
Mdot_outer/Mdot_inner = 0.511844
full residual = 2.438e-6
dominant residual = interval_E
max H/R = 0.1571
integrated advective fraction = 0.03404
```

Interpretation:

```text
This is a source-bookkeeping / source-annulus stiffness benchmark.
It is not the hot stream-fed IMRI branch.
```

### 2.3 New finite stream-fed high-`Mdot` branch at `Mdot_inner/Edd=2`

The finite stream-fed no-wind branch at:

```text
Mdot_inner/Edd = 2
Rout = 300 rg
narrow stream annulus centered at 0.8 Rout
stream_torque_delta_l_fraction = +0.005
N = 640
```

has advanced substantially.

Progression:

```text
old wall:        f_s ≈ 0.65
previous front: f_s ≈ 0.805
current front:  f_s ≈ 0.808585
```

Important accepted points:

```text
f_s = 0.80:
    residual = 3.756e-7
    Mdot_outer/Mdot_inner = 0.203011
    f_adv_global = 0.2039
    f_adv_inner = 0.09463
    f_adv_pos = 0.2731
    Lrad/LEdd = 0.8675
    max H/R = 0.2269
    Rson = 4.66 rg

f_s = 0.805:
    residual = 1.324e-6
    dominant residual = outer_omega
    interval_E = 2.845e-7
    peak_interval_E_rg = 298.2
    Mdot_outer/Mdot_inner = 0.19803

f_s ≈ 0.80796:
    residual = 1.494e-7
    accepted with existing annulus+outer grid
    next attempts near f_s≈0.80821 failed with interval_E ≈ 1.4e-5
    residual peak near R≈298.2--298.6 rg

f_s = 0.808585:
    residual = 8.532e-8
    reached using stronger outer-tail remeshing
    df_s = 1.25e-4
    nfev ≈ 100--130 per accepted step
```

Current branch diagnostics remain mild:

```text
f_adv_global ≈ 0.204
f_adv_inner ≈ 0.095
max H/R ≈ 0.227
Rson ≈ 4.66 rg
```

Interpretation:

```text
This is now a credible first positive-advection finite stream-fed branch,
but only at Mdot_inner/Edd=2 and only mildly advective.
It is not yet the genuinely hot Mdot=5-like stream-fed branch.
```

---

## 3. Diagnosis of the `f_s≈0.8086` wall

### 3.1 Ranking of likely causes

| Cause | Priority | Assessment |
|---|---:|---|
| Mesh / outer-tail resolution | Primary | The old `f_s=0.65` wall was removed by annulus+outer-tail grid improvements, and the newer `f_s≈0.8082` wall was crossed by stronger outer-tail remeshing. That strongly implicates mesh localization near the source tail / outer cells. |
| Outer-boundary closure | Co-primary | Failures and dominant residuals repeatedly involve `outer_omega` or `interval_E` peaks near the outer boundary. The disk at `f_s≈0.81` has only about 19--20% of the inner accretion rate entering through the outer boundary, so the outer edge is no longer a simple through-fed disk edge. |
| Predictor limitation | Important secondary | The simple secant predictor is rejected above `f_s≈0.805`; the current-state seed is often better. This is strong branch curvature. It explains tiny steps and high `nfev`, but probably not the underlying physical/numerical wall. |
| Source formulation / source tail | Possible contributor | The source is narrow and centered at `0.8 Rout`, so its outer tail and induced energy residual sit close to the boundary. The sign is fixed, but the source support and coupling to the reservoir likely matter. |
| Physical loss of branch | Not supported yet | The inner diagnostics are smooth: `Rson`, `H/R`, `f_adv_inner`, and luminosity do not show a collapse. The residual is not sonic/inner; it is source-tail/outer-boundary localized. |

### 3.2 Why this does not look like true inner-branch death

A physical loss of the hot/advective branch should leave signs such as:

```text
Rson jump or disappearance
inner f_adv collapse
H/R divergence
luminosity/advection discontinuity
mesh-independent Jacobian rank defect
fold surviving mesh/source/boundary changes
residual migrating to the sonic point or inner advective region
```

The current notes show the opposite:

```text
Rson ≈ 4.66 rg remains smooth.
max H/R ≈ 0.227 remains smooth.
f_adv_inner ≈ 0.095 remains smooth.
Residual peaks near R≈298 rg, close to the outer tail/cells.
Stronger outer-tail remeshing crosses the previous failure.
```

So do **not** call `f_s≈0.8086` a physical endpoint yet.

### 3.3 But the formulation warning is real

At `f_s≈0.808`,

```text
Mdot_outer/Mdot_inner ≈ 0.19
```

That means about 80% of the inner accretion flow is injected inside the domain. The outer boundary is then acting less like the edge of a smooth through-fed disk and more like a stream-fed reservoir/cavity interface.

So the best phrasing is:

```text
The current wall is a numerical/closure manifestation of an under-modeled
outer reservoir/source-tail interface, not proven physical loss of the branch.
```

---

## 4. Immediate implementation plan

### Step 1 — Freeze regression anchors

Codex should preserve these checkpoints and require them to keep passing after every code change:

```text
A. Standard no-wind Mdot/Edd=5 large-Rout anchor
   residual = 2.293e-6
   f_adv_global = 0.4534
   f_adv_inner = 0.4666
   Lrad/LEdd = 1.541
   max H/R = 0.3164
   Rson = 4.360 rg

B. Finite-Rout no-stream Mdot/Edd=2, Rout=300 anchor
   residual ≈ 4.441e-6
   f_adv_global ≈ 0.1995
   f_adv_inner ≈ 0.0931
   Lrad/LEdd ≈ 0.8844
   max H/R ≈ 0.2269
   Rson ≈ 4.660 rg

C. Stream-fed Mdot_inner/Edd=2, Rout=300, f_s=0.50, no angular source
   residual = 1.743e-6
   dominant = outer_omega
   f_adv_global = 0.2026
   f_adv_inner = 0.09461
   Lrad/LEdd = 0.8735
   max H/R = 0.2269
   Rson = 4.66 rg

D. Stream-fed Mdot_inner/Edd=2, Rout=300, f_s=0.80, torque +0.005
   residual = 3.756e-7
   Mdot_outer/Mdot_inner = 0.203011
   f_adv_global = 0.2039
   f_adv_inner = 0.09463
   f_adv_pos = 0.2731
   Lrad/LEdd = 0.8675
   max H/R = 0.2269
   Rson = 4.66 rg

E. Stream-fed Mdot_inner/Edd=2, Rout=300, f_s≈0.808585, torque +0.005
   residual = 8.532e-8
   strong outer-tail remesh
   expensive tiny-step continuation
```

### Step 2 — Replace hand-tuned outer clustering with residual-based remeshing

Do not keep manually dialing `SOURCE_GRID_OUTER_FRACTION` and `SOURCE_GRID_OUTER_WIDTH` as the main strategy.

Implement a reusable remeshing monitor:

```text
M(R) = 1
     + A * norm(|interval_E|)
     + B * norm(stream_source_prime)
     + C * norm(|dMdot/dlnR|)
     + D * norm(|dQstream/dlnR|)
     + E * outer_boundary_layer_weight
```

Then equidistribute mesh points in cumulative monitor space.

Implementation requirements:

```text
- preserve exact Rout;
- preserve source normalization after remap;
- preserve stream metadata in checkpoints;
- interpolate state carefully, conservatively where possible;
- repolish at fixed f_s after remap;
- write old/new grid, residual profiles, source integrals, and mass budget;
- compare residual localization before/after remap.
```

Initial validation targets:

```text
f_s = 0.70, 0.80, 0.805, 0.8085
N = 640, 768, 896
mesh = current hand-tuned vs residual-remeshed
```

Pass criteria:

```text
residual <= 1e-6 preferred, <= few e-6 acceptable
mass budget closes
f_adv_inner stable to <1--2%
f_adv_global stable to <1--2%
Lrad stable to <1--2%
Rson stable
max H/R stable
residual peak does not merely move to a different unresolved cell
Newton cost does not explode with N
```

### Step 3 — Implement true source-fraction tangent predictor

The current guarded secant predictor helps below the front, but above `f_s≈0.805` it becomes unreliable. Implement the actual tangent continuation equation.

For:

```text
F(z, f_s) = 0
```

compute:

```text
J_z dz/df_s = -F_f_s
```

Practical implementation:

```text
1. Reuse or assemble the square Newton Jacobian J_z.
2. Estimate F_f_s by finite difference in f_s at fixed z.
3. Solve J_z z_f = -F_f_s.
4. Predict z_trial = z + df_s * z_f.
5. Apply damping/line search to the tangent predictor.
6. Compare current-state seed, secant seed, and tangent seed.
7. Choose the seed with the lowest initial residual.
```

Use cost-aware step control:

```text
if nfev < 30 and residual << tolerance:
    grow df_s mildly
elif 30 <= nfev <= 70:
    keep df_s
elif nfev > 70:
    shrink next df_s even if accepted
elif nfev > 120:
    mark as front-like/expensive and do not grow
```

### Step 4 — Add pseudo-arclength continuation before declaring an endpoint

If the tangent norm grows, the branch bends, or the smallest singular value drops, switch from simple `f_s` stepping to pseudo-arclength continuation.

Solve:

```text
F(z, f_s) = 0

t_z · (z - z0) + t_f * (f_s - f_s0) - Delta_s = 0
```

A physical fold can only be claimed if it persists under:

```text
N = 640, 768, 896
residual-based remeshing
soft/Robin outer boundary
source-shape variants
pseudo-arclength continuation
```

Until then, `f_s≈0.8086` should be described as a **current continuation front**, not a physical branch end.

### Step 5 — Implement a soft/Robin outer angular closure

The repeated `outer_omega` sensitivity is too important to treat as tolerance noise. Implement a homotopy from the current hard outer angular closure to a reservoir/Robin closure.

Example form:

```text
R_outer = (1 - chi) * R_hard_omega + chi * R_robin_omega
```

where:

```text
R_robin_omega =
    a_l     * (l_out - l_reservoir) / lK_out
  + a_slope * (dlnOmega/dlnR - slope_reservoir)
```

Scan:

```text
chi = 0, 0.25, 0.50, 0.75, 1.0
```

Use it first at:

```text
Mdot_inner/Edd = 2
Rout = 300 rg
f_s = 0.80, 0.805, 0.8085
```

Then use it for the high-rate compact finite-boundary blockers:

```text
Mdot/Edd = 5:
    Rout = 450 -> 400 -> 350 -> 300
    current blocker: outer_omega near Rout≈400 rg

Mdot/Edd = 3:
    Rout = 3350 -> 3300 -> ... -> 300
    current blocker: interval_E near Rout≈3300 rg
```

### Step 6 — Test source-shape and source-reservoir dependence

The source sign/bookkeeping is fixed, but the current source tail may be interacting badly with the boundary.

Run fixed-integrated-`f_s` comparisons:

```text
A. current narrow source, width = 0.08, Rinj/Rout = 0.80
B. compact C2 bump source with zero tail before Rout
C. source centered slightly inward: Rinj/Rout = 0.70 or 0.75
D. broader source only after soft outer closure is available
```

Use:

```text
f_s = 0.50, 0.70, 0.80, 0.805
Mdot_inner/Edd = 2
Rout = 300 rg
torque_delta_l_fraction = 0 and +0.005
```

Require:

```text
same mass budget
f_adv_inner within ~1--2%
Lrad within ~1--2%
Rson stable
max H/R stable
no qualitative change in residual localization except expected source-tail shifts
```

If the wall moves dramatically when the source tail is kept away from the boundary, then the current wall is primarily a source-tail/boundary-interface artifact.

### Step 7 — Validate before pushing toward `f_s -> 1`

Do not chase `f_s=0.81, 0.82, ...` before proving the already accepted branch is robust.

Validation matrix:

```text
f_s = 0.70, 0.80, 0.805, 0.8085
N = 640, 768, 896
mesh = hand-tuned, residual-remeshed
outer closure = hard, soft/Robin
source shape = current, compact-tapered
```

Call a point scientifically robust only if:

```text
residual <= 1e-6 preferred, <= few e-6 acceptable
mass budget closes
f_adv_inner stable to <1--2%
f_adv_global stable to <1--2%
Lrad stable to <1--2%
Rson stable
max H/R stable
dominant residual is not a single unresolved outer cell
Newton cost does not explode with N
```

---

## 5. Return to the real hot-branch target: `Mdot_inner/Edd=3` and `5`

The `Mdot_inner/Edd=2` stream-fed branch is useful and real, but it is only mildly advective:

```text
f_adv_inner ≈ 0.095
max H/R ≈ 0.227
```

The true hot/slim reference is still closer to the standard `Mdot/Edd=5` branch:

```text
f_adv_inner ≈ 0.4666
f_adv_global ≈ 0.4534
max H/R ≈ 0.3164
```

The latest high-rate finite-boundary status is:

```text
Mdot/Edd = 3:
    accepted down to Rout≈3350 rg
    next failure near Rout≈3300 rg
    dominant = interval_E

Mdot/Edd = 5:
    accepted to Rout≈450 rg
    next failure near Rout≈400 rg
    dominant = outer_omega
```

After implementing residual remeshing and soft/Robin closure, retry:

```text
Mdot/Edd = 3:
    residual-remeshed interval_E monitor
    Rout = 3350 -> 3300 -> 300
    no stream first

Mdot/Edd = 5:
    soft/Robin outer angular closure
    Rout = 450 -> 400 -> 350 -> 300
    no stream first
```

Only once `Mdot=3` or `Mdot=5` reaches `Rout=300 rg` should Codex add stream source:

```text
Mdot_inner/Edd = 3, 5
Rout = 300 rg
f_s = 0.05, 0.10, 0.30 initially
Rinj/Rout = 0.80 first, then 0.70/0.75
width = 0.08 and compact-source variant
torque_delta_l_fraction = 0 and +0.005
wind = 0
heating = 0 initially
```

This is the actual test of whether the finite stream-fed minidisk can access the genuinely hot advective branch.

---

## 6. Wind/heating decision

Do **not** add wind yet.

Wind should wait until at least one of the following is true:

```text
1. residual-based mesh + tangent predictor + soft outer closure still fail;
2. Mdot=3/5 finite-Rout branches cannot reach Rout=300 in no-wind form;
3. stream-fed high-Mdot no-wind branches are too luminous or too thick;
4. equilibrium map lacks a viable upper branch;
5. a physical fold is established robustly under mesh/boundary/source variations.
```

Heating should also wait until mass and angular-momentum source terms are stable. Add in the order:

```text
1. mass source
2. angular-momentum source / torque
3. stream heating
4. wind
```

---

## 7. Highest-risk assumptions

```text
1. That f_s can approach 1 with a hard finite outer boundary.
   At f_s≈0.81, Mdot_outer/Mdot_inner≈0.19.
   The outer boundary is no longer a normal through-fed disk edge.

2. That the narrow source annulus tail is physically harmless.
   The current interval_E wall sits near the source tail / outer edge.

3. That hand-tuned mesh success equals scientific robustness.
   The f_s≈0.808585 result is impressive but expensive and tiny-step dependent.

4. That the Mdot=2 branch is the QPE hot branch.
   It is positive-advective but mild. The true hot target remains Mdot=3--5.

5. That outer_omega residuals are just tolerance noise.
   They repeatedly limit broad-source, compact finite-Rout, and Mdot=5 branches.
```

---

## 8. Acceptance criteria for a robust stream-fed branch

A stream-fed no-wind solution should not be labeled robust unless it satisfies:

```text
1. residual <= 1e-6 preferred, <= few e-6 acceptable;
2. mass budget closes to the integrated stream/wind budget;
3. survives N = 640, 768, 896;
4. survives residual-remeshed grid, not only hand-tuned grid;
5. survives reasonable outer-closure variation;
6. source-shape dependence is small;
7. f_adv_global, f_adv_inner, f_adv_pos, Lrad, H/R, and Rson vary smoothly;
8. no unresolved single-cell outer interval_E wall;
9. Newton cost does not explode with N;
10. pseudo-arclength does not reveal an untracked fold.
```

For the genuinely hot stream-fed branch, additionally require:

```text
Mdot_inner/Edd >= 3, preferably 5;
f_adv_inner clearly above the Mdot=2 mild value;
Lrad grows sublinearly with Mdot_inner;
Rson remains smooth;
max H/R remains in a plausible slim-disk range;
branch connects continuously to the high-Mdot no-wind backbone.
```

---

## 9. Codex-ready prompt

```text
Current status:
- Latest GitHub update advances the Mdot_inner/Edd=2, Rout=300 rg,
  narrow stream-fed no-wind branch with torque_delta_l_fraction=+0.005
  to f_s≈0.808585.
- Old f_s=0.65 wall was fixed by annulus+outer-tail focused grid and guarded secant.
- Existing grid reached f_s≈0.80796 with residual 1.494e-7, then failed near
  f_s≈0.80821 with interval_E≈1.4e-5 localized near R≈298 rg.
- Stronger outer-tail remesh crossed this wall to f_s=0.808585 with residual
  8.532e-8, but required df_s=1.25e-4 and roughly 100--130 function evaluations
  per accepted step.
- Branch remains mildly advective:
    f_adv_global≈0.204
    f_adv_inner≈0.095
    max H/R≈0.227
    Rson≈4.66 rg
- This is not a sonic failure and not convincing physical branch loss.
  It is primarily an outer-tail/source-boundary numerical/closure problem,
  with a predictor limitation and possible source-tail formulation issue.

Do not add wind yet.

Implementation tasks:

1. Freeze regression anchors:
   - standard no-wind Mdot/Edd=5 anchor;
   - Mdot=2 finite-Rout no-stream Rout=300 anchor;
   - Mdot=2 stream f_s=0.50 no-torque anchor;
   - Mdot=2 stream f_s=0.80 torque +0.005 anchor;
   - Mdot=2 stream f_s≈0.808585 strong-remesh anchor.

2. Implement residual-based remeshing:
   - monitor = |interval_E| + stream_source_prime + |dMdot/dlnR|
               + |dQstream/dlnR| + outer-boundary-layer weight.
   - equidistribute mesh;
   - preserve source normalization;
   - repolish after remap;
   - output old/new residual profiles and source integrals.

3. Add N/mesh robustness checks:
   - f_s = 0.70, 0.80, 0.805, 0.8085.
   - N = 640, 768, 896.
   - require stable f_adv_inner, f_adv_global, Lrad, Rson, H/R,
     and mass-budget closure.

4. Implement true source-fraction tangent predictor:
   - solve J_z dz/df_s = -F_f_s using the square Newton Jacobian.
   - compare current-state, secant, and tangent initial residuals.
   - use cost-aware adaptive step control based on nfev.

5. Add pseudo-arclength continuation:
   - use if tangent norm grows or smallest singular value drops.
   - do not call f_s≈0.8086 a physical endpoint unless the fold is
     mesh- and boundary-closure independent.

6. Add soft/Robin outer angular closure:
   - homotopy from hard outer_omega to reservoir/Robin angular condition.
   - test at f_s=0.80 and 0.8085.
   - then use it for Mdot=5 Rout=450->300 and Mdot=3 Rout=3350->300.

7. Test source-shape dependence:
   - current narrow source;
   - compact C2 source with no tail at Rout;
   - Rinj/Rout = 0.70, 0.75, 0.80.
   - fixed integrated f_s and torque.
   - determine whether the wall tracks the source tail or the physical branch.

8. After the outer-boundary/source-tail issue is robustly solved, retry the
   true hot branch:
   - Mdot_inner/Edd = 3 and 5;
   - first no stream to Rout=300;
   - then stream source f_s=0.05, 0.10, 0.30;
   - no wind, no heating initially;
   - torque_delta_l_fraction = 0 and +0.005.

Acceptance criteria for claiming robust stream-fed branch:
- residual <= 1e-6 preferred, <= few e-6 acceptable;
- mass budget closes;
- branch survives N=640/768/896;
- branch survives residual-remeshed and not only hand-tuned mesh;
- branch survives reasonable outer closure variation;
- source-shape dependence is small;
- f_adv diagnostics, Lrad, H/R, and Rson are smooth;
- no unresolved single-cell outer interval_E wall.

Only after this should wind/heating be added.
```

---

## 10. Source references for Codex/human audit

GitHub repository:

```text
https://github.com/huanyang07/IMBH
```

Latest commit page reviewed:

```text
https://github.com/huanyang07/IMBH/commits/main
```

Relevant raw note files:

```text
https://raw.githubusercontent.com/huanyang07/IMBH/main/Note/GPT_PROMPT_STREAM_FRONT_0808.md
https://raw.githubusercontent.com/huanyang07/IMBH/main/Note/CODEX_ADAPTIVE_SOURCE_FRONT_0808_RESULTS.md
https://raw.githubusercontent.com/huanyang07/IMBH/main/Note/CODEX_STREAM_BRANCH_OUTER_GRID_SECANT_RESULTS.md
https://raw.githubusercontent.com/huanyang07/IMBH/main/Note/CODEX_HIGH_MDOT_STREAM_BRIDGE_RESULTS.md
https://raw.githubusercontent.com/huanyang07/IMBH/main/Note/CODEX_POINT2_TO5_STREAM_AND_HIGH_MDOT_RUN_RESULTS.md
```

Uploaded handoff files also used as project context:

```text
IMRI_QPE_CONVERSATION_HANDOFF_SUMMARY.md
GPT_HIGH_MDOT_NOWIND_HANDOFF.md
```
