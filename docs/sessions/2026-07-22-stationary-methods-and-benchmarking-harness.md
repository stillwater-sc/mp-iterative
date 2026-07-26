# Session: stationary methods, benchmarking harness, and Krylov epic organization

**Date**: 2026-07-22 – 2026-07-26
**Duration**: Multi-day continuous session
**Participants**: Theodore Omtzigt (Ravenwater), Claude Code

## Objective

Build out mp-iterative from the composition scaffold into a repo that tests and
*characterizes* mixed-precision iterative methods: deliver the stationary trio
(Jacobi, Gauss-Seidel, SOR) with accumulator strategies, stand up a benchmarking
category that quantifies the arithmetic × accumulator trade-offs, and organize
the Krylov work into a tracked epic. Every code change lands as a small PR,
merged once CI is green on all four platforms (GCC/Clang/MSVC/AppleClang).

## Context

mp-iterative is the integration layer where MTL5's linear algebra meets
Universal's number systems; MTL5 itself must never depend on Universal. The
session opened by migrating `test_cg_quire.cpp` out of MTL5 (where it had crept
in) into this repo, and reorganizing into the stationary → krylov → multigrid
category structure. From there the work followed the epics: stationary methods
(#7) end to end, then the benchmarking harness (#6/#25/#27), with the Krylov
family (#13) captured as a tracked epic for a second developer to advance in
parallel.

Throughout, MTL5-side enablement (the smoother/solver `Accumulator` parameters)
was filed as MTL5 issues and consumed here once merged — keeping the
Universal-free boundary intact.

## Work Completed

### Category structure + quire bridge migration (#1)
Migrated `test_cg_quire.cpp` from MTL5 (removed there via mtl5 #258), converted
from Catch2 to the repo's framework-free style, and reorganized `applications/`,
`tests/` by method category. Moved `quire_accumulator.hpp` into this layer and
dropped its `MTL5_HAS_UNIVERSAL` guard — the coupling is unconditional here.

### Stationary methods (epic #7): Jacobi #8, Gauss-Seidel #10, SOR #11
For each: an MTL5 issue for the smoother `Accumulator` parameter (mtl5 #262–264,
merged as #265–267), then a `test_<m>_quire.cpp` asserting `quire <= naive` for
posits plus a `<m>_precision` demo across number types × strategies. Recurring
finding on the model tridiagonal problem: the strategies **tie** — the 2-term
row sum has nothing to accumulate. Method-specific results: GS reaches tolerance
in ~⅔ the sweeps of Jacobi (in-place sweep dampens error, #10); SOR's empirical
optimal ω drifts down as precision drops and over-relaxation hurts at posit16
(#11). En route, discovered cfloat has a genuine fused `fma` in Universal, so it
earns a real FMA column.

### Benchmarking harness (#6 → #25 → #27)
- **#24** — new `benchmarks/` category (`MPITERATIVE_BUILD_BENCHMARKS`, OFF in
  CI), header-only harness (problems, double-CG reference, forward/backward
  error + convergence-rate metrics, CSV), three stationary drivers, and a
  metrics unit test so the reusable core stays under CI even with drivers off.
  Validated: empirical rate recovers the analytic spectral radius.
- **#26** — 2D (5-point) and 3D (7-point) Poisson families + runner
  generalization. **The headline result**: as the stencil widens the quire
  super-accumulator separates from naive/FMA — inert at 2 terms/row, −44% to
  −48% on the 3D residual floor for posits. FMA ties naive (only quire
  accumulates the whole sum exactly).
- **#27** — SuiteSparse Matrix Market loader (reuses MTL5's `mm_read`; fetch
  script; committed SPD fixture for CI). Reference scope reframed: stationary
  smoothers need SPD/DD operators anyway, so the CG-SPD reference suffices and a
  non-SPD reference is deferred to the future Krylov drivers. Confirmed the
  trend on the real well-conditioned `gr_30_30` (−40%/−42%); documented the
  caveat that ill-conditioned stiffness matrices (nos3, bcsstk14) are
  convergence-limited and mask the effect.

### Krylov epic organization (#13) + provenance
Created the epic and one sub-issue per solver (#14–#23), retroactively closing
CG/BiCGSTAB/GMRES against their already-merged PRs (#1/#2/#9) to recover
provenance, and cross-commenting those PRs. Filed the MTL5 smoother/solver
issues; closed mtl5 #271 (IDR(s) wiring) as superseded by mtl5 #270. Assigned
the open Krylov sub-issues + mtl5 #271 to a second developer (Gurleen-kansray),
sized M / estimate 5 on the "Mixed-Precision Iterative" and "MTL5" project
boards. Created the `stationary`, `benchmarking`, `krylov`, `epic` labels.

### Late-session hardening
- Tightened `test_idr_s_quire` from a weak "within 50%" check to `quire <=
  naive`, matching the sibling Krylov tests (#39) — after verifying the quire
  path is genuinely live (posit32 residual 5.99e-5 → 4.35e-5).
- Closed the completed IDR(s) #17 and QMR #22 sub-issues; corrected the five
  remaining Krylov sub-issue bodies, which wrongly said "file the mtl5 issue
  first" (the MTL5 wiring is now done for all ten solvers — remaining work is
  test-only).
- Linked #32 (full CG mixed-precision dynamics study, needing separate
  matvec/dot accumulators) to roadmap **milestone 2** (#40), keeping it out of
  epic #13's single-accumulator scope.

### Parallel work merged from the second developer
IDR(s) test (#12), CG precision study (#29), QMR test (#31), and a quire
`value()`/fixpnt fix (#34) landed on `main` during the session; verified,
branch-cleaned, and folded into the epic bookkeeping.

## State at End of Session

- **Stationary (epic #7)**: complete — solvers, accumulator strategies, and a
  benchmarking harness that measured the quire payoff on analytic (1D/2D/3D
  Poisson) and real (SuiteSparse) operators.
- **Krylov (epic #13)**: 7 of 10 solvers delivered (CG, BiCGSTAB, GMRES,
  IDR(s), QMR + all-ten MTL5 wiring); 5 tests remain (BiCG #18, BiCGSTAB(ell)
  #19, CGS #20, MINRES #21, TFQMR #23), all unblocked and assigned.
- **Milestone 2**: #32 (CG dynamics study) tracked; needs a `cg<..., MatvecAcc,
  DotAcc>` extension in MTL5 first.

## Follow-ups

- MTL5 #269 — backward-sweep + symmetric GS / SSOR primitives (unblocks the #4
  sweep-order study and #5 SSOR).
- Krylov #18–#21, #23 — the five remaining solver quire tests (pure
  mp-iterative work now).
- #32 — the full four-way CG mixed-precision dynamics study (milestone 2).
- Extend the accumulator sweep to a preconditioned solve so the effect shows on
  ill-conditioned SuiteSparse operators.
