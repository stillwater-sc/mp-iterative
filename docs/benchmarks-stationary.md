# Stationary methods: first mixed-precision characterization

This report is produced by the benchmarking harness (issues #6, #25, #27). It
measures Jacobi, Gauss-Seidel, and SOR on the standard Poisson operator in 1D
(`tridiag(-1, 2, -1)`), 2D (5-point), and 3D (7-point), and on real SPD
operators loaded from the SuiteSparse Matrix Collection, across a value-type
ladder and the three accumulator strategies (naive / FMA / quire), comparing
each low-precision run against a high-accuracy `double` reference solution.

The 1D findings (1–3) established the harness and showed the accumulator
strategies tie on a tridiagonal operator. Finding 4 (added for #25) is the
resolution: **as the stencil widens, the quire super-accumulator separates from
naive/FMA** — the accumulator strategy becomes a real lever once the row sums
are long enough.

## What is measured

For every (method × number type × accumulator strategy) the harness reports:

- **Forward relative error** `‖x − x_ref‖₂ / ‖x_ref‖₂` — distance to the true
  solution (`x_ref` from a tight `double` CG solve).
- **Backward error** — residual `‖Ax − b‖` in the 2- and ∞-norms.
- **Convergence rate** — empirical geometric mean of the per-sweep residual
  reduction, next to the theoretical spectral radius of the iteration matrix.
- **Stagnation floor** — the best residual reached before the low-precision
  iteration plateaus.

Forward and backward error are kept distinct on purpose: a low-precision run can
drive the residual down while the solution itself is off, or stagnate at a
residual floor set by the arithmetic rather than the algorithm.

## How to run

```bash
cmake -B build -DMPITERATIVE_BUILD_BENCHMARKS=ON
cmake --build build -j
# <driver> [--history] [1d|2d|3d] [m] [sweeps]; m = grid points per dimension
./build/benchmarks/stationary/benchmark_sor 1d 48 3000     # summary CSV on stdout
./build/benchmarks/stationary/benchmark_sor 3d 12          # 12^3 = 1728 unknowns
./build/benchmarks/stationary/benchmark_sor --history 2d 32 > sor2d_history.csv
```

Findings 1–3 below are `1d`, `m = 48`, `3000` sweeps, `omega_opt = 1.880`.
Finding 4 sweeps the dimension at the drivers' default grid sizes.

## Finding 1 — the harness recovers the theoretical convergence rate

The measured empirical rate matches the analytic spectral radius, which
validates the measurement pipeline end to end:

| method | empirical rate (double) | theoretical rate |
|--------|-------------------------|------------------|
| Jacobi | 0.99791 | 0.99795 (`cos(π/(n+1))`) |
| Gauss-Seidel | 0.99587 | 0.99590 (`ρ_J²`) |
| SOR @ ω_opt | 0.8942 | 0.8796 (`ω_opt − 1`) |

SOR at the optimum is the only configuration that reaches the `1e-6` tolerance
within budget (**159 sweeps** in `double`); Jacobi and GS on this deliberately
stiff operator (`ρ_J ≈ 1`) do not.

## Finding 2 — the precision floor, not the accumulator, dominates

**Across every method and every number type, the naive / FMA / quire strategies
produce identical results** (to the digits shown). On a tridiagonal operator
each row sum has only two off-diagonal terms, so there is essentially no
accumulation error for an exact quire or a fused multiply-add to remove — the
stagnation floor is set by the value-type representation of the iterate and the
diagonal solve, not by how the short row sum is accumulated.

This is the central quantitative result *for the tridiagonal case*: **the
accumulator strategy is not the lever on a 2-term row sum.** Quire/FMA can only
pay off when rows are long enough that intermediate rounding in the dot product
accumulates. Widening the stencil is exactly what Finding 4 does — and there the
picture changes.

## Finding 3 — the stagnation floor separates the arithmetics sharply

At matched compute budget the floor is a clean function of the number type:

| type | Jacobi floor | GS floor | SOR floor |
|------|-------------:|---------:|----------:|
| double | 1.3e-2 | 2.8e-5 | 6.2e-13 |
| float | 1.3e-2 | 1.2e-4 | 3.0e-4 |
| cfloat⟨32,8⟩ | 1.3e-2 | 1.2e-4 | 3.0e-4 |
| posit⟨32,2⟩ | 1.3e-2 | **3.4e-5** | **6.5e-5** |
| cfloat⟨16,5⟩ | 8.2e-1 | 8.2e-1 | 2.2e0 |
| posit⟨16,2⟩ | 1.2e0 | 1.2e0 | 4.8e0 |

Two things stand out:

- **posit⟨32,2⟩ beats IEEE `float` at matched 32-bit width** on GS and SOR
  (floor 3.4e-5 vs 1.2e-4, and 6.5e-5 vs 3.0e-4) — posit's tapered precision
  near the solution's magnitude buys roughly a half-digit — while tying on
  Jacobi. A concrete "which arithmetic pays off" datapoint.
- **16-bit types never converge** on this operator: their floor sits at
  O(1). Under SOR, posit⟨16,2⟩'s floor is *worse* than cfloat⟨16,5⟩'s (4.8 vs
  2.2) — over-relaxation amplifies the coarse resolution, and the empirical rate
  is reported as `NaN` because there is no converging segment to measure. This
  echoes the SOR ω-vs-precision drift study (#5): at very low precision plain
  Gauss-Seidel is the safer choice.

## Finding 4 — denser stencils separate quire from naive (#25)

Re-running the sweep on the 2D (5-point, 4 off-diagonal terms per interior row)
and 3D (7-point, 6 terms) Poisson operators — same drivers, `benchmark_sor 2d` /
`benchmark_sor 3d` — the quire super-accumulator pulls ahead of naive/FMA, and
the gap grows with the stencil width. The residual **floor** (the backward error
the iteration settles at) for the posit types:

| operator | off-diag/row | posit⟨16,2⟩ naive → quire | posit⟨32,2⟩ naive → quire |
|----------|:------------:|---------------------------|---------------------------|
| 1D tridiagonal | 2 | 10.4 → 10.4 (tie) | 1.50e-4 → 1.50e-4 (tie) |
| 2D 5-point | 4 | 8.74 → **7.57** | 1.32e-4 → **1.11e-4** |
| 3D 7-point | 6 | 0.69 → **0.39** (−44%) | 1.02e-5 → **5.3e-6** (−48%) |
| *(SOR at ω_opt, drivers' default grids)* | | | |

The effect is not SOR-specific — it holds across all three methods in 3D
(posit⟨32,2⟩ residual floor, naive → quire):

| method | posit⟨16,2⟩ | posit⟨32,2⟩ |
|--------|-------------|-------------|
| Jacobi | 0.324 → 0.207 (−36%) | 1.44e-5 → 1.25e-5 (−14%) |
| Gauss-Seidel | 0.331 → 0.212 (−36%) | 5.18e-6 → 3.65e-6 (−30%) |
| SOR | 0.69 → 0.39 (−44%) | 1.02e-5 → 5.3e-6 (−48%) |

Reading of the result:

- **The accumulator strategy is a real lever once `nnz/row` grows.** On the
  2-term tridiagonal row sum there is nothing to accumulate exactly; at 4 and 6
  terms the naive posit accumulation rounds after every add, and the exact quire
  — one rounding for the whole row sum — measurably lowers the floor. This is the
  hypothesis Finding 2 deferred, now confirmed.
- **Quire, not FMA, is what wins.** FMA ties naive throughout: it removes the
  *product* rounding, but the accumulation *across* terms still rounds in the
  value type. Only the quire accumulates the whole row sum exactly.
- **The gain is in backward error (residual floor), not forward error.** The
  forward relative error is essentially tied between naive and quire; quire buys
  a lower residual the iteration can settle to, which is the honest place to
  expect an accumulation improvement to show.
- **Extrapolation.** 6 terms already buys ~2× on the floor; genuine high-`nnz/
  row` operators (SuiteSparse) should widen the gap further. That is the
  follow-up (SuiteSparse loader), for which this same harness re-runs unchanged.

## Finding 5 — real SuiteSparse operators confirm the trend, with a caveat (#27)

The harness loads Matrix Market operators (`benchmark_gauss_seidel --matrix
PATH.mtx`, via MTL5's `mm_read`) and runs the same sweep. The stationary
smoothers converge only for SPD (GS/SOR) or diagonally-dominant (Jacobi)
operators, so the matrices here are SPD and the `double` CG reference stays
valid.

**Well-conditioned real operator — `gr_30_30`** (SuiteSparse `HB/gr_30_30`, a
900-unknown 9-point Laplacian, 9 nnz/row). GS converges in `double` (1107
sweeps to 1e-6), so the low-precision floors are genuine rounding floors. Quire
lowers the posit residual floor, naive → quire:

| type | naive floor | quire floor | change |
|------|------------:|------------:|-------:|
| posit⟨16,2⟩ | 1.230 | **0.715** | −42% |
| posit⟨32,2⟩ | 1.97e-5 | **1.18e-5** | −40% |

That is consistent with — and slightly beyond — the 3D 7-point result (−30% GS
posit⟨32,2⟩), on a real operator at 9 nnz/row. FMA ties naive, as everywhere.

**Caveat — ill-conditioned high-`nnz/row` operators.** The obvious high-`nnz/
row` SPD matrices are stiffness/biharmonic operators (`nos3` ~18 nnz/row,
`bcsstk14` ~35), and they are ill-conditioned enough that **even `double` GS does
not reach a rounding floor** in a fixed budget (`nos3`: `double` floor 15.6 after
1000 sweeps — the iteration is convergence-limited, not accumulation-limited).
Quire still shows a smaller edge there (`nos3` posit⟨16,2⟩ floor 22.9 → 19.4,
−16%), but the honest reading is that the accumulator benefit is cleanest to
measure on **well-conditioned** operators; on stiff ones the slow convergence
dominates the floor. Iterative refinement or a preconditioner would be needed to
expose the accumulator effect on those (a mixed-precision study, milestone 2).

## Conclusions

- The harness measures convergence rate, forward/backward error, and the
  precision floor correctly, validated against analytic Poisson theory.
- **The number system and the accumulator strategy are both levers, and which
  one matters depends on the operator.** On a tridiagonal row sum the number
  system dominates and the accumulator does nothing (Findings 2–3); posit⟨32,2⟩
  edges out `float` at matched width on the sequential methods.
- **As the stencil widens the quire super-accumulator separates from naive/FMA**
  (Finding 4): −44% to −48% on the residual floor for 3D posits, consistent
  across Jacobi/GS/SOR. FMA does not help — exact accumulation of the whole row
  sum is what wins. This confirms the accumulator thesis this whole effort is
  built on: quire pays off exactly when there is a sum long enough to accumulate.
- **Real operators confirm it, with a caveat** (Finding 5): on the
  well-conditioned `gr_30_30` Laplacian quire lowers the posit floor −40% to
  −42%, extending the trend to a real matrix; but ill-conditioned high-`nnz/row`
  stiffness matrices are convergence-limited, so the accumulator effect there is
  masked until a preconditioner or iterative refinement is added (milestone 2).

*Generated from `benchmarks/stationary/` on the standard 1D/2D/3D Poisson
problems; `double`/`float`/`cfloat`/`posit` results are deterministic across GCC
and Clang.*
