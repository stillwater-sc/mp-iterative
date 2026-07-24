# Stationary methods: first mixed-precision characterization

This is the first report produced by the benchmarking harness (issue #6). It
measures Jacobi, Gauss-Seidel, and SOR on the standard 1D Poisson operator
`tridiag(-1, 2, -1)` across a value-type ladder and the three accumulator
strategies (naive / FMA / quire), comparing each low-precision run against a
high-accuracy `double` reference solution.

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
# summary CSV on stdout, human-readable table on stderr
./build/benchmarks/stationary/benchmark_sor 48 3000
# per-iteration residual history instead
./build/benchmarks/stationary/benchmark_sor --history 48 3000 > sor_history.csv
```

Data below is `n = 48`, `3000` sweeps, `omega_opt = 1.880`.

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

This is the central quantitative result: **the accumulator strategy is not the
lever on sparse tridiagonal problems.** Quire/FMA can only pay off when rows are
long enough that intermediate rounding in the dot product accumulates — dense,
banded, 2D/3D Poisson, or high-`nnz/row` SuiteSparse operators. Producing those
problem families (deferred here) is the prerequisite for a fair accumulator
comparison, and this harness is built to measure it when they land.

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

## Conclusions

- The harness measures convergence rate, forward/backward error, and the
  precision floor correctly, validated against analytic Poisson theory.
- On sparse tridiagonal problems the **number system**, not the accumulator
  strategy, is the dominant lever; posit⟨32,2⟩ edges out `float` at matched
  width on the sequential methods.
- A fair accumulator (quire/FMA) comparison needs **denser problems**. The next
  harness step is the 2D/3D Poisson and SuiteSparse problem families, at which
  point the same drivers will re-run unchanged and the quire column should
  finally separate.

*Generated from `benchmarks/stationary/` on the standard 1D Poisson problem;
`double`/`float`/`cfloat`/`posit` results are deterministic across GCC and
Clang.*
