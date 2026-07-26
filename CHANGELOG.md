# Changelog

All notable changes to mp-iterative are documented in this file.
Format follows [Conventional Commits](https://www.conventionalcommits.org/).

## [Unreleased]

### Added

#### Composition scaffold and category structure
- Repo organized by iterative-method category, in order: `stationary/`
  (Jacobi, Gauss-Seidel, SOR), `krylov/` (CG, BiCGSTAB, GMRES, ...),
  `multigrid/` — mirrored across `applications/`, `tests/`, and `benchmarks/` (#1).
- **`include/mtl/math/quire_accumulator.hpp`** — the posit + quire
  `accumulator_traits` specialization, migrated out of MTL5 (which must stay
  Universal-free) into this composition layer. The `MTL5_HAS_UNIVERSAL` opt-in
  guard was dropped: here the coupling is unconditional. A quire is a fixed-size
  super-accumulator for *any* fixed-size arithmetic (integer, fixpnt, posit, lns,
  cfloat); posit is the first instance wired up.
- **`include/sw/mp_iterative/accumulator_strategies.hpp`** — shared naive / FMA /
  quire capability gating (`quire_of`, `has_quire`, `has_fused_multiply_add`).

#### Stationary methods with accumulator strategies (epic #7)
- **Jacobi** (#8), **Gauss-Seidel** (#10), **SOR** (#11): each exercises MTL5's
  accumulator-aware smoother (mtl5 #265/#266/#267) through the naive / FMA /
  quire strategies, with a `tests/stationary/test_<m>_quire.cpp` asserting
  `quire <= naive` for posits and a `applications/stationary/<m>_precision`
  residual table across number types × strategies.
- Method-specific studies: GS-vs-Jacobi convergence rate (in-place sweep
  dampens per-sweep error — 19 vs 30 sweeps to 1e-8, #10); SOR ω-vs-precision
  sweep showing the empirical optimum drifts down as precision drops, and
  over-relaxation *hurts* at posit⟨16,2⟩ (#11).

#### Benchmarking harness (epic #7, issues #6/#25/#27)
- **`benchmarks/`** measurement category, gated by `MPITERATIVE_BUILD_BENCHMARKS`
  (OFF by default; drivers are for measurement, not CI). Shared header-only
  harness under `include/sw/mp_iterative/benchmark/`: model `problems`,
  high-accuracy `reference_solve` (double CG), error/rate `metrics`, `csv`
  writer, `matrix_source` (SuiteSparse), and a `runner` driving each smoother
  across a value-type × accumulator-strategy grid (#24).
- **1D / 2D / 3D Poisson** problem families (#26) and a **SuiteSparse Matrix
  Market loader** (#27, reusing MTL5's `mm_read`; `scripts/fetch_matrices.sh`).
- CI coverage via `tests/benchmark/` unit tests (metrics + loader against a
  committed SPD fixture) even with the drivers OFF.
- Characterization report `docs/benchmarks-stationary.md`.

#### Krylov methods (epic #13)
- Quire-accumulation tests: **CG** (#1), **BiCGSTAB** (#2), **GMRES** (#9),
  **IDR(s)** (#12, assertion tightened to `quire <= naive` in #39), **QMR**
  (#31) — 5 of 10 solvers; MTL5's accumulator wiring now covers all ten.
- **`applications/krylov/cg_precision`** — CG across number types × accumulator
  strategy (#29), the Krylov analogue of `jacobi_precision`.

### Fixed
- Quire `value()` routes through `quire_resolve`, plus a fixpnt accumulator, with
  the mtl-math mirrors kept in sync (#34).

### Key result
As the operator's `nnz/row` grows, the quire super-accumulator separates from
naive/FMA (FMA ties naive throughout — only quire accumulates the whole row sum
exactly): inert on the 2-term tridiagonal row sum, but −44% to −48% on the 3D
7-point residual floor for posits, and −40% to −42% on the real well-conditioned
SuiteSparse `gr_30_30` Laplacian. Ill-conditioned high-`nnz/row` stiffness
matrices are convergence-limited, masking the effect until a preconditioner is
added. The number system is the dominant lever on sparse problems (posit⟨32,2⟩
edges out IEEE `float` at matched width on GS/SOR).
