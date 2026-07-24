# mp-iterative roadmap

## Milestone 0: composition layer bootstrapped (done)

- CMake scaffold replicated from [mp-spice](https://github.com/stillwater-sc/mp-spice):
  INTERFACE library `sw::mp_iterative`, find_package → FetchContent fallback for
  MTL5 + Universal, config-package install, CI matrix (MSVC/GCC/Clang/AppleClang).
- Smoke test: Jacobi iteration on a diagonally dominant sparse system in
  `double`, `float`, `cfloat<16,5>`, and `posit<16,2>`.
- Demo application: `jacobi_precision` residual table across number types.
- Repo organized by iterative-method category: `stationary/` (Jacobi,
  Gauss-Seidel, SOR), `krylov/` (CG, BiCGSTAB, GMRES, ...), `multigrid/`.
- `tests/krylov/test_cg_quire.cpp` migrated from mtl5: CG with quire (exact
  dot product) accumulation of rho/pAp vs naive posit32 (the MTL5 + Universal
  coupling that must not live in MTL5 itself).

## Milestone 1: value-type-generic iterative solver library

- Stationary methods (Jacobi, Gauss-Seidel, SOR) and Krylov methods (CG,
  BiCGSTAB, GMRES) as templates over the number type, under
  `include/sw/mp_iterative/`, reusing MTL5 kernels where available.
- SuiteSparse matrix-market driver (`scripts/fetch_matrices.sh` + MTL5's
  `mm_read`, wired into the benchmark harness as `--matrix PATH`) for realistic
  SPD test problems. Finding: on the well-conditioned `gr_30_30` Laplacian quire
  lowers the posit residual floor −40%/−42%, confirming the 2D/3D trend on a
  real operator; ill-conditioned stiffness matrices are convergence-limited and
  need a preconditioner (milestone 2) to expose the accumulator effect.
- Benchmarking harness (`benchmarks/`, `MPITERATIVE_BUILD_BENCHMARKS`): forward/
  backward error, convergence rate vs analytic spectral radius, and stagnation
  floor across number-type x accumulator-strategy sweeps on the standard Poisson
  operator in 1D/2D/3D. Characterization in
  [docs/benchmarks-stationary.md](benchmarks-stationary.md). Key result: the
  accumulator is inert on tridiagonal (2-term) row sums but the quire
  super-accumulator separates from naive/FMA as the stencil widens (−44% to −48%
  on the 3D 7-point residual floor for posits). SuiteSparse loader (higher
  nnz/row) is the remaining follow-up.

## Milestone 2: mixed-precision studies

- Precision of the residual/orthogonalization vs the preconditioner apply:
  where can low precision live without stalling convergence?
- Mixed-precision restarted GMRES and iterative refinement around low-precision
  inner solves (connect with the MTL5 `iterative_refine` core used by mp-spice).
- Quire (exact dot product) accumulation in CG/GMRES orthogonalization.
