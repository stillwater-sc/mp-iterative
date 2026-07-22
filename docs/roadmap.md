# mp-iterative roadmap

## Milestone 0: composition layer bootstrapped (done)

- CMake scaffold replicated from [mp-spice](https://github.com/stillwater-sc/mp-spice):
  INTERFACE library `sw::mp_iterative`, find_package → FetchContent fallback for
  MTL5 + Universal, config-package install, CI matrix (MSVC/GCC/Clang/AppleClang).
- Smoke test: Jacobi iteration on a diagonally dominant sparse system in
  `double`, `float`, `cfloat<16,5>`, and `posit<16,2>`.
- Demo application: `jacobi_precision` residual table across number types.

## Milestone 1: value-type-generic iterative solver library

- Stationary methods (Jacobi, Gauss-Seidel, SOR) and Krylov methods (CG,
  BiCGSTAB, GMRES) as templates over the number type, under
  `include/sw/mp_iterative/`, reusing MTL5 kernels where available.
- SuiteSparse matrix-market driver (mirror mp-spice's `scripts/fetch_matrices.sh`
  + loader) for realistic test problems.

## Milestone 2: mixed-precision studies

- Precision of the residual/orthogonalization vs the preconditioner apply:
  where can low precision live without stalling convergence?
- Mixed-precision restarted GMRES and iterative refinement around low-precision
  inner solves (connect with the MTL5 `iterative_refine` core used by mp-spice).
- Quire (exact dot product) accumulation in CG/GMRES orthogonalization.
