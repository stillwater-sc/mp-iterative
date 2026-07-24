# mp-iterative

[![CMake](https://github.com/stillwater-sc/mp-iterative/actions/workflows/cmake.yml/badge.svg)](https://github.com/stillwater-sc/mp-iterative/actions/workflows/cmake.yml)

**Mixed-precision iterative methods.** mp-iterative composes two header-only
libraries — [MTL5](https://github.com/stillwater-sc/mtl5) for linear algebra and
[Universal](https://github.com/stillwater-sc/universal) for parameterized number
systems — to explore stationary and Krylov iterative solvers under custom
arithmetic (half precision, posits, ...).

MTL5 deliberately has **no dependency on Universal**: it is the general
linear-algebra layer. mp-iterative is the integration layer where MTL5's
algorithms meet Universal's number types.

## Build

```bash
# Dependencies (MTL5 + Universal) are pulled automatically via FetchContent.
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Run the smoke test
ctest --test-dir build --output-on-failure

# Run the Jacobi precision demo (optional args: problem size, iterations)
./build/applications/stationary/jacobi_precision/jacobi_precision

# Measurement benchmarks are OFF by default; enable and run a sweep
cmake -B build -DMPITERATIVE_BUILD_BENCHMARKS=ON && cmake --build build -j
./build/benchmarks/stationary/benchmark_sor 48 3000   # summary CSV on stdout
```

Using local checkouts instead of fetching from GitHub:

```bash
cmake -B build \
  -DFETCHCONTENT_SOURCE_DIR_MTL5=../mtl5 \
  -DFETCHCONTENT_SOURCE_DIR_UNIVERSAL=../universal
```

## Layout

The repo is organized by iterative-method category, in order: **stationary**
methods (Jacobi, Gauss-Seidel, SOR), **Krylov** subspace methods (CG, BiCGSTAB,
GMRES, ...), and **multigrid**. Three parallel trees serve different purposes:
`tests/` answer "does it work", `applications/` demonstrate, `benchmarks/`
*measure*.

```
applications/
  stationary/{jacobi,gauss_seidel,sor}_precision/  # residual per number type x strategy
benchmarks/                        # measurement drivers (MPITERATIVE_BUILD_BENCHMARKS=ON)
  stationary/                      #   benchmark_{jacobi,gauss_seidel,sor}: CSV sweeps
include/mtl/math/                  # MTL5 trait specializations for Universal types
                                   #   (quire_accumulator.hpp: exact-dot-product bridge)
include/sw/mp_iterative/           # shared composition-layer headers
  accumulator_strategies.hpp       #   naive/fma/quire capability gating
  benchmark/                       #   harness: problems, reference solve, metrics, csv, runner
tests/
  stationary/                      # Jacobi/GS/SOR: quire + strategy tests
  krylov/                          # CG/BiCGSTAB/GMRES with quire accumulation
  benchmark/                       # unit tests for the harness utilities
  multigrid/                       # (no tests yet)
docs/roadmap.md                    # milestones and known integration work
docs/benchmarks-stationary.md      # first mixed-precision characterization report
```

## License

MIT — see [LICENSE](LICENSE).
