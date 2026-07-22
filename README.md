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
GMRES, ...), and **multigrid**.

```
applications/
  stationary/jacobi_precision/   # Jacobi convergence across precisions
include/mtl/math/                # MTL5 trait specializations for Universal types
                                 #   (quire_accumulator.hpp: exact-dot-product bridge)
include/sw/mp_iterative/         # shared composition-layer headers
tests/
  stationary/                    # Jacobi smoke test across number types
  krylov/                        # CG with quire (exact dot product) accumulation
  multigrid/                     # (no tests yet)
docs/roadmap.md                  # milestones and known integration work
```

## License

MIT — see [LICENSE](LICENSE).
