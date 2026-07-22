# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## Project Overview

mp-iterative is the **integration layer** for mixed-precision iterative methods.
It composes two header-only sister libraries:

- [MTL5](https://github.com/stillwater-sc/mtl5) — C++20 linear algebra.
- [Universal](https://github.com/stillwater-sc/universal) — parameterized number
  systems (`cfloat`, `posit`, ...).

**Architectural rule:** MTL5 is the general linear-algebra layer and MUST NOT
depend on Universal. All MTL5 + Universal coupling lives here in mp-iterative.

## Build Commands

```bash
# Dependencies are pulled automatically via FetchContent.
cmake -B build -DCMAKE_BUILD_TYPE=Release -Wno-dev
cmake --build build -j
ctest --test-dir build --output-on-failure

# Use local sister checkouts instead of fetching from GitHub:
cmake -B build -DFETCHCONTENT_SOURCE_DIR_MTL5=../mtl5 \
               -DFETCHCONTENT_SOURCE_DIR_UNIVERSAL=../universal
```

## Architecture

- Header-only composition under `include/sw/mp_iterative/`. Namespace:
  `sw::mp_iterative`.
- CMake: INTERFACE library `sw::mp_iterative` linking MTL5 + Universal. Options:
  `MPITERATIVE_BUILD_APPLICATIONS`, `MPITERATIVE_BUILD_TESTS`.
- `applications/` and `tests/` are organized by iterative-method category, in
  this order: `stationary/` (Jacobi, Gauss-Seidel, SOR), `krylov/` (CG,
  BiCGSTAB, GMRES, ...), `multigrid/`.
- `applications/` — demonstration programs (each its own CMakeLists).
- `tests/` — lightweight self-checking executables (no external framework);
  register with `mpiterative_add_test`.
- `docs/roadmap.md` — milestones and known integration work.

## Conventions

- C++20, header-only. Match the sister repos (mtl5, mp-spice) for style and
  CMake structure.
- Conventional Commits. Feature branches + PRs to `main`; CI must pass.
- Never commit build artifacts or downloaded matrices.
