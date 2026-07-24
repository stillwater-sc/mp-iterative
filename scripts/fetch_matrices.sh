#!/usr/bin/env bash
# Fetch SPD matrices from the SuiteSparse Matrix Collection (https://sparse.tamu.edu/)
# into ./data for the benchmarking harness. Matrices are NOT committed to the repo
# (see .gitignore); only the tiny synthetic fixture under tests/benchmark/data is.
#
# The stationary smoothers (Jacobi/GS/SOR) converge only for SPD (GS/SOR) or
# diagonally dominant (Jacobi) operators, so only SPD matrices are fetched here.
#
# Usage:
#   scripts/fetch_matrices.sh              # small SPD demos (nos3, gr_30_30)
#   scripts/fetch_matrices.sh all          # also the higher-nnz/row bcsstk14
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="$ROOT/data"
mkdir -p "$DATA"

BASE="https://suitesparse-collection-website.herokuapp.com/MM"

fetch() {
    local group="$1" name="$2"
    if [ -f "$DATA/$name/$name.mtx" ]; then
        echo "already have $name"
        return
    fi
    echo "downloading $group/$name ..."
    curl -L --fail -o "$DATA/$name.tar.gz" "$BASE/$group/$name.tar.gz"
    tar -xzf "$DATA/$name.tar.gz" -C "$DATA"
    rm -f "$DATA/$name.tar.gz"
    echo "extracted to $DATA/$name/$name.mtx"
}

# Small SPD demos (low-to-moderate nnz/row).
fetch HB   nos3        # 960x960, ~16 nnz/row, FE biharmonic (SPD)
fetch HB   gr_30_30    # 900x900,  ~8 nnz/row, 9-point Laplacian (SPD)

# Higher nnz/row SPD stiffness matrix (opt-in): the regime where the quire
# super-accumulator is expected to help most.
if [ "${1:-}" = "all" ]; then
    fetch HB bcsstk14   # 1806x1806, ~35 nnz/row, structural stiffness (SPD)
fi

echo
echo "Run the accumulator sweep on a fetched matrix, e.g.:"
echo "  ./build/benchmarks/stationary/benchmark_gauss_seidel --matrix $DATA/nos3/nos3.mtx"
