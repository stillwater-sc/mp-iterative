#pragma once
// mp-iterative benchmarking -- Matrix Market matrix source.
//
// Loads a real sparse operator from a Matrix Market file (e.g. a SuiteSparse
// matrix) so the accumulator sweep can run on high-nnz/row operators, beyond
// the analytic Poisson stencils. The parsing is MTL5's mtl::io::mm_read
// (handles coordinate real/integer, general and symmetric -- symmetric storage
// is expanded to full), so this header only adds the composition-layer piece:
// a value-type converter that rebuilds the loaded double operator in any
// Universal number type for the low-precision runs.
//
// SCOPE: the stationary smoothers (Jacobi/GS/SOR) converge only for
// diagonally dominant (Jacobi) or SPD (GS/SOR) operators, so the matrices used
// with the stationary benchmark must be SPD -- and for SPD the double CG
// reference (reference_solve.hpp) is valid. A general non-symmetric reference
// (GMRES/direct) is deferred to when the Krylov solvers get benchmark drivers;
// those are the solvers that handle general matrices.

#include <cstddef>
#include <string>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/io/matrix_market.hpp>

namespace sw::mp_iterative::benchmark {

/// Load a Matrix Market file as a compressed2D<double> (the reference operator).
inline mtl::mat::compressed2D<double> load_matrix_market(const std::string& path) {
    return mtl::io::mm_read<double>(path);
}

/// Rebuild a loaded double operator in value type T (element-wise conversion),
/// preserving the sparsity pattern. Used to run a loaded matrix in each
/// low-precision arithmetic without re-reading the file.
template <typename T>
mtl::mat::compressed2D<T> convert_matrix(const mtl::mat::compressed2D<double>& A) {
    const std::size_t n = A.num_rows();
    const auto& rp  = A.ref_major();
    const auto& ci  = A.ref_minor();
    const auto& dat = A.ref_data();

    mtl::mat::compressed2D<T> B(n, A.num_cols());
    mtl::mat::inserter<mtl::mat::compressed2D<T>> ins(B);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t k = rp[i]; k < rp[i + 1]; ++k)
            ins[i][ci[k]] << T(dat[k]);
    return B;
}

/// Maximum nonzeros in any row (the stencil width that governs how much a quire
/// super-accumulator can help). Reported alongside the sweep for context.
inline std::size_t max_nnz_per_row(const mtl::mat::compressed2D<double>& A) {
    const auto& rp = A.ref_major();
    std::size_t m = 0;
    for (std::size_t i = 0; i < A.num_rows(); ++i)
        m = std::max(m, static_cast<std::size_t>(rp[i + 1] - rp[i]));
    return m;
}

} // namespace sw::mp_iterative::benchmark
