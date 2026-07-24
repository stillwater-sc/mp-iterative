#pragma once
// mp-iterative benchmarking -- value-type-generic model problems.
//
// Reference problems for characterizing stationary (and later Krylov)
// iterative methods under mixed precision. Each generator returns a sparse
// mtl::mat::compressed2D<T>; the right-hand side used throughout the harness
// is b = ones. Model problems with a KNOWN spectral radius let the harness
// compare the empirical convergence rate against theory.
//
// Centralizes the generators previously duplicated across the demo
// applications (jacobi/gauss_seidel/sor_precision).

#include <cmath>
#include <cstddef>
#include <random>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>

namespace sw::mp_iterative::benchmark {

inline constexpr double pi = 3.14159265358979323846;

/// Standard 1D Poisson operator tridiag(-1, 2, -1), SPD. Jacobi spectral
/// radius rho_J = cos(pi/(n+1)) is close to 1, so it converges slowly and its
/// optimal SOR relaxation is well separated from 1 -- the discriminating case
/// for over-relaxation and accumulator studies.
template <typename T>
mtl::mat::compressed2D<T> poisson_1d(std::size_t n) {
    mtl::mat::compressed2D<T> A(n, n);
    mtl::mat::inserter<mtl::mat::compressed2D<T>> ins(A);
    for (std::size_t i = 0; i < n; ++i) {
        ins[i][i] << T(2);
        if (i > 0)     ins[i][i - 1] << T(-1);
        if (i + 1 < n) ins[i][i + 1] << T(-1);
    }
    return A;
}

/// Shifted 1D Poisson 4I - tridiag(1, 0, 1) = tridiag(-1, 4, -1). Strictly
/// diagonally dominant, so it converges fast; Jacobi spectral radius is
/// (1/2) cos(pi/(n+1)).
template <typename T>
mtl::mat::compressed2D<T> poisson_1d_shifted(std::size_t n) {
    mtl::mat::compressed2D<T> A(n, n);
    mtl::mat::inserter<mtl::mat::compressed2D<T>> ins(A);
    for (std::size_t i = 0; i < n; ++i) {
        ins[i][i] << T(4);
        if (i > 0)     ins[i][i - 1] << T(-1);
        if (i + 1 < n) ins[i][i + 1] << T(-1);
    }
    return A;
}

/// Symmetric, strictly diagonally dominant random tridiagonal system.
/// Deterministic given `seed`. Off-diagonals are drawn in [-1, 1] and the
/// diagonal is set to (row off-diagonal magnitude + 1), guaranteeing SPD and
/// convergence of Jacobi/GS/SOR. A stand-in for "diagonally dominant random"
/// until the SuiteSparse driver lands (roadmap milestone 1).
template <typename T>
mtl::mat::compressed2D<T> diagonally_dominant_random(std::size_t n, unsigned seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    // Draw the sub-diagonal first so A(i,i-1) == A(i-1,i) (symmetric).
    std::vector<double> sub(n, 0.0);
    for (std::size_t i = 1; i < n; ++i) sub[i] = dist(gen);

    mtl::mat::compressed2D<T> A(n, n);
    mtl::mat::inserter<mtl::mat::compressed2D<T>> ins(A);
    for (std::size_t i = 0; i < n; ++i) {
        double off = 0.0;
        if (i > 0)     off += std::abs(sub[i]);
        if (i + 1 < n) off += std::abs(sub[i + 1]);
        if (i > 0)     ins[i][i - 1] << T(sub[i]);
        ins[i][i] << T(off + 1.0);
        if (i + 1 < n) ins[i][i + 1] << T(sub[i + 1]);
    }
    return A;
}

/// Standard 2D Poisson operator, 5-point stencil on an m x m interior grid
/// (diagonal 4, -1 to the four axis neighbors). The matrix is (m*m) x (m*m)
/// with up to 5 nonzeros per row -- a denser row sum than the tridiagonal 1D
/// case, the point of interest for accumulator strategies. Unknown (i,j) maps
/// to linear index i*m + j; columns are inserted in ascending order.
template <typename T>
mtl::mat::compressed2D<T> poisson_2d(std::size_t m) {
    const std::size_t N = m * m;
    mtl::mat::compressed2D<T> A(N, N);
    mtl::mat::inserter<mtl::mat::compressed2D<T>> ins(A);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < m; ++j) {
            const std::size_t r = i * m + j;
            if (i > 0)     ins[r][r - m] << T(-1);   // up
            if (j > 0)     ins[r][r - 1] << T(-1);   // left
            ins[r][r] << T(4);                       // diagonal
            if (j + 1 < m) ins[r][r + 1] << T(-1);   // right
            if (i + 1 < m) ins[r][r + m] << T(-1);   // down
        }
    }
    return A;
}

/// Standard 3D Poisson operator, 7-point stencil on an m x m x m interior grid
/// (diagonal 6, -1 to the six axis neighbors). The matrix is (m^3) x (m^3) with
/// up to 7 nonzeros per row. Unknown (i,j,k) maps to (i*m + j)*m + k.
template <typename T>
mtl::mat::compressed2D<T> poisson_3d(std::size_t m) {
    const std::size_t N = m * m * m;
    const std::size_t m2 = m * m;
    mtl::mat::compressed2D<T> A(N, N);
    mtl::mat::inserter<mtl::mat::compressed2D<T>> ins(A);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < m; ++j) {
            for (std::size_t k = 0; k < m; ++k) {
                const std::size_t r = (i * m + j) * m + k;
                if (i > 0)     ins[r][r - m2] << T(-1);
                if (j > 0)     ins[r][r - m]  << T(-1);
                if (k > 0)     ins[r][r - 1]  << T(-1);
                ins[r][r] << T(6);
                if (k + 1 < m) ins[r][r + 1]  << T(-1);
                if (j + 1 < m) ins[r][r + m]  << T(-1);
                if (i + 1 < m) ins[r][r + m2] << T(-1);
            }
        }
    }
    return A;
}

// --- Problem selection: the standard Poisson operator in 1/2/3 dimensions,
// parameterized on the grid points per dimension (m). ---

enum class poisson_dim { d1, d2, d3 };

/// Matrix dimension (unknowns) for the d-dimensional Poisson operator: m^d.
inline std::size_t poisson_matrix_size(poisson_dim dim, std::size_t m) {
    switch (dim) {
        case poisson_dim::d1: return m;
        case poisson_dim::d2: return m * m;
        case poisson_dim::d3: return m * m * m;
    }
    return m;
}

/// Build the standard Poisson operator in the requested dimension.
template <typename T>
mtl::mat::compressed2D<T> make_poisson(poisson_dim dim, std::size_t m) {
    switch (dim) {
        case poisson_dim::d1: return poisson_1d<T>(m);
        case poisson_dim::d2: return poisson_2d<T>(m);
        case poisson_dim::d3: return poisson_3d<T>(m);
    }
    return poisson_1d<T>(m);
}

inline const char* poisson_dim_name(poisson_dim dim) {
    switch (dim) {
        case poisson_dim::d1: return "poisson1d";
        case poisson_dim::d2: return "poisson2d";
        case poisson_dim::d3: return "poisson3d";
    }
    return "poisson1d";
}

// --- Analytic convergence rates for the standard Poisson operator ---
// (spectral radius of the iteration matrix; smaller = faster). For the
// standard d-dimensional stencil these depend ONLY on the grid points per
// dimension m: rho_J = cos(pi/(m+1)) in 1D, 2D, and 3D alike (the per-axis
// contributions average out), so a single set of formulas serves every
// dimension.

/// Jacobi spectral radius: rho_J = cos(pi/(m+1)).
inline double poisson_jacobi_rate(std::size_t m) {
    return std::cos(pi / static_cast<double>(m + 1));
}

/// Gauss-Seidel spectral radius = rho_J^2 for a consistently ordered operator.
inline double poisson_gauss_seidel_rate(std::size_t m) {
    const double rho = poisson_jacobi_rate(m);
    return rho * rho;
}

/// Optimal SOR relaxation factor omega_opt = 2/(1 + sqrt(1 - rho_J^2)).
inline double poisson_omega_opt(std::size_t m) {
    return 2.0 / (1.0 + std::sin(pi / static_cast<double>(m + 1)));
}

/// SOR spectral radius at omega_opt is omega_opt - 1.
inline double poisson_sor_optimal_rate(std::size_t m) {
    return poisson_omega_opt(m) - 1.0;
}

} // namespace sw::mp_iterative::benchmark
