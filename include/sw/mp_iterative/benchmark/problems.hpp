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

// --- Analytic convergence rates for the standard 1D Poisson operator ---
// (spectral radius of the iteration matrix; smaller = faster).

/// Jacobi spectral radius for the standard 1D Poisson operator.
inline double poisson_1d_jacobi_rate(std::size_t n) {
    return std::cos(pi / static_cast<double>(n + 1));
}

/// Gauss-Seidel spectral radius = rho_J^2 for a consistently ordered operator.
inline double poisson_1d_gauss_seidel_rate(std::size_t n) {
    const double rho = poisson_1d_jacobi_rate(n);
    return rho * rho;
}

/// Optimal SOR relaxation factor omega_opt = 2/(1 + sqrt(1 - rho_J^2)).
inline double poisson_1d_omega_opt(std::size_t n) {
    return 2.0 / (1.0 + std::sin(pi / static_cast<double>(n + 1)));
}

/// SOR spectral radius at omega_opt is omega_opt - 1.
inline double poisson_1d_sor_optimal_rate(std::size_t n) {
    return poisson_1d_omega_opt(n) - 1.0;
}

} // namespace sw::mp_iterative::benchmark
