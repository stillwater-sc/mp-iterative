#pragma once
// mp-iterative benchmarking -- high-accuracy reference solution.
//
// Forward error ||x - x_ref|| / ||x_ref|| needs a trustworthy x_ref. For the
// SPD model problems we solve A x = b in double with a compact conjugate
// gradient to a tight tolerance. The reference lives in plain std::vector<double>
// (not a value-type vector): every low-precision run is compared against this
// same double operator and solution, so the forward-error metric is honest.
//
// CG is implemented inline (rather than via mtl::itl::cg) to keep this header
// self-contained and its correctness obvious -- it is the yardstick, so it
// should have no moving parts beyond double arithmetic.

#include <cmath>
#include <cstddef>
#include <vector>

#include <mtl/mat/compressed2D.hpp>

namespace sw::mp_iterative::benchmark {

/// y = A * x for a compressed2D<double> (raw CRS).
inline std::vector<double> spmv(const mtl::mat::compressed2D<double>& A,
                                const std::vector<double>& x) {
    const std::size_t n = A.num_rows();
    const auto& rp  = A.ref_major();
    const auto& ci  = A.ref_minor();
    const auto& dat = A.ref_data();
    std::vector<double> y(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        double s = 0.0;
        for (std::size_t k = rp[i]; k < rp[i + 1]; ++k)
            s += dat[k] * x[ci[k]];
        y[i] = s;
    }
    return y;
}

inline double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

/// Solve A x = b (SPD) in double via conjugate gradient, x0 = 0.
/// Returns x_ref accurate to `tol` in the 2-norm residual.
inline std::vector<double> reference_solution(const mtl::mat::compressed2D<double>& A,
                                              const std::vector<double>& b,
                                              double tol = 1e-14,
                                              std::size_t maxit = 100000) {
    const std::size_t n = A.num_rows();
    std::vector<double> x(n, 0.0), r = b, p = b;
    double rs_old = dot(r, r);
    const double bnorm = std::sqrt(dot(b, b));
    const double stop = tol * (bnorm > 0.0 ? bnorm : 1.0);

    for (std::size_t k = 0; k < maxit; ++k) {
        if (std::sqrt(rs_old) <= stop) break;
        std::vector<double> Ap = spmv(A, p);
        const double denom = dot(p, Ap);
        if (denom == 0.0) break;
        const double alpha = rs_old / denom;
        for (std::size_t i = 0; i < n; ++i) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }
        const double rs_new = dot(r, r);
        const double beta = rs_new / rs_old;
        for (std::size_t i = 0; i < n; ++i)
            p[i] = r[i] + beta * p[i];
        rs_old = rs_new;
    }
    return x;
}

} // namespace sw::mp_iterative::benchmark
