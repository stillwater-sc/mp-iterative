#pragma once
// mp-iterative benchmarking -- error and convergence metrics.
//
// All metrics operate in double on std::vector<double> data. A low-precision
// solve's final solution is converted to double once and compared against the
// double reference operator/solution, so the numbers below measure the true
// error of the low-precision computation, not error re-expressed in low
// precision.
//
// Two error notions are kept deliberately distinct (issue #6): backward error
// (residual norm) can be tiny while forward error (distance to the true
// solution) is large, so both are reported.

#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include <mtl/mat/compressed2D.hpp>
#include "sw/mp_iterative/benchmark/reference_solve.hpp"  // spmv, dot

namespace sw::mp_iterative::benchmark {

/// Forward relative error ||x - x_ref||_2 / ||x_ref||_2.
inline double forward_relative_error(const std::vector<double>& x,
                                     const std::vector<double>& x_ref) {
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double d = x[i] - x_ref[i];
        num += d * d;
        den += x_ref[i] * x_ref[i];
    }
    const double dn = std::sqrt(den);
    return std::sqrt(num) / (dn > 0.0 ? dn : 1.0);
}

/// Backward error: 2-norm of the residual A*x - b.
inline double residual_2norm(const mtl::mat::compressed2D<double>& A,
                             const std::vector<double>& x,
                             const std::vector<double>& b) {
    std::vector<double> Ax = spmv(A, x);
    double s = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double r = Ax[i] - b[i];
        s += r * r;
    }
    return std::sqrt(s);
}

/// Backward error: inf-norm of the residual A*x - b.
inline double residual_infnorm(const mtl::mat::compressed2D<double>& A,
                               const std::vector<double>& x,
                               const std::vector<double>& b) {
    std::vector<double> Ax = spmv(A, x);
    double m = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i)
        m = std::max(m, std::abs(Ax[i] - b[i]));
    return m;
}

/// Stagnation floor: the best (minimum) residual reached over the history.
/// Once a low-precision iteration hits its rounding floor it plateaus/oscillates
/// around this value.
inline double stagnation_floor(const std::vector<double>& residual_history) {
    double m = std::numeric_limits<double>::infinity();
    for (double r : residual_history) m = std::min(m, r);
    return m;
}

/// Empirical linear convergence rate: geometric mean of consecutive residual
/// ratios over the segment where the iteration is still converging (residual
/// strictly decreasing and comfortably above the floor). Returns NaN if there
/// is no such segment (e.g. immediate stagnation).
inline double empirical_convergence_rate(const std::vector<double>& residual_history) {
    const double floor = stagnation_floor(residual_history);
    const double active = 4.0 * floor;   // "comfortably above the floor"
    double log_sum = 0.0;
    std::size_t count = 0;
    for (std::size_t k = 0; k + 1 < residual_history.size(); ++k) {
        const double a = residual_history[k];
        const double b = residual_history[k + 1];
        if (a > active && b > 0.0 && b < a) {
            log_sum += std::log(b / a);
            ++count;
        }
    }
    if (count == 0) return std::numeric_limits<double>::quiet_NaN();
    return std::exp(log_sum / static_cast<double>(count));
}

/// First iteration index (1-based) at which the residual drops below `tol`,
/// or 0 if it never does within the recorded history.
inline std::size_t iterations_to_tolerance(const std::vector<double>& residual_history,
                                           double tol) {
    for (std::size_t k = 0; k < residual_history.size(); ++k)
        if (residual_history[k] < tol) return k + 1;
    return 0;
}

} // namespace sw::mp_iterative::benchmark
