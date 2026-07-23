// Gauss-Seidel smoother with accumulator strategies: the MTL5 + Universal
// composition for stationary methods (issue #4, mtl5 counterpart #263/#266).
//
// Four checks, no external test framework (matching the repo's lightweight
// style; returns non-zero on failure):
//   1. Smoother with the default (unspecified) Accumulator converges in
//      double on the model problem -- baseline behavior unchanged.
//   2. The actual point: quire-accumulated row sums improve (or at worst
//      match) posit accuracy vs naive same-precision accumulation, at
//      posit<32,2> and at posit<16,2> where the rounding floor is visible.
//   3. The FMA accumulator strategy (mtl5 #259) composes through the
//      smoother with posits (Universal provides an ADL-found fma).
//   4. Intra-sweep error propagation (issue #4 acceptance): Gauss-Seidel
//      reaches a fixed tolerance in fewer sweeps than Jacobi on the model
//      Poisson problem -- the in-place sweep dampens, not amplifies, the
//      per-sweep error at matched precision.
#include <cmath>
#include <cstddef>
#include <iostream>

// pull in the posit number system
#include <universal/number/posit/posit.hpp>
// composition-layer accumulator_traits specializations for Universal's quire
// super-accumulators (currently the posit instance; quires exist for any
// fixed-size arithmetic)
#include <mtl/math/quire_accumulator.hpp>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/itl/smoother/jacobi.hpp>
#include <mtl/itl/smoother/gauss_seidel.hpp>

namespace {

// Shifted 1D Poisson (4I - tridiag(1, 0, 1)), strictly diagonally dominant,
// b = ones: both Jacobi and Gauss-Seidel converge, and the solution is not
// exactly representable, so the converged residual exposes each accumulation
// strategy's floor.
template <typename T>
mtl::mat::compressed2D<T> poisson_shifted(std::size_t n) {
    mtl::mat::compressed2D<T> A(n, n);
    mtl::mat::inserter<mtl::mat::compressed2D<T>> ins(A);
    for (std::size_t i = 0; i < n; ++i) {
        ins[i][i] << T(4);
        if (i > 0)     ins[i][i - 1] << T(-1);
        if (i + 1 < n) ins[i][i + 1] << T(-1);
    }
    return A;
}

template <typename T>
double residual_l2(const mtl::mat::compressed2D<T>& A, const mtl::vec::dense_vector<T>& x) {
    const std::size_t n   = A.num_rows();
    const auto& rp  = A.ref_major();
    const auto& ci  = A.ref_minor();
    const auto& dat = A.ref_data();
    double res2 = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double Axi = 0.0;
        for (std::size_t k = rp[i]; k < rp[i + 1]; ++k)
            Axi += static_cast<double>(dat[k]) * static_cast<double>(x(static_cast<int>(ci[k])));
        double ri = Axi - 1.0;  // b = ones
        res2 += ri * ri;
    }
    return std::sqrt(res2);
}

// Run `sweeps` Gauss-Seidel sweeps with the given Accumulator strategy and
// return ||A*x - b||_2 evaluated in double.
template <typename T, typename Accumulator = void>
double gauss_seidel_residual(const mtl::mat::compressed2D<T>& A, std::size_t sweeps) {
    const std::size_t n = A.num_rows();
    mtl::vec::dense_vector<T> b(n, T(1));
    mtl::vec::dense_vector<T> x(n, T(0));

    mtl::itl::smoother::gauss_seidel<mtl::mat::compressed2D<T>, Accumulator> smoother(A);
    for (std::size_t k = 0; k < sweeps; ++k)
        smoother(x, b);

    return residual_l2(A, x);
}

// --- Sanity: default (unspecified) Accumulator converges as before ---
bool gauss_seidel_default_accumulator_ok() {
    auto A = poisson_shifted<double>(20);
    double r = gauss_seidel_residual<double>(A, 100);
    if (!(r < 1e-10)) {
        std::cerr << "Gauss-Seidel (default Accumulator, double) residual too large: " << r << '\n';
        return false;
    }
    return true;
}

// --- The actual point: quire accumulation vs naive posit ---
template <unsigned nbits, unsigned es>
bool gauss_seidel_quire_beats_naive(const char* name, std::size_t n, std::size_t sweeps) {
    using Posit = sw::universal::posit<nbits, es>;
    using Quire = sw::universal::quire<Posit>;

    auto A = poisson_shifted<Posit>(n);
    double naive_residual = gauss_seidel_residual<Posit>(A, sweeps);
    double quire_residual = gauss_seidel_residual<Posit, Quire>(A, sweeps);

    std::cout << name << " naive residual:             " << naive_residual << '\n';
    std::cout << name << " quire-accumulated residual: " << quire_residual << '\n';

    // Issue #4 acceptance: quire accumulation of the row sums should not be
    // worse than naive same-precision accumulation.
    if (!(quire_residual <= naive_residual)) {
        std::cerr << name << ": quire-accumulated residual (" << quire_residual
                  << ") worse than naive (" << naive_residual << ")\n";
        return false;
    }
    return true;
}

// --- FMA strategy composes through the smoother (mtl5 #259) ---
bool gauss_seidel_fma_ok() {
    using Posit = sw::universal::posit<32, 2>;
    using Fma   = mtl::math::fma_accumulator<Posit>;

    auto A = poisson_shifted<Posit>(20);
    double naive_residual = gauss_seidel_residual<Posit>(A, 100);
    double fma_residual   = gauss_seidel_residual<Posit, Fma>(A, 100);

    std::cout << "posit<32,2> fma-accumulated residual:   " << fma_residual << '\n';

    if (!(fma_residual <= 10.0 * naive_residual + 1e-6)) {
        std::cerr << "fma-accumulated residual (" << fma_residual
                  << ") did not converge comparably to naive (" << naive_residual << ")\n";
        return false;
    }
    return true;
}

// Sweeps to reach `tol` (in double residual), capped at `max_sweeps`; returns
// max_sweeps+1 if it never gets there. Templated on the smoother so Jacobi and
// Gauss-Seidel share one measurement.
template <template <typename, typename> class Smoother, typename T>
std::size_t sweeps_to_tolerance(const mtl::mat::compressed2D<T>& A,
                                double tol, std::size_t max_sweeps) {
    const std::size_t n = A.num_rows();
    mtl::vec::dense_vector<T> b(n, T(1));
    mtl::vec::dense_vector<T> x(n, T(0));
    Smoother<mtl::mat::compressed2D<T>, void> smoother(A);
    for (std::size_t k = 1; k <= max_sweeps; ++k) {
        smoother(x, b);
        if (residual_l2(A, x) < tol)
            return k;
    }
    return max_sweeps + 1;
}

// --- Intra-sweep error propagation: GS converges faster than Jacobi ---
bool gauss_seidel_faster_than_jacobi() {
    // In double, so the comparison measures the methods' convergence rates,
    // not a precision floor. On the model Poisson problem GS's asymptotic
    // rate is the square of Jacobi's, so it needs roughly half the sweeps.
    auto A = poisson_shifted<double>(50);
    const double tol = 1e-8;
    const std::size_t cap = 100000;

    std::size_t jac_sweeps = sweeps_to_tolerance<mtl::itl::smoother::jacobi>(A, tol, cap);
    std::size_t gs_sweeps  = sweeps_to_tolerance<mtl::itl::smoother::gauss_seidel>(A, tol, cap);

    std::cout << "Jacobi sweeps to " << tol << ":       " << jac_sweeps << '\n';
    std::cout << "Gauss-Seidel sweeps to " << tol << ": " << gs_sweeps << '\n';

    if (jac_sweeps > cap || gs_sweeps > cap) {
        std::cerr << "a method failed to reach tolerance within the sweep cap\n";
        return false;
    }
    // GS should be clearly faster; require at least a 1.5x sweep reduction
    // (the asymptotic factor is ~2x, leave margin for the transient).
    if (!(static_cast<double>(gs_sweeps) * 1.5 <= static_cast<double>(jac_sweeps))) {
        std::cerr << "Gauss-Seidel not sufficiently faster than Jacobi: "
                  << gs_sweeps << " vs " << jac_sweeps << " sweeps\n";
        return false;
    }
    return true;
}

} // namespace

int main() {
    int failures = 0;

    if (!gauss_seidel_default_accumulator_ok())                             ++failures;
    if (!gauss_seidel_quire_beats_naive<32, 2>("posit<32,2>", 20, 100))     ++failures;
    if (!gauss_seidel_quire_beats_naive<16, 2>("posit<16,2>", 20, 100))     ++failures;
    if (!gauss_seidel_fma_ok())                                             ++failures;
    if (!gauss_seidel_faster_than_jacobi())                                 ++failures;

    if (failures == 0) std::cout << "test_gauss_seidel_quire passed\n";
    return failures == 0 ? 0 : 1;
}
