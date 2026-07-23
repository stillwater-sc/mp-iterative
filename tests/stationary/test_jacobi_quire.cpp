// Jacobi smoother with accumulator strategies: the MTL5 + Universal
// composition for stationary methods (issue #3, mtl5 counterpart #262).
//
// Three checks, no external test framework (matching the repo's lightweight
// style; returns non-zero on failure):
//   1. Smoother with the default (unspecified) Accumulator converges in
//      double on the model problem -- baseline behavior unchanged.
//   2. The actual point: quire-accumulated row sums improve (or at worst
//      match) posit accuracy vs naive same-precision accumulation, at
//      posit<32,2> and at posit<16,2> where the rounding floor is visible.
//   3. The FMA accumulator strategy (mtl5 #259) composes through the
//      smoother with posits (Universal provides an ADL-found fma).
#include <cmath>
#include <cstddef>
#include <iostream>

// pull in the posit number system
#include <universal/number/posit/posit.hpp>
// composition-layer accumulator_traits specialization for posit + quire
#include <mtl/math/quire_accumulator.hpp>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/itl/smoother/jacobi.hpp>

namespace {

// Shifted 1D Poisson (4I - tridiag(1, 0, 1)), strictly diagonally dominant,
// b = ones: Jacobi converges, and the solution is not exactly representable,
// so the converged residual exposes each accumulation strategy's floor.
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

// Run `sweeps` Jacobi sweeps with the given Accumulator strategy and return
// ||A*x - b||_2 evaluated in double.
template <typename T, typename Accumulator = void>
double jacobi_residual(const mtl::mat::compressed2D<T>& A, std::size_t sweeps) {
    const std::size_t n = A.num_rows();
    mtl::vec::dense_vector<T> b(n, T(1));
    mtl::vec::dense_vector<T> x(n, T(0));

    mtl::itl::smoother::jacobi<mtl::mat::compressed2D<T>, Accumulator> smoother(A);
    for (std::size_t k = 0; k < sweeps; ++k)
        smoother(x, b);

    const auto& rp  = A.ref_major();
    const auto& ci  = A.ref_minor();
    const auto& dat = A.ref_data();
    double res2 = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double Axi = 0.0;
        for (std::size_t k = rp[i]; k < rp[i + 1]; ++k)
            Axi += static_cast<double>(dat[k]) * static_cast<double>(x(static_cast<int>(ci[k])));
        double ri = Axi - 1.0;
        res2 += ri * ri;
    }
    return std::sqrt(res2);
}

// --- Sanity: default (unspecified) Accumulator converges as before ---
bool jacobi_default_accumulator_ok() {
    auto A = poisson_shifted<double>(20);
    double r = jacobi_residual<double>(A, 100);
    if (!(r < 1e-10)) {
        std::cerr << "Jacobi (default Accumulator, double) residual too large: " << r << '\n';
        return false;
    }
    return true;
}

// --- The actual point: quire accumulation vs naive posit ---
template <unsigned nbits, unsigned es>
bool jacobi_quire_beats_naive(const char* name, std::size_t n, std::size_t sweeps) {
    using Posit = sw::universal::posit<nbits, es>;
    using Quire = sw::universal::quire<Posit>;

    auto A = poisson_shifted<Posit>(n);
    double naive_residual = jacobi_residual<Posit>(A, sweeps);
    double quire_residual = jacobi_residual<Posit, Quire>(A, sweeps);

    std::cout << name << " naive residual:             " << naive_residual << '\n';
    std::cout << name << " quire-accumulated residual: " << quire_residual << '\n';

    // The claim under test (issue #3 acceptance): quire accumulation of the
    // row sums should not be worse than naive same-precision accumulation.
    if (!(quire_residual <= naive_residual)) {
        std::cerr << name << ": quire-accumulated residual (" << quire_residual
                  << ") worse than naive (" << naive_residual << ")\n";
        return false;
    }
    return true;
}

// --- FMA strategy composes through the smoother (mtl5 #259) ---
bool jacobi_fma_ok() {
    using Posit = sw::universal::posit<32, 2>;
    using Fma   = mtl::math::fma_accumulator<Posit>;

    auto A = poisson_shifted<Posit>(20);
    double naive_residual = jacobi_residual<Posit>(A, 100);
    double fma_residual   = jacobi_residual<Posit, Fma>(A, 100);

    std::cout << "posit<32,2> fma-accumulated residual:   " << fma_residual << '\n';

    // FMA halves the rounding steps but is not exact: require convergence to
    // the same order as naive, not strict improvement.
    if (!(fma_residual <= 10.0 * naive_residual + 1e-6)) {
        std::cerr << "fma-accumulated residual (" << fma_residual
                  << ") did not converge comparably to naive (" << naive_residual << ")\n";
        return false;
    }
    return true;
}

} // namespace

int main() {
    int failures = 0;

    if (!jacobi_default_accumulator_ok())                              ++failures;
    if (!jacobi_quire_beats_naive<32, 2>("posit<32,2>", 20, 100))      ++failures;
    if (!jacobi_quire_beats_naive<16, 2>("posit<16,2>", 20, 100))      ++failures;
    if (!jacobi_fma_ok())                                              ++failures;

    if (failures == 0) std::cout << "test_jacobi_quire passed\n";
    return failures == 0 ? 0 : 1;
}
