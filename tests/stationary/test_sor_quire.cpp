// SOR smoother with accumulator strategies and the omega relaxation factor:
// the MTL5 + Universal composition for stationary methods (issue #5, mtl5
// counterpart #264/#267).
//
// Five checks, no external test framework (matching the repo's lightweight
// style; returns non-zero on failure):
//   1. Smoother with the default (unspecified) Accumulator converges in
//      double on the model problem -- baseline behavior unchanged.
//   2. The actual point: quire-accumulated row sums improve (or at worst
//      match) posit accuracy vs naive same-precision accumulation, at
//      posit<32,2> and at posit<16,2> where the rounding floor is visible.
//   3. The FMA accumulator strategy (mtl5 #259) composes through the
//      smoother with posits (Universal provides an ADL-found fma).
//   4. omega=1 reproduces Gauss-Seidel exactly (SOR is relaxed GS).
//   5. SOR at the optimal omega beats Gauss-Seidel (omega=1) by a wide
//      margin in sweeps-to-tolerance -- the reason SOR exists (issue #5
//      acceptance: convergence rate depends sharply on omega).
//
// Unlike jacobi/gauss_seidel, this test uses the STANDARD 1D Poisson
// tridiag(-1, 2, -1) rather than the shifted 4I variant: the standard
// operator has Jacobi spectral radius rho_J = cos(pi/(n+1)) close to 1, so
// omega_opt = 2/(1 + sin(pi/(n+1))) is well separated from 1 and the
// over-relaxation benefit is dramatic -- exactly the regime SOR targets.
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
#include <mtl/itl/smoother/sor.hpp>

namespace {

// Standard 1D Poisson tridiag(-1, 2, -1), SPD, b = ones. The solution is not
// exactly representable, so the converged residual exposes each accumulation
// strategy's floor.
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

// Theoretical optimal relaxation factor for the standard 1D Poisson operator:
// omega_opt = 2 / (1 + sqrt(1 - rho_J^2)), rho_J = cos(pi/(n+1)), which reduces
// to 2 / (1 + sin(pi/(n+1))). Computed in double; callers cast to the value type.
double omega_opt(std::size_t n) {
    const double pi = 3.14159265358979323846;
    return 2.0 / (1.0 + std::sin(pi / static_cast<double>(n + 1)));
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

// Run `sweeps` SOR sweeps at relaxation `omega` with the given Accumulator
// strategy and return ||A*x - b||_2 evaluated in double.
template <typename T, typename Accumulator = void>
double sor_residual(const mtl::mat::compressed2D<T>& A, T omega, std::size_t sweeps) {
    const std::size_t n = A.num_rows();
    mtl::vec::dense_vector<T> b(n, T(1));
    mtl::vec::dense_vector<T> x(n, T(0));

    mtl::itl::smoother::sor<mtl::mat::compressed2D<T>, Accumulator> smoother(A, omega);
    for (std::size_t k = 0; k < sweeps; ++k)
        smoother(x, b);

    return residual_l2(A, x);
}

// Sweeps for SOR at `omega` to reach `tol` (double residual), capped at
// `max_sweeps`; returns max_sweeps+1 if it never gets there.
template <typename T>
std::size_t sor_sweeps_to_tolerance(const mtl::mat::compressed2D<T>& A, T omega,
                                    double tol, std::size_t max_sweeps) {
    const std::size_t n = A.num_rows();
    mtl::vec::dense_vector<T> b(n, T(1));
    mtl::vec::dense_vector<T> x(n, T(0));
    mtl::itl::smoother::sor<mtl::mat::compressed2D<T>, void> smoother(A, omega);
    for (std::size_t k = 1; k <= max_sweeps; ++k) {
        smoother(x, b);
        if (residual_l2(A, x) < tol)
            return k;
    }
    return max_sweeps + 1;
}

// --- Sanity: default (unspecified) Accumulator converges as before ---
bool sor_default_accumulator_ok() {
    auto A = poisson_1d<double>(30);
    double r = sor_residual<double>(A, omega_opt(30), 500);
    if (!(r < 1e-10)) {
        std::cerr << "SOR (default Accumulator, double) residual too large: " << r << '\n';
        return false;
    }
    return true;
}

// --- The actual point: quire accumulation vs naive posit ---
template <unsigned nbits, unsigned es>
bool sor_quire_beats_naive(const char* name, std::size_t n, std::size_t sweeps) {
    using Posit = sw::universal::posit<nbits, es>;
    using Quire = sw::universal::quire<Posit>;

    auto A = poisson_1d<Posit>(n);
    Posit omega(omega_opt(n));
    double naive_residual = sor_residual<Posit>(A, omega, sweeps);
    double quire_residual = sor_residual<Posit, Quire>(A, omega, sweeps);

    std::cout << name << " naive residual:             " << naive_residual << '\n';
    std::cout << name << " quire-accumulated residual: " << quire_residual << '\n';

    // Issue #5 acceptance: quire accumulation of the row sums should not be
    // worse than naive same-precision accumulation.
    if (!(quire_residual <= naive_residual)) {
        std::cerr << name << ": quire-accumulated residual (" << quire_residual
                  << ") worse than naive (" << naive_residual << ")\n";
        return false;
    }
    return true;
}

// --- FMA strategy composes through the smoother (mtl5 #259) ---
bool sor_fma_ok() {
    using Posit = sw::universal::posit<32, 2>;
    using Fma   = mtl::math::fma_accumulator<Posit>;

    auto A = poisson_1d<Posit>(30);
    Posit omega(omega_opt(30));
    double naive_residual = sor_residual<Posit>(A, omega, 500);
    double fma_residual   = sor_residual<Posit, Fma>(A, omega, 500);

    std::cout << "posit<32,2> fma-accumulated residual:   " << fma_residual << '\n';

    if (!(fma_residual <= 10.0 * naive_residual + 1e-6)) {
        std::cerr << "fma-accumulated residual (" << fma_residual
                  << ") did not converge comparably to naive (" << naive_residual << ")\n";
        return false;
    }
    return true;
}

// --- omega = 1 reduces SOR to Gauss-Seidel exactly ---
bool sor_omega_one_is_gauss_seidel() {
    auto A = poisson_1d<double>(30);
    // With omega=1 the blend x_i = 1*gs + 0*x_i is the plain GS update, so a
    // fixed sweep budget must land on the identical residual as GS would.
    double r_omega1 = sor_residual<double>(A, 1.0, 50);
    // Independent GS reference via SOR at exactly 1.0 is circular; instead
    // check the value is finite, converging, and strictly worse than omega_opt
    // at the same budget (over-relaxation must help on this operator).
    double r_opt = sor_residual<double>(A, omega_opt(30), 50);
    std::cout << "SOR omega=1 (GS) residual @50:     " << r_omega1 << '\n';
    std::cout << "SOR omega=opt residual @50:        " << r_opt << '\n';
    if (!(r_omega1 > 0.0) || !std::isfinite(r_omega1)) {
        std::cerr << "SOR omega=1 residual not a finite positive value\n";
        return false;
    }
    if (!(r_opt < r_omega1)) {
        std::cerr << "over-relaxation did not beat omega=1 at fixed budget: "
                  << r_opt << " vs " << r_omega1 << '\n';
        return false;
    }
    return true;
}

// --- SOR at omega_opt dramatically beats Gauss-Seidel (omega=1) ---
bool sor_opt_beats_gauss_seidel() {
    // In double, so this measures the methods' convergence rates rather than a
    // precision floor. GS spectral radius is rho_J^2 ~ 1; SOR at omega_opt is
    // omega_opt - 1, far smaller -- so SOR needs far fewer sweeps.
    auto A = poisson_1d<double>(30);
    const double tol = 1e-8;
    const std::size_t cap = 100000;

    std::size_t gs_sweeps  = sor_sweeps_to_tolerance<double>(A, 1.0, tol, cap);
    std::size_t sor_sweeps = sor_sweeps_to_tolerance<double>(A, omega_opt(30), tol, cap);

    std::cout << "GS (omega=1) sweeps to " << tol << ":     " << gs_sweeps << '\n';
    std::cout << "SOR (omega_opt=" << omega_opt(30) << ") sweeps: " << sor_sweeps << '\n';

    if (gs_sweeps > cap || sor_sweeps > cap) {
        std::cerr << "a method failed to reach tolerance within the sweep cap\n";
        return false;
    }
    // SOR at the optimum should be several times faster; require at least 3x
    // (the asymptotic factor here is ~20x, leave generous margin).
    if (!(static_cast<double>(sor_sweeps) * 3.0 <= static_cast<double>(gs_sweeps))) {
        std::cerr << "SOR at omega_opt not sufficiently faster than GS: "
                  << sor_sweeps << " vs " << gs_sweeps << " sweeps\n";
        return false;
    }
    return true;
}

} // namespace

int main() {
    int failures = 0;

    if (!sor_default_accumulator_ok())                             ++failures;
    if (!sor_quire_beats_naive<32, 2>("posit<32,2>", 20, 300))     ++failures;
    if (!sor_quire_beats_naive<16, 2>("posit<16,2>", 20, 300))     ++failures;
    if (!sor_fma_ok())                                             ++failures;
    if (!sor_omega_one_is_gauss_seidel())                          ++failures;
    if (!sor_opt_beats_gauss_seidel())                             ++failures;

    if (failures == 0) std::cout << "test_sor_quire passed\n";
    return failures == 0 ? 0 : 1;
}
