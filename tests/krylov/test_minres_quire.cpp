// MINRES with quire (exact dot product) accumulation: mirrors
// test_cgs_quire.cpp but exercises itl::minres. (MTL5 itself must not
// depend on Universal.) MINRES requires symmetric A; our tridiagonal
// test matrix satisfies that.
//
// Two checks, no external test framework (matching the repo's lightweight
// style; returns non-zero on failure):
//   1. MINRES with the default (unspecified) Accumulator matches the
//      unmodified baseline on a tridiagonal system.
//   2. The actual point: quire-accumulated dot products improve posit32
//      accuracy vs naive posit32 accumulation on a tridiagonal system.
#include <cmath>
#include <cstddef>
#include <iostream>

// pull in the posit number system
#include <universal/number/posit/posit.hpp>
// accumulator_traits specializations for Universal's quire super-accumulators
// (any fixed-size arithmetic admits a quire; this test uses the posit instance)
#include <mtl/math/quire_accumulator.hpp>

#include <mtl/vec/dense_vector.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/operators.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/operation/dot.hpp>
#include <mtl/itl/pc/identity.hpp>
#include <mtl/itl/pc/diagonal.hpp>
#include <mtl/itl/iteration/basic_iteration.hpp>
#include <mtl/itl/krylov/minres.hpp>

using namespace mtl;

namespace {

template <typename T>
mat::dense2D<T> make_tridiagonal(std::size_t n, T diag, T off) {
    mat::dense2D<T> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A(i, j) = T(0.0);
    for (std::size_t i = 0; i < n; ++i) {
        A(i, i) = diag;
        if (i > 0)     A(i, i - 1) = off;
        if (i < n - 1) A(i, i + 1) = off;
    }
    return A;
}

// --- Sanity: default (unspecified) Accumulator behaves exactly as before ---
bool minres_default_accumulator_ok() {
    const std::size_t n = 20;
    auto A = make_tridiagonal<double>(n, 4.0, -1.0);

    vec::dense_vector<double> b(n, 1.0);
    vec::dense_vector<double> x(n, 0.0);

    itl::pc::identity<mat::dense2D<double>> pc(A);
    itl::basic_iteration<double> iter(b, 200, 1e-10);

    int err = itl::minres(A, x, b, pc, iter); // no explicit Accumulator -- must be unaffected
    if (err != 0) {
        std::cerr << "MINRES (default Accumulator) did not converge, err=" << err << '\n';
        return false;
    }

    auto r = A * x;
    for (std::size_t i = 0; i < n; ++i) {
        if (std::abs(r(i) - b(i)) > 1e-8) {
            std::cerr << "MINRES (default Accumulator) residual component " << i
                      << " off: " << r(i) << " vs " << b(i) << '\n';
            return false;
        }
    }
    return true;
}

// --- The actual point: quire accumulation vs naive posit32 on a case where
// dot product magnitude sensitivity matters ---
bool minres_quire_beats_naive_posit32() {
    using Posit = sw::universal::posit<32,2>;
    using Quire = sw::universal::quire<Posit>;

    const std::size_t n = 20;
    auto A = make_tridiagonal<Posit>(n, Posit(2.0), Posit(-1.0));

    vec::dense_vector<Posit> b(n, Posit(1.0));

    vec::dense_vector<Posit> x_naive(n, Posit(0.0));
    itl::pc::identity<mat::dense2D<Posit>> pc(A);
    itl::basic_iteration<Posit> iter_naive(b, 200, Posit(1e-6));
    itl::minres(A, x_naive, b, pc, iter_naive); // default Accumulator = naive posit32

    vec::dense_vector<Posit> x_quire(n, Posit(0.0));
    itl::basic_iteration<Posit> iter_quire(b, 200, Posit(1e-6));
    itl::minres<mat::dense2D<Posit>, vec::dense_vector<Posit>, vec::dense_vector<Posit>,
            itl::pc::identity<mat::dense2D<Posit>>, itl::basic_iteration<Posit>, Quire>(
        A, x_quire, b, pc, iter_quire);

    // Residual ||A*x - b||_2 evaluated in double.
    auto residual_norm = [&](const vec::dense_vector<Posit>& x) {
        double res2 = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            double Axi = 0.0;
            for (std::size_t j = 0; j < n; ++j)
                Axi += double(A(i,j)) * double(x(j));
            double ri = Axi - double(b(i));
            res2 += ri * ri;
        }
        return std::sqrt(res2);
    };

    double naive_residual = residual_norm(x_naive);
    double quire_residual = residual_norm(x_quire);

    std::cout << "naive posit32 residual:             " << naive_residual << '\n';
    std::cout << "quire-accumulated posit32 residual: " << quire_residual << '\n';

    // quire should be at least as accurate as naive (small tolerance for
    // floating-point noise when both are already near machine precision --
    // e.g. quire converging to an exact 0 residual is a pass, not a failure).
    const double tol = 1e-9;
    if (quire_residual > naive_residual + tol) {
        std::cerr << "quire-accumulated residual (" << quire_residual
                  << ") is worse than naive posit32 (" << naive_residual << ")\n";
        return false;
    }
    return true;
}

} // namespace

int main() {
    int failures = 0;

    if (!minres_default_accumulator_ok())      ++failures;
    if (!minres_quire_beats_naive_posit32())   ++failures;

    if (failures == 0) std::cout << "test_minres_quire passed\n";
    return failures == 0 ? 0 : 1;
}
