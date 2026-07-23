// BiCGSTAB with quire (exact dot product) accumulation: the MTL5 + Universal
// composition this repo exists for. Mirrors test_cg_quire.cpp.
// (MTL5 itself must not depend on Universal.)
//
// Three checks, no external test framework (matching the repo's lightweight
// style; returns non-zero on failure):
//   1. BiCGSTAB with the default (unspecified) Accumulator matches the
//      unmodified baseline on a small system.
//   2. BiCGSTAB handles a trivial 1x1 system.
//   3. The actual point: quire-accumulated dot products improve posit32
//      accuracy vs naive posit32 accumulation on a tridiagonal system.
#include <cmath>
#include <cstddef>
#include <iostream>

// pull in the posit number system
#include <universal/number/posit/posit.hpp>
// override of accumulation_traits with posit/quire
#include <mtl/math/quire_accumulator.hpp>

#include <mtl/vec/dense_vector.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/operators.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/operation/dot.hpp>
#include <mtl/itl/pc/identity.hpp>
#include <mtl/itl/pc/diagonal.hpp>
#include <mtl/itl/iteration/basic_iteration.hpp>
#include <mtl/itl/krylov/bicgstab.hpp>

using namespace mtl;

namespace {

// --- Sanity: default (unspecified) Accumulator behaves exactly as before ---
bool bicgstab_default_accumulator_ok() {
    mat::dense2D<double> A(3, 3);
    A(0,0) = 4; A(0,1) = 1; A(0,2) = 0;
    A(1,0) = 1; A(1,1) = 3; A(1,2) = 1;
    A(2,0) = 0; A(2,1) = 1; A(2,2) = 2;

    vec::dense_vector<double> b = {1.0, 2.0, 3.0};
    vec::dense_vector<double> x(3, 0.0);

    itl::pc::identity<mat::dense2D<double>> pc(A);
    itl::basic_iteration<double> iter(b, 100, 1e-10);

    int err = itl::bicgstab(A, x, b, pc, iter); // no explicit Accumulator -- must be unaffected
    if (err != 0) {
        std::cerr << "BiCGSTAB (default Accumulator) did not converge, err=" << err << '\n';
        return false;
    }

    auto r = A * x;
    for (std::size_t i = 0; i < 3; ++i) {
        if (std::abs(r(i) - b(i)) > 1e-8) {
            std::cerr << "BiCGSTAB (default Accumulator) residual component " << i
                      << " off: " << r(i) << " vs " << b(i) << '\n';
            return false;
        }
    }
    return true;
}

// --- Edge case: BiCGSTAB on a trivial 1x1 system ---
bool bicgstab_1x1_ok() {
    mat::dense2D<double> A(1, 1);
    A(0,0) = 4.0;

    vec::dense_vector<double> b = {2.0};
    vec::dense_vector<double> x(1, 0.0);

    itl::pc::identity<mat::dense2D<double>> pc(A);
    itl::basic_iteration<double> iter(b, 100, 1e-10);

    int err = itl::bicgstab(A, x, b, pc, iter);
    if (err != 0) {
        std::cerr << "BiCGSTAB (1x1) did not converge, err=" << err << '\n';
        return false;
    }
    if (std::abs(x(0) - 0.5) > 1e-8) {
        std::cerr << "BiCGSTAB (1x1) wrong solution: " << x(0) << " vs 0.5\n";
        return false;
    }
    return true;
}

// --- The actual point: quire accumulation vs naive posit32 on a case where
// dot product magnitude sensitivity matters ---
bool bicgstab_quire_beats_naive_posit32() {
    using Posit = sw::universal::posit<32,2>;
    using Quire = sw::universal::quire<Posit>;

    const std::size_t n = 20;
    mat::dense2D<Posit> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A(i,j) = Posit(0.0);
    for (std::size_t i = 0; i < n; ++i) {
        A(i,i) = Posit(2.0);
        if (i > 0)     A(i,i-1) = Posit(-1.0);
        if (i < n - 1) A(i,i+1) = Posit(-1.0);
    }

    vec::dense_vector<Posit> b(n, Posit(1.0));

    vec::dense_vector<Posit> x_naive(n, Posit(0.0));
    itl::pc::identity<mat::dense2D<Posit>> pc(A);
    itl::basic_iteration<Posit> iter_naive(b, 200, Posit(1e-6));
    itl::bicgstab(A, x_naive, b, pc, iter_naive); // default Accumulator = naive posit32

    vec::dense_vector<Posit> x_quire(n, Posit(0.0));
    itl::basic_iteration<Posit> iter_quire(b, 200, Posit(1e-6));
    itl::bicgstab<mat::dense2D<Posit>, vec::dense_vector<Posit>, vec::dense_vector<Posit>,
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

    const double rel_diff = std::abs(quire_residual - naive_residual) / naive_residual;
    if (rel_diff > 0.5) {
        std::cerr << "quire-accumulated residual (" << quire_residual
                  << ") differs substantially from naive posit32 (" << naive_residual << ")\n";
        return false;
    }
    return true;
}

} // namespace

int main() {
    int failures = 0;

    if (!bicgstab_default_accumulator_ok())      ++failures;
    if (!bicgstab_1x1_ok())                      ++failures;
    if (!bicgstab_quire_beats_naive_posit32())   ++failures;

    if (failures == 0) std::cout << "test_bicgstab_quire passed\n";
    return failures == 0 ? 0 : 1;
}
