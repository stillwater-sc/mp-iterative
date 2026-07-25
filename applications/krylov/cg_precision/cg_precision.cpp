// cg_precision: Conjugate Gradient on a 1D Poisson (tridiag(-1,2,-1)) SPD system,
// compared across number types AND accumulator strategies.
//
// The Krylov analogue of jacobi_precision. CG runs through
// mtl::itl::cg<..., Accumulator>, which routes BOTH the matrix-vector product
// and the two inner products (rho, pAp) through the accumulator (mtl5 #158):
//   naive -- accumulate in the value type (two roundings per term)
//   fma   -- fused multiply-add, one rounding per term
//   quire -- exact super-accumulation, single rounding at the end
// This reproduces the two "pure" variants of the Universal CG dynamics study
// (mixedprecision/tensor/cg: mv-dot x cmp-dot == naive, mv-fdp x cmp-fdp ==
// quire); the mixed variants need separate matvec/dot accumulators, an MTL5 cg
// enhancement tracked for later. Low-precision types stall at their rounding
// floor -- the strategy columns show how much of that floor is accumulation
// error, and where exact accumulation lets CG make further progress.
#include <cmath>
#include <concepts>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <string>
#include <type_traits>

#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/operators.hpp>
#include <mtl/operation/dot.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/itl/pc/identity.hpp>
#include <mtl/itl/iteration/basic_iteration.hpp>
#include <mtl/itl/krylov/cg.hpp>

#include <universal/number/posit/posit.hpp>
#include <universal/number/cfloat/cfloat.hpp>
#include <mtl/math/quire_accumulator.hpp>

namespace {

template <typename T>
struct quire_of { using type = void; };
template <unsigned nbits, unsigned es>
struct quire_of<sw::universal::posit<nbits, es>> {
    using type = sw::universal::quire<sw::universal::posit<nbits, es>>;
};

template <typename T>
concept has_fused_multiply_add =
    std::is_floating_point_v<T> ||
    requires (T a) { { fma(a, a, a) } -> std::convertible_to<T>; };

// CG on tridiag(-1, 2, -1) x = b with b = ones (SPD Poisson operator). Returns
// the final ||A*x - b||_inf evaluated in double, after at most `maxiter` steps.
template <typename T, typename Accumulator = void>
double cg_residual(std::size_t n, std::size_t maxiter) {
    using Matrix = mtl::mat::dense2D<T>;
    using Vector = mtl::vec::dense_vector<T>;

    Matrix A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) A(i, j) = T(0);
    for (std::size_t i = 0; i < n; ++i) {
        A(i, i) = T(2);
        if (i > 0)     A(i, i - 1) = T(-1);
        if (i + 1 < n) A(i, i + 1) = T(-1);
    }

    Vector b(n, T(1));
    Vector x(n, T(0));

    mtl::itl::pc::identity<Matrix> pc(A);
    mtl::itl::basic_iteration<T> iter(b, static_cast<int>(maxiter), T(1e-12));
    mtl::itl::cg<Matrix, Vector, Vector, mtl::itl::pc::identity<Matrix>,
                 mtl::itl::basic_iteration<T>, Accumulator>(A, x, b, pc, iter);

    double r = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double ax = 0.0;
        for (std::size_t j = 0; j < n; ++j) ax += double(A(i, j)) * double(x[j]);
        r = std::max(r, std::abs(ax - double(b[i])));
    }
    return r;
}

void print_cell(double residual) { std::cout << std::scientific << std::setprecision(3) << std::setw(12) << residual; }
void print_na() { std::cout << std::setw(12) << "n/a"; }

template <typename T>
void report(const std::string& name, std::size_t n, std::size_t maxiter) {
    std::cout << "  " << std::left << std::setw(16) << name << std::right;

    print_cell(cg_residual<T>(n, maxiter));   // naive

    if constexpr (has_fused_multiply_add<T>)
        print_cell(cg_residual<T, mtl::math::fma_accumulator<T>>(n, maxiter));
    else
        print_na();

    using Quire = typename quire_of<T>::type;
    if constexpr (!std::is_void_v<Quire>)
        print_cell(cg_residual<T, Quire>(n, maxiter));
    else
        print_na();

    std::cout << '\n';
}

} // namespace

int main(int argc, char* argv[]) {
    std::size_t n = 200;          // ill-conditioned enough (cond ~ (n/pi)^2) that
    std::size_t maxiter = 500;    // low-precision accumulation error is visible
    if (argc > 1) n = static_cast<std::size_t>(std::stoul(argv[1]));
    if (argc > 2) maxiter = static_cast<std::size_t>(std::stoul(argv[2]));

    std::cout << "CG on 1D Poisson tridiag(-1,2,-1), n = " << n
              << ", <= " << maxiter << " iters, ||Ax-b||inf by accumulator strategy\n";
    std::cout << "  type            " << std::right
              << std::setw(12) << "naive" << std::setw(12) << "fma" << std::setw(12) << "quire" << '\n';
    std::cout << "  ----------------------------------------------------\n";
    report<double>("double", n, maxiter);
    report<float>("float", n, maxiter);
    report<sw::universal::cfloat<16, 5>>("cfloat<16,5>", n, maxiter);
    report<sw::universal::posit<16, 2>>("posit<16,2>", n, maxiter);
    report<sw::universal::posit<32, 2>>("posit<32,2>", n, maxiter);
    return 0;
}
