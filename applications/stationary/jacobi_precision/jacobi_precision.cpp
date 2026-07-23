// jacobi_precision: Jacobi iteration on a shifted 1D Poisson system
// (tridiagonal, diagonally dominant), compared across number types AND
// accumulator strategies (issue #3).
//
// The iteration runs through mtl::itl::smoother::jacobi<Matrix, Accumulator>
// (accumulator-aware since mtl5 #262): the number system is one template
// parameter, the accumulation strategy for the row sums is another.
// Strategies:
//   naive -- accumulate in the value type (two roundings per term)
//   fma   -- fused multiply-add, one rounding per term (mtl5 #259);
//            std::fma for built-ins, ADL-found fma for Universal types
//            (posit and cfloat both provide one)
//   quire -- exact accumulation, single rounding at the end (posit only;
//            composition-layer specialization in mtl/math/quire_accumulator.hpp)
// Low-precision types stagnate at their rounding floor -- the strategy
// columns show how much of that floor is accumulation error.
#include <cmath>
#include <concepts>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <string>
#include <type_traits>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/itl/smoother/jacobi.hpp>

#include <universal/number/posit/posit.hpp>
#include <universal/number/cfloat/cfloat.hpp>
#include <mtl/math/quire_accumulator.hpp>

namespace {

// Quire type for value types that have one (posit); void otherwise.
template <typename T>
struct quire_of { using type = void; };
template <unsigned nbits, unsigned es>
struct quire_of<sw::universal::posit<nbits, es>> {
    using type = sw::universal::quire<sw::universal::posit<nbits, es>>;
};

// The FMA strategy needs a fused multiply-add: std::fma for built-ins, an
// ADL-found fma for custom types (Universal provides one for posit and
// cfloat). Types without one fall back to n/a in the table.
template <typename T>
concept has_fused_multiply_add =
    std::is_floating_point_v<T> ||
    requires (T a) { { fma(a, a, a) } -> std::convertible_to<T>; };

// Jacobi on (4I - tridiag(1, 0, 1)) x = b with b = ones; the solution is NOT
// exactly representable, so the converged residual exposes the rounding floor
// of each number type. Returns ||A*x - b||_inf evaluated in double.
template <typename T, typename Accumulator = void>
double jacobi_residual(std::size_t n, std::size_t iterations) {
    using Sparse = mtl::mat::compressed2D<T>;
    Sparse A(n, n);
    mtl::vec::dense_vector<T> b(n, T(1));
    {
        mtl::mat::inserter<Sparse> ins(A);
        for (std::size_t i = 0; i < n; ++i) {
            ins[i][i] << T(4);
            if (i > 0)     ins[i][i - 1] << T(-1);
            if (i + 1 < n) ins[i][i + 1] << T(-1);
        }
    }
    mtl::vec::dense_vector<T> x(n, T(0));

    mtl::itl::smoother::jacobi<Sparse, Accumulator> smoother(A);
    for (std::size_t iter = 0; iter < iterations; ++iter)
        smoother(x, b);

    const auto& rp  = A.ref_major();
    const auto& ci  = A.ref_minor();
    const auto& dat = A.ref_data();
    double r = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double ax = 0.0;
        for (std::size_t k = rp[i]; k < rp[i + 1]; ++k)
            ax += static_cast<double>(dat[k]) * static_cast<double>(x(static_cast<int>(ci[k])));
        r = std::max(r, std::abs(ax - static_cast<double>(b(static_cast<int>(i)))));
    }
    return r;
}

void print_cell(double residual) {
    std::cout << std::scientific << std::setprecision(3) << std::setw(12) << residual;
}

void print_na() {
    std::cout << std::setw(12) << "n/a";
}

template <typename T>
void report(const std::string& name, std::size_t n, std::size_t iterations) {
    std::cout << "  " << std::left << std::setw(16) << name << std::right;

    print_cell(jacobi_residual<T>(n, iterations));   // naive

    if constexpr (has_fused_multiply_add<T>)
        print_cell(jacobi_residual<T, mtl::math::fma_accumulator<T>>(n, iterations));
    else
        print_na();

    using Quire = typename quire_of<T>::type;
    if constexpr (!std::is_void_v<Quire>)
        print_cell(jacobi_residual<T, Quire>(n, iterations));
    else
        print_na();

    std::cout << '\n';
}

} // namespace

int main(int argc, char* argv[]) {
    std::size_t n = 64;
    std::size_t iterations = 100;
    if (argc > 1) n = static_cast<std::size_t>(std::stoul(argv[1]));
    if (argc > 2) iterations = static_cast<std::size_t>(std::stoul(argv[2]));

    std::cout << "Jacobi on shifted 1D Poisson, n = " << n
              << ", " << iterations << " iterations, ||Ax-b||inf by accumulator strategy\n";
    std::cout << "  type            " << std::right
              << std::setw(12) << "naive" << std::setw(12) << "fma" << std::setw(12) << "quire" << '\n';
    std::cout << "  ----------------------------------------------------\n";
    report<double>("double", n, iterations);
    report<float>("float", n, iterations);
    report<sw::universal::cfloat<16, 5>>("cfloat<16,5>", n, iterations);
    report<sw::universal::posit<16, 2>>("posit<16,2>", n, iterations);
    report<sw::universal::posit<32, 2>>("posit<32,2>", n, iterations);
    return 0;
}
