// sor_precision: SOR (Successive Over-Relaxation) on the standard 1D Poisson
// system tridiag(-1, 2, -1), compared across number types, accumulator
// strategies, AND the relaxation factor omega (issue #5). Companion to
// jacobi_precision / gauss_seidel_precision.
//
// The iteration runs through mtl::itl::smoother::sor<Matrix, Accumulator>
// (accumulator-aware since mtl5 #267): the number system is one template
// parameter, the accumulation strategy for the row sums is another, and omega
// is a constructor argument. The omega blend is scalar arithmetic outside the
// accumulator, so the omega-vs-precision behavior is studied independently of
// the row-sum rounding.
//
// Two studies:
//   1. Accumulator-strategy table at omega_opt (naive / fma / quire per type),
//      identical in shape to jacobi_precision / gauss_seidel_precision.
//   2. Omega sweep: residual after a fixed sweep budget as omega varies, per
//      number type. The omega with the lowest residual is the EMPIRICAL
//      optimum; comparing it to the theoretical omega_opt = 2/(1+sin(pi/(n+1)))
//      shows whether low precision shifts or flattens the optimum.
//
// The standard 1D Poisson (not the shifted 4I variant used by the Jacobi/GS
// demos) is chosen deliberately: its Jacobi spectral radius rho_J = cos(pi/(n+1))
// is close to 1, so omega_opt is well separated from 1 and over-relaxation
// matters -- the regime SOR targets.
#include <cmath>
#include <concepts>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/itl/smoother/sor.hpp>

#include <universal/number/posit/posit.hpp>
#include <universal/number/cfloat/cfloat.hpp>
#include <mtl/math/quire_accumulator.hpp>

namespace {

// Quire super-accumulator for the value type; void when the composition
// layer does not yet provide an accumulator_traits specialization for it.
// Any fixed-size arithmetic admits a quire (integer, fixpnt, posit, lns,
// cfloat) -- posit is just the first one wired up, so the quire column
// reads n/a for the others until their specializations land.
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

const double kPi = 3.14159265358979323846;

// Theoretical optimal relaxation factor for the standard 1D Poisson operator.
double omega_opt(std::size_t n) {
    return 2.0 / (1.0 + std::sin(kPi / static_cast<double>(n + 1)));
}

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

template <typename T>
double residual_inf(const mtl::mat::compressed2D<T>& A, const mtl::vec::dense_vector<T>& x) {
    const std::size_t n = A.num_rows();
    const auto& rp  = A.ref_major();
    const auto& ci  = A.ref_minor();
    const auto& dat = A.ref_data();
    double r = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double ax = 0.0;
        for (std::size_t k = rp[i]; k < rp[i + 1]; ++k)
            ax += static_cast<double>(dat[k]) * static_cast<double>(x(static_cast<int>(ci[k])));
        r = std::max(r, std::abs(ax - 1.0));  // b = ones
    }
    return r;
}

template <typename T, typename Accumulator = void>
double sor_residual(const mtl::mat::compressed2D<T>& A, T omega, std::size_t iterations) {
    const std::size_t n = A.num_rows();
    mtl::vec::dense_vector<T> b(n, T(1));
    mtl::vec::dense_vector<T> x(n, T(0));
    mtl::itl::smoother::sor<mtl::mat::compressed2D<T>, Accumulator> smoother(A, omega);
    for (std::size_t iter = 0; iter < iterations; ++iter)
        smoother(x, b);
    return residual_inf(A, x);
}

void print_cell(double residual) {
    std::cout << std::scientific << std::setprecision(3) << std::setw(12) << residual;
}
void print_na() { std::cout << std::setw(12) << "n/a"; }

// Study 1: accumulator-strategy table at omega_opt.
template <typename T>
void report_strategies(const std::string& name, std::size_t n, std::size_t iterations) {
    auto A = poisson_1d<T>(n);
    T omega(omega_opt(n));
    std::cout << "  " << std::left << std::setw(16) << name << std::right;

    print_cell(sor_residual<T>(A, omega, iterations));   // naive

    if constexpr (has_fused_multiply_add<T>)
        print_cell(sor_residual<T, mtl::math::fma_accumulator<T>>(A, omega, iterations));
    else
        print_na();

    using Quire = typename quire_of<T>::type;
    if constexpr (!std::is_void_v<Quire>)
        print_cell(sor_residual<T, Quire>(A, omega, iterations));
    else
        print_na();

    std::cout << '\n';
}

// Study 2: omega sweep (naive accumulation) -- residual after a fixed budget
// as omega varies. Returns the empirical optimum omega for annotation.
template <typename T>
double report_omega_sweep(const std::string& name, std::size_t n, std::size_t iterations,
                          const std::vector<double>& omegas) {
    auto A = poisson_1d<T>(n);
    std::cout << "  " << std::left << std::setw(16) << name << std::right;
    double best_res = std::numeric_limits<double>::infinity();
    double best_omega = omegas.empty() ? 1.0 : omegas.front();
    for (double w : omegas) {
        double r = sor_residual<T>(A, T(w), iterations);
        print_cell(r);
        if (r < best_res) { best_res = r; best_omega = w; }
    }
    std::cout << "   -> " << std::fixed << std::setprecision(3) << best_omega << '\n';
    return best_omega;
}

} // namespace

int main(int argc, char* argv[]) {
    std::size_t n = 48;
    std::size_t iterations = 500;
    if (argc > 1) n = static_cast<std::size_t>(std::stoul(argv[1]));
    if (argc > 2) iterations = static_cast<std::size_t>(std::stoul(argv[2]));

    const double w_opt = omega_opt(n);

    std::cout << "SOR on standard 1D Poisson tridiag(-1,2,-1), n = " << n
              << ", " << iterations << " iterations\n";
    std::cout << "theoretical omega_opt = 2/(1+sin(pi/(n+1))) = "
              << std::fixed << std::setprecision(4) << w_opt << "\n\n";

    // Study 1: accumulator strategies at omega_opt.
    std::cout << "[1] ||Ax-b||inf by accumulator strategy, at omega_opt\n";
    std::cout << "  type            " << std::right
              << std::setw(12) << "naive" << std::setw(12) << "fma" << std::setw(12) << "quire" << '\n';
    std::cout << "  ----------------------------------------------------\n";
    report_strategies<double>("double", n, iterations);
    report_strategies<float>("float", n, iterations);
    report_strategies<sw::universal::cfloat<16, 5>>("cfloat<16,5>", n, iterations);
    report_strategies<sw::universal::posit<16, 2>>("posit<16,2>", n, iterations);
    report_strategies<sw::universal::posit<32, 2>>("posit<32,2>", n, iterations);

    // Study 2: omega sweep (naive accumulation) -- does the empirical optimum
    // drift from the theoretical omega_opt as precision drops?
    std::vector<double> omegas = {1.0, 1.2, 1.4, 1.6, 1.8, w_opt, 1.95};
    std::cout << "\n[2] ||Ax-b||inf vs omega (naive accumulation); last col = empirical argmin\n";
    std::cout << "  type            " << std::right;
    for (double w : omegas) {
        std::ostringstream tag;
        tag << std::fixed << std::setprecision(2) << w;
        std::cout << std::setw(12) << tag.str();
    }
    std::cout << "   argmin\n";
    std::cout << "  --------------------------------------------------------------------------------------------\n";
    report_omega_sweep<double>("double", n, iterations, omegas);
    report_omega_sweep<float>("float", n, iterations, omegas);
    report_omega_sweep<sw::universal::cfloat<16, 5>>("cfloat<16,5>", n, iterations, omegas);
    report_omega_sweep<sw::universal::posit<16, 2>>("posit<16,2>", n, iterations, omegas);
    report_omega_sweep<sw::universal::posit<32, 2>>("posit<32,2>", n, iterations, omegas);
    return 0;
}
