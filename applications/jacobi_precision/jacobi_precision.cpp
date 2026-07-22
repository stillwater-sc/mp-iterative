// jacobi_precision: Jacobi iteration on a shifted 1D Poisson system
// (tridiagonal, diagonally dominant), compared across number types.
// Demonstrates the MTL5 + Universal composition pattern: the iteration is
// value-type generic, the number system is a template parameter. Low-precision
// types stagnate at their rounding floor -- the starting point for
// mixed-precision iterative studies.
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <string>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>

#include <universal/number/posit/posit.hpp>
#include <universal/number/cfloat/cfloat.hpp>

namespace {

// Jacobi on (4I - tridiag(1, 0, 1)) x = b with b = ones; the solution is NOT
// exactly representable, so the converged residual exposes the rounding floor
// of each number type. Returns ||A*x - b||_inf evaluated in double.
template <typename T>
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
    mtl::vec::dense_vector<T> x(n, T(0)), xn(n, T(0));

    const auto& rp = A.ref_major();
    const auto& ci = A.ref_minor();
    const auto& dat = A.ref_data();
    for (std::size_t iter = 0; iter < iterations; ++iter) {
        for (std::size_t i = 0; i < n; ++i) {
            T sigma(0), diag(0);
            for (std::size_t k = rp[i]; k < rp[i + 1]; ++k) {
                if (ci[k] == i) diag = dat[k];
                else            sigma += dat[k] * x(static_cast<int>(ci[k]));
            }
            xn(static_cast<int>(i)) = (b(static_cast<int>(i)) - sigma) / diag;
        }
        x = xn;
    }

    double r = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double ax = 0.0;
        for (std::size_t k = rp[i]; k < rp[i + 1]; ++k)
            ax += static_cast<double>(dat[k]) * static_cast<double>(x(static_cast<int>(ci[k])));
        r = std::max(r, std::abs(ax - static_cast<double>(b(static_cast<int>(i)))));
    }
    return r;
}

template <typename T>
void report(const std::string& name, std::size_t n, std::size_t iterations) {
    std::cout << "  " << std::left << std::setw(16) << name
              << std::scientific << std::setprecision(3)
              << jacobi_residual<T>(n, iterations) << '\n';
}

} // namespace

int main(int argc, char* argv[]) {
    std::size_t n = 64;
    std::size_t iterations = 100;
    if (argc > 1) n = static_cast<std::size_t>(std::stoul(argv[1]));
    if (argc > 2) iterations = static_cast<std::size_t>(std::stoul(argv[2]));

    std::cout << "Jacobi on shifted 1D Poisson, n = " << n
              << ", " << iterations << " iterations\n";
    std::cout << "  type            ||Ax-b||inf\n";
    std::cout << "  ---------------------------\n";
    report<double>("double", n, iterations);
    report<float>("float", n, iterations);
    report<sw::universal::cfloat<16, 5>>("cfloat<16,5>", n, iterations);
    report<sw::universal::posit<16, 2>>("posit<16,2>", n, iterations);
    return 0;
}
