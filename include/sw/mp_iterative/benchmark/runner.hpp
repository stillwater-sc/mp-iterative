#pragma once
// mp-iterative benchmarking -- stationary sweep runner.
//
// Drives one stationary smoother across a value-type ladder x accumulator
// strategy grid on the standard 1D Poisson problem, measuring forward error,
// backward residual, and convergence rate against a double reference. Each
// driver .cpp is a one-liner over this: it supplies the smoother template and
// its theoretical convergence rate; everything else -- CLI parsing, the type
// ladder, strategy gating, the reference solve, CSV + human output -- lives
// here so it is written once.
//
// Output: a summary CSV (one row per method x type x strategy) on stdout for
// plotting, and a human-readable table on stderr. A `--history` first argument
// switches stdout to a long-format per-iteration residual history instead.

#include <cstddef>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <string>
#include <type_traits>
#include <vector>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/math/quire_accumulator.hpp>   // posit quire accumulator_traits
#include <mtl/math/accumulator_traits.hpp>  // fma_accumulator

#include <universal/number/posit/posit.hpp>
#include <universal/number/cfloat/cfloat.hpp>

#include "sw/mp_iterative/accumulator_strategies.hpp"
#include "sw/mp_iterative/benchmark/problems.hpp"
#include "sw/mp_iterative/benchmark/reference_solve.hpp"
#include "sw/mp_iterative/benchmark/metrics.hpp"
#include "sw/mp_iterative/benchmark/csv.hpp"

namespace sw::mp_iterative::benchmark {

struct config_result {
    std::string type;
    std::string strategy;
    std::size_t sweeps = 0;
    std::size_t iters_to_tol = 0;   // 0 == did not reach tolerance
    double forward_rel_error = 0;
    double residual_2norm = 0;
    double residual_infnorm = 0;
    double empirical_rate = 0;
    double theoretical_rate = 0;
    double floor = 0;
    std::vector<double> history;     // per-sweep residual (double)
};

template <typename T>
std::vector<double> to_double_vector(const mtl::vec::dense_vector<T>& x, std::size_t n) {
    std::vector<double> d(n);
    for (std::size_t i = 0; i < n; ++i) d[i] = static_cast<double>(x(i));
    return d;
}

/// Run one (value type T, Accumulator) configuration of Smoother on the
/// standard 1D Poisson problem, recording the double residual history.
template <template <typename, typename> class Smoother, typename T, typename Accumulator>
config_result run_config(const std::string& type_name, const std::string& strategy_name,
                         std::size_t n, std::size_t sweeps, double tol, double theoretical_rate,
                         double omega,
                         const mtl::mat::compressed2D<double>& A_ref,
                         const std::vector<double>& b_ref,
                         const std::vector<double>& x_ref) {
    using Matrix = mtl::mat::compressed2D<T>;
    using S      = Smoother<Matrix, Accumulator>;

    Matrix A_T = poisson_1d<T>(n);
    mtl::vec::dense_vector<T> b(n, T(1)), x(n, T(0));

    // SOR takes (A, omega); Jacobi/Gauss-Seidel take (A). Dispatch on the ctor.
    auto make = [&]() {
        if constexpr (std::is_constructible_v<S, const Matrix&, T>)
            return S(A_T, T(omega));
        else
            return S(A_T);
    };
    S smoother = make();

    config_result res;
    res.type = type_name;
    res.strategy = strategy_name;
    res.sweeps = sweeps;
    res.theoretical_rate = theoretical_rate;
    res.history.reserve(sweeps);

    for (std::size_t k = 0; k < sweeps; ++k) {
        smoother(x, b);
        res.history.push_back(residual_2norm(A_ref, to_double_vector(x, n), b_ref));
    }

    std::vector<double> xd = to_double_vector(x, n);
    res.forward_rel_error = forward_relative_error(xd, x_ref);
    res.residual_2norm    = residual_2norm(A_ref, xd, b_ref);
    res.residual_infnorm  = residual_infnorm(A_ref, xd, b_ref);
    res.empirical_rate    = empirical_convergence_rate(res.history);
    res.floor             = stagnation_floor(res.history);
    res.iters_to_tol      = iterations_to_tolerance(res.history, tol);
    return res;
}

/// Run every available accumulator strategy for value type T and append the
/// results. naive is always available; fma requires a fused multiply-add;
/// quire requires a composition-layer quire specialization.
template <template <typename, typename> class Smoother, typename T>
void sweep_type(const std::string& type_name, std::size_t n, std::size_t sweeps,
                double tol, double theoretical_rate, double omega,
                const mtl::mat::compressed2D<double>& A_ref,
                const std::vector<double>& b_ref, const std::vector<double>& x_ref,
                std::vector<config_result>& out) {
    out.push_back(run_config<Smoother, T, void>(
        type_name, "naive", n, sweeps, tol, theoretical_rate, omega, A_ref, b_ref, x_ref));

    if constexpr (has_fused_multiply_add<T>)
        out.push_back(run_config<Smoother, T, mtl::math::fma_accumulator<T>>(
            type_name, "fma", n, sweeps, tol, theoretical_rate, omega, A_ref, b_ref, x_ref));

    if constexpr (has_quire<T>)
        out.push_back(run_config<Smoother, T, quire_of_t<T>>(
            type_name, "quire", n, sweeps, tol, theoretical_rate, omega, A_ref, b_ref, x_ref));
}

inline void print_summary_table(std::ostream& os, const std::string& method,
                                const std::vector<config_result>& results) {
    os << "\n" << method << " on standard 1D Poisson -- accumulator strategy sweep\n";
    os << "  " << std::left << std::setw(15) << "type" << std::setw(8) << "strat" << std::right
       << std::setw(13) << "fwd_rel_err" << std::setw(13) << "resid_2" << std::setw(9) << "iters"
       << std::setw(10) << "emp_rate" << std::setw(10) << "theo_rate" << std::setw(12) << "floor" << '\n';
    os << "  " << std::string(88, '-') << '\n';
    for (const auto& r : results) {
        os << "  " << std::left << std::setw(15) << r.type << std::setw(8) << r.strategy << std::right
           << std::scientific << std::setprecision(3)
           << std::setw(13) << r.forward_rel_error
           << std::setw(13) << r.residual_2norm;
        if (r.iters_to_tol) os << std::setw(9) << r.iters_to_tol;
        else                os << std::setw(9) << "-";
        os << std::fixed << std::setprecision(4)
           << std::setw(10) << r.empirical_rate
           << std::setw(10) << r.theoretical_rate
           << std::scientific << std::setprecision(3) << std::setw(12) << r.floor << '\n';
    }
}

inline void write_summary_csv(std::ostream& os, const std::string& method,
                              const std::vector<config_result>& results) {
    csv_writer w({"method", "type", "strategy", "sweeps", "iters_to_tol",
                  "fwd_rel_err", "resid_2norm", "resid_infnorm",
                  "empirical_rate", "theoretical_rate", "floor"});
    w.write_header(os);
    for (const auto& r : results) {
        w.field(method).field(r.type).field(r.strategy).number(r.sweeps).number(r.iters_to_tol)
         .number(r.forward_rel_error).number(r.residual_2norm).number(r.residual_infnorm)
         .number(r.empirical_rate).number(r.theoretical_rate).number(r.floor)
         .end_row(os);
    }
}

inline void write_history_csv(std::ostream& os, const std::string& method,
                              const std::vector<config_result>& results) {
    csv_writer w({"method", "type", "strategy", "iteration", "residual_2norm"});
    w.write_header(os);
    for (const auto& r : results)
        for (std::size_t k = 0; k < r.history.size(); ++k)
            w.field(method).field(r.type).field(r.strategy).number(k + 1).number(r.history[k]).end_row(os);
}

/// Full driver body: parse args, build the double reference, sweep the type
/// ladder, and emit output. Each stationary benchmark .cpp calls this.
template <template <typename, typename> class Smoother>
int run_stationary_benchmark(const std::string& method, int argc, char** argv,
                             double (*theoretical_rate)(std::size_t)) {
    bool history = false;
    int argi = 1;
    if (argc > argi && std::string(argv[argi]) == "--history") { history = true; ++argi; }

    std::size_t n = 48, sweeps = 500;
    const double tol = 1e-6;
    if (argc > argi)     n = static_cast<std::size_t>(std::stoul(argv[argi]));
    if (argc > argi + 1) sweeps = static_cast<std::size_t>(std::stoul(argv[argi + 1]));

    const double omega = poisson_1d_omega_opt(n);
    const double rate  = theoretical_rate(n);

    // Double reference: same problem, high-accuracy CG solve.
    mtl::mat::compressed2D<double> A_ref = poisson_1d<double>(n);
    std::vector<double> b_ref(n, 1.0);
    std::vector<double> x_ref = reference_solution(A_ref, b_ref);

    std::vector<config_result> results;
    sweep_type<Smoother, double>("double", n, sweeps, tol, rate, omega, A_ref, b_ref, x_ref, results);
    sweep_type<Smoother, float>("float", n, sweeps, tol, rate, omega, A_ref, b_ref, x_ref, results);
    sweep_type<Smoother, sw::universal::cfloat<16, 5>>("cfloat<16,5>", n, sweeps, tol, rate, omega, A_ref, b_ref, x_ref, results);
    sweep_type<Smoother, sw::universal::cfloat<32, 8>>("cfloat<32,8>", n, sweeps, tol, rate, omega, A_ref, b_ref, x_ref, results);
    sweep_type<Smoother, sw::universal::posit<16, 2>>("posit<16,2>", n, sweeps, tol, rate, omega, A_ref, b_ref, x_ref, results);
    sweep_type<Smoother, sw::universal::posit<32, 2>>("posit<32,2>", n, sweeps, tol, rate, omega, A_ref, b_ref, x_ref, results);

    // Human-readable table on stderr; machine-readable CSV on stdout.
    print_summary_table(std::cerr, method, results);
    std::cerr << "\n(n=" << n << ", sweeps=" << sweeps << ", omega_opt=" << omega
              << "; CSV on stdout)\n";
    if (history) write_history_csv(std::cout, method, results);
    else         write_summary_csv(std::cout, method, results);
    return 0;
}

} // namespace sw::mp_iterative::benchmark
