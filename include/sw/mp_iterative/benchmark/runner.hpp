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
#include <limits>
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
#include "sw/mp_iterative/benchmark/matrix_source.hpp"
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
/// selected problem, recording the double residual history. For a Poisson
/// problem the operator is generated in T directly; for a loaded matrix
/// (`from_file`) it is converted from the double reference A_ref, whose
/// dimension governs everything.
template <template <typename, typename> class Smoother, typename T, typename Accumulator>
config_result run_config(const std::string& type_name, const std::string& strategy_name,
                         bool from_file, poisson_dim dim, std::size_t m,
                         std::size_t sweeps, double tol,
                         double theoretical_rate, double omega,
                         const mtl::mat::compressed2D<double>& A_ref,
                         const std::vector<double>& b_ref,
                         const std::vector<double>& x_ref) {
    using Matrix = mtl::mat::compressed2D<T>;
    using S      = Smoother<Matrix, Accumulator>;

    const std::size_t N = A_ref.num_rows();
    Matrix A_T = from_file ? convert_matrix<T>(A_ref) : make_poisson<T>(dim, m);
    mtl::vec::dense_vector<T> b(N, T(1)), x(N, T(0));

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
        res.history.push_back(residual_2norm(A_ref, to_double_vector(x, N), b_ref));
    }

    std::vector<double> xd = to_double_vector(x, N);
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
void sweep_type(const std::string& type_name, bool from_file, poisson_dim dim, std::size_t m,
                std::size_t sweeps, double tol, double theoretical_rate, double omega,
                const mtl::mat::compressed2D<double>& A_ref,
                const std::vector<double>& b_ref, const std::vector<double>& x_ref,
                std::vector<config_result>& out) {
    out.push_back(run_config<Smoother, T, void>(
        type_name, "naive", from_file, dim, m, sweeps, tol, theoretical_rate, omega, A_ref, b_ref, x_ref));

    if constexpr (has_fused_multiply_add<T>)
        out.push_back(run_config<Smoother, T, mtl::math::fma_accumulator<T>>(
            type_name, "fma", from_file, dim, m, sweeps, tol, theoretical_rate, omega, A_ref, b_ref, x_ref));

    if constexpr (has_quire<T>)
        out.push_back(run_config<Smoother, T, quire_of_t<T>>(
            type_name, "quire", from_file, dim, m, sweeps, tol, theoretical_rate, omega, A_ref, b_ref, x_ref));
}

inline void print_summary_table(std::ostream& os, const std::string& method,
                                const std::string& problem,
                                const std::vector<config_result>& results) {
    os << "\n" << method << " on standard " << problem << " -- accumulator strategy sweep\n";
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
                              const std::string& problem,
                              const std::vector<config_result>& results) {
    csv_writer w({"method", "problem", "type", "strategy", "sweeps", "iters_to_tol",
                  "fwd_rel_err", "resid_2norm", "resid_infnorm",
                  "empirical_rate", "theoretical_rate", "floor"});
    w.write_header(os);
    for (const auto& r : results) {
        w.field(method).field(problem).field(r.type).field(r.strategy).number(r.sweeps).number(r.iters_to_tol)
         .number(r.forward_rel_error).number(r.residual_2norm).number(r.residual_infnorm)
         .number(r.empirical_rate).number(r.theoretical_rate).number(r.floor)
         .end_row(os);
    }
}

inline void write_history_csv(std::ostream& os, const std::string& method,
                              const std::string& problem,
                              const std::vector<config_result>& results) {
    csv_writer w({"method", "problem", "type", "strategy", "iteration", "residual_2norm"});
    w.write_header(os);
    for (const auto& r : results)
        for (std::size_t k = 0; k < r.history.size(); ++k)
            w.field(method).field(problem).field(r.type).field(r.strategy)
             .number(k + 1).number(r.history[k]).end_row(os);
}

/// Parse a problem selector token ("1d"/"2d"/"3d"); returns d1 for anything else.
inline poisson_dim parse_poisson_dim(const std::string& tok) {
    if (tok == "2d" || tok == "2D") return poisson_dim::d2;
    if (tok == "3d" || tok == "3D") return poisson_dim::d3;
    return poisson_dim::d1;
}

/// Default grid points per dimension, chosen so each problem stays quick to
/// run (matrix dimension m^d comparable across dimensions).
inline std::size_t default_grid(poisson_dim dim) {
    switch (dim) {
        case poisson_dim::d1: return 64;
        case poisson_dim::d2: return 32;
        case poisson_dim::d3: return 12;
    }
    return 64;
}

/// Basename (drop directory and one extension) for labeling a loaded matrix.
inline std::string matrix_label(const std::string& path) {
    std::size_t slash = path.find_last_of("/\\");
    std::string base = (slash == std::string::npos) ? path : path.substr(slash + 1);
    std::size_t dot = base.find_last_of('.');
    if (dot != std::string::npos) base = base.substr(0, dot);
    return base;
}

/// Full driver body: parse args, build the double reference, sweep the type
/// ladder, and emit output. Each stationary benchmark .cpp calls this.
///
///   <driver> [--history] [1d|2d|3d] [m] [sweeps]
///   <driver> [--history] --matrix PATH.mtx [sweeps]
///
/// For a Poisson problem `m` is grid points per dimension (matrix dimension
/// m^d) and the theoretical rate is analytic. For a loaded Matrix Market
/// operator the rate is unknown (reported as NaN) and SOR runs at omega=1
/// (= Gauss-Seidel) since omega_opt needs a spectral radius we do not have.
/// The loaded matrix must be SPD (GS/SOR convergence + valid CG reference).
template <template <typename, typename> class Smoother>
int run_stationary_benchmark(const std::string& method, int argc, char** argv,
                             double (*theoretical_rate)(std::size_t)) {
    bool history = false;
    int argi = 1;
    if (argc > argi && std::string(argv[argi]) == "--history") { history = true; ++argi; }

    bool from_file = false;
    std::string matrix_path;
    poisson_dim dim = poisson_dim::d1;
    if (argc > argi && std::string(argv[argi]) == "--matrix") {
        from_file = true;
        if (argc <= argi + 1) { std::cerr << "error: --matrix needs a path\n"; return 2; }
        matrix_path = argv[argi + 1];
        argi += 2;
    } else if (argc > argi) {
        std::string tok = argv[argi];
        if (tok == "1d" || tok == "2d" || tok == "3d" || tok == "1D" || tok == "2D" || tok == "3D") {
            dim = parse_poisson_dim(tok);
            ++argi;
        }
    }

    const double tol = 1e-6;
    std::size_t m = default_grid(dim), sweeps = 500;

    // Build the double reference operator.
    mtl::mat::compressed2D<double> A_ref;
    std::string problem;
    if (from_file) {
        try {
            A_ref = load_matrix_market(matrix_path);
        } catch (const std::exception& e) {
            std::cerr << "error loading matrix: " << e.what() << '\n';
            return 2;
        }
        problem = matrix_label(matrix_path);
        if (argc > argi) sweeps = static_cast<std::size_t>(std::stoul(argv[argi]));
    } else {
        if (argc > argi)     m = static_cast<std::size_t>(std::stoul(argv[argi]));
        if (argc > argi + 1) sweeps = static_cast<std::size_t>(std::stoul(argv[argi + 1]));
        A_ref = make_poisson<double>(dim, m);
        problem = poisson_dim_name(dim);
    }

    // Poisson problems have an analytic rate and omega_opt; a loaded matrix has
    // neither (rate NaN, SOR at omega=1).
    const double omega = from_file ? 1.0 : poisson_omega_opt(m);
    const double rate  = from_file ? std::numeric_limits<double>::quiet_NaN()
                                   : theoretical_rate(m);

    std::vector<double> b_ref(A_ref.num_rows(), 1.0);
    std::vector<double> x_ref = reference_solution(A_ref, b_ref);

    std::vector<config_result> results;
    sweep_type<Smoother, double>("double", from_file, dim, m, sweeps, tol, rate, omega, A_ref, b_ref, x_ref, results);
    sweep_type<Smoother, float>("float", from_file, dim, m, sweeps, tol, rate, omega, A_ref, b_ref, x_ref, results);
    sweep_type<Smoother, sw::universal::cfloat<16, 5>>("cfloat<16,5>", from_file, dim, m, sweeps, tol, rate, omega, A_ref, b_ref, x_ref, results);
    sweep_type<Smoother, sw::universal::cfloat<32, 8>>("cfloat<32,8>", from_file, dim, m, sweeps, tol, rate, omega, A_ref, b_ref, x_ref, results);
    sweep_type<Smoother, sw::universal::posit<16, 2>>("posit<16,2>", from_file, dim, m, sweeps, tol, rate, omega, A_ref, b_ref, x_ref, results);
    sweep_type<Smoother, sw::universal::posit<32, 2>>("posit<32,2>", from_file, dim, m, sweeps, tol, rate, omega, A_ref, b_ref, x_ref, results);

    // Human-readable table on stderr; machine-readable CSV on stdout.
    print_summary_table(std::cerr, method, problem, results);
    std::cerr << "\n(" << problem << ", " << A_ref.num_rows() << " unknowns, "
              << max_nnz_per_row(A_ref) << " max nnz/row, sweeps=" << sweeps;
    if (!from_file) std::cerr << ", omega_opt=" << omega;
    std::cerr << "; CSV on stdout)\n";
    if (history) write_history_csv(std::cout, method, problem, results);
    else         write_summary_csv(std::cout, method, problem, results);
    return 0;
}

} // namespace sw::mp_iterative::benchmark
