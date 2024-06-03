#include <amgcl/amg.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/backend/cuda.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/relaxation/cusparse_ilu0.hpp>  // Slower than spai0
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/solver/cg.hpp>

#include <cuda_runtime.h>

#include <pybind11/numpy.h>

#include "cpmg/assembler.h"
#include "cpmg/cpamgcl.h"
#include "cpmg/cpmg.h"


namespace cpmg::cpamgcl
{

namespace
{

using Backend = amgcl::backend::cuda<Float>;

using Preconditioner = amgcl::amg<
        Backend,
        amgcl::coarsening::smoothed_aggregation,
        amgcl::relaxation::spai0
>;

using IterativeSolver = amgcl::solver::cg<Backend>;

using Solver = amgcl::make_solver<Preconditioner, IterativeSolver>;


pybind11::tuple
amgcl_solve(
        const pybind11::array_t<bool> & boundaryMask,
        const pybind11::array_t<Float> & laplacian,
        const pybind11::array_t<Float> & boundaryValue,
        const pybind11::array_t<Float> & initialGuess,
        Float relativeTolerance)
{
    std::vector<int> col, ptr;
    std::vector<Float> val, rhs;

    int n2 = assemble2DPoissonProblem(boundaryMask, laplacian, boundaryValue, val, col, ptr, rhs);

    std::vector<Float> x;
    x.reserve(n2);
    x.insert(x.end(), initialGuess.data(), initialGuess.data() + n2);

    Backend::params bprm {};
    cusparseCreate(&bprm.cusparse_handle);

//    TimePoint t1 = Clock::now();
//
//    int * d_col;
//    int * d_ptr;
//    int * d_val;
//
//    cudaMalloc(&d_col, sizeof(int) * col.size());
//    cudaMalloc(&d_ptr, sizeof(int) * ptr.size());
//    cudaMalloc(&d_val, sizeof(Float) * val.size());
//
//    cudaDeviceSynchronize();
//
//    cudaMemcpy(d_col, col.data(), col.size(), cudaMemcpyHostToDevice);
//    cudaMemcpy(d_ptr, ptr.data(), ptr.size(), cudaMemcpyHostToDevice);
//    cudaMemcpy(d_val, val.data(), val.size(), cudaMemcpyHostToDevice);
//
//    cudaDeviceSynchronize();
//
//    cusparseSpMatDescr_t matA;
//
//    cusparseCreateCsr(&matA, n2, n2, static_cast<int>(val.size()),
//                                      d_ptr, d_col, d_val,
//                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
//                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
//
//    cudaDeviceSynchronize();
//
//    TimePoint t2 = Clock::now();
//    Float t3 = std::chrono::duration_cast<Duration>(t2 - t1).count();
//    std::cout << "cusparse initialization took " << t3 << " ms\n";

    auto f_b = Backend::copy_vector(rhs, bprm);
    auto x_b = Backend::copy_vector(x, bprm);

    Solver::params prm {};
    prm.solver.tol = relativeTolerance;

    auto t1 = Clock::now();

    Solver solve(std::tie(n2, ptr, col, val), prm, bprm);

    auto t2 = Clock::now();

    auto [iters, error] = solve(*f_b, *x_b);

    auto t3 = Clock::now();
    auto assemblyTime= std::chrono::duration_cast<Duration>(t2 - t1).count();
    auto solvingTime = std::chrono::duration_cast<Duration>(t3 - t2).count();
    auto totalTime = std::chrono::duration_cast<Duration>(t3 - t1).count();

    thrust::copy(x_b->begin(), x_b->end(), x.begin());

    return pybind11::make_tuple(
            iters,
            error,
            assemblyTime,
            solvingTime,
            totalTime,
            pybind11::array_t<Float>(n2, x.data())
    );
}

}  // namespace anonymous


void init_py_module(pybind11::module_ & m)
{
    namespace py = pybind11;
    using py::literals::operator""_a;

    // This statement must be in the same source file of the function referenced!
    // Otherwise, there will be undefined symbols.
    m.def("amgcl_solve",
          amgcl_solve,
          "boundary_mask"_a,
          "laplacian"_a,
          "boundary_value"_a,
          "initial_guess"_a,
          "relative_tolerance"_a,
          py::return_value_policy::move
    );

}

}  // namespace cpmg::cpamgcl
