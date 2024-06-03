#include <iostream>

#include <amgx_c.h>
#include <cuda_runtime.h>
#include <pybind11/numpy.h>

#include "cpmg/assembler.h"
#include "cpmg/cpamgx.h"
#include "cpmg/cpmg.h"
#include "cpmg/cuda_utils.h"


namespace cpmg::cpamgx
{

namespace
{

constexpr AMGX_Mode kMode = AMGX_mode_dFFI;


void amgx_initialize()
{
    AMGX_CHECK(AMGX_initialize());
    AMGX_CHECK(AMGX_register_print_callback([](const char *, int) -> void {}));
}


void amgx_finalize()
{
    AMGX_CHECK(AMGX_finalize());
}


pybind11::tuple
amgx_solve(
        const pybind11::array_t<bool> & boundaryMask,
        const pybind11::array_t<Float> & laplacian,
        const pybind11::array_t<Float> & boundaryValue,
        const pybind11::array_t<Float> & initialGuess
)
{
    // +---------------------------------------------+
    // | Problem assembly and GPU memory allocation. |
    // +---------------------------------------------+

    std::vector<int> col, ptr;
    std::vector<Float> val, rhs;
    int n2 = assemble2DPoissonProblem(boundaryMask, laplacian, boundaryValue, val, col, ptr, rhs);

    int * cudaCol = nullptr;
    CUDA_CHECK(cudaMalloc(&cudaCol, sizeof(int) * col.size()));
    CUDA_CHECK(cudaMemcpy(cudaCol, col.data(), sizeof(int) * col.size(), cudaMemcpyHostToDevice));

    int * cudaPtr = nullptr;
    CUDA_CHECK(cudaMalloc(&cudaPtr, sizeof(int) * ptr.size()));
    CUDA_CHECK(cudaMemcpy(cudaPtr, ptr.data(), sizeof(int) * ptr.size(), cudaMemcpyHostToDevice));

    Float * cudaVal = nullptr;
    CUDA_CHECK(cudaMalloc(&cudaVal, sizeof(Float) * val.size()));
    CUDA_CHECK(cudaMemcpy(cudaVal, val.data(), sizeof(Float) * val.size(), cudaMemcpyHostToDevice));

    Float * cudaRhs = nullptr;
    CUDA_CHECK(cudaMalloc(&cudaRhs, sizeof(Float) * rhs.size()));
    CUDA_CHECK(cudaMemcpy(cudaRhs, rhs.data(), sizeof(Float) * rhs.size(), cudaMemcpyHostToDevice));

    Float * cudaX = nullptr;
    CUDA_CHECK(cudaMalloc(&cudaX, initialGuess.nbytes()));
    CUDA_CHECK(cudaMemcpy(cudaX, initialGuess.data(), sizeof(Float) * initialGuess.size(), cudaMemcpyHostToDevice));

    // +------+
    // | AMGX |
    // + -----+

    // Configuration string copied from amgx/src/configs/AMG_CLASSICAL_CG.json
    AMGX_config_handle cfg = nullptr;
    std::string options = "{\n"
                          "    \"config_version\": 2, \n"
                          "    \"solver\": {\n"
                          "        \"print_grid_stats\": 0, \n"
                          "        \"solver\": \"AMG\", \n"
                          "        \"print_solve_stats\": 0, \n"
                          "        \"interpolator\": \"D2\",\n"
                          "        \"presweeps\": 1, \n"
                          "        \"obtain_timings\": 0, \n"
                          "        \"max_iters\": 100, \n"
                          "        \"monitor_residual\": 1, \n"
                          "        \"convergence\": \"ABSOLUTE\", \n"
                          "        \"scope\": \"main\", \n"
                          "        \"max_levels\": 100, \n"
                          "        \"cycle\": \"CG\", \n"
                          "        \"tolerance\": 5e-2, \n"  // 5e-2 for benchmark
                          "        \"norm\": \"L2\", \n"
                          "        \"postsweeps\": 1\n"
                          "    }\n"
                          "}";
    AMGX_CHECK(AMGX_config_create(&cfg, options.c_str()));

    AMGX_resources_handle rsc = nullptr;
    AMGX_CHECK(AMGX_resources_create_simple(&rsc, cfg));

    AMGX_matrix_handle A = nullptr;
    AMGX_CHECK(AMGX_matrix_create(&A, rsc, kMode));
    AMGX_CHECK(
            AMGX_matrix_upload_all(
                    A, n2, val.size(), 1, 1,
                    cudaPtr, cudaCol, cudaVal, nullptr
            )
    );

    AMGX_vector_handle x = nullptr;
    AMGX_CHECK(AMGX_vector_create(&x, rsc, kMode));
    AMGX_CHECK(AMGX_vector_upload(x, initialGuess.size(), 1, cudaX));

    AMGX_vector_handle b = nullptr;
    AMGX_CHECK(AMGX_vector_create(&b, rsc, kMode));
    AMGX_CHECK(AMGX_vector_upload(b, boundaryValue.size(), 1, cudaRhs));

    AMGX_solver_handle slv = nullptr;
    AMGX_CHECK(AMGX_solver_create(&slv, rsc, kMode, cfg));

    // +-------------------+
    // | Warmup and reset. |
    // +-------------------+

    AMGX_CHECK(AMGX_solver_setup(slv, A));
    AMGX_CHECK(AMGX_solver_solve(slv, b, x));

    CUDA_CHECK(cudaMemcpy(cudaX, initialGuess.data(), sizeof(Float) * initialGuess.size(), cudaMemcpyHostToDevice));
    AMGX_CHECK(AMGX_vector_upload(x, initialGuess.size(), 1, cudaX));

    // +----------------------+
    // | Solve and benchmark. |
    // +----------------------+

    auto t1 = Clock::now();

    // NO return code checks here for performance
    AMGX_solver_setup(slv, A);

    auto t2 = Clock::now();

    AMGX_solver_solve(slv, b, x);

    auto t3 = Clock::now();
    auto assemblyTime= std::chrono::duration_cast<Duration>(t2 - t1).count();
    auto solvingTime = std::chrono::duration_cast<Duration>(t3 - t2).count();
    auto totalTime = std::chrono::duration_cast<Duration>(t3 - t1).count();

    // +---------------------------+
    // | Solver status collection. |
    // +---------------------------+

    AMGX_SOLVE_STATUS st {};
    AMGX_CHECK(AMGX_solver_get_status(slv, &st));

    if (st != AMGX_SOLVE_SUCCESS)
    {
        throw std::runtime_error("amgx solve failed with status " + std::to_string(st));
    }

    int iters = 0;
    AMGX_CHECK(AMGX_solver_get_iterations_number(slv, &iters));

    std::vector<Float> y(initialGuess.size(), 0.0f);
    AMGX_CHECK(AMGX_vector_download(x, static_cast<void *>(y.data())));

    // +---------------------+
    // | Free all resources. |
    // +---------------------+

    AMGX_CHECK(AMGX_solver_destroy(slv));
    AMGX_CHECK(AMGX_matrix_destroy(A));
    AMGX_CHECK(AMGX_vector_destroy(x));
    AMGX_CHECK(AMGX_vector_destroy(b));

    AMGX_CHECK(AMGX_resources_destroy(rsc));
    AMGX_CHECK(AMGX_config_destroy(cfg));

    CUDA_CHECK(cudaFree(cudaCol));
    CUDA_CHECK(cudaFree(cudaPtr));
    CUDA_CHECK(cudaFree(cudaVal));
    CUDA_CHECK(cudaFree(cudaRhs));
    CUDA_CHECK(cudaFree(cudaX));

    return pybind11::make_tuple(
            iters,
            assemblyTime,
            solvingTime,
            totalTime,
            pybind11::array_t<Float>(static_cast<int>(y.size()), y.data())
    );
}

}  // namespace anonmynous


void init_py_module(pybind11::module_ & m)
{
    namespace py = pybind11;
    using py::literals::operator""_a;

    // This statement must be in the same source file of the function referenced!
    // Otherwise, there will be undefined symbols.
    m.def("amgx_initialize", amgx_initialize);

    m.def("amgx_solve",
          amgx_solve,
          "boundary_mask"_a,
          "laplacian"_a,
          "boundary_value"_a,
          "initial_guess"_a,
          py::return_value_policy::move
    );

    m.def("amgx_finalize", amgx_finalize);
}

}  // namespace cpmg::cpamgx
