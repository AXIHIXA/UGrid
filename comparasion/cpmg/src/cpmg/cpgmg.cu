#include <iostream>

#include <torch/torch.h>
#include <pybind11/numpy.h>

#include "cpmg/cpgmg.h"
#include "cpmg/cpmg.h"
#include "cpmg/cuda_utils.h"


namespace cpmg::cpgmg
{

namespace
{

void
solve_gmg(
        const pybind11::array_t<Float> & boundaryMask,
        const pybind11::array_t<Float> & laplacian,
        const pybind11::array_t<Float> & boundaryValue,
        const pybind11::array_t<Float> & initialGuess,
        Float relativeTolerance
)
{
    torch::Tensor m = torch::from_blob(
            const_cast<Float *>(boundaryMask.data()),
            {boundaryMask.shape(0), boundaryMask.shape(1)}
    ).clone().to(torch::kCUDA);

    torch::Tensor f = torch::from_blob(
            const_cast<Float *>(laplacian.data()),
            {laplacian.shape(0), laplacian.shape(1)}
    ).clone().to(torch::kCUDA);

    torch::Tensor b = torch::from_blob(
            const_cast<Float *>(boundaryValue.data()),
            {boundaryValue.shape(0), boundaryValue.shape(1)}
    ).clone().to(torch::kCUDA);

    torch::Tensor x = torch::from_blob(
            const_cast<Float *>(initialGuess.data()),
            {initialGuess.shape(0), initialGuess.shape(1)}
    ).clone().to(torch::kCUDA);

//    std::cout << "m = \n" << m << "\n\n";
//    std::cout << "f = \n" << f << "\n\n";
//    std::cout << "b = \n" << b << "\n\n";
//    std::cout << "x = \n" << x << "\n\n";


}

}  // namespace anonmyous


void init_py_module(pybind11::module_ & m)
{
    namespace py = pybind11;
    using py::literals::operator""_a;

    // This statement must be in the same source file of the function referenced!
    // Otherwise, there will be undefined symbols.
    m.def("solve_gmg",
          solve_gmg,
          "boundary_mask"_a,
          "laplacian"_a,
          "boundary_value"_a,
          "initial_guess"_a,
          "relative_tolerance"_a,
          py::return_value_policy::move
    );
}

}  // namespace cpmg::cpgmg
