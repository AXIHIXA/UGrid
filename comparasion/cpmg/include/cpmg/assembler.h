#ifndef ASSEMBLER_H
#define ASSEMBLER_H

#include <cstddef>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "cpmg/cpmg.h"


namespace cpmg
{

int assemble2DPoissonProblem(
        const pybind11::array_t<bool> & boundaryMask,
        const pybind11::array_t<Float> & laplacian,
        const pybind11::array_t<Float> & boundaryValue,
        std::vector<Float> & val,
        std::vector<int> & col,
        std::vector<int> & ptr,
        std::vector<Float> & rhs
);


}  // namespace cpmg


#endif  // ASSEMBLER_H
