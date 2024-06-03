#ifndef AMGCU_H
#define AMGCU_H

#include <chrono>

#include <pybind11/pybind11.h>


namespace cpmg
{

using Float = float;

using Clock = std::chrono::high_resolution_clock;

using TimePoint = Clock::time_point;

using Duration = std::chrono::duration<Float, std::milli>;

void init_py_module(pybind11::module_ & m);

}  // namespace cpmg


#endif  // AMGCU_H
