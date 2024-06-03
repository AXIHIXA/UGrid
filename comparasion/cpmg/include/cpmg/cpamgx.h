#ifndef CPAMGX_H
#define CPAMGX_H

#include <pybind11/pybind11.h>


namespace cpmg::cpamgx
{

void init_py_module(pybind11::module_ & m);

}  // namespace cpmg::cpamgx


#endif  // CPAMGX_H
