#ifndef CPAMGCL_H
#define CPAMGCL_H

#include <pybind11/pybind11.h>


namespace cpmg::cpamgcl
{

void init_py_module(pybind11::module_ & m);

} // namespace cpmg::cpamgcl


#endif  // CPAMGCL_H
