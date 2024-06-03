#ifndef CPGMG_H
#define CPGMG_H

#include <pybind11/pybind11.h>


namespace cpmg::cpgmg
{

void init_py_module(pybind11::module_ & m);

}  // namespace cpmg::cpgmg


#endif  // CPGMG_H
