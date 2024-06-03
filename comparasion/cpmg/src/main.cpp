#include <pybind11/pybind11.h>

#include "cpmg/cpmg.h"


PYBIND11_MODULE(MODULE_NAME, m)
{
    cpmg::init_py_module(m);
}
