#include "cpmg/cpamgcl.h"
#include "cpmg/cpamgx.h"
#include "cpmg/cpgmg.h"
#include "cpmg/cpmg.h"


namespace cpmg
{

void init_py_module(pybind11::module_ & m)
{
    cpamgcl::init_py_module(m);
    cpamgx::init_py_module(m);
    cpgmg::init_py_module(m);
}

}  // namespace cpmg
