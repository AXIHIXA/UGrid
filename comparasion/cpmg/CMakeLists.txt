cmake_minimum_required(VERSION 3.27)

project(CPMG CUDA CXX)

# Ensure correct CUDA architecture
# NVIDIA Geforce RTX 2080 Ti has Compute Capability 7.5
# https://developer.nvidia.com/cuda-gpus
# https://stackoverflow.com/questions/67794606/cmake-cuda-architecture-flags
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_VERBOSE_MAKEFILE ON)

find_package(Boost REQUIRED)

set(Python_ROOT_DIR "$ENV{HOME}/opt/anaconda3/envs/py3/")
find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED CONFIG HINTS "${Python_ROOT_DIR}/lib/python3.10/site-packages/pybind11/share/cmake/")

set(CAFFE2_USE_CUDNN ON)
find_package(Torch REQUIRED CONFIG HINTS "$ENV{HOME}/opt/anaconda3/envs/py3/lib/python3.10/site-packages/torch/share/cmake/")

# AMGCL build conflicts with libfmt, go header-only instead
set(AMGCL_INCLUDE_DIRS "$ENV{HOME}/lib/amgcl/")

set(AMGX_INCLUDE_DIRS "$ENV{HOME}/lib/amgx/include/")
set(AMGX_LIBRARIES "$ENV{HOME}/lib/amgx/build/libamgx.a")

set(CPMG
        include/cpmg/assembler.h
        include/cpmg/cpamgcl.h
        include/cpmg/cpamgx.h
        include/cpmg/cpgmg.h
        include/cpmg/cpmg.h
        include/cpmg/cuda_utils.h
        src/cpmg/assembler.cpp
        src/cpmg/cpamgcl.cu
        src/cpmg/cpamgx.cu
        src/cpmg/cpgmg.cu
        src/cpmg/cpmg.cpp
)

set(MAIN
        src/main.cpp
)

set(ALL_SOURCE_FILES
        ${CPMG}
        ${MAIN}
)

set(ALL_INCLUDE_DIRS
        ${AMGCL_INCLUDE_DIRS}
        ${AMGX_INCLUDE_DIRS}
        ${fmt_INCLUDE_DIRS}
        include
)

set(ALL_LIBRARIES
        ${AMGX_LIBRARIES}
        Boost::boost
        cublas
        cublasLt
        cusolver
        cusparse
        nvToolsExt  # Solves AMGX CE undefined symbol: nvtxRangePushEx
        pybind11::module
        tbb
        torch
)

set(PYMODULE cpmg)

set(ALL_COMPILE_DEFS
        -DMODULE_NAME=${PYMODULE}
)

set(ALL_COMPILE_OPTS

)

pybind11_add_module(${PYMODULE} ${ALL_SOURCE_FILES})
target_compile_definitions(${PYMODULE} PUBLIC ${ALL_COMPILE_DEFS})
target_compile_options(${PYMODULE} PUBLIC ${ALL_COMPILE_OPTS})
target_include_directories(${PYMODULE} PUBLIC ${ALL_INCLUDE_DIRS})
target_link_libraries(${PYMODULE} PUBLIC ${ALL_LIBRARIES})


# Note: Do NOT turn CUDA_SEPARABLE_COMPILATION on,
# otherwise there will be undefined symbols in the Python module!
#set_target_properties(${PYMODULE} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
