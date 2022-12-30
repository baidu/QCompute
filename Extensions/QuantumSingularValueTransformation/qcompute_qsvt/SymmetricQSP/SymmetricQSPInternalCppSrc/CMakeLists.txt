cmake_minimum_required(VERSION 3.16)
project(SymmetricQSPInternalCpp)

# static link to c/c++ runtime for linux
if (UNIX)
    link_libraries(-static-libgcc -static-libstdc++)
    # link_libraries(-static)  # not allowed to build an so
    set(BUILD_SHARED_LIBS OFF)
    set(CMAKE_EXE_LINKER_FLAGS "-static")
    # add_definitions(-fPIC)
endif (UNIX)

set(CMAKE_CXX_STANDARD 20)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/../)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

# pybind11
message("---------- pybind11 ----------")
include(FetchContent)
FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://gitee.com/mirrors/pybind11  # https://github.com/pybind/pybind11
        GIT_TAG v2.6.0
)
FetchContent_GetProperties(pybind11)
if (NOT pybind11_POPULATED)
    FetchContent_Populate(pybind11)
    message("pybind11: ${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR}")
    add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
endif ()

# gcc's march native
# please build me on the server
#if (UNIX)
#    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
#    set(CMAKE_CXX_FLAGS "-march=native -msse4.2")
#    set(CMAKE_C_FLAGS "-march=native")
#endif (UNIX)

# mkl
message("---------- mkl ----------")
# https://software.intel.com/content/www/us/en/develop/articles/intel-mkl-link-line-advisor.html
# windows: mkl_intel_ilp64.lib mkl_intel_thread.lib mkl_core.lib libiomp5md.lib
# linux: -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -liomp5 -lpthread -lm -ldl
if (WIN32)
    set(MKL_ROOT "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries_2020.4.311/windows/mkl")
    set(MKL_INCLUDE_DIRS ${MKL_ROOT}/include)
    set(MKL_LIBRARIES ${MKL_ROOT}/lib/intel64_win/mkl_intel_lp64.lib ${MKL_ROOT}/lib/intel64_win/mkl_intel_thread.lib ${MKL_ROOT}/lib/intel64_win/mkl_core.lib ${MKL_ROOT}/../compiler/lib/intel64_win/libiomp5md.lib)
    message("MKL: ${MKL_INCLUDE_DIRS}, ${MKL_LIBRARIES}")
    include_directories(${MKL_INCLUDE_DIRS})
    link_libraries(${MKL_LIBRARIES})
endif (WIN32)
if (UNIX)
    find_package(MKL REQUIRED)
    message("MKL: ${MKL_INCLUDE_DIRS}, ${MKL_LIBRARIES}")
    message("MKL_ROOT: ${MKL_ROOT}")
    include_directories(${MKL_INCLUDE_DIRS})
    link_libraries(${MKL_ROOT}/lib/intel64/libmkl_intel_lp64.a ${MKL_ROOT}/lib/intel64/libmkl_gnu_thread.a ${MKL_ROOT}/lib/intel64/libmkl_core.a)  # link_libraries(${MKL_LIBRARIES})

    # solve run-time error: undefined symbol: GOMP_critical_end
    get_filename_component(MKL_ROOT_PARENT ${MKL_ROOT} DIRECTORY)
    message("MKL_ROOT_PARENT: ${MKL_ROOT_PARENT}")
    if (EXISTS ${MKL_ROOT_PARENT}/lib/intel64/libiomp5.a)
        message("find ${MKL_ROOT_PARENT}/lib/intel64/libiomp5.a")
        link_libraries(${MKL_ROOT_PARENT}/lib/intel64/libiomp5.a)
    elseif (EXISTS ${MKL_ROOT_PARENT}/compiler/lib/intel64/libiomp5.a)
        message("find ${MKL_ROOT_PARENT}/compiler/lib/intel64/libiomp5.a")
        link_libraries(${MKL_ROOT_PARENT}/compiler/lib/intel64/libiomp5.a)
    else ()
        message(FATAL_ERROR "can not find intel libiomp5.a")
    endif ()

    # dl, used by mkl static lib
    link_libraries(${CMAKE_DL_LIBS})
endif (UNIX)

# # boost
# SET(CMAKE_INCLUDE_PATH ${CMAKE_INCLUDE_PATH} "D:/sdk/boost_1_74_0")
# SET(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} "D:/sdk/boost_1_74_0/lib64-msvc-14.1")
# set(Boost_USE_STATIC_LIBS ON)
# set(Boost_USE_MULTITHREADED ON)
# find_package(Boost COMPONENTS program_options date_time chrono)
# message("Boost: ${Boost_INCLUDE_DIR}, ${Boost_LIBRARIES}")
# include_directories(${Boost_INCLUDE_DIRS})
# link_libraries(${Boost_LIBRARIES})  # ubuntu
# # link_libraries(libboost_program_options.so)  # centos

# eigen
message("---------- eigen ----------")
include(FetchContent)
FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen
        GIT_TAG 3.4
)
FetchContent_GetProperties(eigen)
if (NOT eigen_POPULATED)
    FetchContent_Populate(eigen)
    message("eigen: ${eigen_SOURCE_DIR} ${eigen_BINARY_DIR}")
    # add_subdirectory(${eigen_SOURCE_DIR} ${eigen_BINARY_DIR})
    include_directories(${eigen_SOURCE_DIR})
endif ()

# omp
find_package(OpenMP)
if (NOT OpenMP_CXX_FOUND)
    message(FATAL_ERROR "OpenMP not found")
endif ()
link_libraries(OpenMP::OpenMP_CXX)

# final
message("---------- final ----------")
pybind11_add_module(SymmetricQSPInternalCpp main.cpp)
