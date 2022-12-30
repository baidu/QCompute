#pragma once

#define EIGEN_USE_MKL_ALL
#define KEEP_REFORMAT_CODE_SAFE

#include <iostream>
#include <iomanip>
#include <limits>
#include <cmath>
#include <ctime>
// #include <boost/chrono.hpp>

#include "omp.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <Eigen/Dense>
#include <unsupported/Eigen/KroneckerProduct>
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/MatrixFunctions>

// #include <mkl.h>

#ifdef NDEBUG
#undef NDEBUG

#include <cassert>

#define NDEBUG
#else

#include <cassert>

#endif

using namespace std;
using namespace Eigen;
namespace py = pybind11;
using namespace pybind11::literals;

using MatrixXcdr = Matrix<dcomplex, Dynamic, Dynamic, RowMajor>;
using MatrixXcdrMap = Map<MatrixXcdr>;
using MatrixXdr = Matrix<double, Dynamic, Dynamic, RowMajor>;
using MatrixXdrMap = Map<MatrixXdr>;

template<typename T>
Map<Matrix<T, Dynamic, Dynamic, RowMajor>> py_array_to_matrix_map(const py::array_t<T> &arr) {
    py::buffer_info bi = arr.request();
    assert(bi.format == py::format_descriptor<T>::format());
    assert(bi.shape.size() == 2 || bi.shape.size() == 1);
    if (bi.shape.size() == 2) {
        Map<Matrix<T, Dynamic, Dynamic, RowMajor>> mm(static_cast<T *>(bi.ptr), bi.shape[0], bi.shape[1]);  // matrix
        return mm;
    } else if (bi.shape.size() == 1) {
        Map<Matrix<T, Dynamic, Dynamic, RowMajor>> mm(static_cast<T *>(bi.ptr), 1, bi.shape[0]);  // row vector
        return mm;
    } else {
        assert(false);
        return Map<Matrix<T, Dynamic, Dynamic, RowMajor>>(static_cast<T *>(nullptr), -1, -1);
    }
}

template<typename T>
inline T square(T x) {
    return x * x;
}

// cout vector
template<typename T>
ostream &operator<<(ostream &os, const vector<T> &vec) {
    for (auto &item: vec) {
        os << item << ", ";
    };
    return os;
}

// cout map
template<typename T1, typename T2>
ostream &operator<<(ostream &os, const map<T1, T2> &arr) {
    // json counts
    int count = 0;
    for (auto &item: arr) {
        os << "\"" << item.first << "\": " << item.second;
        if (++count != arr.size()) {
            os << ", ";
        }
        os << endl;
    }
    return os;
}
