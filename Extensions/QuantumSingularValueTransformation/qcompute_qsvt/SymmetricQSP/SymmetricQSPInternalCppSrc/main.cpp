#pragma once

#include "common.hpp"
// #include "profiler.hpp"

// def __func_expiZ_map(vec_phi):
vector<MatrixXcdr> __func_expiZ_map(vector<double> vec_phi) {
    // profilerBegin("__func_expiZ_map");

    // list_mat = []
    // for float_phi in vec_phi:
    //     comp_entry = np.exp(1j * float_phi)
    //     list_mat.append(np.array([[comp_entry, 0], [0, comp_entry.conjugate()]]))
    vector<MatrixXcdr> list_mat;
    for (auto &float_phi: vec_phi) {
        dcomplex comp_entry = exp(dcomplex(0.0, 1.0) * float_phi);
        MatrixXcdr mat(2, 2);
        mat << comp_entry, 0,
                0, conj(comp_entry);
        list_mat.push_back(mat);
    }

    // profilerEnd("__func_expiZ_map");

    // return np.array(list_mat)
    return list_mat;
}

// def func_symQSP_A_map(vec_phi, list_Wx, bool_parity):
py::array_t<double>
func_symQSP_A_map(vector<double> vec_phi, vector<py::array_t<dcomplex>> list_Wx_np, int bool_parity) {
    // profilerBegin("func_symQSP_A_map");

    // list_Wx N*2*2 dcomple -> vector of eigen matirx
    vector<MatrixXcdrMap> list_Wx;
    for (auto &i: list_Wx_np) {
        list_Wx.push_back(py_array_to_matrix_map(i));
    }

    // int_n = len(list_Wx)  # 一般用 int_n 记 x 或 Wx 的数量
    // list_expiZ = __func_expiZ_map(vec_phi)
    // list_symQSP_half = [list_expiZ[0][:1].copy() for _ in range(int_n)]  # 初始化为 [list_expiZ[0], ... , list_expiZ[0]]
    int int_n = list_Wx.size();
    int int_d = vec_phi.size();
    vector<MatrixXcdr> list_expiZ = __func_expiZ_map(vec_phi);
    vector<MatrixXcdr> list_symQSP_half(int_n, list_expiZ[0](seq(0, 0), Eigen::all));

    // # 以下二重循环的外层循环可以并行计算
    // for idx_x in range(int_n):
    //     for mat_expiZ in list_expiZ[1:]:
    //         list_symQSP_half[idx_x] = (list_symQSP_half[idx_x] @ list_Wx[idx_x]) @ mat_expiZ
    //         # 这里 mat_expiZ 是对角阵，是不是有办法通过换成其他矩阵乘法加速？
    // # 如此得到了 func_QSP_A_map 的 QSP 矩阵
    // # 接下来分奇偶性计算func_symQSP_A_map，这里没有矩阵计算了，只是一步代入公式
#pragma omp parallel for
    for (int idx_x = 0; idx_x < int_n; idx_x++) {
        for (int mat_expiZ_idx = 1; mat_expiZ_idx < int_d; mat_expiZ_idx++) {
            auto mat_expiZ = list_expiZ[mat_expiZ_idx];
            list_symQSP_half[idx_x] = (list_symQSP_half[idx_x] * list_Wx[idx_x]) * mat_expiZ;
        }
    }

    // vec_A = np.zeros(int_n)  # 使用 vec_A 储存返回值
    MatrixXdr vec_A = MatrixXdr::Zero(1, int_n);

    // if bool_parity == 0:
    //     for idx_x in range(int_n):
    //         mat_half0 = list_symQSP_half[idx_x][0]
    //         # 之所以叫 mat_half0，是因为其为对称化后前一半 vec_phi 所对应的 QSP 矩阵的第 0 行
    //         # 偶情形调用 list_expiZ[-1] 的两个对角元
    //         vec_A[idx_x] = (mat_half0[0] ** 2 * list_expiZ[-1][1, 1] +
    //                         mat_half0[1] ** 2 * list_expiZ[-1][0, 0]).real
    if (bool_parity == 0) {
        for (int idx_x = 0; idx_x < int_n; idx_x++) {
            auto mat_half0 = list_symQSP_half[idx_x].row(0);
            vec_A(idx_x) = (square(mat_half0(0)) * list_expiZ[list_expiZ.size() - 1](1, 1) +
                            square(mat_half0(1)) * list_expiZ[list_expiZ.size() - 1](0, 0)).real();
        }
    }
        // todo: the else branch was never tested
        // else:
        //     for idx_x in range(int_n):
        //         mat_half0 = list_symQSP_half[idx_x][0]
        //         # 奇情形区分 xj 调用 Wx(xj) 的第 0 行
        //         vec_A[idx_x] = ((mat_half0[0] ** 2 + mat_half0[1] ** 2) * list_Wx[idx_x][0, 0] +
        //                         2 * mat_half0[0] * mat_half0[1] * list_Wx[idx_x][0, 1]).real
    else {
        for (int idx_x = 0; idx_x < int_n; idx_x++) {
            auto mat_half0 = list_symQSP_half[idx_x].row(0);
            vec_A(idx_x) = ((square(mat_half0(0)) + square(mat_half0(1))) * list_Wx[idx_x](0, 0) +
                            2.0 * mat_half0(0) * mat_half0(1) * list_Wx[idx_x](0, 1)).real();
        }
    }

    // return vec_A
    py::array_t<double> vec_A_np({int_n});
    py_array_to_matrix_map(vec_A_np) = vec_A;
    // profilerEnd("func_symQSP_A_map");
    return vec_A_np;
}

/*
// def func_symQSP_gradA_map(vec_phi, list_Wx, bool_parity):  # checked
py::tuple func_symQSP_gradA_map(vector<double> vec_phi, vector<py::array_t<dcomplex>> list_Wx_np, int bool_parity) {
    // profilerBegin("func_symQSP_gradA_map");

    // list_Wx N*2*2 dcomple -> vector of eigen matirx
    vector<MatrixXcdrMap> list_Wx;
    for (auto &i: list_Wx_np) {
        list_Wx.push_back(py_array_to_matrix_map(i));
    }

    // int_d = len(vec_phi)
    // int_n = len(list_Wx)
    // list_expiZ = __func_expiZ_map(vec_phi)
    int int_d = vec_phi.size();
    int int_n = list_Wx.size();
    vector<MatrixXcdr> list_expiZ = __func_expiZ_map(vec_phi);

    // reg_mat_front = np.zeros((int_n, int_d, 2, 2), dtype=complex)  # 这里选择了先创建 0 列表再赋值，也许逐步 append 会更快？
    // # 以下二重循环的外层循环可以并行计算
    // for idx_x in range(int_n):
    //     reg_mat_front[idx_x, 0] = np.eye(2)
    //     for idx_phi in range(int_d - 1):
    //         reg_mat_front[idx_x, idx_phi + 1] = reg_mat_front[idx_x, idx_phi] @ (list_expiZ[idx_phi] @ list_Wx[idx_x])
    //         # 这里 list_expiZ[idx_phi] 是对角阵，是不是有办法通过换成其他矩阵乘法加速？
    vector<vector<MatrixXcdr>> reg_mat_front(int_n);
#pragma omp parallel for
    for (int idx_x = 0; idx_x < int_n; idx_x++) {
        reg_mat_front[idx_x].push_back(MatrixXcdr::Identity(2, 2));
        for (int idx_phi = 0; idx_phi < int_d - 1; idx_phi++) {
            reg_mat_front[idx_x].push_back(reg_mat_front[idx_x][idx_phi] * (list_expiZ[idx_phi] * list_Wx[idx_x]));
        }
    }

    // reg_mat_mid = np.zeros((int_n, int_d, 2, 2), dtype=complex)  # 这里选择了先创建 0 列表再赋值，也许逐步 append 会更快？
    // for idx_x in range(int_n):
    //     reg_mat_mid[idx_x, -1] = list_expiZ[-1].copy()
    vector<vector<MatrixXcdr>> reg_mat_mid(int_n, vector<MatrixXcdr>(int_d, MatrixXcdr::Zero(2, 2)));
    for (int idx_x = 0; idx_x < int_n; idx_x++) {
        reg_mat_mid[idx_x][list_expiZ.size() - 1] = list_expiZ[list_expiZ.size() - 1];
    }

    // # 以下二重循环的外层循环可以并行计算
    // for idx_x in range(int_n):
    //     for idx_phi in range(-2, -int_d - 1, -1):
    //         reg_mat_mid[idx_x, idx_phi] = (list_expiZ[idx_phi] @ list_Wx[idx_x]) @ reg_mat_mid[idx_x, idx_phi + 1]
    //         # 这里 list_expiZ[idx_phi] 是对角阵，是不是有办法通过换成其他矩阵乘法加速？
#pragma omp parallel for
    for (int idx_x = 0; idx_x < int_n; idx_x++) {
        for (int idx_phi = -2; idx_phi > -int_d - 1; idx_phi--) {
            reg_mat_mid[idx_x][reg_mat_mid.size() + idx_phi] =
                    (list_expiZ[list_expiZ.size() + idx_phi] * list_Wx[idx_x]) *
                    reg_mat_mid[idx_x][list_expiZ.size() + (idx_phi + 1)];
        }
    }

    // vec_A = np.zeros(int_n)  # 初始化 vec_A 寄存器
    // mat_gradA = np.zeros((int_n, int_d))  # 初始化 mat_gradA 寄存器
    MatrixXdr vec_A = MatrixXdr::Zero(1, int_n);
    MatrixXdr mat_gradA = MatrixXdr::Zero(int_n, int_d);

    // if bool_parity == 0:
    //     for idx_x in range(int_n):
    //         for idx_phi in range(int_d - 1):
    //             # 引入两个 temp 矩阵只是为了代码增加可读性，直接代入下面公式可就太长了
    //             mat_temp_front = reg_mat_front[idx_x, idx_phi, 0]
    //             mat_temp_mid = reg_mat_mid[idx_x, idx_phi]
    //             mat_gradA[idx_x, idx_phi] = (list_expiZ[-1][0, 0] * ((mat_temp_front[1] * mat_temp_mid[1, 1]) ** 2
    //                                                                  - (mat_temp_front[0] * mat_temp_mid[0, 1]) ** 2) +
    //                                          list_expiZ[-1][1, 1] * ((mat_temp_front[1] * mat_temp_mid[1, 0]) ** 2
    //                                                                  - (mat_temp_front[0] * mat_temp_mid[
    //                                 0, 0]) ** 2)).imag * 2
    //         # 这里 bool_parity 为偶时，对 vec_phi[-1] 求偏导的公式不符合前面的规律
    //         # 引入 temp 矩阵只是为了代码增加可读性，直接代入下面公式可就太长了
    //         mat_temp_front = reg_mat_front[idx_x, -1, 0]
    //         mat_gradA[idx_x, -1] = (list_expiZ[-1][1, 1] * mat_temp_front[1] ** 2
    //                                 - list_expiZ[-1][0, 0] * mat_temp_front[0] ** 2).imag
    //         mat_half0 = reg_mat_mid[idx_x, 0, 0]
    //         vec_A[idx_x] = (mat_half0[0] ** 2 * list_expiZ[-1][1, 1] + mat_half0[1] ** 2 * list_expiZ[-1][0, 0]).real
    if (bool_parity == 0) {
        for (int idx_x = 0; idx_x < int_n; idx_x++) {
            for (int idx_phi = 0; idx_phi < int_d - 1; idx_phi++) {
                auto f = reg_mat_front[idx_x][idx_phi].row(0);
                auto &m = reg_mat_mid[idx_x][idx_phi];
                auto tmp = list_expiZ[list_expiZ.size() - 1](0, 0) * (square(f(1) * m(1, 1)) - square(f(0) * m(0, 1))) +
                           list_expiZ[list_expiZ.size() - 1](1, 1) * (square(f(1) * m(1, 0)) - square(f(0) * m(0, 0)));
                mat_gradA(idx_x, idx_phi) = tmp.imag() * 2;
            }
            auto f = reg_mat_front[idx_x][reg_mat_front[idx_x].size() - 1].row(0);
            mat_gradA(idx_x, mat_gradA.cols() - 1) = (list_expiZ[list_expiZ.size() - 1](1, 1) * square(f(1)) -
                                                      list_expiZ[list_expiZ.size() - 1](0, 0) * square(f[0])).imag();
            auto mat_half0 = reg_mat_mid[idx_x][0].row(0);
            vec_A(idx_x) = (square(mat_half0(0)) * list_expiZ[list_expiZ.size() - 1](1, 1) +
                            square(mat_half0(1)) * list_expiZ[list_expiZ.size() - 1](0, 0)).real();
        }
    }
        // else:
        //     for idx_x in range(int_n):
        //         for idx_phi in range(int_d):
        //             # 引入两个 temp 矩阵只是为了代码增加可读性，直接代入下面公式可就太长了
        //             mat_temp_front = reg_mat_front[idx_x, idx_phi, 0]
        //             mat_temp_mid = reg_mat_mid[idx_x, idx_phi]
        //             mat_gradA[idx_x, idx_phi] = \
        //                 (list_Wx[idx_x][0, 0]
        //                  * (mat_temp_front[1] ** 2 * (mat_temp_mid[1, 0] ** 2 + mat_temp_mid[1, 1] ** 2)
        //                     - mat_temp_front[0] ** 2 * (mat_temp_mid[0, 0] ** 2 + mat_temp_mid[0, 1] ** 2))
        //                  + 2 * list_Wx[idx_x][0, 1] * (mat_temp_front[1] ** 2 * mat_temp_mid[1, 0] * mat_temp_mid[1, 1]
        //                                                - mat_temp_front[0] ** 2 * mat_temp_mid[0, 0] * mat_temp_mid[0, 1])
        //                  ).imag * 2
        //         mat_half0 = reg_mat_mid[idx_x, 0, 0]
        //         vec_A[idx_x] = ((mat_half0[0] ** 2 + mat_half0[1] ** 2) * list_Wx[idx_x][0, 0] +
        //                         2 * mat_half0[0] * mat_half0[1] * list_Wx[idx_x][0, 1]).real
    else {
        for (int idx_x = 0; idx_x < int_n; idx_x++) {
            for (int idx_phi = 0; idx_phi < int_d; idx_phi++) {
                auto f = reg_mat_front[idx_x][idx_phi].row(0);
                auto &m = reg_mat_mid[idx_x][idx_phi];
                auto tmp =
                        list_Wx[idx_x](0, 0) *
                        (square(f(1)) * (square(m(1, 0)) + square(m(1, 1))) -
                         square(f(0)) * (square(m(0, 0)) + square(m(0, 1)))) +
                        2.0 * list_Wx[idx_x](0, 1) *
                        (square(f(1)) * m(1, 0) * m(1, 1) - square(f(0)) * m(0, 0) * m(0, 1));
                mat_gradA(idx_x, idx_phi) = tmp.imag() * 2;
            }
            auto mat_half0 = reg_mat_mid[idx_x][0].row(0);
            vec_A(idx_x) = ((square(mat_half0(0)) + square(mat_half0(1))) * list_Wx[idx_x](0, 0) +
                            2.0 * mat_half0(0) * mat_half0(1) * list_Wx[idx_x](0, 1)).real();
        }
    }

    // return mat_gradA, vec_A
    py::array_t<double> mat_gradA_np({int_n, int_n});
    py_array_to_matrix_map(mat_gradA_np) = mat_gradA;
    py::array_t<double> vec_A_np({int_n});
    py_array_to_matrix_map(vec_A_np) = vec_A;
    // profilerEnd("func_symQSP_gradA_map");
    return pybind11::make_tuple(mat_gradA_np, vec_A_np);
}
*/

py::tuple func_symQSP_gradA_map(vector<double> vec_phi, vector<py::array_t<dcomplex>> list_Wx_np, int bool_parity) {
    // profilerBegin("func_symQSP_gradA_map");

    // list_Wx N*2*2 dcomple -> vector of eigen matirx
    vector<MatrixXcdrMap> list_Wx;
    for (auto &i: list_Wx_np) {
        list_Wx.push_back(py_array_to_matrix_map(i));
    }

    // int_d = len(vec_phi)
    // int_n = len(list_Wx)
    // list_expiZ = __func_expiZ_map(vec_phi)
    int int_d = vec_phi.size();
    int int_n = list_Wx.size();
    vector<MatrixXcdr> list_expiZ = __func_expiZ_map(vec_phi);

    // reg_mat_front = np.zeros((int_n, int_d, 2, 2), dtype=complex)  # 这里选择了先创建 0 列表再赋值，也许逐步 append 会更快？
    // reg_mat_QSP = np.zeros((int_n, 2, 2), dtype=complex)  # 存对称 QSP 矩阵的列表 (关于 x)
    // vec_A = np.zeros(int_n)  # 初始化 vec_A 寄存器
    vector<vector<MatrixXcdr>> reg_mat_front(int_n);
    vector<MatrixXcdr> reg_mat_QSP(int_n);
    MatrixXdr vec_A = MatrixXdr::Zero(1, int_n);

    // # 这里分奇偶的代码区别只在 reg_mat_QSP[idx_x] 的计算公式
    // if bool_parity == 0:
    //     # 以下二重循环的外层循环可以并行计算
    //     for idx_x in range(int_n):
    //         reg_mat_front[idx_x, 0] = np.eye(2)
    //         for idx_phi in range(int_d - 1):
    //             reg_mat_front[idx_x, idx_phi + 1] = reg_mat_front[idx_x, idx_phi] @ (list_expiZ[idx_phi] @ list_Wx[idx_x])
    //             # 这里 list_expiZ[idx_phi] 是对角阵，是不是有办法通过换成其他矩阵乘法加速？
    //         # 这一行和 else 情形的两行产生区别，inner(A,B) == A.dot(transpose(B))
    //         reg_mat_QSP[idx_x] = np.inner(reg_mat_front[idx_x, -1] @ list_expiZ[-1], reg_mat_front[idx_x, -1])
    //         vec_A[idx_x] = reg_mat_QSP[idx_x, 0, 0].real
    if (bool_parity == 0) {
#pragma omp parallel for
        for (int idx_x = 0; idx_x < int_n; idx_x++) {
            reg_mat_front[idx_x].push_back(MatrixXcdr::Identity(2, 2));
            for (int idx_phi = 0; idx_phi < int_d - 1; idx_phi++) {
                reg_mat_front[idx_x].push_back(reg_mat_front[idx_x][idx_phi] * (list_expiZ[idx_phi] * list_Wx[idx_x]));
            }
            reg_mat_QSP[idx_x] = (
                    (reg_mat_front[idx_x][reg_mat_front.size() - 1] * list_expiZ[list_expiZ.size() - 1]) *
                    (reg_mat_front[idx_x][reg_mat_front.size() - 1]).transpose()
            );
            vec_A(0, idx_x) = reg_mat_QSP[idx_x](0, 0).real();
        }
    }

        // else:
        //     # 以下二重循环的外层循环可以并行计算
        //     for idx_x in range(int_n):
        //         reg_mat_front[idx_x, 0] = np.eye(2)
        //         for idx_phi in range(int_d - 1):
        //             reg_mat_front[idx_x, idx_phi + 1] = reg_mat_front[idx_x, idx_phi] @ (list_expiZ[idx_phi] @ list_Wx[idx_x])
        //             # 这里 list_expiZ[idx_phi] 是对角阵，是不是有办法通过换成其他矩阵乘法加速？
        //         # 这两行和 if 情形的一行产生区别，inner(A,B) == A.dot(transpose(B))
        //         mat_temp = reg_mat_front[idx_x, -1] @ list_expiZ[-1]
        //         reg_mat_QSP[idx_x] = np.inner(mat_temp @ list_Wx[idx_x], mat_temp)
        //         vec_A[idx_x] = reg_mat_QSP[idx_x, 0, 0].real
    else {
#pragma omp parallel for
        for (int idx_x = 0; idx_x < int_n; idx_x++) {
            reg_mat_front[idx_x].push_back(MatrixXcdr::Identity(2, 2));
            for (int idx_phi = 0; idx_phi < int_d - 1; idx_phi++) {
                reg_mat_front[idx_x].push_back(reg_mat_front[idx_x][idx_phi] * (list_expiZ[idx_phi] * list_Wx[idx_x]));
            }
            auto mat_temp = reg_mat_front[idx_x][reg_mat_front.size() - 1] * list_expiZ[reg_mat_front.size() - 1];
            reg_mat_QSP[idx_x] = (
                    (mat_temp * list_Wx[idx_x]) *
                    mat_temp.transpose()
            );
            vec_A(0, idx_x) = reg_mat_QSP[idx_x](0, 0).real();
        }
    }

    // mat_gradA = np.zeros((int_n, int_d))  # 初始化 mat_gradA 寄存器
    MatrixXdr mat_gradA = MatrixXdr::Zero(int_n, int_d);

    // # 以下二重循环均可以并行计算
    // for idx_x in range(int_n):
    //     for idx_phi in range(int_d):
    //         # 引入 temp 矩阵只是为了代码增加可读性，直接代入下面公式可就太长了
    //         mat_temp_front = reg_mat_front[idx_x, idx_phi, :, 0]  # 这是一个 2 维复向量
    //         # (2 维向量) 乘 (2x2 矩阵) 乘 (2维向量) 得到一个数，怕 C 里面语法不一样？
    //         mat_gradA[idx_x, idx_phi] = -2 * (np.conjugate(mat_temp_front) @ reg_mat_QSP[idx_x] @ mat_temp_front).imag
#pragma omp parallel for
    for (int idx_x = 0; idx_x < int_n; idx_x++) {
        for (int idx_phi = 0; idx_phi < int_d; idx_phi++) {
            auto mat_temp_front = reg_mat_front[idx_x][idx_phi](Eigen::all, 0);
            auto mat_temp_front2 = (mat_temp_front.conjugate().transpose() * reg_mat_QSP[idx_x]) * mat_temp_front;
            mat_gradA(idx_x, idx_phi) = -2.0 * mat_temp_front2(0).imag();
        }
    }

    // if bool_parity == 0:
    //     # 以下循环均可以并行计算，或者化成向量计算？
    //     ###for idx_x in range(int_n):
    //     ###    mat_gradA[idx_x, -1] = mat_gradA[idx_x, -1] * 0.5
    //     mat_gradA[:, -1] *= 0.5 ### ???
    if (bool_parity == 0) {
        mat_gradA(Eigen::all, mat_gradA.cols() - 1) *= 0.5;
    }

    // return mat_gradA, vec_A
    py::array_t<double> mat_gradA_np({int_n, int_n});
    py_array_to_matrix_map(mat_gradA_np) = mat_gradA;
    py::array_t<double> vec_A_np({int_n});
    py_array_to_matrix_map(vec_A_np) = vec_A;
    // profilerEnd("func_symQSP_gradA_map");
    return pybind11::make_tuple(mat_gradA_np, vec_A_np);
}

KEEP_REFORMAT_CODE_SAFE PYBIND11_MODULE(SymmetricQSPInternalCpp, m) {
    // optional module docstring
    m.doc() = "no doc";

    // expose run function, and run keyword arguments and default arguments
    m.def("func_symQSP_A_map", &func_symQSP_A_map, "no doc",
          py::arg("vec_phi"),
          py::arg("list_Wx"),
          py::arg("bool_parity"));
    m.def("func_symQSP_gradA_map", &func_symQSP_gradA_map, "no doc",
          py::arg("vec_phi"),
          py::arg("list_Wx"),
          py::arg("bool_parity"));

    // m.def("profilerBegin", &profilerBegin, "no doc", py::arg("vec_phi"));
    // m.def("profilerEnd", &profilerEnd, "no doc", py::arg("vec_phi"));
    // m.def("profilerGetSummary", &profilerGetSummary, "no doc");
    // m.def("profilerReset", &profilerReset, "no doc");
}
