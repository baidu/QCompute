#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Baidu, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Module for utility functions in quantum package.
"""

from argparse import ArgumentTypeError
from typing import List, Union, Any, Tuple

import numpy
from numpy import conj, transpose, log, sqrt, power, pi, arcsin, exp, matmul, arctan, sin, cos, real
from numpy.linalg import linalg
import matplotlib.pyplot as plt

from Extensions.QuantumNetwork.qcompute_qnet import EPSILON
from Extensions.QuantumNetwork.qcompute_qnet.quantum.gate import Gate

__all__ = [
    "COLOR_TABLE",
    "kron",
    "complex_log",
    "decompose_to_u_gate",
    "dagger",
    "to_projector",
    "to_superoperator",
    "find_keys_by_value",
    "print_progress",
    "plot_results",
]


# Color table for terminal print
COLOR_TABLE = {
    "red": "\033[31m",
    "blue": "\033[34m",
    "yellow": "\033[33m",
    "green": "\033[32m",
    "purple": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
    "end": "\033[0m",
}


def kron(matrices: List[numpy.ndarray]) -> numpy.ndarray:
    r"""Take the kronecker product of a list of matrices.

    .. math::

        [A, B, C, \cdots] \to A \otimes B \otimes C \otimes \cdots

    Args:
        matrices (List[numpy.ndarray]): a list of matrix to product

    Returns:
        numpy.ndarray: the kronecker result
    """
    if not isinstance(matrices, list):
        raise ArgumentTypeError(f"Input {matrices} should be a list.")

    result = matrices[0]
    if len(matrices) > 1:  # kron together
        for i in range(1, len(matrices)):
            result = numpy.kron(result, matrices[i])
    return result


def complex_log(complex_number: complex) -> complex:
    r"""Calculate the logarithm of a complex number.

    .. math::

        \text{Let a complex number be } c = a + b i, \text{. The logarithm is obtained by: }

        log(c) = log (a + b i) = i \times \arctan(b / a)

    Args:
        complex_number (complex): a complex number

    Returns:
        complex: logarithm of the complex number
    """
    # If the type is ``float`` or ``int``, add a small imaginary part
    if not isinstance(complex_number, complex):
        complex_number = complex(complex_number, EPSILON)
    # If the type is ``complex``, check if the real part and the imaginary part are zero
    else:
        if complex_number.real == 0:
            complex_number += EPSILON
        if complex_number.imag == 0:
            complex_number += EPSILON * 1j

    # Zero
    if abs(complex_number) <= EPSILON:
        raise ArgumentTypeError(
            f"Invalid complex number: ({complex_number})!\n" "The length of a complex number must be larger than 0."
        )

    real_in = complex_number.real
    real_phase = real_in / abs(real_in)
    imag_in = complex_number.imag
    imag_phase = imag_in / abs(imag_in)

    real_out = log(sqrt(power(real_in, 2) + power(imag_in, 2)))

    # If the real part is zero
    if abs(real_in) < EPSILON:
        imag_out = pi / 2 if imag_phase > 0 else 3 * pi / 2 / 2

    # If the imaginary part is zero
    elif abs(imag_in) < EPSILON:
        imag_out = 0 if real_phase > 0 else pi

    # If both the real part and the imaginary part are not zero
    else:
        if real_phase > 0 and imag_phase > 0:
            imag_out = arcsin(imag_in / exp(real_out))
        elif real_phase > 0 > imag_phase:
            imag_out = arcsin(imag_in / exp(real_out))
        else:
            imag_out = pi - arcsin(imag_in / exp(real_out))

    return real_out + imag_out * 1j


def decompose_to_u_gate(u_mat: numpy.ndarray) -> Tuple[float, float, float]:
    r"""Decompose a 2 X 2 unitary gate to the product of rotation gates.

    Warning：
        Unlike the commonly used ``U3`` gate,
        a unitary gate is decomposed to the product of 'Rz, Rx, Rz' in this method.

    The matrix form is:

    .. math::

        U(\theta, \phi, \lambda) = Rz(\phi) Rx(\theta) Rz (\lambda)
        = \begin{pmatrix}
        e^{-i(\lambda / 2 + \phi / 2)} cos(\theta / 2)      &   -e^{i(\lambda / 2 - \phi / 2)} sin(\theta / 2) i \\
        -e^{-i(\lambda / 2 - \phi / 2)} sin(\theta / 2) i   &    e^{i(\lambda / 2 + \phi / 2)} cos(\theta / 2)
        \end{pmatrix}

    Args:
        u_mat (numpy.ndarray): the unitary gate to be decomposed

    Returns:
        Tuple[float, float, float]: a tuple containing the rotation angles of the Rx gate,
        the left Rz gate and the right Rz gate
    """
    if not isinstance(u_mat, numpy.ndarray):
        raise ArgumentTypeError(
            f"Invalid matrix ({u_mat}) with the type: `{type(u_mat)}`!\n"
            "Only `numpy.ndarray` is supported as the type of the matrix."
        )
    if u_mat.shape != (2, 2):
        raise ArgumentTypeError(
            f"Invalid matrix ({u_mat}) with the shape: {u_mat.shape}!\n"
            "Only (2, 2) is supported as the shape of the matrix."
        )

    u_error = linalg.norm(matmul(conj(transpose(u_mat)), u_mat) - Gate.I())
    is_unitary = u_error < EPSILON
    if not is_unitary:
        raise ArgumentTypeError(f"Invalid matrix ({u_mat}) with the norm: {u_error}! Only unitary matrix is supported.")

    a = u_mat[0][0]
    b = u_mat[0][1]
    d = u_mat[1][0]
    e = u_mat[1][1]

    a_is_zero = abs(a) <= EPSILON
    d_is_zero = abs(d) <= EPSILON

    if a_is_zero:
        theta = pi
        gamma = 0
        phi = complex_log(b / d) * 1j

    elif d_is_zero:
        gamma = 0
        theta = 0
        phi = complex_log(e / a) * -1j

    else:
        gamma = (complex_log(e / a) + complex_log(b / d)) * -1j / 2
        phi = (complex_log(e / a) - complex_log(b / d)) * -1j / 2
        theta = arctan(b / a * sin(gamma) + b / a * cos(gamma) * 1j) * 2

    return real(theta), real(phi), real(gamma)


def dagger(matrix: numpy.ndarray) -> numpy.ndarray:
    r"""Compute the conjugate transpose of a matrix.

    Args:
        matrix (numpy.ndarray): input matrix

    Returns:
        numpy.ndarray: output matrix
    """
    return transpose(conj(matrix))


def to_projector(vec: numpy.ndarray) -> numpy.ndarray:
    r"""Convert a column vector to a projector.

    Args:
        vec (numpy.ndarray): input vector

    Returns:
        numpy.ndarray: equivalent projector
    """
    return vec @ dagger(vec)


def to_superoperator(kraus_list: List[numpy.ndarray]) -> numpy.ndarray:
    r"""Convert a list of kraus operators to a Liouville-superoperator.

    ..math::
        S = \sum_i \bar E_i \otimes E_i

    Args:
        kraus_list (List[numpy.ndarray]): a list of kraus operators

    Returns:
        numpy.ndarray: Liouville-superoperator
    """
    superop = 0
    for kraus in kraus_list:
        superop += kron([conj(kraus), kraus])

    return superop


def find_keys_by_value(d: dict, value: Any) -> List[Any]:
    r"""Find all the keys in a dict whose values match the given value.

    Args:
        d (dict): a dict to search
        value (Any): a value to match

    Returns:
        list: resulting list
    """
    return [k for k, v in d.items() if v == value]


def print_progress(current_progress: Union[float, int], progress_name: str, track=True) -> None:
    r"""Print a progress bar.

    Args:
        current_progress (Union[float, int]): current progress percentage
        progress_name (str): name of the progress bar
        track (bool): whether to print the progress on the terminal
    """
    if current_progress < 0 or current_progress > 1:
        raise ArgumentTypeError(
            f"Invalid current progress: ({current_progress})!\n"
            f"'current_progress' must be a percentage between 0 and 1"
        )
    if not isinstance(track, bool):
        raise ArgumentTypeError(
            f"Invalid parameter ({track}) with the type `{type(track)}`!\n"
            f"Only `bool` is supported as the parameter."
        )
    if track:
        print(
            "\r" + f"{progress_name.ljust(30)}"
            f"|{'■' * int(50 * current_progress):{50}s}| "
            f"\033[94m {'{:6.2f}'.format(100 * current_progress)}% \033[0m ",
            flush=True,
            end="",
        )
        if current_progress == 1:
            print(" (Done)")


def plot_results(result_lst: list, legend_lst: list, title=None, xlabel=None, ylabel=None, xticklabels=None) -> None:
    r"""Plot the results for comparison.

    Args:
        result_lst (list): results to plot
        legend_lst (list): legends of results
        title (str, optional): title of the figure
        xlabel (str, optional): xlabel of the figure
        ylabel (str, optional): ylabel of the figure
        xticklabels (list, optional): xticklabels of the figure
    """
    title = "Sampling results" if title is None else title
    xlabel = "Outcomes" if xlabel is None else xlabel
    ylabel = "Counts" if ylabel is None else ylabel

    if not isinstance(result_lst, List):
        raise ArgumentTypeError(
            f"Invalid dictionary list ({result_lst}) with the type `{type(result_lst)}`!\n"
            "Only `List` is supported as the dictionary list."
        )
    if not isinstance(legend_lst, List):
        raise ArgumentTypeError(
            f"Invalid bar labels ({legend_lst}) with the type `{type(legend_lst)}`!\n"
            "Only `List` is supported as the bar labels."
        )
    if len(result_lst) != len(legend_lst):
        raise ArgumentTypeError(
            f"Invalid dictionary list ({result_lst}) and bar labels ({legend_lst})!\n"
            "Please check your input as the number of dictionaries and bar labels must be the same."
        )

    # Cross-check the dictionaries and complete the empty results
    key_lst = []
    for result in result_lst:
        key_lst += result.keys()
    key_set = set(key_lst)
    new_dict_lst = []
    for result in result_lst:
        key_to_complete = key_set.difference(result.keys())
        for key in key_to_complete:
            result[key] = 0
        from Extensions.QuantumNetwork.qcompute_qnet.quantum.circuit import Circuit

        new_dict = Circuit.sort_results(result)
        new_dict_lst.append(new_dict)

    result_lst = new_dict_lst

    bars_num = len(result_lst)
    bar_width = 1 / (bars_num + 1)
    plt.ion()
    plt.figure()
    for bar_num in range(bars_num):
        plot_dict = result_lst[bar_num]
        # Obtain the y label and xticks in order
        keys = list(plot_dict.keys())
        values = list(plot_dict.values())
        xlen = len(keys)
        xticks = [((bar_num) / (bars_num + 1)) + xnum for xnum in range(xlen)]
        # Plot bars
        plt.bar(xticks, values, width=bar_width, align="edge", label=legend_lst[bar_num])
        plt.yticks()
    if xticklabels is None:
        plt.xticks(list(range(xlen)), keys, rotation=90)
    else:
        assert len(xticklabels) == xlen, "the 'xticklabels' should have the same length with 'x' length."
        plt.xticks(list(range(xlen)), xticklabels, rotation=90)
    plt.legend()
    plt.title(title, fontproperties="SimHei", fontsize="x-large")
    plt.xlabel(xlabel, fontproperties="SimHei")
    plt.ylabel(ylabel, fontproperties="SimHei")
    plt.ioff()
    plt.show()
