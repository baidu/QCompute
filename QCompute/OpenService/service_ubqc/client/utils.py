# Copyright (c) 2022 Institute for Quantum Computing, Baidu Inc. All Rights Reserved.
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

# !/usr/bin/env python3

"""
utils
"""
FileErrorCode = 7

from numpy import array, pi, linalg, ndarray
from numpy import exp, log, cos, sin, sqrt, arcsin, arctan
from numpy import eye, real, power, matmul
from numpy import transpose as t
from numpy import conj as c

from QCompute.OpenService import ModuleErrorCode
from QCompute.QPlatform import Error

__all__ = ["plus_state",
           "minus_state",
           "zero_state",
           "one_state",
           "h_gate",
           "s_gate",
           "t_gate",
           "cz_gate",
           "cnot_gate",
           "pauli_gate",
           "rotation_gate",
           "u3_gate",
           "u_gate",
           "_log",
           "decompose"
           ]

eps = 1e-12  # error


def plus_state():
    r"""Define a ``plus`` state.

    The matrix form is:

    .. math::

        \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ 1 \end{bmatrix}

    Returns:
        ndarray: the ``ndarray`` form of a ``plus`` state

    Code Example:

    .. code-block:: python

        from QCompute.OpenService.service_ubqc.client.utils import plus_state
        print("The vector form of a plus state is: \n", plus_state())

    ::

        The vector form of a plus state is:
         [[0.70710678]
         [0.70710678]]
    """
    return array([[1 / sqrt(2)], [1 / sqrt(2)]], dtype='float64')


def minus_state():
    r"""Define a ``minus`` state.

    The matrix form is:

    .. math::

        \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ -1 \end{bmatrix}

    Returns:
        ndarray: the ``ndarray`` form of a ``minus`` state.

    Code Example:

    .. code-block:: python

        from QCompute.OpenService.service_ubqc.client.utils import minus_state
        print("The vector form of a minus state is: \n", minus_state())

    ::

        The vector form of a minus state is:
         [[ 0.70710678]
         [-0.70710678]]
    """
    return array([[1 / sqrt(2)], [-1 / sqrt(2)]], dtype='float64')


def zero_state():
    r"""Define a ``zero`` state.

    The matrix form is:

    .. math::

        \begin{bmatrix} 1 \\ 0 \end{bmatrix}

    Returns:
        ndarray: the ``ndarray`` form of a ``zero`` state.

    Code Example:

    .. code-block:: python

        from QCompute.OpenService.service_ubqc.client.utils import zero_state
        print("The vector form of a zero state is: \n", zero_state())

    ::

        The vector form of a zero state is:
         [[1.]
         [0.]]
    """
    return array([[1], [0]], dtype='float64')


def one_state():
    r"""Define a ``one`` state.

    The matrix form is:

    .. math::

        \begin{bmatrix} 0 \\ 1 \end{bmatrix}

    Returns:
        ndarray: the ``ndarray`` form of a ``one`` state.

    Code Example:

    .. code-block:: python

        from QCompute.OpenService.service_ubqc.client.utils import one_state
        print("The vector form of a one state is: \n", one_state())

    ::

        The vector form of a one state is:
         [[0.]
         [1.]]
    """
    return array([[0], [1]], dtype='float64')


def h_gate():
    r"""Define a ``Hadamard`` gate.

    The matrix form is:

    .. math::

        \frac{1}{\sqrt{2}} \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}

    Returns:
        ndarray: the ``ndarray`` form of a ``Hadamard`` gate.

    Code Example:

    .. code-block:: python

        from QCompute.OpenService.service_ubqc.client.utils import h_gate
        print("The matrix form of a Hadamard gate is: \n", h_gate())

    ::

        The matrix form of a Hadamard gate is:
         [[ 0.70710678  0.70710678]
         [ 0.70710678 -0.70710678]]
    """
    return (1 / sqrt(2)) * array([[1, 1], [1, -1]], dtype='float64')


def s_gate():
    r"""Define a ``S`` gate.

    The matrix form is:

    .. math::

        \begin{bmatrix} 1 & 0 \\ 0 & i \end{bmatrix}

    Returns:
        ndarray: the ``ndarray`` form of a ``S`` gate.

    Code Example:

    .. code-block:: python

        from QCompute.OpenService.service_ubqc.client.utils import s_gate
        print("The matrix form of a S gate is:\n", s_gate())

    ::

        The matrix form of a S gate is:
         [[1.+0.j 0.+0.j]
         [0.+0.j 0.+1.j]]
    """
    return array([[1, 0], [0, 1j]], dtype='complex128')


def t_gate():
    r"""Define a ``T`` gate.

    The matrix form is:

    .. math::

        \begin{bmatrix} 1 & 0 \\ 0 & e^{i \pi / 4} \end{bmatrix}

    Returns:
        ndarray: the ``ndarray`` form of a ``T`` gate.

    Code Example:

    .. code-block:: python

        from QCompute.OpenService.service_ubqc.client.utils import t_gate
        print("The matrix form of a T gate is: \n", t_gate())

    ::

        The matrix form of a T gate is:
         [[1.        +0.j         0.        +0.j        ]
         [0.        +0.j         0.70710678+0.70710678j]]
    """
    return array([[1, 0], [0, exp(1j * pi / 4)]], dtype='complex128')


def cz_gate():
    r"""Define a ``Control Z`` gate.

    The matrix form is:

    .. math::

        \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & -1 \end{bmatrix}

    Returns:
        ndarray: the ``ndarray`` form of a ``Control Z`` gate.

    Code Example:

    .. code-block:: python

        from QCompute.OpenService.service_ubqc.client.utils import cz_gate
        print("The matrix form of a CZ gate is: \n", cz_gate())

    ::

        The matrix form of a CZ gate is:
         [[ 1.  0.  0.  0.]
         [ 0.  1.  0.  0.]
         [ 0.  0.  1.  0.]
         [ 0.  0.  0. -1.]]
    """
    return array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, -1]], dtype='float64')


def cnot_gate():
    r"""Define a ``Control NOT (CNOT)`` gate.

    The matrix form is:

    .. math::

        \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{bmatrix}

    Returns:
        ndarray: the ``ndarray`` form of a ``Control NOT (CNOT)`` gate.

    Code Example:

    .. code-block:: python

        from QCompute.OpenService.service_ubqc.client.utils import cnot_gate
        print("The matrix form of a CNOT gate is: \n", cnot_gate())

    ::

        The matrix form of a CNOT gate is:
         [[1. 0. 0. 0.]
         [0. 1. 0. 0.]
         [0. 0. 0. 1.]
         [0. 0. 1. 0.]]
    """
    return array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 1],
                  [0, 0, 1, 0]], dtype='float64')


def pauli_gate(gate):
    r"""Define a ``Pauli`` gate.

    The matrix form of an Identity gate ``I`` is:

    .. math::

        \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}

    The matrix form of a Pauli gate ``X`` is:

    .. math::

        \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}

    The matrix form of a Pauli gate ``Y`` is:

    .. math::

        \begin{bmatrix} 0 & - i \\ i & 0 \end{bmatrix}

    The matrix form of a Pauli gate ``Z`` is:

    .. math::

        \begin{bmatrix} 1 & 0 \\ 0 & - 1 \end{bmatrix}

    Args:
        gate (str): index of a Pauli gate. 'I', 'X', 'Y', and 'Z' are the indexes of the corresponding Pauli gate

    Returns:
        ndarray: the ``ndarray`` form of a ``Pauli`` gate.

    Code Example:

    .. code-block:: python

        from QCompute.OpenService.service_ubqc.client.utils import pauli_gate
        I = pauli_gate('I')
        X = pauli_gate('X')
        Y = pauli_gate('Y')
        Z = pauli_gate('Z')
        print('The matrix form of an Identity gate is: \n', I)
        print('The matrix form of a Pauli X gate is: \n', X)
        print('The matrix form of a Pauli Y gate is: \n', Y)
        print('The matrix form of a Pauli Z gate is: \n', Z)

    ::

        The matrix form of an Identity gate is:
         [[1. 0.]
         [0. 1.]]
        The matrix form of a Pauli X gate is:
         [[0. 1.]
         [1. 0.]]
        The matrix form of a Pauli Y gate is:
         [[ 0.+0.j -0.-1.j]
         [ 0.+1.j  0.+0.j]]
        The matrix form of a Pauli Z gate is:
         [[ 1.  0.]
         [ 0. -1.]]
    """
    if gate == 'I':  # identity gate
        return eye(2, 2, dtype='float64')
    elif gate == 'X':  # Pauli X gate
        return array([[0, 1], [1, 0]], dtype='float64')
    elif gate == 'Y':  # Pauli Y gate
        return array([[0, -1j], [1j, 0]], dtype='complex128')
    elif gate == 'Z':  # Pauli Z gate
        return array([[1, 0], [0, -1]], dtype='float64')
    else:
        raise Error.ArgumentError(f"Invalid Pauli gate index: ({gate})!\nOnly 'I', 'X', 'Y' and 'Z' are supported as the index of Pauli gate.", ModuleErrorCode, FileErrorCode, 1)

def rotation_gate(axis, theta):
    r"""Define a rotation gate.

    The matrix form is:

    .. math::

        R_{x}(\theta) = \cos(\theta / 2) I - i\sin(\theta / 2) X

        R_{y}(\theta) = \cos(\theta / 2) I - i\sin(\theta / 2) Y

        R_{z}(\theta) = \cos(\theta / 2) I - i\sin(\theta / 2) Z

    Args:
        axis (str): the index of rotation axis. ‘x’, ‘y’ and ‘z’ are indexes of the rotation axis
        theta (float / int): rotation angle

    Returns:
        ndarray: the ``ndarray`` form of a rotation gate

    Code Example:

    .. code-block:: python

        from numpy import pi
        from QCompute.OpenService.service_ubqc.client.utils import rotation_gate

        theta = pi / 6
        Rx = rotation_gate('x', theta)
        Ry = rotation_gate('y', theta)
        Rz = rotation_gate('z', theta)
        print("The matrix form of a rotation X gate with angle pi/6 is: \n", Rx)
        print("The matrix form of a rotation Y gate with angle pi/6 is: \n", Ry)
        print("The matrix form of a rotation Z gate with angle pi/6 is: \n", Rz)

    ::

        The matrix form of a rotation X gate with angle pi/6 is:
         [[0.96592583+0.j         0.        -0.25881905j]
         [0.        -0.25881905j 0.96592583+0.j        ]]
        The matrix form of a rotation Y gate with angle pi/6 is:
         [[ 0.96592583+0.j -0.25881905+0.j]
         [ 0.25881905+0.j  0.96592583+0.j]]
        The matrix form of a rotation Z gate with angle pi/6 is:
         [[0.96592583-0.25881905j 0.        +0.j        ]
         [0.        +0.j         0.96592583+0.25881905j]]
    """
    if not isinstance(theta, float) and not isinstance(theta, int):
        raise Error.ArgumentError(f'Invalid rotation angle ({theta}) with the type: `{type(theta)}`!\nOnly `float` and `int` are supported as the type of rotation angle.', ModuleErrorCode, FileErrorCode, 2)

    # Calculate half of the input theta
    half_theta = theta / 2

    if axis == 'x':  # rotation-x gate matrix
        return pauli_gate('I') * cos(half_theta) + sin(half_theta) * -1j * pauli_gate('X')
    elif axis == 'y':  # rotation-y gate matrix
        return pauli_gate('I') * cos(half_theta) + sin(half_theta) * -1j * pauli_gate('Y')
    elif axis == 'z':  # rotation-z gate matrix
        return pauli_gate('I') * cos(half_theta) + sin(half_theta) * -1j * pauli_gate('Z')
    else:
        raise Error.ArgumentError(f"Invalid rotation axis: ({axis})!\nOnly 'x', 'y' and 'z' are supported as the index of rotation axis.", ModuleErrorCode, FileErrorCode, 3)

def u3_gate(theta, phi, lamda):
    r"""Define a single qubit unitary gate。

    Warning：
        This is the same ``U3`` gate as in other common packages.
        It has a decomposition form of 'Rz, Ry, Rz'.

    The matrix form is:

    .. math::

        U (\theta, \phi, \lambda) = Rz(\phi) Ry(\theta) RZ(\lambda)
        = \begin{bmatrix}
        \cos\frac{theta}{2} & -e^{i\lambda} \sin\frac{theta}{2}\\
        e^{i\phi}\sin\frac{theta}{2} & e^{i(\phi+\lambda)}\cos\frac{theta}{2}
        \end{bmatrix}

    Args:
        theta (float / int): the rotation angle of the Ry gate
        phi (float / int): the rotation angle of the left Rz gate
        lamda (float / int): the rotation angle of the right Rz gate

    Returns:
        ndarray: the ``ndarray`` form of a single qubit unitary gate
    """
    if not isinstance(theta, float) and not isinstance(theta, int):
        raise Error.ArgumentError(f'Invalid rotation angle ({theta}) with the type: `{type(theta)}`!\nOnly `float` and `int` are supported as the type of rotation angle.', ModuleErrorCode, FileErrorCode, 4)

    if not isinstance(phi, float) and not isinstance(phi, int):
        raise Error.ArgumentError(f'Invalid rotation angle ({phi}) with the type: `{type(phi)}`!\nOnly `float` and `int` are supported as the type of rotation angle.', ModuleErrorCode, FileErrorCode, 5)

    if not isinstance(lamda, float) and not isinstance(lamda, int):
        raise Error.ArgumentError(f'Invalid rotation angle ({lamda}) with the type: `{type(lamda)}`!\nOnly `float` and `int` are supported as the type of rotation angle.', ModuleErrorCode, FileErrorCode, 6)

    u_mat = matmul(matmul(rotation_gate('z', phi), rotation_gate('y', theta)), rotation_gate('z', lamda))
    return u_mat


def u_gate(theta, phi, lamda):
    r"""Define a single qubit unitary gate。

    Warning：
        The unitary gate generated by this method is a unique gate in MBQC.
        Unlike the commonly used ``U3`` gate, it has a decomposition form of 'Rz, Rx, Rz' .

    The matrix form is:

    .. math::

        U(\theta, \phi, \lambda) = Rz(\phi) Rx(\theta) Rz (\lambda)
        = \begin{pmatrix}
        e^{-i(\lambda / 2 + \phi / 2)} cos(\theta / 2)      &   -e^{i(\lambda / 2 - \phi / 2)} sin(\theta / 2) i \\
        -e^{-i(\lambda / 2 - \phi / 2)} sin(\theta / 2) i   &    e^{i(\lambda / 2 + \phi / 2)} cos(\theta / 2)
        \end{pmatrix}

    Args:
        theta (float / int): the rotation angle of the Rx gate
        phi (float / int): the rotation angle of the left Rz gate
        lamda (float / int): the rotation angle of the right Rz gate

    Returns:
        ndarray: the ``ndarray`` form of a single qubit unitary gate
    """
    if not isinstance(theta, float) and not isinstance(theta, int):
        raise Error.ArgumentError(f'Invalid rotation angle ({theta}) with the type: `{type(theta)}`!\nOnly `float` and `int` are supported as the type of rotation angle.', ModuleErrorCode, FileErrorCode, 7)


    if not isinstance(phi, float) and not isinstance(phi, int):
        raise Error.ArgumentError(f'Invalid rotation angle ({phi}) with the type: `{type(phi)}`!\nOnly `float` and `int` are supported as the type of rotation angle.', ModuleErrorCode, FileErrorCode, 8)


    if not isinstance(lamda, float) and not isinstance(lamda, int):
        raise Error.ArgumentError(f'Invalid rotation angle ({lamda}) with the type: `{type(lamda)}`!\nOnly `float` and `int` are supported as the type of rotation angle.', ModuleErrorCode, FileErrorCode, 9)

    u_mat = matmul(matmul(rotation_gate('z', phi), rotation_gate('x', theta)), rotation_gate('z', lamda))

    return u_mat


def _log(complex_number):
    r"""Calculate the logarithm of a complex number.

    .. math::

        \text{Let a complex number be } c = a + b i, \text{. The logarithm is obtained by: }

        log(c) = log (a + b i) = i \times \arctan(b / a)

    Note：
        This method is used to calculate the logarithm of a complex number.

    Args:
        complex_number (complex): a complex number

    Returns:
        complex: logarithm of the complex number
    """
    # If the type is ``float`` or ``int``, add a small imaginary part
    if not isinstance(complex_number, complex):
        complex_number = complex(complex_number, eps)
    # If the type is ``complex``, check if the real part and the imaginary part are zero
    else:
        if complex_number.imag == 0:
            complex_number += eps * 1j
        if complex_number.real == 0:
            complex_number += eps

    # Zero
    if abs(complex_number) <= eps:
        raise Error.ArgumentError(f'Invalid complex number: ({complex_number})!\nThe length of a complex number must be larger than 0.', ModuleErrorCode, FileErrorCode, 10)

    real_in = complex_number.real
    real_phase = real_in / abs(real_in)
    imag_in = complex_number.imag
    imag_phase = imag_in / abs(imag_in)

    real_out = log(sqrt(power(real_in, 2) + power(imag_in, 2)))

    # If the real part is zero
    if abs(real_in) < eps:
        if imag_phase > 0:
            imag_out = pi / 2
        else:
            imag_out = 3 * pi / 2 / 2

    # If the imaginary part is zero
    elif abs(imag_in) < eps:
        if real_phase > 0:
            imag_out = 0
        else:
            imag_out = pi

    # If both the real part and the imaginary part are not zero
    else:
        if real_phase > 0 and imag_phase > 0:
            imag_out = arcsin(imag_in / exp(real_out))
        elif real_phase > 0 > imag_phase:
            imag_out = arcsin(imag_in / exp(real_out))
        else:
            imag_out = pi - arcsin(imag_in / exp(real_out))

    return real_out + imag_out * 1j


def decompose(u_mat):
    r"""Decompose a 2 X 2 unitary gate to the product of rotation gates.

    Warning：
        Unlike the commonly used ``U3`` gate, a unitary gate is decomposed to the product of 'Rz, Rx, Rz' in this method.

    The matrix form is:

    .. math::

        U(\theta, \phi, \lambda) = Rz(\phi) Rx(\theta) Rz (\lambda)
        = \begin{pmatrix}
        e^{-i(\lambda / 2 + \phi / 2)} cos(\theta / 2)      &   -e^{i(\lambda / 2 - \phi / 2)} sin(\theta / 2) i \\
        -e^{-i(\lambda / 2 - \phi / 2)} sin(\theta / 2) i   &    e^{i(\lambda / 2 + \phi / 2)} cos(\theta / 2)
        \end{pmatrix}

    Args:
        u_mat (ndarray): the unitary gate to be decomposed

    Returns:
        float: the rotation angle of the Rx gate
        float: the rotation angle of the left Rz gate
        float: the rotation angle of the right Rz gate
    """
    if not isinstance(u_mat, ndarray):
        raise Error.ArgumentError(f'Invalid matrix ({u_mat}) with the type: `{type(u_mat)}`!\nOnly `ndarray` is supported as the type of the matrix.', ModuleErrorCode, FileErrorCode, 11)

    if u_mat.shape != (2, 2):
        raise Error.ArgumentError(f'Invalid matrix ({u_mat}) with the shape: {u_mat.shape}!\nOnly (2, 2) is supported as the shape of the matrix.', ModuleErrorCode, FileErrorCode, 12)

    u_error = linalg.norm(matmul(c(t(u_mat)), u_mat) - pauli_gate('I'))
    is_unitary = u_error < eps
    if not is_unitary:
        raise Error.ArgumentError(f'Invalid matrix ({u_mat}) with the norm: {u_error}!\nOnly unitary matrix is supported.', ModuleErrorCode, FileErrorCode, 13)

    a = u_mat[0][0]
    b = u_mat[0][1]
    d = u_mat[1][0]
    e = u_mat[1][1]

    a_is_zero = abs(a) <= eps
    d_is_zero = abs(d) <= eps

    if a_is_zero:
        theta = pi
        lamda = 0
        phi = _log(b / d) * 1j

    elif d_is_zero:
        lamda = 0
        theta = 0
        phi = _log(e / a) * -1j

    else:
        lamda = (_log(e / a) + _log(b / d)) * -1j / 2
        phi = (_log(e / a) - _log(b / d)) * -1j / 2
        theta = arctan(b / a * sin(lamda) + b / a * cos(lamda) * 1j) * 2

    return real(theta), real(phi), real(lamda)