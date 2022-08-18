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
from typing import List, Union
import numpy

__all__ = [
    "kron",
    "print_progress"
]


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


def print_progress(current_progress: Union[float, int], progress_name, track=True):
    r"""Print a progress bar.

    Args:
        current_progress (Union[float, int]): current progress percentage
        progress_name (str): name of the progress bar
        track (bool): whether to print the progress on the terminal
    """
    if current_progress < 0 or current_progress > 1:
        raise ArgumentTypeError(f"Invalid current progress: ({current_progress})!\n"
                                f"'current_progress' must be a percentage between 0 and 1")
    if not isinstance(track, bool):
        raise ArgumentTypeError(f"Invalid parameter ({track}) with the type `{type(track)}`!\n"
                                f"Only `bool` is supported as the parameter.")
    if track:
        print("\r" + f"{progress_name.ljust(30)}"
                     f"|{'■' * int(50 * current_progress):{50}s}| "
                     f"\033[94m {'{:6.2f}'.format(100 * current_progress)}% \033[0m ", flush=True, end="")
        if current_progress == 1:
            print(" (Done)")
