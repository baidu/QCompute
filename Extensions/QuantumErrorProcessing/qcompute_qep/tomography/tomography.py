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
This file aims to collect functions related to the Tomography.

"""
import abc
from typing import Any
import numpy as np

from qcompute_qep.utils.types import QProgram, QComputer


class Tomography(abc.ABC):
    """The Tomography abstract class.


    Quantum tomography is an experimental procedure to reconstruct a description of part of quantum system
    from the measurement outcomes of a specific set of experiments.

    The abstract Tomography class for the quantum tomography module.
    The implementations---``StateTomography``, ``ProcessTomography``, ``GateSetTomography``, ``SpectralTomography``
    and ``DetectorTomography``---must inherit this abstract class.
    """
    def __init__(self, qp: QProgram = None, qc: QComputer = None, **kwargs: Any):
        """
        The init function of the Quantum State Tomography class.

        :param qp: QProgram, quantum program for creating the target quantum state
        :param qc: QComputer, the quantum computer

        Optional keywords list are:

            + `method`: default to ``inverse``, specify the tomography method
            + `shots`: default to :math:`4096`, the number of shots each measurement should carry out
            + `ptm`: default to ``False``, if the quantum object should be in the Pauli transfer matrix form

        """
        self._qp: QProgram = qp
        self._qc: QComputer = qc
        self._method: str = None
        self._shots: int = None
        self._ptm: bool = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.fit(*args, **kwargs)

    @abc.abstractmethod
    def fit(self, qp: QProgram = None, qc: QComputer = None, **kwargs) -> np.ndarray:
        """
        Every implementation of the Tomography class must inherit the `fit` function,
        which aims to estimate the target quantum object.

        :param qp: QProgram, quantum program for creating the target quantum state
        :param qc: QComputer, the quantum computer instance
        :return: the estimated quantum object

        Optional keywords list are:

            + `method`: default to ``inverse``, specify the tomography method
            + `shots`: default to :math:`4096`, the number of shots each measurement should carry out
            + `ptm`: default to ``False``, if the quantum object should be in the Pauli transfer matrix form

        """
        raise NotImplementedError
