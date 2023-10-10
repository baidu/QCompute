#!/usr/bin/python3
# -*- coding: utf8 -*-

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
Quantum Environment Operation
"""
FileErrorCode = 9

from typing import TYPE_CHECKING, Optional, Callable, Union, List

from QCompute.QPlatform import Error, ModuleErrorCode
from QCompute.QPlatform.QEnv import QEnv
from QCompute.QPlatform.QOperation.Barrier import Barrier
from QCompute.QPlatform.QOperation.CompositeGate import MS, CK
from QCompute.QPlatform.QOperation.FixedGate import ID, X, Y, Z, H, S, SDG, T, TDG, \
    CX, CY, CZ, CH, SWAP, \
    CCX, CSWAP
from QCompute.QPlatform.QOperation.Measure import MeasureZ
from QCompute.QPlatform.QOperation.RotationGate import U, RX, RY, RZ, \
    CU, CRX, CRY, CRZ
from QCompute.QPlatform.QRegPool import QRegStorage

if TYPE_CHECKING:
    from QCompute.QPlatform.QOperation import RotationArgument
    from QCompute.QPlatform.QOperation.RotationGate import RotationGateOP
    from QCompute.QPlatform.QOperation.CustomizedGate import CustomizedGateOP
    from QCompute.QPlatform.QOperation.CompositeGate import CompositeGateOP
    from QCompute.QPlatform.QOperation.QProcedure import QProcedure


class QEnvOperation(QEnv):
    """
    Quantum Environment Operation
    """

    # Fixed Gate
    def ID(self, *qRegIndexList: int) -> 'QEnvOperation':
        ID(*[self.Q(qRegIndex) for qRegIndex in qRegIndexList])
        return self

    def X(self, *qRegIndexList: int) -> 'QEnvOperation':
        X(*[self.Q(qRegIndex) for qRegIndex in qRegIndexList])
        return self

    def Y(self, *qRegIndexList: int) -> 'QEnvOperation':
        Y(*[self.Q(qRegIndex) for qRegIndex in qRegIndexList])
        return self

    def Z(self, *qRegIndexList: int) -> 'QEnvOperation':
        Z(*[self.Q(qRegIndex) for qRegIndex in qRegIndexList])
        return self

    def H(self, *qRegIndexList: int) -> 'QEnvOperation':
        H(*[self.Q(qRegIndex) for qRegIndex in qRegIndexList])
        return self

    def S(self, *qRegIndexList: int) -> 'QEnvOperation':
        S(*[self.Q(qRegIndex) for qRegIndex in qRegIndexList])
        return self

    def SDG(self, *qRegIndexList: int) -> 'QEnvOperation':
        SDG(*[self.Q(qRegIndex) for qRegIndex in qRegIndexList])
        return self

    def T(self, *qRegIndexList: int) -> 'QEnvOperation':
        T(*[self.Q(qRegIndex) for qRegIndex in qRegIndexList])
        return self

    def TDG(self, *qRegIndexList: int) -> 'QEnvOperation':
        TDG(*[self.Q(qRegIndex) for qRegIndex in qRegIndexList])
        return self

    def CX(self, *qRegIndexList: int) -> 'QEnvOperation':
        CX(*[self.Q(qRegIndex) for qRegIndex in qRegIndexList])
        return self

    def CY(self, *qRegIndexList: int) -> 'QEnvOperation':
        CY(*[self.Q(qRegIndex) for qRegIndex in qRegIndexList])
        return self

    def CZ(self, *qRegIndexList: int) -> 'QEnvOperation':
        CZ(*[self.Q(qRegIndex) for qRegIndex in qRegIndexList])
        return self

    def CH(self, *qRegIndexList: int) -> 'QEnvOperation':
        CH(*[self.Q(qRegIndex) for qRegIndex in qRegIndexList])
        return self

    def SWAP(self, *qRegIndexList: int) -> 'QEnvOperation':
        SWAP(*[self.Q(qRegIndex) for qRegIndex in qRegIndexList])
        return self

    def CCX(self, *qRegIndexList: int) -> 'QEnvOperation':
        CCX(*[self.Q(qRegIndex) for qRegIndex in qRegIndexList])
        return self

    def CSWAP(self, *qRegIndexList: int) -> 'QEnvOperation':
        CSWAP(*[self.Q(qRegIndex) for qRegIndex in qRegIndexList])
        return self

    # Rotation Gate
    def U(self, theta: 'RotationArgument',
          phi: Optional['RotationArgument'] = None,
          lamda: Optional['RotationArgument'] = None) -> 'gateWrapFunc':
        gate = U(theta, phi, lamda)
        return self.gateWrap(gate)

    def RX(self, theta: 'RotationArgument') -> 'gateWrapFunc':
        gate = RX(theta)
        return self.gateWrap(gate)

    def RY(self, theta: 'RotationArgument') -> 'gateWrapFunc':
        gate = RY(theta)
        return self.gateWrap(gate)

    def RZ(self, lamda: 'RotationArgument') -> 'gateWrapFunc':
        gate = RZ(lamda)
        return self.gateWrap(gate)

    def CU(self, theta: 'RotationArgument',
           phi: 'RotationArgument',
           lamda: 'RotationArgument') -> 'gateWrapFunc':
        gate = CU(theta, phi, lamda)
        return self.gateWrap(gate)

    def CRX(self, theta: 'RotationArgument') -> 'gateWrapFunc':
        gate = CRX(theta)
        return self.gateWrap(gate)

    def CRY(self, theta: 'RotationArgument') -> 'gateWrapFunc':
        gate = CRY(theta)
        return self.gateWrap(gate)

    def CRZ(self, lamda: 'RotationArgument') -> 'gateWrapFunc':
        gate = CRZ(lamda)
        return self.gateWrap(gate)

    # Customized Gate
    def CustomizedGate(self, gate: 'CustomizedGateOP') -> 'gateWrapFunc':
        return self.gateWrap(gate)

    # Composite Gate
    def MS(self, theta: Optional['RotationArgument'] = None) -> 'gateWrapFunc':
        gate = MS(theta)
        return self.gateWrap(gate)

    def CK(self, kappa: Optional['RotationArgument'] = None) -> 'gateWrapFunc':
        gate = CK(kappa)
        return self.gateWrap(gate)

    # QProcedure
    def QProcedure(self, procedure: Union[str, 'QProcedure']) -> 'ProcedureWrapFunc':
        if type(procedure) == str:
            procedure = self.procedureMap[procedure]
        return self.procedureWrap(procedure)

    # Barrier
    def Barrier(self, *qRegIndexList: List[int]) -> 'QEnvOperation':
        Barrier(*[self.Q(qRegIndex) for qRegIndex in qRegIndexList])
        return self

    # Measure
    def MeasureZ(self, qRegIndexList: Optional[List[Union[int, 'QRegStorage']]] = None,
                 cRegIndexList: Optional[Union[List[int], range]] = None) -> 'QEnvOperation':
        if qRegIndexList is None and cRegIndexList is None:
            qRegIndexList, cRegIndexList = self.Q.toListPair()
        elif qRegIndexList is None or cRegIndexList is None:
            raise Error.ArgumentError('Mismatched qRegIndexList and cRegIndexList', ModuleErrorCode, FileErrorCode, 1)
        MeasureZ([self.Q(qRegIndex) if isinstance(qRegIndex, int) else qRegIndex for qRegIndex in qRegIndexList],
                 cRegIndexList)
        return self

    def gateWrap(self, gate: Union['RotationGateOP', 'CustomizedGateOP', 'CompositeGateOP']) -> \
            'gateWrapFunc':
        def func(*qRegIndexList: int) -> 'QEnvOperation':
            gate(*[self.Q(qRegIndex) for qRegIndex in qRegIndexList])
            return self

        func.data = gate
        return func

    def procedureWrap(self, procedure: 'QProcedure') -> 'ProcedureWrapFunc':
        def func(*argumentList: 'RotationArgument') -> 'gateWrapFunc':
            procedureOP = procedure(*argumentList)
            return self.gateWrap(procedureOP)

        func.data = procedure
        return func


gateWrapFunc = Callable[[*[int]], 'QEnvOperation']
ProcedureWrapFunc = Callable[[*['RotationArgument']], 'gateWrapFunc']