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
Unroll Circuit To IOPCAS
"""
FileErrorCode = 16

from QCompute.OpenModule import ModuleImplement
from typing import List, Dict, Optional, Union
from QCompute.QProtobuf import PBProgram, PBCircuitLine, PBFixedGate, PBRotationGate




class UnrollCircuitToIoPCASModule(ModuleImplement):
    """
    Unroll supported gates to Ry, Rz, barrier, measure

    Supported fixed gates: ID, X, Y, Z, H, S, SDG, T, TDG, CX, CY, CZ, CH, SWAP, CCX, CSWAP

    Supported rotation gates: U, R, RX, RY, RZ, CU, CRX, CRY, CRZ

    Composite gates are supported since they can be processed by the CompositeGateModule module in advance.

    Must unrollProcedure before, because rotation gate must hve all rotation arguments.

    Example:

    env.module(UnrollCircuitToIoPCASModule())

    env.module(UnrollCircuitToIoPCASModule({'disable': True}))  # Disable

    env.module(UnrollCircuitToIoPCASModule({'errorOnUnsupported': True, 'targetGates': ['RY', 'RZ', 'CZ']}))

    env.module(UnrollCircuitToIoPCASModule({'errorOnUnsupported': False, 'targetGates': ['RY', 'RZ', 'CZ'], 'sourceGates': ['CH', 'CSWAP']}))

    env.serverModule(ServerModule.UnrollCircuitToIoPCAS, {"disable": True})
    """
    

    def __init__(self, arguments: Optional[Dict[str, Union[List[str], bool]]] = None):
        """
        Initialize the Module.

        Json serialization is allowed by the requested parameter.
        """
        super().__init__(arguments)
        

    def __call__(self, program: 'PBProgram') -> 'PBProgram':
        """
        Process the Module

        :param program: the program

        :return: unrolled circuit
        """
        from QCompute.QPlatform import Error; raise Error.RuntimeError('Not implemented at local sdk')

    
