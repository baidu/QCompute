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

FileErrorCode = 5

from copy import deepcopy
from typing import List, Dict, Optional

from QCompute.OpenModule import ModuleImplement
from QCompute.QProtobuf import PBProgram, PBCircuitLine


class ReverseCircuitModule(ModuleImplement):
    """
    Reverse the circuit
    change the gate order, reverse the customized gate matrix.

    Example:

    env.module(ReverseCircuitModule())

    env.module(ReverseCircuitModule({'disable': True}))  # Disable
    """

    def __init__(self, arguments: Optional[Dict[str, bool]] = None):
        """
        Initialize the Module.

        Json serialization is allowed by the requested parameter.
        """
        super().__init__(arguments)

    def __call__(self, program: 'PBProgram') -> 'PBProgram':
        """
        Process the Module

        :param program: the program
        :return: reversed circuit
        """
        if self.disable:
            return program

        ret = deepcopy(program)

        for name, procedure in program.body.procedureMap.items():
            targetProcedure = ret.body.procedureMap[name]
            del targetProcedure.circuit[:]
            self._reverse_circuit(targetProcedure.circuit, procedure.circuit)
        del ret.body.circuit[:]
        self._reverse_circuit(ret.body.circuit, program.body.circuit)
        return ret

    def _reverse_circuit(self, circuitOut: List['PBCircuitLine'], circuitIn: List['PBCircuitLine']):
        """
        Reverse a circuit

        :param circuitOut: input circuit
        :param circuitIn: output circuit
        """

        for circuitLine in reversed(deepcopy(circuitIn)):
            circuitOut.append(circuitLine)