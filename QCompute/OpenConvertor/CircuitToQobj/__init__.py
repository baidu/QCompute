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
Convert the circuit to qobj
"""
FileErrorCode = 7

import uuid
from typing import TYPE_CHECKING, Dict, List, Optional

from bidict import bidict

from QCompute.OpenConvertor import ConvertorImplement, ModuleErrorCode
from QCompute.QPlatform import Error
from QCompute.QPlatform.QOperation.RotationGate import createRotationGateInstance
from QCompute.QProtobuf import PBQObject, PBExperiment, PBInstruction, PBFixedGate, PBRotationGate, PBMeasure

if TYPE_CHECKING:
    from QCompute.QProtobuf import PBProgram, PBCircuitLine


class CircuitToQobj(ConvertorImplement):
    """
    Circuit To Qobj
    """

    def __init__(self):
        self._instructions: Optional[List['PBInstruction']] = None
        self._measuredQRegsToCRegsBidict: Optional[bidict] = None
        self._measured = False

    def convert(self, program: 'PBProgram', shots: int) -> 'PBQObject':
        """
        Convert the circuit to qobj.

        For Aer simulator

        Example:

        env.publish()  # need to generate protobuf circuit data

        qobj = CircuitToQObj().convert(env.program, 1024)

        :param program: Protobuf format of the circuit
        :param shots: Shot count
        :return: QObj
        """

        # list all quantum registers, then reassign the indices (only indices are used in qiskit register, no names)
        qRegMap = {qReg: index for index, qReg in enumerate(program.head.usingQRegList)}

        # fill in the qobj list
        qObject = PBQObject()
        qObject.type = 'QASM'
        qObject.schema_version = '1.1.0'
        qObject.qobj_id = uuid.uuid4().hex
        experiment = PBExperiment()
        experiment.config.n_qubits = len(program.head.usingQRegList)  # the number of quantum registers
        experiment.config.memory_slots = 1 if len(
            program.head.usingCRegList) == 0 else len(
            program.head.usingCRegList)  # the number of classical registers must not be less than 1
        experiment.config.shots = shots  # number of shot

        self._instructions = experiment.instructions
        self._measuredQRegsToCRegsBidict = bidict()
        self._measured = False
        self._convertCircuit(program.body.circuit, qRegMap)
        qObject.experiments.append(experiment)

        return qObject

    def _convertCircuit(self, circuit: List['PBCircuitLine'], qRegMap: Dict[int, int]) -> None:
        # fill in the circuit
        for circuitLine in circuit:
            op = circuitLine.WhichOneof('op')

            if self._measured and op != 'measure':
                raise Error.ArgumentError('Measure must be the last operation', ModuleErrorCode, FileErrorCode, 1)

            instruction = PBInstruction()
            if op == 'fixedGate':  # fixed gate
                fixedGate: int = circuitLine.fixedGate
                instruction.name = PBFixedGate.Name(fixedGate).lower()  # the name of fixed gate
                instruction.qubits[:] = [qRegMap[qReg] for qReg in circuitLine.qRegList]  # quantum register lists
            elif op == 'rotationGate':  # rotation gate
                rotationGate: int = circuitLine.rotationGate
                gate = createRotationGateInstance(PBRotationGate.Name(rotationGate), *circuitLine.argumentValueList)
                instruction.name = f'u{len(gate.uGateArgumentList)}'  # rotation gate types: U1, U2, and U3
                instruction.params[:] = gate.uGateArgumentList  # parameters for U gate
                instruction.qubits[:] = [qRegMap[qReg] for qReg in circuitLine.qRegList]  # quantum register lists
            elif op == 'customizedGate':  # customized gate
                raise Error.ArgumentError('Unsupported operation customizedGate!', ModuleErrorCode, FileErrorCode, 2)
            elif op == 'compositeGate':  # composite gate
                raise Error.ArgumentError('Unsupported operation compositeGate!', ModuleErrorCode, FileErrorCode, 3)
            elif op == 'procedureName':  # procedure
                raise Error.ArgumentError(
                    'Unsupported operation procedure, please flatten by UnrollProcedureModule!',
                    ModuleErrorCode, FileErrorCode, 4)

                # it is not implemented, flatten by UnrollProcedureModule
            elif op == 'measure':  # measure
                measure: 'PBMeasure' = circuitLine.measure
                if measure.type == PBMeasure.Type.Z:  # only Z measure is supported
                    pass
                else:  # unsupported measure types
                    raise Error.ArgumentError(
                        f'Unsupported operation measure {measure.type.name}!', ModuleErrorCode, FileErrorCode, 5)

                instruction.name = 'measure'  # instruction name
                instruction.memory[:] = measure.cRegList  # classical register lists
                instruction.qubits[:] = [qRegMap[qReg] for qReg in circuitLine.qRegList]  # quantum register lists
                for index, qReg in enumerate(circuitLine.qRegList):
                    if self._measuredQRegsToCRegsBidict.get(qReg) is not None:
                        raise Error.ArgumentError('Measure must be once on a QReg!', ModuleErrorCode, FileErrorCode, 6)
                    if self._measuredQRegsToCRegsBidict.inverse.get(measure.cRegList[index]) is not None:
                        raise Error.ArgumentError('Measure must be once on a CReg!', ModuleErrorCode, FileErrorCode, 7)
                    self._measuredQRegsToCRegsBidict[qReg] = measure.cRegList[index]
                self._measured = True
            elif op == 'barrier':  # barrier
                instruction.name = 'barrier'  # instruction name
                instruction.qubits[:] = [qRegMap[qReg] for qReg in circuitLine.qRegList]  # quantum register lists
            else:  # unsupported operation
                raise Error.ArgumentError(
                    f'Unsupported operation {circuitLine}!', ModuleErrorCode, FileErrorCode, 8)
            self._instructions.append(instruction)