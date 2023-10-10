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
Convert the circuit to Qasm
"""
FileErrorCode = 6

import re
from typing import List, Optional, Tuple, Set, Dict, Any

from QCompute.OpenConvertor import ModuleErrorCode, ConvertorImplement
from QCompute.QPlatform import Error
from QCompute.QProtobuf import PBProgram, PBFixedGate, PBRotationGate, PBCompositeGate


class CircuitToQasm(ConvertorImplement):
    """
    Circuit To Qasm
    """

    def __init__(self):
        self.containMeasure = False
        # Sort the order of procedures to analyze procedure calling.
        self.procedureNameList = []
        self.proceduresCode = {}
        self.proceduresDepends = {}

    def getArgumentList(self, params: List[str], paramIds: List[int]) -> List:
        """
        Load called params according to arguments.

        :param params: type: List[str]. List of params.
        :param paramIds: type: List[int]. List of param indices.

        :return: type: List. List of called params.
        """
        paramCall = []

        checkParams = params and len(params)
        checkParamIds = paramIds and len(paramIds)

        if checkParams and checkParamIds:
            for i in range(len(paramIds)):
                paramId = paramIds[i]
                if paramId != -1:
                    # paramId
                    paramItem = f'param{paramId}'
                    paramCall.append(paramItem)
                else:
                    # paramValue
                    paramItem = params[i]
                    paramCall.append(paramItem)
        else:
            # Only checkParamIds or checkParams is available.
            # param0 -> paramx
            if checkParamIds:
                for i in range(len(paramIds)):
                    paramId = paramIds[i]
                    paramItem = f'param{paramId}'
                    paramCall.append(paramItem)

            if checkParams:
                paramCall = params

        return paramCall

    continueZeroRe = re.compile('0+$')

    def getFixedFloatNumber(self, realNum: float) -> str:
        """
        Format floating point number.

        :param realNum: type: float. The input floating point number.

        :return: type: str. The string of formatted floated point number.
        """
        paramFixed = format(realNum, '.7f')
        # Remove 0s.
        paramFixed = self.continueZeroRe.sub('', paramFixed)

        # Insert a 0 after decimal point.
        charCheck = paramFixed[-1]
        if charCheck == '.':
            paramFixed += '0'
        return paramFixed

    def getFixedArgumentList(self, argumentValueList: List[float]) -> List[str]:
        params = []
        for param in argumentValueList:
            paramFixed = self.getFixedFloatNumber(param)
            params.append(paramFixed)
        return params

    def getTrimmedArgumentList(self, argumentValueList: List[float]) -> List[str]:
        """ 
        Convert float point number into string. Value None is converted to 0.

        :param argumentValueList: type: List[float]. List of argument values. 

        :return: type: List[str]. List of converted argument values.
        """
        convertedParams = [self.getFixedFloatNumber(p) for p in argumentValueList]
        return convertedParams

    def getMeasureCommandCode(self, measure, qRegs: List[int], regName: str) -> str:
        """
        Convert measure code to command code.

        :param measure: type: object. Measure code.
        :param qRegs: type: List[int]. List of quantum registers.
        :param regName: type: str. Name of register.

        :return: type: str. Command of measure code.
        """
        command = ''
        # Convert from quantum registers to conventional registers.
        for i in range(len(measure.cRegList)):
            cr = measure.cRegList[i]  # conventional register
            qr = qRegs[i]  # quantum register
            command += f'measure {regName}[{qr}] -> c[{cr}];\n'
        return command

    def getFixedCommandCode(self, fixedGate: int, regs: List[int], regName: str, usingIndex: bool = True) -> str:
        """
        Convert Fixed Gate to command code. 

        :param fixedGate: type: int. Index of fixed gate.
        :param regs: type: List[int]. List of registers.
        :param regName: type: str. Name of register.
        :param usingIndex: type: bool. If True, adds braket [] for register index. Default: True.
        
        :return: type: str. Command code of fixed gate.
        """

        # Get name of gate.
        gateName = PBFixedGate.Name(fixedGate).lower()
        # Convert registers.
        if usingIndex:
            regOp = [f'{regName}[{r}]' for r in regs]
            command = ''

        # Index can not be used in the procedure. 
        # Set usingIndex=False to remove braket [].
        else:
            regOp = [f'{regName}{r}' for r in regs]
            # Indent = 2
            command = '  '
        command += f'{gateName} {", ".join(regOp)};\n'
        return command

    def getRotationCommandCode(self, rotationGate: int, regs: List[int], regName: str, usingIndex: bool = True,
                               paramValues: Optional[List[float]] = None, paramIds: Optional = None) -> str:
        """
        Convert Rotation Gate to command code. 

        :param rotationGate: type: int. Index of fixed gate.
        :param regs: type: List[int]. List of registers.
        :param regName: type: str. Name of register.
        :param usingIndex: type: bool. If True, adds braket [] for register index. Default: True.
        :param paramValues: type: List[float], optional. List of param values.
        :param paramIds: type: List[int], optional. List of param indices.
        # NOTE: paramValues/paramIds may be None.

        :return: type: str. Command code of rotation gate.
        """

        # Get name of gate.
        gateName = PBRotationGate.Name(rotationGate).lower()

        # Check if optional arguments paramValues and paramIds are None.
        params = self.getFixedArgumentList(paramValues)
        # Convert params.
        paramCall = self.getArgumentList(params, paramIds)

        # Convert registers
        if usingIndex:
            regOp = [f'{regName}[{r}]' for r in regs]
            command = ''

        # Index can not be used in the procedure. 
        # Set usingIndex=False to remove braket [].
        else:
            regOp = [f'{regName}{r}' for r in regs]
            # Indent=2.
            command = '  '

        command += f'{gateName} ({", ".join(paramCall)}) {", ".join(regOp)};\n'

        return command

    def getBarrierCommandCode(self, regs: List[int], regName: str, usingIndex: bool = True) -> str:
        """
        Convert barrier code to command code.

        :param regs: type: List[int]. List of registers.
        :param regName: type: str. Name of register.
        :param usingIndex: type: bool. If True, adds braket [] for register index. Default: True.

        :return: type: str. Command of barrier code.
        """
        # Convert registers.
        if usingIndex:
            regOp = [f'{regName}[{r}]' for r in regs]
            command = f'barrier {", ".join(regOp)};\n'
        else:
            command = ''
            for r in regs:
                # Indent=2.
                command += f'  barrier {regName}{r};\n'
        return command

    def getProcCommandCode(self, procedureName: str, regs: List[int], regName: str, usingIndex: bool = True,
                           paramValues: Optional[List[float]] = None, paramIds: Optional = None) -> str:
        """
        Convert procedure to command code. 

        :param procedureName: type: str. Name of called procedure.
        :param regs: type: List[int]. List of registers.
        :param regName: type: str. Name of quantum register. Default: 'q'.
        :param usingIndex: type: bool. If True, adds braket [] for register index. Default: True.
        :param paramValues: type: List[float], optional. List of param values.
        :param paramIds: type: List[int], optional. List of param indices.
        
        :return: type: str. Command code of procedure.
        """

        # Convert None argument to 0.
        convertedParams = self.getTrimmedArgumentList(paramValues)
        # Convert params.
        paramCall = self.getArgumentList(convertedParams, paramIds)

        if len(paramCall) > 0:
            params = f'({", ".join(paramCall)})'
        else:
            params = ''

        # Example of command: cu1(pi/2) q[0],q[1];
        if usingIndex:
            regOp = [f'{regName}[{r}]' for r in regs]
            command = ''

        # Call procedure.
        else:
            regOp = [f'{regName}{r}' for r in regs]
            command = '  '
        command += f'{procedureName}{params} {", ".join(regOp)};\n'
        return command

    def getCompositeCommandCode(self, compositeGate: int, regs: List[int], regName: str, usingIndex: bool = True,
                                paramValues: Optional[List[float]] = None, paramIds: Optional = None) -> str:
        """
        Convert Composite Gate to command code. 

        :param compositeGate: type: int. Index of composite gate.
        :param regs: type: List[int]. List of registers.
        :param regName: type: str. Name of register.
        :param usingIndex: type: bool. If True, adds braket [] for register index. Default: True.
        :param paramValues: type: List[float], optional. List of param values.
        :param paramIds: type: List[int], optional. List of param indices.

        :return: type: str. Command code of composite gate.
        """

        # Get name of gate.
        gateName = PBCompositeGate.Name(compositeGate)
        # Check params.
        params = self.getFixedArgumentList(paramValues)
        # Convert params.
        paramCall = self.getArgumentList(params, paramIds)

        # Convert registers.
        if usingIndex:
            regOp = [f'{regName}[{r}]' for r in regs]
            command = ''
        else:
            regOp = [f'{regName}{r}' for r in regs]
            command = '  '
        command += f'{gateName} ({", ".join(paramCall)}) {", ".join(regOp)};\n'
        return command

    def getCircuitsCode(self, circuit: List, regName: str, usingIndex: bool = True) -> Tuple[str, Set[str]]:
        """
        Convert Circuits to qasm code. 

        :param circuit: type: List. List of gate operations.
        :param regName: type: str. Name of register. The name of register used in the procedure is different from that in external program.
        :param usingIndex: type: bool. If True, adds braket [] for register index. Default: True.

        :return qasmCode: type: str. Qasm code of circuit.
        :return depends: type: str. Dependencies of called procedures.
        """

        # Process circuits recurrently.
        qasmCode = ''
        # If op == 'measure', adds measure code to the end of command code. 
        measureCode = ''
        # Dependency of procedures.
        depends: Set[str] = set()
        for gate in circuit:
            # Type of gate operation.
            op = gate.WhichOneof('op')
            if op == 'fixedGate':
                qasmCode += self.getFixedCommandCode(gate.fixedGate, gate.qRegList, regName, usingIndex)
            elif op == 'rotationGate':
                qasmCode += self.getRotationCommandCode(gate.rotationGate, gate.qRegList, regName, usingIndex,
                                                        gate.argumentValueList, gate.argumentIdList)
            elif op == 'compositeGate':
                qasmCode += self.getCompositeCommandCode(gate.compositeGate, gate.qRegList, regName, usingIndex,
                                                         gate.argumentValueList, gate.argumentIdList)
            elif op == 'procedureName':
                qasmCode += self.getProcCommandCode(gate.procedureName, gate.qRegList, regName, usingIndex,
                                                    gate.argumentValueList, gate.argumentIdList)
                # Store names of dependent procedures
                if gate.procedureName in depends:
                    depends.add(gate.procedureName)
            elif op == 'barrier':
                qasmCode += self.getBarrierCommandCode(gate.qRegList, regName, usingIndex)
            elif op == 'measure':
                measureCode += self.getMeasureCommandCode(gate.measure, gate.qRegList, regName)
            else:
                raise Error.ArgumentError(f'Invalid gate operation: {gate}', ModuleErrorCode, FileErrorCode, 1)

        if measureCode is not None:
            qasmCode += measureCode
            self.containMeasure = True

        return qasmCode, depends

    def getProcedureCode(self, procedureMaps: Dict[str, Any]) -> str:
        """
        Process procedure code

        :param procedureMaps: type: Dict[str, Any]. Map of procedures.
        
        :return: type: str. Procedure code.
        """
        procCode = ''
        for name, content in procedureMaps.items():
            # key: name of procedure
            # val: object of procedure content
            gateDefine = self.getProcDefineCode(name, content)

            # Make sure definition of procedure is in front of procedure code.
            self.procedureNameList.append(name)
            self.proceduresCode[name] = gateDefine

        # Reorder definitions of procedures according to procedureMaps.
        usedProcedure = []  # List[str]
        for name in self.procedureNameList:
            # detect dependencies.
            depends = self.proceduresDepends.get(name)
            # Process the dependent procedure first.
            if depends is not None and len(depends) > 0:
                for item in depends:
                    if item in usedProcedure:
                        # Procedure has been used.
                        continue
                    depCode = self.proceduresCode.get(item)
                    if depCode is not None:
                        # Dependent code is at the front.
                        depCode += procCode
                        # Reset 
                        procCode = depCode
                        # Flag of used procedure to skip loop.
                        usedProcedure.append(item)
                    else:
                        # Dependent code not found.
                        raise Error.ArgumentError(f'Invalid procedure name: {item}', ModuleErrorCode, FileErrorCode, 2)

            if name in usedProcedure:
                # Procedure has been used.
                continue

            # Assemble procedure code.
            procCode += self.proceduresCode.get(name)
            usedProcedure.append(name)

        return procCode

    def getProcDefineCode(self, name: str, content) -> str:
        """
        Get definition code of procedure.

        :param name: type: str. Name of procedure.
        :param content: type: object. Content includes {parameterCount, usingQRegList, circuit}.

        parameterCount: type: int. Number of defined params ranges from 1 to 3, corresponding to theta, phi, and lambda.
        usingQRegList: type: List[int]. List of quantum registers. Example: qb0-qbn.
        circuit: type: List. List of gate operations.
 
        :return: type: str. Definition code of procedure.
        """

        if content.parameterCount > 0:
            paramsArray = [f'param{i}' for i in range(content.parameterCount)]
            paramsDef = f'({", ".join(paramsArray)})'
        else:
            paramsDef = ''

        # Quantum registers.
        qRegList = [f'qb{r}' for r in content.usingQRegList]
        qRegs = ', '.join(qRegList)

        # Get type of gate operation. Measure code is excluded.
        if content.circuit is not None:
            circuitCode, depends = self.getCircuitsCode(content.circuit, 'qb', False)
            # Construct map of dependencies. 
            self.proceduresDepends[name] = depends
        else:
            circuitCode = ''

        gateDefine = f'gate {name}{paramsDef} {qRegs}\n{{\n{circuitCode}}}\n'

        return gateDefine

    def convert(self, program: PBProgram) -> str:
        """
        Convert PBProgram to qasmCode. 

        :param program: type: object. PBProgram.

        :return: type: str. qasm code.
        """
        qasmCode = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'

        if program.source.qasm != '':
            return program.source.qasm

        # Size is maximum+1, since index starts from 0.
        maxQregSize = max(program.head.usingQRegList) + 1
        maxCregSize = max(program.head.usingCRegList) + 1

        qasmCode += f'qreg q[{maxQregSize}];\n'
        qasmCode += f'creg c[{maxCregSize}];\n'

        # Make sure procedure is defined before it is called.
        if program.body.procedureMap:
            procedureCode = self.getProcedureCode(program.body.procedureMap)
            qasmCode += procedureCode

        circuitCode, depends = self.getCircuitsCode(program.body.circuit, 'q')
        qasmCode += circuitCode

        return qasmCode