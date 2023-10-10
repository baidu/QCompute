#!/usr/bin/python3
# -*- coding: utf8 -*-

# Copyright (c) 2022 Baidu, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use self file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Convert the qasm text to protobuf circuit
"""
FileErrorCode = 8

import json
import re
from enum import IntEnum
from typing import List, Dict, Union, Tuple, Any

from antlr4.error.ErrorListener import ErrorListener
from py_expression_eval import Parser

from QCompute import Define
from QCompute.OpenConvertor import ModuleErrorCode, ConvertorImplement
from QCompute.OpenConvertor.QasmToCircuit.BNF_Antlr4.gen.QASMLexer import InputStream, QASMLexer, CommonTokenStream
from QCompute.OpenConvertor.QasmToCircuit.BNF_Antlr4.gen.QASMParser import QASMParser
from QCompute.OpenConvertor.QasmToCircuit.BNF_Antlr4.gen.QASMVisitor import QASMVisitor
from QCompute.QPlatform import Error
from QCompute.QProtobuf import PBProgram, PBCircuitLine, PBFixedGate, PBRotationGate, PBCompositeGate, \
    PBMeasure


class CircuitLine:
    def __init__(self):
        self.pbCircuitLine = PBCircuitLine()
        self.pendingArgumentList: List[str] = None


class data(object):
    pass


class QasmToCircuit(ConvertorImplement):
    """
    Qasm To Circuit
    """

    def convert(self, qasmText) -> PBProgram:
        """
        QASM->Protobuf Circuit

        :param qasm: QASM text

        :return: Protobuf format of the circuit
        """

        inputStream = InputStream(qasmText)  # Convert string to byte stream
        lexer = QASMLexer(inputStream)  # lexical parsing
        stream = CommonTokenStream(lexer)  # Lexicon to symbol stream
        parser = QASMParser(stream)  # Symbol stream to syntax parsing
        parser.addErrorListener(_Listener())  # Mount error listener
        tree = parser.mainprog()  # Parsing starts from the top element of the syntax tree
        visitor = _Visitor()  # Syntax tree visitor
        visitor.visit(tree)  # Traversing the syntax tree using the visitor
        visitor.complete()
        visitor.program.source.qasm = qasmText
        return visitor.program


class QasmParseState(IntEnum):
    Main = 0
    Procedure = 1


FixedGateIdMap = dict(PBFixedGate.items())

RotationGateIdMap = dict(PBRotationGate.items())

CompositeGateIdMap = dict(PBCompositeGate.items())


class _Listener(ErrorListener):
    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        raise Error.ArgumentError(msg, ModuleErrorCode, FileErrorCode, 1)


class ScopeLevel:
    def __init__(self, qReg: Dict[str, int], cReg: Dict[str, int], state: QasmParseState, regMap: Dict,
                 argumentMap: Dict,
                 argumentExprList: List, argumentIdList: List[int], argumentValueList: List[str], procName: str,
                 circuitCounter: int):
        self.qReg = qReg
        self.cReg = cReg
        self.state = state
        self.regMap = regMap
        self.argumentMap = argumentMap
        self.argumentExprList = argumentExprList
        self.argumentIdList = argumentIdList
        self.argumentValueList = argumentValueList
        self.procName = procName
        self.circuitCounter = circuitCounter


class _Visitor(QASMVisitor):
    def __init__(self):
        self.program = PBProgram()
        self.program.sdkVersion = Define.sdkVersion

        # Predefine list of one argument operations.
        self.oneArgumentOp = {'ID', 'X', 'Y', 'Z', 'H', 'S', 'SDG', 'T', 'TDG', 'U', 'RX', 'RY', 'RZ'}
        # Predefine list of two argument operations.
        self.twoArgumentsOp = {'CX', 'CY', 'CZ', 'CH', 'SWAP', 'CU', 'CRX', 'CRY', 'CRZ'}
        # Predefine list of three argument operations.
        self.threeArgumentsOp = {'CCX', 'CSWAP'}

        self.containMeasure = False
        self.scopeStack: List[ScopeLevel] = []
        self.qRegVarMap: Dict[str, int] = {}
        self.cRegVarMap: Dict[str, int] = {}

        self.procedureVarSet = set()
        self.procedureMap = {}
        self.procedureArgumentMap = {}
        # Head
        self.usingQRegList: List[int] = []
        self.usingCRegList: List[int] = []
        # Body - circuit
        self.circuitList: List[CircuitLine] = []

        # Default scope level: global. 
        # Default state: main procedure.
        globalScopeLevel = ScopeLevel(
            qReg=self.qRegVarMap,
            cReg=self.cRegVarMap,
            state=QasmParseState.Main,
            regMap={},
            argumentMap={},
            argumentExprList=[],
            argumentIdList=[],
            argumentValueList=[],
            procName='',
            circuitCounter=0,
        )
        self.scopeStack.append(globalScopeLevel)

    def convertArguments(self, arguments: str) -> List[float]:
        """
        Convert arguments to float.

        :param arguments: type: str. Arguments.

        :return: type: List[float]. List of arguments (floating point data type).
        """
        argumentList = arguments.split(',')
        return [float(argument) for argument in argumentList]

    def prepareArguments(self, arguments: str) -> List[str]:
        argumentList = arguments.split(',')
        return argumentList

    def getArguments(self, ctx) -> Tuple[str, List[int]]:
        """
        Check arguments.
        # NOTE: explist may be undefined. According to grammar, arguments are seperated by ','.
        Traverse parser tree to check four arithmetic operations.
        The argumentExprList, argumentIdList and argumentValueList in the current scope are cleaned at last.

        :param ctx: type: object. QASMParser.UopContext.

        :return: type: Tuple[str, List[int]]. Converted arguments and list of argument ids.
        """
        arguments = ''
        argumentIdList: List[int] = []

        explist = ctx.explist()
        if explist:
            arguments = self.visitExplist(explist)
            scopeData = self.getScopeData()
            argumentIdList = scopeData.argumentIdList
            self.cleanArgumentExprList()
            self.cleanArgumentIdAndValue(True, True)

        return arguments, argumentIdList

    def getRealArgumentList(self, procArguments: str, pendingArgumentList: List[str], argumentMap) -> List[float]:
        """
        Get real values of arguments.

        :param procArguments: type: str. Procedure arguments.
        :param pendingArgumentList: type: List[str]. List of pending arguments.
        :param argumentMap: type: object. Argument map of procedure.
        
        :return realArgumentList: type: List[float]. List of real values of arguments.
        """
        argumentArray = procArguments.split(',')
        realArgumentList: List[float] = []
        if pendingArgumentList:
            for argument in pendingArgumentList:
                realVal = argument
                try:
                    reVal = float(realVal)
                    realArgumentList.append(reVal)
                    continue
                except ValueError:
                    pass

                # Check if pending arguments are params declared in procedure.
                for key, val in argumentMap.items():
                    callVal = argumentArray[val]
                    realVal = re.sub(f'\\b{key}\\b', callVal, realVal)

                # Real-valued calculation.
                try:
                    parser = Parser()
                    reVal = parser.parse(realVal).evaluate({})
                    realArgumentList.append(reVal)
                except Exception as ex:
                    raise Error.ArgumentError(
                        f'Calculation expression {argument} parsing error: {ex}.', ModuleErrorCode, FileErrorCode, 2)

        return realArgumentList

    def strMapToObj(strMap: Dict[str, Any]):
        obj = object()
        for k, v in strMap:
            setattr(obj, k, v)
        return obj

    def cleanArgumentExprList(self):
        scopeLen = len(self.scopeStack)
        scope = self.scopeStack[scopeLen - 1]
        del scope.argumentExprList[:]

    def cleanArgumentIdAndValue(self, cleanIdList: bool, cleanArgumentList: bool):
        """
        Clean arguments.

        :param cleanIdList: type: bool. If True, cleans argumentIdList.
        :param cleanArgumentList: type: bool. If True, cleans argumentValueList.
        """
        scopeLen = len(self.scopeStack)
        scope = self.scopeStack[scopeLen - 1]
        if cleanIdList:
            del scope.argumentIdList[:]

        if cleanArgumentList:
            del scope.argumentValueList[:]

    def getScopeData(self):
        """
        Get the top element of scopeStack.
        """
        scopeLen = len(self.scopeStack)
        scope = self.scopeStack[scopeLen - 1]
        return scope

    def getRegOrderNumber(self, regName: str, line: int, position: int) -> int:
        """
        Get the order number of regName at this scope.

        :param regName: type: str. Name of register.
        :param line: type: int. Row.
        :param position: type: int. Column.

        :return: type: int. order number.
        """
        regMap = self.getScopeData().regMap
        orderNum = regMap.get(regName)
        if orderNum is None:
            raise Error.ArgumentError(
                f'Qubit register name {regName} is undefined. line: {line}, position: {position}.', ModuleErrorCode,
                FileErrorCode, 3)
        return orderNum

    def checkQregVar(self, qRegId: str, line: int, position: int) -> bool:
        if qRegId is None:
            raise Error.ArgumentError(f'QReg {qRegId} is undefined. line: {line}, position: {position}.',
                                      ModuleErrorCode, FileErrorCode, 4)
        qReg = self.getScopeData().qReg
        # Check gate variable exist
        checkVar = qRegId in qReg
        if not checkVar:
            raise Error.ArgumentError(f'QReg {qRegId} doses not declared. line: {line}, position: {position}.',
                                      ModuleErrorCode, FileErrorCode, 5)
        return True

    def checkCregVar(self, cRegId: str, line: int, position: int) -> bool:
        if cRegId is None:
            raise Error.ArgumentError(f'CReg {cRegId} is undefined. line: {line}, position: {position}.',
                                      ModuleErrorCode, FileErrorCode, 6)
        # check gate variable exist
        cReg = self.getScopeData().cReg
        checkVar = cRegId in cReg
        if not checkVar:
            raise Error.ArgumentError(f'CReg {cRegId} doses not declared. line: {line}, position: {position}.',
                                      ModuleErrorCode, FileErrorCode, 7)
        return True

    def checkRegSizeInner(self, regVar: str, regSize: int, reqSize: int, line: int, position: int) -> bool:
        """
        Check if the access is out-of-bounds.
        :param regVar: type: str. Varaible of register.
        :param regSize: type: int. Size of register.
        :param reqSize: type: int. Size of request.
        :param line: type: int. Row.
        :param position: type: int. Columm.

        :return: type: bool. If False, the access is out-of-bounds. If True, the access is valid.
        """
        if regSize == 0:
            raise Error.ArgumentError(f'Variable {regVar} size is zero. line: {line}, position: {position}.',
                                      ModuleErrorCode, FileErrorCode, 8)
        elif reqSize >= regSize:
            raise Error.ArgumentError(f'Variable {regVar} out-of-bounds access. line: {line}, position: {position}.',
                                      ModuleErrorCode, FileErrorCode, 9)
        return True

    def checkQregSize(self, qRegVar: str, reqSize: int, line: int, position: int) -> bool:
        """
        Check size of quantum register.

        :param qRegVar: type: str. Quantum register variable.
        :param reqSize: type: int. Size of request.
        :param line: type: int. Row.
        :param position: type: int. Column. 

        :return: type: bool. If False, the access is out-of-bounds. If True, the access is valid.
        """
        self.checkQregVar(qRegVar, line, position)
        qReg = self.getScopeData().qReg
        regSize = qReg[qRegVar]
        return self.checkRegSizeInner(qRegVar, regSize, reqSize, line, position)

    def checkCregSize(self, cRegVar: str, reqSize: int, line: int, position: int) -> bool:
        """
        Check size of classic register.

        :param cRegVar: type: str. Classic register variable.
        :param reqSize: type: int. Size of request.
        :param line: type: int. Row.
        :param position: type: int. Column. 

        :return: type: bool. If False, the access is out-of-bounds. If True, the access is valid.
        """
        self.checkCregVar(cRegVar, line, position)
        cReg = self.getScopeData().cReg
        regSize = cReg[cRegVar]
        return self.checkRegSizeInner(cRegVar, regSize, reqSize, line, position)

    def checkQregLimit(self, qRegVal: str, line: int, position: int) -> bool:
        """
        Check limit of quantum register.

        :param line: type: int. Row.
        :param position: type: int. Column.

        :return: type: bool. True. If quantum register variable is declared more than once, throws exception.
        """
        qReg = self.getScopeData().qReg
        rSize = len(qReg)
        if rSize > 0:
            raise Error.ArgumentError(
                f'Qubit register variable declaration more than once. line: {line}, position: {position}.',
                ModuleErrorCode, FileErrorCode, 10)
        return True

    def checkCregLimit(self, qRegVal: str, line: int, position: int) -> bool:
        """
        Check limit of classic register.

        :param line: type: int. Row.
        :param position: type: int. Column.

        :return: type: bool. True. If classic register variable is declared more than once, throws exception.
        """
        cReg = self.getScopeData().cReg
        rSize = len(cReg)
        if rSize > 0:
            raise Error.ArgumentError(
                f'Classic register variable declaration more than once. line: {line}, position: {position}.',
                ModuleErrorCode, FileErrorCode, 11)
        return True

    def checkUOpArgumentIdList(self, idArgumentList: List, line: int, position: int) -> bool:
        """
        Check if idlist has been declared.
        
        :param idArgumentList: type: List. List of argument ids.
        :param line: type: int. Row.
        :param position: type: int. Column.

        :return: type: bool. True. If the same register is used, throws exception.
        """
        checkIds: List[str] = []
        for id in idArgumentList:
            rId = id.getText()
            # Check if quantum register is defined.
            self.checkQregVar(rId, line, position)
            if rId in checkIds:
                raise Error.ArgumentError(f'Illegal operation on same register. line: {line}, position: {position}.',
                                          ModuleErrorCode, FileErrorCode, 12)
            checkIds.append(rId)
        return True

    def checkBarrierIdList(self, idArgumentList: List, line: int, position: int) -> bool:
        """
        Check params in barrier operation.

        :params idArgumentList: type: List. List of params in barrier operation. 
        :params line: type: int. Row.
        :params position: type: int. Column. 

        :return: type: bool. If True. Barrier operation is valid. 
        """

        if len(idArgumentList):
            raise Error.ArgumentError(
                f'Illegal barrier operation on multiple register. line: {line}, position: {position}.', ModuleErrorCode,
                FileErrorCode, 13)
        rId = idArgumentList[0].getText()
        # Check if quantum register is defined.
        return self.checkQregVar(rId, line, position)

    def checkMeasureSize(self, qregId: str, cregId: str, line: int, position: int) -> bool:
        """
        Check registers of measurement instruction.

        :param qregId: type: str. The index of quantum register.
        :param cregId: type: str. The index of classic register.
        :param line: type: int. Row.
        :param position: type: int. Column.

        :return: bool. If True, measurement instruction is legal. 
        If size of quantum register is larger than 
        that of classic register, throws exception.
        """
        scopeData = self.getScopeData()
        qReg = scopeData.qReg
        cReg = scopeData.cReg
        qRegSize = qReg[qregId]
        cRegSize = cReg[cregId]
        if qRegSize > cRegSize:
            raise Error.ArgumentError(
                f'Illegal measure operation on different size of register. line:{line}, position:{position}.',
                ModuleErrorCode, FileErrorCode, 14)
        return True

    def checkArgumentsCount(self, opId: str, argumentCount: int, line: int, position: int) -> bool:
        """
        Check the count of arguments.
        The count of arguments must matches the requirement of 
        operation (one argument operation, two arguments operation,
        and three arguments operation).

        :param opId: type: str. The name of operation. For example, 'ID', 'CX', and 'CSWAP'.
        :param argumentCount: type: int. The count of argument(s).
        :param line: type: int. Row.
        :param position: type: int. Column.

        :return: bool. True. If the count of arguments is different from operation, throws exception.
        """
        checkOp = opId.upper()
        error = False
        if checkOp in self.oneArgumentOp:
            if argumentCount != 1:
                error = True
        elif checkOp in self.twoArgumentsOp and argumentCount != 2:
            if argumentCount != 2:
                error = True
        elif checkOp in self.threeArgumentsOp and argumentCount != 3:
            if argumentCount != 3:
                error = True

        if error:
            raise Error.ArgumentError(f'Illegal operation on gate. line: {line}, position: {position}.',
                                      ModuleErrorCode, FileErrorCode, 15)
        return True

    def checkProcName(self, procName: str, gateName: str, line: int, position: int) -> bool:
        """
        Check recursive call.
        # NOTE: recursive call is PROHIBITED.

        :param procName: type: str. The name of procedure.
        :param gateName: type: str. The name of gate.
        :param line: type: int. Row.
        :param position: type: int. Column.

        :return: type: bool. True. If recursive call exists, throws exception.
        """
        if procName == gateName:
            raise Error.ArgumentError(
                f'Illegal operation on gate, recursive call is prohibited. line: {line}, position: {position}.',
                ModuleErrorCode, FileErrorCode, 16)
        return True

    def checkProcArgumentsCount(self, procName: str, arguments: str, line: int, position: int) -> bool:
        """
        Check the count of argument(s) in procedure. The count 
        of arguments must matches the requirement of operation.

        :param procName: type: str. The name of procedure. 
        :param arguments: type: str. The arguments of procedure.
        :param line: type: int. Row.
        :param position: type: int. Column.

        :return: type: bool. True. If the count of arguments is different from operation, throws exception.
        """
        argumentArray = arguments.split(',')
        argumentCount = len(argumentArray) if len(arguments) > 0 else 0
        procArgumentCount = self.procedureMap[procName].argumentCount
        if argumentCount != procArgumentCount:
            raise Error.ArgumentError(f'Illegal operation on gate. line: {line}, position: {position}.',
                                      ModuleErrorCode, FileErrorCode, 17)
        return True

    def checkArgumentDeclared(self, argumentVal: str) -> bool:
        """
        Check if the argument is declared.

        :param argumentVal: type: str. The 
        """
        argumentMap = self.getScopeData().argumentMap
        return argumentVal in argumentMap

    def checkMeasureCommand(self, line: int, position: int) -> bool:
        """
        Check if measurement instruction exists. The measurement instruction must be the last instruction.

        :param line: type: int. Row.
        :param position: type: int. Column.
        
        :return: type: bool. True. If measurement instruction is not the last instruction, throws exception.
        """
        if self.containMeasure:
            raise Error.ArgumentError(
                f'Illegal operation in code. The measurement instruction must be the last instruction. line: {line}, position: {position}.',
                ModuleErrorCode, FileErrorCode, 18)

    def genSequenceArray(self, size: int):
        return [i for i in range(size)]

    def getProcCircuitList(self, procName: str, procArguments: str, regIdList: List[int]):
        """
        Recursively get circuits of procedures. 
        # NOTE: Use shallowCopy to get values of objects. Objects can not be modified. 
        Each pending argument is calculated by the currently used arguments.
        Specifically, the current procParams are real arguments. Their positions indicate the order of called procedures.
        Get the names and positions of parameters in the procedure. Use these names to check if these parameters exist in the current circuit. 
        If parameters of procedure are in the pendingArgumentList, get the positions of these parameters. Load arguments of these position from procParams.

        Procedures are called recursively. Circuits are processed recurrently. Merges all circuits to get the circuit list.

        Swap register is a mapping. For example, procedure q[3], q[2], q[1]
        Default declaration of procedure: qb0, qb1, qb2. The mapping is qb0 -> q[3], qb1 -> q[2], qb2 -> q[1]


        :param procName: type: str. The name of procedure. 
        :param procArguments: type: str. The arguments of procedure.
        :param regIdList: type: List[int]. List of register ids. 
        
        :return: List[CircuitLine]. The circuits of procedures.
        """
        circuitList: List[CircuitLine] = []
        procedureCircuitList = self.procedureMap[procName].circuitList

        argumentMap = self.procedureArgumentMap[procName]

        for circuit in procedureCircuitList:
            procedureName = circuit.pbCircuitLine.procedureName
            pendingArgumentList = circuit.pendingArgumentList

            newQRegList: List[int] = []
            for i in circuit.pbCircuitLine.qRegList:
                q = regIdList[i]
                newQRegList.append(q)

            realArgumentList = self.getRealArgumentList(procArguments, pendingArgumentList, argumentMap)
            if procedureName:
                gateName = circuit.pbCircuitLine.procedureName
                procArgumentsStr = ','.join([str(num) for num in realArgumentList])
                subCircuits = self.getProcCircuitList(gateName, procArgumentsStr, newQRegList)
                for sub in subCircuits:
                    circuitList.append(sub)

            # Store circuits.
            newCircuit = CircuitLine()
            newCircuit.pbCircuitLine.CopyFrom(circuit.pbCircuitLine)
            newCircuit.pbCircuitLine.argumentValueList[:] = realArgumentList
            newCircuit.pbCircuitLine.argumentIdList[:] = []
            newCircuit.pbCircuitLine.qRegList[:] = newQRegList

            if not procedureName:
                circuitList.append(newCircuit)
        return circuitList

    def createCircuit(self, circuitList: List[CircuitLine], arguments: str, argumentIdList: List[int],
                      argumentIdField: bool,
                      gateType: str, gateName: Union[str, int], regIdList: List[int]):
        circuit = CircuitLine()
        if gateType == 'fixedGate':
            circuit.pbCircuitLine.fixedGate = gateName
        if gateType == 'rotationGate':
            circuit.pbCircuitLine.rotationGate = gateName
        circuit.pbCircuitLine.qRegList[:] = regIdList

        if arguments:
            circuit.pbCircuitLine.argumentValueList[:] = self.convertArguments(arguments)

        if argumentIdField:
            circuit.pbCircuitLine.argumentIdList[:] = argumentIdList

        circuitList.append(circuit)

    def createProcCircuit(self, circuitList: List[CircuitLine], arguments: str, argumentIdList: List[int],
                          argumentIdField: bool, gateType: str, gateName: Union[str, int], regIdList: List[int]):
        """
        Create circuits of procedure. 

        :param circuitList: type: List[CircuitLine]. List of circuitLines.
        :param arguments: type: str. Arguments.
        :param argumentIdList: type: List[int]. List of argument ids.
        :param argumentIdField: type: bool. Mask of procedure.
        :param gateType: type: str. Type of gate.
        :param gateName: type: Union[str, int]. GateIdMap.
        :param regIdList: type: List[int]. List of register ids.

        """
        # Store circuits
        circuit = CircuitLine()
        setattr(circuit.pbCircuitLine, gateType, gateName)
        circuit.pbCircuitLine.qRegList[:] = regIdList

        if arguments:
            circuit.pendingArgumentList = self.prepareArguments(arguments)

        if argumentIdField:
            circuit.pbCircuitLine.argumentIdList[:] = argumentIdList

        circuitList.append(circuit)

    def expandCircuitList(self, circuitList: List[CircuitLine], arguments: str, argumentIdList: List[int],
                          argumentIdField: bool, gateType: str, gateName: Union[str, int], regIdList: List[int]):
        """
        Expands all the called circuits in procedure, and copys to external procedure. 
        
        :param circuitList: type: List[CircuitLine]. List of circuits.
        :param arguments: type: str. Arguments.
        :param gateName: type: Union[str, int]. GateIdMap.
        :param regIdList: type: List[int]. List of register ids.

        :return: None.
        """

        procCircuits = self.getProcCircuitList(gateName, arguments, regIdList);

        for sub in procCircuits:
            circuitList.append(sub)

    def createMeasureCircuit(self, circuitList: List[CircuitLine], qRegId: int, cRegId: int):
        circuit = CircuitLine()
        circuit.pbCircuitLine.measure.type = PBMeasure.Type.Z
        circuit.pbCircuitLine.measure.cRegList.append(cRegId)
        circuit.pbCircuitLine.qRegList.append(qRegId)

        circuitList.append(circuit)

    def createBarrierCircuit(self, circuitList: List[CircuitLine], qRegIdList: List[int]):
        circuit = CircuitLine()
        circuit.pbCircuitLine.barrier = True
        circuit.pbCircuitLine.qRegList[:] = qRegIdList
        circuitList.append(circuit)

    def visitQReg(self, ctx):
        """
        Check quantum registers. 
        Declaring multiple variables is temporarily not support.
        Check the same variables declared in classic registers.

        :param ctx: type: object. QASMParser.QRegContext
        """
        line: int = ctx.start.line
        position: int = ctx.start.column

        id: str = ctx.ID().getText()
        self.checkQregLimit(id, line, position)
        regSize = int(ctx.INT().getText())
        # create sequence number list for array
        regList = self.genSequenceArray(regSize)
        self.qRegVarMap[id] = regSize
        self.usingQRegList = regList

        if id in self.cRegVarMap:
            raise Error.ArgumentError(
                f'CReg {id} already declared. line: {line}, position: {position}.', ModuleErrorCode, FileErrorCode, 19)

        return ''

    def visitCReg(self, ctx):
        """
        Check classic registers. 
        Declaring multiple variables is temporarily not support.
        Check the same variables declared in quantum registers.

        :param ctx: type: object. QASMParser.CRegContext
        """
        line: int = ctx.start.line
        position: int = ctx.start.column

        id: str = ctx.ID().getText()
        self.checkCregLimit(id, line, position)
        regSize = int(ctx.INT().getText())
        # create sequence number list for array
        regList = self.genSequenceArray(regSize)
        self.cRegVarMap[id] = regSize
        self.usingCRegList = regList

        if id in self.qRegVarMap:
            raise Error.ArgumentError(f'QReg {id} already declared. line: {line}, position: {position}.',
                                      ModuleErrorCode, FileErrorCode, 20)

        return ''

    def getOpName(self, ctx) -> str:
        """
        Get the name of operation instruction.

        :param ctx: type: object. QASMParser.UopContext
        """
        # line number
        line: int = ctx.start.line
        position: int = ctx.start.column

        opCmd = ctx.ID()

        if opCmd:
            opId: str = opCmd.getText()
        else:
            raise Error.ArgumentError(f'Illegal operation on gate. line: {line}, position: {position}.',
                                      ModuleErrorCode, FileErrorCode, 21)

        return opId

    def getGateNameType(self, ctx, opId: str) -> Tuple[str, Union[str, int], bool]:
        """
        Get the type of gate operation.

        :param ctx: type: object. QASMParser.UopContext
        :param opId: type: str. The name of operation. For example, 'ID', 'CX', and 'CSWAP'.

        :return: type: Tuple[str, Union[str,int], bool]. (gateType, gateName, argumentIdField)
        """
        line: int = ctx.start.line
        position: int = ctx.start.column
        argumentIdField = False
        state = self.getScopeData().state
        if state == QasmParseState.Procedure:
            argumentIdField = True

        gateIdCheck = opId.upper()
        gateF = FixedGateIdMap.get(gateIdCheck)
        gateR = RotationGateIdMap.get(gateIdCheck)
        gateC = CompositeGateIdMap.get(gateIdCheck)
        gateProcedure = opId in self.procedureVarSet

        if gateF is not None:
            gateType = 'fixedGate'
            gateName = gateF
        elif gateR is not None:
            gateType = 'rotationGate'
            gateName = gateR
        elif gateC is not None:
            gateType = 'compositeGate'
            gateName = gateC
        elif gateProcedure:
            gateType = 'procedureName'
            gateName = opId
        else:
            raise Error.ArgumentError(
                f'Illegal operation on gate. line: ${line}, position: ${position}. line: {line}, position: {position}.',
                ModuleErrorCode, FileErrorCode, 22)
        return gateType, gateName, argumentIdField

    def getCurrentCircuitsContainer(self) -> List[CircuitLine]:
        """
        Get the current circuitList. If state is procedure, skip to the circuitList in procedureMap.
        """
        circuitList = self.circuitList
        scopeData = self.getScopeData()
        state = scopeData.state
        procName = scopeData.procName
        if state == QasmParseState.Procedure:
            circuitList = self.procedureMap[procName].circuitList
        return circuitList

    def idListOperation(self, ctx, opId: str, idList, circuitList: List[CircuitLine]):
        """
        Registers without index, such as h q;
        # NOTE: operation with multiple registers is not supported so far. Index can not be used in the procedure. Hence, operations are single-qubit operations.

        :param ctx: type: object. QASMParser.UopContext.
        :param opId: type: str. The name of operation. For example, 'ID', 'CX', and 'CSWAP'.
        :param idList: type: object. List of ids.
        :param circuitList: type: List[CircuitLine]. List of circuitLines.
        """
        # line number
        line: int = ctx.start.line
        position: int = ctx.start.column
        gateType, gateName, argumentIdField = self.getGateNameType(ctx, opId)
        arguments, argumentIdList = self.getArguments(ctx)

        idArguments = idList.ID()
        self.checkUOpArgumentIdList(idArguments, line, position)
        # Check variable declaration. Only one id argument uses.
        argument: str = idArguments[0].getText()
        scopeData = self.getScopeData()
        state = scopeData.state
        qReg = scopeData.qReg
        regMap = scopeData.regMap
        procName = scopeData.procName

        # Check the number of used registers. Only one register can be expanded so far. 
        idLens = len(idArguments)
        if idLens > 1:
            # command of procedure. Call registers declared in multiple procedures. 
            # Get list of registers according to their positions.
            regList: List[int] = [regMap[id.getText()] for id in idArguments]

            if gateType == 'procedureName':
                self.checkProcName(procName, gateName, line, position)
                self.checkProcArgumentsCount(gateName, arguments, line, position)

            self.createProcCircuit(circuitList, arguments, argumentIdList, argumentIdField, gateType, gateName, regList)
        else:
            # Since some operations require external data, creates objects for return.
            if state == QasmParseState.Procedure:
                # Get id of used register.
                regVal = self.getRegOrderNumber(argument, line, position)
                self.createProcCircuit(circuitList, arguments, argumentIdList, argumentIdField, gateType, gateName,
                                       [regVal])
            else:
                size = qReg.get(argument)
                for i in range(size):
                    self.createCircuit(circuitList, arguments, argumentIdList, argumentIdField, gateType, gateName, [i])

    def mixListOperation(self, ctx, opId: str, mixList, circuitList):
        """
        Operation registers with index, such as h q[1];

        :param ctx: type: object. QASMParser.UopContext
        :param opId: type: str. Operation name such as u, cx
        :param mixList: type: object. QASMParser.MixedlistContext
        :param circuitList: type: List[CircuitLine]. List of circuitLines.
        """
        line: int = ctx.start.line
        position: int = ctx.start.column
        gateType, gateName, argumentIdField = self.getGateNameType(ctx, opId)
        arguments, argumentIdList = self.getArguments(ctx)
        scopeData = self.getScopeData()
        state = scopeData.state

        mixId = mixList.ID()
        mixArgument = mixList.INT()

        if len(mixId) != len(mixArgument):
            raise Error.ArgumentError(
                f'Illegal operation on gate. line: {line}, position: {position}.', ModuleErrorCode, FileErrorCode, 23)

        # Check number of arguments.
        self.checkArgumentsCount(opId, len(mixId), line, position)

        # Check validity of variables.
        regName: str = None
        for m in mixId:
            mid = m.getText()
            if not regName:
                regName = mid
            self.checkQregVar(mid, line, position)
        # Check out-of-bounds access.
        usageIds = []
        for p in mixArgument:
            rIndex = int(p.getText())
            self.checkQregSize(regName, rIndex, line, position)
            usageIds.append(rIndex)

        # to pb has two situations,
        # 1. Procedure calls other procedures;
        # 2. Directly calls procedure.
        # Arguments are calculated. Called circuits are expanded and stored.
        if state == QasmParseState.Procedure:
            self.createProcCircuit(circuitList, arguments, argumentIdList, argumentIdField, gateType, gateName,
                                   usageIds)
        else:
            if gateType == 'procedureName':
                self.checkProcArgumentsCount(gateName, arguments, line, position);
                self.expandCircuitList(circuitList, arguments, argumentIdList, argumentIdField, gateType, gateName,
                                       usageIds)
            else:
                self.createCircuit(circuitList, arguments, argumentIdList, argumentIdField, gateType, gateName,
                                   usageIds)

    def visitUop(self, ctx):
        """
        :param ctx: type: object. QASMParser.UopContext

        As noted in document, declaring multiple registers is temporarily not supported.
        In terms of one register operation, it allows to declare registers without specific indices. For example, h q; It means the parallel operation on all qubits of q. q[5] ==> h q[0] --> h q[4].
        The multiple registers operation does not work, such as swap q1 q2.
        But swap q[0], q[1] is available. 
        This function will be further improved for multiple registers operation.

        Description: 
            According to the definition of OpenQasm, registers can be used parallel, such as U(0,0,0) q; q is parallel operated based on its value. 
            If q[5] is defined, the above operation is U(0,0,0) q[0] -> U(0,0,0) q[5]
            It is similar to other operations like CX q, r;
            Moreover, there exists partial matching which is common for two arguments operation. For example, CX q[0], r; It means q[0] is mapping to each qubit of r. This operation is executed according to the value of r.
        
        Regular check:
            Valid command. 
            Valid variable. 
            Out-of-bounds cccess. 
        """

        # line number
        line: int = ctx.start.line
        position: int = ctx.start.column

        # Check the current state. If procedure is being processed, 
        # turns to process the circuitList of procedureMap.
        circuitList = self.getCurrentCircuitsContainer()

        # Operation command, such as u, cx, swap, ry, etc.
        opId = self.getOpName(ctx)

        # Check if measurement instruction exists. If exists, it is add to the last.
        self.checkMeasureCommand(line, position)

        anyList = ctx.anylist()
        # anyList may be undefined.
        if anyList is not None:
            idList = anyList.idlist()
            mixList = anyList.mixedlist()
            # Since params have no index, length of idList is needed.
            if idList is not None:
                self.idListOperation(ctx, opId, idList, circuitList)
            elif mixList is not None:
                # Register with indices, such as swap q[0], q[1];
                # Temporarily NOT support one-to-many and many-to-one registers.
                self.mixListOperation(ctx, opId, mixList, circuitList)
        return ''

    def visitMeasureOp(self, ctx):
        """
        Visit measurement instruction.

        :param ctx: type: object. ctx is QASMParser.MeasureOpContext.

        Format: measure q -> c;
        # NOTE: the size of q is not greater than c.

        measure q[0] -> c[0];
        """
        line: int = ctx.start.line
        position: int = ctx.start.column

        argument = ctx.argument()
        qReg = argument[0]
        cReg = argument[1]

        qRegId: str = qReg.ID().getText()
        cRegId: str = cReg.ID().getText()

        self.checkQregVar(qRegId, line, position)
        self.checkCregVar(cRegId, line, position)

        qRegInt = qReg.INT()
        cRegInt = cReg.INT()

        if qRegInt is None and cRegInt is None:
            # Register without index.
            # Check the sizes of qReg and cReg. The size of 
            # qReg must be less than or equal to that of cReg.
            self.checkMeasureSize(qRegId, cRegId, line, position)

            qRegSize = self.qRegVarMap.get(qRegId)
            for i in range(qRegSize):
                self.createMeasureCircuit(self.circuitList, i, i)

            # Measurement instruction exists.
            self.containMeasure = True

        elif qRegInt is not None and cRegInt is not None:
            # Registers with indices.
            # Check out-of-bounds access.
            qInt = int(qRegInt.getText())
            cInt = int(cRegInt.getText())
            self.checkQregSize(qRegId, qInt, line, position)
            self.checkCregSize(cRegId, cInt, line, position)

            self.createMeasureCircuit(self.circuitList, qInt, cInt)

            # Measurement instruction exists.
            self.containMeasure = True

        else:
            # Error. One register uses indices, the other one does not use index.
            raise Error.ArgumentError(f'Illegal measure operation on gate. line: {line}, position: {position}.',
                                      ModuleErrorCode, FileErrorCode, 24)

        return ''

    def visitBarrierOp(self, ctx):
        """
        Visit barrier operation.

        :param ctx: type: object. ctx is BarrierOpContext.
        
        Example: 
            barrier q;
            barrier q[0], ...;
        """
        line = ctx.start.line
        position = ctx.start.column

        anyList = ctx.anylist()

        # Get current circuit list. Check if procedure needs to be turn. 
        circuitList = self.getCurrentCircuitsContainer()

        idList = anyList.idlist()
        mixList = anyList.mixedlist()

        if idList is not None:
            idArguments = idList.ID()
            self.checkBarrierIdList(idArguments, line, position)
            # Check variable declaration. Only one id argument uses.
            argument = idArguments[0].getText()
            qReg = self.getScopeData().qReg
            size = qReg.get(argument)

            for i in range(size):
                self.createBarrierCircuit(circuitList, [i])

        # Register with indices, such as barrier q[0], q[1].
        if mixList is not None:
            mixId = mixList.ID()
            mixArgument = mixList.INT()
            if len(mixId) != len(mixArgument):
                raise Error.ArgumentError(f'Illegal operation on gate. line: {line}, position: {position}.',
                                          ModuleErrorCode, FileErrorCode, 25)

            # Check validity of variables.
            regName: str = None
            for m in mixId:
                mid = m.getText()
                if not regName:
                    regName = mid
                self.checkQregVar(mid, line, position)
            # Check out-of-bounds access.
            usageIds = []
            for p in mixArgument:
                rIndex = int(p.getText())
                self.checkQregSize(regName, rIndex, line, position)
                usageIds.append(rIndex)

            self.createBarrierCircuit(circuitList, usageIds)
        return ''

    def visitExplist(self, ctx):
        """
        Visit expression list.
        
        :param ctx: type: object. QASMParser.UopContext.

        :return: expression. 
        """
        line: int = ctx.start.line
        position: int = ctx.start.column
        scopeData = self.getScopeData()
        argumentIdList = scopeData.argumentIdList
        argumentValueList = scopeData.argumentValueList
        argumentMap = scopeData.argumentMap
        argumentExprList = scopeData.argumentExprList

        expList = []
        exp = ctx.exp()
        for item in exp:
            expVal = item.accept(self)

            obj = json.loads(expVal)
            flag = obj['flag']
            text = obj['text']
            val = obj['val']
            expr = obj['expr']

            if flag == 0:
                # Get the index.
                index = argumentMap[text]
                argumentIdList.append(index)
                # param is marked with '0'. 
                argumentValueList.append('0')
                expList.append(text)
            else:
                argumentIdList.append(-1)
                # param is marked with text. 
                argumentValueList.append(text)
                expList.append(val)
            # Save expression parsed by antlr.
            argumentExprList.append(expr)

        # Clean argument ids and values. Set ret=False to cancel.
        ret = True
        for argumentId in argumentIdList:
            if argumentId != -1:
                ret = False
                break
        if ret:
            self.cleanArgumentIdAndValue(True, False)

        retExp = ','.join(expList)

        return retExp

    def visitExp(self, ctx):
        """
        Visit expression.
        # NOTE: both params and arguments may exist in expression. For example, u(param0, pi/2, param1).
        # Only arguments can be used for calculation.

        :param ctx: type: object. QASMParser.ExpContext

        :return: type: json. Expression.
        Example: 
            {
                'flag': -1,
                'text': calcText,
                'val': str(reVal),
                'expr': exprList
            }
        """

        line: int = ctx.start.line
        position: int = ctx.start.column

        exprList = []

        calcList = []

        count = ctx.getChildCount()

        calcArgumentList = []

        checkItem: List[bool] = []
        for i in range(count):
            item = ctx.getChild(i)
            itemText = item.getText()
            # check if argument has been declared.
            checkFlag = self.checkArgumentDeclared(itemText)
            checkItem.append(checkFlag)
            if checkFlag:
                calcArgumentList.append(itemText)
            calcList.append(itemText)
            # pi => PI for parsing.
            checkPI = itemText.find('pi')
            if checkPI != -1:
                itemText = itemText.upper()
            exprList.append(itemText)
        # Check arguments.
        calcText = ''.join(calcList)
        exprText = ''.join(exprList)
        checkParam = self.checkArgumentDeclared(calcText)
        if checkParam:
            return json.dumps({
                'flag': 0,
                'text': calcText,
                'val': calcText,
                'expr': exprList
            })

        # Parse expression.
        try:
            # Check if params exist in expression. If yes, get the mapping values of params for parsing.
            # Since the values of params are not defined in the declaration of procedure, expressions are stored temporarily till they are called.
            ret = True
            for item in checkItem:
                if item != False:
                    ret = False
                    break
            if not ret:
                # store expression.
                return json.dumps({
                    'flag': -1,
                    'text': exprText,
                    'val': exprText,
                    'expr': exprList
                })

            parser = Parser()
            reVal = parser.parse(exprText).evaluate({})

        except Exception as ex:
            raise Error.ArgumentError(
                f'Calculation expression {calcText} parsing error: {ex}. line: {line}, position: {position}.',
                ModuleErrorCode, FileErrorCode, 26)

        return json.dumps({
            'flag': -1,
            'text': calcText,
            'val': str(reVal),
            'expr': exprList
        })

    def visitGatedecl(self, ctx):
        """
        Declaration of procedure.
        Current scope is stacked into procStack.

        :param ctx: type: object. QASMParser.GatedeclContext
        """
        # line number
        line: int = ctx.start.line
        position: int = ctx.start.column

        gateId: str = ctx.ID().getText()
        # Check duplication of procedure names.
        if gateId in self.procedureVarSet:
            # Duplication
            raise Error.ArgumentError(
                f'Duplicated procedure name ${gateId}, line: ${line}, position: ${position}.', ModuleErrorCode,
                FileErrorCode, 27)

        # Check if function bodies exist. 
        noBody = False
        parent = ctx.parentCtx

        gopList = parent.goplist()
        if gopList is None:
            noBody = True

        arguments = None
        regs = None
        # According to grammar of Qasm, if length of idlist is 2, the first content is argument list, the second content is register list.
        # If length of idlist is 1, instruction only contains register list.
        contentList = ctx.idlist()
        idListLen = len(contentList)
        if idListLen == 1:
            regs = contentList[0].ID()
        elif idListLen == 2:
            arguments = contentList[0].ID()
            regs = contentList[1].ID()

        # Check arguments of procedure.
        argumentMap = {}
        if arguments:
            for index, argument in enumerate(arguments):
                argumentMap[argument.getText()] = index

        # List of used registers.
        regList = self.genSequenceArray(len(regs))

        # Construct data structure declared in procedure.
        procData = data()
        procData.argumentCount = len(arguments) if arguments else 0
        procData.usingQRegList = regList
        procData.circuitList = []

        # Store names of procedures for parsing.
        self.procedureVarSet.add(gateId)

        # Store argumentMap of procedures for converting.
        self.procedureArgumentMap[gateId] = argumentMap
        # If no function body, directly adds declaration and returns.
        self.procedureMap[gateId] = procData
        if not noBody:
            regMap = {}
            procVarMap = {}
            for index, reg in enumerate(regs):
                procVarMap[reg.getText()] = 1
                regMap[reg.getText()] = index

            # Stack of procedures.
            procStack = ScopeLevel(
                qReg=procVarMap,
                cReg={},
                state=QasmParseState.Procedure,
                regMap=regMap,
                argumentMap=argumentMap,

                argumentExprList=[],
                argumentIdList=[],
                argumentValueList=[],

                procName=gateId,
                circuitCounter=0,
            )
            self.scopeStack.append(procStack)

        return ''

    def visitGoplist(self, ctx):
        """
        Internal declaration of procedure. 
        State: procedure declaration.
        State is recovered to procedure after parsing.

        :param ctx: type: object. QASMParser.GoplistContext
        """
        self.visitChildren(ctx)
        self.scopeStack.pop()

        return ''

    def complete(self):
        head = self.program.head
        head.usingQRegList[:] = self.usingQRegList
        head.usingCRegList[:] = self.usingCRegList
        body = self.program.body
        for circuit in self.circuitList:
            body.circuit.append(circuit.pbCircuitLine)
