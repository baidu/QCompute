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
import json
import re
from enum import IntEnum
from typing import List, Dict, Union, Tuple, Any, Set

from antlr4.error.ErrorListener import ErrorListener
from py_expression_eval import Parser

from QCompute import sdkVersion
from QCompute.OpenConvertor import ModuleErrorCode, ConvertorImplement
from QCompute.OpenConvertor.QasmToCircuit.BNF_Antlr4.gen.QASMLexer import InputStream, QASMLexer, CommonTokenStream
from QCompute.OpenConvertor.QasmToCircuit.BNF_Antlr4.gen.QASMParser import QASMParser
from QCompute.OpenConvertor.QasmToCircuit.BNF_Antlr4.gen.QASMVisitor import QASMVisitor
from QCompute.QPlatform import Error
from QCompute.QProtobuf import PBProgram, PBCircuitLine, PBFixedGate, PBRotationGate, PBCompositeGate, \
    PBMeasure

FileErrorCode = 5


class CircuitLine:
    def __init__(self):
        self.pbCircuitLine = PBCircuitLine()
        self.pendingArgumentList = None  # type: List[str]


class data(object):
    pass


class QasmToCircuit(ConvertorImplement):
    """
    Qasm To Circuit
    """

    def convert(self, qasmText):
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
        visitor = _Visitor()  # set up visitor
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
    program = PBProgram()
    program.sdkVersion = sdkVersion

    # 预定义单参数指令列表
    oneArgumentOp = {'ID', 'X', 'Y', 'Z', 'H', 'S', 'SDG', 'T', 'TDG', 'U', 'RX', 'RY', 'RZ'}
    # 预定义双参数指令列表
    twoArgumentsOp = {'CX', 'CY', 'CZ', 'CH', 'SWAP', 'CU', 'CRX', 'CRY', 'CRZ'}
    # 预定义三参数指令列表
    threeArgumentsOp = {'CCX', 'CSWAP'}

    # 记录分析程序内部信息，变量等
    containMeasure = False  # type: bool
    # 作用域保存栈
    scopeStack = []  # type: List[ScopeLevel]
    # qreg map
    qRegVarMap = {}  # type: Dict[str, int]
    # creg map
    cRegVarMap = {}  # type: Dict[str, int]
    # procedure map，子程序使用的变量记录
    procedureVarSet = set()
    # 子程序记录
    procedureMap = {}  # type: Dict
    # 子程序参数信息记录
    procedureArgumentMap = {}  # type: Dict
    # Head
    usingQRegList = []  # type: List[int]
    usingCRegList = []  # type: List[int]
    # Body - circuit
    circuitList = []  # type: List[CircuitLine]

    def __init__(self):
        # 默认是全局作用域,
        # 状态机，默认是主程序
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
        # 字符串转换成数字
        argumentList = arguments.split(',')
        ret = []  # type: List[float]
        for argument in argumentList:
            ret.append(float(argument))
        return ret

    # 预先分解列表参数，等形参有具体的数值时候再进行计算填充
    def prepareArguments(self, arguments: str) -> List[str]:
        # 每个参数
        argumentList = arguments.split(',')
        return argumentList

    # 参数检验处理
    def getArguments(self, ctx) -> Tuple[str, List[int]]:
        arguments = ''
        argumentIdList = []  # type: List[int]
        # 四则运算检查
        explist = ctx.explist()
        # may be undefined - 检查下QASMParser对应的定义
        if explist:
            # 这是参数列表，可能不存在，根据语法定义，多参数使用,分隔
            # 这里进行实数的计算和处理
            arguments = self.visitExplist(explist)
            # 记录了实际的参数转换信息
            scopeData = self.getScopeData()
            argumentIdList = scopeData.argumentIdList
            # 数据要清理，后续调用需要使用
            self.cleanArgumentExprList()
            self.cleanArgumentIdAndValue(True, True)

        return arguments, argumentIdList

    # 进行参数的转换和计算
    def getRealArgumentList(self, procArguments: str, pendingArgumentList: List[str], argumentMap) -> List[float]:
        argumentArray = procArguments.split(',')
        realArgumentList = []  # type: List[float]
        if pendingArgumentList:
            for argument in pendingArgumentList:
                realVal = argument
                # 如果已经是数字，跳过
                try:
                    reVal = float(realVal)
                    realArgumentList.append(reVal)
                    continue
                except ValueError:
                    pass

                # 查看待决参数，是否包含子程序声明的形参
                for key, val in argumentMap.items():
                    # 需要总体的字符匹配，不能匹配部分，不然会出错误，比如param10 可能被识别成param1
                    callVal = argumentArray[val]
                    realVal = re.sub(f'\\b{key}\\b', callVal, realVal)
                # 尝试进行实数计算
                # 尝试解析，解析错误会抛出异常
                try:
                    parser = Parser()
                    reVal = parser.parse(realVal).evaluate({})
                    realArgumentList.append(reVal)
                except Exception as ex:
                    # 转换错误，抛出自定义的异常
                    raise Error.ArgumentError(
                        f'Calculation expression {argument} parsing error: {ex}.', ModuleErrorCode, FileErrorCode, 2)

        return realArgumentList

    # 转换map到对象
    def strMapToObj(strMap: Dict[str, Any]):
        obj = object()
        for k, v in strMap:
            setattr(obj, k, v)
        return obj

    def cleanArgumentExprList(self):
        scopeLen = len(self.scopeStack)
        scope = self.scopeStack[scopeLen - 1]
        del scope.argumentExprList[:]

    # 清理变量
    def cleanArgumentIdAndValue(self, cleanIdList: bool, cleanArgumentList: bool):
        scopeLen = len(self.scopeStack)
        scope = self.scopeStack[scopeLen - 1]
        if cleanIdList:
            del scope.argumentIdList[:]

        if cleanArgumentList:
            del scope.argumentValueList[:]

    # 返回当前作用域下的检测结构信息，也就是栈顶元素
    def getScopeData(self):
        scopeLen = len(self.scopeStack)
        scope = self.scopeStack[scopeLen - 1]
        return scope

    # 返回当前作用域下寄存器符号映射的顺序数字
    def getRegOrderNumber(self, regName: str, line: int, position: int) -> int:
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
        # check gate variable exist
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

    # 检查是否是越界请求
    def checkRegSizeInner(self, regVar: str, regSize: int, reqSize: int, line: int, position: int) -> bool:
        if regSize == 0:
            # 尺寸为0
            raise Error.ArgumentError(f'Variable {regVar} size is zero. line: {line}, position: {position}.',
                                      ModuleErrorCode, FileErrorCode, 8)
        elif reqSize >= regSize:
            # 越界访问
            raise Error.ArgumentError(f'Variable {regVar} out-of-bounds access. line: {line}, position: {position}.',
                                      ModuleErrorCode, FileErrorCode, 9)
        return True

    # 查验寄存器变量，是否正确请求引用的尺寸
    def checkQregSize(self, qRegVar: str, reqSize: int, line: int, position: int) -> bool:
        self.checkQregVar(qRegVar, line, position)
        qReg = self.getScopeData().qReg
        regSize = qReg[qRegVar]
        return self.checkRegSizeInner(qRegVar, regSize, reqSize, line, position)

    # 查验寄存器变量，是否正确请求引用的尺寸
    def checkCregSize(self, cRegVar: str, reqSize: int, line: int, position: int) -> bool:
        self.checkCregVar(cRegVar, line, position)
        cReg = self.getScopeData().cReg
        regSize = cReg[cRegVar]
        return self.checkRegSizeInner(cRegVar, regSize, reqSize, line, position)

    # 查验寄存器声明限制 - qubit
    def checkQregLimit(self, qRegVal: str, line: int, position: int) -> bool:
        qReg = self.getScopeData().qReg
        rSize = len(qReg)
        if rSize > 0:
            # 暂时禁止多于1个的变量声明
            raise Error.ArgumentError(
                f'Qubit register variable declaration more than once. line: {line}, position: {position}.',
                ModuleErrorCode, FileErrorCode, 10)
        return True

    # 查验寄存器声明限制 - 经典寄存器
    def checkCregLimit(self, qRegVal: str, line: int, position: int) -> bool:
        cReg = self.getScopeData().cReg
        rSize = len(cReg)
        if rSize > 0:
            # 暂时禁止多于1个的变量声明
            raise Error.ArgumentError(
                f'Classic register variable declaration more than once. line: {line}, position: {position}.',
                ModuleErrorCode, FileErrorCode, 11)
        return True

    # 暂时不支持多寄存器， 语句参数中，多寄存器名字也不能一样
    def checkUOpArgumentIdList(self, idArgumentList: List, line: int, position: int) -> bool:
        # 查验idlist是否已经声明
        checkIds = []  # type: List[str]
        for id in idArgumentList:
            rId = id.getText()
            # 检查引用是否声明过
            self.checkQregVar(rId, line, position)
            if rId in checkIds:
                # 检验是否重复
                raise Error.ArgumentError(f'Illegal operation on same register. line: {line}, position: {position}.',
                                          ModuleErrorCode, FileErrorCode, 12)
            checkIds.append(rId)
        return True

    # 拦截器参数检验
    def checkBarrierIdList(self, idArgumentList: List, line: int, position: int) -> bool:
        # 查验idlist是否已经声明
        checkIds = []  # type: List[str]
        if len(idArgumentList):
            raise Error.ArgumentError(
                f'Illegal barrier operation on multiple register. line: {line}, position: {position}.', ModuleErrorCode,
                FileErrorCode, 13)
        rId = idArgumentList[0].getText()
        # 检查引用是否声明过
        return self.checkQregVar(rId, line, position)

    # 检查测量操作中，寄存器的尺寸要求
    def checkMeasureSize(self, qregId: str, cregId: str, line: int, position: int) -> bool:
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

    # 检查指令的参数数量是否符合要求
    def checkArgumentsCount(self, opId: str, argumentCount: int, line: int, position: int) -> bool:
        checkOp = opId.upper()
        # 单个
        error = False  # type: bool
        if checkOp in self.oneArgumentOp:
            if argumentCount != 1:
                error = True
        elif checkOp in self.twoArgumentsOp and argumentCount != 2:
            if argumentCount != 2:
                error = True
        elif checkOp in self.threeArgumentsOp and argumentCount != 3:
            if argumentCount != 3:
                error = True
        # 查验检查结果
        if error:
            raise Error.ArgumentError(f'Illegal operation on gate. line: {line}, position: {position}.',
                                      ModuleErrorCode, FileErrorCode, 15)
        return True

    # 禁止递归调用检查
    def checkProcName(self, procName: str, gateName: str, line: int, position: int) -> bool:
        if procName == gateName:
            raise Error.ArgumentError(
                f'Illegal operation on gate, recursive call is prohibited. line: {line}, position: {position}.',
                ModuleErrorCode, FileErrorCode, 16)
        return True

    # 检查调用子程序指令的参数数量是否符合要求
    def checkProcArgumentsCount(self, procName: str, arguments: str, line: int, position: int) -> bool:
        # 获取子程序注册的参数数量
        argumentArray = arguments.split(',')
        argumentCount = len(argumentArray) if len(arguments) > 0 else 0
        procArgumentCount = self.procedureMap[procName].argumentCount
        if argumentCount != procArgumentCount:
            raise Error.ArgumentError(f'Illegal operation on gate. line: {line}, position: {position}.',
                                      ModuleErrorCode, FileErrorCode, 17)
        return True

    # 检查参数是否是形参
    def checkArgumentDeclared(self, argumentVal: str) -> bool:
        argumentMap = self.getScopeData().argumentMap
        return argumentVal in argumentMap

    # 检查测量指令是否已经存在
    def checkMeasureCommand(self, line: int, position: int) -> bool:
        if self.containMeasure:
            raise Error.ArgumentError(
                f'Illegal operation in code. The measurement instruction must be the last instruction. line: {line}, position: {position}.',
                ModuleErrorCode, FileErrorCode, 18)

    def genSequenceArray(self, size: int):
        # create sequence number list for array
        regList = []  # type: List[int]
        for i in range(size):
            regList.append(i)
        return regList

    # 递归获取子程序的电路，进行内联处理，并且在必要时计算参数
    def getProcCircuitList(self, procName: str, procArguments: str, regIdList: List[int]):
        circuitList = []  # type: List[CircuitLine]
        procedureCircuitList = self.procedureMap[procName].circuitList
        # 子程序形式参数数组
        argumentMap = self.procedureArgumentMap[procName]

        # 循环处理 - 注意，这里不能直接对象赋值，因为是引用关系，会干扰破坏后续操作，需要使用浅拷贝赋值
        for circuit in procedureCircuitList:
            procedureName = circuit.pbCircuitLine.procedureName
            pendingArgumentList = circuit.pendingArgumentList
            # 逐个解析尚未计算的参数，把当前调用使用的参数注入并且计算
            # 注入计算的方式：当前的procParams都是实值，这些参数值在数组中的位置，代表了子程序代码调用的顺序关系；
            # 查询子程序声明阶段注册的形式参数名和位置关系，并且使用这些形式参数名，查询当前电路调用的参数中，是否有这些形参名，调用列表是：pendingParams
            # 如果pendingParams中的参数中，包含了子程序内的形参，则查找出这个形参的位置属性，用同样位置procParams的实参替换，再进行计算，得到实参值
            # 如果是子程序，则递归调用，如果是普通的电路，则循环处理，最后归并出完整的电路列表
            # 注意，此时的电路信息内，已经都是实参信息，没有形参，因为已经被计算完毕

            # 寄存器替换，它是一种映射： 举例，外部调用是： procedure q[3], q[2], q[1]
            # 子程序声明默认都是： qb0, qb1, qb2，所以映射关系就是 qb0 -> q[3], qb1 -> q[2], qb2 -> q[1]
            # 最终展开时，寄存器需要还原到子程序外部的，实际使用的寄存器，所以要进行映射处理
            newQRegList = []  # type: List[int]
            for i in circuit.pbCircuitLine.qRegList:
                # 获取外部的寄存器值
                q = regIdList[i]
                newQRegList.append(q)

            realArgumentList = self.getRealArgumentList(procArguments, pendingArgumentList, argumentMap)
            if procedureName:
                # 子程序调用，需要递归处理
                gateName = circuit.pbCircuitLine.procedureName
                # 解析子程序调用的参数值，需要得到具体的数值才能继续推进解析
                procArgumentsStr = ','.join([str(num) for num in realArgumentList])
                # 变换一下寄存器，需要使用最外层的原始寄存器进行展开处理
                subCircuits = self.getProcCircuitList(gateName, procArgumentsStr, newQRegList)
                for sub in subCircuits:
                    circuitList.append(sub)

            # 电路参数替换
            # 存储数据
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
        # 存储数据
        circuit = CircuitLine()
        if gateType == 'fixedGate':
            circuit.pbCircuitLine.fixedGate = gateName
        if gateType == 'rotationGate':
            circuit.pbCircuitLine.rotationGate = gateName
        circuit.pbCircuitLine.qRegList[:] = regIdList

        if arguments:
            # 设置规范格式
            circuit.pbCircuitLine.argumentValueList[:] = self.convertArguments(arguments)
        # else:
        #     circuit.argumentValueList.clear()

        if argumentIdField:
            circuit.pbCircuitLine.argumentIdList[:] = argumentIdList
            # else:
            #     circuit.argumentIdList.clear()

        circuitList.append(circuit)

    # 针对子程序中的参数，特殊处理
    def createProcCircuit(self, circuitList: List[CircuitLine], arguments: str, argumentIdList: List[int],
                          argumentIdField: bool, gateType: str, gateName: Union[str, int], regIdList: List[int]):
        # 存储数据
        circuit = CircuitLine()
        setattr(circuit.pbCircuitLine, gateType, gateName)
        circuit.pbCircuitLine.qRegList[:] = regIdList

        if arguments:
            circuit.pendingArgumentList = self.prepareArguments(arguments)
        # else:
        #     circuit.argumentValueList.clear()

        if argumentIdField:
            circuit.pbCircuitLine.argumentIdList[:] = argumentIdList
            # else:
            #     circuit.argumentIdList.clear()

        circuitList.append(circuit)

    def expandCircuitList(self, circuitList: List[CircuitLine], arguments: str, argumentIdList: List[int],
                          argumentIdField: bool, gateType: str, gateName: Union[str, int], regIdList: List[int]):
        # 对子程序的调用，需要递归展开处理，把所有此子程序的电路调用，都复制到外部程序，并且解析计算参数值
        procCircuits = self.getProcCircuitList(gateName, arguments, regIdList);
        # combine it
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
        # 存储数据
        circuitList.append(circuit)

    def visitQReg(self, ctx):
        # 暂时禁止1个以上变量声明
        # line number
        line = ctx.start.line  # type: int
        position = ctx.start.column  # type: int

        id = ctx.ID().getText()  # type: str
        self.checkQregLimit(id, line, position)
        regSize = int(ctx.INT().getText())
        # create sequence number list for array
        regList = self.genSequenceArray(regSize)
        self.qRegVarMap[id] = regSize
        self.usingQRegList = regList

        # 同时检查cReg, 不能有同名变量
        if id in self.cRegVarMap:
            raise Error.ArgumentError(
                f'CReg {id} already declared. line: {line}, position: {position}.', ModuleErrorCode, FileErrorCode, 19)

        return ''

    def visitCReg(self, ctx):
        # 暂时禁止1个以上变量声明
        # line number
        line = ctx.start.line  # type: int
        position = ctx.start.column  # type: int

        id = ctx.ID().getText()  # type: str
        self.checkCregLimit(id, line, position)
        regSize = int(ctx.INT().getText())
        # create sequence number list for array
        regList = self.genSequenceArray(regSize)
        self.cRegVarMap[id] = regSize
        self.usingCRegList = regList

        # 同时检查qReg, 不能有同名变量
        if id in self.qRegVarMap:
            raise Error.ArgumentError(f'QReg {id} already declared. line: {line}, position: {position}.',
                                      ModuleErrorCode, FileErrorCode, 20)

        return ''

    # 解析操作命令名字
    def getOpName(self, ctx) -> str:
        # line number
        line = ctx.start.line  # type: int
        position = ctx.start.column  # type: int

        opCmd = ctx.ID()  # type: str

        if opCmd:
            opId = opCmd.getText()
        else:
            raise Error.ArgumentError(f'Illegal operation on gate. line: {line}, position: {position}.',
                                      ModuleErrorCode, FileErrorCode, 21)

        return opId

    # 查验转换操作门的类型
    def getGateNameType(self, ctx, opId: str) -> Tuple[str, Union[str, int], bool]:
        # line number
        line = ctx.start.line  # type: int
        position = ctx.start.column  # type: int
        argumentIdField = False
        state = self.getScopeData().state
        if state == QasmParseState.Procedure:
            # 子程序类型需要附加paramIds参数，这里是标记
            argumentIdField = True

        gateIdCheck = opId.upper()
        gateF = FixedGateIdMap.get(gateIdCheck)
        gateR = RotationGateIdMap.get(gateIdCheck)
        gateC = CompositeGateIdMap.get(gateIdCheck)
        gateProcedure = opId in self.procedureVarSet

        if gateF is not None:
            # 固定门
            gateType = 'fixedGate'
            gateName = gateF
        elif gateR is not None:
            # 旋转门
            gateType = 'rotationGate'
            gateName = gateR
        elif gateC is not None:
            # 组合门
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

    # 查看是否需要切换电路容器
    def getCurrentCircuitsContainer(self) -> List[CircuitLine]:
        # 查验解析状态，是否位于子程序解析过程中，需要单独处理
        # 解析结果看需要切换，如果是子程序，转换到procedureMap
        circuitList = self.circuitList
        scopeData = self.getScopeData()
        state = scopeData.state
        procName = scopeData.procName
        if state == QasmParseState.Procedure:
            circuitList = self.procedureMap[procName].circuitList
        return circuitList

    # 操作寄存器没有使用索引的情形，比如: h q;
    # 注意，目前不支持多寄存器操作，子程序中也有要求不能使用索引，所以都是单qubit的操作
    def idListOperation(self, ctx, opId: str, idList, circuitList: List[CircuitLine]):
        # line number
        line = ctx.start.line  # type: int
        position = ctx.start.column  # type: int
        gateType, gateName, argumentIdField = self.getGateNameType(ctx, opId)
        arguments, argumentIdList = self.getArguments(ctx)

        idArguments = idList.ID()
        self.checkUOpArgumentIdList(idArguments, line, position)
        # 经过变量声明、变量重复检验，只能有一个id参数通过
        argument = idArguments[0].getText()  # type: str
        scopeData = self.getScopeData()
        state = scopeData.state
        qReg = scopeData.qReg
        regMap = scopeData.regMap
        procName = scopeData.procName

        # 查验使用的寄存器数量，目前只有单个可以展开，其它的不展开处理，需要等到支持多寄存器操作再进行改进
        idLens = len(idArguments)
        if idLens > 1:
            # 目前这种情况是在子程序内部的指令，调用多个子程序声明的寄存器，所以需要根据使用寄存器的位置来生成列表
            regList = []  # type: List[int]
            for id in idArguments:
                regList.append(regMap[id.getText()])
            # 需要检验参数数量
            if gateType == 'procedureName':
                self.checkProcName(procName, gateName, line, position)
                self.checkProcArgumentsCount(gateName, arguments, line, position)
            # 目前此状态只有子程序内部调用电路指令
            self.createProcCircuit(circuitList, arguments, argumentIdList, argumentIdField, gateType, gateName, regList)
        else:
            # 循环转换; 有的操作需要外部继续附加数据，所以要创建可以返回处理的对象
            # 需要判定是否在子程序状态，子程序状态不牵扯展开问题，单独处理
            if state == QasmParseState.Procedure:
                # 获取使用的寄存器id
                regVal = self.getRegOrderNumber(argument, line, position)
                self.createProcCircuit(circuitList, arguments, argumentIdList, argumentIdField, gateType, gateName,
                                       [regVal])
            else:
                size = qReg.get(argument)
                for i in range(size):
                    # 设置规范格式
                    self.createCircuit(circuitList, arguments, argumentIdList, argumentIdField, gateType, gateName, [i])

    # 操作寄存器使用索引的情形，比如: h q[1];
    def mixListOperation(self, ctx, opId: str, mixList, circuitList):
        # line number
        line = ctx.start.line  # type: int
        position = ctx.start.column  # type: int
        gateType, gateName, argumentIdField = self.getGateNameType(ctx, opId)
        arguments, argumentIdList = self.getArguments(ctx)
        scopeData = self.getScopeData()
        state = scopeData.state

        mixId = mixList.ID()
        mixArgument = mixList.INT()

        if len(mixId) != len(mixArgument):
            raise Error.ArgumentError(
                f'Illegal operation on gate. line: {line}, position: {position}.', ModuleErrorCode, FileErrorCode, 23)

        # 检查多参数的指令操作参数数量
        self.checkArgumentsCount(opId, len(mixId), line, position)

        # 检验变量合法性
        regName = None  # type: str
        for m in mixId:
            mid = m.getText()
            if not regName:
                regName = mid
            self.checkQregVar(mid, line, position)
        # 检验是否对此变量越界访问
        usageIds = []
        for p in mixArgument:
            rIndex = int(p.getText())
            self.checkQregSize(regName, rIndex, line, position)
            usageIds.append(rIndex)
        # 存储数据
        # to pb要区分调用的情况，子程序内部对其它子程序的调用，子程序的直接调用
        # 对这两种调用，需要计算参数值，展开、存储调用的电路，而不设定子程序的map
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
        # 文档已经标注。因为前面禁止多寄存器声明，所以这里的限制要注明
        # 对于单个寄存器参数的操作，允许只声明寄存器，不带具体索引地址，比如: h q;
        # 它的意思是，并行对q包含的所有qubit一起操作，q[5] => h q[0] -> h q[4]
        # 对于多寄存器参数的操作，目前暂时禁止只使用寄存器，因为没有意义。
        # swap q q; 是没有意义的，但是swap q[0], q[1]; 完全合法
        # 等逐渐完善，可以支持多寄存器操作的时候再进行改进，就可以支持：
        # swap q1, q2; 这样的操作了。

        # 说明： 根据OpenQasm的定义，对寄存器的使用有并行的情况，比如 -
        # U(0,0,0) q; 这里的q要根据它的大小，并行操作，并不是只操作第一个。
        # 如果前面定义q[5]，那么上面的操作相当于： U(0,0,0) q[0] -> U(0,0,0) q[5]
        # 其它的情况也是类似的，比如： CX q, r; 同样是并行操作
        # 而且还有部分匹配的情况，一般是两个参数的指令，比如CX q[0], r;
        # 它的含义是： q[0]分别映射操作到r的每一个qubit，根据r的大小执行
        # 所以要根据实际的指令进行判定后，做出转换

        # 常规检验：
        # 指令合法性检验，是否支持
        # 引用的变量合法性检验，名称、是否越界访问

        # line number
        line = ctx.start.line  # type: int
        position = ctx.start.column  # type: int

        # 查验解析状态，是否位于子程序解析过程中，需要单独处理
        # 解析结果看需要切换，如果是子程序，转换到procedureMap
        circuitList = self.getCurrentCircuitsContainer()

        # 操作指令，比如u, cx, swap，ry等等
        # antlr 根据定义编译了多个归属的指令字段，哪个有效就证明是哪个指令
        # 需要注意检查
        opId = self.getOpName(ctx)

        # 查看是否已经有测量指令，测量指令必须放在最后
        self.checkMeasureCommand(line, position)

        anyList = ctx.anylist()
        # may be undefined
        if anyList is not None:
            idList = anyList.idlist()
            mixList = anyList.mixedlist()
            # idlist有效时候，是因为命令行的参数不带索引，所以要判定数量
            if idList is not None:
                self.idListOperation(ctx, opId, idList, circuitList)
            elif mixList is not None:
                # 这是寄存器带索引参数的情形，比如swap q[0], q[1];
                # 暂时不支持多对一，一对多
                self.mixListOperation(ctx, opId, mixList, circuitList)

        return ''

    def visitMeasureOp(self, ctx):
        # 调用格式： measure q -> c;
        # q的尺寸要小于等于c
        # measure q[0] -> c[0];
        line = ctx.start.line  # type: int
        position = ctx.start.column  # type: int

        argument = ctx.argument()
        qReg = argument[0]
        cReg = argument[1]

        qRegId = qReg.ID().getText()
        cRegId = cReg.ID().getText()

        self.checkQregVar(qRegId, line, position)
        self.checkCregVar(cRegId, line, position)

        qRegInt = qReg.INT()
        cRegInt = cReg.INT()

        if qRegInt is None and cRegInt is None:
            # 这是寄存器没有使用索引的情况
            # 循环测量，检查尺寸对应关系，前者尺寸必须小于等于后者尺寸
            self.checkMeasureSize(qRegId, cRegId, line, position)
            # 循环生成测量操作，以第一个寄存器尺寸为准
            qRegSize = self.qRegVarMap.get(qRegId)
            for i in range(qRegSize):
                # 生成转换，存储数据
                self.createMeasureCircuit(self.circuitList, i, i)

            # 标记已经包含了测量指令
            self.containMeasure = True

        elif qRegInt is not None and cRegInt is not None:
            # 这是同时使用寄存器索引的情况
            # 检验是否越界访问
            qInt = int(qRegInt.getText())
            cInt = int(cRegInt.getText())
            self.checkQregSize(qRegId, qInt, line, position)
            self.checkCregSize(cRegId, cInt, line, position)

            # 生成转换，存储数据
            self.createMeasureCircuit(self.circuitList, qInt, cInt)
            # 标记已经包含了测量指令
            self.containMeasure = True

        else:
            # 本操作不能一个使用索引，另一个不使用索引
            raise Error.ArgumentError(f'Illegal measure operation on gate. line: {line}, position: {position}.',
                                      ModuleErrorCode, FileErrorCode, 24)

        return ''

    def visitBarrierOp(self, ctx):
        # 这个命令格式是： barrier q;
        # barrier q[0], ...;

        # line number
        line = ctx.start.line  # type: int
        position = ctx.start.column  # type: int

        anyList = ctx.anylist()

        # 子程序支持拦截器，这里查看是否需要切换
        circuitList = self.getCurrentCircuitsContainer()
        # idlist有效，是因为命令行的参数不带索引
        idList = anyList.idlist()
        mixList = anyList.mixedlist()

        if idList is not None:
            idArguments = idList.ID()
            self.checkBarrierIdList(idArguments, line, position)
            # 经过变量声明、变量重复检验，只能有一个id参数通过
            argument = idArguments[0].getText()
            qReg = self.getScopeData().qReg
            size = qReg.get(argument)

            # 循环转换
            for i in range(size):
                # 设置规范格式
                self.createBarrierCircuit(circuitList, [i])

        # 这是寄存器带索引参数的情形，比如barrier q[0], q[1]
        if mixList is not None:
            mixId = mixList.ID()
            mixArgument = mixList.INT()
            if len(mixId) != len(mixArgument):
                raise Error.ArgumentError(f'Illegal operation on gate. line: {line}, position: {position}.',
                                          ModuleErrorCode, FileErrorCode, 25)

            # 检验变量合法性
            regName = None  # type: str
            for m in mixId:
                mid = m.getText()
                if not regName:
                    regName = mid
                self.checkQregVar(mid, line, position)
            # 检验是否对此变量越界访问
            usageIds = []
            for p in mixArgument:
                rIndex = int(p.getText())
                self.checkQregSize(regName, rIndex, line, position)
                usageIds.append(rIndex)

            # 存储数据
            self.createBarrierCircuit(circuitList, usageIds)
        return ''

    def visitExplist(self, ctx):
        line = ctx.start.line  # type: int
        position = ctx.start.column  # type: int
        scopeData = self.getScopeData()
        argumentIdList = scopeData.argumentIdList
        argumentValueList = scopeData.argumentValueList
        argumentMap = scopeData.argumentMap
        argumentExprList = scopeData.argumentExprList

        expList = []
        exp = ctx.exp()
        for item in exp:
            # 计算表达式的值
            expVal = item.accept(self)
            # 分解，根据情况判定如何设置参数
            obj = json.loads(expVal)
            flag = obj['flag']
            text = obj['text']
            val = obj['val']
            expr = obj['expr']

            if flag == 0:
                # 这里使用了形参，记录索引值
                # 查找索引值
                index = argumentMap[text]
                argumentIdList.append(index)
                # 参数使用字符串0标记
                argumentValueList.append('0')
                expList.append(text)
            else:
                # flag == -1
                # 实参，使用-1标记
                argumentIdList.append(-1)
                # Composer 使用文本值便于观察使用，pb使用转换后的数值计算
                argumentValueList.append(text)
                expList.append(val)
            # 保存antlr分解的计算表达式
            argumentExprList.append(expr)

        # 检测清理，如果不需要则设置为空
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
        # line number
        line = ctx.start.line  # type: int
        position = ctx.start.column  # type: int

        # 需要注意的是，这里既有形参也有实参的可能。在子程序语法中，有形参也有实参
        # 比如u(param0, pi/2, param1);
        # 这种需要区别对待，发现是形参的，直接返回，只有实参才进行计算。
        exprList = []
        # 保留原始的文本信息，因为pi会被转换成大写，而Composer不需要大写
        calcList = []
        # 获取解析出的计算表达式列表，进行计算解析.
        # sin(0.1)
        count = ctx.getChildCount()
        # 记录哪些形参参与了计算
        calcArgumentList = []
        # 记录检查形参参与情况
        checkItem = []  # type: List[bool]
        for i in range(count):
            item = ctx.getChild(i)
            itemText = item.getText()
            # 检查标记
            checkFlag = self.checkArgumentDeclared(itemText)
            checkItem.append(checkFlag)
            if checkFlag:
                calcArgumentList.append(itemText)
            calcList.append(itemText)
            # 对于pi要单独处理，转换成大写才可识别解析
            checkPI = itemText.find('pi')
            if checkPI != -1:
                itemText = itemText.upper()
            exprList.append(itemText)
        # 检查，是否总体参数是形参。形参不需要这里解析计算
        calcText = ''.join(calcList)
        exprText = ''.join(exprList)
        checkParam = self.checkArgumentDeclared(calcText)
        if checkParam:
            # 返回对象并序列化，因为返回值要求是字符串
            return json.dumps({
                'flag': 0,
                'text': calcText,
                'val': calcText,
                'expr': exprList
            })

        # 尝试解析，解析错误会抛出异常
        try:
            # 检查是否有形参在表达式中作为一部分，参与了计算，如果有，需要检查并获取形参的映射值，转换后进行解析
            # 因为代码解析先后顺序的原因，子程序声明的时候，无法确定形参的值，所以必须暂存表达式，等待产生调用的时刻，重新解析
            # 表达式的值，更新电路信息
            ret = True
            for item in checkItem:
                if item != False:
                    ret = False
                    break
            if not ret:
                # 形参作为表达式的一部分有参与计算，暂存表达式，以待后续调用重新扫描，继续解析
                return json.dumps({
                    'flag': -1,
                    'text': exprText,
                    'val': exprText,
                    'expr': exprList
                })

            parser = Parser()
            reVal = parser.parse(exprText).evaluate({})
            # bad expression test
            # Parser.evaluate('pi/2');
        except Exception as ex:
            # 转换错误，抛出自定义的异常
            raise Error.ArgumentError(
                f'Calculation expression {calcText} parsing error: {ex}. line: {line}, position: {position}.',
                ModuleErrorCode, FileErrorCode, 26)
        # 保留原始值，Composer使用便利
        return json.dumps({
            'flag': -1,
            'text': calcText,
            'val': str(reVal),
            'expr': exprList
        })

    def visitGatedecl(self, ctx):
        # 遇到子程序声明，解析状态要进行切换，作用域检测压栈处理，便于建立子程序数据结构
        # line number
        line = ctx.start.line  # type: int
        position = ctx.start.column  # type: int

        gateId = ctx.ID().getText()
        # 查验子程序是否同名
        if gateId in self.procedureVarSet:
            # 重名
            raise Error.ArgumentError(
                f'Duplicated procedure name ${gateId}, line: ${line}, position: ${position}.', ModuleErrorCode,
                FileErrorCode, 27)

        # 需要查验是否包含了函数体，如果没有包含，则没有后续的函数体解析过程
        # 需要通过检验父上下文的内容，必须强制转换
        noBody = False
        parent = ctx.parentCtx

        gopList = parent.goplist()
        if gopList is None:
            noBody = True

        arguments = None
        regs = None
        # 根据qasm语法定义声明，idlist长度如果是2,则第一个是参数列表，第二个是寄存器列表
        # 如果idlist长度是1, 则表明只有寄存器列表，没有参数列表
        contentList = ctx.idlist()
        idListLen = len(contentList)
        if idListLen == 1:
            regs = contentList[0].ID()
        elif idListLen == 2:
            arguments = contentList[0].ID()
            regs = contentList[1].ID()

        # 处理参数，检查并设定存储, 用于子程序的参数检查
        argumentMap = {}
        if arguments:
            for index, argument in enumerate(arguments):
                argumentMap[argument.getText()] = index

        # 使用的寄存器范围列表
        regList = self.genSequenceArray(len(regs))

        # 构建子程序声明的数据结构
        procData = data()
        procData.argumentCount = len(arguments) if arguments else 0
        procData.usingQRegList = regList
        procData.circuitList = []

        # 设置子程序名称记录，保存给后续的分析使用
        self.procedureVarSet.add(gateId)

        # 记录每个子程序使用的参数信息，后期计算转换需要使用
        self.procedureArgumentMap[gateId] = argumentMap
        # 分情况处理，如果没有函数体，则直接加入声明信息返回，否则需要后续的解析处理
        self.procedureMap[gateId] = procData
        if not noBody:
            # 构建寄存器和顺序的结构，因为有需要使用它的数字标记
            regMap = {}
            # 设置子程序范围内的变量检查记录
            procVarMap = {}
            for index, reg in enumerate(regs):
                procVarMap[reg.getText()] = 1
                regMap[reg.getText()] = index

            # 构建数据栈
            procStack = ScopeLevel(
                qReg=procVarMap,
                cReg={},
                state=QasmParseState.Procedure,
                regMap=regMap,
                argumentMap=argumentMap,
                # 这几项，用于精确处理子程序的参数使用
                argumentExprList=[],
                argumentIdList=[],
                argumentValueList=[],
                # 通过此关联信息，获取必须的数据
                procName=gateId,
                circuitCounter=0,
            )
            self.scopeStack.append(procStack)

        return ''

    def visitGoplist(self, ctx):
        # 这是子程序的内部声明部分，此时解析状态已经切换到子程序模式，
        # 解析完毕要切换回来
        self.visitChildren(ctx)
        # 解析状态恢复
        self.scopeStack.pop()

        return ''

    def complete(self):
        head = self.program.head
        head.usingQRegList[:] = self.usingQRegList
        head.usingCRegList[:] = self.usingCRegList
        body = self.program.body
        for circuit in self.circuitList:
            body.circuit.append(circuit.pbCircuitLine)
        # 因为电路必须展开，所以移除子程序
        # for procName, proc in self.procedureMap.items():
        #     procedure = body.procedureMap[procName]
        #     procedure.parameterCount = proc.argumentCount
        #     procedure.usingQRegList[:] = proc.usingQRegList
        #     for circuit in proc.circuitList:
        #         procedure.circuit.append(circuit.pbCircuitLine)
