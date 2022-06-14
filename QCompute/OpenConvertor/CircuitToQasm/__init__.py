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
import re
from typing import List, Optional, Tuple, Set, Dict, Any

from QCompute.OpenConvertor import ModuleErrorCode, ConvertorImplement
from QCompute.QPlatform import Error
from QCompute.QProtobuf import PBProgram, PBFixedGate, PBRotationGate, PBCompositeGate

FileErrorCode = 4


class CircuitToQasm(ConvertorImplement):
    """
    Circuit To Qasm
    """

    def __init__(self):
        self.containMeasure = False
        # 子程序代码排序，分析调用关系使用
        self.procedureNameList = []
        self.proceduresCode = {}
        self.proceduresDepends = {}

    # 进行形参、实参的分析处理，组合使用
    def getArgumentList(self, params: List[str], paramIds: List[int]) -> List:
        paramCall = []
        # 查看paramIds的值，是否有-1的情况，如果有需要寻找对应的paramValues内容
        checkParams = params and len(params)
        checkParamIds = paramIds and len(paramIds)

        if checkParams and checkParamIds:
            for i in range(len(paramIds)):
                paramId = paramIds[i]
                if paramId != -1:
                    # 形参
                    paramItem = f'param{paramId}'
                    paramCall.append(paramItem)
                else:
                    # 实参，使用实际数值参数
                    paramItem = params[i]
                    paramCall.append(paramItem)
        else:
            # 只有单一参数有效，或者同时为空
            # 全部使用形参，param0 -> paramx
            if checkParamIds:
                for i in range(len(paramIds)):
                    paramId = paramIds[i]
                    paramItem = f'param{paramId}'
                    paramCall.append(paramItem)
            # 全部使用实参
            if checkParams:
                paramCall = params

        return paramCall

    continueZeroRe = re.compile('0+$')

    # 小数点修饰
    def getFixedFloatNumber(self, realNum: float) -> str:
        paramFixed = format(realNum, '.7f')
        # 剔除连续的0，不能从.开始判断，因为可能有0.010000这种情况
        paramFixed = self.continueZeroRe.sub('', paramFixed)
        # 判断最后的字符，如果是.，则需要补一个0，符合阅读习惯
        charCheck = paramFixed[-1]
        if charCheck == '.':
            paramFixed += '0'
        return paramFixed

    def getFixedArgumentList(self, argumentValueList: List[float]) -> List[str]:
        # 查看是否有参数
        params = []
        for param in argumentValueList:
            paramFixed = self.getFixedFloatNumber(param)
            params.append(paramFixed)
        return params

    def getTrimmedArgumentList(self, argumentValueList: List[float]) -> List[str]:
        # 先进行转换，空值转换为0,其它有效值不动
        convertedParams = [self.getFixedFloatNumber(p) for p in argumentValueList]
        return convertedParams

    # 测量指令，这里要使用传统寄存器
    def getMeasureCommandCode(self, measure, qRegs: List[int], regName: str) -> str:
        command = ''
        # 转换使用的寄存器
        # 要支持多参数紧凑测量，measure.cRegList size为准
        for i in range(len(measure.cRegList)):
            # 第一个是量子寄存器，第二个是传统寄存器
            cr = measure.cRegList[i]
            qr = qRegs[i]
            command += f'measure {regName}[{qr}] -> c[{cr}];\n'
        return command

    # usingIndex的含义 - 子程序中不能使用索引，所以命令要去掉索引的表示[]
    # 分解circuit内门的记录，转换成命令文本
    # 固定门
    def getFixedCommandCode(self, fixedGate: int, regs: List[int], regName: str, usingIndex: bool = True) -> str:
        # 寻找映射的名称
        gateName = PBFixedGate.Name(fixedGate)
        # 转换使用的寄存器
        if usingIndex:
            regOp = [f'{regName}[{r}]' for r in regs]
            command = ''
        else:
            regOp = [f'{regName}{r}' for r in regs]
            # 子程序调用，命令加缩进
            command = '  '
        command += f'{gateName} {", ".join(regOp)};\n'
        return command

    # usingIndex的含义 - 子程序中不能使用索引，所以命令要去掉索引的表示[]
    # 旋转门, 注意， paramValues/paramIds可能为空
    def getRotationCommandCode(self, rotationGate: int, regs: List[int], regName: str, usingIndex: bool = True,
                               paramValues: Optional[List[float]] = None, paramIds: Optional = None) -> str:
        # 寻找映射的名称
        gateName = PBRotationGate.Name(rotationGate)

        # 需要区分情况处理
        # 查看是否有参数
        params = self.getFixedArgumentList(paramValues)
        # 参数分析转换
        paramCall = self.getArgumentList(params, paramIds)

        # 转换使用的寄存器
        if usingIndex:
            regOp = [f'{regName}[{r}]' for r in regs]
            command = ''
        else:
            regOp = [f'{regName}{r}' for r in regs]
            # 子程序调用，命令加缩进
            command = '  '

        command += f'{gateName} ({", ".join(paramCall)}) {", ".join(regOp)};\n'

        return command

    # 拦截器指令
    def getBarrierCommandCode(self, regs: List[int], regName: str, usingIndex: bool = True) -> str:
        # 转换使用的寄存器
        if usingIndex:
            regOp = [f'{regName}[{r}]' for r in regs]
            command = ''
        else:
            regOp = [f'{regName}{r}' for r in regs]
            # 子程序调用，命令加缩进
            command = '  '
        command += f'barrier {", ".join(regOp)};\n'
        return command

    # 转换子程序的调用指令， 参数说明：
    # procedureName - 调用的子程序名字； regs - 引用的寄存器列表，
    # regName - 输出使用的基础名字，典型值是q，但是解析子程序的时候，会用到其它的名字
    # usingIndex - 是否使用寄存器的索引，子程序内指令不能使用索引
    # paramValues - 调用子程序的参数值
    def getProcCommandCode(self, procedureName: str, regs: List[int], regName: str, usingIndex: bool = True,
                           paramValues: Optional[List[float]] = None, paramIds: Optional = None) -> str:
        # 先进行转换，空值转换为0,其它有效值不动
        convertedParams = self.getTrimmedArgumentList(paramValues)
        # 参数分析转换
        paramCall = self.getArgumentList(convertedParams, paramIds)
        # 连接
        if len(paramCall) > 0:
            params = f'({", ".join(paramCall)})'
        else:
            params = ''
        # 调用格式例子： cu1(pi/2) q[0],q[1];
        if usingIndex:
            regOp = [f'{regName}[{r}]' for r in regs]
            command = ''
        else:
            regOp = [f'{regName}{r}' for r in regs]
            # 子程序调用，命令加缩进
            command = '  '
        command += f'{procedureName}{params} {", ".join(regOp)};\n'
        return command

    # 组合门的处理
    def getCompositeCommandCode(self, compositeGate: int, regs: List[int], regName: str, usingIndex: bool = True,
                                paramValues: Optional[List[float]] = None, paramIds: Optional = None) -> str:
        # 寻找映射的名称
        gateName = PBCompositeGate.Name(compositeGate)
        # 查看是否有参数
        params = self.getFixedArgumentList(paramValues)
        # 参数分析转换
        paramCall = self.getArgumentList(params, paramIds)

        # 转换使用的寄存器
        if usingIndex:
            regOp = [f'{regName}[{r}]' for r in regs]
            command = ''
        else:
            regOp = [f'{regName}{r}' for r in regs]
            # 子程序调用，命令加缩进
            command = '  '
        command += f'{gateName} ({", ".join(paramCall)}) {", ".join(regOp)};\n'
        return command

    # 处理电路映射，circuit作为对象，如何在ts标记？
    # regName - 使用的寄存器名称，因为子程序使用的寄存器名字与外部程序需要分开
    def getCircuitsCode(self, circuit: List, regName: str, usingIndex: bool = True) -> Tuple[str, Set[str]]:
        # 循环处理电路标记
        # 需要标记，是否有测量指令，此外，测量指令自动放在后面
        # 查找记录依赖项
        qasmCode = ''
        measureCode = ''
        depends = set()  # type: Set[str]
        for gate in circuit:
            # 判定操作类型，目前支持的几类，后续要支持子程序
            # 不同的操作, 需要反向查询具体类型信息
            op = gate.WhichOneof('op')
            if op == 'fixedGate':
                qasmCode += self.getFixedCommandCode(gate.fixedGate, gate.qRegList, regName, usingIndex)
            elif op == 'rotationGate':
                qasmCode += self.getRotationCommandCode(gate.rotationGate, gate.qRegList, regName, usingIndex,
                                                        gate.argumentValueList, gate.argumentIdList)
            elif op == 'compositeGate':
                # 组合门
                qasmCode += self.getCompositeCommandCode(gate.compositeGate, gate.qRegList, regName, usingIndex,
                                                         gate.argumentValueList, gate.argumentIdList)
            elif op == 'procedureName':
                # 子程序
                qasmCode += self.getProcCommandCode(gate.procedureName, gate.qRegList, regName, usingIndex,
                                                    gate.argumentValueList, gate.argumentIdList)
                # 记录依赖的子程序名称
                if gate.procedureName in depends:
                    depends.add(gate.procedureName)
            elif op == 'barrier':
                qasmCode += self.getBarrierCommandCode(gate.qRegList, regName, usingIndex)
            elif op == 'measure':
                # 测量指令单独处理
                measureCode += self.getMeasureCommandCode(gate.measure, gate.qRegList, regName)
            else:
                raise Error.ArgumentError(f'Invalid gate operation: {gate}', ModuleErrorCode, FileErrorCode, 1)

        if measureCode is not None:
            qasmCode += measureCode
            self.containMeasure = True

        return qasmCode, depends

    # 处理子程序的定义，子程序的结构定义，可以参考测试用例，比较全面
    # 格式说明：
    # "nG1": {
    #         "paramCount": 2,
    #         "usingQRegs": [
    #         ],
    #         "circuit": [
    #           {
    #             "fixedGate": 5,
    #             "qRegs": []
    #           },
    #         ]
    #       }
    def getProcedureCode(self, procedureMaps: Dict[str, Any]) -> str:
        procCode = ''
        for name, content in procedureMaps.items():
            # key 是子程序的名字
            # val 是子程序的设定内容对象
            gateDefine = self.getProcDefineCode(name, content)
            # 需要对子程序的声明顺序做标记，确保声明在前，使用在后
            self.procedureNameList.append(name)
            self.proceduresCode[name] = gateDefine
        # 根据依赖信息表，重排设定子程序的代码声明顺序，确保正确
        # 算法使用过的子程序信息
        usedProcedure = []  # List[str]
        # 处理代码依赖
        for name in self.procedureNameList:
            # 循环检测依赖信息
            depends = self.proceduresDepends.get(name)
            # 如果有依赖，先使用依赖的代码
            if depends is not None and len(depends) > 0:
                for item in depends:
                    # 查看是否已经使用过
                    if item in usedProcedure:
                        # 已经使用过
                        continue
                    depCode = self.proceduresCode.get(item)
                    if depCode is not None:
                        # 依赖代码放在最前面
                        depCode += procCode
                        # 重设
                        procCode = depCode
                        # 设置使用标志，避免循环依赖、死循环
                        usedProcedure.append(item)
                    else:
                        # 依赖代码没有找到，报告错误
                        raise Error.ArgumentError(f'Invalid procedure name: {item}', ModuleErrorCode, FileErrorCode, 2)
            # 查看是否已经使用过
            if name in usedProcedure:
                # 已经使用过
                continue
            # 代码计入
            procCode += self.proceduresCode.get(name)
            usedProcedure.append(name)

        return procCode

    # 获取单个子程序的定义代码
    def getProcDefineCode(self, name: str, content) -> str:
        # {parameterCount, usingQRegList, circuit} = content

        # 分解三个参数；parameterCount 是定义了几个参数，1-3个，一般对应是：theta,phi,lambda，
        # 但是我们这里先用param0 -> paramX 这样的命名
        # 更多的使用其它的参数名字
        # usingQRegList是使用了几个Qubit，用qb0-qbn表达
        # circuit 是具体的操作指令
        if content.parameterCount > 0:
            paramsArray = [f'param{i}' for i in range(content.parameterCount)]
            paramsDef = f'({", ".join(paramsArray)})'
        else:
            paramsDef = ''

        # 寄存器
        qRegList = [f'qb{r}' for r in content.usingQRegList]
        qRegs = ', '.join(qRegList)

        # 循环处理内部定义的电路指令 - 分解
        # 判定操作类型，不支持测量指令
        if content.circuit is not None:
            circuitCode, depends = self.getCircuitsCode(content.circuit, 'qb', False)
            # 构建依赖信息表
            self.proceduresDepends[name] = depends
        else:
            circuitCode = ''
        # gate cu1(lambda) a,b
        # {
        # U(0,0,theta/2) a
        # CX a,b
        # U(0,0,-theta/2) b
        # CX a,b
        # U(0,0,theta/2) b
        # }

        # 输出声明
        gateDefine = f'gate {name}{paramsDef} {qRegs}\n{{\n{circuitCode}}}\n'

        return gateDefine

    # 循环转换处理
    def convert(self, program: PBProgram) -> str:
        qasmCode = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'

        # 如果已经有了source，则直接返回
        if program.source.qasm != '':
            return program.source.qasm
        # usingQRegList, usingCRegList 是数组，最大值是使用的索引值，因为从0开始，需要+1
        maxQregSize = max(program.head.usingQRegList) + 1
        maxCregSize = max(program.head.usingCRegList) + 1

        qasmCode += f'qreg q[{maxQregSize}];\n'
        qasmCode += f'creg c[{maxCregSize}];\n'

        # 处理子程序声明部分，需要处理额外的子程序调用关系，确保先声明，后调用
        if program.body.procedureMap:
            procedureCode = self.getProcedureCode(program.body.procedureMap)
            qasmCode += procedureCode

        # 循环处理电路标记
        circuitCode, depends = self.getCircuitsCode(program.body.circuit, 'q')
        qasmCode += circuitCode

        return qasmCode
