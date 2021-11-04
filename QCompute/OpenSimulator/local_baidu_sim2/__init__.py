#!/usr/bin/python3
# -*- coding: utf8 -*-

# Copyright (c) 2020 Baidu, Inc. All Rights Reserved.
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
To adapt Sim2 to PySDK
"""
import subprocess
from enum import Enum
from pathlib import Path
from sys import executable
from typing import List

from QCompute.Define import outputPath
from QCompute.Define.Settings import outputInfo, inProcessSimulator
from QCompute.Define.Utils import filterConsoleOutput, findUniError
from QCompute.OpenConvertor.CircuitToJson import CircuitToJson
from QCompute.OpenSimulator import QImplement, ModuleErrorCode
from QCompute.OpenSimulator.local_baidu_sim2.Simulator import runSimulator
from QCompute.QPlatform import Error

FileErrorCode = 1


class Backend(QImplement):
    """
    Call the local Baidu Simulator.

    Note of the data field is in base class QImplement.
    """

    def commit(self) -> None:
        """
        Commit the circuit to local baidu simulator.

        The combinations are not 2**3=8 cases. We only implement several combination checks:

        1)DENSE-EINSUM-SINGLE

        2)DENSE-EINSUM-PROB

        3)DENSE-MATMUL-SINGLE

        4)DENSE-MATMUL-PROB

        

        Param of above is used in env.backend()

        Example:

        env = QEnv()

        env.backend(BackendName.LocalBaiduSim2, Sim2Param.Dense_Matmul_Probability)

        Can be self.runInProcess()  # in this process

        Or self.runOutProcess()  # in another process
        """

        if len(self.program.head.usingQRegList) > 32:
            raise Error.RuntimeError('The dimension of ndarray does not support more than 32 qubits! '
                                     f'Currently, QReg in the program counts {self.program.head.usingQRegList}.',
                                     ModuleErrorCode, FileErrorCode, 1)

        if inProcessSimulator:
            self.runInProcess()
        else:
            self.runOutProcess()

    def runInProcess(self):
        """
        Executed in the process (for debugging purpose)
        """

        self.result = runSimulator(self._makeArguments(), self.program)
        self.result.output = self.result.toJson()

        if outputInfo:
            print('Shots', self.result.shots)
            print('Counts', self.result.counts)
            print('State', self.result.state)
            print('Seed', self.result.seed)

    def runOutProcess(self):
        """
        Executed separately
        """

        jsonStr = CircuitToJson().convert(self.program)

        # write the circuit to a temporary json file
        programFilePath = outputPath / 'program.json'
        if outputInfo:
            print('Program file:', programFilePath)  # print the output filename
        with open(programFilePath, 'wt', encoding='utf-8') as file:
            file.write(jsonStr)

        cmd = (executable,) + (str(Path(__file__).parent / 'Simulator.py'),) + tuple(self._makeArguments()) + (
            '-inputFile', programFilePath)
        if outputInfo:
            print(f'{cmd}')

        # call the simulator
        completedProcess = subprocess.run(
            # Compatible with Python3.6, in Python3.7 and above, the more understandable alias of 'universal_newlines' is 'text'.
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, encoding='utf-8')

        self.result.log = filterConsoleOutput(completedProcess.stdout)
        # collect the result to simulator for the subsequent invoking
        self.result.code = completedProcess.returncode
        if self.result.code != 0:
            self.result.vendor = findUniError(completedProcess.stdout) or \
                                    f'QC.3.{ModuleErrorCode}.{FileErrorCode}.2'
            return

        countsFilePath = outputPath / 'counts.json'
        if outputInfo:
            print('Counts file:', countsFilePath)  # print the input filename
        with open(countsFilePath, 'rt', encoding='utf-8') as file:
            text = file.read()

        self.result.fromJson(text)
        self.result.output = self.result.toJson()

        if outputInfo:
            print('UsedQRegList', self.result.ancilla.usedQRegList)
            print('UsedCRegList', self.result.ancilla.usedCRegList)
            print('CompactedQRegDict', self.result.ancilla.compactedQRegDict)
            print('CompactedCRegDict', self.result.ancilla.compactedCRegDict)
            print('Shots', self.result.shots)
            print('Counts', self.result.counts)
            print('Seed', self.result.seed)

    def _makeArguments(self) -> List[str]:
        """
        Generate arguments
        """

        if self.backendArgument and len(self.backendArgument) > 0:
            if isinstance(self.backendArgument[0], Enum):
                args = self.backendArgument[0].value
            else:
                args = self.backendArgument[0]
            return (f'{args} -shots {self.shots}').split()
        else:
            return f'-shots {self.shots}'.split()
