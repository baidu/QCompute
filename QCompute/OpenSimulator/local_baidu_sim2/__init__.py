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

import json
import os
import subprocess
from enum import Enum
from os import path
from sys import executable

from google.protobuf.json_format import MessageToJson

from QCompute import outputPath
from QCompute.Define.Settings import outputInfo, inProcessSimulator
from QCompute.Define.Utils import _filterConsoleOutput
from QCompute.OpenSimulator import QuantumImplement
from QCompute.OpenSimulator.local_baidu_sim2.Simulator import runSimulator
from QCompute.QuantumPlatform import Error


class Backend(QuantumImplement):
    """
    Call the local Baidu Simulator.

    Note of the data field is in base class QuantumImplement.
    """

    def commit(self):
        """
        Commit the circuit to local baidu simulator.

        The combinations are not 2**3=8 cases. We only implement several combination checks:

        1)DENSE-EINSUM-SINGLE

        2)DENSE-EINSUM-PROB

        3)DENSE-MATMUL-SINGLE

        4)DENSE-MATMUL-PROB

        

        Param of above is used in env.backend()

        Example:

        env = QuantumEnvironment()

        env.backend(BackendName.LocalBaiduSim2, Sim2Param.Dense_Matmul_Probability)

        Can be self.runInProcess()  # in this process

        Or self.runOutProcess()  # in another process
        """

        if len(self.program.head.usingQRegs) > 32:
            raise Error.RuntimeError('The dimension of ndarray does not support more than 32 qubits. '
                                     f'Currently, QReg in the program counts {self.program.head.usingQRegs}.')

        if inProcessSimulator:
            self.runInProcess()
        else:
            self.runOutProcess()

    def runInProcess(self):
        """
        Executed in the process (for debugging purpose)
        """

        self.result = runSimulator(self._makeParams(), self.program)

        if outputInfo:
            print('Shots', self.result.shots)
            print('Counts', self.result.counts)
            print('Seed', self.result.seed)

    def runOutProcess(self):
        """
        Executed separately
        """

        jsonStr = MessageToJson(self.program, preserving_proto_field_name=True)

        # write the circuit to a temporary json file
        programFilePath = os.path.join(outputPath, 'program.json')
        if outputInfo:
            print('program file:', programFilePath)  # print the output filename
        with open(programFilePath, 'wt', encoding='utf-8') as file:
            file.write(jsonStr)

        cmd = (executable,) + (path.join(path.dirname(__file__), 'Simulator.py'),) + tuple(self._makeParams()) + (
            '-inputFile', programFilePath)
        if outputInfo:
            print(f"{cmd}")

        # call the simulator
        completedProcess = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # collect the result to simulator for the subsequent invoking
        self.result.code = completedProcess.returncode
        if self.result.code != 0:
            self.result.output = completedProcess.stderr
            print(self.result.output)
            return

        countsFilePath = os.path.join(outputPath, 'counts.json')
        if outputInfo:
            print('counts file:', countsFilePath)  # print the input filename
        with open(countsFilePath, 'rt', encoding='utf-8') as file:
            text = file.read()

        self.result.fromJson(text)

        if outputInfo:
            print('Shots', self.result.shots)
            print('Counts', self.result.counts)
            print('Seed', self.result.seed)

    def _makeParams(self):
        """
        Generate params
        """

        if len(self.backendParam) > 0:
            param = ''
            if isinstance(self.backendParam[0], Enum):
                param = self.backendParam[0].value
            else:
                param = self.backendParam[0]
            return (f'{param} -shots {self.shots}').split()
        else:
            return f'-shots {self.shots}'.split()
