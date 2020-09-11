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
import subprocess
import tempfile
from enum import Enum
from os import path, fdopen
from sys import executable

from google.protobuf.json_format import MessageToJson

from QCompute.Define.Settings import outputInfo, inProcessSimulator
from QCompute.Define.Utils import _filterConsoleOutput
from QCompute.OpenSimulator import QuantumImplement
from QCompute.OpenSimulator.local_baidu_sim2.Simulator import runSimulator


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

        if inProcessSimulator:
            self.runInProcess()
        else:
            self.runOutProcess()

    def runInProcess(self):
        """
        Executed in the process (for debugging purpose)
        """

        self.result = runSimulator(self.makeParams(), self.program)
        self.result.output = json.dumps({
            'shots': self.result.shots,
            'counts': self.result.counts,
            'seed': self.result.seed,
            'startTimeUtc': self.result.startTimeUtc,
            'endTimeUtc': self.result.endTimeUtc,
        })

        if outputInfo:
            print('Shots', self.result.shots)
            print('Counts', self.result.counts)
            print('Seed', self.result.seed)

    def runOutProcess(self):
        """
        Executed separately
        """

        jsonStr = MessageToJson(self.program, preserving_proto_field_name=True)

        # write the qiskit qobj to a temporary json file
        tmpFd, tmpFn = tempfile.mkstemp(suffix=".json")
        with fdopen(tmpFd, "wt") as fObj:
            fObj.write(jsonStr)
        if outputInfo:
            print('json file:', tmpFn)  # print the json filename

        cmd = (executable,) + (path.join(path.dirname(__file__), 'Simulator.py'),) + tuple(self.makeParams()) + (
            '-inputFile', tmpFn)
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

        self.result.output = _filterConsoleOutput(completedProcess.stdout)
        result = json.loads(self.result.output)
        self.result.shots = result['shots']
        self.result.counts = result['counts']
        self.result.seed = result['seed']
        self.result.startTimeUtc = result['startTimeUtc']
        self.result.endTimeUtc = result['endTimeUtc']

        if outputInfo:
            print('Shots', self.result.shots)
            print('Counts', self.result.counts)
            print('Seed', self.result.seed)

    def makeParams(self):
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
