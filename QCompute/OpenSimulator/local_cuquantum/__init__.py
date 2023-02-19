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
To adapt cuQuantum to PySDK
"""
import platform
import subprocess
from enum import Enum
from pathlib import Path
from sys import executable
from typing import List

from QCompute import Define
from QCompute.Define import Settings
from QCompute.Define.Utils import filterConsoleOutput, findUniError
from QCompute.OpenConvertor.CircuitToJson import CircuitToJson
from QCompute.OpenSimulator import QImplement, ModuleErrorCode
from QCompute.QPlatform import Error
from QCompute.OpenSimulator.local_cuquantum.CheckEnv import nvidia_gpu_driver_installed, cuda_11_installed, openmpi_installed
from QCompute.OpenSimulator.local_cuquantum.Simulator import runSimulator

FileErrorCode = 1


class Backend(QImplement):
    """
    Call the local Baidu Simulator.

    Note of the data field is in base class QImplement.
    """

    def commit(self) -> None:
        """
        Commit the circuit to local cuquantum simulator.

        Param of above is used in env.backend()

        Example:

        env = QEnv()

        env.backend(BackendName.LocalCuquantum)

        Can be self.runInProcess()  # in this process

        Or self.runOutProcess()  # in another process
        """

        if platform.system() != 'Linux':
            raise Error.RuntimeError('cuQuantum needs to run in Linux!\n'
                                     'Ubuntu 18-22 is suggested.',
                                     ModuleErrorCode, FileErrorCode, 1)

        if not nvidia_gpu_driver_installed():
            raise Error.RuntimeError('NVIDIA GPU dirver is not installed!\n'
                                     'Make sure you have a mainstream NVIDIA GPU first.\n'
                                     'Then, try to install the driver as well as CUDA Toolkit 11.\n'
                                     'Please see https://developer.nvidia.com/cuda-11-8-0-download-archive\n'
                                     'The runfile(local) installer is suggested.\n'
                                     'Or, try to install the driver by "Software & Updates" tools -> '
                                     '"Additional Drivers" tab when you use a mainstream ubuntu operating system.\n'
                                     'Please confirm the compatibility among the GPU, OS, driver, cuda versions.',
                                     ModuleErrorCode, FileErrorCode, 2)

        if not cuda_11_installed():
            raise Error.RuntimeError('CUDA 11 is not installed!\n'
                                     'Make sure you have a mainstream NVIDIA GPU first.\n'
                                     'Then, try to install the driver as well as CUDA Toolkit 11.\n'
                                     'Please see https://developer.nvidia.com/cuda-11-8-0-download-archive\n'
                                     'The runfile(local) installer is suggested.\n'
                                     'Please confirm the compatibility among the GPU, OS, driver, cuda versions.',
                                     ModuleErrorCode, FileErrorCode, 3)

        if not openmpi_installed():
            raise Error.RuntimeError('libopenmpi is not installed!\n'
                                     'Please install it.\n'
                                     'If you are use ubuntu 18-22, you can try:\n'
                                     'sudo apt install libopenmpi-dev.',
                                     ModuleErrorCode, FileErrorCode, 4)

        try:
            import mpi4py
            import cupy
            import cuquantum
        except ImportError:
            raise Error.RuntimeError('cuquantum or its dependencies cannot be satisfied!\n'
                                     'Please use command `pip install -U mpi4py==3.1.4 cupy-cuda11x cuquantum-python-cu11==22.11.0.1`\n'
                                     'If using conda, please use this command before `conda install gcc_linux-64\n'
                                     'For more information, visit also https://docs.nvidia.com/cuda/cuquantum/python/README.html#installation',
                                     ModuleErrorCode, FileErrorCode, 5)

        if len(self.program.head.usingQRegList) > 32:
            raise Error.RuntimeError('The dimension of ndarray does not support more than 32 qubits! '
                                     f'Currently, QReg in the program counts {self.program.head.usingQRegList}.',
                                     ModuleErrorCode, FileErrorCode, 6)

        if Settings.inProcessSimulator:
            self.runInProcess()
        else:
            self.runOutProcess()

    def runInProcess(self):
        """
        Executed in the process (for debugging purpose)
        """

        self.result = runSimulator(self._makeArguments(), self.program)
        self.result.output = self.result.toJson()

        if Settings.outputInfo:
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
        programFilePath = Define.outputDirPath / 'program.json'
        if Settings.outputInfo:
            print('Program file:', programFilePath)  # print the output filename
        programFilePath.write_text(jsonStr, encoding='utf-8')

        cmd = (executable,) + (str(Path(__file__).parent / 'Simulator.py'),) + tuple(self._makeArguments()) + (
            '-inputFile', programFilePath)
        if Settings.outputInfo:
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
                                 f'QC.3.{ModuleErrorCode}.{FileErrorCode}.7'
            return

        countsFilePath = Define.outputDirPath / 'counts.json'
        if Settings.outputInfo:
            print('Counts file:', countsFilePath)  # print the input filename
        text = countsFilePath.read_text(encoding='utf-8')

        self.result.fromJson(text)
        self.result.output = self.result.toJson()

        if Settings.outputInfo:
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
