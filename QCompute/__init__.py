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
Export the entire directory as a library
"""

from QCompute.Define import (
    sdkVersion,
    outputPath,
)

from QCompute.Define.Settings import outputInfo

from QCompute.OpenModule.CompositeGateModule import CompositeGate
from QCompute.OpenModule.UnrollCircuitModule import UnrollCircuit
from QCompute.OpenModule.UnrollProcedureModule import UnrollProcedure

from QCompute.OpenSimulator import (
    QuantumResult,
)

from QCompute.QuantumPlatform import (
    QuantumTask,
    BackendName,
    Sim2Param,
)
from QCompute.QuantumPlatform.QuantumEnvironment import QuantumEnvironment

from QCompute.QuantumPlatform.QuantumOperation.Barrier import Barrier
from QCompute.QuantumPlatform.QuantumOperation.CompositeGate import RZZ
from QCompute.QuantumPlatform.QuantumOperation.CustomizedGate import CustomizedGate
from QCompute.QuantumPlatform.QuantumOperation.FixedGate import (
    ID,
    X,
    Y,
    Z,
    H,
    S,
    SDG,
    T,
    TDG,
    CX,
    CY,
    CZ,
    CH,
    SWAP,
    CCX,
    CSWAP,
)
from QCompute.QuantumPlatform.QuantumOperation.Measure import (
    MeasureZ
)
from QCompute.QuantumPlatform.QuantumOperation.RotationGate import (
    U,
    RX,
    RY,
    RZ,
    CU,
    CRX,
    CRY,
    CRZ,
)
import QCompute.QuantumPlatform.Utilities
from QCompute.Test.PostInstall.PostInstall import testAll

