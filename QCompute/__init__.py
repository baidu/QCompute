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
Export the entire directory as a library
"""

from QCompute import Define
from QCompute.Define import (
    sdkVersion,
    outputDirPath,
)
from QCompute.Define.Utils import matchSdkVersion
from QCompute.Module.MappingToBaiduQPUQianModule import MappingToBaiduQPUQianModule
from QCompute.Module.UnrollCircuitToBaiduQPUQianModule import UnrollCircuitToBaiduQPUQianModule
from QCompute.Module.MappingToIoPCASModule import MappingToIoPCASModule
from QCompute.Module.UnrollCircuitToIoPCASModule import UnrollCircuitToIoPCASModule
from QCompute.Module.MappingToIonAPMModule import MappingToIonAPMModule
from QCompute.Module.UnrollCircuitToIonAPMModule import UnrollCircuitToIonAPMModule
from QCompute.OpenModule.CompositeGateModule import CompositeGateModule
from QCompute.OpenModule.CompressGateModule import CompressGateModule
from QCompute.OpenModule.InverseCircuitModule import InverseCircuitModule
from QCompute.OpenModule.ReverseCircuitModule import ReverseCircuitModule
from QCompute.OpenModule.UnrollCircuitModule import UnrollCircuitModule
from QCompute.OpenModule.UnrollProcedureModule import UnrollProcedureModule
from QCompute.OpenModule.UnrollNoiseModule import UnrollNoiseModule
from QCompute.OpenModule.CompressNoiseModule import CompressNoiseModule



from QCompute.OpenSimulator import (
    QResult,
)

from QCompute.QPlatform import (
    QTask,
    BackendName,
    Sim2Argument,
    ServerModule,
)
from QCompute.QPlatform.QEnv import QEnv, concatQEnv
from QCompute.QPlatform.QEnvOperation import QEnvOperation
from QCompute.QPlatform.InteractiveModule import InteractiveModule

from QCompute.QPlatform.QOperation.Barrier import Barrier, BarrierOP
from QCompute.QPlatform.QOperation.CompositeGate import MS, CK
from QCompute.QPlatform.QOperation.CustomizedGate import CustomizedGateOP
from QCompute.QPlatform.QOperation.FixedGate import (
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
from QCompute.QPlatform.QOperation.Measure import (
    MeasureZ
)
from QCompute.QPlatform.QOperation.RotationGate import (
    U,
    RX,
    RY,
    RZ,
    CU,
    CRX,
    CRY,
    CRZ,
)
from QCompute.QPlatform.QOperation.Photonic.PhotonicGaussianGate import (
    PhotonicDX as GaussianDX,
    PhotonicDP as GaussianDP,
    PhotonicPHA as GaussianPHA,
    PhotonicBS as GaussianBS,
    PhotonicCZ as GaussianCZ,
    PhotonicCX as GaussianCX,
    PhotonicDIS as GaussianDIS,
    PhotonicSQU as GaussianSQU,
    PhotonicTSQU as GaussianTSQU,
    PhotonicMZ as GaussianMZ,  # CompositeGate
)
from QCompute.QPlatform.QOperation.Photonic.PhotonicGaussianMeasure import (
    MeasureHomodyne as GaussianMeasureHomodyne,
    MeasureHeterodyne as GaussianMeasureHeterodyne,
    MeasurePhotonCount as GaussianMeasurePhotonCount,
)
from QCompute.QPlatform.QOperation.Photonic.PhotonicFockGate import (
    PhotonicAP as FockAP,
    PhotonicPHA as FockPHA,
    PhotonicBS as FockBS,
    PhotonicMZ as FockMZ,  # CompositeGate
)
from QCompute.QPlatform.QOperation.Photonic.PhotonicFockMeasure import (
    MeasurePhotonCount as FockMeasurePhotonCount,
)
from QCompute.OpenConvertor.CircuitToDrawConsole import (
    CircuitToDrawConsole
)
from QCompute.OpenConvertor.CircuitToJson import (
    CircuitToJson
)
from QCompute.OpenConvertor.JsonToCircuit import (
    JsonToCircuit
)
from QCompute.OpenConvertor.CircuitToInternalStruct import (
    CircuitToInternalStruct
)
from QCompute.OpenConvertor.InternalStructToCircuit import (
    InternalStructToCircuit
)
from QCompute.OpenConvertor.CircuitToQasm import (
    CircuitToQasm
)
from QCompute.OpenConvertor.QasmToCircuit import (
    QasmToCircuit
)
from QCompute.OpenConvertor.CircuitToIonQ import (
    CircuitToIonQ
)
from QCompute.OpenConvertor.IonQToCircuit import (
    IonQToCircuit
)
from QCompute.OpenConvertor.CircuitToQobj import (
    CircuitToQobj
)
from QCompute.OpenConvertor.CircuitToXanaduSF import (
    CircuitToXanaduSF, TwoQubitsGate
)
from QCompute.QPlatform import QStatus
from QCompute.QPlatform import Utilities
from QCompute.Calibration import CalibrationUpdate, CalibrationReadData



from QCompute.QPlatform.QNoise.AmplitudeDamping import AmplitudeDamping
from QCompute.QPlatform.QNoise.BitFlip import BitFlip
from QCompute.QPlatform.QNoise.BitPhaseFlip import BitPhaseFlip
from QCompute.QPlatform.QNoise.CustomizedNoise import CustomizedNoise
from QCompute.QPlatform.QNoise.Depolarizing import Depolarizing
from QCompute.QPlatform.QNoise.PauliNoise import PauliNoise
from QCompute.QPlatform.QNoise.PhaseDamping import PhaseDamping
from QCompute.QPlatform.QNoise.PhaseFlip import PhaseFlip
from QCompute.QPlatform.QNoise.ResetNoise import ResetNoise
