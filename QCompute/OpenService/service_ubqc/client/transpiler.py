# Copyright (c) 2022 Institute for Quantum Computing, Baidu Inc. All Rights Reserved.
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

# !/usr/bin/env python3

"""
transpiler
"""
FileErrorCode = 6


from QCompute.OpenService import ModuleErrorCode
from QCompute.QPlatform import Error

from QCompute.OpenService.service_ubqc.client.qobject import Circuit
from QCompute.OpenService.service_ubqc.client.mcalculus import MCalculus

__all__ = [
    "transpile_to_brickwork"
]


def transpile_to_brickwork(circuit, to_xy_measurement=True):
    r"""Translate a quantum circuit to its equivalent brickwork pattern.

    In this method, the quantum circuit is translated to its equivalent measurement pattern in MBQC model.
    This pattern is called 'brickwork pattern' because it has a specific brickwork structure which is crucial in UBQC.
    Please see the reference [arXiv:0807.4154] for more details.

    Args:
        circuit (Circuit): quantum circuit
        to_xy_measurement (bool): whether or not to convert all measurements to the measurements in XY plane
                                  (In UBQC, all measurements are set to be in XY plane as default.)

    Returns:
        Pattern: a brickwork pattern equivalent to the original quantum circuit
    """
    if not isinstance(circuit, Circuit):
        raise Error.ArgumentError(f'Invalid circuit ({circuit}) with the type: ({type(circuit)})!\nOnly `Circuit` is supported as the type of quantum circuit.', ModuleErrorCode, FileErrorCode, 1)

    mc = MCalculus()
    circuit.simplify_by_merging(to_xy_measurement)
    circuit.to_brickwork()
    mc.set_circuit(circuit)
    mc.to_brickwork_pattern()
    mc.standardize()
    pattern = mc.get_pattern()
    return pattern