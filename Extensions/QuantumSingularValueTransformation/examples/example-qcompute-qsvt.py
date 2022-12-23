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

r"""
This is a simple example to test if you have successfully installed the QSVT toolkit.
"""

import numpy as np
from QCompute.Define import Settings as QC_Settings

from qcompute_qsvt.Application.HamiltonianSimulation import func_HS_QSVT

# not to draw the quantum circuit locally
QC_Settings.drawCircuitControl = []
QC_Settings.outputInfo = False
QC_Settings.autoClearOutputDirAfterFetchMeasure = True


if __name__ == "__main__":
    from qcompute_qsvt.SymmetricQSP import Settings as SQSP_Settings
    SQSP_Settings.INTERNAL = "python"  # test SymmetricQSPExternalPy
    # the return should be approximated to `{'000000': 5000, '100000': 1000, '010000': 1000, '110000': 1000}`
    print(func_HS_QSVT(list_str_Pauli_rep=[(1, 'X0X1'), (1, 'X0Z1'), (1, 'Z0X1'), (1, 'Z0Z1')],
                       num_qubit_sys=2, float_tau=-np.pi / 8, float_epsilon=1e-6, circ_output=False)['counts'])
    SQSP_Settings.INTERNAL = "cpp"  # test SymmetricQSPExternalCpp
    # the return should be approximated to `{'000000': 5000, '100000': 1000, '010000': 1000, '110000': 1000}`
    print(func_HS_QSVT(list_str_Pauli_rep=[(1, 'X0X1'), (1, 'X0Z1'), (1, 'Z0X1'), (1, 'Z0Z1')],
                       num_qubit_sys=2, float_tau=-np.pi / 8, float_epsilon=1e-6, circ_output=False)['counts'])
