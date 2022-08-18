#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
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
Module for discrete-event simulation environment with extra functionalities.
"""

from enum import Enum
from typing import List, Union

from qcompute_qnet.core.des import DESEnv

__all__ = [
    "QuantumEnv"
]


class QuantumEnv(DESEnv):
    r"""Class for discrete-event simulation environment with extra functionalities.

    Important:
        A ``QuantumEnv`` is a ``DESEnv`` with extra functionalities that can implement quantum circuits
        with virtual or real quantum backends.

        These quantum circuits are automatically generated from quantum network protocols.
    """

    def __init__(self, name: str, default=False):
        r"""Constructor for QuantumEnv class.

        Args:
            name (str): name of the discrete-event simulation environment
            default (bool): whether to set the current environment as default
        """
        super().__init__(name, default)

    def run(self, shots=1, backend=None, token=None, print_cir=True,
            end_time=None, logging=False, summary=True) -> Union[dict, List[dict]]:
        r"""Run the simulation and the quantum circuit with a given backend.

        Args:
            shots (int): number of circuit samples for a single circuit
            backend (Enum): backend to run the quantum circuit
            token (str): your token for QCompute backend
            print_cir (bool): whether to print the circuit
            end_time (float): end time of the simulation in picoseconds
            logging (bool): whether to output the simulation log file
            summary (bool): whether to print the simulation report on the terminal

        Returns:
            Union[dict, List[dict]]: circuit results
        """
        super().run(end_time, logging, summary)

        if len(self.network.circuits) == 0:  # single circuit
            cir = self.network.default_circuit
            cir.defer_measurement()
            if print_cir:
                cir.print_circuit(color=True)

            results = cir.run(shots, backend, token)

        else:  # multiple circuits
            circuits = {self.network.circuits.pop(0): 1}  # record the same circuits with their repeated times
            results = []

            for comp_cir in self.network.circuits:
                diff = True
                for cir in circuits:
                    if comp_cir.is_equal(cir):
                        diff = False
                        circuits[cir] += 1
                        break
                if diff is True:
                    circuits[comp_cir] = 1
            self.network.circuits = list(circuits.keys())

            for cir in circuits.keys():
                cir.defer_measurement()
                if print_cir:
                    cir.print_circuit(color=True)

                result = cir.run(circuits[cir], backend, token)
                results.append(result)

        return results

    def reset(self) -> None:
        pass
