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

"""
Implementation of the Zero-Noise Extrapolation method in the Quantum Circuit Model.
"""
from typing import Callable, List, Union, Any
from matplotlib.figure import Figure

from qcompute_qep.exceptions.QEPError import ArgumentError
from qcompute_qep.mitigation import Mitigator
import qcompute_qep.mitigation.zne as zne
from qcompute_qep.utils.types import QProgram, QComputer


class ZNEMitigator(Mitigator):
    r"""
    Implementation of the Zero-Noise Extrapolation method in the Quantum Circuit Model.
    """
    def __init__(self,
                 folder: Union[zne.Folder, str] = None,
                 extrapolator: Union[zne.Extrapolator, str] = None,
                 **kwargs: Any) -> None:
        r"""
        Initialization function of the Zero-Noise Extrapolation method,

        :param folder: Union[Folder, str], a `Folder` instance or a `str` representing the name of the a Folder class
        :param extrapolator: Union[Extrapolator, str], an `Extrapolator` instance or
                                        a `str` representing the name of the an Extrapolator class
        :param args: optional arguments
        :param kwargs: other optional keyword arguments
        """
        super(ZNEMitigator, self).__init__()

        # Setup the folder instance from the input data
        if folder is None:  # set default folder instance
            self._folder = zne.folder.CircuitFolder(method='right')
        elif isinstance(folder, zne.folder.Folder):
            self._folder = folder
        else:  # Construct the folder from its name
            folder = folder.lower()
            if folder not in zne.folder.__SUPPORTED_FOLDERS__:
                raise ArgumentError('{} is not a supported folder type!'.format(folder))
            else:
                self._folder = getattr(zne.folder, folder.capitalize() + 'Folder')()

        # Setup the extrapolator instance from the input data
        if extrapolator is None:
            self._extrapolator = zne.extrapolator.ExponentialExtrapolator()
        elif isinstance(extrapolator, zne.extrapolator.Extrapolator):
            self._extrapolator = extrapolator
        else:  # Construct the extrapolator from its name
            extrapolator = extrapolator.lower()
            if extrapolator not in zne.extrapolator.__SUPPORTED_EXTRAPOLATORS__:
                raise ArgumentError('{} is not a supported extrapolator type!'.format(extrapolator))
            else:
                self._extrapolator = getattr(zne.extrapolator, extrapolator.capitalize() + 'Extrapolator')()

        self._scale_factors: List[float] = kwargs.get('scale_factors', self._config_default_scale_factors())
        self._history: dict = {'folder': self._folder.name, 'extrapolator': self._extrapolator.name}

    def mitigate(self,
                 qp: QProgram,
                 qc: QComputer,
                 calculator: Callable,
                 scale_factors: List[float] = None, **kwargs) -> float:
        r"""Implement the Zero-Noise Extrapolation method.

        Using the Zero-Noise Extrapolation method to improve the computation accuracy of the quantum algorithm.
        This method accepts

        + a quantum circuit @qp that describes the quantum algorithm,
        + a quantum computer @qc on which the quantum circuit will run, and
        + a calculator @calculator with which the expectation value will be estimated from the measurement counts

        as its inputs, use the 'first folding then extrapolating' technique to obtain an error-mitigated
        expectation value. Roughly, ZNE works as follows.

        .. admonition:: Procedure

            + Step 1: For each factor in @scale_factors, fold @qp accordingly and store the folded_qp.
            + Step 2: Run all folded_qps in the quantum computer @qp and estimate the noisy values using @calculator.
            + Step 3: Extrapolate these noisy values with respect to the scaling factors to obtain an error-mitigated expectation value.

        .. note::
            The function `calculator` must accept `QProgram` and `QComputer` as inputs and output a float number.
            Within the function, it will run the `QProgram` in the `QComputer`, collect the measurement outcomes,
            and estimate the expectation value.
            Here is an example that runs `QProgram` in the `QComputer`, collect the measurement outcomes,
            and estimate the expectation value of the Pauli z observable.

            .. code-block:: python
                :linenos:

                def calculator(qp: QProgram, qc: QComputer) -> float:
                    # Obtain the output raw counts
                    counts = execute(qp, qc, shots='4096')
                    # Estimate the expectation value of the Pauli z observable
                    return expval_z_from_counts(counts)

        :param qp: QProgram, a quantum program instance
        :param qc: QComputer, a quantum computer instance
        :param calculator: Callable, a callable object accepting qp and qc as inputs and returning a float number
        :param scale_factors: List[float], scaling factors
        :param kwargs: other optional key word arguments
        :return: float, error-mitigated expectation value obtained by the ZNE method
        """
        # If the input @scale_factor is not None, update the property
        if scale_factors is not None:
            self._scale_factors = scale_factors
        elif self._scale_factors is not None:
            pass
        else:  # Both the input and property @scale_factor are None, use the default configuration
            self._config_default_scale_factors()

        # Store the noisy expectation values
        expvals = []

        # For each factor in @scale_factors, fold @qp according to the scale factor,
        # run it in the quantum computer @qp, and estimate the noisy values using @calculator
        for i, lam in enumerate(scale_factors):
            qp_folded = self._folder(qp, lam)
            val = calculator(qp_folded, qc)
            expvals.append(val)

        # Extrapolate these noisy values to obtain an error-mitigated expectation value
        miti_value = self._extrapolator(scale_factors,
                                        expvals,
                                        order=kwargs.get('order', None),
                                        asymptote=kwargs.get('asymptote', None),
                                        ansatz=kwargs.get('ansatz', None),
                                        init_paras=kwargs.get('init_paras', None))

        # Record the mitigation history
        self._history['scale_factors'] = scale_factors
        self._history['expectations'] = expvals
        self._history['extrapolation_result'] = self._extrapolator.extrapolation_result
        self._history['mitigated_value'] = miti_value

        return miti_value

    def _config_default_scale_factors(self) -> None:
        r"""
        Configure default scaling factors.
        """
        self._scale_factors = list(range(1, 3, 5))

    def plot_extrapolation_results(self, save: bool = False, fname: str = None) -> Figure:
        r"""
        Plot the extrapolation result, using the method implemented in `extrapolator`.

        :param save: bool, indicate if to save the figure or not
        :param fname: str, if save=True, save the figure with the given figure name
        :return: Figure, a `matplotlib.figure.Figure` instance
        """
        return self._extrapolator.plot_extrapolation_results(save, fname)

    def __str__(self) -> str:
        s = 'A {} object, with folder: {} and extrapolator: {}.'.format(
            self.__class__.__name__,
            self.history['folder'],
            self.history['extrapolator'])
        return s

    @property
    def history(self):
        r"""
        A dict-type property, used to record ZNE mitigating data, including the scaling_factors,
        expectations, extrapolation_result and the final mitigated_value.
        """
        if self._history is None:
            raise NotImplementedError
        else:
            return self._history

    @property
    def scale_factors(self):
        r"""
        Return an array_like object, representing the scaling factors that are used for ZNE mitigation.
        """
        return self._scale_factors

    @property
    def folder(self):
        return self._folder

    @property
    def extrapolator(self):
        return self._extrapolator
