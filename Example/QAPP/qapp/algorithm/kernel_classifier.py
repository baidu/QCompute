# -*- coding: UTF-8 -*-
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
Quantum Kernel Classifier
"""

import numpy as np
from sklearn import svm
from QCompute.QPlatform.QEnv import QEnv
from qapp.circuit import KernelEstimationCircuit


class KernelClassifier:
    """Kernel Classifier class
    """
    def __init__(self, backend: str, encoding_style: str = 'IQP', kernel_type: str = 'qke', shots: int = 1024):
        """The constructor of the KernelClassifier class

        :param encoding_style: Encoding scheme to be used, defaults to 'IQP', which uses the default encoding scheme
        :param kernel_type: Type of kernel to be used, defaults to 'qke', i.e., <x1|x2>
        :param backend: Backend to be used in this task. Please refer to https://quantum-hub.baidu.com/quickGuide for details
        :param shots: Number of measurement shots, defaults to 1024
        """
        self._backend = backend
        self._shots = shots
        self._clf = svm.SVC()
        self._num_evaluation = 0
        if kernel_type == 'qke':
            self._kernel_circuit = KernelEstimationCircuit
            if encoding_style == 'IQP':
                self._encoding_style = encoding_style
            else:
                raise ValueError(
                    'The encoding style {} is not compatible with kernel type {}'.format(encoding_style, kernel_type))
        else:
            raise ValueError('The kernel type is not supported yet')
        self._kernel_type = kernel_type

    def _compute_kernel_entry(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Computes the kernel entry value between two classical data vectors

        :param x1: First classical data
        :param x2: Second classical data
        :return: Kernel entry value
        """

        qubit_num = len(x1)
        env = QEnv()
        env.backend(self._backend)
        q = env.Q.createList(qubit_num)
        circuit = self._kernel_circuit(num=qubit_num, encoding_style=self._encoding_style)
        circuit.add_circuit(q, x1, x2)
        # Submit job
        counts = env.commit(self._shots, fetchMeasure=True)['counts']
        self._num_evaluation += 1
        # Expectation
        if str(0) * qubit_num in counts.keys():
            return counts[str(0) * qubit_num] / self._shots
        else:
            return 0

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Trains the classifier with known data

        :param X: Set of classical data vectors as the training data
        :param y: Known labels of the training data
        """

        X = X / abs(X).max() * np.pi

        def query_kernel_matrix(X1, X2):
            return np.array([[self._compute_kernel_entry(x1, x2) for x2 in X2] for x1 in X1])

        self._clf.kernel = query_kernel_matrix
        self._clf.fit(X, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predicts labels of new data

        :param x: Set of data vectors with unknown labels
        :return: Predicted labels of the input data
        """
        x = x / abs(x).max() * np.pi
        return self._clf.predict(x)
