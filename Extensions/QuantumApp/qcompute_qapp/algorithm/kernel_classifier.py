# !/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2021 Baidu, Inc. All Rights Reserved.
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
from ..circuit import KernelEstimationCircuit


class KernelClassifier:
    """Kernel Classifier class"""

    def __init__(self, backend: str, encoding_style: str = "IQP", kernel_type: str = "qke", shots: int = 1024):
        r"""The constructor of the KernelClassifier class

        Args:
            backend (str): Backend to be used in this task. Please refer to https://quantum-hub.baidu.com/quickGuide
                           for details
            encoding_style (str): Encoding scheme to be used, defaults to 'IQP', which uses the default encoding scheme
            kernel_type (str): Type of kernel to be used, defaults to 'qke', i.e., <x1|x2>
            shots (int): Number of measurement shots, defaults to 1024

        """
        self._backend = backend
        self._shots = shots
        self._clf = svm.SVC()
        self._num_evaluation = 0
        self._kernel_matrix = None
        self._normalized_training_vectors = None
        self._unnormalized_training_vectors = None
        self._training_labels = None
        if kernel_type == "qke":
            self._kernel_circuit = KernelEstimationCircuit
            if encoding_style == "IQP":
                self._encoding_style = encoding_style
            else:
                raise ValueError(
                    f"Error EA01001(QAPP): The encoding style {encoding_style:s} "
                    f"is not compatible with kernel type {kernel_type:s}"
                )
        else:
            raise ValueError("Error EA01002(QAPP): The kernel type is not supported yet")
        self._kernel_type = kernel_type

    def _compute_kernel_entry(self, x1: np.ndarray, x2: np.ndarray) -> float:
        r"""Computes the kernel entry value between two classical data vectors

        Args:
            x1 (np.ndarray): First classical data
            x2 (np.ndarray): Second classical data

        Returns:
            float: Kernel entry value

        """
        qubit_num = len(x1)
        env = QEnv()
        env.backend(self._backend)
        q = env.Q.createList(qubit_num)
        circuit = self._kernel_circuit(num=qubit_num, encoding_style=self._encoding_style)
        circuit.add_circuit(q, x1, x2)
        # Submit job
        counts = env.commit(self._shots, fetchMeasure=True)["counts"]
        self._num_evaluation += 1
        # Expectation
        if str(0) * qubit_num in counts.keys():
            return counts[str(0) * qubit_num] / self._shots
        else:
            return 0

    def _get_kernel_matrix(self, data_vectors_1, data_vectors_2) -> np.ndarray:
        r"""Compute the kernel matrix between two set of data vectors

        Args:
            data_vectors_1 (np.ndarray): Set of data vectors of shape (n_features, n_data_1)
            data_vectors_2 (np.ndarray): Set of data vectors of shape (n_features, n_data_2) or None. If None, will be
                set automatically to the same as data_vectors_1. (Note: it's preferable to set data_vectors_2 to None
                when computing the kernel matrix of the same data)

        Returns:
            np.ndarray: A kernel matrix of shape (n_data_1, n_data_2)

        """
        if data_vectors_2 is not None:
            kernel_matrix = np.zeros([len(data_vectors_1), len(data_vectors_2)])
            for i in range(len(data_vectors_1)):
                for j in range(len(data_vectors_2)):
                    kernel_matrix[i, j] = self._compute_kernel_entry(data_vectors_1[i], data_vectors_2[j])
            kernel_matrix = kernel_matrix.transpose()
        else:
            kernel_matrix = np.zeros([len(data_vectors_1), len(data_vectors_1)])
            for i in range(len(data_vectors_1)):
                for j in range(i, len(data_vectors_1)):
                    kernel_matrix[i, j] = self._compute_kernel_entry(data_vectors_1[i], data_vectors_1[j])
                    kernel_matrix[j, i] = kernel_matrix[i, j]
        return kernel_matrix

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        r"""Trains the classifier with known data

        Args:
            X (np.ndarray): Set of classical data vectors as the training data
            y (np.ndarray): Known labels of the training data

        """
        self._unnormalized_training_vectors = X
        self._training_labels = y
        X = X / abs(X).max() * np.pi
        self._normalized_training_vectors = X
        self._kernel_matrix = self._get_kernel_matrix(self._normalized_training_vectors, None)
        self._clf.kernel = "precomputed"
        self._clf.fit(self._kernel_matrix, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        r"""Predicts labels of new data

        Args:
            x (np.ndarray): Set of data vectors with unknown labels

        Return:
            np.ndarray: Predicted labels of the input data

        """
        x = x / abs(x).max() * np.pi
        prediction_kernel_matrix = self._get_kernel_matrix(self._normalized_training_vectors, x)
        return self._clf.predict(prediction_kernel_matrix)
