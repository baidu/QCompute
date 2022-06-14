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
Classify the hand written digits (MNIST) data with a quantum kernel classifier
"""

import sys

sys.path.append('../../..')  # "from QCompute import *" requires this
from QCompute import *

sys.path.append('..')
import time
import json
import numpy as np
from skimage.transform import resize

from QCompute.QPlatform import BackendName
from qapp.algorithm.kernel_classifier import KernelClassifier

# If user have a Quantum-hub account and wish to use a cloud simulator/qpu, please

# Readers should get their tokens from quantum-hub.baidu.com to be connected to real quantum devices and cloud backend.
# from QCompute import Define
# Define.hubToken = 'your token'
# backend = BackendName.CloudIoPCAS
# backend = BackendName.CloudBaiduSim2Water
backend = BackendName.LocalBaiduSim2
myKernelClassifier = KernelClassifier(backend=backend)

# Load local MNIST data
print('Loading local MNIST data...')
fp = open('./data/MNIST_data.json', 'rt+')
mnist = json.load(fp)
fp.close()
print('Complete!')


# A utility function used to generate the data set
def get_data(first_num=1,
             second_num=8,
             data_size=10,
             scale=2):
    """function used to generate a bi-classification dataset from the MNIST data set with reduced scale and size

    :param first_num: The label of the first class
    :param second_num: The label of the second class
    :param data_size: Size of the generated data set
    :param scale: n, the returned data will be a set of nxn pictures
    :return: data, label
    """
    data_first_num = np.array(mnist['data'])[np.array(mnist['target']) == str(first_num)]
    data_second_num = np.array(mnist['data'])[np.array(mnist['target']) == str(second_num)]

    data_unsampled = np.vstack([data_first_num, data_second_num])
    label_unsampled = np.hstack([np.zeros(len(data_first_num)), np.ones(len(data_second_num))])

    idx = np.random.randint(low=0, high=len(label_unsampled), size=data_size)

    data_uncompressed = data_unsampled[idx]
    label = label_unsampled[idx]

    data_unnormalized = [resize(image.reshape([28, 28]), [scale, scale]).flatten() for image in data_uncompressed]

    data = [(image / image.max()) * np.pi for image in data_unnormalized]

    return np.array(data), label


# Generate the training and testing set
data_train, label_train = get_data(data_size=18, scale=3)
data_test, label_test = get_data(data_size=10, scale=3)

start = time.time()

# Train the classifier with the training set
print('Training the classifier...')
myKernelClassifier.fit(data_train, label_train)
print('Complete!')

# Make prediction on the testing set
print('Predicting on testing data...')
predict_svm_qke_test = myKernelClassifier.predict(data_test)
print('Complete!')

run_time = time.time() - start
print('Kernel classification run time:', run_time)

# Calculate the error rate
error_rate_test = sum(abs(predict_svm_qke_test - label_test)) / len(predict_svm_qke_test)
print('Testing Error Rate:', error_rate_test)
