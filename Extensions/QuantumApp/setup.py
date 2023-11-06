#!/usr/bin/env python
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

r"""
Install library to site-packages
"""

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="qcompute-qapp",
    version="0.0.1",
    author="Institute for Quantum Computing, Baidu INC.",
    author_email="quantum@baidu.com",
    description=(
        "A quantum computing toolbox based on the QCompute component of Quantum Leaf "
        "which provides quantum computing services for solving practical problems."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://quantum-hub.baidu.com/qapp/",
    packages=[
        "qcompute_qapp",
        "qcompute_qapp.algorithm",
        "qcompute_qapp.application",
        "qcompute_qapp.application.chemistry",
        "qcompute_qapp.application.optimization",
        "qcompute_qapp.circuit",
        "qcompute_qapp.optimizer",
        "qcompute_qapp.utils",
    ],
    install_requires=[
        "networkx>=2.5.1",
        "numpy>=1.20.3",
        "tqdm>=4.58.0",
        "scipy>=1.6.3",
        "noisyopt>=0.2.2",
        "qcompute>=2.0.0",
        "scikit-learn>=0.24.2",
        "scikit-image",
    ],
    python_requires=">=3.6, <4",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    project_urls={
        "Documentation": "https://quantum-hub.baidu.com/qapp/tutorial-overview",
        "Source": "https://github.com/baidu/QCompute/tree/master/Extensions/QuantumApplication",
        "Tracker": "https://github.com/PaddlePaddle/Quantum/issues",
    },
)
