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
Install library to site-packages.
"""

from __future__ import absolute_import

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='qcompute-qnet',
    version='1.1.0',
    description='A Quantum NETwork toolkit developed by the Institute for Quantum Computing at Baidu Research.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Institute for Quantum Computing, Baidu INC.',
    author_email='quantum@baidu.com',
    url="https://quantum-hub.baidu.com/qnet/tutorial-introduction",
    python_requires='>=3.7, <3.10',
    packages=find_packages(),
    install_requires=[
        'numpy==1.21.3',
        'pandas==1.4.3',
        'networkx==2.8.3',
        'matplotlib==3.5.2'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    license='Apache 2.0',
)
