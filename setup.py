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
  The setup script to install for python
"""

from __future__ import absolute_import

from pathlib import Path

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup

SDK_VERSION = '3.3.5'

DESC = Path('./README.md').read_text(encoding='utf-8')

setup(
    name='qcompute',
    version=SDK_VERSION,
    install_requires=[
        # PySDK
        'protobuf==4.21.1',
        'numpy>=1.17.3',
        'requests==2.31.0',
        'bidict==0.22.1',
        'bce-python-sdk==0.8.87',
        'antlr4-python3-runtime==4.13.0',
        'py-expression-eval==0.3.14',
        'websocket-client==1.6.1',
        'tqdm>=4.5.0',
        'nanoid==2.0.0',
        'multiprocess==0.70.15',
        

        # Example
        'scipy>=1.8.0',
        'matplotlib>=3.3.0',
        'sympy==1.12',
        'pyprimes==0.1',
    ],
    python_requires='>=3.8, <3.11',
    packages=find_packages(),
    url='https://quantum.baidu.com',
    license='Apache License 2.0',
    author='Baidu Quantum',
    author_email='quantum@baidu.com',
    description='QCompute is a Python-based quantum software development kit (SDK). '
                'It provides a full-stack programming experience for advanced users '
                'via hybrid quantum programming language features and a high-performance simulator.',
    long_description=DESC,
    long_description_content_type='text/markdown'
)
