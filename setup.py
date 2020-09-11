#!/usr/bin/python3
# -*- coding: utf8 -*-

# Copyright (c) 2020 Baidu, Inc. All Rights Reserved.
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
  The setup script to install  for python
"""

from __future__ import absolute_import

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup

SDK_VERSION = '0.0.1'

with open('./README.md', 'r', encoding='utf-8') as srcfd:
    DESC = srcfd.read()

setup(
    name='qcompute',
    version=SDK_VERSION,
    install_requires=[
        'protobuf==3.13.0',
        'numpy==1.19.1',
        'scipy==1.5.2',
        'requests==2.24.0',
        'pprint==0.1',
        'bidict==0.21.0',
        'bce-python-sdk-reborn==0.8.32',
        'matplotlib==3.3.1',
    ],
    python_requires='>=3.6, <4',
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
