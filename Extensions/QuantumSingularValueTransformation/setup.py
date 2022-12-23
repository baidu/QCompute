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

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='qcompute-qsvt',
    version='0.1.1',
    install_requires=[
        'qcompute==3.2.0',
    ],
    python_requires='>=3.9, <3.10',
    packages=find_packages(),
    package_data={
        'qcompute_qsvt.SymmetricQSP.x64': [
            'libiomp5md.dll',
            'msvcp140.dll',
            'vcruntime140.dll',
            'SymmetricQSPInternalCpp.cp39-win_amd64.pyd',
            'SymmetricQSPInternalCpp.cpython-39-darwin.so',
            'SymmetricQSPInternalCpp.cpython-39-x86_64-linux-gnu.so',
        ],
        'qcompute_qsvt.SymmetricQSP.arm': [
            'SymmetricQSPInternalCpp.cpython-39-darwin.so',
        ],
    },
    url='https://quantum.baidu.com',
    license='Apache License 2.0',
    author='Baidu Quantum',
    author_email='quantum@baidu.com',
    description='Quantum Singular Value Transformation toolkit developed '
                'by the Institute for Quantum Computing at Baidu Research.',
    long_description=long_description,
    long_description_content_type='text/markdown'
)
