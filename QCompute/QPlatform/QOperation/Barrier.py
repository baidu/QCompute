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
Barrier Operation
"""
from typing import TYPE_CHECKING

from QCompute.QPlatform.QOperation import QOperation

if TYPE_CHECKING:
    from QCompute.QPlatform.QRegPool import QRegStorage


class BarrierOP(QOperation):
    """
    The barrier instruction

    Barrier does nothing for implementing circuits on simulator but does STOP optimization between two barriers
    """

    def __call__(self, *qRegList: 'QRegStorage') -> None:
        self._op(list(qRegList))


Barrier = BarrierOP()
