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
__init__ file of the `qcompute_qsvt.Oracle` module.
"""

from qcompute_qsvt.Oracle.BlockEncoding import circ_block_encoding, circ_ctrl_Sel_multiPauli, circ_j_ctrl_multiPauli
from qcompute_qsvt.Oracle.StatePreparation import circ_state_pre, circ_state_pre_inverse

__all__ = [
    'circ_block_encoding', 'circ_ctrl_Sel_multiPauli', 'circ_j_ctrl_multiPauli',
    'circ_state_pre', 'circ_state_pre_inverse'
]