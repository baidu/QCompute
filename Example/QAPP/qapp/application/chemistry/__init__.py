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
Export the entire directory as a library
"""

from .molecular_ground_state_energy import MolecularGroundStateEnergy

__all__ = [
    'MolecularGroundStateEnergy',
]


H2_HAMILTONIAN = (4,
    [[-0.09706626816762864, 'iiii'], [0.17141282644776892, 'ziii'], [0.17141282644776895, 'izii'], 
     [-0.223431536908136, 'iizi'], [-0.223431536908136, 'iiiz'], [0.16868898170361213, 'zzii'], 
     [0.12062523483390426, 'zizi'], [0.16592785033770355, 'ziiz'], [0.16592785033770355, 'izzi'], 
     [0.12062523483390426, 'iziz'], [0.17441287612261608, 'iizz'], [-0.0453026155037993, 'xxyy'], 
     [0.0453026155037993, 'xyyx'], [0.0453026155037993, 'yxxy'], [-0.0453026155037993, 'yyxx']])

# Only valid for solving ground state
LiH_HAMILTONIAN = (4, 
    [[-7.506303018570541, 'iiii'], [0.15814738862267735, 'ziii'], [0.013623738087373657, 'xzxi'], 
     [0.013623738087373657, 'yzyi'], [0.15814738862267735, 'izii'], [0.013623738087373658, 'ixzx'], 
     [0.013623738087373658, 'iyzy'], [-0.01449187180210662, 'iizi'], [-0.01449187180210662, 'iiiz'], 
     [0.1227406966266831, 'zzii'], [0.01192593972942561, 'xixi'], [0.01192593972942561, 'yiyi'], 
     [0.01192593972942561, 'zxzx'], [0.01192593972942561, 'zyzy'], [0.003139712545543703, 'xyyx'], 
     [-0.003139712545543703, 'xxyy'], [-0.003139712545543703, 'yyxx'], [0.003139712545543703, 'yxxy'], 
     [0.05314831356196822, 'zizi'], [0.056288026107511914, 'ziiz'], [-0.0016977952301184007, 'xzxz'], 
     [-0.0016977952301184007, 'yzyz'], [0.056288026107511914, 'izzi'], [-0.0016977952301184007, 'ixix'], 
     [-0.0016977952301184007, 'iyiy'], [0.05314831356196822, 'iziz'], [0.08460108207835978, 'iizz']])
