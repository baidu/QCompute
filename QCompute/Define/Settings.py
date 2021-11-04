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
Configuration
"""


from QCompute.Define import MeasureFormat

outputInfo = True
"""
Output info setting.

Output the returned information from quantum-hub to console.

The information also can be found from quantum-hub website.

Values: True, False
"""

drawCircuitControl = None
"""
Draw circuit control.

Draw the circuit to console.

Values: None, [], [...]
"""

drawCircuitCustomizedGateHashLength = 2
"""
Customized gate hash length, when draw circuit.

Hex str, length * 2

|Custom[xxxx]|

Values: int
"""

measureFormat = MeasureFormat.Bin
"""
Measure output format setting. 

Measure key is '00', '01', '10', '11' or '0x0', '0x1', '0x2', '0x3'

This format can reduce the object size of the results. The default is 'Bin'. 

Values: MeasureFormat.Bin, MeasureFormat.Hex
"""

inProcessSimulator = True
"""
Run simulator in or out process.

The 'inProcessSimulator' option can significantly accelerate the calculation of simulator while 
the outProcessSimulator can enhance the stability of simulator.  

Supporting local_baidu_sim2

Values: True, False
"""

linuxDirectSim2Cpp = False
"""
Use local sim2 cpp directly in linux.
"""



