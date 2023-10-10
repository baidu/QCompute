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
Commit QASM Circuit directly for execution with an option parameter.
"""

qasmSource = '''
OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
creg c[6];
gate nG0(param0,param1) qb0,qb1,qb2
{
  u(pi, param0, param1) qb0;
  u(param0/2, pi/2, param1) qb2;
  u(param0/2, pi/2, param1) qb1;
}
gate nG1(param0,param1) qb0,qb1,qb2
{
  nG0(pi, param0/2) qb0, qb1, qb2;
  nG0(pi/4, param0/2) qb2, qb1, qb0;
  cx qb0, qb2;
  cx qb2, qb1;
}
nG1(pi, pi/2)  q[3], q[1], q[2];
measure q[1] -> c[1];
measure q[0] -> c[0];
'''

import sys
from pprint import pprint

sys.path.append('../..')
from QCompute import *

matchSdkVersion('Python 3.3.5')

# Create environment
env = QEnv()
# Choose backend Baidu local simulator
env.backend(BackendName.LocalBaiduSim2)

program = QasmToCircuit().convert(qasmSource)

# Commit the task with 1024 shots
taskResult = env.commit(1024, fetchMeasure=True, program=program)

pprint(taskResult)
