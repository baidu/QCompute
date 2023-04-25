# !/usr/bin/env python3

"""
superdense coding example
"""

from QCompute import *

# Set the times of measurements
shots = 4096

# Please input you Token here
# Define.hubToken= 'Your Token'

# Set the message that Alice wishes to send to Bob
message = '11'

# Set up environment
env = QEnv()

# You can choose backend here. When choose 'Quantum Device' or 'Cloud Simulator',
# Please input your Token of QUANTUM LEAF first, otherwise, the code cannot excute.

# Using Local Simulator
env.backend(BackendName.LocalBaiduSim2)
# Using Quantum Device
# env.backend(BackendName.CloudIoPCAS)
# Using Cloud Simulator
# env.backend(BackendName.CloudBaiduSim2Water)

# Initialize all qubits
q = [env.Q[0], env.Q[1]]
# Prepare Bell state
H(q[0])
CX(q[0], q[1])

# Alice operates its qubits according to the information that needs to be transmitted
if message == '01':
    X(q[0])
elif message == '10':
    Z(q[0])
elif message == '11':
    Z(q[0])
    X(q[0])

# Bob decodes
CX(q[0], q[1])
H(q[0])

# Bob makes measurements
MeasureZ(q, range(2))
taskResult = env.commit(shots, fetchMeasure=True)
