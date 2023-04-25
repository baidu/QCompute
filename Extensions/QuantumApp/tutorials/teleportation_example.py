# !/usr/bin/env python3

"""
quantum teleportation example
"""

from QCompute import *
import numpy as np

# Set number of qubits
qubit_num = 3
# Set shots
shots = 10000
# Fix randim number seed
np.random.seed(14)

# Generate 3 random angles
angle = 2 * np.pi * np.random.randn(3)

# Please input you Token here
# Define.hubToken= 'Your Token'


def choose_backend():
    """
    Choose the backend we wish to use
    """
    # You can choose backend here. When choose 'Quantum Device' or 'Cloud Simulator',
    # Please input your Token of QUANTUM LEAF first, otherwise, the code cannot excute.

    # Using Local Simulator
    backend = BackendName.LocalBaiduSim2
    # Using Quantum Device
    # backend = BackendName.CloudIoPCAS
    # Using Cloud Simulator
    # backend = BackendName.CloudBaiduSim2Earth
    return backend


def prepare_state():
    """
    prepare the state that we wish to teleport
    """
    # Set up environment
    env = QEnv()
    env.backend(choose_backend())

    # Initialize all qubits
    q = [env.Q[i] for i in range(qubit_num)]

    # Prepare quantum state |psi> 
    U(angle[0], angle[1], angle[2])(q[0])

    # Measurements
    MeasureZ([q[0]], [0])

    taskResult = env.commit(shots, fetchMeasure=True)
    return taskResult['counts']


def main():
    """
    Execute the main function
    """
    # Set up environment
    env = QEnv()
    env.backend(choose_backend())

    # Initialize all qubits
    q = [env.Q[i] for i in range(qubit_num)]

    # Prepare entangled state between Alice and Bob.
    H(q[1])
    CX(q[1], q[2])

    # Prepare quantum state |psi>
    U(angle[0], angle[1], angle[2])(q[0])

    # Alice acts gates on her qubits
    CX(q[0], q[1])
    H(q[0])

    # Bob acts gate to recover |psi>
    CZ(q[0], q[2])
    CX(q[1], q[2])

    # Bob makes measurements on qubit q2
    MeasureZ([q[2]], [2])

    taskResult = env.commit(shots, fetchMeasure=True)
    return taskResult['counts']


if __name__ == '__main__':
    prepare_state()
    main()
