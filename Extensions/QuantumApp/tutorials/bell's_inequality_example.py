# !/usr/bin/env python3

"""
bell's inequality example
"""

from QCompute import *
import numpy as np
from collections import Counter
from random import choice

# Please input you Token here
# Define.hubToken= 'your token'


def choose_backend():
    r"""
    Choose the backend of the QuLeaf

    :return: The backend.
    :rtype: BackendName
    """
    # You can choose backend here. When choose 'Quantum Device' or 'Cloud Simulator',
    # Please input your Token of QUANTUM LEAF first, otherwise, the code cannot excute.

    # Using Local Simulator
    backend = BackendName.LocalBaiduSim2
    # Using Quantum Device
    # backend = BackendName.CloudIoPCAS
    # Using Cloud Simulator
    # backend = BackendName.CloudBaiduSim2Water
    return backend


# Create a dictionary to record the measurement results of the first term
result1 = {"QS": [], "QT": [], "RS": [], "RT": []}

# Create a dictionary to record the measurement results of the second term
result2 = {"QS": [], "QT": [], "RS": [], "RT": []}

# run 100 times
times = 100
for i in range(times):
    # Alice randomly chooses between 'Q' and 'R' for measurement
    ranA = choice(["Q", "R"])
    # Bob randomly chooses between 'S' and 'T' for measurement
    ranB = choice(["S", "T"])
    ran = str(ranA) + str(ranB)

    # Every measurement only has one shot
    shots = 1
    env = QEnv()
    env.backend(choose_backend())

    q = [env.Q[0], env.Q[1]]
    # Prepare Bell state
    X(q[0])
    X(q[1])
    H(q[0])
    CX(q[0], q[1])

    if ran[0] == "R":
        H(q[0])

    MeasureZ(q, range(2))
    taskResult = env.commit(shots, fetchMeasure=True)["counts"]
    # Record the measurement result of the first term
    for key, value in taskResult.items():
        if value == 1:
            result1[ran].append(key)

    # Measure the second term
    shots = 1
    env = QEnv()
    env.backend(choose_backend())

    q = [env.Q[0], env.Q[1]]
    # Prepare Bell state
    X(q[0])
    X(q[1])
    H(q[0])
    CX(q[0], q[1])
    H(q[1])

    if ran[0] == "R":
        H(q[0])

    MeasureZ(q, range(2))
    taskResult = env.commit(shots, fetchMeasure=True)["counts"]
    # Record the measurement result of the second term
    for key, value in taskResult.items():
        if value == 1:
            result2[ran].append(key)

# Post-process experiment results
QS1 = Counter(result1["QS"])
QS2 = Counter(result2["QS"])
RS1 = Counter(result1["RS"])
RS2 = Counter(result2["RS"])
RT1 = Counter(result1["RT"])
RT2 = Counter(result2["RT"])
QT1 = Counter(result1["QT"])
QT2 = Counter(result2["QT"])


def exp(Measure):
    r"""
    Calculate the expectation value of measurement

    :param Measure: The counts of the measurement.
    :type Measure: Counter
    :return: the expectation value
    :rtype: float
    """
    # Calculate the expected value of measurement
    summary = Measure["00"] - Measure["01"] - Measure["10"] + Measure["11"]
    total = Measure["00"] + Measure["01"] + Measure["10"] + Measure["11"]
    return 1 / np.sqrt(2) * summary / total


a_list = [QS1, QS2, RS1, RS2, RT1, RT2, QT1, QT2]

# Combine two sub-terms to get the expected value
QS = -exp(QS1) - exp(QS2)
RS = -exp(RS1) - exp(RS2)
RT = exp(RT1) - exp(RT2)
QT = exp(QT1) - exp(QT2)

print("E(QS)=", QS)
print("E(RS)=", RS)
print("E(RT)=", RT)
print("E(QT)=", QT)

print("Expected value: E(QS)+E(RS)+E(RT)-E(QT)=", QS + RS + RT - QT)
