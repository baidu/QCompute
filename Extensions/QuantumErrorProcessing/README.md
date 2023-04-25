*Copyright (c) 2022 Institute for Quantum Computing, Baidu Inc. All Rights Reserved.*

![](https://img.shields.io/badge/release-v1.1.0-blue)
[![](https://img.shields.io/badge/docs-API-blue)](https://quantum-hub.baidu.com/docs/qep/)
[![](https://img.shields.io/badge/Python-3.8+-green)](https://www.python.org/)
![](https://img.shields.io/badge/OS-MacOS%20|%20Windows%20|%20Linux-green)
[![](https://img.shields.io/badge/license-Apache%202.0-orange)](https://github.com/baidu/QCompute/blob/master/LICENSE)

## About QEP

**QEP** is a **Q**uantum **E**rror **P**rocessing toolkit developed by the [Institute for 
Quantum Computing](https://quantum.baidu.com) at [Baidu Research](http://research.baidu.com). It 
aims to deal with quantum errors inherent in quantum devices using software solutions. Currently,
it offers four powerful quantum error processing functions: performance evaluation, 
quantum error characterization, quantum error mitigation, and quantum error correction:

+ **Performance Evaluation** is used for assessing the capabilities and extendibilities of 
  quantum computing hardware platforms, through estimating the error rates of quantum states, 
  quantum gates, and quantum measurement apparatus. It provides 
  standard randomized benchmarking, interleaved randomized benchmarking, cross-entropy 
  benchmarking, unitarity randomized benchmarking, direct fidelity estimation, and 
  cross-platform fidelity estimation methods.

+ **Quantum Error Characterization** is used for reconstructing the comprehensive information in 
  quantum computing hardware platforms, through many partial and limited experimental results. 
  It provides quantum state tomography, quantum process tomography, quantum detector tomography, 
  quantum gateset tomography, and spectral quantum tomography.

+ **Quantum Error Mitigation** is used for improving the accuracy of quantum computational 
  results, through post-processing the experiment data obtained by varying noisy experiments,  
  extending the computational reach of a noisy superconducting quantum processor. It provides 
  zero-noise extrapolation technique to mitigate quantum gate noise, and a collection of methods 
  such as inverse, least-square, iterative Bayesian unfolding, Neumann series to mitigate quantum measurement noise.

+ **Quantum Error Correction (QEC)** is used to protect quantum information from errors due to 
  environmental noise and imperfections in hardware. Quantum computers rely on the delicate 
  properties of quantum systems, which are susceptible to errors, and thus, QEC has become an 
  essential tool for realizing fault-tolerant quantum computing. It provides a simulator that 
  can simulate error correction codes based on the stabilizer formalism, allowing users to study 
  the effects of various types of noise, assess the performance of different error correction 
  codes, and evaluate the robustness of quantum algorithms to errors.

QEP is based on [QCompute](https://quantum-hub.baidu.com/opensource), 
which a Python-based open-source quantum computing platform SDK also developed 
by [Institute for Quantum Computing](https://quantum.baidu.com). 
It provides a full-stack programming experience for senior users via hybrid quantum programming language 
features and high-performance simulators. 
You can install QCompute via [pypi](https://pypi.org/project/qcompute/). 
When you install QEP, QCompute will be automatically installed. 
Please refer to QCompute's official [Open Source](https://quantum-hub.baidu.com/opensource) page for more details.

## Installation

### Install QEP

The package QEP is compatible with 64-bit Python 3.8 and 3.9, on Linux, MacOS (10.14 or later) and Windows. We highly recommend the users to install QEP via `pip`. Open the Terminal and run

```bash
pip install qcompute-qep
```

This will install the QEP binaries as well as the QEP package. For those using an older version of QEP, keep up to date by installing with the `--upgrade` flag for additional features and bug fixes.

### Run Examples

After installation, you can try the following simple program to check whether QEP has been successfully installed.

```python
from QCompute import *
import qcompute_qep.tomography as tomography


# Step 1. Initialize a quantum program for preparing the Bell state
qp = QEnv()  # qp is short for "quantum program", instance of QProgram
qp.Q.createList(2)
H(qp.Q[0])
CX(qp.Q[0], qp.Q[1])

# Step 2. Set the quantum computer (instance of QComputer).
# For debugging on ideal simulator, change qc to BackendName.LocalBaiduSim2
qc = BackendName.LocalBaiduSim2

# For test on real quantum hardware, change qc to BackendName.CloudBaiduQPUQian
# You must set your VIP token in order to access the hardware
# Define.hubToken = "Token"
# qc = BackendName.CloudBaiduQPUQian

# Step 3. Perform Quantum State Tomography, check how well the Bell state is prepared.
st = tomography.StateTomography()

# Alternatively, you may initialize the StateTomography instance as follows:
# st = StateTomography(qp, qc, method='inverse', shots=4096)

# Call the tomography procedure and obtain the noisy quantum state
st.fit(qp, qc, method='inverse', shots=4096)

print("***********************************************************************")
print("Testing whether 'qcompute-qep' is successfully installed or not now ...\n")

print('Fidelity of the Bell state is: F = {:.5f}'.format(st.fidelity))
print("Please change 'qc' to other quantum computers for more tests.\n")

print("Package 'qcompute-qep' is successfully installed! Please enjoy!")
print("***********************************************************************")
```

Note that more examples are provided in the [API documentation](https://quantum-hub.baidu.com/docs/qep/), 
the [QEP Tutorials](https://quantum-hub.baidu.com/qep/), and the source 
file of QEP hosted in [GitHub](https://github.com/baidu/QCompute/tree/master/Extensions/QuantumErrorProcessing/).
You can get started from there.

## Tutorials

QEP provides detailed and comprehensive tutorials for performance evaluation, quantum error 
characterization, quantum error mitigation, and quantum error correction, 
ranging from theoretical analysis to practical application. 
We recommend the interested researchers or developers to download the Jupyter Notebooks and try it. 
The tutorials are listed as follows:

+ **Performance Evaluation**

  + [Standard Randomized Benchmarking](https://quantum-hub.baidu.com/qep/tutorial-standardrb)
  + [Interleaved Randomized Benchmarking](https://quantum-hub.baidu.com/qep/tutorial-interleavedrb)
  + [Cross-Entropy Benchmarking](https://quantum-hub.baidu.com/qep/tutorial-xeb)
  + [Unitarity Randomized Benchmarking](https://quantum-hub.baidu.com/qep/tutorial-unitarityrb)
  + [Direct Fidelity Estimation of Quantum States](https://quantum-hub.baidu.com/qep/tutorial-dfe-state)
  + [Direct Fidelity Estimation of Quantum Processes](https://quantum-hub.baidu.com/qep/tutorial-dfe-process)
  + [Cross-Platform Estimation of Quantum States](https://quantum-hub.baidu.com/qep/tutorial-cpe-state)

+ **Quantum Error Characterization**

  + [Quantum State Tomography](https://quantum-hub.baidu.com/qep/tutorial-qst)
  + [Quantum Process Tomography](https://quantum-hub.baidu.com/qep/tutorial-qpt)
  + [Quantum Detector Tomography](https://quantum-hub.baidu.com/qep/tutorial-qdt)
  + [Quantum Gateset Tomography](https://quantum-hub.baidu.com/qep/tutorial-gst)
  + [Spectral Quantum Tomography](https://quantum-hub.baidu.com/qep/tutorial-sqt)

+ **Quantum Error Mitigation**

  + [Zero-Noise Extrapolation](https://quantum-hub.baidu.com/qep/tutorial-zne)
  + [Measurement Error Mitigation](https://quantum-hub.baidu.com/qep/tutorial-mem)
  + [Applications of Measurement Error Mitigation](https://quantum-hub.baidu.com/qep/tutorial-mem-applications)

+ **Quantum Error Correction**
  
  + [Stabilizer Code Simulator](https://quantum-hub.baidu.com/qep/tutorial-qec-simulator)

More tutorials and demonstrations will be included in the future release.

## API Documentation

For those who are looking for explanation on the python classes and functions in QEP, 
please refer to our [API documentation](https://quantum-hub.baidu.com/docs/qep/).

## Feedbacks

Users are encouraged to contact us via email quantum@baidu.com with general questions, 
unfixed bugs, and potential improvements. We hope to make QEP better together with the community!

## Research based on QEP

We encourage researchers and developers to use QEP to explore quantum error processing. 
If your work uses QEP, please feel free to send us a notice via quantum@baidu.com and 
cite us with the following BibTeX:

```BibTex
@misc{QEP, 
    author={{Baidu Quantum}},
    title={{Quantum Error Processing Toolkit (QEP)}}, 
    year={2022}, 
    url={https://quantum-hub.baidu.com/qep/}, 
    version={1.1.0}
}
```

## Copyright and License

QEP uses [Apache-2.0 license](https://github.com/baidu/QCompute/blob/master/LICENSE).
