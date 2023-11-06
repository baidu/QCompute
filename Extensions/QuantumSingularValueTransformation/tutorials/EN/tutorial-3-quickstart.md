# Quick Start

*Copyright (c) 2022 Institute for Quantum Computing, Baidu Inc. All Rights Reserved.*

QSVT toolkit provides quantum circuits generation for Hamiltonian simulations, users could call these quantum circuits when using QCompute to create a quantum circuit. This page focuses on the usage of the Hamiltonian Simulation module to help users quickly get started with QSVT toolkit.

Detailed introduction for related theories refers to [following sections](https://quantum-hub.baidu.com/qsvt/tutorial-introduction); the implementation principle of this toolkit can also refer to its [API documentation](https://quantum-hub.baidu.com/docs/qsvt/).

## Demo

After the installation of QSVT toolkit, we can create a new python script file and enter the following code to initialize this demo.

```python{.line-numbers}
import numpy as np
from QCompute import QEnv, BackendName, MeasureZ
from Extensions.QuantumSingularValueTransformation.qcompute_qsvt.Application.HamiltonianSimulation.HamiltonianSimulation import circ_HS_QSVT
```

In the task of Hamiltonian simulation, we need to input the Hamiltonian to be simulated first. We use Hamiltonians' representations in Pauli basis, that is, to express the Hamiltonians in the form of linear combinations of multi-qubit Pauli matrices. Each term in the linear combinations is given by a double-tuple consisting of a float and a string, where the float represents the coefficient of this term, and the string encodes the information of the Pauli matrix. As an example, Hamiltonian

$$
H\otimes H=\frac12\left(X\otimes X+X\otimes Z+Z\otimes X+Z\otimes Z\right)
$$ 

is represented as a list

```python
list_str_Pauli_rep_HH = [(0.5, 'X0X1'), (0.5, 'X0Z1'), (0.5, 'Z0X1'), (0.5, 'Z0Z1')]
```

More complicatedly, the Hamiltonian of a hydrogen molecule $\operatorname{H_2}$ could be represented as:

```python
list_str_pauli_rep_H2 = [
    (-0.09706626861762556, 'I'), (-0.04530261550868938, 'X0X1Y2Y3'),
    (0.04530261550868938, 'X0Y1Y2X3'), (0.04530261550868938, 'Y0X1X2Y3'),
    (-0.04530261550868938, 'Y0Y1X2X3'), (0.1714128263940239, 'Z0'),
    (0.16868898168693286, 'Z0Z1'), (0.12062523481381837, 'Z0Z2'),
    (0.16592785032250773, 'Z0Z3'), (0.17141282639402394, 'Z1'),
    (0.16592785032250773, 'Z1Z2'), (0.12062523481381837, 'Z1Z3'),
    (-0.2234315367466399, 'Z2'), (0.17441287610651626, 'Z2Z3'),
    (-0.2234315367466399, 'Z3')]
```

We take `list_str_pauli_rep_HH` as an example, and  continue to show the following procedure. Also, we need to specify the number $n$ (@`num_qubit_sys`) of qubits involved in the system, the simulation time $\tau$ (@`float_tau`) and the precision $\epsilon$ (@`float_epsilon`) for the task, such as:

```python{.line-numbers}
num_qubit_sys = 2
float_tau = np.pi / 4
float_epsilon = 1e-5
```

Then we need to declare the quantum environment and quantum registers. Here we need to introduce the system register `reg_sys` corresponding to the Hamiltonian, several ancilla qubits `reg_blocking` to encode such Hamiltonian (where the number of ancilla qubits is related to the length of such Hamiltonian), and finally two ancilla qubits `reg_ancilla` to implement such Hamiltonian simulation. Since we are more accustomed to putting control qubits before controlled qubits, we reverse the order of the aforementioned quantum registers:

```python{.line-numbers}
# create the quantum environment, choose local backend
env = QEnv()
env.backend(BackendName.LocalBaiduSim2)

# the two ancilla qubit introduced from HS
reg_ancilla = [env.Q[0], env.Q[1]]
# compute the number of qubits needed in the block-encoding, and form a register
num_qubit_blocking = max(1, int(np.ceil(np.log2(len(list_str_Pauli_rep_HH)))))
reg_blocking = list(env.Q[idx] for idx in range(2, 2 + num_qubit_blocking))
# create the system register for the Hamiltonian
reg_sys = list(env.Q[idx] for idx in range(2 + num_qubit_blocking, 2 + num_qubit_blocking + num_qubit_sys))
```

Then we call the Hamiltonian simulation circuit, equivalent to operate the time evolution operator $e^{-iH\otimes H\tau}$ on the system register:

```python
circ_HS_QSVT(reg_sys, reg_blocking, reg_ancilla, list_str_Pauli_rep_HH, float_tau, float_epsilon)
```

We may introduce other quantum gates before or after calling this quantum circuit to realize more complicate algorithms. **However, it must be noted that all qubits in `reg_ancilla` and `reg_blocking` should be at state $|0\rangle$ before calling the Hamiltonian simulation circuit, otherwise we would not obtain such time evolution operator we wanted.**

Measure the final state and print the task result,

```python{.line-numbers}
# measure
MeasureZ(*env.Q.toListPair())
# commit
print(env.commit(8000, downloadResult=False)['counts'])
```

and then we may obtain

```python
{'100000': 1003, '000000': 5002, '010000': 960, '110000': 1035}
```

which indicates that all four zeroed ancilla qubits in `reg_ancilla` and `reg_blocking` remain state $|0\rangle$ really, and the population number of the system register is approximately in the ratio of $5:1:1:1$, which matches

$$
e^{-i\pi H\otimes H/4}|00\rangle=\frac{-i}{2\sqrt 2}\left((1+2i)|00\rangle+|01\rangle+|10\rangle+|11\rangle\right).
$$

Additionally, if users just want to experience Hamiltonian simulation a little, you may call the function `qcompute_qsvt.Application.HamiltonianSimulation.HamiltonianSimulation.func_HS_QSVT` to implement aforementioned procedure easily:

```python{.line-numbers}
import numpy as np
from Extensions.QuantumSingularValueTransformation.qcompute_qsvt.Application.HamiltonianSimulation.HamiltonianSimulation import func_HS_QSVT
print(func_HS_QSVT(list_str_Pauli_rep=[(0.5, 'X0X1'), (0.5, 'X0Z1'), (0.5, 'Z0X1'), (0.5, 'Z0Z1')], 
      num_qubit_sys=2, float_tau=np.pi / 4, float_epsilon = 1e-5, circ_output=False)['counts'])
```

Users may refer to the [source code](https://github.com/baidu/QCompute/tree/master/Extensions/QuantumSingularValueTransformation/qcompute_qsvt/Applications/HamiltonianSimulation/HamiltonianSimulation.py) to learn more information.

## More Examples

Users can download more [examples](https://github.com/baidu/QCompute/tree/master/Extensions/QuantumSingularValueTransformation/examples/) on [GitHub](https://github.com/baidu/QCompute/tree/master/Extensions/QuantumSingularValueTransformation/), where [**`example-HamiltonianSimulation.py`**](https://github.com/baidu/QCompute/tree/master/Extensions/QuantumSingularValueTransformation/examples/example-HamiltonianSimulation.py) includes two demo functions：

- `func_MS_test_QSVT` We test the output results of the five quantum circuits in turn to verify that the circuit implementation of the operator $e^{i\pi X\otimes X/4}$ is correct. Here precision `1e-5` is chosen, and if it fails to detect errors within $10000$ shots, then it prints `"MS test passed."`
- `func_HH_test_QSVT` We could also test the correctness for the operator $e^{-i\pi H\otimes H/4}$ in a similar way as `func_MS_test_QSVT`.

Here the time evolution operator $e^{i\pi X\otimes X/4}$ with Hamiltonian $X\otimes X$ and evolution time $-\pi/4$ is exactly the two-qubit native quantum gate $\operatorname{MS}$ gate in trapped ion quantum computing. Users may refer to [QuanlseTrappedIon](https://quanlse.baidu.com/#/doc/tutorial-general-MS-gate) for more details. Especially, we could prepare a superposition state

$$
\operatorname{MS}|00\rangle=\frac{1}{\sqrt 2}\left(|00\rangle + i|11\rangle\right).
$$

in only one step by operating an $\operatorname{MS}$ gate on ground state $|00\rangle$.

Moreover, users may add some post-processing by themselves to prove that the phase difference of the two components is indeed $\pi/2$ in $\operatorname{MS}|00\rangle$.
