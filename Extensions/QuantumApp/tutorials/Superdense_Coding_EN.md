# Quantum Superdense Coding

*Copyright (c) 2021 Institute for Quantum Computing, Baidu Inc. All Rights Reserved.*

> If you run this tutorial with cloud computing power, you will consume about 1 Quantum-hub points.

## Background

The coding and operation of classical computers are based on Boolean logic algebra, using binary coding, and a single bit is a binary number of 0 and 1, which means one classical bit can encode one bit of information at most. However, with the discovery of quantum entanglement and the introduction of Bell states, Bennett and Wiesner proposed a quantum coding protocol in 1992 [1]. This protocol uses the entangled nature of qubits to transmit the information of two classical bits through one qubit, thereby realizing a communication method with larger capacity and higher efficiency. This way of using quantum entanglement to achieve high-capacity coding is called quantum superdense coding, and it is an important application of quantum mechanics in coding.

It should be highlighted that two methods of quantum superdense coding and quantum teleportation should not be confused. Here we will make a simple comparison:

- Quantum superdense coding transmits two classical bits of information through one qubit.
- Quantum teleportation transmits one of qubit information through two classical bits of information.

At the same time, both protocols use quantum entanglement as a resource.

## Quantum superdense coding protocol

Like quantum teleportation, the basic assumption of quantum superdense coding is that Alice and Bob share a pair of Bell states. The difference is that they share a quantum channel instead of a classical channel. How to use only one qubit to transmit the information of two classical bits $00,01,10,11$? We could use the following method: Alice and Bob share a pair of Bell states prepared in advance, and according to the classical information that needs to be transmitted, Alice performs corresponding encoding operations on the qubits she holds, and then transmits the operated qubits to Bob through the quantum channel. Bob makes a joint measurement on the received qubits and the previously held qubits to decode the classical information that Alice wants to transmit. Experimentally, the quantum superdense coding protocol was verified by Mattle et al. through photon polarization experiments in 1996 [2].

The detailed steps are as follows:

**Step 1: Preparation**

Firstly, before separating, Alice and Bob need to prepare a Bell state $|\Phi^+\rangle=\frac{1}{\sqrt{2}}(|00\rangle+|11\rangle)$, and take one of the qubits.

**Step 2: Alice applies encoding operations and transmits qubits**

Depending on the classic bit information that Alice wants to send to Bob, Alice operates on the bits she holds according to the following table.

| Information | Alice operation|
| :------: | :------: |
| 00 | I |
| 01 | Z |
| 10 | X |
| 11 | ZX |

<div style="text-align:center">Table 1: Table for quantum superdense coding. </div>

Suppose Alice wants to send the classical bit $11$, she only needs to act on her qubit $X$ gate first, and then act on the $Z$ gate:

$$
|\Phi^+\rangle_{AB} \stackrel{X_A}{\longrightarrow} \frac{1}{\sqrt{2}}(|10\rangle_{AB}+|01\rangle_{AB}) \stackrel{Z_A}{\longrightarrow} \frac{1}{\sqrt{2}}(|01\rangle_{AB}-|10\rangle_{AB}) = |\psi\rangle_{AB}, \tag{1}
$$

Then, Alice sends the qubits she manipulated to Bob through the shared quantum channel.

**Step 3: Bob performs the decoding operation**

After Bob receives the qubits transmitted by Alice, he obtains the quantum state describing the entire system |\psi\rangle_{AB}. Bob uses $A$ as the control bit to act on the CNOT gate, and then performs the $H$ gate operation on the qubit $A$. Finally, the computational basis $\{|00\rangle,|01\rangle,|10\rangle,|11\rangle\}$ is measured for the two systems to decode the classic information that Alice wants to transmit:

$$
|\psi\rangle_{AB} \stackrel{\text{CNOT}}{\longrightarrow} \frac{1}{\sqrt{2}}(|01\rangle_{AB}-|11\rangle_{AB}) \stackrel{H_A}{\longrightarrow} |11\rangle \tag{2}
$$

Note that Bob's measurement result can only be $11$ at this time. This completely decodes the information that Alice wants to transmit. In the same way, if Alice's encoding in step 2 is otherwise, the same operation can help Bob decode completely. We can verify against the following table.

| Information | Alice's operation | Bob's received state | After Bob's operation | Decoded information |
| :-----: | :----: | :----: | :----: | :----: |
| 00 | $I$ | $\frac{1}{\sqrt{2}}$ (&#124;00$\rangle$ + &#124;11$\rangle$) | &#124;00$\rangle$ | 00 |
| 01 | $Z$ | $\frac{1}{\sqrt{2}}$ (&#124;00$\rangle$ - &#124;11$\rangle)$ | &#124;01$\rangle$ | 01 |
| 10 | $X$ | $\frac{1}{\sqrt{2}}$ (&#124;01$\rangle$ + &#124;10$\rangle)$ | &#124;10$\rangle$ | 10 |
| 11 | $ZX$ | $\frac{1}{\sqrt{2}}$ (&#124;01$\rangle$ - &#124;10$\rangle)$ | &#124;11$\rangle$ | 11 |

<div style="text-align:center">Table 2: Encoding and decoding table for quantum superdense coding. </div>

Lastly, the complete circuit diagram of quantum superdense coding is as follows:

![Superdense-coding](figures/superdensecoding-fig-message.png "Figure 2: Circuit for transmit 'ij'.")

## Superdense coding with Quantum Leaf

After being familiar with the above quantum mechanics derivation, we can simulate the quantum superdense coding protocol on the Quantum Leaf platform. The specific example code is as follows:

```python
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
```

```
Shots 4096
Counts {'11': 4096}
State None
Seed 1019639350
```

The result obtained shows that the message $'11'$ that Alice wants to transmit was correctly decoded by Bob.

---

## Reference

[1] Bennett, Charles H., and Stephen J. Wiesner. "Communication via one-and two-particle operators on Einstein-Podolsky-Rosen states." [Physical Review Letters 69.20 (1992): 2881.](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.69.2881)

[2] Mattle, Klaus, et al. "Dense coding in experimental quantum communication." [Physical Review Letters 76.25 (1996): 4656.](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.76.4656)
