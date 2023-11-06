# Measurement-based quantum computation

*Copyright (c) 2022 Institute for Quantum Computing, Baidu Inc. All Rights Reserved.*

## Introduction

Quantum computation utilizes the peculiar laws of the quantum world and provides us with a novel and promising way of information processing. The essence of quantum computation is to evolve the initially prepared quantum state into another expected one, and then make measurements on it to obtain the required classical results. However, the approaches of quantum state evolution are varied in different computation models. The widely used quantum circuit model [1,2] completes the evolution by performing quantum gate operations, which can be regarded as a quantum analog of the classical computing model. In contrast, measurement-based quantum computation (MBQC) provides a completely different approach for quantum computing.

As its name suggests, the entire evolution in MBQC is completed via quantum measurements. There are mainly two variants of measurement-based quantum computation in the literature: teleportation-based quantum computing (TQC) model [3-5] and one-way quantum computer (1WQC) model [6-9]. The former requires joint measurements on multiple qubits, while the latter only requires single-qubit measurements. After these two variants were proposed, they were proved to be highly correlated and admit a one-to-one correspondence [10]. So without further declaration, all of the following discussions about MBQC will refer to the 1WQC model.

MBQC is a unique model in quantum computation and has no classical analog. The model controls the computation by measuring part of the qubits of an entangled state, with those remaining unmeasured undergoing the evolution correspondingly. By controlling measurements, we can complete any desired evolution. The computation in MBQC is mainly divided into three steps. The first step is to prepare a resource state, which is a highly entangled many-body quantum state. This state can be prepared offline and can be independent of specific computational tasks. The second step is to sequentially perform single-qubit measurements on each qubit of the prepared resource state, where subsequent measurements can depend on previous measurement outcomes, that is, measurements can be adaptive. The third step is to perform byproduct corrections on the final state. Finally, we do classical data processing on measurement outcomes to obtain the required computation results. 

A typical example of MBQC algorithms is shown in Figure 1. The grid represents a commonly used quantum resource state (called cluster state, see below for details). Each vertex on the grid represents a qubit, while the entire grid represents a highly entangled quantum state. We measure each qubit one by one in a specific measurement basis (In the vertices, X, Y, Z, XY, etc., represent the corresponding measurement basis), and then perform byproduct corrections (to eliminate the effect of Pauli X and Pauli Z operators), to complete the computation.

![MBQC example](./figures/mbqc-fig-general_pattern.jpg "Figure 1: A typical example of MBQC algorithm where computation is proceeded by measuring each qubit on the vertex.")

The "three-step" process of MBQC has brought us quantities of benefits. For example, if the quantum state prepared in the first step is too noisy, we can simply discard this state before computation begins (that is, before any measurement is implemented), and prepare it again to ensure the accuracy of the computational results. Since the resource state can be prepared offline and independent of specific computing tasks, it can also be applied to secure delegated quantum computation [11,12] to protect clients' privacy. In addition, single-qubit measurement is easier to be implemented in practice than quantum gates. Non-adaptive quantum measurements can even be carried out simultaneously, thereby, reducing the computation depth and requiring less coherence time of the quantum system. The difficulty of realizing MBQC mainly lies in resource state preparation in the first step. Such a quantum state is highly entangled and the number of qubits required is much larger than that of the usual circuit model. For recent progress on the resource state preparation, please refer to [13,14]. The following table briefly summarizes both advantages and limitations of MBQC and quantum circuit models.

|    | Quantum circuit model     | MBQC model    |
|:---: | :---: | :---: |
| Pros|  has classical analog; easy to understand; and develop applications | resource state can be prepared offline; easy to implement single-qubit measurement; measurements can be implemented simultaneously; leading to lower implementation depth |
|Cons| implementation order fixed; depth restricted by coherence time| no classical analog thus super-intuitive; resource state requires a large number of qubits; thus hard to prepare in practice| 


## Prerequisites

Before introducing the MBQC module in more detail, let's briefly review the two building blocks of MBQC.

### 1. Graph and graph state
    
Given a graph $G=(V, E)$ with vertices set $V$ and edges set $E$, we can prepare an entangled quantum state by initializing a plus state $|+\rangle = (|0\rangle + |1\rangle) / \sqrt{2}$ to each vertex of $G$ and performing a control Z operation $CZ = |0\rangle\langle 0| \otimes I + |1\rangle\langle1|\otimes Z$ between each connected qubit pair. The resulting quantum state is called the graph state of $G$, denoted by $|G\rangle$, such that:
    
$$
|G\rangle = \prod_{(a,b) \in E} CZ_{ab} \left(\bigotimes_{v \in V}|+\rangle_v\right). \tag{1}
$$

The concept of graph state is nothing particular. Actually, the well-known Bell state and GHZ state are both graph states up to local unitary transformations. Besides, if the underlying graph we consider is a 2D grid then the corresponding graph state is called cluster state, depicted in Figure 2.

![Graph states](./figures/mbqc-fig-graph_states.jpg "Figure 2：(i) The graph of a Bell state; (ii) The graph of a 4-qubit GHZ state; (iii) The graph of a cluster state")

### 2. Projective measurement

Quantum measurement is one of the main concepts in quantum information processing. In the circuit model, measurements are performed usually at the end of the circuit to extract classical results from the quantum state. However, in MBQC, quantum measurements are also used to drive the computation. In the MBQC model, we use single-qubit measurements by default and mainly use 0/1 projection measurement. According to Born's rule [17], given a projective measurement basis $\{|\psi_0\rangle, |\psi_1\rangle\}$ and a quantum state $|\phi\rangle$, the probability that the outcome $s \in \{0,1\}$ occurs is given by $p(s) = |\langle \psi_s|\phi\rangle|^2$, and the corresponding post-measurement state is $| \psi_s\rangle\langle\psi_s|\phi\rangle / \sqrt{p(s)}$. In other words, the state of the measured qubit collapses into $|\psi_s\rangle$, while the state of other qubits evolves to $\langle\psi_s|\phi\rangle / \sqrt{p(s)}$.

Single-qubit measurements are commonly used, especially the binary projective measurements on the $XY$, $YZ$ and $XZ$ planes, defined respectively by the following orthonormal bases,

- XY-plane measurement: $M^{XY}(\theta) = \{R_z(\theta) |+\rangle, R_z(\theta) |-\rangle \}$, reducing to $X$ measurement if $\theta = 0$ and $Y$ measurement if $\theta = \frac{\pi}{2}$;

- YZ-plane measurement: $M^{YZ}(\theta) = \{R_x(\theta)|0\rangle, R_x(\theta)|1\rangle\}$, reducing to $Z$ measurement if $\theta = 0$;

- XZ-plane measurement: $M^{XZ}(\theta) = \{R_y(\theta)|0\rangle, R_y(\theta)|1\rangle\}$, reducing to $Z$ measurement if $\theta = 0$.

In the above definitions, we use $|+\rangle = (|0\rangle + |1\rangle)/ \sqrt{2},|-\rangle = (|0\rangle - |1\rangle)/ \sqrt{2}$, and $R_x, R_y, R_z$ are rotation gates around $x,y,z$ axes respectively.

## MBQC module


### 1. Model and code implementation

#### "Three-step" process

As is mentioned above, MBQC is different from the quantum circuit model. The computation in MBQC is driven by measuring each qubit on a graph state. To be specific, the MBQC model consists of the following three steps.

- **Graph state preparation**: that is, to prepare a many-body entangled state. Given vertices and edges in a graph, we can prepare a graph state by initializing a plus state on each vertex and performing a control Z operation between each connected qubit pair. Since a graph state and its underlying graph have a one-to-one correspondence, it suffices to work with the graph only. In addition, we can selectively replace some of the plus states in the graph with a customized input state if necessary.

- **Single-qubit measurement**: that is, to perform single-qubit measurements on the prepared graph state with specific measurement bases. The measurement angles can be adaptively adjusted according to previous outcomes. Non-adaptive measurements commute with each other in simulation and can even be performed simultaneously in experiments. 

- **Byproduct correction**: Due to the random nature of quantum measurement, the evolution of the unmeasured quantum state cannot be uniquely determined. In other words, the unmeasured quantum state may undergo some extra evolutions, called byproducts. So the last step is to correct these to obtain the expected result. If the final output is not a quantum state but the measurement outcomes, it suffices to eliminate the effect of byproducts via classical data processing only.

In conclusion, the "three-step" process of MBQC includes graph state preparation, single-qubit measurement, and byproduct correction. The first two steps are indispensable while the implementation of the third step depends on the form of expected results.

#### Measurement pattern and "EMC" language

Besides the "three-step" process, an MBQC model can also be described by the "EMC" language from the measurement calculus [18]. As is mentioned above, MBQC admits a one-to-one correspondence to the circuit model. We can usually call the MBQC equivalent of a quantum circuit as a measurement pattern while the equivalent of a specific gate/measurement is called a subpattern [18]. In the "EMC" language, we usually call an entanglement operation "an entanglement command", denoted by "E"; call a measurement operation "a measurement command", denoted by "M"; call a byproduct correction operation "a byproduct correction command", denoted by "C". Therefore, in parallel with the"three-step" process, MBQC is also characterized by an "EMC" command list. The process of computation is to execute commands in the command list in order. To familiarize ourselves with MBQC quickly, we will adopt the conventional "three-step" process to describe MBQC in this tutorial. It should be noted that the "three-step" process and the "EMC" language are essentially the same things with different denotations.

#### Code implementation

In terms of code implementation, the most important part of the MBQC module is a class `MBQC` with attributes and methods necessary for MBQC simulation. We can instantiate an MBQC class and call the class methods step by step to complete the MBQC computation process. Here, we briefly introduce some frequently used methods and their functionalities. Please refer to the API documentation for details.

|MBQC class method|Functionality|
|:---:|:---:|
|`set_graph`|input a graph for MBQC|
|`set_pattern`|input a measurement pattern for MBQC|
|`set_input_state`|input initial quantum state|
|`draw_process`|draw the dymanical process of MBQC computation|
|`track_progress`|track the running progress of MBQC computation|
|`measure`|perform single-qubit measurement|
|`sum_outcomes`|sum outcomes of the measured qubits|
|`correct_byproduct`|correct byproduct operators|
|`run_pattern`|run the input measurement pattern|
|`get_classical_output`|return classical results|
|`get_quantum_output`|return quantum results|

In the MBQC module, we provide two simulation modes, "graph" and "pattern", corresponding to the two equivalent descriptions of the MBQC computation process respectively. If we set a graph, the whole computation needs to follow the "three-step" process. It is worth mentioning that we design a vertex dynamic classification algorithm to simulate the MBQC computation process efficiently. Roughly speaking, we integrate the first two steps of the process, change the execution order of entanglement and measurement operations automatically to reduce the number of effective qubits involved in the computation and thereby improve the efficiency. The outline to use the simulation module is as follows:

```python
"""
MBQC module usage (set a graph and proceed with the "three-step" process)
"""
from Extensions.QuantumNetwork.qcompute_qnet.quantum.backends.mbqc import MBQC

# Instantiate MBQC and create an MBQC model
mbqc = MBQC()

# First step of the "three-step" process, set a graph
mbqc.set_graph(graph)

# Set an initial input state (optional)
mbqc.set_input_state(input_state)

# Second step of the "three-step" process, perform single-qubit measurements
mbqc.measure(which_qubit, basis)
mbqc.measure(which_qubit, basis)
...

# Third step of the "three-step" process, correct byproducts
mbqc.correct_byproduct(gate, which_qubit, power)

# Obtain the classical and quantum outputs
classical_output = mbqc.get_classical_output()
quantum_output = mbqc.get_quantum_output()
```

If we set a pattern to the ``MBQC`` class, we need to call the `run_pattern` method to complete the simulation.

```python
"""
MBQC module usage (set a pattern and simulate by "EMC" commands)
"""
from Extensions.QuantumNetwork.qcompute_qnet.quantum.backends.mbqc import MBQC

# Instantiate MBQC and create an MBQC model
mbqc = MBQC()

# Set a measurement pattern
mbqc.set_pattern(pattern)

# Set an initial input state (optional) 
mbqc.set_input_state(input_state)

# Run the measurement pattern
mbqc.run_pattern()

# Obtain the classical and quantum outputs
classical_output = mbqc.get_classical_output()
quantum_output = mbqc.get_quantum_output()
```

After the above introduction, I am sure you already have a basic understanding of MBQC and our simulation module. Now, let's do some exercises with the following two examples!

### 2. Example: general single-qubit unitary gate in MBQC

For a general single-qubit unitary gate $U$, it can be decomposed to $ U = R_x(\gamma)R_z(\beta)R_x(\alpha)$ up to a global phase [17]. In MBQC, this unitary gate can be realized in the following way [15]. As shown in Figure 3: prepare five qubits, with input on the leftmost qubit while output on the rightmost qubit; input a state $|\psi\rangle$ and initialize other qubits with $|+\rangle$; apply a $CZ$ operation to each connected qubit pair; perform $X$-measurement on the first qubit and adaptive measurements in the $XY$-plane on the middle three qubits, with the four measured qubits' outcomes recorded as $s_1$, $s_2$, $s_3$, $s_4$; correct byproducts to the state on qubit $5$ after all measurements. Then, the output state on qubit 5 will be $U|\psi\rangle$.


![Single qubit pattern](./figures/mbqc-fig-single_qubit_pattern.jpg "Figure 3: Realizing a general single-qubit unitary gate in MBQC")


**Note**: after measuring the first four qubits, state on qubit $5$ has the form of $X^{s_2 + s_4}Z^{s_1 + s_3} U|\psi\rangle$, where $X^{s_2 + s_4}$ and $Z^{s_1 + s_3}$ are the so-called byproducts. We need to correct them according to the measurement outcomes to get the desired state of $U|\psi\rangle$.

Here is the code implementation:

####  Import relevant modules

```python
from numpy import pi, random

from Extensions.QuantumNetwork.qcompute_qnet.quantum.backends.mbqc import MBQC
from Extensions.QuantumNetwork.qcompute_qnet.quantum.basis import Basis
from Extensions.QuantumNetwork.qcompute_qnet.quantum.gate import Gate
from Extensions.QuantumNetwork.qcompute_qnet.quantum.state import PureState
```

#### Set graph and state

Then, we can set the graph on our own. For this instance in Figure 3, we need five vertices (recorded as `['1', '2', '3', '4', '5']`) and four edges (recorded as  (`[('1', '2'), ('2', '3'), ('3', '4'), ('4', '5')]`)). We need to set an input the state on vertex `'1'` and initialize measurement angles.


```python
# Construct the underlying graph
V = ['1', '2', '3', '4', '5']
E = [('1', '2'), ('2', '3'), ('3', '4'), ('4', '5')]
G = [V, E]
# Generate a random state vector
input_vec = PureState.random_state_vector(1, is_real=False)

# Initialize measurement angles
alpha = pi * random.uniform()
beta = pi * random.uniform()
gamma = pi * random.uniform()
```

####  Instantiate an MBQC model

Then we can construct our own MBQC model by instantiating the class `MBQC` and setting the graph and input state.


```python
# Instantiate MBQC
mbqc = MBQC()
# Set the graph
mbqc.set_graph(G)
# Set the input state
mbqc.set_input_state(PureState(input_vec, ['1']))
```

Then, we perform measurements on the first four vertices.

####  Measure the first vertex

As shown in Figure 3, we perform $X$-measurement on the first vertex, that is, the measurement in $XY$-plane with an angle of $\theta_1 = 0$。 


```python
# Calculate the angle for the first measurement
theta_1 = 0
# Measure the first vertex
mbqc.measure('1', Basis.Plane('XY', theta_1))
```

Measurement on the first vertex is straightforward because it is not adaptive. However, things will be tougher for the second, third, and fourth vertices, as the measurement angles are set adaptively according to the previous measurement outcomes.

####  Measure the second vertex

As shown in Figure 3, the measurement on the second vertex has a form of $M^{XY}(\theta_2)$, where

$$
\theta_2 = (-1)^{s_1 + 1} \alpha, \tag{2}
$$

This is a measurement in the $XY$-plane with an adaptive angle $(-1)^{s_1 + 1} \alpha$, where $s_1$ is the outcome of the first vertex. 

There is a method `sum_outcomes` in the class `MBQC` to calculate the summation of outcomes for vertices in the first argument. If we want to add an extra value "$x$" on top of the summation, we can set the second argument to be $x$. Otherwise, the default value of the second argument is $0$.


```python
# Calculate the angle for the second measurement
theta_2 = (-1) ** mbqc.sum_outcomes(['1'], 1) * alpha
# Measure the second vertex
mbqc.measure('2', Basis.Plane('XY', theta_2))
```

####  Measure the third vertex

As shown in Figure 3, the measurement on the third vertex has a form of $M^{XY}(\theta_3)$, where

$$
\theta_3 = (-1)^{s_2 + 1} \beta, \tag{3}
$$

This is a measurement in the $XY$-plane with an adaptive angle $(-1)^{s_2 + 1} \beta$, where $s_2$ is the outcome of the second vertex.


```python
# Calculate the angle for the third measurement
theta_3 = (-1) ** mbqc.sum_outcomes(['2'], 1) * beta
# Measure the third vertex
mbqc.measure('3', Basis.Plane('XY', theta_3))
```

#### Measure the fourth vertex

As shown in Figure 3, the measurement on the fourth vertex has a form of $M^{XY}(\theta_4)$, where

$$
\theta_4 = (-1)^{s_1 + s_3 + 1} \gamma, \tag{4}
$$

This is a measurement in the $XY$-plane with an adaptive angle $(-1)^{s_1 + s_3 + 1} \gamma$, where $s_1$ and $s_3$ are respectively the outcomes of the first and the third vertices.


```python
# Calculate the angle for the fourth measurement
theta_4 = (-1) ** mbqc.sum_outcomes(['1', '3'], 1) * gamma
# Measure the fourth vertex
mbqc.measure('4', Basis.Plane('XY', theta_4))
```

####  Correct byproducts on the fifth vertex

After measurements on the first four vertices, the state on the fifth vertex is not exactly $U|\psi\rangle$, but a state with byproducts $X^{s_2 + s_4}Z^{s_1 + s_3} U|\psi\rangle$. To obtain the desired $U|\psi\rangle$, we must correct byproducts on the fifth vertex.


```python
# Correct byproducts on the fifth vertex
mbqc.correct_byproduct('X', '5', mbqc.sum_outcomes(['2', '4']))
mbqc.correct_byproduct('Z', '5', mbqc.sum_outcomes(['1', '3']))
```

#### Obtain the final output state and compare it with the expected one

We can call `get_classical_output` and `get_quantum_output` to obtain the classical and quantum outputs after simulation. 

```python
# Obtain the quantum result
state_out = mbqc.get_quantum_output()

# Compute the expected state vector
vec_std = Gate.Rx(gamma) @ Gate.Rz(beta) @ Gate.Rx(alpha) @ input_vec
# Construct the expected state on vertex '5'
state_std = PureState(vec_std, ['5'])

# Compare with the expected state
print(state_out.compare_by_vector(state_std))
```

### 3. Example: CNOT gate in MBQC

The CNOT gate is one of the most frequently used gates in the circuit model. In MBQC, the realization of a CNOT gate is shown in Figure 4 [7]: prepare $15$ qubits, with $1$, $9$ being the input qubits and $7$, $15$ being the output qubits; input a state $|\psi\rangle$ and initialize other vertices to $|+\rangle$; apply a CZ operator to each connected qubit pairs; perform $X$-measurements on the vertices $1, 9, 10, 11, 13, 14$ and $Y$-measurement on the vertices $2, 3, 4, 5, 6, 8, 12$ (Note: All of these measurements are non-adaptive measurements, so the order of their executions can be permuted); correct byproducts on $7$ and $15$ to obtain the output state $\text{CNOT}|\psi\rangle$.

![CNOT pattern](./figures/mbqc-fig-cnot_pattern.jpg "Figure 4: Realization of CNOT gate in MBQC")

**Note**: Similar to the first example, byproduct corrections are necessary to get the desired $\text{CNOT}|\psi\rangle$.

Here is a complete code implementation:


```python
from numpy import pi, random

from Extensions.QuantumNetwork.qcompute_qnet.quantum.backends.mbqc import MBQC
from Extensions.QuantumNetwork.qcompute_qnet.quantum.basis import Basis
from Extensions.QuantumNetwork.qcompute_qnet.quantum.gate import Gate
from Extensions.QuantumNetwork.qcompute_qnet.quantum.state import PureState

# Define X, Y measurement bases
X_basis = Basis.X()
Y_basis = Basis.Y()

# Define the underlying graph for computation
V = [str(i) for i in range(1, 16)]
E = [('1', '2'), ('2', '3'), ('3', '4'), ('4', '5'),
     ('5', '6'), ('6', '7'), ('4', '8'), ('8', '12'),
     ('9', '10'), ('10', '11'), ('11', '12'),
     ('12', '13'), ('13', '14'), ('14', '15')]
G = [V, E]

# Generate a random state vector
input_psi = PureState.random_state_vector(2, is_real=True)

# Instantiate a MBQC class
mbqc = MBQC()
# Set the graph state
mbqc.set_graph(G)
# Set the input state
mbqc.set_input_state(PureState(input_psi, ['1', '9']))

# Measure each qubit step by step
mbqc.measure('1', X_basis)
mbqc.measure('2', Y_basis)
mbqc.measure('3', Y_basis)
mbqc.measure('4', Y_basis)
mbqc.measure('5', Y_basis)
mbqc.measure('6', Y_basis)
mbqc.measure('8', Y_basis)
mbqc.measure('9', X_basis)
mbqc.measure('10', X_basis)
mbqc.measure('11', X_basis)
mbqc.measure('12', Y_basis)
mbqc.measure('13', X_basis)
mbqc.measure('14', X_basis)

# Compute the power of byproduct operators
cx = mbqc.sum_outcomes(['2', '3', '5', '6'])
tx = mbqc.sum_outcomes(['2', '3', '8', '10', '12', '14'])
cz = mbqc.sum_outcomes(['1', '3', '4', '5', '8', '9', '11'], 1)
tz = mbqc.sum_outcomes(['9', '11', '13'])

# Correct the byproduct operators
mbqc.correct_byproduct('X', '7', cx)
mbqc.correct_byproduct('X', '15', tx)
mbqc.correct_byproduct('Z', '7', cz)
mbqc.correct_byproduct('Z', '15', tz)

# Obtain the quantum result
state_out = mbqc.get_quantum_output()

# Construct the expected result
vec_std = Gate.CNOT() @ input_psi
state_std = PureState(vec_std, ['7', '15'])

# Compare with the expected result
print(state_out.compare_by_vector(state_std))
```

---

## References

[1] Deutsch, David Elieser. "Quantum computational networks." [Proceedings of the Royal Society of London. A. 425.1868 (1989): 73-90.](https://royalsocietypublishing.org/doi/abs/10.1098/rspa.1989.0099)

[2] Barenco, Adriano, et al. "Elementary gates for quantum computation." [Physical Review A 52.5 (1995): 3457.](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.52.3457)

[3] Gottesman, Daniel, and Isaac L. Chuang. "Demonstrating the viability of universal quantum computation using teleportation and single-qubit operations." [Nature 402.6760 (1999): 390-393.](https://www.nature.com/articles/46503?__hstc=13887208.d9c6f9c40e1956d463f0af8da73a29a7.1475020800048.1475020800050.1475020800051.2&__hssc=13887208.1.1475020800051&__hsfp=1773666937)

[4] Nielsen, Michael A. "Quantum computation by measurement and quantum memory." [Physics Letters A 308.2-3 (2003): 96-100.](https://www.sciencedirect.com/science/article/abs/pii/S0375960102018030)

[5] Leung, Debbie W. "Quantum computation by measurements." [International Journal of Quantum Information 2.01 (2004): 33-43.](https://www.worldscientific.com/doi/abs/10.1142/S0219749904000055)

[6] Robert Raussendorf, et al. "A one-way quantum computer." [Physical Review Letters 86.22 (2001): 5188.](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.86.5188)

[7] Raussendorf, Robert, and Hans J. Briegel. "Computational model underlying the one-way quantum computer." [Quantum Information & Computation 2.6 (2002): 443-486.](https://dl.acm.org/doi/abs/10.5555/2011492.2011495)

[8] Robert Raussendorf, et al. "Measurement-based quantum computation on cluster states." [Physical Review A 68.2 (2003): 022312.](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.68.022312)

[9] Briegel, Hans J., et al. "Measurement-based quantum computation." [Nature Physics 5.1 (2009): 19-26.](https://www.nature.com/articles/nphys1157)

[10] Aliferis, Panos, and Debbie W. Leung. "Computation by measurements: a unifying picture." [Physical Review A 70.6 (2004): 062314.](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.70.062314)

[11] Broadbent, Anne, et al. "Universal blind quantum computation." [2009 50th Annual IEEE Symposium on Foundations of Computer Science. IEEE, 2009.](https://arxiv.org/abs/0807.4154)

[12] Morimae, Tomoyuki. "Verification for measurement-only blind quantum computing." [Physical Review A 89.6 (2014): 060302.](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.89.060302)

[13] Larsen, Mikkel V., et al. "Deterministic generation of a two-dimensional cluster state." [Science 366.6463 (2019): 369-372.](https://science.sciencemag.org/content/366/6463/369)

[14] Asavanant, Warit, et al. "Generation of time-domain-multiplexed two-dimensional cluster state." [Science 366.6463 (2019): 373-376.](https://science.sciencemag.org/content/366/6463/373)

[15] Richard Jozsa, et al. "An introduction to measurement based quantum computation." [arXiv:quant-ph/0508124](https://arxiv.org/abs/quant-ph/0508124v2)

[16] Nielsen, Michael A. "Cluster-state quantum computation." [Reports on Mathematical Physics 57.1 (2006): 147-161.](https://www.sciencedirect.com/science/article/abs/pii/S0034487706800145)

[17] Nielsen, Michael A., and Isaac Chuang. "Quantum computation and quantum information."[Cambridge University Press (2010).](https://www.cambridge.org/core/books/quantum-computation-and-quantum-information/01E10196D0A682A6AEFFEA52D53BE9AE)

[18] Danos, Vincent, et al. "The measurement calculus." [Journal of the ACM (JACM) 54.2 (2007): 8-es.](https://dl.acm.org/doi/abs/10.1145/1219092.1219096)
