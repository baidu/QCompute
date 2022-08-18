# Grover's Search Algorithm

## Background

A salesman wants to promote sales of a newly designed quantum computer by travelling among $n$ cities (New York $\rightarrow$ Shanghai $\rightarrow ...$) and finally travels back to where he starts. He really needs to find the shortest possible route to save his time. This is known as the [Travelling Salesman Porblem (TSP)](https://en.wikipedia.org/wiki/Travelling_salesman_problem) and in its essence an **unstructured** search problem. 

Mathematically, a type of these problems can be formulated as:

> Given a set $S = \{0, 1, \cdots, N-1 \}$ with $N$ elements, and a Boolean function $f : S \rightarrow \{0, 1\}$, the task is to find an element $x\in S$ satisfying $f(x) = 1$, which is called the solution for such search problem. It is assumed that such soltions are unique.

Here "unstructured" means that we have no prior knowledge about this dataset, e.g. the data is sorted by size, first letter, etc.

For such unstructured search problem afore defined, **classical search algorithms** can only enumerate elements $x$ in the set $S$ and check whether $f(x)=1$ one by one. Since the solution is unique, the complexities for classical search algorithms are all $\mathcal{O}(N)$ in the worst case.

However, there exists a **quantum algorithm**$^{[1]}$ named after *Lov Grover* in 1996 that can help us achieve a quadratic speed-up where $\mathcal{O}{(\sqrt{N})}$ queries are needed. For a set with $10^6$ elements, you only have to search a few thousand times.

---
## The Contents for Grover's Search Alogrithm

Briefly, we only demonstrate the workflow of the Grover's search algorithm here, rather than introduce its principle and complexity in detail. More theoretical details of it can be found in [1,2], or [QULEARN$^{[3]}$](https://qulearn.baidu.com/textbook/chapter3/%E6%A0%BC%E7%BD%97%E5%BC%97%E7%AE%97%E6%B3%95.html) in Chinese.

### Encoding of Data

Firstly, we need to specify the database we search on, denoted by $S=\{|0\rangle,|1\rangle,\cdots,|N-1\rangle\}$. The quantum system should have at least $n=\lceil \log_2 N\rceil$ qubits to containing $N$ quantum states pairwise orthogonal. Without loss of generality, we assume thate $N$ is a power of $2$, i.e.$N=2^n$.

### Encoding of Verifier

In order to take full advantage of the properties of quantum superposition and coherence, quantum verifier has also been designed better. As a classical verifier verifies the equation $f(x)=1$ to check whether $x$ is the search target after inputting $x$, a quantum verifier will be an evolution distinguishing search target and others after inputing a superposition state. **Grover oracle** is such a quantum verifier, defined as

$$
\hat O = I - 2|x^*\rangle\langle x^*|.    \tag{1}
$$

It is observed that there exists a relative phase $\pi$ between the search target $x^*$ and other states in computational basis.

### Amplitude Amplification

Furthermore in order to take full advantage of Grover oracle, Grover devise a quantum circuit, whose ideas are abstracted into amplitude amplification algorithm$^{[4,5]}$. Shortly to say, we introduce the **diffusion operator** $\hat D$, and call Grover oracle $\hat O$ and the diffusion operator $\hat D$ alternately, so that (the **absolutely value** of) the amplitude of the intermediate quantum state on the search target component $|x^*\rangle$ gradually enlarges, until it is almost $1$. Meanwhile the state of such quantum system could be considerred as the search target $|x^*\rangle$ approximately.

![The quantum circuit for Grover's search algorithm](./PIC/pipeline.png)

Let's take a simple example to demonstrate this process:

> **Problem:**   To obtain approxiamtely $|6\rangle$ from $S=\{|0\rangle,|1\rangle,\cdots,|7\rangle\}$.
**Supplying:** Grover oracle $\hat O = I - 2|6\rangle\langle 6|$.

It can be simply envisioned that we should start from a superposition state:

$$
|\psi_0\rangle=H^{\otimes 3}|0\rangle^{\otimes3}=\frac{1}{\sqrt 8}\sum_{j=0}^7|j\rangle=:\frac{1}{\sqrt 8}|6\rangle+\frac{\sqrt 7}{\sqrt 8}|6^\perp\rangle.\tag{2}
$$

Then we operate Grover oracle $\hat O$ on the quantum state $|\psi_0\rangle$, and obtain

$$
|\psi_1\rangle=\hat O|\psi_0\rangle=-\frac{1}{\sqrt 8}|6\rangle+\frac{\sqrt 7}{\sqrt 8}|6^\perp\rangle.\tag{3}
$$

Introduce the diffusion operator

$$
\hat D= I - 2|\psi_0\rangle\langle \psi_0|,\tag{4}
$$

operate it on $|\psi_1\rangle$, and we obtain

$$
|\psi_2\rangle=\hat D|\psi_1\rangle=-\frac{5}{2\sqrt 8}|6\rangle-\frac{1}{2\sqrt 8}|6^\perp\rangle.\tag{5}
$$

We find (**the absolutely value** of) the amplitude of $|\psi_2\rangle$ is larger than that of $|\psi_0\rangle$ on the target component $|6\rangle$, so this algorithm is called **amplitude amplification**. By further computation, we find the amplitude of

$$
|\psi_4\rangle=\hat D\hat O|\psi_2\rangle=\frac{11}{4\sqrt 8}|6\rangle-\frac{1}{4\sqrt 8}|6^\perp\rangle\tag{6}
$$

on the target component $|6\rangle$ enlarges futhermore. However, the amplitude of

$$
|\psi_6\rangle=\hat D\hat O|\psi_4\rangle=-\frac{13}{8\sqrt 8}|6\rangle+\frac{7}{8\sqrt 8}|6^\perp\rangle\tag{7}
$$

on the target component $|6\rangle$ becomes smaller than that of $|\psi_4\rangle$. After accurate calculation, we know that when the number of times of alternately calling Grover oracles and the diffusion operators is approximately

$$
\frac{\pi}{4\arcsin(1/\sqrt{2^n})} -\frac12,\tag{8}
$$

the amplitude on the target component $|6\rangle$ reaches the maximum. For $n=3$, the value of afore formula equals to $1.673$, so when we alternate $1$ or $2$ times, such amiplitude reaches maximum.

>Here $\hat D$ is called diffusion operator, becasue it maps each state $|j\rangle$ in computational basis into
>
>$$
>\hat D|j\rangle=|j\rangle-2|\psi_0\rangle\langle \psi_0|j\rangle=\frac{N-2}{N}|j\rangle-\sum_{k\ne j}\frac{2}{N}|k\rangle,\tag{9}
>$$
>
>where all components $|k\rangle$ except $|j\rangle$ recieve equal amounts of amplitude, as diffusing from $|j\rangle$ to $|k\rangle$. Since such diffusing amplitude will cancellate between positives and negatives, the amplitude amplification occurs on the target component $|6\rangle$.
---
## Code Demonstration and Explanation

### A Simple Sample of 3 Qubits
We continue to search $|6\rangle$ in $S=\{|0\rangle,|1\rangle,\cdots,|7\rangle\}$ as an example and give a simple code demonstration. The implement of quantum circuit refers to the python file `Grover_3qubit.py`.

- First, we initialize the three qubits as a superposition state;
    ```python{.line-numbers}
    # Create environment
    env = QEnv()
    # Choose backend Baidu Cloud Quantum Simulator-Sim2
    env.backend(BackendName.CloudBaiduSim2Water)

    # Initialize the three-qubit register
    q = env.Q.createList(3)

    # The first step of Grover's search algorithm, superposition
    H(q[0])
    H(q[1])
    H(q[2])
    ```
- then call Grover oracle $\hat O$, which is prepared using ```CCX``` gates, ```X``` gates and ```H``` gates;

    ```python{.line-numbers}
    # Call the Grover oracle
    # The first layer of X gates in Grover oracle 
    X(q[0])

    # The CCZ gate in Grover oracle
    H(q[2])
    CCX(q[0], q[1], q[2])
    H(q[2])

    # The second layer of X gates in Grover oracle 
    X(q[0])
    ```
- then call the diffusion operator $\hat D$, which is prepared using ```CCX``` gates, ```X``` gates and ```H``` gates;

    ```python{.line-numbers}
    # Call the diffusion operator
    # The first layer of Hadamard gates in the diffusion operator
    H(q[0])
    H(q[1])
    H(q[2])

    # The first layer of X gates in the diffusion operator
    X(q[0])
    X(q[1])
    X(q[2])

    # The CCZ gate in the diffusion operator
    H(q[2])
    CCX(q[0], q[1], q[2])
    H(q[2])

    # The second layer of X gates in the diffusion operator
    X(q[0])
    X(q[1])
    X(q[2])

    # The second layer of Hadamard gates in the diffusion operator
    H(q[0])
    H(q[1])
    H(q[2])
    ```
- we may alternately call Grover oracle $\hat O$ and the diffusion operator $\hat D$ another time, or measure the quantum system directly. Here we take ```1000``` shots.
    ```python{.line-numbers}
    # Finally, we measure the quantum system
    MeasureZ(*env.Q.toListPair())
    # Commit the quest to the cloud
    env.commit(1000, downloadResult=False)
    ```

For the case alternating $1$ time, the following picture shows the circuit demonstration and the rerunning task result in QComposer of Quantum Leaf with "Little Endian". We find that the quantum state $|6\rangle=|110\rangle$ is measured $773$ shots among $1000$ shots.

![the quantum circuit and task result for Grover's search algorithm on 3 qubits](./PIC/Grover_3qubit_1_EN.jpg)

If alternating $2$ times instead, (replacing the following `1` with `2`) 

```python{.line-numbers}
# Alternate calling Grover oracles and the diffusion operators
for _ in range(1):
```

we will obtain the following task result, where the state $|6\rangle=|110\rangle$ is measured for $955$ shots. In other words, the sucess probability of search increases significantly.

![another tast result for Grover's search algorithm on 3 qubits](./PIC/Grover_3qubit_2_EN.jpg)

### More Qubits
We also supply demonstration code for more qubits, which refers to python file `Grover.py`.

```python{.line-numbers}
def Grover(num_qubit, int_target=None):
    """
    :param num_qubit: n, the number of qubits which will encode the database to search
    :param int_target: t, the index of the search target, defaulted to be generated randomly
    """
    # input your token
    Define.hubToken = ""
    # create environment
    env = QEnv()
    # choose backend Baidu Cloud Quantum Simulator-Sim2
    env.backend(BackendName.CloudBaiduSim2Water)
    # create the quantum register encoding the database
    reg_sys = env.Q.createList(num_qubit)
    # generate the search target randomly if unspecified
    if int_target is None:
        int_target = randint(0, 2 ** num_qubit - 1)
    else:
        assert int_target < 2 ** num_qubit
    # prepare the initial state in Grover's algorithm
    for idx_qubit in reg_sys:
        H(idx_qubit)
    # Alternate the Grover oracle and diffusion operator for certain times,
    # which only depends on the size of the database.
    for _ in range(round(numpy.pi/(4 * numpy.arcsin(2 ** (-num_qubit / 2))) - 1 / 2)):
        Barrier(*reg_sys)
        Barrier(*reg_sys)
        # Call the Grover oracle
        circ_Grover_oracle(reg_sys, int_target)

        Barrier(*reg_sys)
        Barrier(*reg_sys)
        # Call the diffusion operator
        circ_diffusion_operator(reg_sys)

    # Finally, we measure reg_sys to verify Grover's algorithm works correctly.
    # Here the result of measurement is shown in positive sequence.
    MeasureZ(reg_sys, range(num_qubit - 1, -1, -1))
    # Commit the quest to the cloud
    env.commit(16000, downloadResult=False)
```

We can call the function `Grover` as following:

```python
# searching the state |3> on a 4-qubit circuit
Grover(4, int_target=3)
```

which will search the quantum state $|3\rangle$ in a $4$ qubits quantum circuit. If `int_target` is undifined, it will be randomly generated legally in line `14-15`. It is also noted that in the above code block
- line `20-21` correspond the preparation of initial superposition state;
- the loop starting from line `24` corresponds the alternating times of Grover oracles and diffusion operators in Grover's search algorithm;
- line `27,32` corrsepond Grover oracle and diffusion operator, respectively.

To distinguish different steps in the algorthm, we insert two layers of `Barrier`.

**Grover oracle** is implemented using `CnZ` gates (multi-controlled Pauli-$Z$ gates) and two layers of `X` gates. Here it encodes Grover oracle which qubis are operated `X` gates on, and `CnZ` gates will be implemented by calling the function `circ_multictrl_Z` in line `22`. We represent $\hat O$ by `GO` in code comments.

```python{.line-numbers}
def circ_Grover_oracle(reg_sys, int_target):
    """
    This function give a circuit to implement the Grover oracle in Grover's algorithm.
    Generally, the search target should be unknown, and encoded by Grover oracle.
    However, in this implement we suppose the search target is known, such that we can implement an oracle to encode it.
    :param reg_sys: |s>, the system register to operate the Grover oracle
    :param int_target: t, the search target we want.
    :return: GO == I - 2|s><s|,
             GO |s> == -|s>, if s == t;
                       |s>,  else.
    """
    num_qubit = len(reg_sys)
    # Since CnZ == I - 2|11...1><11...1|, we can flip CnZ into GO by two layers of X gates.
    # Meanwhile, those X gates encode the search target s.
    # the first layer of X gates encoding the search target s
    for int_k in range(num_qubit):
        if (int_target >> int_k) % 2 == 0:
            X(reg_sys[-1 - int_k])

    Barrier(*reg_sys)
    # the multictrl gate CnZ
    circ_multictrl_Z(reg_sys[-1], reg_sys[:-1])

    Barrier(*reg_sys)
    # the second layer of X gates encoding the search target s
    for int_k in range(num_qubit):
        if (int_target >> int_k) % 2 == 0:
            X(reg_sys[-1 - int_k])
```

**Diffusion operator** is implemented using `CnZ` gates, two layers of `X` gates and two layers of `H` gates. Here `CnZ` gates will be implemented by calling the function `circ_multictrl_Z` in line `18`. We represent $\hat D$ by `DO` in code comments.

```python{.line-numbers}
def circ_diffusion_operator(reg_sys):
    """
    This function give a circuit to implement the diffusion operator in Grover's algorithm.
    The diffusion operator flip the phase along the state |++...+>,
    which could be implemented by CnZ and two layers of H gates and two layers of X gates.
    :param reg_sys: |s>, the system register to operate the diffusion operator
    :return: DO == I - 2|++...+><++...+|,
             DO |s> == -|s>, if |s> == |++...+>;
                       |s>,  else.
    """
    # the first layer of H gates and the first layer of X gates
    for idx_qubit in reg_sys:
        H(idx_qubit)
        X(idx_qubit)

    Barrier(*reg_sys)
    # the multictrl gate CnZ
    circ_multictrl_Z(reg_sys[-1], reg_sys[:-1])

    Barrier(*reg_sys)
    # the second layer of X gates and the second layer of H gates
    for idx_qubit in reg_sys:
        H(idx_qubit)
        X(idx_qubit)
```

The implement of **CnZ** gate (or CnX gate equivalently) is beyond the scope of this tutorial. Readers may refers to [6] or [7] for more information. It is worth adding that if you look at the circuit generated by this code, the quantum circuit of the $n$ qubits search algorithm may appear $n+1$ qubits. Don't panic, this qubit is used as a borrowed ancilla qubit, we do not need to measure it at the end of the circuit, and its introduction will not affect our calculation results. The following figure shows the interception of the quantum circuit generated by this code when $4$ qubits are used. It can be seen that some `CCX` gates will operate on `Q[4]`.
![ancilla introduced](./PIC/Grover_ancilla.png)

---
## Reference

[1] Grover, Lov K. "A fast quantum mechanical algorithm for database search." [Proceedings of the 28th Annual ACM Symposium on Theory of Computing](https://dl.acm.org/doi/10.1145/237814.237866). 1996.

[2] Nielsen, Michael A., and Isaac Chuang. "[Quantum computation and quantum information](https://www.cambridge.org/highereducation/books/quantum-computation-and-quantum-information/01E10196D0A682A6AEFFEA52D53BE9AE)." (2002): 558-559.

[3] IQC, Baidu Research, "Grover alogrithm." [QULEARN](https://qulearn.baidu.com/textbook/chapter3/格罗弗算法.html), 2022.

[4] Brassard, Gilles, and Hoyer, Peter. "An exact quantum polynomial-time algorithm for Simon's problem." [Proceedings of the Fifth Israeli Symposium on Theory of Computing and Systems](https://ieeexplore.ieee.org/abstract/document/595153). IEEE, 1997.

[5] Grover, Lov K. "Quantum computers can search rapidly by using almost any transformation." [Physical Review Letters](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.80.4329) 80.19 (1998): 4329.

[6] Barenco, Adriano, et al. "Elementary gates for quantum computation." [Physical review A](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.52.3457) 52.5 (1995): 3457.

[7] Gidney, Craig. "[Constructing Large Controlled Nots.](https://algassert.com/circuits/2015/06/05/Constructing-Large-Controlled-Nots.html)", Web. 05 Jun. 2015.

[8] Wikipedia contributors. "Grover's algorithm." Wikipedia, The Free Encyclopedia. [Wikipedia](https://en.wikipedia.org/wiki/Grover%27s_algorithm), The Free Encyclopedia, Web. 26 Nov. 2020.



*Last Updated: 24 May 2022*