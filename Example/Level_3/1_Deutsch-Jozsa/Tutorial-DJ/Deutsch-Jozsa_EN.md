# Deutsch-Jozsa Algorithm

Deutsch-Jozsa is an alogrithm to display advantage of quantum computation over classical computation.

Suppose there is a function $f$, defined on all n-bit binary strings, taking value 0 or 1, and it is either constant  or balanced (i.e. half of the results are 1 and the other half 0). Now the task is to check whether it is a constant function. Although the result can be determined immediately when two different values occur, it's still tough to determine whether it is indeed a constant function, for all $2^{n-1}+1$ values must be examined without any one ignored. Exponential-grown resource is need. But in a quantum computer, given a unitary transformation with information of $f$ (often referred as a quantum oracle), $U_f$, only one examination is needed to get the result.

The quantum circuit of this algorithm is:

![](PIC/circuit.png)

where the initial state $|x\rangle=|0^{\otimes n}\rangle$, $|y\rangle=|1\rangle$. $U_f$ is defined as: 
$$
\begin{aligned}
U_f:\{0,1\}^n\times\{0,1\}\rightarrow\{0,1\}^n\times\{0,1\}\\
|x\rangle \otimes|y\rangle\rightarrow|x\rangle \otimes|y\oplus f(x)\rangle\\
\end{aligned}
$$
$U_f$ will act on the prepared state:
$$
\frac{1}{\sqrt{2^{n+1}}}\sum_{x=0}^{2^n-1}|x\rangle(|0\rangle-|1\rangle)
$$
Then measure the front n qubits, we get:
$$
\begin{aligned}
&\bigg ( \frac{1}{\sqrt{2^{n}}}\sum_{x=0}^{2^n-1}\langle x|\otimes I\bigg )\bigg(\frac{1}{\sqrt{2^{n+1}}}\sum_{x=0}^{2^n-1}|x\rangle\big(|f(x)\rangle-|1\oplus f(x)\rangle\big)\bigg)\\
&=\bigg ( \frac{1}{\sqrt{2^{n}}}\sum_{x=0}^{2^n-1}\langle x|\otimes I\bigg )\bigg(\frac{1}{\sqrt{2^{n+1}}}\sum_{x=0}^{2^n-1}(-1)^{f(x)}|x\rangle\big(|0\rangle-|1\rangle\big)\bigg)\\
&=\frac{1}{2^n}\sum_{x=0}^{2^n-1}(-1)^{f(x)}|-\rangle
\end{aligned}
$$
And the measurement result shall be 1 or 0 if $f$ is constant or not.

If $f$ is constant the probability is$|\frac{1}{2^n}\times 2^n|^2=1$, else it is $|\frac{1}{2^n}\times\big( (-1)\times2^{n-1}+1\times2^{n-1}\big)|^2=0$.


Now let's try it out on Quantum Leaf. Examine two examples: $f_1=0$, $f_2(x)$ = the first qubit of $x$. ​
```python{.line-numbers, highlight=7}
"""
Deutsch-Jozsa Algorithm.
"""

from QCompute import *

matchSdkVersion('Python 3.0.0')

# In this example we use 10 qubits as the main register,
# and also an ancillary qubit else
MainReg_num = 10


def main():
    """
    main
    """
    # Create two environment separately, and choose backend
    # We will execute D-J algorithm for f1 and f2 simultaneously
    env1 = QEnv()
    env1.backend(BackendName.LocalBaiduSim2)
    env2 = QEnv()
    env2.backend(BackendName.LocalBaiduSim2)
    
    # Initialize two registers on 11 qubits respectively,
    # where the last qubit in each register refers to the ancillary qubit,
    # and q1 and q2 correspond to f1 and f2 respectively.
    q1 = env1.Q.createList(MainReg_num + 1)
    q2 = env2.Q.createList(MainReg_num + 1)
    
    # As a preparation for D-J algorithm, we flip the ancillary qubit from |0> to |1>
    X(q1[MainReg_num])
    X(q2[MainReg_num])
    
    # In D-J algorithm, we apply a Hadamard gate on each qubit
    # in main register and the ancillary qubit
    for i in range(MainReg_num + 1):
        H(q1[i])
        H(q2[i])
        
    # Then apply U_f:
    # for f1 = 0, we need to do nothing on q1;
    # for f2 = the value of first qubit,so if f2 = 0 do nothing,
    # else to flip the ancillary qubit in q2, which is exactly a CX gate
    CX(q2[0], q2[MainReg_num])
    
    # Then we apply a Hadamard gate on each qubit in main register again
    for i in range(MainReg_num):
        H(q1[i])
        H(q2[i])
        
    # Measure the main registers with the computational basis
    MeasureZ(q1[:-1], range(MainReg_num))
    MeasureZ(q2[:-1], range(MainReg_num))
    # Commit the quest, where we need only 1 shot to distinguish that
    # f1 is constant for the measurement result |0>,
    # and f2 is balanced for the measurement result unequal to |0>
    env1.commit(shots=1, downloadResult=False)
    env2.commit(shots=1, downloadResult=False)


if __name__ == '__main__':
    main()
```

The result is like: (The highest position is $|y\rangle = |-\rangle$, and it may be 0 or 1)
```python
Result {'0000000000': 1}
Outcome |0>^n|y> appears, so the function is constant.
Result {'0000000001': 1}
Outcome other than |0>^n|y> appears, so the function is balanced.
```
---
## Reference
[David Deutsch and Richard Jozsa (1992). "Rapid solutions of problems by quantum computation". Proceedings of the Royal Society of London A. 439 (1907): 553–558.](https://royalsocietypublishing.org/doi/abs/10.1098/rspa.1992.0167)