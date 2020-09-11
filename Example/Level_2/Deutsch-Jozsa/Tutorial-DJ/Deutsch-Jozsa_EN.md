# Deutsch-Jozsa Algorithm

Deutsch-Jozsa is an alogrithm to display advantage of quantum computation over classical computation.

Suppose there is a function $f$, defined on all n-bit binary strings, taking value 0 or 1, and it is either constant  or balanced (i.e. half of the results are 1 and the other half 0). Now the task is to check whether it is a constant function. Although the result can be determined immediately when two different values occur, it's still tough to determine whether it is indeed a constant function, for all $2^{n-1}+1$ values must be examined without any one ignored. Exponential-grown resource is need. But in a quantum computer, given a unitary transformation with information of $f$ (often referred as a quantum oracle), $U_f$, only one examination is needed to get the result.

The quantum circuit of this algorithm is:

![](PIC/circuit.png)

where the initial state $|x\rangle=|0^{\otimes n}\rangle$, $|y\rangle=|1\rangle$. $U_f$ is defined as: 
$$
\begin{align}
U_f:\{0,1\}^n\times\{0,1\}\rightarrow\{0,1\}^n\times\{0,1\}\\
|x\rangle \otimes|y\rangle\rightarrow|x\rangle \otimes|y\oplus f(x)\rangle\\
\end{align}
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


Now let's try it out on Quantum Leaf. Examine two examples: $f_1=0$, $f_2(x)$ = the first bit of $x$. ​
```python{.line-numbers, highlight=7}
"""
Deutsch-Jozsa Algorithm.
Suppose: f1 = 0, f2 = first bit.
"""

from QCompute import *
n = 10  # total qubit number


def analyze_result(result):
    print("Result", result)
    for key, values in result.items():
        binstring = list(reversed(list(key)))
        oct = 0
        for pos in range(len(binstring)):
            if binstring[pos] == '1':
                oct += 2**pos
        if oct % (2**n) == 0:
            print("Outcome |0>^n|y> appears, so the function is constant.")
            break
        else:
            print("Outcome other than |0>^n|y> appears, so the function is balanced.")


def main():
    """
    main
    """
    env1 = QuantumEnvironment()
    env1.backend(BackendName.LocalBaiduSim2)
    env2 = QuantumEnvironment()
    env2.backend(BackendName.LocalBaiduSim2)

    # Prepare the state:
    q1 = []
    q2 = []
    for i in range(n):
        q1.append(env1.Q[i])
        q2.append(env2.Q[i])
        H(q1[i])
        H(q2[i])
    q1.append(env1.Q[n])
    q2.append(env2.Q[n])
    X(q1[n])
    X(q2[n])
    H(q1[n])
    H(q2[n])

    # Apply U_f:
    # f1 = 0, so do nothing on q1.
    # f2 = first bit, so if the first bit is 0 do nothing, else swap q2[n].
    CX(q2[0], q2[n])

    # Measure:
    for i in range(n):
        H(q1[i])
        H(q2[i])
    MeasureZ(q1, range(n+1))
    MeasureZ(q2, range(n+1))
    taskResult1 = env1.commit(shots=1, fetchMeasure=True)
    taskResult2 = env2.commit(shots=1, fetchMeasure=True)

    # Analyze:
    analyze_result(taskResult1['counts'])
    analyze_result(taskResult2['counts'])


if __name__ == '__main__':
    main()
```

The result is like: (The highest position is $|y\rangle = |-\rangle$, and it may be 0 or 1)
```python
Result {'-0000000000': 1}
Outcome |0>^n|y> appears, so the function is constant.
Result {'-0000000001': 1}
Outcome other than |0>^n|y> appears, so the function is balanced.
```
---
## Reference
[David Deutsch and Richard Jozsa (1992). "Rapid solutions of problems by quantum computation". Proceedings of the Royal Society of London A. 439 (1907): 553–558.](https://royalsocietypublishing.org/doi/abs/10.1098/rspa.1992.0167)