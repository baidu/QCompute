# Shor's Algorithm

## Preface

**Integer Factorization** denotes a decomposition of a given positive integer $N$ as a product of several smaller positive integers (greater than 1), such as $45=5\times9$.

**Shor's Algorithm** is a quantum algorithm proposed by Shor in 1994$^{[2]}$, which can give the integer factorization of $N$ in time

$$
O\left((\log N)^2(\log \log N)(\log \log \log N)\right).\tag{1}
$$

Here we will show Shor's algorithm in three sections. Firstly, we will show the framework of Shor's algorithm, referring to section **Overview of Shor's Algorithm** for details. That introduces two core steps of Shor's algorithm: reducing factorizations into orders finding and the quantum order-finding algorithm, referring to sections **From Integer Factorization to Order-Finding** and **Quantum Order-Finding Algorithm** for details, respectively. In the final of these two sectinos, we given OPENQASM and QCompute codes to factor integers $15$ and $63$ for readers to try. Since the methods above are of less generality, we can't use these to factor all integers. To overcome these problem, we give the code for the alogrithm in reference [7] for readers to try and study, referring to section **Implements of Gate $C(U_{a,N})$** for more details.

## Overview of Shor's Algorithm

Shor's algorithm can be considered as a quantum improvement of an classical algorithm for integer factorization (the ideas came from [1]):

1. both algorithms transfer the integer factorization into Order-Finding;
2. Shor proposes the Quantum Order-Finding Algorithm, which achieves **exponential speedup**;
3. replace the Order-Finding algorithm by the quantum one, and we will obtain a quantum improvement, namely Shor's algorithm.

We will introduce the transfer procedure (from integer factorization to order-finding), the quantum order-finding algorithm, and code examples of them in Quantum Leaf successively. However, we haven't given a general method to factor any integer, because of the key problem how to implement the **Oracle** (quantum blackbox) efficiently. We will present an idea on implementing the oracle by (classical) **Reversible Calculation** in the first half of the last subsection, and the Quantum Leaf code of the implement in the other half for readers to learn.

## From Integer Factorization to Order-Finding

In this section, we will show a way to transfer integer factorization into order-finding. As a result of that, as long as we solve order-finding, we can give a factorization of corresponding integer.

Some basic mathematical background knowledge is required to read the following contents of this tutorial:

1. Given two integers $N,a$, if $N/a$ is also an integer, we say $a$ **divides** $N$,$a$ is a **factor** of $N$, denoted as $a\,|\,N$; otherwise, it is denoted as $a\nmid N$;
2. The **nontrivial** factors of $N$ are those factors except $\pm 1$ and $\pm N$; **Integer factorization** can also be considered as nontrivial factors solving;
4. The **Greatest Common Divisor** (GCD) of several integers is defined as the maximal integers which divides each of these integers, denoted as ${\rm gcd}(\cdot,\cdots,\cdot)$, which can be computed by [**Euclidean algorithm**](https://baike.baidu.com/item/%E6%AC%A7%E5%87%A0%E9%87%8C%E5%BE%97%E7%AE%97%E6%B3%95/1647675?fr=aladdin) effectively; We say two integers **coprime** if their GCD is $1$;
5. Given integers $a,b,N$, we say $a$ and $b$ are congruent modulo $N$ if $N|(a-b)$, denoted as $a\equiv b\pmod{N}$; otherwise, it is denoted as $a\not\equiv b\pmod{N}$;
6. Given two integers $a$ and $N$, we say $r$ is the **(multiplicative) order** of $a$ modulo $N$ if $r$ is the minimal positive integer such that $a^r\equiv 1\pmod{N}$, denoted as $r={\rm ord}(a,N)$; We say $a$ and $N$ the **base** and the **modulus** of $r$.

If we know an integer $a$ and its order $r$ modulo $N$, then we have $N\,|\,(a^r-1)$. Now assume that $r$ is even (say $2\,|\,r$), and then we have $N\,|\,(a^{r/2}-1)(a^{r/2}+1)$ by the formula for the difference of square. We know that $N\nmid(a^{r/2}-1)$ by the minimality in the definition of the orders. Now further assume that $N\nmid(a^{r/2}+1)$, i.e. $a^{r/2}\not\equiv-1\pmod{N}$, then we have $N\nmid(a^{r/2}\pm1)$ but $N\mid(a^{r/2}-1)(a^{r/2}+1)$. Thus it can be proved that ${\rm gcd}(N,a^{r/2}\pm1)$ are both nontrivial factors of $N$. In summary, if we have a pair $(a,r)$ satisfying above conditions, we can find a nontrivial factor of $N$ effectively (equivalently, we have a factorization of $N$). To simplify the expression, writers give the above conditions a title (not general formal):

<div class="theorem">

**Definition 1 (Inducer).** Given two coprime integers $a$ and $N$ with $2\nmid N$ (say $N$ is odd), denote $r={\rm ord}(a,N)$. We call $a$ an **inducer modulo** $N$ if $r$ is even and $a^{r/2}\not\equiv-1\pmod{N}$.

</div>

We have known an inducer $a$ and its order $r$ can induce an integer factorization of $N$, but it is unresolved how to find an inducer and how to compute the order. However if we have an algorithm to compute orders effectively, then we can select $a$ randomly, then compute its order, and then determine whether $a$ is an inducer. If so,we can give a factorization of $N$ through ${\rm gcd}(N,a^{r/2}\pm1)$; otherwise, we can re-select $a$ and repeat the above procedure. Especially, if we have selected an integer $a$ not coprime with $N$, although the order of $a$ doesn't exist, ${\rm gcd}(a,N)$ has been a nontrivial factor of $N$. Meanwhile we do not need the order-finding step to obtain the factorization of $N$.

<div class="theorem">

**Example 2.** Let $N=315$, $a=2$. It can be proved that ${\rm ord}(2,315)=12$ and $2^{12/2}\equiv 64\not\equiv-1\pmod{315}$. Thus $2$ is an inducer modulo $315$, both ${\rm gcd}(2^{6}-1,315)=63$ and ${\rm gcd}(2^{6}+1,315)=5$ are nontrivial factors of $315$. Especially, we have $315=63\times5$ exactly.

</div>

We will expand the step of order-finding in the next section, which is the only quantum part in Shor's algorithm. As the last of this section, we summary the procedure of Shor's algorithm as following:

<div class="theorem">

**Algorithm 3 (Shor's algorithm).**

**Input:** a positive odd $N$ which has two coprime nontrivial factors.

**Output:** an nontrivial factor of $N$.

1. Select $a$ in $\{2,3,\cdots,N-2\}$ randomly;
2. Use [Euclidean algorithm](https://baike.baidu.com/item/%E6%AC%A7%E5%87%A0%E9%87%8C%E5%BE%97%E7%AE%97%E6%B3%95/1647675?fr=aladdin) to compute ${\rm gcd}(a,N)$;
3. **if** ${\rm gcd}(a,N)\ne1$, **return** ${\rm gcd}(a,N)$;
4. Use (quantum) order-finding algorithm to compute $r={\rm ord}(a,N)$;
5. **if** $r$ is even **and** $a^{r/2}\not\equiv -1\pmod{N}$,**return** ${\rm gcd}(a^{r/2}+1,N)$ (or ${\rm gcd}(a^{r/2}-1,N)$);
6. **goto** step 1.

</div>

## Quantum Order-Finding Algorithm

### Theory

Since the algorithm is to find the order of $a$ modulo $N$, we call it quantum order-finding algorithm. It can be proved that

$$
f(x)= a^x\text{ mod }{N}\tag{2}
$$

is a period function, whose (minimum positive) period is ${\rm ord}(a,N)$ exactly, where $x \text{ mod }y:=x-y\lfloor x/y\rfloor\in\{0,\cdots,y-1\}$ is the remainder of $x$ divided by $y$. Then, quantum order-finding algorithm can be considered as a special case for the **Quantum Period-Finding Algorithm**. In essence, quantum phase estimation algorithm is also a special case for quantum period-finding algorithm, but it is more difficult to understand order-finding and phase estimation through period-finding. Here interested readers may refer to [5].

Now we can consider that Shor's order-finding algorithm is based on thg phase estimation algorithm: 
1. **encode** the order into the eigen phase of some quantum gate (or circuit);
2. use quantum phase estimation algorithm to **estimate** the eigen phase;
3. use the continued fraction expansion algorithm to **decode** the value of order.

#### Encoding the Order into Eigen Phase

Given $a\in\{2,\cdots,N-2\}$ coprime with $N$, our goal is to solve $r={\rm ord}(a,N)$.Introducing $L=\lceil \log_2(N+1)\rceil$ qubits to encode the quantum states $|0\rangle,|1\rangle,\cdots,|N-1\rangle$, set a quantum gate

$$
U_{a,N}=\sum_{x=0}^{N-1}|{ax\ \textrm{mod } {N}}\rangle\langle{x}|+\sum_{x=N}^{2^L-1}|{x}\rangle\langle{x}|.\tag{3}
$$

It can be proved that $|1\rangle$ is an average superposition state of some eigenstates of $U_{a,N}$. Operating the quantum phase estimation algorithm on the state $|1\rangle$ and the quantum gate $U_{a,N}$, we will obtain an approximation of 

$$
s/r, \ s=0,1,\cdots,r-1,\tag{4}
$$

with $2\pi s/r$ an eigen phase of $U_{a,N}$. When the precision of quantum phase estimation algorithm is less than $N^{-2}$, we can use continued fraction expansion algorithm to recovery $r$ from the approximation.

#### Continued Fraction Expansion Decodes eigen phases into Orders

Here we demonstrate the continued fraction expansion algorithm through an example. It is clear to see that the continued fraction expansion algorithm is just a deformation of Euclidean algorithm: in each iteration we reform the fraction as a mixed fraction, reduce the numerator into $1$, and input the new denominator into next iteration until the denominator is an integer.

<div class="theorem">

**Example 4.** Set $N=63$, $k=2729/8192$. We have the continued fraction expansion for $k$:

$$
\begin{aligned}
\frac{2729}{8192}
=&\frac{1}{\frac{8192}{2729}}=\frac{1}{3+\frac{5}{2729}}=\frac{1}{3+\frac{1}{545+\frac45}}\\
=&0 + \frac{1}{3+\frac{1}{545+\frac{1}{1+\frac{1}{4}}}}=:[0,3,545,1,4],
\end{aligned}\tag{5}
$$

We call $[0]=0$,$[0,3]=0+\frac{1}{3}=\frac{1}{3}$,$[0,3,545]=0+\frac{1}{3+\frac{1}{545}}=\frac{545}{1636}$ and $[0,3,545,1]=0+\frac{1}{3+\frac{1}{545+\frac{1}{1}}}=\frac{546}{1639}$ the $0$th, $1$st, $2$nd and $3$rd **convergent** of continued fraction $[0,3,545,1,4]$. The continued fraction expansion algorithm **returns** the maximal denominator less than $N$ of each convergent of $k$, which is $3$ in this example.

</div> 

<div class="theorem">

**Remark 5.** When the precision of quantum phase estimation algorithm is less than $N^{-2}$, the continued fraction expansion algorithm will return the denominator of $s/r$. However, the denominator of $s/r$ may not be $r$: when ${\rm gcd}(s,r)\ne1$, the denominator of $s/r$ is $r/{\rm gcd}(s,r)$. Meanwhile the probability of recovering the correct order is related to $N$, but trending to $0$ for the worst case.

</div> 

In order to increase the probability of recovering the order, the best method is operating the above algorithm twice, and returning the least common multiple of the two returns:

<div class="theorem">

**Theorem 6.** By calling the quantum order-finding algorithm twice, we can obtain two denominators $q$ and $q'$. Compute the least common multiple of them $\tilde r:=q q'/\textrm{gcd}(q,q')$, and then the probability of $\tilde r=r$ is greater than ${6(1-\epsilon)^2}/{\pi^2} \ge60.7\%\cdot(1-\epsilon)^2$, where $\epsilon$ is the failure probability of the phase estimation algorithm. Here $q$ and $q'$ are returns of continued fraction expansion algorithm with $k$ and $k'$ of as inputs, respectively, where $k$ and $k'$ are the two returns of quantum phase estimation algorithm.

</div>

#### Summary

At the last of this section, we summary the quantum order-finding algorithm.

<div class="theorem">

**Algorithm 7 (Quantum order-finding algorithm.).**

**Input:** Integer $a\in\{2,\cdots,N-2\}$ coprime with $N$, the failure probability of quantum phase estimation $\epsilon$.

**Output:** ${\rm ord}(a,N)$.


1. Set $L=\lceil \log_2(N+1)\rceil$;
2. Call twice: using $t=2L+1+\lceil\log_2(1+\frac{2}{\epsilon\pi^2})\rceil$ qubits as the ancilla register to operate the quantum phase estimation algorithm on the quantum gate $U_{a,N}$ and quantum state $|1\rangle$ and measuring the ancilla register. Obtain two quantum state $|m\rangle$ and $|m'\rangle$, respectively;
3. Operate the continued fraction expansion algorithm on $k=2^{-t}m$ and $k'=2^{-t}m'$ and find the maximal denominator less than $N$ of each convergent $q_j$ 和 $q_k'$, respectively;
4. Setting $r_{jk}=q_jq_k'/{\rm gcd}(q_j,q_k')$, **if** $a^{r_{jk}}\equiv 1\pmod N$ **return** ${\rm ord}(a,N)=r_{jk}$; **else** **goto** step 2 and restart the computation.

</div>

Here the probability of returning ${\rm ord}(a,N)$ in the step 4 of algorithm 7 is greater than $60.7\%\cdot(1-\epsilon)^2$. Especially, set $t=2L+1+3$, $\epsilon=2.895\%$, the probability is greater than $57.3\%$. Meanwhile, the probability of finding the correct order is greater than $96.6\%$ for four iterations.

Thus, we have shown all content of Shor's algorithm, and will demonstration two simple example in the next subsection.

### Simple Examples and Experiments

In this subsection, we will show Shor's algorithm in [Quantum Leaf](https://quantum-hub.baidu.com/#/). Firstly, we use QComposer to solve the order of $2$ modulo $15$ and thus give a factorization of $15$. Secondly, we use PyOnline to solve the order of $2$ modulo $63$ and thus give a factorization of $63$.

#### Find ${\rm ord}(2,15)$ and Factor $15$

$L=\lceil \log_2(15+1)\rceil=4$ qubits are needed to encode the quantum gate $U_{2,15}$ (abbreviated as $U$ in this subsection):

$$
U_{2,15}=\left[\begin{array}{cccccccccccccccc}
1&0&0&0&0&0&0&0&0&0&0&0&0&0&0&0\\
0&0&0&0&0&0&0&0&1&0&0&0&0&0&0&0\\
0&1&0&0&0&0&0&0&0&0&0&0&0&0&0&0\\
0&0&0&0&0&0&0&0&0&1&0&0&0&0&0&0\\
0&0&1&0&0&0&0&0&0&0&0&0&0&0&0&0\\
0&0&0&0&0&0&0&0&0&0&1&0&0&0&0&0\\
0&0&0&1&0&0&0&0&0&0&0&0&0&0&0&0\\
0&0&0&0&0&0&0&0&0&0&0&1&0&0&0&0\\
0&0&0&0&1&0&0&0&0&0&0&0&0&0&0&0\\
0&0&0&0&0&0&0&0&0&0&0&0&1&0&0&0\\
0&0&0&0&0&1&0&0&0&0&0&0&0&0&0&0\\
0&0&0&0&0&0&0&0&0&0&0&0&0&1&0&0\\
0&0&0&0&0&0&1&0&0&0&0&0&0&0&0&0\\
0&0&0&0&0&0&0&0&0&0&0&0&0&0&1&0\\
0&0&0&0&0&0&0&1&0&0&0&0&0&0&0&0\\
0&0&0&0&0&0&0&0&0&0&0&0&0&0&0&1
\end{array}\right].\tag{6}
$$

A decomposition of $C(U_{2,15})$ is shown as following, where the $0$th qubit is the controlling qubit, and the $1$st, $2$nd, $3$rd and $4$th qubit are the target qubits of $U_{2,15}$ in sequence:

![$\operatorname{controlled^0-}U_{2,15}^{1,2,3,4}$](figures/CU215.png)

The corresponding OPENQASM code is following:

```cpp{.line-numbers}
OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];
cswap q[0], q[1], q[2];
cswap q[0], q[2], q[3];
cswap q[0], q[3], q[4];
```

which can be represented as

$$
\begin{aligned}
C(U_{2,15})&= \operatorname{CSWAP}_{0,3,4}\cdot\operatorname{CSWAP}_{0,2,3}\cdot\operatorname{CSWAP}_{0,1,2},\tag{7}
\end{aligned}
$$

where $\operatorname{CSWAP}_{j,k,l}$ is a $\operatorname{CSWAP}$ gate with the $j$-th qubit as controlling qubit and the $k$-th and $l$-th qubit as target qubits. Other $C(U_{2,15}^{2^j})$ gate are needed in the quantum phase estimation algorithm:

$$
\begin{aligned}
C(U_{2,15}^2)&= \operatorname{CSWAP}_{0,2,4}\cdot \operatorname{CSWAP}_{0,1,3};\tag{8}
\end{aligned}
$$

$$
\begin{aligned}
C(U_{2,15}^{2^j}) &= I,\quad j\ge3.\tag{9}
\end{aligned}
$$

Using $2L+1=9$ qubits as the ancilla register to estimate the eigen phase of $|0001\rangle$, the quantum circuit of that in QComposer is shown in the following figure (where we omit those omissible $I$ gates):

![the circuit for finding $\textrm{Ord}(2,15)$ in QComposer](figures/OrderFinding215.png)

The corresponding OPENQASM [code](https://quantum-hub.baidu.com/#/) is following:

```cpp{.line-numbers}OPENQASM 2.0;
include "qelib1.inc";
qreg q[13]; // q[0],...,q[8] are ancilla register, q[9],...,q[12] are system register
creg c[13];
x q[12]; // Prepare the initial state |0>|1>, and will operate the Quantum Phase Estimation algorithm (QPE)
// The first step of QPE where we prepare an average superposition state
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
h q[6];
h q[7];
h q[8]; 
// An implement of C(U) with q[8] as the controlling qubit
cswap q[8], q[9], q[10];
cswap q[8], q[10], q[11];
cswap q[8], q[11], q[12]; 
// An implement of C(U^2) with q[7] as the controlling qubit
cswap q[7], q[9], q[11];
cswap q[7], q[10], q[12]; 
// Omit other C(U^(2^j)) gates equal to I
// Operate inversed Fourier tranform on the ancilla register
// The SWAP step in inversed Fourier tranform
swap q[0], q[8];
swap q[1], q[7];
swap q[2], q[6];
swap q[3], q[5];
// The control-rotation step in inversed Fourier tranform
h q[8];
cu(0, 0, -pi/2) q[8], q[7];
h q[7];
cu(0, 0, -pi/4) q[8], q[6];
cu(0, 0, -pi/2) q[7], q[6];
h q[6];
cu(0, 0, -pi/8) q[8], q[5];
cu(0, 0, -pi/4) q[7], q[5];
cu(0, 0, -pi/2) q[6], q[5];
h q[5];
cu(0, 0, -pi/16) q[8], q[4];
cu(0, 0, -pi/8) q[7], q[4];
cu(0, 0, -pi/4) q[6], q[4];
cu(0, 0, -pi/2) q[5], q[4];
h q[4];
cu(0, 0, -pi/32) q[8], q[3];
cu(0, 0, -pi/16) q[7], q[3];
cu(0, 0, -pi/8) q[6], q[3];
cu(0, 0, -pi/4) q[5], q[3];
cu(0, 0, -pi/2) q[4], q[3];
h q[3];
cu(0, 0, -pi/64) q[8], q[2];
cu(0, 0, -pi/32) q[7], q[2];
cu(0, 0, -pi/16) q[6], q[2];
cu(0, 0, -pi/8) q[5], q[2];
cu(0, 0, -pi/4) q[4], q[2];
cu(0, 0, -pi/2) q[3], q[2];
h q[2];
cu(0, 0, -pi/128) q[8], q[1];
cu(0, 0, -pi/64) q[7], q[1];
cu(0, 0, -pi/32) q[6], q[1];
cu(0, 0, -pi/16) q[5], q[1];
cu(0, 0, -pi/8) q[4], q[1];
cu(0, 0, -pi/4) q[3], q[1];
cu(0, 0, -pi/2) q[2], q[1];
h q[1];
cu(0, 0, -pi/256) q[8], q[0];
cu(0, 0, -pi/128) q[7], q[0];
cu(0, 0, -pi/64) q[6], q[0];
cu(0, 0, -pi/32) q[5], q[0];
cu(0, 0, -pi/16) q[4], q[0];
cu(0, 0, -pi/8) q[3], q[0];
cu(0, 0, -pi/4) q[2], q[0];
cu(0, 0, -pi/2) q[1], q[0];
h q[0]; 
// The inversed Fourier tranform is completed
// Measurement
measure q[8] -> c[8];
measure q[7] -> c[7];
measure q[6] -> c[6];
measure q[5] -> c[5];
measure q[4] -> c[4];
measure q[3] -> c[3];
measure q[2] -> c[2];
measure q[1] -> c[1];
measure q[0] -> c[0];
```

The results of a run are as follows:

```python
{
    "000000011": 237,
    "000000010": 257,
    "000000000": 269,
    "000000001": 261
}
```

About equal frequency
We can see that the measurement are $|000000000\rangle$, $|010000000\rangle$, $|100000000\rangle$ and $|110000000\rangle$ of about equal frequency, which are corresponding to fractions $0$, $1/4$, $1/2$ and $3/4$. Here the continued fraction expansion of $3/4$ is

$$
\frac{3}{4}=\frac{1}{1+\frac{1}{3}},\tag{10}
$$ 

whose $1$st convergent is $1$。

Thus we have probability about $1/2$ to obtain ${\rm ord}(2,15)=4$ correctly in each call of quantum phase estimation algorithm, and $3/4$ in each two calls. Since $2^{4/2} = 4\not\equiv -1\pmod{15}$, we have $15|(4-1)(4+1)$ induces a factorization of $15$:

$$
15=\textrm{gcd}(15,4-1)\times\textrm{gcd}(15,4+1)=3\times5.\tag{11}
$$

Now the show is done that using Shor's algorithm to factor $15$.

Afterword: It is not a general method to omit $C(U^{2^j})$ gates equal to $I$ in the above OPENQASM code, because of rejecting what is near at hand and seeking what is far away. The right way is to decompose those gates by a general method instead of to prove it equal to $I$. In the next subsection we will avoid this when showing the factorization of $63$.

#### Find ${\rm ord}(2,63)$ and Factor $63$

$L=\lceil \log_2(63+1)\rceil=6$ qubits are needed to encode the quantum gate $U_{2,63}$ (also abbreviated as $U$ in this subsection). We will introduce $2L+1=13$ qubits as the ancilla register to estimate the eigen phase. Considering $j>0$ implies

$$
U^{2^j}=U^{2^{j\rm{\ mod\ }2}},\tag{12}
$$ 

we only need to implement quantum gates $C(U)$, $C(U^{2})$ and $C(U^{4})$:

$$
\begin{aligned}
C(U)&= \operatorname{CSWAP}_{0,1,2}\cdot\operatorname{CSWAP}_{0,2,3}\cdot\operatorname{CSWAP}_{0,3,4}\cdot\operatorname{CSWAP}_{0,4,5}\cdot\operatorname{CSWAP}_{0,5,6};\tag{13}\\
\end{aligned}
$$

$$
\begin{aligned}
C(U^2)&= \operatorname{CSWAP}_{0,3,5}\cdot \operatorname{CSWAP}_{0,1,3}\cdot\operatorname{CSWAP}_{0,4,6}\cdot \operatorname{CSWAP}_{0,2,4};\tag{14}\\
\end{aligned}
$$

$$
\begin{aligned}
C(U^{4}) &=\operatorname{CSWAP}_{0,1,3} \cdot \operatorname{CSWAP}_{0,3,5}\cdot\operatorname{CSWAP}_{0,2,4}\cdot \operatorname{CSWAP}_{0,4,6}.\tag{15}
\end{aligned}
$$

The [PyOnline](https://quantum-hub.baidu.com/#/) code is following:

```python{.numberLines}
from QCompute import *
from numpy import pi

matchSdkVersion('Python 3.0.0')


def func_order_finding_2_mod_63():
    """
    This function will give an approximation related to the eigen phase s/6 for some s=0,1,2,3,4,5
    where 6 is the order of 2 mod 63.
    """
    env = QEnv()  # Create environment
    env.backend(BackendName.LocalBaiduSim2)  # Choose backend Baidu Local Quantum Simulator-Sim2

    L = 6  # The number of qubits to encode the gate U, also the number of qubits in the system register
    N = 3 * L + 1  # The total number of qubits in this algorithm
    # The number of ancilla qubits used in the quantum phase estimation algorithm (QPE), also the number of qubits in
    # the ancilla register
    t = 2 * L + 1

    # Create a register, the first t qubits of which form the ancilla register, and the others form the system register.
    q = env.Q.createList(N)

    X(q[N - 1])  # We prepare the state |1> in the system register, and will operate QPE

    for i in range(t):
        H(q[i])  # The first step in QPE, we prepare an average superposition state,

    # The following is the transfer step in QPE, we will operate several C(U^(2^j)) gates

    # The following is a decomposition of the gate C(U), the ctrlling qubit is the last qubit in the system register
    CSWAP(q[2 * L], q[t + 4], q[t + 5])
    CSWAP(q[2 * L], q[t + 3], q[t + 4])
    CSWAP(q[2 * L], q[t + 2], q[t + 3])
    CSWAP(q[2 * L], q[t + 1], q[t + 2])
    CSWAP(q[2 * L], q[t + 0], q[t + 1])

    s = 2 * L - 1  # For the other C(U^(2^j)) gates, where q[s] is just the ctrlling qubit
    while s >= 0:
        if s % 2 == 1:
            # The decomposition of C(U^2) under this condition
            CSWAP(q[s], q[t + 1], q[t + 3])
            CSWAP(q[s], q[t + 3], q[t + 5])
            CSWAP(q[s], q[t + 0], q[t + 2])
            CSWAP(q[s], q[t + 2], q[t + 4])
        else:
            # The decomposition of C(U^4) under this condition
            CSWAP(q[s], q[t + 3], q[t + 5])
            CSWAP(q[s], q[t + 1], q[t + 3])
            CSWAP(q[s], q[t + 2], q[t + 4])
            CSWAP(q[s], q[t + 0], q[t + 2])
        s -= 1  # Move the pointer to a higher ancilla qubit

    # We need to operate an inverse Quantum Fourier Transform (QFT) on the ancilla register in the last step of QPE
    # The SWAP step in inverse QFT
    for i in range(t // 2):
        SWAP(q[i], q[t - i - 1])

    # The ctrl-rotation step in inverse QFT
    for i in range(t - 1):
        H(q[t - i - 1])
        for j in range(i + 1):
            CU(0, 0, -pi / pow(2, (i - j + 1)))(q[t - j - 1], q[t - i - 2])
    H(q[0])

    # We have completed the inverse QFT and also QPE, and will measure the quantum state we have obtained
    MeasureZ(q[:t], range(t))  # Only the ancilla register (i.e. the first t qubits) need to be measured
    env.commit(8192, downloadResult=False)


if __name__ == "__main__":
    func_order_finding_2_mod_63()
```

Call the quantum algorithm for $8192$ times, omit those results of $5\%$ total frequency, and we process the remainder data as following, where `"4096": 1419` indicates that there are $1419$ shots of $|4096\rangle$ in quantum measurement:

```python
{
    "4096": 1419,
    "0": 1287,
    "5461": 955,
    "2731": 955,
    "1365": 940,
    "6827": 896,
    "6826": 252,
    "1366": 250,
    "2730": 246,
    "5462": 219,
    "2732": 56,
    "5460": 53,
    "6828": 52,
    "1364": 49,
    "2729": 47,
    "1367": 41,
    "5463": 36,
    "6825": 34,
} 
```

Call the continued fraction expansion algorithm and compute all convergent, and we obtain:

```python
{
    {{0, 1/2}: 1419},
    {{0}: 1287},
    {{0, 1, 1/2, 2/3, 5461/8192}: 955},
    {{0, 1/2, 1/3, 2731/8192}: 955},
    {{0, 1/6, 682/4093, 1365/8192}: 940}, 
    {{0, 1, 5/6, 3411/4093, 6827/8192}: 896}, 
    {{0, 1, 4/5, 5/6, 1704/2045, 3413/4096}: 252}, 
    {{0, 1/5, 1/6, 341/2045, 683/4096}: 250}, 
    {{0, 1/3, 1365/4096}: 246}, 
    {{0, 1, 2/3, 2731/4096}: 219}, 
    {{0, 1/2, 1/3, 683/2048}: 56}, 
    {{0, 1, 1/2, 2/3, 1365/2048}: 53}, 
    {{0, 1, 5/6, 851/1021, 1707/2048}: 52},
    {{0, 1/6, 170/1021, 341/2048}: 49}, 
    {{0, 1/3, 545/1636, 546/1639, 2729/8192}: 47}, 
    {{0, 1/5, 1/6, 136/815, 137/821, 410/2457, 1367/8192}: 41},
    {{0, 1, 2/3, 1091/1636, 1093/1639, 5463/8192}: 36},
    {{0, 1, 4/5, 5/6, 679/815, 684/821, 2047/2457, 6825/8192}: 34}
} 
```

The maximal denominators less than $N=63$ in each line is concerned (here is only $1,2,3,6$). The frequency of denominator $2$, $3$ and $6$ are $1419$, $2567$ and $2514$, respectively. Considering $\rm{ord}(2,63)=6$, we have probability about $2514/8192 \approx 30.7\%$ to obtain ${\rm ord}(2,15)=4$ correctly in each call of quantum phase estimation algorithm, and

$$
1-\left(1-\frac{2514}{8192}\right)^2+2 * \frac{1419}{8192} * \frac{2567}{8192}\approx 62.8\%\tag{16}
$$

in each two calls, where the last adder indicates that one of the two denominators is $2$ and the other is $3$. Since $2^{6/2} = 8\not\equiv -1\pmod{63}$, we have $63|(8-1)(8+1)$ induces a factorization of $63$:

$$
63=\textrm{gcd}(63,8-1)\times\textrm{gcd}(63,8+1)=7\times9.\tag{17}
$$

Even though we do not use $U^{2^j}=I$ in this example, 

$$
U^{2^3}=U^2\tag{18}
$$

implies $U^{2^3-2}=I$ and $\rm{ord}(2,63)\,|\,(2^3-2)=6$, which is also rejecting what is near at hand and seeking what is far away.


## Implements of Gate $C(U_{a,N})$

To achieve the general implement of Shor's algorithm, we should find a general algorithm to decompose the quantum gate $C(U_{a,N})$. In the last section, we introduce algebraic method to decompose gate $C(U_{a,N})$ into several $\operatorname{CSWAP}$ gates, and simplify gates $C(U_{a,N}^{2^j})$. However, it doesn't make sense logically, not only the procedure of decomposition but also the omissions of gates equal to $I$, which refer to or implies the fact that the factorization of $N$ has been known. More general algorithms on implementing gates $C(U_{a,N})$ refer to $[6,7]$.

Besides these complicated method for reducing the number of ancilla qubits, we can implement the gate $C(U_{a,N})$ by **Reversible Calculation**.

### Reversible Calculation

#### From (Classical) Reversible Calculation on

What's the classical reversible calculation? It is known to us that classical computers are based on binary numbers and logic gates, and there are three elementary logic gates $\mathtt{AND}$ gate $\wedge$, $\mathtt{OR}$ gate $\vee$ and $\mathtt{NOT}$ gate $\neg$. It is clear that both $\mathtt{AND}$ gate and $\mathtt{OR}$ gate have two inputs and one output:

$$
\forall a,b\in\{0,1\},\ a\wedge b=ab,\ a\vee b=\neg(\neg a\wedge\neg b)=1-(1-a)(1-b),\ \neg a=1-a.\tag{19}
$$

We call these calculation where we cannot recover the inputs from outputs **irreversible calculation**, and **reversible calculation** otherwise. It is clear to see that for a logical operator in reversible calculation, the number of output bits should be equal to that of input bits, such as the $\mathtt{NOT}$ gate is reversible calculation.

It can be predicted that if we maintain the input unchanged, and record the output on ancilla register, we will implement any function in reversible calculation, such as:

$$
(a,b,c)\rightarrow(a,b,c\oplus f(a,b)),\tag{20}
$$

where $f(a,b)$ is an arbitrary function $\{0,1\}^2\rightarrow\{0,1\}$, and $x\oplus y:=x+y\text{ mod }2$ is the result of $\mathtt{XOR}$ operator on $x$ and $y$.

Finally, by introducing ancilla qubits, one can use quantum gates $X$, $C(X)$ and $C^2(X)$ to simulate any reversible calculation in quantum computing, where:

$$
\begin{aligned}
X|a\rangle=&\ |\neg a\rangle,\\
C^2(X)|a\rangle|b\rangle|0\rangle
=&\ |a\rangle|b\rangle|a\wedge b\rangle,\\
C(X)|a\rangle|0\rangle
=&\ |a\rangle|a\rangle=:\operatorname{FANOUT}|a\rangle.
\end{aligned}\tag{21}
$$

Here $\operatorname{FANOUT}$ will make two copies of a classical bit, which can be simulated for the computational basis in quantum computing. According to the decomposition of $\mathtt{OR}$, we have the quantum version for it:

$$
(X\otimes X\otimes X)\cdot C^2(X)\cdot(X\otimes X\otimes I)|a\rangle|b\rangle|0\rangle
=\ |a\rangle|b\rangle|a\vee b\rangle.\tag{22}
$$

![Quantum $\mathtt{OR}$ gate](figures/vee.png)

#### Go Back to Gates $U_{a,N}$

Recall the definition of

$$
U_{a,N}=\sum_{x=0}^{N-1}|{ax\ \textrm{mod } {N}}\rangle\langle{x}|+\sum_{x=N}^{2^L-1}|{x}\rangle\langle{x}|.\tag{23}
$$

Given an input $x\in\{0,1,\cdots,2^L-1\}$, we can always determine whether $x< N$ classically, and compute $ax\textrm{ mod } N$ if $x< N$. The underlying implement of these determinations and computations must come down to logical gates, i.e. $\mathtt{AND}$, $\mathtt{OR}$ and $\mathtt{NOT}$ gates. Thus these could be implemented quantumly. Say we finally obtain a quantum circuit $\operatorname{C}$ consist of several $X$, $C(X)$ and $C^2(X)$ gates, which implements

$$
\operatorname{C}|x\rangle|0\rangle^{\otimes L}|0\rangle^{\otimes t}=|x\rangle(U_{a,N}|x\rangle)|0\rangle^{\otimes t},\tag{24}
$$

where $t+L$ is the number of ancilla qubits introduced in the whole quantum circuit. According to the property of reversible computation, the number of quantum gates in the above quantum circuit and $t$ must be less than twice the number of logical gates in classical computing, which ensures effective implement of the quantum circuit $\operatorname{C}$.

$(24)$ is different from our purpose $|x\rangle\rightarrow U_{a,N}|x\rangle$. By the way, we may use quantum gate $\operatorname{SWAP}$ to swap two registers, and implement a quantum gate $U_{a^{-1},N}^\dagger$ analogously to erase the information $|x\rangle$:

$$
|x\rangle(U_{a,N}|x\rangle)|0\rangle^{\otimes t}\xrightarrow{\operatorname{SWAP}}
(U_{a,N}|x\rangle)|x\rangle|0\rangle^{\otimes t}\xrightarrow{\operatorname{\hat C^\dagger}}
(U_{a,N}|x\rangle)|0\rangle|0\rangle^{\otimes t},\tag{25}
$$

where $\hat{\operatorname C}^\dagger$ is the inverse circuit for $\hat{\operatorname C}$:

$$
(U_{a,N}|x\rangle)|0\rangle|0\rangle^{\otimes t}
\xrightarrow{\operatorname{\hat C}}
(U_{a,N}|x\rangle)(U_{a^{-1},N}U_{a,N}|x\rangle)|0\rangle^{\otimes t}=(U_{a,N}|x\rangle)|x\rangle|0\rangle^{\otimes t}.\tag{26}
$$

**Example 8.** We will deduct this procedure by an example $U_{2,3}$:

![$U_{2,3}$](figures/2xmod3.png)

$3$ zeroed qubits are introduced here. The first of them is used as a comparator storing whether $x < 3$, and will perform a controlling qubit to distinguish two cases. The other two zeroed qubits forms an ancilla register. We implement a map $|x\rangle|0\rangle\rightarrow|x\rangle|2x\text{ mod }3\rangle$ with the comparator as a controlling qubit. Then swap two registers to move $|2x\text{ mod }3\rangle$ into the system register also with the comparator as a controlling qubit. Then we may use $|2x\text{ mod }3\rangle$ to erase the information $|x\rangle$ in the ancilla register and obtain $|2x\text{ mod }3\rangle|0\rangle$, because $2(2x\text{ mod 3})\text{ mod 3} = x$. Finally, we erase the information in the comparator to implement a $U_{2,3}$ gate. The detailed evolution procedure of the five qubits are as follows:

$$
\begin{aligned}
|x\rangle|0\rangle|0\rangle\rightarrow
\begin{cases}
|x\rangle|1\rangle|0\rangle\rightarrow|x\rangle|1\rangle|2x\text{ mod }3\rangle\rightarrow|2x\text{ mod }3\rangle|1\rangle|x\rangle\rightarrow|2x\text{ mod }3\rangle|1\rangle|0\rangle\rightarrow|2x\text{ mod }3\rangle|0\rangle|0\rangle,&x<3;\\
|x\rangle|0\rangle|0\rangle\rightarrow|x\rangle|0\rangle|0\rangle\rightarrow|x\rangle|0\rangle|0\rangle\rightarrow|x\rangle|0\rangle|0\rangle\rightarrow|x\rangle|0\rangle|0\rangle,&x=3.
\end{cases}
\end{aligned}\tag{27}
$$

Of course, the computation for $2x\text{ mod }3$ is simplified greatly. For general cases, we may need more ancilla qubits to store carries and intermediate results of multiplication.

### Implements of General $C(U_{a,N})$ Gates

In the quantum order-finding algorithm, the oracles are unitary matrices of form

$$
U_{a,N}=\sum_{x=0}^{N-1}|{ax\ \textrm{mod } {N}}\rangle\langle{x}|+\sum_{x=N}^{2^L-1}|{x}\rangle\langle{x}|.\tag{6}
$$

Also some powers 

$$
U_{a,N}^{2^k}=U_{a^{2^k}\text{ mod }N,N},
$$

of $U_{a,N}$ are required, who have the same form $U_{\tilde a,N}$. Thus it is a general quantum circuit implementing $U_{\tilde a,N}$ effectively that we need.

Hence, we refer to the algorithm in

[7] Beauregard, Stephane. "Circuit for Shor's algorithm using 2n+3 qubits." Quantum Information & Computation 3.2 (2003): 175-185.

to give the implement code for quantum gate $C(U_{a,N})$. Similar to the procedure in example 8, here we also adopt the route

$$
|x\rangle|0\rangle^{\otimes L}|0\rangle^{\otimes t}\rightarrow|x\rangle(U_{a,N}|x\rangle)|0\rangle^{\otimes t}\xrightarrow{\operatorname{SWAP}}
(U_{a,N}|x\rangle)|x\rangle|0\rangle^{\otimes t}\rightarrow
(U_{a,N}|x\rangle)|0\rangle^{\otimes L}|0\rangle^{\otimes t}.\tag{28}
$$

More detailed, to implement the quantum gate $U_{a,N}$, we need to implement the following quantum gates and their controlling version ($C(U):=|0\rangle\langle0|\otimes I_*+|1\rangle\langle1|\otimes U$) in order:

$$
\begin{aligned}
\Phi_{L,+a}&:=&\sum_{j=0}^{2^L-1}\Phi_{L}|j+a\text{ mod }2^L\rangle\langle j|\Phi_{L}^\dagger,\tag{29}
\end{aligned}
$$

$$
\begin{aligned}
\Phi_{L,+a,\text{mod }N}&:=&\Phi_{L}\left(\sum_{j=0}^{N-1}|j+a\text{ mod }N\rangle\langle j|+\sum_{j=N}^{2^L-1}|j\rangle\langle j|\right)\Phi_{L}^\dagger,\tag{30}
\end{aligned}
$$

$$
\begin{aligned}
U_{L,+a\cdot\text{ mod }N}&:=&\sum_{f=0}^{2^L-1}|f\rangle\langle f|\otimes\left(\sum_{y=0}^{N-1}|(y+af)\text{ mod }N\rangle\langle y|+\sum_{y=N}^{2^L-1}|y\rangle\langle y|\right)\tag{31}
\end{aligned}
$$

where $N<2^L$, $\Phi_L$ is the quantum Fourier transform $\operatorname{QFT}_ L$ without the $\operatorname{SWAP}$ step. Here we can understand those gates roughly: $\Phi_{L,+a}$ implements the operation $+a$ modulo $2^L$; $\Phi_{L,+a,\text{mod }N}$ implements the operation $+a$ modulo $N$; $U_{L,+a\cdot\text{ mod }N}$ implements the operation $\times a$ modulo $N$, which is the first step in $(28)$.


#### Implements of $~\Phi$$_{L,+a}$

Considering

$$
\begin{aligned}
\operatorname{QFT}_{L}|j\rangle&=&\bigotimes_{k=0}^{L-1}\frac{|0\rangle+e^{2\pi ij*2^{-k-1}}|1\rangle}{2},\tag{32}
\end{aligned}
$$

$$
\begin{aligned}
\left(\bigotimes_{k=0}^{L-1}U_1(2\pi a*2^{-k-1})\right)\cdot
\operatorname{QFT}_{L}|j\rangle&=&\bigotimes_{k=0}^{L-1}\left(U_1(2\pi a*2^{-k-1})\cdot\frac{|0\rangle+e^{2\pi ij*2^{-k-1}}|1\rangle}{2}\right)
\end{aligned}
$$

$$
\begin{aligned}
&=&\bigotimes_{i=0}^{L-1}\frac{|0\rangle+e^{2\pi i(j+a)*2^{-k-1}}|1\rangle}{2}\tag{33}\\
&=&\operatorname{QFT}_{L}|j+a\text{ mod }2^L\rangle,
\end{aligned}
$$

we have

$$
\bigotimes_{k=0}^{L-1}U_1(2\pi a*2^{-k-1})=\sum_{j=0}^{2^L-1}\operatorname{QFT}_{L}|j+a\text{ mod }2^L\rangle\langle j|\operatorname{QFT}_{L}^\dagger.\tag{34}
$$

Since there is a sequence of $\operatorname{SWAP}$ gates in the implement of $\operatorname{QFT}$ which are unnecessary for the implement in $(34)$, we redefine $\Phi_L$ as $\operatorname{QFT}_{L}$ without the $\operatorname{SWAP}$ step. Thus we have

$$
\begin{aligned}
\Phi_{L}|j\rangle&=&\bigotimes_{k=0}^{L-1}\frac{|0\rangle+e^{2\pi ij*2^{k-L}}|1\rangle}{2},\tag{35}
\end{aligned}
$$

$$
\begin{aligned}
\left(\bigotimes_{k=0}^{L-1}U_1(2\pi a*2^{k-L})\right)\cdot
\Phi_{L}|j\rangle&=&\Phi_{L}|j+a\text{ mod }2^L\rangle,\tag{36}
\end{aligned}
$$

$$
\begin{aligned}
\bigotimes_{k=0}^{L-1}U_1(2\pi a*2^{k-L})&=&\sum_{j=0}^{2^L-1}\Phi_{L}|j+a\text{ mod }2^L\rangle\langle j|\Phi_{L}^\dagger,\tag{37}
\end{aligned}
$$

The following is the code for $\Phi_L$, where it is noted that ```QFT``` refers to $\operatorname{QFT}$ **WITHOUT** $\operatorname{SWAP}$ step in all codes in this subsection, which will be omitted after:

```python{.line-numbers}
def func_qft_without_swap(reg_system):
    """
    Quantum Fourier Transform without the swap step, |s> -> QFT|s>
    :param reg_system: |s>
    :return:
    """
    number_qubit = len(reg_system)
    for idx1_qubit in range(0, number_qubit - 1):  # The outer loop
        H(reg_system[idx1_qubit])  # Operate a H gate on the idx1-th qubit
        for idx2_qubit in range(2, number_qubit - idx1_qubit + 1):  # The inner loop
            # where we will operate a CU1 gate in each loop
            idx3_qubit = idx1_qubit + idx2_qubit - 1  # idx3 is the idx for the ctrlling qubit
            # idx1 is the ctrlled qubit and idx2 is related to the rotation angle
            CU(0, 0, 2 * pi / pow(2, idx2_qubit))(reg_system[idx3_qubit], reg_system[idx1_qubit])
    H(reg_system[number_qubit - 1])  # Do not forget there is a H gate operating on the last qubit
```

Then we have the decomposition and code implement for $\Phi_{L,+a}$:

![$\Phi_{+a}$](figures/PhiAdda.png)

```python{.line-numbers}
def func_qftadd(reg_system, int_adder):
    """
    a circuit implement the addition under the Fourier bases
    :param reg_system: QFT|s>, we write the state as a image of the Fourier transform
    :param int_adder: a
    :return: a circuit which implement the map: QFT|s> -> QFT|s+a>
    """
    for idx_qubit in range(len(reg_system)):  # For each qubit in the reg_s, we operate a U1 gate on it
        U(2 * pi * (int_adder % 2 ** (idx_qubit + 1)) / (2 ** (idx_qubit + 1)))(
            reg_system[-1 - idx_qubit])
```

#### Implements of $C^2(\Phi_{L,+a,\text{mod }N})$

Here we present the implement of $C^2(\Phi_{L,+a,\text{mod }N})$ directly. We can obtain the implements of $C(\Phi_{L,+a,\text{mod }N})$ and $\Phi_{L,+a,\text{mod }N}$ easily by removing the controlling qubits.

![$C^2(\Phi_{L,+a,\text{mod }N})$](figures/PhiAddMod.png)

Here $|c_1\rangle$ and $|c_2\rangle$ are two controlling qubits. The last line $|0\rangle$ is called a **zeroed qubit**, where as long as input state is $|0\rangle$, the output state must also be $|0\rangle$. It needs more explanation here that only when

$$
L\ge\lceil \log_2 N\rceil+1,\tag{38}
$$

and

$$
s=0,1,\cdots,N-1,\tag{39}
$$

the above quantum circuit $\mathcal C_{C^2(\Phi_{L,+a,\text{mod }N})}$ implements the mapping:

$$
\begin{aligned}
\mathcal C_{C^2(\Phi_{L,+a,\text{mod }N})}|c_1\rangle|c_2\rangle\left(\Phi_L|s\rangle\right)|0\rangle
&=&\left(C^2(\Phi_{L,+a,\text{mod }N})|c_1\rangle|c_2\rangle\left(\Phi_L|s\rangle\right)\right)|0\rangle\\
&=&|c_1\rangle|c_2\rangle\left(\Phi_L|s+c_1c_2a\text{ mod }N\rangle\right)|0\rangle.
\end{aligned}\tag{40}
$$

If the conditions are not satisfied, there may occur some leakage or some zeroed qubits may not reset to $|0\rangle$ and be entangled with other qubits. Meanwhile the evolution may not be as expected.

Additionally, quantum computing allows superposition. When two states are legal, their superposition states will also be legal. Thus the state in the third and fourth line may also be

$$
\Phi_L\sum_{s=0}^{N-1}p_s|s\rangle,\tag{41}
$$

where

$$
\sum_{s=0}^{N-1}|p_s|^2=1.\tag{42}
$$

For similar cases, we will not repeat it later and still say $s=0,\cdots,N-1$ simply.

In the following code implement, we denote the first and second line as ```reg_ctrlling```,the third and fourth line as ```reg_system``` with the third line as ```reg_system[0]```, the last line as ```qubit_zeroed```, integers $a$ and $N$ as ```int_adder``` and ```int_divisor```, respectively. Those functions named with the end ```_inverse``` denotes the inverse circuit of the corresponding function; those functions named with the head ```func_ctrl``` or ```func_double_ctrl``` denotes the control version or double control version of the corresponding function.

```python{.line-numbers}
def func_double_ctrl_qftaddmod(reg_system, reg_ctrlling, qubit_zeroed, int_adder, int_divisor):
    """
    CC-qftadd(int_adder)mod(int_divisor)
    |c_1>|c_2>QFT|s> -> <0|c_1*c_2>*|c_1>|c_2>QFT|s> + <1|c_1*c_2>*|c_1>|c_2>QFT|s+a mod d>
    the complement comes from the Figure 5 in arXiv quant-ph/0205095
    :param reg_system: QFT|s> with s < d
    :param reg_ctrlling: [|c_1>,|c_2>]
    :param qubit_zeroed: |0>
    :param int_adder: a with a < d
    :param int_divisor: d
    :return: |c_1>|c_2>QFT|s>|0> -> <0|c_1*c_2>*|c_1>|c_2>QFT|s>|0> + <1|c_1*c_2>*|c_1>|c_2>QFT|s+a mod d>|0>
    """
    func_double_ctrl_qftadd(reg_system, reg_ctrlling, int_adder)
    func_qftadd_inverse(reg_system, int_divisor)
    func_qft_without_swap_inverse(reg_system)
    CX(reg_system[0], qubit_zeroed)
    func_qft_without_swap(reg_system)
    func_ctrl_qftadd(reg_system, qubit_zeroed, int_divisor)
    func_double_ctrl_qftadd_inverse(reg_system, reg_ctrlling, int_adder)
    func_qft_without_swap_inverse(reg_system)
    X(reg_system[0])
    CX(reg_system[0], qubit_zeroed)
    X(reg_system[0])
    func_qft_without_swap(reg_system)
    func_double_ctrl_qftadd(reg_system, reg_ctrlling, int_adder)
```

Especially, the double control gate $C^2(U_1)$ is required in the double control version of some quantum subcircuits, whose implements are as follows:

```python{.line-numbers}
def CCU1(q1, q2, q3, float_theta):
    """
    a single-parameter three-qubit gate, which is the double-ctrl version for U1 gate,
    the matrix form of CCU1(theta) is the diagonal matrix {1,1,1,1,1,1,1,e^{i theta}}
    in fact we do not distinguish which qubit is the ctrlling or ctrlled qubit
    :param q1: a qubit
    :param q2: another qubit
    :param q3: a third qubit
    :param float_theta: the rotation angle
    :return: |q1>|q2>|q3> -> <0|q1*q2*q3>*|q1>|q2>|q3> + e^{i theta}*<1|q1*q2*q3>*|q1>|q2>|q3>
    """
    float_theta_half = float_theta / 2
    CU(0, 0, float_theta_half)(q2, q3)
    CX(q1, q2)
    CU(0, 0, -float_theta_half)(q2, q3)
    CX(q1, q2)
    CU(0, 0, float_theta_half)(q1, q3)
```

Afterword: Even though we say we want to implement quantum gates $\Phi_{+a,\text{mod }N}$, in fact it is a projection $B$ of its matrix form what we need. When we have a quantum circuit $\mathcal C$ satisfying

$$
\mathcal C\cdot\Phi_{L}\sum_{j=0}^{N-1}|j\rangle\langle j|\Phi_{L}^\dagger=B:=\Phi_{L,+a,\text{mod }N}\cdot\Phi_{L}\sum_{j=0}^{N-1}|j\rangle\langle j|\Phi_{L}^\dagger,\tag{43}
$$

we can regard $\mathcal C$ as an effectively implement of $B$. We call this method block-encoding that encoding effective information $B$ into a quantum circuit $\mathcal C$ as a projection. Meanwhile, $\mathcal C$ has form:

$$
\begin{aligned}
\mathcal C&:=&\Phi_{L}\left(\sum_{j=0}^{N-1}|j+a\text{ mod }N\rangle\langle j|+\sum_{j=N}^{2^L-1}|\tilde j\rangle\langle j|\right)\Phi_{L}^\dagger,\tag{44}
\end{aligned}
$$

$$
\begin{aligned}
\Phi_{L,+a,\text{mod }N}&:=&\Phi_{L}\left(\sum_{j=0}^{N-1}|j+a\text{ mod }N\rangle\langle j|+\sum_{j=N}^{2^L-1}|j\rangle\langle j|\right)\Phi_{L}^\dagger.\tag{45}
\end{aligned}
$$

where $\{|\tilde j\rangle\}$ should be an orthonormal bases of $\text{Span}\{|j\rangle\,|\,j=N,\cdots2^L-1\}$. It is clear to see that $\Phi_{L,+a,\text{mod }N}$ is a special case of implements for $\mathcal C$.

#### Implements of $U_{L,+a\cdot\text{ mod }N}$

Recall the definition of $U_{L,+a\cdot\text{ mod }N}$ as follows:

$$
U_{L,+a\cdot\text{ mod }N}:=\sum_{f=0}^{2^L-1}|f\rangle\langle f|\otimes\left(\sum_{y=0}^{N-1}|(y+af)\text{ mod }N\rangle\langle y|+\sum_{y=N}^{2^L-1}|y\rangle\langle y|\right).\tag{46}
$$

Expand $f$ as binary to $f=\sum_{j=0}^*2^jf_j$, and we decompose the multiplication into additions:

$$
af=\sum_{j=0}^*2^jf_j\cdot a.\tag{47}
$$

Furtherly, we have

$$
\begin{aligned}
&&s+af\text{ mod }N=\left(\sum_{j=0}^*2^jf_j\cdot a\right)\text{ mod }N\\
&=&s+_{\text{mod }N}(2^0a\text{ mod }N)f_0+_{\text{mod }N}(2^1a\text{ mod }N)f_1+_{\text{mod }N}\cdots+_{\text{mod }N} (2^*a\text{ mod }N)f_*,
\end{aligned}\tag{48}
$$

where $+_{\text{mod }N}$ denotes the sum modulo $N$, which make sure that the output satisfies the input condition, i.e. outputs stay in the integer interval $\{0,1,\cdots,N-1\}$.

![$C^2(\Phi_{+a,\text{mod }N})$](figures/UAddProdMod.png)

It is checked that if $L\ge\lceil \log_2 N\rceil+1$ and $s=0,1,\cdots N-1$, the above circuit satisfies

$$
\mathcal C_{U_{L,+a\cdot\text{ mod }N}}|c\rangle|f\rangle|s\rangle
={U_{L,+a\cdot\text{ mod }N}}|c\rangle|f\rangle|s\rangle
=|c\rangle|f\rangle|s+caf\text{ mod }N\rangle.\tag{49}
$$

It is worth noting that there're requirements for $|s\rangle$ but no for $|f\rangle$. Finally we denote $|s\rangle$,$|f\rangle$,$|c\rangle$ as ```reg_system```,```reg_factor_1```, ```qubit_ctrlling```, and $a,N$ as ```int_factor_2```,```int_divisor```, respectively, and introduce ```qubit_zeroed``` as the zeroed qubit needed in the implement of $\mathcal C_{C^2(\Phi_{L,+a,\text{mod }N})}$. Then we have a code implement as following:

```python{.line-numbers}
def func_ctrl_addprodmod(reg_system, reg_factor_1, qubit_ctrlling, qubit_zeroed, int_factor_2, int_divisor):
    """
    :param reg_system: |s> with s < d
    :param reg_factor_1: |f_1>, a quantum state encoding the factor_1
    :param qubit_ctrlling: |c>
    :param qubit_zeroed: |0>
    :param int_factor_2: f_2, a classical data
    :param int_divisor: d
    :return: |c>|f_1>|s> -> <0|c>|c>|f_1>|s> + <1|c>|c>|f_1>|s+ f_1*f_2 mod d>
    the complement comes from the Figure 6 in arXiv quant-ph/0205095
    """
    func_qft_without_swap(reg_system)
    for idx_qubit in range(len(reg_factor_1)):  # For each qubit in reg_f_1, we will operate a CC-qftaddmod gate where
        # regarding idx_qubit as one of the two ctrlling qubit
        func_double_ctrl_qftaddmod(reg_system, [qubit_ctrlling, reg_factor_1[-1 - idx_qubit]], qubit_zeroed,
                                   (int_factor_2 * (2 ** idx_qubit)) % int_divisor, int_divisor)
    func_qft_without_swap_inverse(reg_system)
```

#### Implements of $U_{a,N}$

Considering

$$
s-a^{-1}as\equiv 0\pmod N,\tag{50}
$$

where $a^{-1}$ is the inverse of $a$ modulo $N$ ($a^{-1}a\equiv1\pmod N$), which can be computed by extended Euclidean algorithm effectively, then the procedure in formula $(28)$ can be represented as

$$
\begin{aligned}
|s\rangle|0\rangle
\rightarrow U_{L,+a\cdot\text{ mod }N}|s\rangle|0\rangle
=|s\rangle|as\rangle
\stackrel{\operatorname{SWAP}}\longrightarrow|as\rangle|s\rangle
\rightarrow U_{L,+a^{-1}\cdot\text{ mod }N}^\dagger|as\rangle|s\rangle
=|as\rangle|0\rangle.
\end{aligned}\tag{51}
$$

We show the circuit decomposition of $C(U_{a,N})$ as following:

![$C^2(\Phi_{+a,\text{mod }N})$](figures/UaN.png)

Here $f < N$, $|0\rangle$ is a sequence of zeroed qubits, and $\operatorname{SWAP}$ is a sequence of $\operatorname{SWAP}$ gates operating on corresponding qubits in two registers. It is noted that we need $\lceil\log_2 N\rceil$ qubits to encode the quantum state $|f\rangle$, and $\lceil\log_2 N\rceil+1$ to $|0\rangle$. Also noted that the $\operatorname{SWAP}$ gates align from low position of the two registers, and there's no $\operatorname{SWAP}$ gates operating on the highest position of $|0\rangle$. A highest zeroed qubit is needed in the implement of $C^2(\Phi_{L,+a,\text{mod }N})$ (as a carry), so $|0\rangle$ has one qubit more than $|f\rangle$.

Then we show the code implement. Here we denote $|f\rangle$,$|0\rangle$,$|c\rangle$ as ```reg_system```,```reg_zeroed```, ```qubit_ctrlling```, and $a,N$ as ```int_factor```,```int_divisor```, respectively, and introduce ```qubit_zeroed``` as the zeroed qubit required in the implement of $\mathcal C_{C^2(\Phi_{L,+a,\text{mod }N})}$.

```python{.line-numbers}
def func_ctrl_multmod(reg_system, reg_zeroed, qubit_ctrlling, qubit_zeroed, int_factor, int_divisor):
    """
    |c>|s> -> <0|c>|c>|s>  + <1|c>|c>|s * f mod d>
    the complement comes from the Figure 7 in arXiv quant-ph/0205095
    :param reg_system: |s>
    :param reg_zeroed: |0*>, a register initialled into |0*>
    :param qubit_ctrlling: |c>
    :param qubit_zeroed: |0>, a qubit at state |0>
    :param int_factor: f
    :param int_divisor: d
    :return: |c>|s>|0*>|0> -> <0|c>|c>|s>|0*>|0>  + <1|c>|c>|s * f mod d>|0*>|0>
    """
    func_ctrl_addprodmod(reg_zeroed, reg_system, qubit_ctrlling, qubit_zeroed, int_factor, int_divisor)
    for idx_qubit in range(min(len(reg_system), len(reg_zeroed))):  # We CSWAP the corresponding qubit in those two reg
        # from the end since maybe the two reg has different length
        CSWAP(qubit_ctrlling, reg_system[-1 - idx_qubit], reg_zeroed[-1 - idx_qubit])
    func_ctrl_addprodmod_inverse(reg_zeroed, reg_system, qubit_ctrlling, qubit_zeroed, pow(int_factor, -1, int_divisor),
                                 int_divisor)
```

Here $a^{-1}\pmod N$ is computed by calling the function ```pow``` in python.

#### Implements of Quantum Order-Finding Algorithm

Denote ```int_factor```,```int_divisor```,``` int_shots```,```number_qubit_ancilla``` as the base, the modulus, the shot number for each quantum circuit, the number of ancilla qubits in the quantum phase estimation algorithm, respectively. Then we have the code implement of quantum order-finding algorithm as follows:

```python{.line-numbers}
def func_quantum_order_finding(int_factor, int_divisor, int_shots, number_qubit_ancilla):
    """
    :param int_factor: f
    :param int_divisor: d
    :param int_shots: the shots number for each quantum circuit
    :param number_qubit_ancilla: the number of qubits for the estimating the phase
    :return: an estimation for the fraction r/ord(f,d)/2^t, where ord(f,d) is the order of f mod d, r is a random in
             {0,1,...,ord(f,d)-1}, and t = number_qubit_ancilla as following corresponding to the precision.
    """
    # Create the quantum environment
    env = QEnv()
    Define.hubToken = ''
    # Choose backend
    env.backend(BackendName.LocalBaiduSim2)
    # env.backend(BackendName.CloudBaiduSim2Water)

    # Decide the number of qubit which will be used to encode the eigenstate
    number_qubit_system = int(math.ceil(math.log(int_divisor, 2)))

    # Create the quantum register
    # The ancilla qubit used for phase estimation
    reg_ancilla = [env.Q[idx_qubit] for idx_qubit in range(0, number_qubit_ancilla)]
    number_qubit_part2 = number_qubit_ancilla + number_qubit_system
    # The system register holding the eigenstate
    reg_system = [env.Q[idx_qubit] for idx_qubit in range(number_qubit_ancilla, number_qubit_part2)]
    number_qubit_part3 = number_qubit_ancilla + 2 * number_qubit_system + 1
    # The zeroed register used in the circuit of func_ctrl_multmod
    reg_zeroed = [env.Q[idx_qubit] for idx_qubit in range(number_qubit_part2, number_qubit_part3)]
    qubit_zeroed = env.Q[number_qubit_part3]  # The other zeroed qubit used in the circuit of func_ctrl_multmod

    # Initialise the state |0...01> as a superposition of concerned eigenstates
    X(reg_system[-1])

    # The following is the quantum phase estimation algorithm
    for idx_qubit in range(len(reg_ancilla)):
        H(reg_ancilla[idx_qubit])
        func_ctrl_multmod(reg_system, reg_zeroed, reg_ancilla[idx_qubit], qubit_zeroed, pow(int_factor, 2 ** idx_qubit,
                                                                                            int_divisor), int_divisor)
    func_qft_without_swap_inverse(reg_ancilla)

    # We only measure the reg_ancilla, which gives the estimation of the phase
    MeasureZ(reg_ancilla, range(number_qubit_ancilla))

    env.module(CompositeGateModule())

    return func_qof_data_processing(int_divisor, number_qubit_ancilla, env.commit(
        int_shots, fetchMeasure=True)["counts"])
```

There're five parts in this code:

1. set the quantum register;
2. initial the quantum state;
3. call the quantum circuit for quantum phase estimation algorithm;
4. measure the final quantum state;
5. call the function for data post processing.

The following function is used for data post processing:

```python{.line-numbers}
def func_qof_data_processing(int_divisor, number_qubit_ancilla, dict_task_result_counts):
    """
    :param int_divisor: d
    :param number_qubit_ancilla: the number of qubits for the estimating the phase
    :param dict_task_result_counts: a dict storing the counts data in the task result
    :return: a dict {"order":shots} storing the order and the shots, such as {"2":5,"4":7} means that 5 shots indicate
             the order may be 2 and 7 shots indicate the order may be 4.
    dict_task_result_counts of form {"quantum_output":shots} is a quantum output from an estimation for the fraction
    r/ord(f,d)/2^t, where ord(f,d) is the order of f mod d, r is a random in {0,1,...,ord(f,d)-1}, and
    t = number_qubit_ancilla as following corresponding to the precision.
    For the case that the ancilla is enough we compute the maximal denominator, and for the case that the ancilla is not
    enough we compute all the possible denominators.
    """
    dict_order = {}
    # The case that the number of ancilla is enough
    if number_qubit_ancilla >= math.log(int_divisor, 2) * 2:
        for idx_key in dict_task_result_counts.keys():  # We need to transform the key in dict_task
            if dict_task_result_counts[idx_key] <= 0:  # Skip the measurement results with counts <= 0
                continue
            # From a numerator to the order by calling func_result_to_order
            int_order_maybe = func_result_to_order(int(idx_key[::-1], 2), number_qubit_ancilla, int_divisor)
            str_int_order_maybe = "{0}".format(int_order_maybe)
            if str_int_order_maybe not in dict_order.keys():
                dict_order[str_int_order_maybe] = dict_task_result_counts[idx_key]
            else:
                dict_order[str_int_order_maybe] += dict_task_result_counts[idx_key]
    else:  # The case that the number of ancilla is not enough
        for idx_key in dict_task_result_counts.keys():  # We need to transform the key in dict_task
            # from a numerator to the order by calling func_result_to_order
            if dict_task_result_counts[idx_key] <= 0:  # Skip the measurement results with counts <= 0
                continue
            list_int_order_maybe = func_result_to_order_list(int(idx_key[::-1], 2), number_qubit_ancilla, int_divisor)
            for int_order_maybe in list_int_order_maybe:
                str_int_order_maybe = "{0}".format(int_order_maybe)
                if str_int_order_maybe not in dict_order.keys():
                    dict_order[str_int_order_maybe] = dict_task_result_counts[idx_key]
                else:
                    dict_order[str_int_order_maybe] += dict_task_result_counts[idx_key]
    return dict_order
```

whose input is the measurement result of the quantum circuit ```dict_task_result_counts```, the number of ancilla qubits ```number_qubit_ancilla``` for computing the denominator of $k$ and the modulus ```int_divisor``` for bounding the order. Here ```dict_task_result_counts``` is a ```dict``` in python, the keys of whose items are quantum states as a ```str``` and values are corresponding measurement times. The output of this function is also a ```dict```, the keys of whose items are "orders", and values are also corresponding measurement times. We use "orders" here, because these "order" may be wrong, but are just results after data processing. There are two cases to distinguish whether the ancilla qubits introduced are enough. If the ancilla qubits are enough, we select the maximal denominator less than the modulus of each convergent as the returns of continued fraction expansion algorithm, corresponding to the function ```func_result_to_order```; otherwise we list the denominators of all convergent as the returns, corresponding to the function ```func_result_to_order_list```. Here ```func_result_to_order``` and ```func_result_to_order_list``` are classical computing, whose essence is just Euclidean algorithm, where we won't go into much detail.

Finally, we come back to

#### Implements of Shor's Algorithm

```python{.line-numbers}
def func_Shor_algorithm(int_divisor, number_qubit_ancilla=None, int_shots=2, int_factor=None):
    """
    We want to factor the int int_divisor
    :param int_divisor: d, which we want to factor
    :param number_qubit_ancilla: the number of qubit which will be used for quantum phase estimation
    :param int_shots: the number of shots whose default value is 2; when int_shots > 2 means we want to know the
                      distribution of the state after quantum phase estimation
    :param int_factor: an integer whose order will be computed by the quantum order finding algorithm
    :return: a factor of d
    here it will print the computation process such as ord(4 mod 15) = 2 and the factorization such as 15 = 3 * 5,
    where "not quantum" means the current factorization comes from a classical part of Shor's algorithm.
    For int_shots > 2, we will print the number of shots where we obtain a correct factorization.
    """
    # Some classical cases
    if int_divisor < 0:
        int_divisor = -int_divisor
    if int_divisor == 0:
        print("{0} is zero.".format(int_divisor))
        return 0
    if int_divisor == 1:
        print("{0} is unit.".format(int_divisor))
        return 1
    elif isprime(int_divisor):
        print("{0} is prime.".format(int_divisor))
        return 1
    else:
        # The case that d is a power of some prime
        for idx_int in range(2, int(math.floor(math.log(int_divisor, 2))) + 1):
            if pow(int(pow(int_divisor, 1 / idx_int)), idx_int) == int_divisor:
                print("{0[0]} is a power of {0[1]}.".format([int_divisor, int(pow(int_divisor, 1 / idx_int))]))
                return idx_int
        # The case that d is even
        if int_divisor % 2 == 0:
            print("{0[0]} = {0[1]} * {0[2]}".format([int_divisor, 2, int_divisor // 2]))
            return 2
        else:
            # Generate a random f (int_factor) which can be assigned when called
            if int_factor is None:
                int_factor = randint(2, int_divisor - 2)
            # If the input int_factor is invalid to introduce the factorization, we reset the value of int_factor
            elif int_factor % int_divisor == 1 or int_factor % int_divisor == int_divisor - 1 or \
                    int_factor % int_divisor == 0:
                print('The value of int_factor is invalid!')
                return func_Shor_algorithm(int_divisor, number_qubit_ancilla=number_qubit_ancilla, int_shots=int_shots)
            int_gcd = math.gcd(int_factor, int_divisor)
            # The case that f and d are not co-prime
            if int_gcd != 1:
                print('{0[0]} = {0[1]} * {0[2]}, not quantum'.format([int_divisor, int_gcd, int_divisor // int_gcd]))
                return int_gcd
            else:
                # From now on, we entry the quantum part
                number_qubit_system = int(math.ceil(math.log(int_divisor, 2)))
                # The number_qubit_ancilla used for phase estimation should be 2 * n_q_s + 4 such that the successful
                # probability of the phase estimation will > 98.6%
                if number_qubit_ancilla is None:
                    number_qubit_ancilla = 2 * number_qubit_system + 4
                # The case that the number of ancilla is enough
                if number_qubit_ancilla >= math.log(int_divisor, 2) * 2:
                    # A dict storing the possible order and its corresponding number of shots
                    dict_order = func_quantum_order_finding(int_factor, int_divisor, int_shots, number_qubit_ancilla)
                    # The list of possible order
                    list_order = [int(idx_key) for idx_key in dict_order.keys()]
                    int_order = 0
                    if int_shots == 1 or int_shots == 2 or len(list_order) == 1:
                        if int_shots == 1 or len(list_order) == 1:
                            int_order = list_order[0]
                        elif int_shots == 2 and len(list_order) == 2:
                            # For two shots, we compute the least common multiple as the order
                            int_order = int(list_order[0] * list_order[1] / math.gcd(list_order[0], list_order[1]))
                        int_pow_half = pow(int_factor, int_order // 2, int_divisor)
                        # To check whether int_factor and its order can introduce the factorization
                        if pow(int_factor, int_order, int_divisor) == 1 and int_order % 2 == 0 and int_pow_half != \
                                int_divisor - 1 and int_pow_half != 1:
                            print('ord({0[0]} mod {0[1]}) = {0[2]}'.format([int_factor, int_divisor, int_order]))
                            # An f which satisfies some appropriate conditions will give a factorization of d
                            print('{0[0]} = {0[1]} * {0[2]}'.format(
                                [int_divisor, math.gcd(int_divisor, int_pow_half - 1),
                                 math.gcd(int_divisor, int_pow_half + 1)]))
                            return math.gcd(int_divisor, int_pow_half - 1)
                        else:
                            # Maybe we compute a wrong order, maybe f and its order cannot give the factorization of d
                            print('Perhaps ord({0[0]} mod {0[1]}) = {0[2]},\n'.format([int_factor, int_divisor,
                                                                                       int_order]))
                            print('but it cannot give the factorization of {0}.'.format(int_divisor))
                            # We haven't compute the correct order and need to recompute
                            func_Shor_algorithm(int_divisor, number_qubit_ancilla=number_qubit_ancilla,
                                                int_shots=int_shots)
                    elif int_shots > 2:
                        # Here we use sympy to compute the order to confirm which possible order is correct
                        int_order_true = n_order(int_factor, int_divisor)
                        if int_order_true in list_order:
                            print('We obtain ord({0[0]} mod {0[1]}) = {0[2]} for {0[3]} of {0[4]} times.'.format(
                                [int_factor, int_divisor, int_order_true, dict_order["{0}".format(int_order_true)],
                                 int_shots]))
                            int_pow_half_true = pow(int_factor, int_order_true // 2, int_divisor)
                            # To check whether int_factor and its order can introduce the factorization
                            if pow(int_factor, int_pow_half_true, int_divisor) == 1 and int_order_true % 2 == 0 and \
                                    int_pow_half_true != int_divisor - 1:
                                print('{0[0]} = {0[1]} * {0[2]}'.format([int_divisor, math.gcd(
                                    int_divisor, int_pow_half_true - 1), math.gcd(int_divisor, int_pow_half_true + 1)]))
                                return math.gcd(int_divisor, int_pow_half_true - 1)
                            else:  # int_factor cannot introduce the factorization of d
                                print('But it cannot give the factorization of {0}.'.format(int_divisor))
                                func_Shor_algorithm(int_divisor, number_qubit_ancilla=number_qubit_ancilla,
                                                    int_shots=int_shots)
                        else:
                            # We haven't compute the correct order and need to recompute
                            print("we haven't computed the correct order of {0[0]} mod {0[1]}.".format(
                                [int_factor, int_divisor]))
                            func_Shor_algorithm(int_divisor, number_qubit_ancilla=number_qubit_ancilla,
                                                int_shots=int_shots)
                else:  # The case that the number of ancilla is not enough, where the cost is more.
                    print("Since the ancilla qubits are not enough to estimate the order with a high probability,")
                    print("we need to traverse all probable denominator in the step of continued fraction expansions.")
                    print("Thus some order may be obtained for twice as many times as the number of shots.\n")
                    # A dict storing the possible order and its corresponding number of shots
                    dict_order = func_quantum_order_finding(int_factor, int_divisor, int_shots, number_qubit_ancilla)
                    # The list of possible order
                    list_order = [int(idx_key) for idx_key in dict_order.keys()]
                    int_order = 0
                    if int_shots == 1 or len(list_order) == 1:  # For the case 1-shot
                        for int_order_maybe in list_order:  # Check "maybe" is the correct order one by one
                            if int_order_maybe != 0 and pow(int_factor, int_order_maybe, int_divisor) == 1:
                                int_order = int_order_maybe  # If correct, recorded in int_order
                                break
                        int_pow_half = pow(int_factor, int_order // 2, int_divisor)
                        # To check whether int_factor and its order can introduce the factorization
                        if pow(int_factor, int_order, int_divisor) == 1 and int_order % 2 == 0 and int_pow_half != \
                                int_divisor - 1 and int_pow_half != 1:
                            print("ord({0[0]} mod {0[1]}) = {0[2]}".format([int_factor, int_divisor, int_order]))
                            # An f which satisfies some appropriate conditions will give a factorization of d
                            print("{0[0]} = {0[1]} * {0[2]}".format(
                                [int_divisor, math.gcd(int_divisor, int_pow_half - 1),
                                 math.gcd(int_divisor, int_pow_half + 1)]))
                            return math.gcd(int_divisor, int_pow_half - 1)
                        else:
                            # Maybe we compute a wrong order, maybe f and its order cannot give the factorization of d
                            if int_order == 0:  # We haven't computed the correct order of int_factor
                                print("We haven't computed the correct order of {0[0]} mod {0[1]}.\n".format(
                                    [int_factor, int_divisor]))
                            else:  # int_factor cannot introduce the factorization of d
                                print("Perhaps ord({0[0]} mod {0[1]}) = {0[2]},".format(
                                    [int_factor, int_divisor, int_order]))
                                print("but it cannot give the factorization of {0}.".format(int_divisor))
                            # And we need to recompute
                            func_Shor_algorithm(int_divisor, number_qubit_ancilla=number_qubit_ancilla,
                                                int_shots=int_shots)
                    elif int_shots > 1:  # For the case int_shots > 1
                        # Here we use sympy to compute the order to confirm which possible order is correct
                        int_order_true = n_order(int_factor, int_divisor)
                        if int_order_true in list_order:
                            print("We obtain ord({0[0]} mod {0[1]}) = {0[2]} for {0[3]} of {0[4]} times.".format(
                                [int_factor, int_divisor, int_order_true, dict_order["{0}".format(int_order_true)],
                                 int_shots]))
                            int_pow_half_true = pow(int_factor, int_order_true // 2, int_divisor)
                            # To check whether int_factor and its order can introduce the factorization
                            if pow(int_factor, int_pow_half_true, int_divisor) == 1 and int_order_true % 2 == 0 and \
                                    int_pow_half_true != int_divisor - 1:
                                print("{0[0]} = {0[1]} * {0[2]}".format([int_divisor, math.gcd(
                                    int_divisor, int_pow_half_true - 1), math.gcd(int_divisor, int_pow_half_true + 1)]))
                                return math.gcd(int_divisor, int_pow_half_true - 1)
                            else:  # int_factor cannot introduce the factorization of d
                                print("But it cannot give the factorization of {0}.".format(int_divisor))
                                func_Shor_algorithm(int_divisor, number_qubit_ancilla=number_qubit_ancilla,
                                                    int_shots=int_shots)
                        else:
                            # We haven't compute the correct order and need to recompute
                            print("we haven't computed the correct order of {0[0]} mod {0[1]}.\n".format(
                                [int_factor, int_divisor]))
                            func_Shor_algorithm(int_divisor, number_qubit_ancilla=number_qubit_ancilla,
                                                int_shots=int_shots)
```

We denote the integer $N$ to be factored as ```int_divisor```, the integer $a\in\{2,\cdots,N-2\}$ selected in the first iteration as ```int_factor``` and to be randomly generated in the ```38th``` line by default, the number of ancilla qubits in quantum phase estimation algorithm as ```number_qubit_ancilla``` to be set a little larger than twice the number of qubits encoding $N$ in the ```55th``` line by default, and the operation times of each quantum circuit as ```int_shots``` to be set $2$ corresponding to the step 2 in algorithm 7 by default.

We reduce the cases for $N$ negative, $0$, $1$, prime, even, powers of integers in the ```14-35th``` lines of the above codes, corresponding to the input condition in algorithm 3. The ```37-49th``` lines are corresponding to the step 1-3 of algorithm 3. Then there're also two cases to distinguish whether the ancilla qubits are enough. If the ancilla qubits are enough or the shots number $\le 2$, we prefer trying to factor $N$; otherwise, we prefer estimating the probability of obtaining correct factorization. We achieve the result from quantum order-finding algorithm in the ```60th``` or ```119th``` line. For the case trying to factor $N$, we determine whether $a$ is an inducer in the ```72nd``` or ```130th``` line, corresponding to the step 5 in algorithm 3, and furtherly give the factorization in the ```76th``` or ```134th``` line. For the case estimating probabilities, we call function ```sympy.ntheory.residue_ntheory.n_order``` to obtain correct order directly in ```90th``` and ```152nd``` line, and then compare it with the result from quantum order-finding algorithm to obtain the frequency for obtaining correct factorization.

We can call Shor's algorithm as follows:

```python
func_Shor_algorithm(15, number_qubit_ancilla=8, int_shots=2, int_factor=2)
```

The results of a run are following:

```python
Shots 2
Counts {'00000010': 1, '00000001': 1}
State None
Seed 245569767
ord(2 mod 15) = 4
15 = 3 * 5
```

We can see that the quantum order-finding algorithm finds the correct order of $2$ modulo $15$.

## Reference

[1] Pomerance, Carl. "A tale of two sieves." Notices Amer. Math. Soc. 1996.

[2] Shor, Peter W. "Algorithms for quantum computation: discrete logarithms and factoring." Proceedings 35th annual symposium on foundations of computer science. IEEE, 1994.

[3] Agrawal, Manindra, Neeraj Kayal, and Nitin Saxena. "PRIMES is in P." Annals of Mathematics (2004): 781-793.

[4] Lenstra Jr, H. W., and Carl Pomerance. "Primality testing with Gaussian periods. Archived version 20110412, Dartmouth College, US." (2011).

[5] Nielsen, Michael A., and Isaac L. Chuang. "Quantum computation and quantum information." Phys. Today 54.2 (2001): 60.

[6] Vedral, Vlatko, Adriano Barenco, and Artur Ekert. "Quantum networks for elementary arithmetic operations." Physical Review A 54.1 (1996): 147.

[7] Beauregard, Stephane. "Circuit for Shor's algorithm using 2n+3 qubits." arXiv preprint quant-ph/0205095 (2002).