# Hamiltonian Simulation

*Copyright (c) 2022 Institute for Quantum Computing, Baidu Inc. All Rights Reserved.*

## Definition

In this page, we continue to introduce **Hamiltonian Simulation**. Suppose the Hamiltonian $\check H$ of a given system is time-independent, according to Schrödinger equation (in natural units)

$$
i\frac{d}{dt}|\psi(t)\rangle=\check H|\psi(t)\rangle,
$$

we find the relationship between the quantum states of the system at $\tau$ time and $0$ time is

$$
|\psi(\tau)\rangle= e^{-i\check H\tau}|\psi(0)\rangle.
$$

As long as we can realize the operator $e^{-i\check H\tau}$ quantumly, we can simulate the $\tau$ time evolution of such system. We call $e^{-i\check H\tau}$ the $\tau$ time evolution operator of Hamiltonian $\check H$, and call the process of using quantum circuits to approximate the time evolution operator **Digital Quantum Simulation**.

Specifically, we need to construct a quantum circuit $\check C$ after inputting some encode of Hamiltonian $\check H$ and time $\tau$, so that the error between $\check C$ and $e^{-i\check H\tau}$ is less than the given precision $\epsilon$. Later, we will continue to realize digital quantum simulation around quantum eigenvalue transformation and quantum singular value transformation.

## Splitting Transformation Functions

As mentioned in previous sections, we assume the Hamiltonian $\check H$ of an $n$-qubit system is encoded in the upper left corner of an $m+n$ qubit Hermitian oracle $\check U$. By QET, if we found processing parameters $\Phi$, such that the QSP function $P_\Phi(x)\approx e^{-i\tau x}$, then QET circuit $W_\Phi(\check H,\check Z)$ would be approximate to a block-encoding of $e^{-i\check H\tau}$.

**Remark**. Considering $e^{-i\check H\tau}=e^{-i\left(\check H/s\right)\cdot\left(s\tau\right)}$, when $\|\check H\|_2>1$, we assume $\check U$ is a block-encoding of $\check H/s$ for some $s>\|\check H\|_2$, and then realize the $s\tau$ time evolution operator of Hamiltonian $\check H/s$ instead, where $\|\check H\|_2$ is defined as the maximal singular value of $\check H$ (which is also the maximal eigenvalue here).

Because the QSP function has parity limitations itself, in order to improve the simulation accuracy better, we split the QSP function into two parts, such as

$$
e^{-i\tau x}=\cos(\tau x)-i\sin(\tau x),
$$

and then find processing functions $\Phi_{\mathfrak{R}}$ and $\Phi_{\mathfrak{I}}$, such that

$$
\forall\,x\in[-1,1],\ P_{\Phi_{\mathfrak{R}}}\left(x\right)\approx \cos(\tau x),\ P_{\Phi_{\mathfrak{I}}}\left(x\right)\approx -i\sin(\tau x).
$$

Then we could use LCU of block-encodings to obtain the block-encoding of $\frac{P_{\Phi_{\mathfrak{R}}}(\check H)+ P_{\Phi_{\mathfrak{I}}}(\check H)}{2}\approx\frac{e^{-i\check H\tau}}{2}$.

![时间演化算符的一半](./figures/HS-HS_half.JPG)


## Approximate Polynomials

In the previous subsection, we mentioned that we need to find the processing parameters to make the QSP function approximate to the trigonometric functions. The actual method is to find the approximation polynomials of the trigonometric functions first, and then start from the approximation polynomials to find the corresponding processing parameters.

Consider Jacobi-Anger expansion$^{[6]}$

$$
\begin{aligned}
&\cos(\tau x)= \frac12J_0(\tau)+\sum_{j=1}^\infty J_{2j}(\tau)T_{4j}(x),\\
&\sin(\tau x)= \sum_{j=1}^\infty J_{2j-1}(\tau)U_{4j-3}(x), 
\end{aligned}\tag{1}
$$

where $T_k(x)=\cos(k\arccos x)$ and $U_k(x)=\sin((k+1)\arccos x)/\sqrt{1-x^2}$ are Chebyshev polynomials of the first and second kind, respectively, and

$$
J_k(\tau)=\frac{1}{\pi}\int_0^\pi \cos\left(kt-\tau\sin t\right)dt 
$$

is Bessel function of the first kind. It is shown that the two series in $(1)$ converge exponentially, so when we choose an appropriate truncation for the two series, we will obtain approximated polynomials $A_{\Phi_{\mathfrak{R}}}$ and $A_{\Phi_{\mathfrak{I}}}$. Finally, we can compute such processing parameters $\Phi_{\mathfrak{R}}$ and $\Phi_{\mathfrak{I}}$ by those algorithms for processing parameters computing, such that

$$
P_{\Phi_{\mathfrak{R}}}\approx f_{\cos},\ P_{\Phi_{\mathfrak{I}}}\approx -if_{\sin}.
$$

## Symmetric Quantum Signal Processing

However, in practice, in order to realize Hamiltonian simulation for a longer period of time, we adopt optimization algorithm based on SQSP with better algorithm stability to find $\Phi_{\mathfrak{R}}$ and $\Phi_{\mathfrak{I}}$, such that

$$
\forall\,x\in[-1,1],\ \frac{P_{\Phi_{\mathfrak{R}}}(x)+P_{\Phi_{\mathfrak{R}}}^*(x)}{2}\approx \frac{f_{\cos}(x)}{2},\ \frac{P_{\Phi_{\mathfrak{I}}}(x)+P_{\Phi_{\mathfrak{I}}}^*(x)}{2}\approx \frac{f_{\sin}(x)}{2}.
$$

Furthermore, we have

$$
\frac{(P_{\Phi_{\mathfrak{R}}}(x)+P_{\Phi_{\mathfrak{R}}}^*(x))-i(P_{\Phi_{\mathfrak{I}}}(x)+P_{\Phi_{\mathfrak{I}}}^*(x))}{4}\approx\frac{f_{\cos}(x)-if_{\sin}(x)}{4}\approx\frac{e^{-i\tau x}}{4},
$$

and can realize a QET circuit $\check C$ approximate to a block-encoding of $e^{-i\check H\tau}/4$ based on LCU of block-encodings:

$$
\langle0|_{cba}\check C_{cbas}|0\rangle_{cba}\approx\left(e^{-i\check H\tau}/4\right)_s.
$$

We denote it in circuit representation as follows.

![C 块编码了 exp(-iHt)/4](./figures/HS-BE_C.JPG)

## Fixed-Point Amplitude Amplification

For the case that $\check C$ is approximate to a block-encoding of $e^{-i\check H\tau}/4$, the singular values of the block of $\check C$ are all approximate to $1/4$. As long as we find an odd QSP function $P_\Phi$ with $z:=P_\Phi(1/4)$ a unit complex number, the QSVT circuit $W_\Phi(\check C,\hat Z)$ will be approximate to a block-encoding of 

$$
P_\Phi^{(SV)}\left(e^{-i\check H\tau}/4\right)\approx ze^{-i\check H\tau}.
$$

Omitting the global phase factor $z$, we have obtained a circuit $W_\Phi(\check C,\hat Z)$ approximate to a block-encoding of $e^{-i\check H\tau}$, where

$$
\hat Z=(2|0\rangle\langle0|-I)_{cb}.
$$

Set

$$
\begin{aligned}
\Phi_{\operatorname{AA}}:=\left(\hat\varphi_0,\hat\varphi_1,\cdots,\hat\varphi_7\right):=\frac{1}{2}\left(0,-\arccos\frac{1}{3},-\arccos\frac{7}{9},0,0,\arccos\frac{7}{9},\arccos\frac{1}{3},0\right)\in\mathbb R^8,
\end{aligned}
$$
 
we have 

$$
P_{\Phi_{\operatorname{AA}}}(x)=\frac{1024 x^7}{27}-\frac{640 x^5}{9}+\frac{364 x^3}{9}-\frac{169 x}{27}\in\mathbb R[x]
$$

satisfies $P_{\Phi_{\operatorname{AA}}}(1/4)=-1$. So QSVT circuit $\check C_{\operatorname{AA}}:=W_{\Phi_{\operatorname{AA}}}(\check C,\hat Z)$ is approximate to a block-encoding of

$$
P_\Phi^{(SV)}\left(e^{-i\check H\tau}/4\right)= -e^{-i\check H\tau}.
$$

**Remark**. For the case that $\check C_{\operatorname{AA}}$ is approximate to a block-encoding of a unitary operation,

$$
\langle0|_{cba}(\check C_{\operatorname{AA}})_{cbas}|0\rangle_{cba}\approx-\left(e^{-i\check H\tau}\right)_s
$$

is equivalent to 

$$
(\check C_{\operatorname{AA}})_{cbas}|0\rangle_{cba}\approx-|0\rangle_{cba}\left(e^{-i\check H\tau}\right)_s.
$$

Moreover, the circuit representation could be rewritten as

![CAA 块编码了 exp(-iHt)](./figures/HS-BE_CAA.JPG)

Now, all qubits in register $cba$ could be considered as a zeroed qubit$^{[7]}$.

---

## References

[1] Low, Guang Hao, and Isaac L. Chuang. "Optimal Hamiltonian simulation by quantum signal processing." Physical review letters 118.1 (2017): 010501.  
[2] Low, Guang Hao, and Isaac L. Chuang. "Hamiltonian simulation by qubitization." Quantum 3 (2019): 163.  
[3] Gilyén, András, et al. "Quantum singular value transformation and beyond: exponential improvements for quantum matrix arithmetics." Proceedings of the 51st Annual ACM SIGACT Symposium on Theory of Computing. 2019.  
[4] Martyn, John M., et al. "Grand unification of quantum algorithms." PRX Quantum 2.4 (2021): 040203.  
[5] Dong, Yulong, et al. "Efficient phase-factor evaluation in quantum signal processing." Physical Review A 103.4 (2021): 042419.  
[6] wiki, Jacobi-Anger expansion, https://en.wikipedia.org/wiki/Jacobi%E2%80%93Anger_expansion  
[7] Craig Gidney. “Constructing Large Controlled Nots.” https://algassert.com/circuits/2015/06/05/Constructing-Large-Controlled-Nots.html  
