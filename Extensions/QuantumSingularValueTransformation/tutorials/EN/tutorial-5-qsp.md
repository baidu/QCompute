# Quantum Signal Processing

*Copyright (c) 2022 Institute for Quantum Computing, Baidu Inc. All Rights Reserved.*

## Definitions

**Quantum Signal Processing** (QSP) was designed as a single-qubit evolution model$^{[1]}$, which realizes the effect of signal processing by alternately calling two kinds of quantum gates, quantum signal gates and quantum processing gates, to form a quantum circuit. In particular, here multiple processing is performed on a single signal. In this tutorial, we define QSP as follows:

Given $x\in[-1,1]$, $\Phi=(\varphi_0,\varphi_1,\cdots,\varphi_d)\in\mathbb R^{d+1}$, we call the following single-qubit circuit

![W_phi(x)的定义](./figures/SQSP-WPhix.JPG)

QSP circuit, in math defined strictly as

$$
W_{\Phi}(x):=e^{iZ\varphi_0}\prod_{j=1}^d\left(W(x)e^{iZ\varphi_j}\right),
$$

where $i=\sqrt{-1}$ is the imaginary unit, $e$ is the base of natural logarithm, $X$ and $Z$ are Pauli $X$ and $Z$ matrices respectively. We call

- $x$ signal parameter, which is the single independent variable for the matrix function $W_\Phi$;
- $\Phi$ processing parameters, which is the parametric variable for the matrix function $W_\Phi$ and also encodes such function;
- $W(x):=e^{iX\arccos x}=\begin{pmatrix}x&i\sqrt{1-x^2}\\i\sqrt{1-x^2}&x\end{pmatrix}$ signal gate;
- $e^{iZ\varphi_j}=\begin{pmatrix}e^{i\varphi_j}&0\\0&e^{-i\varphi_j}\end{pmatrix}$ processing gate.

**Remark.** To say it more precisely, QSP is a functional, and also a special encoding method, which maps the processing parameter $\Phi$ to the processing method $W_\Phi$. Furthermore, $W_\Phi$ is a matrix function that maps the signal parameter $x$ to the processing result operation $W_\Phi(x)$.

## Properties

By definition, we find the matrix form for operation $W_\Phi(x)$ must be a $2$-dimensional unitary matrix with determinant $1$. Then $[1]$ has proved that $W_\Phi(x)$ must be of form:

$$
W_{\Phi}(x)= \begin{pmatrix}
P_\Phi(x)&i\sqrt{1-x^2}Q_\Phi(x)\\
i\sqrt{1-x^2}Q_\Phi^*(x)&P_\Phi^*(x)
\end{pmatrix},
$$

where

- $P_{\Phi}(x)$ is a complex polynomial with degree no more than $d$ and parity the same as the integer $d$;
- $Q_{\Phi}(x)$ is a complex polynomial with degree no more than $d-1$ and parity the same as the integer $d-1$;
- $P_\Phi(x)P_\Phi^*(x)+(1-x^2)Q_\Phi(x)Q_\Phi^*(x)\equiv1$.

Here $P_\Phi^*$ and $Q_\Phi^*$ are the polynomials obtained by taking the complex conjugate of each coefficient in $P_\Phi$ and $Q_\Phi$, respectively.

Additionally, if a complex polynomial pair $(P,Q)$ satisfies the above three properties of pair $(P_\Phi,Q_\Phi)$ simultaneously for some integer $d$, then we can use the undetermined coefficient method to calculate each component of $\Phi$ recursively, such that $(P_\Phi,Q_\Phi)=(P,Q)$.

## Applications

Because of the limitation of the application of QSP in quantum singular value transformation, only $P_\Phi$ is cared about, but not $Q_\Phi$. In this tutorial, we call $P_\Phi$ QSP function, corresponding to which transformation used in quantum singular value transformation.

For those complex polynomials $P$ satisfying certain conditions, we can always find another polynomial $Q$, so that they satisfy the aforementioned three properties, and then we can find their processing parameters $\Phi$ to complete the decoding for the QSP function $P$. For those processing functions or transformations which are not polynomials, we can always find their polynomial approximations, and then realize the corresponding QSP or quantum singular value transformation approximately.

## Symmetric Quantum Signal Processing

It always works to find $Q$ from $P$ in theory, but it will inevitably involve the problem of calculation accuracy in numeral calculations. With the increase of $d$, the finding of $Q$ becomes more and more difficult. In $[3]$, it is proposed that the processing parameters of Symmetric Quantum Signal Processing (SQSP) can be calculated based on the optimization algorithm to bypass the problem of finding $Q$.

The so-called SQSP is just defined as QSP with processing parameters $\Phi=\{\varphi_0,\varphi_1,\cdots,\varphi_d\}\in\mathbb R^{d+1}$ **symmetric**:

$$
\forall j\in\{0,1,\cdots,d\},\ \varphi_j=\varphi_{d-j}
$$

Denote $A_\Phi:=\frac{P_\Phi+P_\Phi^*}{2}$ as the real part of polynomial $P_\Phi$. As a last resort, we gave up the direct calculation of the processing parameters encoding such QSP function $P$, and instead calculated two groups of processing parameters $\Phi$ so that $A_\Phi$s approximate to the real and imaginary parts of $P$ respectively.

**Remark.** Considering the real and imaginary parts of $P$ may not have parity sometimes, we need to split them by parity and then calculate the corresponding processing parameters in turn, or add parity restrictions when calculating approximate polynomials.

Denote $\tilde d$ as $\frac{d+1}{2}$ rounded up, also the number of free variables in the processing parameters under symmetric restrictions; denote each $x_j=\cos\frac{(2j-1) \pi}{4\tilde d}$ for $j=1,2,\cdots,\tilde d$ as the anchor point for the following function; define the objective optimization function:

$$
L_f(\Phi):=\frac{1}{2\tilde d}\sum_{j=1}^{\tilde d}\left(A_\Phi(x_j)-f(x_j)\right)^2.
$$

It is easy to see that the smaller $L_f(\Phi)$ is, the closer $A_\Phi$ is to $f$. For $f=\frac{P+P^*}{2}$ and $\frac{P-P^*}{2i}$, under the limitation of $\Phi$ symmetric, we can find the minimum points $\Phi_{\mathfrak{R}}$ and $\Phi_{\mathfrak{I}}$ for $L_f$ respectively. Then we can construct

$$
\frac{P_{\Phi_{\mathfrak{R}}}+P_{\Phi_{\mathfrak{R}}}^*+iP_{\Phi_{\mathfrak{I}}}+iP_{\Phi_{\mathfrak{I}}}^*}{2}\approx P.
$$

That is, the target processing function $P$ is split into a linear combination of four QSP functions. In particular, according to the properties of QSP, we can obtain $\Phi'$ with $P_{\Phi'}=P_\Phi^*$ from $\Phi$  efficiently.

Specifically, we can use L-BFGS algorithm$^{[5]}$ to optimize $L_f(\Phi)$, where the initial value set as $\Phi=(\pi/4,0,0,\cdots,0,\pi/4)$, which is integrated in `qcompute_qsvt.SymmetricQSP` module. For the specific realization method, please refer to [API Documentation](https://quantum-hub.baidu.com/docs/qsvt/SymmetricQSP/SymmetricQSPExternal.html), and we won’t go into details here.

---

## References
[1] Low, Guang Hao, Theodore J. Yoder, and Isaac L. Chuang. "Methodology of resonant equiangular composite quantum gates." Physical Review X 6.4 (2016): 041067.  
[2] Low, Guang Hao, and Isaac L. Chuang. "Optimal Hamiltonian simulation by quantum signal processing." Physical review letters 118.1 (2017): 010501.  
[3] Dong, Yulong, et al. "Efficient phase-factor evaluation in quantum signal processing." Physical Review A 103.4 (2021): 042419.  
[4] Wang, Jiasu, Yulong Dong, and Lin Lin. "On the energy landscape of symmetric quantum signal processing." arXiv preprint arXiv:2110.04993 (2021).  
[5] Liu, D. C.; Nocedal, J. (1989). "On the Limited Memory Method for Large Scale Optimization". Mathematical Programming B. 45 (3): 503–528. CiteSeerX 10.1.1.110.6443. doi:10.1007/BF01589116. S2CID 5681609.  