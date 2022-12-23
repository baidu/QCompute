# 量子信号处理

*版权所有 (c) 2022 百度量子计算研究所，保留所有权利。*

## 定义

**量子信号处理**（Quantum Signal Processing, QSP）最初被设计成一种单量子比特演化模型$^{[1]}$，通过交替的调用量子信号和量子处理两种量子门形成量子电路，来实现对信号的处理效果。特别地，这里是对单一信号执行多次处理。

在本教程中，我们如下定义量子信号处理：

给定 $x\in[-1,1]$, $\Phi=(\varphi_0,\varphi_1,\cdots,\varphi_d)\in\mathbb R^{d+1}$，称如下单比特量子电路

![W_phi(x)的定义](./figures/SQSP-WPhix.JPG)

为量子信号处理电路，对应的数学定义为

$$
W_{\Phi}(x):=e^{iZ\varphi_0}\prod_{j=1}^d\left(W(x)e^{iZ\varphi_j}\right),
$$

其中 $i=\sqrt{-1}$ 为虚数单位，$e$ 是自然对数的底数，$X,Z$ 分别为泡利 $X,Z$ 矩阵，

- $x$ 为信号参数，是矩阵函数 $W_\Phi$ 的自变量；
- $\Phi$ 为处理参数，是矩阵函数 $W_\Phi$ 的参变量，其编码了这个矩阵函数；
- $W(x):=e^{iX\arccos x}=\begin{pmatrix}x&i\sqrt{1-x^2}\\i\sqrt{1-x^2}&x\end{pmatrix}$ 为信号量子门；
- $e^{iZ\varphi_j}=\begin{pmatrix}e^{i\varphi_j}&0\\0&e^{-i\varphi_j}\end{pmatrix}$ 为处理量子门。

注：更准确的讲，量子信号处理是一个泛函，是一种特殊的编码方式，它将处理参数 $\Phi$ 映射为处理方式 $W_\Phi$，而这里 $W_\Phi$ 又是一个矩阵函数，其将信号参数 $x$ 映射为处理结果矩阵 $W_\Phi(x)$。

## 性质

根据定义，我们发现矩阵 $W_\Phi(x)$ 一定是一个行列式为 $1$ 的 $2$ 维酉矩阵。进一步地，$[1]$ 证明了 $W_\Phi(x)$ 一定形如：

$$
W_{\Phi}(x)= \begin{pmatrix}
P_\Phi(x)&i\sqrt{1-x^2}Q_\Phi(x)\\
i\sqrt{1-x^2}Q_\Phi^*(x)&P_\Phi^*(x)
\end{pmatrix},
$$

其中 

- $P_{\Phi}(x)$ 为次数不超过 $d$ 的，与 $d$ 具有相同奇偶性的复系数多项式；
- $Q_{\Phi}(x)$ 为次数不超过 $d-1$ 的，与 $d-1$ 具有相同奇偶性的复系数多项式；
- $P_\Phi(x)P_\Phi^*(x)+(1-x^2)Q_\Phi(x)Q_\Phi^*(x)\equiv1$。

$P_\Phi^*$ 和 $Q_\Phi^*$ 分别表示复系数多项式 $P_\Phi$ 和 $Q_\Phi$ 各项系数取复共轭后得到的共轭多项式。

另外，若复系数多项式对 $(P,Q)$ 作为 $(P_\Phi,Q_\Phi)$ 同时满足前述三条性质，那么我们可以使用待定系数法递归地计算出 $\Phi$ 的各个分量，使得 $(P_\Phi,Q_\Phi)=(P,Q)$。

## 应用

由于在量子奇异值变换中的应用限制，我们一般只关心 $P_\Phi$。在本教程中，我们称 $P_\Phi$ 为量子信号处理函数，对应于量子奇异值变换中具体要采用的哪种变换。

对于满足特定条件的复系数多项式 $P$，我们总能计算出另一个多项式 $Q$，使得它们满足前述三条性质，进而可以计算出 $\Phi$，完成从处理函数 $P$ 到处理参数 $\Phi$ 的解码。对于并不是多项式的处理函数或变换，我们总可以找到其多项式逼近，进而近似地实现对应的量子信号处理或量子奇异值变换。

## 对称量子信号处理

从 $P$ 计算 $Q$ 的这个步骤在理论上总是可行的，但在具体数值实现的时候，难免要涉及到计算精度的问题，随着 $d$ 的增加，$Q$ 的计算变得愈加困难。$[3]$ 中提出可以基于优化算法计算对称量子信号处理的处理参数，以绕过 $Q$ 的计算这一难题。

所谓对称量子信号处理，就是处理参数**对称**的量子信号处理：

$$
\forall j\in\{0,1,\cdots,d\},\ \varphi_j=\varphi_{d-j}
$$

记 $A_\Phi:=\frac{P_\Phi+P_\Phi^*}{2}$ 是 $P_\Phi$ 的实部，退而求其次，我们放弃了直接计算编码复多项式 $P$ 的处理参数，改为计算出两个 $\Phi$ 使得 $P_\Phi$ 的实部 $A_\Phi$ 分别近似 $P$ 的实部、虚部。

注：有时 $P$ 的实虚部不具有奇偶性，我们需要再对它们做奇偶性分拆然后依次计算对应的处理参数；或者在计算近似多项式时，就加入奇偶性的限制。

记 $\tilde d$ 为 $\frac{d+1}{2}$ 的向上取整，对应处理参数在对称限制后的自由变量数量，以及 $x_j=\cos\frac{(2j-1)\pi}{4\tilde d}$ 作为函数逼近的锚点，定义目标优化函数：

$$
L_f(\Phi):=\frac{1}{2\tilde d}\sum_{j=1}^{\tilde d}\left(A_\Phi(x_j)-f(x_j)\right)^2.
$$

可以看出 $L_f(\Phi)$ 越小，$A_\Phi$ 越接近 $f$。分别对 $f=\frac{P+P^*}{2},\frac{P-P^*}{2i}$，在 $\Phi$ 对称的限制下，计算出 $L_f(\Phi)$ 的最小值点 $\Phi_{\mathfrak{R}}$ 和 $\Phi_{\mathfrak{I}}$。那么我们就有 

$$
\frac{P_{\Phi_{\mathfrak{R}}}+P_{\Phi_{\mathfrak{R}}}^*+iP_{\Phi_{\mathfrak{I}}}+iP_{\Phi_{\mathfrak{I}}}^*}{2}\approx P.
$$

即将目标处理函数，分拆成了四个量子信号处理函数的线性组合。特别地，根据量子信号处理的性质，我们可以高效计算出 $\Phi'$，使得 $P_{\Phi'}=P_\Phi^*$。

具体对 $L_f(\Phi)$ 的优化可以采用 L-BFGS 算法$^{[5]}$，其中初值选择 $\Phi=(\pi/4,0,0,\cdots,0,\pi/4)$，在 `qcompute_qsvt.SymmetricQSP` 模块中，我们就采用了这一算法给出了实现。具体的实现方式请参照 [API 文档](https://quantum-hub.baidu.com/docs/qsvt/SymmetricQSP/SymmetricQSPExternal.html)，这里就不再赘述了。

---

## 参考资料
[1] Low, Guang Hao, Theodore J. Yoder, and Isaac L. Chuang. "Methodology of resonant equiangular composite quantum gates." Physical Review X 6.4 (2016): 041067.  
[2] Low, Guang Hao, and Isaac L. Chuang. "Optimal Hamiltonian simulation by quantum signal processing." Physical review letters 118.1 (2017): 010501.  
[3] Dong, Yulong, et al. "Efficient phase-factor evaluation in quantum signal processing." Physical Review A 103.4 (2021): 042419.  
[4] Wang, Jiasu, Yulong Dong, and Lin Lin. "On the energy landscape of symmetric quantum signal processing." arXiv preprint arXiv:2110.04993 (2021).  
[5] Liu, D. C.; Nocedal, J. (1989). "On the Limited Memory Method for Large Scale Optimization". Mathematical Programming B. 45 (3): 503–528. CiteSeerX 10.1.1.110.6443. doi:10.1007/BF01589116. S2CID 5681609.  