# 量子特征值与奇异值变换

*版权所有 (c) 2022 百度量子计算研究所，保留所有权利。*

本篇中我们继续介绍**量子特征值变换**$^{[1,2]}$（Quantum Eigenvalue Transformation）和**量子奇异值变换**$^{[3]}$。

## 量子特征值变换

量子特征值变换，顾名思义，我们要量子地实现特征值的变换。变换的无外乎涉及两件事情，一者为初值，一者为映射。初值，我们可以采用块编码来实现量子的输入；映射，我们采用量子信号处理函数来完成映射。所以说，量子特征值变换可以是块编码与量子信号处理的结合，而它们之所以能结合，它们结合的正确性，均来自于量子比特化$^{[2]}$（Qubitization）理论。

沿用块编码的符号，我们假定 $2^n$ 维厄米矩阵 $\check H$ 被块编码在 $m+n$ 比特量子黑盒 $\check U$ 的左上角，简记为：

$$
\langle0|_a \check U_{as}|0\rangle_a=\check H_s,
$$

这里额外假设黑盒 $\check U$ 满足 $\check U^\dagger=\check U$，即黑盒 $\check U$ 同时也是厄米的。设 $\check H$ 有谱分解 $\check H=\sum_\lambda \lambda|u_\lambda\rangle\langle u_\lambda|$，其中 $\lambda$ 遍历 $\check H$ 的特征值，$|u_\lambda\rangle$ 为特征值 $\lambda$ 所对应的特征态。

记

$$
|0_\lambda\rangle_{as}:=|0\rangle_a|u_\lambda\rangle_s,\ 
|0_\lambda^\perp\rangle_{as}:=\frac{\check U|0_\lambda\rangle-\lambda|0_\lambda\rangle}{\sqrt{1-\lambda^2}},
$$

引入系统 $a$ 上的关于子空间 $|0\rangle\langle0|$ 的反射算子 $\check Z$ 和高维旋转算子 $e^{i\check Z\varphi}$

$$
\begin{aligned}
\check Z:=&\ 2|0\rangle\langle0|-I,\\
e^{i\check Z\varphi}=&\ e^{i\varphi}|0\rangle\langle0|+e^{-i\varphi}(I-|0\rangle\langle0|),
\end{aligned}\tag{1}
$$

那么我们发现如下等式成立

$$
\begin{aligned}
e^{i\check Z\varphi}\cdot(|0_\lambda\rangle,|0_\lambda^\perp\rangle)&=(|0_\lambda\rangle,|0_\lambda^\perp\rangle)\cdot e^{iZ\varphi},\\
ie^{-i\pi\check Z/4}\check Ue^{-i\pi\check Z/4}\cdot(|0_\lambda\rangle,|0_\lambda^\perp\rangle)&=
(|0_\lambda\rangle,|0_\lambda^\perp\rangle)\cdot W(\lambda),
\end{aligned}
$$

这里 

$$
(|0_\lambda\rangle,|0_\lambda^\perp\rangle):=|0_\lambda\rangle\langle0|+|0_\lambda^\perp\rangle\langle1|
$$ 

是一个规模为 $2^{m+n}\times2$ 的矩阵，$e^{iZ\varphi}$ 和 $W(\lambda)$ 恰分别是量子信号处理中所定义的处理量子门和信号量子门。可以看出 $(|0_\lambda\rangle,|0_\lambda^\perp\rangle)$ 左边的多比特操作可以被等效成其右边的处理量子门或信号量子门，或者说这些多比特操作在限制在 $|0_\lambda\rangle$ 和 $|0_\lambda^\perp\rangle$ 所张成的二维空间上时，可以分别被等同于处理量子门和信号量子门。

我们可以将这些等同后的处理量子门和信号信号量子门交替相乘得到量子信号处理电路 $W_\Phi(\lambda)$，用 $(|0_\lambda\rangle,|0_\lambda^\perp\rangle)$ 右乘这个量子信号处理电路，再根据等式 $(1)$ 以及矩阵乘法的结合率，将处理矩阵和信号矩阵移动到 $(|0_\lambda\rangle,|0_\lambda^\perp\rangle)$ 的左边，便得到了等同前的量子电路：

$$
W_\Phi(\check U,\check Z):=i^de^{i\check Z\varphi_0}\prod_{j=1}^d\left(e^{-i\pi\check Z/4}\check Ue^{-i\pi\check Z/4} e^{i\check Z\varphi_j}\right),
$$

其满足

$$
W_\Phi(\check U,\check Z)\cdot(|0_\lambda\rangle,|0_\lambda^\perp\rangle)=(|0_\lambda\rangle,|0_\lambda^\perp\rangle)\cdot W_\Phi(\lambda).
$$

相应地，$W_\Phi(\check U,\check Z)$ 的电路表示为

![W_Phi(U,Z)的定义](./figures/QSVT-WPhiU.JPG)


因为其中 $i^d$ 为全局相位因子，在电路表示中就没有体现出来了。进一步，根据叠加性，可以证明（附于附录）

$$
\langle0|_aW_\Phi(\check U,\check Z)_{as}|0\rangle_a
=\sum_\lambda P_\Phi(\lambda)|u_\lambda\rangle_{s}\langle u_\lambda|_s
=:P_\Phi(\check H)_s,\tag{2}
$$

即电路 $W_\Phi(\check U,\check Z)$ 是 $P_\Phi(\check H)$ 的块编码。

![W_Phi(U,Z) 块编码了 P_Phi(H)](./figures/QSVT-BE_WPhiU.JPG)

我们称 $W_\Phi(\check U,\check Z)$ 为量子特征值变换电路，这里 $P_\Phi(\check H)$ 是单变量函数 $P_\Phi$ 在厄米矩阵 $\check H$ 处的矩阵函数值。宏观地看，$W_\Phi(\check U,\check Z)$ 可以区分、识别 $\check H$ 的不同的特征空间，并将特征值作为信号参数导入量子信号处理，**并行**地处理全部的特征值，同时实现各个特征值沿函数 $P_\Phi$ 的变换。这种将多个 $(|0_\lambda\rangle,|0_\lambda^\perp\rangle)$ 所张成的二维空间当作量子比特并进行并行操作的技巧被称为**量子比特化**。

**注：** 前面我们虽然假设了 $\check U^\dagger= \check U$，但这个条件并不苛刻。对于这个假设不成立的情形，我们总可以引入一个辅助比特 $b$，控制版本的量子黑盒 $C(\check U)=|0\rangle\langle0|\otimes I + |1\rangle\langle1|\otimes\check U$ 及其逆 $C(\check U)^\dagger$ 来构造一个新的量子黑盒 $\check V$：

![V 的定义](./figures/QSVT-V(U).JPG)

数学上，可以描述为：

$$
\check V_{bas}= H_b C(\check U)_{bas}^\dagger X_b C(\check U)_{bas}H_b,
$$

其中 $H$ 是阿达玛门。那么可以验证 $\check V$ 仍是 $\check H$ 的块编码，且满足 $\check V^\dagger=\check V$，只不过矩阵的维数相比于 $\check U$ 增加了一倍。

![V 块编码了 H](./figures/QSVT-BE_V.JPG)

## 量子奇异值变换

作为量子特征值变换的推广版本，接下来我们来简要介绍下**量子奇异值变换**的一种特例情形。在讲量子奇异值变换之前，我们先来了解下什么是奇异值。复矩阵 $\check N$ 的奇异值被定义为 $\check N^\dagger\check N$ 的特征值的算术平方根（相等特征值仍被认为是不同的特征值），故而 $\check N$ 为方阵的时候，它的奇异值数量总与矩阵维数相等。对于矩阵 $\check N$ 并非方阵的时候，我们一般只讨论它非零的奇异值，或者直接假定奇异值都是非零的。基于厄米矩阵 $\check N^\dagger\check N$ 的谱分解，我们可以得到矩阵 $\check N$ 的奇异值分解：

$$
\check N=\sum_\sigma \sigma|u_\sigma\rangle\langle v_\sigma|
$$

其中各 $\sigma>0$，各 $|u_\sigma\rangle$ 两两正交，各 $|v_\sigma\rangle$ 也两两正交。数学上，矩阵 $\check N$ 作为线性变换会将高维单位球映射为一个高维椭球，此时 $\check N$ 的非零奇异值正是该高维椭球的各个半轴长。正因为 $0$ 奇异值对应高维椭球的退化掉的维度，我们才会不去关心那些 $0$ 奇异值。

回到量子奇异值变换，我们依旧假设 $2^n$ 维方阵 $\check N$ 被块编码在 $m+n$ 比特量子黑盒 $\check U$ 的左上角，依旧引入系统 $a$ 上的关于子空间 $|0\rangle\langle0|$ 的反射算子 $\check Z$ 和高维旋转算子 $e^{i\check Z\varphi}$

$$
\begin{aligned}
\check Z&:=2|0\rangle\langle0|-I,\\
e^{i\check Z\varphi}&:=e^{i\varphi}|0\rangle\langle0|+e^{-i\varphi}(I-|0\rangle\langle0|),
\end{aligned}
$$

奇数 $d$ 和处理参数 $\Phi=(\varphi_0,\varphi_1,\cdots,\varphi_d)\in\mathbb R^{d+1}$，此时改记

![W_Phi(U,Z)的定义](./figures/QSVT-WPhiU_SV.JPG)

数学上为：

$$
W_\Phi(\check U,\check Z)=i^de^{i\check Z\varphi_0}\prod_{j=1}^d\left(e^{-i\pi\check Z/4}\check U^{(-1)^{j+1}}e^{-i\pi\check Z/4} e^{i\check Z\varphi_j}\right),
$$

同样地，其中 $i^d$ 为全局相位因子，在电路表示中没有体现。那么可以验证 $W_\Phi(\check U,\check Z)$ 是 $\check N$ 奇异值变换结果 $P_\Phi^{(SV)}(\check N):=\sum_\sigma P_\Phi(\sigma)|u_\sigma\rangle\langle v_\sigma|$ 的块编码：

$$
\langle0|_aW_\Phi(\check U,\check Z)_{as}|0\rangle_a
=P^{(SV)}_\Phi(\check N).
$$

![W_Phi(U,Z) 块编码了 P_Phi(N)](./figures/QSVT-BE_WPhiU_SV.JPG)

在讲量子奇异值变换且不发生歧义时，我可以简记 $P^{(SV)}_\Phi$ 为 $P_\Phi$。可以验证量子特征值变换就是量子奇异值变换本特例情形在 $\check U^\dagger=\check U$ 且 $\check N^\dagger=\check N$ 时的特例。

---

## 附录

#### $(2)$ 的证明

$$
\begin{aligned}
 &\langle0|_aW_\Phi(\check U,\check Z)_{as}|0\rangle_a\\
=&\langle0|_aW_\Phi(\check U,\check Z)_{as}|0\rangle_a\sum_\lambda|u_\lambda\rangle_s\langle u_\lambda|_s\\
=&\sum_\lambda\langle0|_a\left(W_\Phi(\check U,\check Z)|0_\lambda\rangle\right)_{as}\langle u_\lambda|_s\\
=&
\sum_\lambda\langle0|_a\cdot\left(
    |0_\lambda\rangle_{as}\cdot\left(\langle0|W_\Phi(\lambda)|0\rangle\right)+
    |0_\lambda^\perp\rangle_{as}\cdot\left(\langle1|W_\Phi(\lambda)|0\rangle\right)
    \right)\langle u_\lambda|_s\\
=&
\sum_\lambda\left(
    P_\Phi(\lambda)\langle0|_a\cdot|0_\lambda\rangle_{as}+
    i\sqrt{1-\lambda^2}Q_\Phi(\lambda)\langle0|_a\cdot|0_\lambda^\perp\rangle_{as}
    \right)\langle u_\lambda|_s\\
=&
\sum_\lambda P_\Phi(\lambda)|u_\lambda\rangle_{s}\langle u_\lambda|_s\\
=:&P_\Phi(\check H)_s,
\end{aligned}
$$

## 参考资料

[1] Low, Guang Hao, and Isaac L. Chuang. "Optimal Hamiltonian simulation by quantum signal processing." Physical review letters 118.1 (2017): 010501.  
[2] Low, Guang Hao, and Isaac L. Chuang. "Hamiltonian simulation by qubitization." Quantum 3 (2019): 163.  
[3] Gilyén, András, et al. "Quantum singular value transformation and beyond: exponential improvements for quantum matrix arithmetics." Proceedings of the 51st Annual ACM SIGACT Symposium on Theory of Computing. 2019.  