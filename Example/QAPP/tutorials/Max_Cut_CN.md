# 最大割问题

<em> 版权所有 (c) 2021 百度量子计算研究所，保留所有权利。 </em>

> 若使用云端算力运行本教程，将消耗约 300 Quantum-hub 点数。

## 概览

最大割问题（Max-Cut Problem）是图论中常见的一个组合优化问题，在统计物理学和电路设计中都有重要应用。最大割问题是一个 NP 困难问题，因此目前并不存在一个高效的算法能完美地解决该问题。

在图论中，一个图是由一对集合 $G=(V, E)$ 表示，其中集合 $V$ 中的元素为该图的顶点，集合 $E$ 中的每个元素是一对顶点，表示连接这两个顶点的一条边。例如下方图片中的图可以由 $V=\{0,1,2,3\}$ 和 $E=\{(0,1),(1,2),(2,3),(3,0)\}$ 表示。

```python
import networkx as nx

# num_vertices 是图 G 的顶点数，同时也是量子比特的个数
num_vertices = 4
G = nx.Graph()
V = range(num_vertices)
G.add_nodes_from(V)
E = [(0, 1), (1, 2), (2, 3), (3, 0)]
E.sort()
G.add_edges_from(E)
```

![fig0](./figures/max-cut-fig-0.png)

一个图上的割（cut）是指将该图的顶点集 $V$ 分割成两个互不相交的集合的一种划分，每个割都对应一个边的集合，这些边的两个顶点被划分在不同的集合中。于是我们可以将这个割的大小定义为与其对应的边的集合的大小，即被割开的边的条数。最大割问题就是要找到一个割使得被割开的边的条数最多。下图展示了上图中图的一个最大割，该最大割的大小为 $4$，即割开了图中所有的边。

![fig1](./figures/max-cut-fig-1.png)

假设输入的图 $G=(V, E)$ 有 $n=|V|$ 个顶点和 $m=|E|$ 条边，那么我们可以将最大割问题描述为 $n$ 个比特和 $m$ 个子句的组合优化问题。每个比特对应图 $G$ 中的一个顶点 $v$，其取值 $z_v$ 为 $0$ 或 $1$，分别对应该顶点属于集合 $S_{0}$ 或 $S_{1}$，因此这 $n$ 个比特的每种取值 $z$ 都对应一个割。每个子句则对应图 $G$ 中的一条边 $(u,v)$，一个子句要求其对应的边连接的两个顶点的取值不同，即 $z_u\neq z_v$，表示该条边被割开。也就是说，当该条边连接这的两个顶点被割划分到不同的集合上时，我们说该子句被满足。因此，对于图 $G$ 中的每条边 $(u,v)$，我们有

$$
C_{(u,v)}(z) = z_u+z_v-2z_uz_v,
\tag{1}
$$

其中 $C_{(u,v)}(z) = 1$ 当且仅当该条边被割开。否则，该函数等于 $0$。整个组合优化问题的目标函数是

$$
C(z) = \sum_{(u,v)\in E}C_{(u,v)}(z) = \sum_{(u,v)\in E}z_u+z_v-2z_uz_v.
\tag{2}
$$

因此，解决最大割问题就是要找到一个取值 $z$ 使得公式（2）中的目标函数最大。

### 编码最大割问题

为了将最大割问题转化为一个量子问题，我们要用到 $n$ 个量子比特，每个量子比特对应图 $G$ 中的一个顶点。一个量子比特处于量子态 $|0\rangle$ 或 $|1\rangle$，表示其对应的顶点属于集合 $S_{0}$ 或 $S_{1}$。值得注意的是，$|0\rangle$ 和 $|1\rangle$ 是 Pauli $Z$ 门的两个本征态，并且它们的本征值分别为 $1$ 和 $-1$，即

$$
\begin{align}
Z|0\rangle&=|0\rangle,\tag{3}\\
Z|1\rangle&=-|1\rangle.\tag{4}
\end{align}
$$

因此我们可以使用 Pauli $Z$ 门来构建该最大割问题的哈密顿量 $H_C$。因为通过映射 $f(x):x\to(x+1)/2$ 可以将 $-1$ 映射到 $0$ 上 并且仍将 $1$ 映射到 $1$ 上，所以我们可以将式（2）中的 $z$ 替换为 $(Z+I)/2$（$I$ 是单位矩阵），得到原问题目标函数对应的哈密顿量

$$
\begin{align}
H_C &= \sum_{(u,v)\in E} \frac{Z_u+I}{2} + \frac{Z_v+I}{2} - 2\cdot\frac{Z_u+I}{2}\frac{Z_v+I}{2}\tag{5}\\
&= \sum_{(u,v)\in E} \frac{Z_u+Z_v+2I - (Z_uZ_v+Z_u+Z_v+I)}{2}\tag{6}\\
&= \sum_{(u,v)\in E} \frac{I - Z_uZ_v}{2}.\tag{7}
\end{align}
$$

该哈密顿量关于一个量子态 $|\psi\rangle$ 的期望值为

$$
\begin{align}
\langle\psi|H_C|\psi\rangle &= \langle\psi|\sum_{(u,v)\in E} \frac{I - Z_uZ_v}{2}|\psi\rangle\tag{8}\\
&= \langle\psi|\sum_{(u,v)\in E} \frac{I}{2}|\psi\rangle - \langle\psi|\sum_{(u,v)\in E} \frac{Z_uZ_v}{2}|\psi\rangle\tag{9}\\
&= \frac{|E|}{2} - \frac{1}{2}\langle\psi|\sum_{(u,v)\in E} Z_uZ_v|\psi\rangle.\tag{10}
\end{align}
$$

如果我们记

$$
H_D = -\sum_{(u,v)\in E} Z_uZ_v,
\tag{11}
$$

那么找到量子态 $|\psi\rangle$ 使得 $\langle\psi|H_C|\psi\rangle$ 最大等价于找到量子态 $|\psi\rangle$ 使得 $\langle\psi|H_D|\psi\rangle$ 最大。  

## QAPP 和 QCompute 实现

加载 QAPP 和 QCompute 相关的所有模块。

```python
# 加载 QAPP 和 QCompute 相关模块
from qapp.circuit import ParameterizedCircuit
from qapp.circuit import QAOAAnsatz
from qapp.optimizer import SPSA, Powell
from QCompute.QPlatform import BackendName
from qapp.algorithm import QAOA
from qapp.application.optimization.max_cut import MaxCut

# 加载额外需要用到的包
import numpy as np
```

使用上述方法，读者可以通过以下代码对最大割问题进行编码。

```python
# 构建哈密顿量
max_cut = MaxCut(num_qubits = num_vertices)
max_cut.graph_to_hamiltonian(G)
print(max_cut._hamiltonian)
```
```
[[-1.0, 'zzii'], [-1.0, 'ziiz'], [-1.0, 'izzi'], [-1.0, 'iizz']]
```

通过编码，我们将最大割问题转化为了量子优化问题，并使用量子近似优化算法 [1]（quantum approximate optimization algorithm, QAOA）来求解。对 QAOA 感兴趣的读者可以参考量桨提供的[教程](https://qml.baidu.com/tutorials/combinatorial-optimization/quantum-approximate-optimization-algorithm.html)。这里，我们使用 QAPP 提供的 ``QAOAAnsatz`` 。首先，我们定义参数化电路中的参数，在此教程中即 QAOA 电路中的参数。

```python
layer = 2 # 量子电路的层数
parameters = 2 * np.pi * np.random.rand(layer * 2)
iteration_num = 100


ansatz = QAOAAnsatz(num_vertices, parameters, max_cut._hamiltonian, layer)
```

然后我们定义用来找到编码好的哈密顿量的最大特征值的优化器。这里，我们使用 QAPP 提供的 ``SPSA`` 。读者也可以使用别的优化器，比如 ``Powell`` 。

```python
# 使用 SPSA 优化器
opt = SPSA(iteration_num, ansatz, a=0.5, c=0.15)
```

我们现在可以运行 ``QAOA`` 算法来解决最大割问题。我们预设的后端是本地后端 ``BackendName.LocalBaiduSim2`` ，读者可以在[量易伏](https://quantum-hub.baidu.com/)的网站上找到更多的后端。特别地，我们也提供真机接口，读者可以通过 ``BackendName.CloudIoPCAS`` 调用。

```python
backend = BackendName.LocalBaiduSim2
# 调用真机或者云服务器需要从量易伏网站个人中心里获取 token
# from QCompute import Define
# Define.hubToken = 'your token'
# backend = BackendName.CloudIoPCAS # 调用真机
# backend = BackendName.CloudBaiduSim2Water # 调用云服务器 CloudBaiduSim2Water
measure = 'SimMeasure' # 设置测量方式
qaoa = QAOA(num_vertices, max_cut._hamiltonian, ansatz, opt, backend, measure)
qaoa.run(shots=4096)
print("找到的最大特征值: ", qaoa.maximum_eigenvalue)
```
```
terminated after reaching max number of iterations
找到的最大特征值:  3.99951171875
```

### 解码量子答案
当求得损失函数的最小值后，我们的任务还没有完成。为了进一步求得 Max-Cut 问题的近似解，需要从 QAOA 输出的量子态中解码出经典优化问题的答案。物理上，解码量子态需要对量子态进行测量，然后统计测量结果的概率分布。  

通常情况下，某个比特串出现的概率越大，意味着其对应 Max-Cut 问题最优解的可能性越大。我们可以使用下面的代码来得到概率最高的比特串：

```python
# 重复测量电路输出态 2048 次
counts = qaoa.get_measure(shots=2048)
# 找到测量结果中出现几率最大的比特串
cut_bitstring = max(counts, key=counts.get)
solution = max_cut.decode_bitstring(cut_bitstring)
print("对于这个图形，我们找到的最大割解是: ", solution)
```
```
我们找到的最大割解是： {0: '1', 1: '0', 2: '1', 3: '0'}
```

此时，记其在该比特串中对应的比特取值为 $0$ 的顶点属于集合 $S_0$ 以及对应比特取值为 $1$ 的顶点属于集合 $S_1$，这两个顶点集合之间存在的边就是该图的一个可能的最大割方案。

下面的代码选取测量结果中出现几率最大的比特串，然后将其映射回经典解，并且画出对应的最大割方案：
- 红色顶点属于集合 $S_0$，
- 蓝色顶点属于集合 $S_1$，
- 虚线表示被割的边。

![fig2](./figures/max-cut-fig-2.png)

_______

## 参考文献

[1] Farhi, Edward, Jeffrey Goldstone, and Sam Gutmann. "A quantum approximate optimization algorithm." arXiv preprint [arXiv:1411.4028 (2014)](https://arxiv.org/abs/1411.4028).
