# 快速上手

*版权所有 (c) 2022 百度量子计算研究所，保留所有权利。*

量子奇异值变换目前提供了哈密顿量模拟的量子电路生成，用户可以在使用 QCompute 创建量子电路的时候调用哈密顿量模拟电路。本篇着重于代码的调用使用，帮助用户快速上手使用量子奇异值变换。

具体的理论介绍参见教程的[后续章节](https://quantum-hub.baidu.com/qsvt/tutorial-introduction)，代码的实现原理也可以参考量子奇异值变换的 [API 文档](https://quantum-hub.baidu.com/docs/qsvt/)。

## 使用演示

在完成量子奇异值变换的安装后，我们可以新建一个 python 脚本文件，输入以下代码完成本次演示的初始化。

```python{.line-numbers}
import numpy as np
from QCompute import QEnv, BackendName, MeasureZ
from Extensions.QuantumSingularValueTransformation.qcompute_qsvt.Application.HamiltonianSimulation.HamiltonianSimulation import circ_HS_QSVT
```

在实现哈密顿量模拟之前，我们要输入想要模拟的哈密顿量。我们采用哈密顿量的泡利基表示，即将哈密顿量表示成多比特泡利矩阵的线性组合的形式。线性组合中的每一项用一个浮点数和字符串构成的二元组给出，其中浮点数表述这一项的系数，字符串编码了泡利矩阵的信息。如哈密顿量 

$$
H\otimes H=\frac12\left(X\otimes X+X\otimes Z+Z\otimes X+Z\otimes Z\right)
$$ 

可以表示为列表

```python
list_str_Pauli_rep_HH = [(0.5, 'X0X1'), (0.5, 'X0Z1'), (0.5, 'Z0X1'), (0.5, 'Z0Z1')]
```

更复杂地，如氢分子 $\operatorname{H_2}$ 的哈密顿量可以表示为：

```python
list_str_pauli_rep_H2 = [
    (-0.09706626861762556, 'I'), (-0.04530261550868938, 'X0X1Y2Y3'),
    (0.04530261550868938, 'X0Y1Y2X3'), (0.04530261550868938, 'Y0X1X2Y3'),
    (-0.04530261550868938, 'Y0Y1X2X3'), (0.1714128263940239, 'Z0'),
    (0.16868898168693286, 'Z0Z1'), (0.12062523481381837, 'Z0Z2'),
    (0.16592785032250773, 'Z0Z3'), (0.17141282639402394, 'Z1'),
    (0.16592785032250773, 'Z1Z2'), (0.12062523481381837, 'Z1Z3'),
    (-0.2234315367466399, 'Z2'), (0.17441287610651626, 'Z2Z3'),
    (-0.2234315367466399, 'Z3')]
```

以 `list_str_pauli_rep_HH` 为例我们继续讲述哈密顿量模拟电路的调用。我们还需要明确哈密顿量涉及系统的量子比特数 $n$ (@`num_qubit_sys`)，哈密顿量模拟的时间 $\tau$ (@`float_tau`) 及模拟的精度 $\epsilon$ (@`float_epsilon`)，如：

```python{.line-numbers}
num_qubit_sys = 2
float_tau = np.pi / 4
float_epsilon = 1e-5
```

然后声明量子环境和量子寄存器，我们需要引入哈密顿量涉及系统所对应的系统寄存器 `reg_sys`，若干辅助比特 `reg_blocking` 实现哈密顿量的编码（比特数量与哈密顿量的表示长度相关），以及两个辅助比特 `reg_ancilla` 来实现哈密顿量的模拟。由于我们更习惯于将控制比特置于受控比特之前，我们颠倒了前述各量子寄存器的引入顺序：

```python{.line-numbers}
# create the quantum environment, choose local backend
env = QEnv()
env.backend(BackendName.LocalBaiduSim2)

# the two ancilla qubit introduced from HS
reg_ancilla = [env.Q[0], env.Q[1]]
# compute the number of qubits needed in the block-encoding, and form a register
num_qubit_blocking = max(1, int(np.ceil(np.log2(len(list_str_Pauli_rep_HH)))))
reg_blocking = list(env.Q[idx] for idx in range(2, 2 + num_qubit_blocking))
# create the system register for the Hamiltonian
reg_sys = list(env.Q[idx] for idx in range(2 + num_qubit_blocking, 2 + num_qubit_blocking + num_qubit_sys))
```

此时所有的量子比特都处于基态 $|0\rangle$，而后我们调用哈密顿量模拟电路，其相当于在系统寄存器 `reg_sys` 上作用了对应的哈密顿量时间演化算符 $e^{-iH\otimes H\tau}$：

```python
circ_HS_QSVT(reg_sys, reg_blocking, reg_ancilla, list_str_Pauli_rep_HH, float_tau, float_epsilon)
```

我们可以在该电路前后增加其他的量子门，以实现更复杂的量子算法。**但需要注意的是，调用哈密顿量模拟电路 `circ_HS_QSVT` 前，所有 `reg_ancilla` 和 `reg_blocking` 中的量子比特都必须处于 $|0\rangle$ 态，否则不会实现我们想要的哈密顿量时间演化算子。**

测量并打印测量结果，

```python{.line-numbers}
# measure
MeasureZ(*env.Q.toListPair())
# commit
print(env.commit(8000, downloadResult=False)['counts'])
```

我们可以得到

```python
{'100000': 1003, '000000': 5002, '010000': 960, '110000': 1035}
```

考虑到 QCompute 采用大端输出方式，从中我们可以读出前四个量子比特，即 `reg_ancilla` 和 `reg_blocking` 中共计四个辅助比特的输出态均为 $|0\rangle$，也就是说它们确实被还原到了 $|0\rangle$ 态。另外系统寄存器布居数近似处于 $5:1:1:1$ 的比例与

$$
e^{-i\pi H\otimes H/4}|00\rangle=\frac{-i}{2\sqrt 2}\left((1+2i)|00\rangle+|01\rangle+|10\rangle+|11\rangle\right)
$$

相符。

除了上述一步一步实现量子电路之外，如果读者只想简单体验一下哈密顿量模拟的效果，可以直接调用函数 `func_HS_QSVT` 实现上述演示过程：

```python{.line-numbers}
import numpy as np
from Extensions.QuantumSingularValueTransformation.qcompute_qsvt.Application.HamiltonianSimulation.HamiltonianSimulation import func_HS_QSVT
print(func_HS_QSVT(list_str_Pauli_rep=[(0.5, 'X0X1'), (0.5, 'X0Z1'), (0.5, 'Z0X1'), (0.5, 'Z0Z1')], 
      num_qubit_sys=2, float_tau=np.pi / 4, float_epsilon = 1e-5, circ_output=False)['counts'])
```

读者可以参考该函数的[源代码](https://github.com/baidu/QCompute/tree/master/Extensions/QuantumSingularValueTransformation/qcompute_qsvt/Applications/HamiltonianSimulation/HamiltonianSimulation.py)了解更多应用信息。

## 更多例子 

用户可以在 [github](https://github.com/baidu/QCompute/tree/master/Extensions/QuantumSingularValueTransformation/) 上下载更多的[用例](https://github.com/baidu/QCompute/tree/master/Extensions/QuantumSingularValueTransformation/examples/)。其中 [**`example-HamiltonianSimulation.py`**](https://github.com/baidu/QCompute/tree/master/Extensions/QuantumSingularValueTransformation/examples/example-HamiltonianSimulation.py) 包含两个演示函数：

- `func_MS_test_QSVT` 我们依次测试五个既定量子电路的输出结果，以验证哈密顿量模拟 $e^{i\pi X\otimes X/4}$ 的电路实现是正确的。比如这里在模拟精度取 $10^{-5}$，且各 10000 shots 未能检测出误差时，测试通过并会打印出 `"MS test passed."`。
- `func_HH_test_QSVT` 我们也可以用类似 `func_MS_test_QSVT` 的方法测试哈密顿量模拟 $e^{-i\pi H\otimes H/4}$ 的电路实现的正确性。

这里哈密顿量 $\check H= X\otimes X$ 在时间 $-\pi/4$ 处的演化算子 $e^{i\pi X\otimes X/4}$ 正是离子阱量子计算中的两比特原生量子门 $\operatorname{MS}$ 门，读者可以参考[量脉离子阱](https://quanlse.baidu.com/#/doc/tutorial-ion-trap-single-and-two-qubit-gate)了解更多信息。特别地，在基态 $|00\rangle$ 上作用 $\operatorname{MS}$ 门可以一步制备叠加态

$$
\operatorname{MS}|00\rangle=\frac{1}{\sqrt 2}\left(|00\rangle + i|11\rangle\right).
$$

读者可以自行尝试添加一些后处理来证明两个分量的相位差确实是 $\pi/2$。
