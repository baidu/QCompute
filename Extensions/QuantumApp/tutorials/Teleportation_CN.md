# 量子隐形传态

*版权所有 (c) 2021 百度量子计算研究所，保留所有权利。*

> 若使用云端算力运行本教程，将消耗约 1 Quantum-hub 点数。

## 背景介绍

这里我们将学习量子信息理论中最经典的一个量子协议——量子隐形传态 （Quantum teleportation）。它将告诉我们如何远距离的"传输"一个量子态。假设 Alice 在实验室里获得了一个特殊的单比特量子态 $|\psi\rangle$，她想把这个量子态分享给相距很远的同事 Bob。如果他们之间存在一条很理想的量子信道，最直接的方法就是将 $|\psi\rangle$ 通过量子信道发送给 Bob。但如果他们之间根本不存在已搭建好的量子信道该怎么办呢？经过之前的学习，我们知道量子态可以表示为： $|\psi\rangle=\alpha|0\rangle+\beta|1\rangle$。Alice 可以想办法测量得到两个复数概率幅的值 $\alpha, \beta\in\mathbb{C}$ ，然后通过经典信道告诉 Bob。完成经典通讯之后，Bob 再想办法搭建量子电路去制备量子态 $|\psi\rangle$。由于 Alice 每次测量后会破原来的量子态，要想准确的获得 $\alpha,\beta$ 的估计，则需要大量拷贝的 $|\psi\rangle$ 进行反复实验。这种方法往往是非常低效的，并且实际情况中 Alice 往往只有一个拷贝的量子态，从而根本无法测得 $\alpha$ 和 $\beta$ 的准确值。所以是否还有其他的方法将量子态从 Alice 传给 Bob 呢？

## 量子隐形传态协议

在1993年，一组物理学家 Bennett et al. 提出了一套引入量子纠缠作为资源的通讯协议 [1]。这就是本节重点讨论的量子隐形传态协议。在传输的开始之前，Alice 和 Bob 需要提前准备好一对共享的 Bell 态 $|\Phi^+\rangle$,

$$
|\Phi^+\rangle_{AB} = \frac{1}{\sqrt{2}}(|0\rangle_A|0\rangle_B + |1\rangle_A|1\rangle_B). \tag{1}
$$

其中角标 $A$、$B$ 分别指代 Alice 和 Bob 所持有的量子比特系统。他们还需要保证之间共享一条经典信道以此来传输经典信息。在确定了实验条件之后，我们记需要被传输的态为 $|\psi\rangle_C$ ，Alice 对她所持有的两个量子比特系统 $C,A$ 以 $C$ 为控制位进行 CNOT 操作，然后在量子比特 $C$ 作用 $H$ 门操作。完成上述量子门操作后，Alice 对两个量子比特进行计算基测量得到经典信息比特 $z,x\in\{0,1\}$ 。Alice 将这些信息通过经典信道传输给 Bob。最后 Bob 按照接收到的通讯结果来对他持有的量子比特系统 B 进行量子门操作 $Z^zX^x$。比如说，$x=1,z=1$，Bob 需要对其量子态先进行 $X$ 门操作再进行 $Z$ 门操作。其他情况同理。完成上述步骤后， Bob 的量子比特就会处于量子态 $|\psi\rangle$ 。至此，Alice 实现了对量子态 $|\psi\rangle$ 的传输。实验上，量子隐形传态协议在1997年被 Bouwmeester et al. 通过光子的偏振实验验证 [2]。

![QCL](figures/teleportation-circuit.jpg "图 1: 量子隐形传态协议的电路图表示。")

对于量子隐形传态的理解有以下一些注意事项：

- 隐形传态协议所传输的是量子比特的状态信息，而不是量子比特本身。
- 由于量子测量的性质，Alice 在测量的时候会破坏了手上的量子态 $|\psi\rangle$，而这个量子态会在 Bob 这一侧被恢复出来。 整个过程中并没有两个拷贝的 $|\psi\rangle$ 同时存在，所以不会违反量子不可克隆原理。
- 在这个传输协议里没有任何超越光速的信息传输。Bob 恢复量子态时要先得到 Alice 的测量结果，进而做出对应的恢复操作。所以协议中量子态的传输过程会受到了经典通信速度限制。
- 当量子隐形传态完成之后，Alice 和 Bob 共享的纠缠资源会被消耗。因此在每一次传输之前，都需要重新建立纠缠资源。
- 隐形传态过程中，我们并不需要知道被传输的量子态 $|\psi\rangle$ 的任何信息。

关于更多的相关讨论可以参见文献 [3]。

**步骤1：实验准备**

首先，Alice 和 Bob 在分开之前制备一对 Bell 态 $|\Phi^+\rangle_{AB}$ 并且两人分别拿走一个量子比特。因为系统是对称的，这里两人具体拿走了哪一个并不重要。我们记 Alice 需要传输的量子态为 $|\psi\rangle_C$。我们不妨假设 $|\psi\rangle_C=\alpha|0\rangle_C+\beta|1\rangle_C$。这时整个系统的量子态可以描述为：

$$
\begin{align}
|\phi_0\rangle_{CAB} &= |\psi\rangle_C\otimes|\Phi^+\rangle_{AB} \tag{2}\\
&= (\alpha|0\rangle_C + \beta|1\rangle_C)\otimes\frac{1}{\sqrt{2}}(|0\rangle_A|0\rangle_B+|1\rangle_A|1\rangle_B) \tag{3}\\
&= \frac{1}{\sqrt{2}}(\alpha|000\rangle+\alpha|011\rangle + \beta|100\rangle +\beta|111\rangle), \tag{4}
\end{align}
$$

**步骤2：Alice 施加操作**

然后 Alice 进行 CNOT 操作并得到 $|\phi_1\rangle$,

$$
\begin{align}
|\phi_1\rangle_{CAB} &= (\text{CNOT}\otimes I)|\phi_0\rangle_{CAB \tag{5}}\\
&= (\text{CNOT}\otimes I)\frac{1}{\sqrt{2}}(\alpha|000\rangle+\alpha|011\rangle + \beta|100\rangle +\beta|111\rangle) \tag{6}\\
&= \frac{1}{\sqrt{2}}(\alpha|000\rangle+\alpha|011\rangle + \beta|110\rangle +\beta|101\rangle), \tag{7}
\end{align}
$$

Alice 对第一个量子比特施加 $H$ 门得到 $|\phi_2\rangle$，

$$
\begin{align}
|\phi_2\rangle_{CAB} &= (H \otimes I \otimes I)|\phi_1\rangle_{CAB} \tag{8}\\
&= (H\otimes I\otimes I)\frac{1}{\sqrt{2}}(\alpha|000\rangle+\alpha|011\rangle + \beta|110\rangle +\beta|101\rangle) \tag{9}\\
&= \frac{1}{2}(\alpha|000\rangle+\alpha|100\rangle+\alpha|011\rangle+\alpha|111\rangle + \beta|010\rangle -\beta|110\rangle+\beta|001\rangle-\beta|101\rangle). \tag{10}
\end{align}
$$

**步骤3：Alice 进行系统测量并传输经典比特**

为了方便看清 Alice 对所持有的两个量子比特做测量的效果，我们按照计算基重新整理整个系统的量子态 $|\phi_2\rangle_{CAB}$ 得到：

$$
\begin{align}
|\phi_2\rangle_{CAB} =\frac{1}{2}(&|00\rangle_{CA}\otimes(\alpha|0\rangle_B+\beta|1\rangle_B) \tag{11}\\
+&|01\rangle_{CA}\otimes(\alpha|1\rangle_B+\beta|0\rangle_B) \tag{12}\\
+&|10\rangle_{CA}\otimes(\alpha|0\rangle_B-\beta|1\rangle_B) \tag{13}\\
+&|11\rangle_{CA}\otimes(\alpha|1\rangle_B-\beta|0\rangle_B)). \tag{14}
\end{align}
$$

接着 Alice 对持有的两个量子比特关于计算基 $\{|00\rangle,|01\rangle,|10\rangle,|11\rangle\}$ 做测量。所得到的四种测量结果如下：

| Alice 测量结果 | 概率 | 测量后 Bob 的量子态 | 复原所需的操作 |
| :-----:| :----: | :----: | :----: |
| 00 | 1/4 | $\alpha$  &#124; 0 $\rangle+\beta$ &#124; 1 $\rangle$ | $I$|
| 01 | 1/4 | $\alpha$  &#124; 1 $\rangle+\beta$ &#124; 0 $\rangle$ | $X$|
| 10 | 1/4 | $\alpha$  &#124; 0 $\rangle-\beta$ &#124; 1 $\rangle$ | $Z$|
| 11 | 1/4 | $\alpha$  &#124; 1 $\rangle-\beta$ &#124; 0 $\rangle$ | $ZX$|

**步骤4：Bob 进行操作并复原量子态**

在 Alice 测量结束后，Bob 手上的量子比特所处的状态已经非常接近 $|\psi\rangle$。他只需要按照 Alice 发送的经典比特 00、01、10、11 的信息，进行对应的量子门操作就可以复原得到 $|\psi\rangle = \alpha|0\rangle+\beta|1\rangle$。

## 量易伏平台演示

为了加深对量子隐形传态协议的理解，我们可以在量易伏平台上进行模拟演示。

首先我们通过广义旋转门 $U_3$，

$$
U_3(\theta,\phi,\psi) = \left[
\begin{matrix}
   cos(\frac{\theta}{2}) & -e^{i\psi}sin(\frac{\theta}{2}) \\
   e^{i\phi}sin(\frac{\theta}{2}) & e^{i(\psi+\phi)}cos(\frac{\theta}{2}) \tag{15}
\end{matrix}\right]
$$

随机的生成一个量子态 $|\psi\rangle$，作为我们想要传输的量子态，示例代码如下：

```python
from QCompute import *
import numpy as np

# 设置量子比特数量
qubit_num = 3        
# 设置测量次数     
shots = 10000       
# 固定随机种子     
np.random.seed(14)        

# 生成3个随机的角度
angle = 2*np.pi*np.random.randn(3)

# 请输入您的 Token
# Define.hubToken= 'Your Token'


def choose_backend():
    # 您可以在这里选择您想要使用的后端，在选择‘量子真机’和‘云端模拟器’时
    # 请先输入您 量易伏 平台的 Token，否则无法正常运行

    # 使用本地模拟器
    backend = BackendName.LocalBaiduSim2
    # 使用云端设备
    # backend = BackendName.CloudIoPCAS
    # 使用云端模拟器
    # backend = BackendName.CloudBaiduSim2Earth
    return backend


def main():
    """
    main
    """
    # 创建环境
    env = QEnv()
    
    # 选择 Backend 为本地模拟器
    env.backend(choose_backend())

    # 初始化全部量子比特
    q = [env.Q[i] for i in range(qubit_num)]

    # 制备量子态 |psi> 
    U(angle[0], angle[1], angle[2])(q[0])

    # 测量结果
    MeasureZ([q[0]],[0])
    taskResult = env.commit(shots, fetchMeasure=True)
    return taskResult['counts']


if __name__ == '__main__':
    main()
```

    Shots 10000
    Counts {'1': 9747, '0': 253}
    State None
    Seed 56335848

按照测量的结果可以得到生成的量子态 $|\psi\rangle=\alpha|0\rangle+\beta|1\rangle$，其中 $|\alpha|^2\approx0.0253$ 并且 $|\beta|^2\approx 0.9747$。然后我们尝试将这个态从 Alice 传输给 Bob。这里需要注意的是，我们使用了量子控制门 CZ 和 CNOT 来替代 Bob 根据 Alice 测量结果所进行的经典控制，具体示例代码如下：

```python
def main():
    """
    main
    """
    # 创建环境
    env = QEnv()
    # 选择 Backend 为本地模拟器
    env.backend(choose_backend())

    # 初始化全部量子比特
    q = [env.Q[i] for i in range(qubit_num)]

    # Alice 和 Bob之间制备纠缠态
    H(q[1])
    CX(q[1], q[2])
    
    # 制备量子态 |psi> 
    U(angle[0], angle[1], angle[2])(q[0])
    
    # Alice 对持有的 q0 和 q1 进行操作
    CX(q[0], q[1])
    H(q[0])
    
    # Bob 进行操作复原量子态 |psi> 
    CZ(q[0], q[2]) 
    CX(q[1], q[2])

    # Bob 对他持有的量子比特 q2 进行测量
    MeasureZ([q[2]],[2])
    taskResult = env.commit(shots, fetchMeasure=True)
    return taskResult['counts']


if __name__ == '__main__':
    main()
```

```
Shots 10000
Counts {'1': 9718, '0': 282}
State None
Seed 695386758
```

按照测量的结果得到量子态 $\alpha'|0\rangle+\beta'|1\rangle$，其中 $|\alpha'|^2\approx0.0282$ 并且 $|\beta'|^2\approx 0.9718$ 。 与被传输的量子态 $|\psi\rangle = \alpha|0\rangle+\beta|1\rangle$ 非常接近。其中两个态之间稍许的误差来源于我们测试量子态时所做的测量次数有限 （即 shots = 10000），增大这个数值的设定可以减少误差。

---

## 参考资料

[1] Bennett, Charles H., et al. "Teleporting an unknown quantum state via dual classical and Einstein-Podolsky-Rosen channels." [Physical Review Letters 70.13 (1993): 1895.](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.70.1895)

[2] Bouwmeester, Dik, et al. "Experimental quantum teleportation." [Nature 390.6660 (1997): 575-579.](https://www.nature.com/articles/37539)

[3] Peres, Asher. "What is actually teleported?." [IBM Journal of Research and Development 48.1 (2004): 63-69.](https://arxiv.org/pdf/quant-ph/0304158.pdf)
