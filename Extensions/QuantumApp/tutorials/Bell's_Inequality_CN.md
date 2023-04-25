# Bell 不等式

*版权所有 (c) 2021 百度量子计算研究所，保留所有权利。*

> 若使用云端算力运行本教程，将消耗约 200 Quantum-hub 点数。

## 背景介绍

自量子力学诞生以来，它在描述原子结构和更小尺寸物理现象上已经取得了重要的进展。在微观尺度上，量子力学的预言结果和实验吻合得非常好，人们逐渐接受量子力学作为描述现实世界规律的合适理论。然而，量子力学关于波函数 $|\psi\rangle$ 的概率诠释引发了 Einstein 和 Bohr 一场旷日持久的争论 [1]——量子力学是否完备？ 1935年 Einstein、Podolsky、Rosen 三人发表了一篇名为“量子力学关于物理实在性的描述是否完备”的文章 [2]，试图通过基于实在性和局域性的思想论证量子力学的不完备性，这就是EPR (Einstein-Podolsky-Rosen）佯谬。 

EPR 佯谬的目的是通过说明量子力学缺乏某些本质的实在性，从而论证量子力学不是一个完备的理论。在 EPR 佯谬文章发表多年后，1964年 Bell 提出了一类不等式，被称之为 Bell 不等式 [3]。这类不等式说的是，假如 EPR 佯谬中的两个假设是成立的，那么有一类实验结果一定满足某个特定的不等式，从而我们可以在真实物理世界中构造相关的实验装置，验证自然界是否遵循 EPR 佯谬的假设，进而判断量子力学是否完备。让实验结果决定哪种理论是更为贴切的关于自然的理论描述。

首先要明确的是，Bell 不等式是指一类不等式，而不是特指某个不等式。这里我们以常用的一种 Bell 不等式——CHSH（Clauser-Horne-Shimony-Holt）不等式为例，来说明经典世界和量子世界的异同。因为 Bell 不等式不是量子力学得到的结果，我们可以暂时忘却关于量子力学的背景知识，假定如下图所示实验：

![exp](figures/bell-fig-experiment.png "图 1: 假想实验示意图。")

1. Charlie 制备了两个粒子并分别将两个粒子分发给 Alice 和 Bob。
2. Alice 有两台测量设备，每次收到粒子后随机选取某台设备进行测量，两台设备测量结果 $Q,R$ 只取值于 $\pm 1$。
3. Bob 也有两台测量设备，每次收到粒子后随机选取某台设备进行测量，两台设备测量结果 $S,T$ 也只取值于 $\pm 1$。
4. 重复以上1-3步，记录实验结果并计算出均值 $\mathbf{E}(QS)$、$\mathbf{E}(QT)$、$\mathbf{E}(RS)$、$\mathbf{E}(RT)$。

注意在第一步中要求 Charlie 每次制备的粒子状态相同。为了保证 Alice 和 Bob 之间的测量互不影响，我们可以假定 Alice 和 Bob 离得足够远，且和 Charlie 的距离相同，Charlie制备的两个粒子以相同的速度分别朝 Alice 和 Bob 运动，两者在收到粒子后马上进行测量，这种绝对精确的同时测量和足够远的距离都保证了 Alice 和 Bob 所做的测量之间不存在相互干扰。人们发现 Alice 和 Bob 进行联合测量后上述四项的平均值总是满足不等式

$$
\mathbf{E}(QS)+\mathbf{E}(RS)+\mathbf{E}(RT)-\mathbf{E}(QT) \le 2.
\tag{4}
$$

这个不等式被称为 CHSH 不等式，是 Bell 不等式常见的一种。

## 通过量子力学预测结果

如果我们引入量子力学的相关知识并进行如下实验：
Charlie 可以制备两个纠缠的粒子，为了方便，我们假设两个粒子处于纠缠 Bell 态

$$
|\psi\rangle = \frac{(|01\rangle-|10\rangle)}{\sqrt{2}}.
\tag{5}
$$

其中 Dirac 符号 $|ab\rangle$ 中 $a,b$ 分别代表粒子1和2所处的状态，这里我们假定了两个粒子状态由 $|0\rangle,|1\rangle$ 两个基矢展开。Charlie 制备好上述 Bell 态后将第一个粒子发给 Alice，第二个粒子发给 Bob。而 Alice 和 Bob 可以对两个粒子进行如下测量

$$
\begin{align}
Q=Z_1&, R = X_1, \\
T = \frac{Z_2-X_2}{\sqrt{2}}&, S=\frac{-Z_2-X_2}{\sqrt{2}},
\end{align}
\tag{6}
$$

这里 $Z_1,Z_2$ 分别表示作用在粒子1和2上的 Pauli $Z$ 算符，$X_1, X_2$ 分别表示作用在粒子1和2的 Pauli $X$ 算符。这四个算符都具有 $\pm 1$ 的两个本征值。量子力学中，状态 $|\psi\rangle$ 在观测量 $W$ 下的期望值为 $\mathbf{E}(W)=\langle\psi|W|\psi\rangle$，利用 Pauli 算符的性质 $Z|0\rangle=|0\rangle, Z|1\rangle=-|1\rangle$ 和 $X|0\rangle=|1\rangle, X|1\rangle=|0\rangle$，可以方便的得到

$$
\begin{align}
\mathbf{E}(QS)&=\langle\psi|Z_1\otimes (\frac{-Z_2-X_2}{\sqrt{2}})|\psi\rangle \\
&= \frac{1}{2\sqrt{2}}(\langle 01|-\langle 10|)[Z_1\otimes(-Z_2-X_2)](|01\rangle-|10\rangle)\\
&= \frac{1}{2\sqrt{2}}(\langle 01|-\langle 10|)(|01\rangle-|10\rangle-|00\rangle-|11\rangle)\\
&= \frac{1}{2\sqrt{2}}(\langle 01|01\rangle + \langle 10|10\rangle)\\
&= \frac{1}{\sqrt{2}}.
\end{align}
\tag{7}
$$

同样的，我们能得到

$$
\mathbf{E}(RS) = \frac{1}{\sqrt{2}}, \mathbf{E}(RT) = \frac{1}{\sqrt{2}}, \mathbf{E}(QT) = -\frac{1}{\sqrt{2}}.
\tag{8}
$$

从而我们发现

$$
\mathbf{E}(QS) + \mathbf{E}(RS) +\mathbf{E}(RT)-\mathbf{E}(QT)=2\sqrt{2}>2,
\tag{9}
$$

之前的 CHSH 不等式是被违背的！这说明 EPR 中的实在性和局域性的假设并不适用于量子力学，同时这也是量子力学有别于经典力学的地方。

## 量易伏平台演示

从式 7，我们可以看出

$$
\begin{align}
\mathbf{E}(QS)&=\langle\psi|Z_1\otimes (\frac{-Z_2-X_2}{\sqrt{2}})|\psi\rangle \\
&= -\frac{1}{\sqrt{2}}\langle\psi|Z_1\otimes Z_2|\psi\rangle - \frac{1}{\sqrt{2}}\langle\psi|Z_1\otimes X_2|\psi\rangle,
\end{align}
\tag{10}
$$

该期望值 $\mathbf{E}(QS)$ 由两部分组成，一部分是 $Z\otimes Z$ 测量的期望值，另一部分是 $Z\otimes X$ 测量的期望值。同样的，我们可以得到 $\mathbf{E}(RS)$，$\mathbf{E}(RT)$，$\mathbf{E}(QT)$ 也都含有两项。

$$
\begin{align}
\mathbf{E}(RS)&=-\frac{1}{\sqrt{2}}\langle\psi|X_1\otimes Z_2|\psi\rangle - \frac{1}{\sqrt{2}}\langle\psi|X_1\otimes X_2|\psi\rangle \\
\mathbf{E}(RT)&=\frac{1}{\sqrt{2}}\langle\psi|X_1\otimes Z_2|\psi\rangle - \frac{1}{\sqrt{2}}\langle\psi|X_1\otimes X_2|\psi\rangle \\
\mathbf{E}(QT)&=\frac{1}{\sqrt{2}}\langle\psi|Z_1\otimes Z_2|\psi\rangle - \frac{1}{\sqrt{2}}\langle\psi|Z_1\otimes X_2|\psi\rangle. \\
\end{align}
\tag{11}
$$

```python
from QCompute import *
import numpy as np
from collections import Counter
from random import choice

# 请输入您的 Token
# Define.hubToken= 'your token'


def choose_backend():
    # 您可以在这里选择您想要使用的后端，在选择‘量子真机’和‘云端模拟器’时
    # 请先输入您 量易伏 平台的 Token，否则无法正常运行
    # 使用本地模拟器
    backend = BackendName.LocalBaiduSim2
    # 使用量子设备
    # backend = BackendName.CloudIoPCAS
    # 使用云端模拟器
    # backend = BackendName.CloudBaiduSim2Water
    return backend


# 创建字典用来记录第一项的测量结果
result1 = {'QS': [],
           'QT': [],
           'RS': [],
           'RT': []}

# 创建字典用来记录第二项的测量结果
result2 = {'QS': [],
           'QT': [],
           'RS': [],
           'RT': []}

# 进行 100 次实验
times = 100
for i in range(times):
    # Alice 在 'Q' 和 'R' 中随机挑选进行测量
    ranA = choice(['Q', 'R'])
    # Bob 在 'S' 和 'T' 中随机挑选进行测量
    ranB = choice(['S', 'T'])
    ran = str(ranA) + str(ranB)
    
    # 每次测量只有1个 shot
    shots = 1
    env = QEnv()
    env.backend(choose_backend())
                    
    q = [env.Q[0], env.Q[1]]
    # 制备 Bell 态
    X(q[0])
    X(q[1])
    H(q[0])
    CX(q[0], q[1])
    
    if ran[0] == 'R':
        H(q[0])
        
    MeasureZ(q, range(2))
    taskResult = env.commit(shots, fetchMeasure=True)['counts']
    # 记录第一项的测量结果
    for key, value in taskResult.items():
        if value == 1:
            result1[ran].append(key)
    
    # 对第二项进行测量
    shots = 1
    env = QEnv()
    env.backend(choose_backend())

    q = [env.Q[0], env.Q[1]]
    # 制备 Bell 态
    X(q[0])
    X(q[1])
    H(q[0])
    CX(q[0], q[1])
    H(q[1])
    
    if ran[0] == 'R':
        H(q[0])
        
    MeasureZ(q, range(2))
    taskResult = env.commit(shots, fetchMeasure=True)['counts']
    # 记录第二项的测量结果
    for key, value in taskResult.items():
        if value == 1:
            result2[ran].append(key)
```

```python
QS1 = Counter(result1['QS'])
QS2 = Counter(result2['QS'])
RS1 = Counter(result1['RS'])
RS2 = Counter(result2['RS'])
RT1 = Counter(result1['RT'])
RT2 = Counter(result2['RT'])
QT1 = Counter(result1['QT'])
QT2 = Counter(result2['QT'])

def exp(Measure):
    # 计算测量结果的期望值
    summary = Measure["00"]-Measure["01"]-Measure["10"]+Measure["11"]
    total = Measure["00"]+Measure["01"]+Measure["10"]+Measure["11"]
    return 1/np.sqrt(2)*summary/total

a_list = [QS1, QS2, RS1, RS2, RT1, RT2, QT1, QT2]
```

```python
# 合并两个子项得到大项的期望值
QS = -exp(QS1)-exp(QS2)
RS = -exp(RS1)-exp(RS2)
RT = exp(RT1)-exp(RT2)
QT = exp(QT1)-exp(QT2)
```

```python
print('E(QS)=',QS)
print('E(RS)=',RS)
print('E(RT)=',RT)
print('E(QT)=',QT)

print('Expected value: E(QS)+E(RS)+E(RT)-E(QT)=', QS+RS+RT-QT)
```

```
E(QS)= 0.6934172386033232
E(RS)= 0.7002912941389663
E(RT)= 0.7008340597405378
E(QT)= -0.6960715012880779
Expected value: E(QS)+E(RS)+E(RT)-E(QT)= 2.7906140937709054
```

从模拟的结果来看，$\mathbf{E}(QS)\approx \frac{1}{\sqrt{2}}, \mathbf{E}(RS)\approx \frac{1}{\sqrt{2}}, \mathbf{E}(RT)\approx \frac{1}{\sqrt{2}}, \mathbf{E}(QT)\approx -\frac{1}{\sqrt{2}}$，这个结果和理论推导吻合（式 7，式 8）。模拟所得的期望值 $\mathbf{E}(QS)+\mathbf{E}(RS)+\mathbf{E}(RT)-\mathbf{E}(QT) \approx 2\sqrt{2}>2$，不满足 Bell 不等式。

---

## 参考资料

[1] Skibba, Ramin. "Einstein, Bohr and the war over quantum theory." [Nature 555.7698 (2018).](https://www.nature.com/articles/d41586-018-03793-2)

[2] Einstein, Albert, Boris Podolsky, and Nathan Rosen. "Can quantum-mechanical description of physical reality be considered complete?." [Physical Review 47.10 (1935): 777.](https://journals.aps.org/pr/abstract/10.1103/PhysRev.47.777)

[3] Bell, John S. "On the Einstein-Podolsky-Rosen paradox." [Physics 1.3 (1964): 195.](https://journals.aps.org/ppf/pdf/10.1103/PhysicsPhysiqueFizika.1.195)
