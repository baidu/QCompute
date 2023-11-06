# 基于测量的量子计算

*版权所有 (c) 2022 百度量子计算研究所，保留所有权利。*

## 概述

量子计算利用量子世界中特有的运行规律，为我们提供了一种全新的并且非常有前景的信息处理方式。其计算的本质是通过特定的方式将初始制备的量子态演化成我们预期的另一个量子态，然后在演化后的量子态上做测量以获得计算结果。但是在不同模型下，量子态的演化方式各异。比较常用的量子电路模型 (quantum circuit model) [1,2] 通过对量子态进行量子门操作来完成演化，该模型可以理解为经典计算模型的量子版本，被广泛地应用在量子计算领域中。而基于测量的量子计算 （Measurement-Based Quantum Computation, MBQC) 是一种完全不同于量子电路模型的计算方式。

顾名思义，MBQC 模型的计算过程是通过量子测量完成的。基于测量的量子计算主要有两种子模型：隐形传态量子计算 (Teleportation-based Quantum Computing, TQC) 模型 [3-5]
和单向量子计算机 (One-Way Quantum Computer, 1WQC) 模型
[6-9]。其中，前者需要用到多量子比特的联合测量，而后者只需要单比特测量即可。有趣的是，这两种子模型在被分别提出后，被证明是高度相关并且一一对应的 [10]。所以我们此后将不加声明地，默认讨论的 MBQC 模型为 1WQC 模型。

与电路模型不同，MBQC 是量子计算特有的一种模型，没有经典计算模型的对应。该模型的核心思想在于对一个量子纠缠态的部分比特进行测量，未被测量的量子系统将会实现相应的演化，并且通过对测量方式的控制，我们可以实现任意需要的演化。MBQC
模型下的计算过程主要分为三个步骤：第一步，准备一个资源态 (resource state)，即一个高度纠缠的多体量子态，该量子态可以在计算开始之前准备好，且可以与具体计算任务无关；第二步，对准备好的资源态的每个比特依次做单比特测量，其中后续比特的测量方式可以根据已经被测量的比特的测量结果做出相应调整，即允许适应性测量 (adaptive measurement)；第三步，对测量后得到的量子态进行副产品纠正 (byproduct correction)。最后，我们对所有比特的测量结果进行经典数据处理，即可得到需要的计算结果。

图 1 给出了一个 MBQC 模型下的算法示例。图中的网格代表了一种常用的量子资源态,（称为团簇态，cluster state，详见下文），网格上的每个节点都代表了一个量子比特，整个网格则代表了一个高度纠缠的量子态。我们依次对每个比特进行测量（节点中的 $X, Y, Z, XY$ 等表示对应的测量基），对测量后的量子态进行副产品纠正（消除 Pauli $X$ 算符和 Pauli $Z$ 算符），即可完成计算。

![MBQC example](./figures/mbqc-fig-general_pattern.jpg "图 1: 通过对网格上的每个比特进行测量来完成计算")

MBQC 模型的计算方式给我们带来了诸多好处。比如说，如果第一步制备的量子态噪声太大，我们则可以在计算开始之前（即测量之前)
丢弃这个态，并重新制备，以此保证计算结果的准确性；由于资源态可以与计算任务无关，因此可以应用在安全代理计算中 [11,
12]，保护计算的隐私；另外，单比特量子测量在实验上比量子门更容易实现，保真度更高，并且无适应性依赖关系的量子测量可以同步进行，从而降低整个计算的深度，对量子系统相干时间要求更低。MBQC
模型实现上的技术难点主要在于第一步资源态的制备，该量子态高度纠缠，并且制备所需的比特数比通常电路模型的多很多。关于资源态制备的相关进展，有兴趣的读者可以参见 [13,14] 。下表概括了 MBQC 模型与量子电路模型的优势和限制。

|    | 量子电路模型     | 基于测量的量子计算模型    |
|:---: | :---: | :---: |
| 优势|  与经典计算模型对应；易于理解和拓展应用 | 资源态可与计算无关；单比特测量易于操作；可并行测量，算法深度低 |
|限制| 量子门执行顺序固定；电路深度受相干时间限制| 无经典对应，不直观；资源态比特数多，制备难度高 | 

## 预备知识

在正式介绍 QNET 中的 MBQC 模块之前，我们首先回顾一下掌握 MBQC 模型需要用到的两个核心知识点。

### 1. 图与图态
    
对于任意给定一个图 $G=(V, E)$，其中，$V$ 是点的集合，$E$ 是边的集合，我们可以定义一个量子态与之对应。具体的做法为，将图 $G$ 中每个节点对应一个加态 $|+\rangle = (|0\rangle +
|1\rangle) / \sqrt{2}$，如果图中两个节点之间有边相连，则将对应节点上的加态之间作用控制 Z 门， $CZ = |0\rangle\langle0| \otimes I +
|1\rangle\langle1|\otimes Z$。由此步骤生成的量子态称为图 $G$ 的图态 (graph state)，记为 $|G\rangle$，具体数学表达式如下：
    
$$
|G\rangle = \prod_{(a,b) \in E} CZ_{ab} \left(\bigotimes_{v \in V}|+\rangle_v\right). \tag{1}
$$

图态其实并不是一个陌生的概念。通过局部的酉变换，我们所熟知的 Bell 态、GHZ 态等都可以表示为一个图对应的图态；此外，如果我们考虑的图具有周期网状的晶格结构（简单理解为二维坐标系的网格图），那么其对应的图态称为团簇态 (cluster state)，如图 2 所示。

![Graph states](./figures/mbqc-fig-graph_states.jpg "图 2：图 (i) 对应的图态为 Bell 态，图 (ii) 对应的图态为一个 4-qubit 的 GHZ 态，图 (iii) 对应一个团簇态")

### 2. 投影测量

量子测量是量子信息处理中的核心概念之一，在电路模型中，量子测量往往出现在电路末端，用于从量子态中解码出我们需要的经典结果。但是在 MBQC 模型中，量子测量不仅用于解码算法答案，还用于控制量子态的演化过程，即：通过对纠缠的多体量子态进行部分测量，驱动未测量的量子态进行演化。在 MBQC 模型中，我们默认使用单比特测量，且以 0/1 投影测量为主。根据测量公理 [17]，假设待测量的量子态为 $|\phi\rangle$，投影测量由一对正交基 $\{|\psi_0\rangle, |\psi_1\rangle\}$ 给出，那么测量结果为 $s \in \{0,1\}$ 的概率为 $p(s) = |\langle \psi_s|\phi\rangle|^2$，测量后对应的量子态坍缩为 $|\psi_s\rangle\langle\psi_s|\phi\rangle / \sqrt{p(s)}$，即被测量的比特坍缩为 $|\psi_s\rangle$，其他比特演化为 $\langle\psi_s|\phi\rangle / \sqrt{p(s)}$。

特别地，我们常用到的单比特测量为 $XY$, $YZ$, $XZ$ 三个平面上的投影测量，它们分别由如下的正交基给出，

- XY 平面测量：$M^{XY}(\theta) = \{R_z(\theta) |+\rangle, R_z(\theta) |-\rangle \}$，其中，当 $\theta = 0$ 时为 $X$ 测量；当 $\theta = \frac{\pi}{2}$ 时为 $Y$ 测量；

- YZ 平面测量：$M^{YZ}(\theta) = \{R_x(\theta)|0\rangle, R_x(\theta)|1\rangle\}$，其中，当 $\theta = 0$ 时为 $Z$ 测量；

- XZ 平面测量：$M^{XZ}(\theta) = \{R_y(\theta)|0\rangle, R_y(\theta)|1\rangle\}$，其中，当 $\theta = 0$ 时为 $Z$ 测量；

以上 $|+\rangle = (|0\rangle + |1\rangle)/ \sqrt{2},|-\rangle = (|0\rangle - |1\rangle)/ \sqrt{2}$, 且 $R_x, R_y, R_z$ 分别为绕 $x,y,z$ 轴旋转的单比特旋转门。

## MBQC 模块


### 1. 技术路线及代码实现

#### “三步走”流程

前面提到，MBQC 模型不同于常见的量子电路模型，该模型中量子态的演化是通过对量子图态上的部分比特进行测量来实现的。具体地，MBQC 模型由以下三个步骤构成。

- **量子图态准备**：即准备一个多体纠缠态。一般地，我们给出图（点和边）的信息，初始化图中节点为加态，根据图中节点的连线方式作用控制 Z 门，便可以生成量子图态。以此对应关系，每当我们给定一个图的信息，我们便可以在其上定义对应的量子图态。此外，我们还可以根据需要选择性替换图态中某些节点上的加态为指定的输入态。
- **单比特测量**：按照特定的测量方式对上一步准备好的量子图态进行单比特测量，测量角度可以根据已获得的测量结果进行动态调整。无适应性依赖关系的测量可以交换顺序或同时进行。
- **副产品纠正**：由于测量结果的随机性，未测量量子态的演化方式不能唯一确定，换句话说，未测量的量子态有可能会进行一些多余的演化。我们称这些多余的演化为副产品（byproduct
）。因而算法的最后一步就是对副产品进行纠正，得到我们预期的演化结果。如果算法最后要求输出的不是一个量子态，而是对演化完的量子态继续进行测量并获取经典结果的话，副产品的影响只需要通过经典数据处理来修正即可。因此，MBQC
模型的主要步骤为前两步，第三步是否进行则是取决于我们想要获得的是量子态的输出还是测量结果的经典输出。

依次进行上述三个步骤，我们可以概括出 MBQC 模型“三步走”的流程，即：量子图态准备、单比特测量和副产品纠正。

#### 测量模式与 "EMC" 语言

除了常用的“三步走”流程之外，一个 MBQC 模型还可以用 "EMC" 语言 [18] 来描述。如前所述，MBQC 模型与电路模型具有一一对应关系。我们可以把由电路模型对应的 MBQC 模型称为该电路模型的测量模式 (pattern) ，把电路中的单个量子门或对输出态的单个测量对应的 MBQC 模型称为该量子门或测量对应的子模式 (subpattern) [18]。在描述 MBQC 的 "EMC" 语言中，我们将纠缠操作对应 “纠缠命令”，用符号 "E" 来表示；将测量操作对应 “测量命令”，用符号 "M" 来表示；将副产品纠正操作对应 “副产品纠正命令”，用符号 "C" 来表示。于是，对应于上述“三步走”流程，一个完整的 MBQC 运算过程还可以用“命令列表” \[EMC\] 来表示。运算过程则是按照命令列表从左至右的顺序执行各个命令。为了让大家快速地熟悉 MBQC 模型，在本教程中，我们采用经典的“三步走”流程来描述 MBQC 模型的运算过程。需要注意的是，“三步走”流程和 "EMC" 语言只是 MBQC 模型运算过程的不同表示方式，两者本质是一样的。

#### 代码实现

代码实现上，MBQC 计算引擎的核心内容是 ``MBQC`` 类，该类具有与 MBQC 模型相关的属性和类方法。我们可以根据具体情况自行实例化 MBQC 类，然后通过依次调用相关类方法就可以完成 MBQC 模型的运算过程。接下来我们通过如下表格简单介绍一下常用的类方法及其功能。更为详细和全面的介绍请参考相关 API 文档。

|        类方法        |              功能              |
| :--------------------: | :------------------------------: |
|      `set_graph`      |          输入图的信息          |
|     `set_pattern`     |       输入测量模式的信息       |
|   `set_input_state`   |      输入初始量子态的信息      |
|     `draw_process`     | 画出 MBQC 模型运算过程的动态图 |
|    `track_progress`    |   查看 MQBC 模型运算的进度条   |
|       `measure`       |         执行单比特测量         |
|     `sum_outcomes`     |  对指定节点的测量结果进行求和  |
|  `correct_byproduct`  |           纠正副产品           |
|     `run_pattern`     |          运行测量模式          |
| `get_classical_output` |        获取经典输出结果        |
|  `get_quantum_output`  |        获取量子输出结果        |

在 MBQC 模块中，为了方便大家使用，我们设置了“图”（graph）和“模式”（pattern）的两种输入方式，分别对应于 MBQC 模型运算过程的两种描述方式。如果输入为图，则后续运算过程需要我们自行按照“三步走”流程完成。值得一提的是，我们设计了节点动态分类算法来模拟 MBQC 的运算过程，简单来说，就是将 MBQC “三步走”流程中的第一、二步进行整合，交换某些纠缠和测量操作，从而降低实际参与运算的比特数，提高运算效率。MBQC 模拟的具体调用方式如下：

```python
"""
MBQC 模块调用格式（以图为输入，进行“三步走”流程）
"""
from Extensions.QuantumNetwork.qcompute_qnet.quantum.backends.mbqc import MBQC

# 实例化 MBQC 类创建一个模型
mbqc = MBQC()

# “三步走”中第一步，设置图
mbqc.set_graph(graph)

# 设置初始态 （可选）
mbqc.set_input_state(input_state)

# “三步走”中第二步，单比特测量
mbqc.measure(which_qubit, basis)
mbqc.measure(which_qubit, basis)
...

# “三步走”中第三步，纠正副产品
mbqc.correct_byproduct(gate, which_qubit, power)

# 输出运行后的经典和量子输出结果
classical_output = mbqc.get_classical_output()
quantum_output = mbqc.get_quantum_output()
```

如果输入为测量模式，只需调用 `run_pattern` 类方法即可完成 MBQC 模型运算过程，格式如下：

```python
"""
MBQC 模块调用格式（以测量模式为输入，执行 "EMC" 命令）
"""
from Extensions.QuantumNetwork.qcompute_qnet.quantum.backends.mbqc import MBQC

# 实例化 MBQC 类创建一个模型
mbqc = MBQC()

# 设置测量模式
mbqc.set_pattern(pattern)

# 设置初始态 （可选）
mbqc.set_input_state(input_state)

# 运行测量模式
mbqc.run_pattern()

# 输出运行后的经典和量子输出结果
classical_output = mbqc.get_classical_output()
quantum_output = mbqc.get_quantum_output()
```

跟据前面的介绍，相信大家对 MBQC 模型以及我们设计的 MBQC 模块有了大致的了解。下面我们用两个示例带领大家进行一些实战。

### 2. 使用示例：用 MBQC 实现任意单比特量子门

跟据量子门的分解，我们知道任意单比特量子门 $U$ 都可以分解为 $ U = R_x(\gamma)R_z(\beta)R_x(\alpha)$ 的形式（忽略全局相位）[17] 。在 MBQC 模型中，这样的单比特量子门可以按如下的方式实现（参见图 3） [15] ：准备五个量子比特，最左侧是输入比特，最右侧为输出比特。输入量子态 $|\psi\rangle$，其余量子比特初始化为加态，相邻比特作用控制 Z 门，对第一个比特作 $X$ 测量，对中间三个比特依次进行适应性测量，前四个比特的测量结果依次记为 $s_1$, $s_2$, $s_3$, $s_4$，根据测量结果对得到的量子态进行副产品修正，则在第五个比特上输出的结果为 $U|\psi\rangle$。

![Single qubit pattern](./figures/mbqc-fig-single_qubit_pattern.jpg "图 3: MBQC 模型下任意单比特量子门的实现方式")

**注意**：测量完前四个比特后，第五个量子比特的状态为 $X^{s_2 + s_4}Z^{s_1 + s_3} U|\psi\rangle$，其中 $X^{s_2 + s_4}Z^{s_1 + s_3}$ 就是所谓的副产品，我们需要跟据测量结果，对此进行修正，才能得到想要的 $U|\psi\rangle$。

以下是代码展示：

#### 引入计算所需要的模块

```python
from numpy import pi, random

from Extensions.QuantumNetwork.qcompute_qnet.quantum.backends.mbqc import MBQC
from Extensions.QuantumNetwork.qcompute_qnet.quantum.basis import Basis
from Extensions.QuantumNetwork.qcompute_qnet.quantum.gate import Gate
from Extensions.QuantumNetwork.qcompute_qnet.quantum.state import PureState
```

#### 输入图和量子态

接下来，我们可以自定义要输入的图，在此例中，如图 $3$ 所示，我们需要输入的是五个带标签的节点（记作 `['1', '2', '3', '4', '5']` ）和图中的四条边（记作 `[('1', '2'), ('2', '3'), ('3', '4'), ('4', '5')]` ），并在最左侧的比特 `'1'` 上输入量子态，同时初始化测量的角度。

```python
# 构造用于 MBQC 计算的图
V = ['1', '2', '3', '4', '5']
E = [('1', '2'), ('2', '3'), ('3', '4'), ('4', '5')]
G = [V, E]
# 生成一个随机的量子态向量
input_vec = PureState.random_state_vector(1, is_real=False)

# 初始化角度
alpha = pi * random.uniform()
beta = pi * random.uniform()
gamma = pi * random.uniform()
```

#### 初始化 MBQC 模型

实例化 `MBQC` 类并设置图和输入量子态的信息，就可以搭建属于自己的 MBQC 模型了。

```python
# 实例化 MBQC 类
mbqc = MBQC()
# 输入图的信息
mbqc.set_graph(G)
# 输入初始量子态的信息
mbqc.set_input_state(PureState(input_vec, ['1']))
```

之后，我们依次对四个节点进行测量。

#### 测量第一个节点

根据图 3，第一个比特的测量方式为 $X$ 测量，也就是 $XY$ 平面测量角度为 $0$ 的情形，即 $\theta_1 = 0$。 

```python
# 计算第一个比特的测量角度
theta_1 = 0
# 对第一个比特进行测量
mbqc.measure('1', Basis.Plane('XY', theta_1))
```

第一个比特的测量不涉及适应性的问题，所以比较简单，但对于第二、三和四个比特而言，其测量角度就需要考虑前面的测量结果了。

#### 测量第二个节点

根据图 3，第二个比特的测量方式为 $M^{XY}(\theta_2)$ 测量，其中，

$$
\theta_2 = (-1)^{s_1 + 1} \alpha, \tag{2}
$$

也就是 $XY$ 平面测量角度为 $(-1)^{s_1 + 1} \alpha$ 的测量，其中 $s_1$ 为第一个节点的测量结果。 

在 `MBQC` 类中，我们定义了类方法 `sum_outcomes` ，可以对指定输入标签的量子比特的测量结果进行求和运算，如果想要对求和结果额外加上一个数字 $x$，则可在第二个参数处赋值 $x$，否则为 $0$。


```python
# 计算第二个比特的测量角度
theta_2 = (-1) ** mbqc.sum_outcomes(['1'], 1) * alpha
# 对第二个比特进行测量
mbqc.measure('2', Basis.Plane('XY', theta_2))
```

#### 测量第三个节点

根据图 3，第三个比特的测量方式为 $M^{XY}(\theta_3)$ 测量，其中，

$$
\theta_3 = (-1)^{s_2 + 1} \beta, \tag{3}
$$

也就是 $XY$ 平面测量角度为 $(-1)^{s_2 + 1} \beta$ 的测量，其中 $s_2$ 为第二个节点的测量结果。 


```python
# 计算第三个比特的测量角度
theta_3 = (-1) ** mbqc.sum_outcomes(['2'], 1) * beta
# 对第三个比特进行测量
mbqc.measure('3', Basis.Plane('XY', theta_3))
```

#### 测量第四个节点

根据图 3，第四个比特的测量方式为 $M^{XY}(\theta_4)$ 测量，其中，

$$
\theta_4 = (-1)^{s_1 + s_3 + 1} \gamma, \tag{4}
$$

也就是 $XY$ 平面测量角度为 $(-1)^{s_1 + s_3 + 1} \gamma$ 的测量，其中 $s_1$ 为第一个节点的测量结果，其中 $s_3$ 为第三个节点的测量结果。 


```python
# 计算第四个比特的测量角度
theta_4 = (-1) ** mbqc.sum_outcomes(['1', '3'], 1) * gamma
# 对第四个比特进行测量
mbqc.measure('4', Basis.Plane('XY', theta_4))
```

#### 对第五个节点输出的量子态进行修正

前四个节点测量结束之后，第五个节点上的输出量子态并不是 $U|\psi\rangle$，而是附带有副产品的量子态 $X^{s_2 + s_4}Z^{s_1 + s_3} U|\psi\rangle$， 如果希望输出量子态为 $U|\psi\rangle$，需要在测量结束之后对副产品进行修正。


```python
# 对量子态的副产品进行修正
mbqc.correct_byproduct('X', '5', mbqc.sum_outcomes(['2', '4']))
mbqc.correct_byproduct('Z', '5', mbqc.sum_outcomes(['1', '3']))
```

#### 读取修正后的量子态并与预期的量子态进行比较

调用 `get_classical_output` 和 `get_quantum_output` 分别获取经典和量子输出结果。

```python
# 读取量子输出结果
state_out = mbqc.get_quantum_output()

# 计算预期的量子态列向量
vec_std = Gate.Rx(gamma) @ Gate.Rz(beta) @ Gate.Rx(alpha) @ input_vec
# 构造预期的量子态
state_std = PureState(vec_std, ['5'])

# 与预期的输出态进行比较
print(state_out.compare_by_vector(state_std))
```

### 3. 使用示例： 用 MBQC 实现 CNOT 门

CNOT 门是电路模型中常用的两比特门，在 MBQC 模型中， CNOT 门的实现方案如下（参见图 4） [7]：准备 $15$ 个量子比特，第 $1$、$9$ 比特是输入比特，最右侧 $7$ 和 $15$ 为输出比特。输入量子态 $|\psi\rangle$，其余量子比特初始化为加态，图中相连接的比特作用控制 Z 门。对第 $1, 9, 10, 11, 13, 14$ 做 $X$ 测量，对 $2, 3, 4, 5, 6, 8, 12$ 做 $Y$ 测量（注意：这些测量的角度无依赖关系，交换测量顺序对测量结果没有影响），对副产品算符进行修正后，在 $7$ 和 $15$ 输出的量子比特将会为 $\text{CNOT}|\psi\rangle$。

![CNOT pattern](./figures/mbqc-fig-cnot_pattern.jpg "图 4: MBQC 模型下 CNOT 门的一种实现方式")

**注意**：与前面的单比特量子门类似，我们需要在测量完之后对副产品进行修正才能得到预期的 $\text{CNOT}|\psi\rangle$。

以下是完整的代码展示：

```python
from numpy import pi, random

from Extensions.QuantumNetwork.qcompute_qnet.quantum.backends.mbqc import MBQC
from Extensions.QuantumNetwork.qcompute_qnet.quantum.basis import Basis
from Extensions.QuantumNetwork.qcompute_qnet.quantum.gate import Gate
from Extensions.QuantumNetwork.qcompute_qnet.quantum.state import PureState

# 定义 X 测量和 Y 测量
X_basis = Basis.X()
Y_basis = Basis.Y()

# 定义用于 MBQC 计算的图
V = [str(i) for i in range(1, 16)]
E = [('1', '2'), ('2', '3'), ('3', '4'), ('4', '5'),
     ('5', '6'), ('6', '7'), ('4', '8'), ('8', '12'),
     ('9', '10'), ('10', '11'), ('11', '12'),
     ('12', '13'), ('13', '14'), ('14', '15')]
G = [V, E]

# 生成一个随机的量子态列向量
input_psi = PureState.random_state_vector(2, is_real=True)

# 初始化 MBQC 类
mbqc = MBQC()
# 输入图的信息
mbqc.set_graph(G)
# 输入初始量子态的信息
mbqc.set_input_state(PureState(input_psi, ['1', '9']))

# 依次对节点进行测量，注意以下测量顺序可以任意交换
mbqc.measure('1', X_basis)
mbqc.measure('2', Y_basis)
mbqc.measure('3', Y_basis)
mbqc.measure('4', Y_basis)
mbqc.measure('5', Y_basis)
mbqc.measure('6', Y_basis)
mbqc.measure('8', Y_basis)
mbqc.measure('9', X_basis)
mbqc.measure('10', X_basis)
mbqc.measure('11', X_basis)
mbqc.measure('12', Y_basis)
mbqc.measure('13', X_basis)
mbqc.measure('14', X_basis)

# 计算副产品的系数
cx = mbqc.sum_outcomes(['2', '3', '5', '6'])
tx = mbqc.sum_outcomes(['2', '3', '8', '10', '12', '14'])
cz = mbqc.sum_outcomes(['1', '3', '4', '5', '8', '9', '11'], 1)
tz = mbqc.sum_outcomes(['9', '11', '13'])

# 对测量后的量子态进行副产品修正
mbqc.correct_byproduct('X', '7', cx)
mbqc.correct_byproduct('X', '15', tx)
mbqc.correct_byproduct('Z', '7', cz)
mbqc.correct_byproduct('Z', '15', tz)

# 读取量子输出结果
state_out = mbqc.get_quantum_output()

# 构造预期的量子态
vec_std = Gate.CNOT() @ input_psi
state_std = PureState(vec_std, ['7', '15'])

# 与预期的量子态作比较
print(state_out.compare_by_vector(state_std))
```

---

## 参考文献

[1] Deutsch, David Elieser. "Quantum computational networks." [Proceedings of the Royal Society of London. A. 425.1868 (1989): 73-90.](https://royalsocietypublishing.org/doi/abs/10.1098/rspa.1989.0099)

[2] Barenco, Adriano, et al. "Elementary gates for quantum computation." [Physical Review A 52.5 (1995): 3457.](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.52.3457)

[3] Gottesman, Daniel, and Isaac L. Chuang. "Demonstrating the viability of universal quantum computation using teleportation and single-qubit operations." [Nature 402.6760 (1999): 390-393.](https://www.nature.com/articles/46503?__hstc=13887208.d9c6f9c40e1956d463f0af8da73a29a7.1475020800048.1475020800050.1475020800051.2&__hssc=13887208.1.1475020800051&__hsfp=1773666937)

[4] Nielsen, Michael A. "Quantum computation by measurement and quantum memory." [Physics Letters A 308.2-3 (2003): 96-100.](https://www.sciencedirect.com/science/article/abs/pii/S0375960102018030)

[5] Leung, Debbie W. "Quantum computation by measurements." [International Journal of Quantum Information 2.01 (2004): 33-43.](https://www.worldscientific.com/doi/abs/10.1142/S0219749904000055)

[6] Robert Raussendorf, et al. "A one-way quantum computer." [Physical Review Letters 86.22 (2001): 5188.](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.86.5188)

[7] Raussendorf, Robert, and Hans J. Briegel. "Computational model underlying the one-way quantum computer." [Quantum Information & Computation 2.6 (2002): 443-486.](https://dl.acm.org/doi/abs/10.5555/2011492.2011495)

[8] Robert Raussendorf, et al. "Measurement-based quantum computation on cluster states." [Physical Review A 68.2 (2003): 022312.](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.68.022312)

[9] Briegel, Hans J., et al. "Measurement-based quantum computation." [Nature Physics 5.1 (2009): 19-26.](https://www.nature.com/articles/nphys1157)

[10] Aliferis, Panos, and Debbie W. Leung. "Computation by measurements: a unifying picture." [Physical Review A 70.6 (2004): 062314.](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.70.062314)

[11] Broadbent, Anne, et al. "Universal blind quantum computation." [2009 50th Annual IEEE Symposium on Foundations of Computer Science. IEEE, 2009.](https://arxiv.org/abs/0807.4154)

[12] Morimae, Tomoyuki. "Verification for measurement-only blind quantum computing." [Physical Review A 89.6 (2014): 060302.](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.89.060302)

[13] Larsen, Mikkel V., et al. "Deterministic generation of a two-dimensional cluster state." [Science 366.6463 (2019): 369-372.](https://science.sciencemag.org/content/366/6463/369)

[14] Asavanant, Warit, et al. "Generation of time-domain-multiplexed two-dimensional cluster state." [Science 366.6463 (2019): 373-376.](https://science.sciencemag.org/content/366/6463/373)

[15] Richard Jozsa, et al. "An introduction to measurement based quantum computation." [arXiv:quant-ph/0508124](https://arxiv.org/abs/quant-ph/0508124v2)

[16] Nielsen, Michael A. "Cluster-state quantum computation." [Reports on Mathematical Physics 57.1 (2006): 147-161.](https://www.sciencedirect.com/science/article/abs/pii/S0034487706800145)

[17] Nielsen, Michael A., and Isaac Chuang. "Quantum computation and quantum information."[Cambridge University Press (2010).](https://www.cambridge.org/core/books/quantum-computation-and-quantum-information/01E10196D0A682A6AEFFEA52D53BE9AE)

[18] Danos, Vincent, et al. "The measurement calculus." [Journal of the ACM (JACM) 54.2 (2007): 8-es.](https://dl.acm.org/doi/abs/10.1145/1219092.1219096)
