# 量子隐形传态

*版权所有 (c) 2022 百度量子计算研究所，保留所有权利。*

本教程中，我们将结合一个经典的量子网络协议——量子隐形传态对 QPU 模块的功能展开进一步详细的介绍。首先，我们将对量子隐形传态的原理和相关流程进行介绍；随后，我们将介绍如何借助 QNET 对该协议进行仿真，并获取量子电路运行后的返回结果。


## 1. 背景介绍

在量子通信中，存在这样一个常见的场景：Alice 想要将她手中所持有的一份未知的单比特量子态 $| \psi \rangle$ 发送给与她相距很远的 Bob。然而他们之间并没有直接相连的量子信道，因此无法完成对量子态的直接传输。由于量子力学的性质，Alice 也无法像复制经典消息那样复制多份量子态，并通过统计分析的方式获得量子态的相关准确信息，从而通过经典信道告知 Bob 相关参数来还原这个量子态。因此，在这种情境下，如何实现量子态的远距离传输成为一个关键的问题。在 1993 年，Charles Bennett 及其合作者提出了量子隐形传态协议 [1]，通过引入量子力学的纠缠特性，成功解决了对未知量子态的远距离传输。

## 2. 协议流程

量子隐形传态是量子信息理论当中非常经典的一个量子协议，该协议的流程大致如下：

1. 首先，Alice 和 Bob 需要共享一对贝尔态粒子 $| \Phi^+ \rangle _{AB}$；
2. 其次，Alice 将在本地对她手中的两个粒子（贝尔态中 Alice 持有的粒子和需要传输的粒子）进行联合测量；
3. 随后，她将对这两个粒子共同测量的经典结果 $M_1M_2$ 通过经典信道发送给 Bob；
4. 最后，由 Bob 根据 Alice 所发送的测量结果对他手中的粒子进行相应的泡利门操作，从而恢复出 Alice 一开始想要传输的量子态。

接下来，我们使用 QNET 对量子隐形传态协议进行仿真，并借此向大家简单演示如何使用 QNET 的语法实现一个量子网络协议。

## 3. 协议实现

在 QPU 模块中，我们提供了 ``Teleportation`` 类用以对量子隐形传态协议进行仿真，在该协议中我们定义了三个子协议（``SubProtocol``），分别描述了该协议中的三种角色各自的行为：负责制备并分发纠缠粒子的纠缠源（``Source``）、对量子态进行传送的发送方（``Sender``）以及对量子态进行还原的接收方（``Receiver``）。该协议中，所有角色在收到量子比特后都将存储在本地的量子寄存器中，方便后续访问其量子态并执行操作。


```python
class Teleportation(Protocol):    
    
    def __init__(self, name=None):
        super().__init__(name)
        self.role = None

    class Message(ClassicalMessage):

        def __init__(self, src: "Node", dst: "Node", protocol: type, data: Dict):
            super().__init__(src, dst, protocol, data)

        @unique
        class Type(Enum):

            ENT_REQUEST = "Entanglement request"
            OUTCOME_FROM_SENDER = "Measurement outcome from the sender"

    def start(self, **kwargs) -> None:
        role = kwargs['role']
        # 根据角色来实例化对应子协议，例如：若 role 的值为 "Sender"，则实例化 Sender 子协议
        self.role = getattr(Teleportation, role)(self)
        # 启动子协议
        self.role.start(**kwargs)  

    def receive_classical_msg(self, msg: "ClassicalMessage", **kwargs) -> None:
        # 由对应的子协议接收经典消息
        self.role.receive_classical_msg(msg) 

    def receive_quantum_msg(self, msg: "QuantumMsg", **kwargs) -> None:
        # 存储接收到的量子比特
        self.node.qreg.store_qubit(msg.data, kwargs['src'])  
        # 由对应的子协议做进一步的操作
        self.role.receive_quantum_msg()  
```

### 1. 纠缠源（``Source``）

纠缠源负责制备并分发纠缠。作为纠缠源，Charlie 在收到纠缠分发请求后，需要在本地制备一对贝尔态 $| \Phi^+ \rangle = \tfrac{1}{\sqrt{2}} (|00 \rangle + |11 \rangle)$，并通过量子信道分别将两个纠缠粒子发送给需要共享纠缠的通信双方。


```python
class Teleportation(Protocol):
    ...
    class Source(SubProtocol):

        def __init__(self, super_protocol: Protocol):
            super().__init__(super_protocol)

        def start(self, **kwargs) -> None:
            pass

        def receive_classical_msg(self, msg: "ClassicalMessage") -> None:
            # 如果接收到纠缠分发请求
            if msg.data['type'] == Teleportation.Message.Type.ENT_REQUEST:
                # 在本地生成一对纠缠粒子
                self.node.qreg.h(0)
                self.node.qreg.cnot([0, 1])

                # 对生成的纠缠粒子进行分发
                # 将本地寄存器地址为 0 的量子比特发送给请求纠缠的节点
                self.node.send_quantum_msg(dst=msg.src, qreg_address=0)  
                # 将本地寄存器地址为 1 的量子比特发送给另一个共享纠缠的节点
                self.node.send_quantum_msg(dst=msg.data['peer'], qreg_address=1)  
```

### 2. 发送方（``Sender``）

作为量子态的发送方，Alice 需要主动向纠缠源发起纠缠请求，获取与接收方共享的纠缠粒子。在收到纠缠粒子后，她将对该粒子以及她想要传输的量子比特 $| \psi \rangle$ 共同作用贝尔态测量，并将测量结果 $M_1M_2$ 通过经典信道发送给接收方 Bob。


```python
class Teleportation(Protocol):
    ...
    class Sender(SubProtocol):

        def __init__(self, super_protocol: Protocol):
            super().__init__(super_protocol)
            self.peer = None
            self.ent_source = None
            self.address_to_teleport = None

        def start(self, **kwargs) -> None:
            # 量子态的接收方
            self.peer = kwargs['peer']  
            # 对应的纠缠源节点
            self.ent_source = kwargs['ent_source']  
            # 等待传输的量子比特在寄存器中的地址
            self.address_to_teleport = kwargs['address_to_teleport']  
            # 向纠缠源发起纠缠请求
            self.request_entanglement()  

        def request_entanglement(self) -> None:
            # 生成一则纠缠请求消息
            ent_request_msg = Teleportation.Message(
                src=self.node, dst=self.ent_source, protocol=Teleportation,
                data={'type': Teleportation.Message.Type.ENT_REQUEST, 'peer': self.peer}
            )
            # 向纠缠源发送纠缠请求
            self.node.send_classical_msg(dst=self.ent_source, msg=ent_request_msg)  

        def receive_quantum_msg(self) -> None:
            # 获取接收到的量子比特在寄存器中的存储地址
            address_reception = self.node.qreg.get_address(self.ent_source)  
            # 在本地对待传输的量子态和所持有的纠缠态进行贝尔测量
            self.node.qreg.bsm([self.address_to_teleport, address_reception])  

            # 获取测量结果
            m1 = self.node.qreg.units[self.address_to_teleport]['outcome']
            m2 = self.node.qreg.units[address_reception]['outcome']

            # 将测量结果通过经典信道发送给接收方
            outcome_msg = Teleportation.Message(
                    src=self.node, dst=self.peer, protocol=Teleportation,
                    data={'type': Teleportation.Message.Type.OUTCOME_FROM_SENDER,
                          'outcome_from_sender': [m1, m2]}
                )
            self.node.send_classical_msg(dst=self.peer, msg=outcome_msg)
```

### 3. 接收方（``Receiver``）

作为量子态的接收方，Bob 需要同时等待纠缠源 Charlie 从量子信道发来的纠缠粒子以及 Alice 从经典信道发来的测量结果 $M_1M_2$。随后，他根据经典测量结果对他的粒子进行对应的泡利门操作来恢复 Alice 一开始想要传输的量子态 $| \psi \rangle$。本教程中，为了验证该量子态就是一开始发送方 Alice 想要传输的量子态 $| \psi \rangle$，在协议中我们规定，接收方在恢复量子态后，继续对其进行测量。


```python
class Teleportation(Protocol):
    ...
    class Receiver(SubProtocol):

        def __init__(self, super_protocol: Protocol):
            super().__init__(super_protocol)
            self.peer = None
            self.ent_source = None
            self.outcome_from_sender = None

        def start(self, **kwargs) -> None:
            # 量子态的发送方
            self.peer = kwargs['peer']  
            # 对应的纠缠源节点
            self.ent_source = kwargs['ent_source']  

        def receive_classical_msg(self, msg: "ClassicalMessage") -> None:
            # 如果接收到的消息是发送方的测量结果
            if msg.data['type'] == Teleportation.Message.Type.OUTCOME_FROM_SENDER:
                # 保存发送方的测量结果
                self.outcome_from_sender = msg.data['outcome_from_sender']  
                # 获取接收到的量子比特在寄存器中的存储地址
                address_reception = self.node.qreg.get_address(self.ent_source)  
                # 若已经接收到来自纠缠源的量子比特
                if self.node.qreg.units[address_reception]['qubit'] is not None:
                    # 进行对应的操作来恢复初始发送方想要传输的量子态
                    self.correct_state()

        def receive_quantum_msg(self) -> None:
            # 若接收到来自纠缠源的量子比特时，已经接收到来自发送方的经典测量结果
            if self.outcome_from_sender is not None:
                # 进行对应操作恢复发送方想要传输的量子态
                self.correct_state()

        def correct_state(self) -> None:
            # 获取纠缠源发送的量子比特在寄存器中的存储地址
            address_reception = self.node.qreg.get_address(self.ent_source)  
            # 根据发送方第一个量子比特的测量结果判断是否执行 Z 门
            self.node.qreg.z(address_reception, condition=self.outcome_from_sender[0])  
            # 根据发送方第二个量子比特的测量结果判断是否执行 X 门
            self.node.qreg.x(address_reception, condition=self.outcome_from_sender[1])  

            # 为验证恢复的量子态的正确性，对该量子态进行测量
            self.node.qreg.measure(address_reception)
```

## 4. 代码示例

接下来，我们使用 QNET 对量子隐形传态的完整使用场景进行模拟。

首先，我们创建一个仿真环境 ``QuantumEnv``。相比于之前教程中经常用到的 ``DESEnv``，``QuantumEnv`` 可以连接本地或云端模拟器运行由量子网络协议自动生成的量子电路并返回仿真结果。


```python
from qcompute_qnet.models.qpu.env import QuantumEnv

# 创建一个仿真环境
env = QuantumEnv("Teleportation", default=True)
```

随后，分别创建量子隐形传态协议中涉及到的三位角色：发送方 Alice、接收方 Bob 和纠缠源 Charlie，并配置他们之间的通信链路。

**注意**：为了方便使用，如果我们在创建链路时指定各条链路的距离值，在链路初始化时将自动装载 4 条逻辑信道（2 条经典信道和 2 条量子信道）。

在这里，我们使用了节点模板 ``QuantumNode``，在该类节点中预装了用以存储和处理量子信息的量子寄存器，通过传入 ``qreg_size`` 参数可以指定寄存器的大小。此外，可以通过传入 ``protocol`` 参数直接指定希望预装的协议，对应的协议实例将自动被创建并添加到节点的协议栈中。这里，我们指定为网络中每个节点预装 ``Teleportation`` 协议实例。


```python
from qcompute_qnet.models.qpu.node import QuantumNode
from qcompute_qnet.models.qpu.protocol import Teleportation
from qcompute_qnet.topology.link import Link

# 创建装有量子寄存器的量子节点并指定其中预装的协议类型为 Teleportation
alice = QuantumNode("Alice", qreg_size=2, protocol=Teleportation)
bob = QuantumNode("Bob", qreg_size=1, protocol=Teleportation)
charlie = QuantumNode("Charlie", qreg_size=2, protocol=Teleportation)

# 创建通信链路
link_ab = Link("Link_ab", ends=(alice, bob), distance=1e3)
link_ac = Link("Link_ac", ends=(alice, charlie), distance=1e3)
link_bc = Link("Link_bc", ends=(bob, charlie), distance=1e3)
```

然后，我们创建一个量子网络，并将配置好的节点和通信链路装入网络中。在 ``Network`` 中装有默认的全局量子电路 ``default_circuit``，用以存储由量子网络协议编译而成的量子电路。


```python
from qcompute_qnet.topology.network import Network

# 创建一个量子网络并将各节点和链路装入量子网络中
network = Network("Teleportation network")
network.install([alice, bob, charlie, link_ab, link_ac, link_bc])
```

接下来，我们让 Alice 随机生成一个用以传输的量子态 $| \psi \rangle$。通过调用 ``start`` 方法，可以启动节点协议栈中所装载的协议。


```python
import numpy

# 随机生成用于传输的量子态
theta, phi, gamma = numpy.random.randn(3)
print(f"Rotation angles (rad) of U3: theta: {theta:.4f}, phi: {phi:.4f}, gamma: {gamma:.4f}")
alice.qreg.u3(0, theta, phi, gamma)

# 启动量子隐形传态协议
alice.start(role="Sender", peer=bob, ent_source=charlie, address_to_teleport=0)
bob.start(role="Receiver", peer=alice, ent_source=charlie)
charlie.start(role="Source")
```

最后，对仿真环境 ``QuantumEnv`` 进行初始化并运行仿真。通过调用 ``QuantumEnv`` 的 ``run`` 方法，我们可以连接不同的量子后端对量子网络协议编译而成的量子电路进行运行并返回结果。其中，通过指定 ``shots`` 参数，用户可以设置量子电路的采样次数。通过设置 ``backend`` 连接本地、云端模拟器或量子真机来运行量子电路。此外，若在 ``backend`` 中选择使用云端模拟器，则需要额外传入 ``token`` 值进行访问。

完成仿真运行后，``QuantumEnv`` 将调用 ``Circuit`` 类的 ``print_circuit`` 方法以可视化的形式打印所生成的量子电路，并以字典形式返回电路运行结果。运行结果中将包含电路名称（``circuit_name``)、电路的采样次数（``shots``）以及电路的采样结果（``counts``）。


```python
from qcompute_qnet.quantum.backends import Backend

# 初始化仿真环境并运行仿真
env.init()
results = env.run(shots=1024, backend=Backend.QCompute.LocalBaiduSim2, summary=False)
# 查看电路运行结果
print(f"\nCircuit results:\n", results)  
```

对量子隐形传态协议进行仿真的运行结果如下所示。

```
Rotation angles (rad) of U3: theta: 0.7683, phi: 0.3789, gamma: 0.4121
```

![图 1：量子隐形传态协议的电路运行结果](./figures/teleportation-circuit_results.png "图 1：量子隐形传态协议的电路运行结果")

```
Colors: {'Alice': 'red', 'Charlie': 'blue', 'Bob': 'green'}

Circuit results:
 {'circuit_name': 'Circuit', 'shots': 1024, 'counts': {'000': 206, '001': 41, '010': 221, '011': 47, '100': 210, '101': 40, '110': 228, '111': 31}}
```

为了查看最终接收方 Bob 所持量子比特的测量结果，可以调用 ``Circuit`` 的 ``reduce_results`` 方法，传入电路采样结果，并通过 ``indices`` 参数传入想要查看测量结果的量子比特在电路中的全局索引，来获取对应的返回结果。为了验证协议仿真的正确性，我们可以另行创建一个电路，输入与发送方 Alice 一致的量子态并进行测量，通过比对测量结果以验证本次协议仿真的正确性。

```python
from qcompute_qnet.quantum.circuit import Circuit

# 查看接收方所持量子比特的测量结果
reduced_indices = [2]
reduced_results = network.default_circuit.reduce_results(results['counts'], indices=reduced_indices)
print(f"\nMeasurement results of the receiver:\n", reduced_results)

# 创建一个对比电路并作用 U3 门，采用相同的参数生成发送方初始想要发送的量子态并测量
comp_cir = Circuit("Circuit for verification")
comp_cir.u3(0, theta, phi, gamma)
comp_cir.measure(0)

# 查看测量结果并进行对比，以验证协议仿真的正确性
results = comp_cir.run(shots=1024, backend=Backend.QCompute.LocalBaiduSim2)
print(f"\nMeasurement results of the origin state for verification:\n", results['counts'])
```

运行结果对比如下：

```
Measurement results of the receiver:
 {'0': 865, '1': 159}

Measurement results of the origin state for verification:
 {'0': 888, '1': 136}
```

---

## 参考文献

[1] Bennett, Charles H, et al. "Teleporting an unknown quantum state via dual classical and Einstein-Podolsky-Rosen channels." [Physical Review Letters 70.13 (1993): 1895.](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.70.1895)
