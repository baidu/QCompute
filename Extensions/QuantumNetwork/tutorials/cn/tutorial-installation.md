# 安装

*版权所有 (c) 2022 百度量子计算研究所，保留所有权利。*

## 安装 Python 环境

我们建议使用 [Anaconda](https://www.anaconda.com/products/distribution) 作为开发环境。Anaconda 是一个强大的 Python 包和环境管理器，支持 Windows、Linux 和 MacOS 等操作系统，提供 Scipy、Numpy、Matplotlib 等科学计算和作图包，其自带 Python 开发环境的管理器 conda，可以用来安装和更新主流 Python 包。Anaconda 安装成功后，可以启动 Anaconda Prompt 命令行窗口（Windows 用户）或终端（Linux 和 MacOS 用户）来使用 conda 创建和管理环境。本教程中将介绍如何使用 conda 创建和管理虚拟环境。

1. 首先，打开命令行窗口： Windows 用户可以输入 ``Anaconda Prompt``；MacOS 用户可以使用快捷键 ``command + space`` 并输入 ``Terminal``。
2. 在打开命令行窗口后，输入
```bash
conda create --name qnet_env python=3.8
```
来创建一个 Python 版本为 3.8 的环境 qnet_env。然后使用如下命令来激活这个环境，
```bash
conda activate qnet_env
```

关于 conda 更详细的介绍请参考[官方教程](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html)。

## 安装 QNET

QNET 需要 64-bit Python 3.8+ 的支持，兼容 Linux，MacOS (10.14+)以及 Windows 系统。我们推荐使用如下 ``pip`` 方式进行安装。在激活 conda 环境之后，输入
```bash
pip install qcompute-qnet
```
将会自动完成 QNET 的安装。对于已经安装过 QNET 的用户，可以在安装时加入 ``--upgrade`` 参数对 QNET 包进行更新。

## 运行测试

现在，我们可以尝试编写简单的例子来测试 QNET 是否安装成功。例如：


```python
from qcompute_qnet.core.des import DESEnv
from qcompute_qnet.topology import Network, Node, Link

# 创建模拟环境
env = DESEnv("Simulation Environment", default=True)
# 创建网络
network = Network("First Network")  
# 创建节点 Alice
alice = Node("Alice")  
# 创建节点 Bob
bob = Node("Bob")  
# 创建链路并连接节点 Alice 和 Bob
link = Link("Alice_Bob", ends=(alice, bob)) 
# 将节点和链路安装到网络中
network.install([alice, bob, link])  
# 初始化模拟环境
env.init()
# 模拟运行
env.run()
```

注意 API 文档的 Examples 栏目中我们提供了更丰富的代码演示案例，欢迎用户进行测试学习。
