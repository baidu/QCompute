# 安装说明

*版权所有 (c) 2022 百度量子计算研究所，保留所有权利。*

## 安装 Python 环境

我们建议使用 Anaconda 作为开发环境。Anaconda 是一个强大的 Python 包和环境管理器，支持 Windows、Linux 和 MacOS 等操作系统，提供 Scipy、Numpy、Matplotlib 等科学计算和作图包，其自带 Python 开发环境的管理器 conda，可以用来安装和更新主流 Python 包。Anaconda 安装成功后，可以启动 Anaconda Prompt 命令行窗口（Windows 用户）或终端（Linux 和 MacOS 用户）来使用 conda 创建和管理环境。本教程中将介绍如何使用 conda 创建和管理虚拟环境。

1. 首先，打开命令行窗口： Windows 用户可以输入 ``Anaconda Prompt``；MacOS 用户可以使用快捷键 ``command + space`` 并输入 ``Terminal``。

2. 在打开命令行窗口后，输入

    ```bash
    conda create --name qsvt_env python=3.9
    ```

    来创建一个 Python 版本为 3.9 的环境 ``qsvt_env``。然后使用如下命令来激活这个环境，

    ```bash
    conda activate qsvt_env
    ```
关于 conda 更详细的介绍请参考其[官方教程](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html)。

## 安装量子奇异值变换

量子奇异值变换需要 64-bit Python 3.9 的支持，兼容 Linux，MacOS (10.14+)以及 Windows 系统。我们推荐使用如下 `pip` 方式进行安装。在激活 conda 环境之后，输入

```bash
pip install qcompute-qsvt
```

将会自动完成量子奇异值变换的安装。对于已经安装过量子奇异值变换的用户，可以在安装时加入 `--upgrade` 参数对量子奇异值变换进行更新。

## 运行测试

现在，我们可以尝试编写简单的例子来测试量子奇异值变换是否安装成功。例如：

```python
import numpy as np
from Extensions.QuantumSingularValueTransformation.qcompute_qsvt.Application.HamiltonianSimulation.HamiltonianSimulation import func_HS_QSVT

print(func_HS_QSVT(list_str_Pauli_rep=[(1, 'X0X1'), (1, 'X0Z1'), (1, 'Z0X1'), (1, 'Z0Z1')], 
                   num_qubit_sys=2, float_tau=-np.pi / 8, float_epsilon=1e-6, circ_output=False)['counts'])
```

这将在初态 $|00\rangle$ 上作用哈密顿量 $X\otimes X + X\otimes Z + Z\otimes X + Z\otimes Z$ 在时间 $-\pi/8$ 处的时间演化算符，精度为 `1e-6`，然后测量末量子态，并打印测量结果的布居数。

注意代码源文件的 [examples](https://github.com/baidu/QCompute/tree/master/Extensions/QuantumSingularValueTransformation/examples) 中提供了更丰富的代码演示案例，欢迎用户进行测试学习。