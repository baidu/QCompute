#!/usr/bin/python3
# -*- coding: utf8 -*-

# Copyright (c) 2022 Baidu, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
这是一个使用量子求阶算法计算 2 模 63 的阶的简单例子。
"""


from QCompute import *
from numpy import pi

matchSdkVersion('Python 3.0.0')


def func_order_finding_2_mod_63():
    """
    这个函数将会返回特征相位因子 s/6 的逼近，其中 6 是 2 模 63 的阶，s=0,1,...,5
    """
    env = QEnv()  # 创建环境
    env.backend(BackendName.LocalBaiduSim2)  # 选择后端 LocalBaiduSim2

    L = 6  # 编码 U 门需要的量子比特数、工作寄存器的量子比特数
    N = 3 * L + 1  # 算法需要使用的总量子比特数
    t = 2 * L + 1  # 相位估计算法所需的辅助比特数、辅助寄存器的量子比特数

    q = env.Q.createList(N)  # 生成量子比特的列表

    X(q[N - 1])  # 在工作寄存器制备初始状态 |1>，下面开始执行相位估计算法

    for i in range(t):
        H(q[i])  # 相位估计的第一步，对辅助寄存器取平均叠加完毕，下面开始将相位估计算法的转存步骤

    CSWAP(q[2 * L], q[t + 4], q[t + 5])
    CSWAP(q[2 * L], q[t + 3], q[t + 4])
    CSWAP(q[2 * L], q[t + 2], q[t + 3])
    CSWAP(q[2 * L], q[t + 1], q[t + 2])
    CSWAP(q[2 * L], q[t + 0], q[t + 1])
    # 上一注释至此实现了门 C(U)，辅助寄存器的最后一个量子比特控制工作寄存器

    s = 2 * L - 1  # 准备依次作用其他 C(U^(2^j)) 门，q[s] 即门的控制量子位
    while s >= 0:
        if s % 2 == 1:
            CSWAP(q[s], q[t + 1], q[t + 3])
            CSWAP(q[s], q[t + 3], q[t + 5])
            CSWAP(q[s], q[t + 0], q[t + 2])
            CSWAP(q[s], q[t + 2], q[t + 4])  # 本条件下的门组合后即门 C(U^2)
        else:
            CSWAP(q[s], q[t + 3], q[t + 5])
            CSWAP(q[s], q[t + 1], q[t + 3])
            CSWAP(q[s], q[t + 2], q[t + 4])
            CSWAP(q[s], q[t + 0], q[t + 2])  # 本条件下的门组合后即门 C(U^4)
        s -= 1  # 指针移动到更高位的辅助比特上去

    # 逆 Fourier 变换中的 SWAP 步骤
    for i in range(t // 2):
        SWAP(q[i], q[t - i - 1])

    # 逆 Fourier 变换中的控制旋转步骤
    for i in range(t - 1):
        H(q[t - i - 1])
        for j in range(i + 1):
            CU(0, 0, -pi / pow(2, (i - j + 1)))(q[t - j - 1], q[t - i - 2])
    H(q[0])

    # 完成逆 Fourier 变换，完成相位估计算法，等待测量
    MeasureZ(q[:t], range(t))  # 只测量辅助寄存器即前 t 个比特
    env.commit(8192, downloadResult=False)


if __name__ == "__main__":
    func_order_finding_2_mod_63()
