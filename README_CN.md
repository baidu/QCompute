简体中文 | [English](README.md)

# 量易伏 - QComputeSDK

- [特色](#特色)
- [安装步骤](#安装步骤)
   - [环境设置](#环境设置)
   - [安装 QComputeSDK](#安装-QComputeSDK)
   - [运行](#运行)
   - [重大更新](#重大更新)
- [入门与开发](#入门与开发)
   - [案例入门](#案例入门)
   - [API 文档](#API-文档)
   - [开发](#开发)
- [交流与反馈](#交流与反馈)
- [使用 QComputeSDK 的工作](#使用-qcomputesdk-的工作)
- [FAQ](#faq)
- [Copyright and License](#copyright-and-license)


[Quantum Leaf (量易伏)](https://quantum-hub.baidu.com/) 是[百度量子计算研究所](https://quantum.baidu.com/)旗下全球首个云原生量子计算平台。用户可以使用量易伏进行量子编程，量子模拟和运行真实量子计算机。量易伏旨在为量子基础设施即服务 (Quantum infrastructure as a Service, QaaS) 提供量子基础开发环境。

![](https://release-data.cdn.bcebos.com/github-qleaf%2F%E9%87%8F%E6%98%93%E4%BC%8F%E5%9B%BE%E6%A0%87.png)

[![](https://img.shields.io/badge/license-Apache%202.0-green)](./LICENSE) ![](https://img.shields.io/badge/build-passing-green) ![](https://img.shields.io/badge/Python-3.9--3.11-blue) ![](https://img.shields.io/badge/release-v3.3.0-blue) 

![](https://img.shields.io/badge/OS-MacOS%20|%20Windows%20|%20Linux-lightgrey.svg?style=flat-square)

本安装包是 QComputeSDK 的 Python 语言实现的全量量子开源计算框架。它采用经典量子混合的编程模式并预制多种先进模块，用户不仅可以在量子环境对象 (QEnv) 下快速搭建电路，也可以将它用于各类复杂量子算法的研发。QComputeSDK 内置多类本地高性能模拟器和云端模拟器/真机调用接口，用户可以将电路在本地模拟器快速模拟验证也可以将电路任务进一步提交至云端真实量子硬件（超导、离子阱）以及高性能模拟器执行。

## 特色

- 轻松上手
   - 近 50 篇教程案例，还在不断地增加
   - 量子电路本地可视化
   - 全自动调用相关计算模块，完成预订流程
- 功能丰富
   - 支持电路嵌套的量子子程序功能
   - 本地高性能模拟器支持 32 量子比特的模拟运算
   - 云端高性能异构模拟器支持更大规模量子模拟
   - 支持多种噪声模型的模拟
   - 基于英伟达 cuQuantum 的本地 GPU 模拟器
   - 基于 Gaussian/Fork 态的本地光量子模拟器
- 真实量子算力
   - 接入百度自研超导量子计算机 QPUQian
   - 接入中科院精密测量院离子阱量子计算机 IonAPM 
   - 接入中科院物理所超导量子计算机 IoPCAS

## 安装步骤

### 环境设置

推荐使用 Anaconda 创建虚拟环境，
```bash
conda create -n qcompute_env python=3.10
conda activate qcompute_env
```
> Anaconda 请从[官网下载](https://www.anaconda.com/download#downloads)

> 注意： 无论使用 Anaconda 还是原生 Python ，Python 版本都应 >= 3.9 



### 安装 QComputeSDK

通过 `pip` 完成安装，

```bash
pip install qcompute
```

用户也可以选择下载全部文件后进行本地安装。我们推荐此种方式安装以及二次 SDK 开发，可以方便的形成本地开发闭环，更方便调试等动作。

```bash
git clone https://github.com/baidu/QCompute.git
cd QCompute
pip install -e .
```

### 运行
如果用户选择下载全部文件，现在可以试着运行一段程序来验证是否安装成功。这里我们运行 QComputeSDK 提供的测试脚本，

```bash
python -m Test.PostInstall.PostInstall_test
```
该脚本中包括执行本地与云端任务测试，云端测试前需要在命令行输入用户 Token ，Token 可登陆[量易伏官网](https://quantum-hub.baidu.com/token)查看。如不需要做云端测试可运行 `Ctrl+c` 结束。

> 注意：通过 pip 安装请跳过此步。

### 重大更新

从 QComputeSDK 3.0.0 版本开始，开发者可以通过 QComputeSDK 运行百度自研超导量子计算机乾始。设备定期对外提供服务，可以从[量易伏真机详情页](https://quantum-hub.baidu.com/services)查看真机开放时间。该设备在 QComputeSDK 中的后端名为`CloudBaiduQPUQian`.

## 入门与开发

### 案例入门

QComputeSDK 是一个实现后台接入真实量子硬件的量子计算开发框架。建立起了量子计算与量子硬件的桥梁，为量子算法和应用的研发落地提供强有力的支撑，也提供了丰富的案例供开发者学习。

在这里，我们提供了初级、中级、高级案例供大家学习。初级案例中展示了使用 QComputeSDK 可以快速上手的简单示例，包括量子态制备、经典量子混合编程、以及将电路任务提交到量子计算机上执行等。中级案例中是 QComputeSDK 的进阶用法，包括模块的使用、内置转换器的使用等。高级案例中则是进阶量子算法在 QComputeSDK 上的实现示例，我们为这些算法都配套了详细的教程文档。建议用户下载 QComputeSDK 全部文件安装，本地运行进行实践。

- [初级案例](./Example/Level_1)
  1. [GHZ 态制备（本地）](./Example/Level_1/GHZ_Local.py)
  2. [GHZ 态制备（云端）](./Example/Level_1/GHZ_Cloud.py)
  3. [贝尔态制备（本地）](./Example/Level_1/HCNOT_Local.py)
  4. [贝尔态制备（云端）](./Example/Level_1/HCNOT_Cloud.py)
  5. [经典量子混合语言示例（本地）](./Example/Level_1/Hybrid_Local.py)
  6. [经典量子混合语言示例（云端）](./Example/Level_1/Hybrid_Cloud.py)
  7. [经典量子信息交互示例（本地）](./Example/Level_1/Interact_Local.py)
  8. [经典量子信息交互示例（云端）](./Example/Level_1/Interact_Cloud.py)
  9. [百度量子自研超导真机运行示例](./Example/Level_1/QPUCase_BaiduQPUQian.py)
  10. [中科院精密测量院离子阱运行示例](./Example/Level_1/QPUCase_IonAPM.py)
  11. [中科院物理所超导真机运行示例](./Example/Level_1/QPUCase_IoPCAS.py)
  12. [基于 Fock 态的光量子线路模拟](./Example/Level_1/PhotonicFookCase_local.py)
  13. [基于 Gaussian 态的光量子线路模拟](./Example/Level_1/PhotonicGaussianCase_local.py)
  14. [基于 cuQuantum 的 GPU 模拟器](./Example/Level_1/SimulatorCase_CuQuantum_Local.py)
  15. [盲量子计算示例](./Example/Level_1/Ubqc.py)
  - [量子噪声模拟](./Example/Level_1/Noise)
    1. [对电路添加噪声示例](./Example/Level_1/Noise/AddNoise.py)
    2. [量子噪声压缩模块示例](./Example/Level_1/Noise/CompressNoiseTest.py)
    3. [一量子位电路含噪模拟](./Example/Level_1/Noise/OneQubitNoiseTest.py)
    4. [多进程并行的噪声模拟](./Example/Level_1/Noise/ParallelNoiseSimulationTest.py)
    5. [两量子位电路含噪模拟](./Example/Level_1/Noise/TwoQubitNoiseTest.py)

- [中级案例](./Example/Level_2)
  - [输出信息设置](./Example/Level_2/0_OutputFormatControl)
    1. [教程](./Example/Level_2/0_OutputFormatControl/Turorials/OutputFormatControl_ZH.md)
    2. [结果打印信息设置示例](./tutorials/machine_learning/QClassifier_CN.ipynb)
    3. [输出文件自动清理示例](./tutorials/machine_learning/VSQL_CN.ipynb)
  - [通用模块](./Example/Level_2/1_OpenModules)
    1. [教程](./Example/Level_2/1_OpenModules/Tutorials/OpenModules_ZH.md)
    2. [模块使用示例](./Example/Level_2/1_OpenModules/0_OpenModules.py)
    3. [量子电路逆操作模块](./Example/Level_2/1_OpenModules/1_InverseCircuitModule.py)
    4. [量子电路反操作模块](./Example/Level_2/1_OpenModules/2_ReverseCircuitModule.py)
    5. [子程序展开模块](./Example/Level_2/1_OpenModules/3_UnrollProcedureModule.py)
    6. [量子门分解模块](./Example/Level_2/1_OpenModules/4_UnrollCircuitModule.py)
    7. [量子门压缩模块](./Example/Level_2/1_OpenModules/5_CompressGateModule.py)
  - [转换器](./Example/Level_2/2_OpenConvertors)
    1. [教程](./Example/Level_2/2_OpenConvertors/Tutorials/OpenConvertors_ZH.md)
    2. [电路序列化示例](./Example/Level_2/2_OpenConvertors/0_Circuit.py)
    3. [终端绘图示例](./Example/Level_2/2_OpenConvertors/1_DrawConsole.py)
    4. [电路序列和反序列示例](./Example/Level_2/2_OpenConvertors/2_InternalStruct.py)
    5. [电路转 JSON 示例](./Example/Level_2/2_OpenConvertors/3_JSON.py)
    6. [电路转 QASM 示例](./Example/Level_2/2_OpenConvertors/4_QASM.py)
    7. [电路转 QOBJ 示例](./Example/Level_2/2_OpenConvertors/5_Qobj.py)
    8. [电路转 IonQ 示例](./Example/Level_2/2_OpenConvertors/6_IonQ.py)
    9. [电路转 Xanadu 示例](./Example/Level_2/2_OpenConvertors/7_XanaduSF.py)

- [高级案例](./Example/Level_3)
  1. [量子超密编码](./Example/Level_3/0_SuperdenseCoding/Tutorial-Superdense/Superdense_CN.md)
  2. [Deutsch-Jozsa 算法](./Example/Level_3/1_Deutsch-Jozsa/Tutorial-DJ/Deutsch-Jozsa_CN.md)
  3. [量子相位估计 (QPE)](./Example/Level_3/2_PhaseEstimation/Tutorial-phase/Phase_CN.md)
  4. [格罗弗搜索算法](./Example/Level_3/3_Grover/Tutorial-Grover/Grover_CN.md)
  5. [Shor 算法](./Example/Level_3/4_ShorAlgorithm/tutorial/Shor_CN.md)
  6. [变分量子基态求解器 (VQE)](./Example/Level_3/5_VQE/Tutorial-VQE/VQE_CN.md)
  7. [变分量子态对角化 (VQSD)](./Example/Level_3/6_VQSD/Tutorial-VQSD/VQSD_CN.md)

在最近的更新中，QComputeSDK 加入了本地光量子计算模拟器 (LocalBaiduSimPhotonic) 。与传统的量子电路模型不同，光量子计算具有其独特的运行方式。QComputeSDK 在架构上支撑起光学体系，也成为了首个集成通用量子计算与光量子计算双体系的量子开发套件。感兴趣的读者请参见[光量子计算模拟器教程](https://quantum-hub.baidu.com/pqs/tutorial-introduction)。

### API 文档

了解更多 QComputeSDK 使用方法，请参考 [API 文档](https://quantum-hub.baidu.com/docs/qcompute/latest/)，包含了供用户使用的所有函数和类的详细说明与用法。

### 开发
QComputeSDK 中包括量子计算架构、量子模拟器、量子案例以及扩展功能等。对于需要涉及架构或模拟器源码的开发者，建议下载全部文件并本地安装调试。对于使用 QComputeSDK 研发算法应用的开发者或科研工作者，建议以 [GHZ_Cloud.py](./Example/Level_1/GHZ_Cloud.py) 作为代码框架，修改和使用这个文件可以有效帮助熟悉本量子开发套件的语法。建议开发者熟悉 QComputeSDK 的电路模型构造，注意量子位输出顺序为高位。 

## 交流与反馈

- 我们非常欢迎您提交问题、报告与建议，您可以通过以下渠道反馈
  - [GitHub Issues](https://github.com/baidu/QCompute/issues) / [Gitee Issues](https://gitee.com/baidu/qcompute/issues)
  - [量易伏官网-意见反馈](https://quantum-hub.baidu.com/feedback)
  - 量易伏官方邮箱 quantum@baidu.com
- 技术交流 QQ 群：1147781135，欢迎扫码进群

![](https://release-data.cdn.bcebos.com/github-qleaf%2Fqrcode.png)

## 使用 QComputeSDK 的工作

我们非常欢迎开发者使用 QComputeSDK 进行量子应用研发，如果您的工作有使用 QComputeSDK，也非常欢迎联系我们。以下为基于 QComputeSDK 开发的量子应用：
- [量噪 (QEP, Quantum Error Processing)](https://quantum-hub.baidu.com/qep/tutorial-overview)，百度量子计算研究所研发的量子噪声处理工具集，主要功能包括量子性能评估、量子噪声刻画、量子噪声缓释和量子纠错。
- [盲量子计算 (UBQC, Universal Blind Quantum Computation)](https://quantum-hub.baidu.com/bqc/tutorial-bqc)，百度量子计算研究所研发的基于 UBQC 协议的盲计算代理服务。
- [QAPP](https://quantum-hub.baidu.com/qapp/tutorial-overview) 是基于 QComputeSDK 开发的量子计算解决方案工具集，提供包括量子化学、组合优化、机器学习在内的诸多领域问题的量子计算求解服务。
- [量子奇异值变换 (QSVT, Quantum Singular Value Transformation)](https://quantum-hub.baidu.com/qsvt/tutorial-overview)，百度量子计算研究所研发的量子奇异值变换工具集，主要功能包括量子奇异值变换，对称量子信号处理，以及哈密顿量模拟。
- [量子金融 QFinance](https://quantum-hub.baidu.com/qfinance/tutorial-option-pricing) ，百度量子计算研究所研发的量子金融库，提供用于期权定价的量子蒙特卡罗方法。
- [光量子计算模拟器 (PQS, Photonic Quantum Simulator)](https://quantum-hub.baidu.com/pqs/tutorial-introduction)，百度研究院量子计算研究所研发的光量子计算模拟器，支持基于 Gaussian 态和 Fock 态的光量子线路模拟。

## FAQ
1. 问：**使用 QComputeSDK 可以做什么？它有哪些应用场景？**

    答：QComputeSDK 是一个基于 Python 的量子计算开发框架，可以用于构建、运行和优化量子算法。我们在 QComputeSDK 建设了全面且完善的基础设施用于支持各类量子算法的实现，因此在量子应用的研发上它具有广泛的应用场景。具体工作可以参考但不限于 QComputeSDK 中的[扩展功能](./Extensions)。

2. 问：**想用 QComputeSDK 做量子编程，但对量子计算不是很了解，该如何入门？**

    答：Nielsen 和 Chuang 所著的《量子计算与量子信息》是量子计算领域公认的经典入门教材。建议读者首先学习这本书的第一、二、四章，介绍了量子计算中的基本概念、数学和物理基础、以及量子电路模型。读者也可以在[量易简](https://qulearn.baidu.com/)上学习，这是一个在线量子学习知识库，不仅包含量子计算教程，还有丰富的视频课程。读者还可以下载[量易伏APP](https://quantum-hub.baidu.com/qmobile)，APP上的量子小调包含丰富有趣的量子样例，帮助读者随时随地的学习。

3. 问：**QComputeSDK 是否免费？**

    答：QComputeSDK 是免费的。QComputeSDK 是开源 SDK 并携带多类本地模拟器，用户执行本地模拟任务是免费的。当用户通过 QComputeSDK 将任务提交给云端模拟器或真机运行时，会扣除一定点数。详细的扣点规则可以参考[用户指南](https://quantum-hub.baidu.com/quickGuide/account)。用户在创建账户时我们会赠送点数，点数余额可以在[个人中心](https://quantum-hub.baidu.com/profile)查看。

4. 问：**点数不足怎么办？**

    答：点数目前仅用于资源控制。点数不足时可以从量易伏官网的[意见反馈](https://quantum-hub.baidu.com/feedback)或[量易伏APP](https://quantum-hub.baidu.com/qmobile)的用户反馈提交点数申请。我们会在三个工作日内处理您的请求。


## Copyright and License

QComputeSDK 使用 [Apache-2.0 license](./LICENSE) 许可证。

## 作者
- 刘树森
- 贺旸
- 江云帆
- 张文学
- 孙文赟
- 付永凡
- 陈建萧
- 沈豪杰
- 吕申进
- 王友琪
