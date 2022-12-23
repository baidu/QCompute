# 简介

*版权所有 (c) 2022 百度量子计算研究所，保留所有权利。*

## 关于量子奇异值变换

**量子奇异值变换**是由[百度量子计算研究所](https://quantum.baidu.com)研发的量子奇异值变换（**Q**uantum **S**ingular **V**alue **T**ransformation）工具集，旨在量子计算机或量子计算模拟器上更便捷地实现量子模拟等算法。目前，该工具集提供了量子奇异值变换，对称量子信号处理，以及它们在哈密顿量模拟中的应用等三个主要模块：

+ **量子奇异值变换** 用于实现量子操作的奇异值变换，输入输出均为以块编码形式实现的量子操作或量子电路。

+ **对称量子信号处理** 可用于编码量子奇异值变换的变换函数，以补全量子奇异值变换的量子电路。引入对称量子信号处理可以更加高效地完成编码步骤。

+ **哈密顿量模拟** 为量子奇异值变换在量子模拟，乃至量子计算领域中最重要的应用。该模块提供了相应的函数，用于生成哈密顿量时间演化算符的量子电路。

## 教程

量子奇异值变换提供了使用哈密顿量模拟模块的[快速上手](https://quantum-hub.baidu.com/qsvt/tutorial-quickstart)，以及[理论简述](https://quantum-hub.baidu.com/qsvt/tutorial-introduction)供用户学习入门。目前教程的内容安排如下：

- [理论简述](https://quantum-hub.baidu.com/qsvt/tutorial-introduction)
- [量子信号处理](https://quantum-hub.baidu.com/qsvt/tutorial-qsp)
- [块编码与酉线性组合](https://quantum-hub.baidu.com/qsvt/tutorial-be)
- [量子特征值与奇异值变换](https://quantum-hub.baidu.com/qsvt/tutorial-qet)
- [哈密顿量模拟](https://quantum-hub.baidu.com/qsvt/tutorial-hs)

建议初学者按照顺序阅读学习。我们将会在之后的版本更新中提供更细节、更详尽的教程。

## 常见问题

**Q: 使用量子奇异值变换需要安装哪些依赖项？**

**A:** 量子奇异值变换是基于量易伏 [QCompute](https://quantum-hub.baidu.com/opensource) 开源 SDK 开发，所以在使用量子奇异值变换之前需要预先安装 QCompute。当用户安装量子奇异值变换时会自动下载安装这个关键依赖包。

**Q: 我的点数用完了该怎么办？**

**A:** 请通过 [Quantum Hub](https://quantum-hub.baidu.com/) 联系我们。首先，登录 [Quantum Hub](https://quantum-hub.baidu.com/)，然后进入“意见反馈”页面，点击“获取点数”，然后输入必要的信息。提交您的反馈并等待回复，我们将会第一时间联系您。

**Q: 我应该如何在研究工作中引用量子奇异值变换？**

**A:** 我们热诚欢迎科研工作者使用量子奇异值变换进行量子算法相关问题的研发与探索。如果您的工作使用了量子奇异值变换，请通过如下 BibTeX 进行引用：

```BibTex
@misc{QSVT,
      title = {{Quantum Sigular Value Transformation toolkit in Baidu Quantum Platform}},
      year = {2022},
      url = {https://quantum-hub.baidu.com/qsvt/}
}
```

## 用户反馈
我们热诚欢迎用户通过邮件方式对工具集的各种相关问题，未解决的 bug 和可能的改善方向进行反馈。相关问题请邮件至 [quantum@baidu.com](mailto:quantum@baidu.com)

## 版权和许可证

量子奇异值变换使用 [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) 作为许可证。