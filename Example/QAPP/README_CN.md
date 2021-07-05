简体中文 | [English](README.md)

# QCompute-QAPP 使用指南

<em> 版权所有 (c) 2021 百度量子计算研究所，保留所有权利。</em> 

## QAPP 简介

QAPP 是基于[量易伏](https://quantum-hub.baidu.com/)组件 [QCompute](https://quantum-hub.baidu.com/opensource) 开发的量子计算解决方案工具集，提供包括量子化学、组合优化、机器学习在内的诸多领域问题的量子计算求解服务。QAPP 为用户提供了一站式的量子计算应用开发功能，直接对接用户在人工智能、金融科技、教育科研等方面的真实需求。

## QAPP 架构

QAPP 架构遵循从应用到真机的完整开发逻辑，包含 Application, Algorithm, Circuit, Optimizer 四个模块。其中 Application 模块将用户需求转换成相应的数学问题；Algorithm 模块选择合适的量子算法对该数学问题进行求解；在求解过程中，用户可以指定 Optimizer 模块中提供的优化器，也可自行设计自定义优化器；求解过程中所需的量子电路由 Circuit 模块支持，Circuit 模块直接调用 [QCompute](https://quantum-hub.baidu.com/opensource) 平台，支持对[量易伏](https://quantum-hub.baidu.com/services)模拟器/量子芯片的调用。

![exp](tutorials/figures/QAPPlandscape.png "QAPP 全景图")
<div style="text-align:center">QAPP 架构全景图</div>

## QAPP 案例入门

我们提供 QAPP 求解[分子基态能量](tutorials/VQE_CN.md)、求解[组合优化问题](tutorials/Max_Cut_CN.md)、以及求解[分类问题](tutorials/Kernel_Classifier_CN.md)的三个 QAPP 实用案例。这些用例旨在帮助用户快速上手 QAPP 各模块功能的调用以及自定义算法的开发。

## API 文档

我们提供 QAPP 的 [API](API_Documentation.pdf) 文档以供用户查阅。
