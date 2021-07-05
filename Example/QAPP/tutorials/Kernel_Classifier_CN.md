# 量子核方法

<em> 版权所有 (c) 2021 百度量子计算研究所，保留所有权利。 </em>

> 若使用云端算力运行本教程，将消耗约 500 Quantum-hub 点数。

## 概览

量子核方法是量子计算和经典机器学习中的核方法的结合，在近年来被提出之后便吸引了人们广泛的关注 [1-7]。简单地来说，利用量子计算机可以高效地计算两个量子态之间的内积的优点，量子核方法可以借助高维希尔伯特特征空间的特点在一些机器学习任务中取得优势 [6-7]。在本教程中，我们将首先对其理论背景进行简单的介绍，并展示如何利用 `QAPP` 中内置的 `KernelClassifier` 来完成真实数据的分类任务。


### 理论背景

在经典机器学习中，核方法一般指的是将低维的数据向量通过特征映射（feature map）映射到高维的特征空间（feature space）中来识别低维数据中难以分辨的模式的方法。如图1的例子所示，通过将一维的线性不可分数据映射到二维空间中，映射后的数据在二维空间中是线性可分的。

![feature map](figures/Qkernel-fig-featuremap.png "图1：核方法中的特征映射")
<div style="text-align:center">图1：核方法中的特征映射 </div>

不过，在实际应用中，由于特征空间的维数可能会十分巨大，我们往往并不希望直接对映射后的特征向量进行分析。相反，通过核方法中的另一个核心概念-核函数（kernel function），我们可以隐式地引入特征映射在模式识别上的优势。核函数的定义为数据向量在特征空间里的内积，其具体形式为

$$
K(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_j)^T \phi(\mathbf{x}_i),
\tag{1}
$$

其中 $\phi()$ 就代表着特征映射。需要注意的是，在核方法中我们并不需要显式地写出特征映射，而只需要定义核函数的形式即可。

在经典机器学习中，核方法最具代表性的应用就是支持向量机（support vector machine, SVM）。简单地来说，支持向量机解决的是线性分类问题：以一个二分类的问题举例，给定数据集为 $T = \{ (\mathbf{x}_1, y_1), ..., (\mathbf{x}_m, y_m) \} \subset \mathcal{X}\times\mathbb{Z}_2$，通过超平面 $(\mathbf{w}, b)$，支持向量机可以通过如下决策函数的正负来预测每个数据点 $\mathbf{x}$ 的标签：

$$
y_{\rm pred} = {\rm sign}(\langle \mathbf{w}, \mathbf{x} \rangle + b).
\tag{2}
$$

但是对于在原始数据空间中线性不可分的数据而言，这样的做法往往并不可行。所以如图1所示，通过引入我们上文中提到的特征映射，我们可以将原始数据空间中的数据向量映射到特征空间中来进行分类，从而得到更好的分类效果。此时，我们标记特征空间中的超平面为 $(\mathbf{w}', b')$, 此时决策函数就变成了：

$$
y_{\rm pred} = {\rm sign}(\langle \mathbf{w}', \phi(\mathbf{x}) \rangle + b').
\tag{3}
$$

更进一步的是，我们可以通过对偶化的方法，引入拉格朗日乘子 $\alpha_i$ 来表示此时的分割超平面 $\mathbf{w}' = \sum_i \alpha_i \phi(\mathbf{x_i})$ [8]。此时，我们可以在 $\alpha_i \geq 0$，$\sum_i y_i \alpha_i=0$ 的约束下，通过最大化

$$
\sum_i \alpha_i - \frac{1}{2} \sum_{i, j} \alpha_i \alpha_j y_i y_j \phi(\mathbf{x}_j)^T \phi(\mathbf{x}_i)
\tag{4}
$$

来计算最优参数 $\alpha_i^*$。不难发现，此时我们只需要计算数据向量在特征空间里的内积 $\phi(\mathbf{x}_j)^T \phi(\mathbf{x}_i) = K(x_i, x_j)$，正是我们上文中提到的核函数。换言之，在支持向量机中，我们不需要显式地知道特征映射的形式，而只需要计算原始数据在特征空间里的内积，就可以实现在特征空间中对数据进行分类。并且，对于任何新的数据向量 $\mathbf{x}'$，我们只需要通过核函数 $K(,)$ 计算

$$
y_{\rm pred} = {\rm sign}(\sum_i \alpha^*_i  \langle \phi(\mathbf{x_i}), \phi(\mathbf{x}' \rangle + b') = 
{\rm sign}(\sum_i \alpha^*_i  K(\mathbf{x}_i, \mathbf{x}') + b'),
\tag{5}
$$

就可以对数据的标签进行预测。

借助这种思想，我们就可以很简单的理解量子核方法的内涵。首先，可以认为一个编码电路 $U(\mathbf{x}) $ 实际上将经典数据向量 $\mathbf{x} $ 编码到某个希尔伯特空间中的一个量子态 $| \phi(\mathbf{x}) \rangle$ 上：

$$
U(\mathbf{x}) | 0^{\otimes N} \rangle = | \phi(\mathbf{x}) \rangle.
\tag{6}
$$

这种编码也往往被称为一种量子特征映射。关于编码电路的具体形式我们这里不做展开，感兴趣的读者可以阅读量桨中的[编码教程](https://qml.baidu.com/tutorials/machine-learning/encoding-classical-data-into-quantum-states.html)来了解不同的量子编码电路形式。此时我们的量子特征映射就是从经典数据空间到量子态所处的希尔伯特空间的一种特殊特征映射。在这个基础上，我们将量子核函数（quantum kernel function）定义为经典数据向量在量子特征空间的内积，其具体形式就为

$$
K^Q_{ij} = |\langle \phi(\mathbf{x}_j) | \phi(\mathbf{x}_i) \rangle |^2,
\tag{7}
$$

上式可以进一步写为

$$
|\langle \phi(\mathbf{x}_j) | \phi(\mathbf{x}_i) \rangle |^2 =  |\langle 0^{\otimes N} | U^\dagger(\mathbf{x}_j) U(\mathbf{x}_i) | 0^{\otimes N} \rangle |^2.
\tag{8}
$$

不难发现，通过运行如图2中所示的量子电路，并在统计其测量结果为 $| 0^{\otimes N} \rangle $ 的概率，我们就可以估计式（8）中的量子核函数。这种方法也被称为量子核估计（quantum kernel estimation, QKE）方法。也就是说，如果我们通过量子核估计方法来计算式（4-5) 中的核函数的话，我们就可以利用支持向量机的方法在量子特征空间来完成数据分类。借助量子特征映射的量子性，人们希望这种量子核方法可以更好地分类具有复杂模式的数据。人们已经证明，通过精心设计量子特征映射，量子核方法就可以用来分辨任何经典方法都无法识别的数据模式 [7]。

![QKE](figures/Qkernel-fig-QKE.png "图2：量子核估计电路")
<div style="text-align:center">图2：量子核估计电路 </div>

![illustration](figures/Qkernel-fig-illustrationCN.png "图3：经典核方法和量子核方法的对比示意图")
<div style="text-align:center">图3：经典核方法和量子核方法的对比示意图 </div>

## 量子核方法分类手写数字

下面，以手写数字（MNIST）数据的二分类问题为例，我们将展示如何使用 QCompute 和 QAPP 中的相关模块完成实际的数据分类任务。

![MNIST](figures/MNISt.png "图1：核方法中的特征映射")
<div style="text-align:center"> 手写数字示例 </div>


首先，加载 QAPP、QCompute、skimage 以及相关的模块（请确保上述模块已经正确安装到当前环境）。

```python
import numpy as np
import time
import json
from skimage.transform import resize
from qapp.algorithm.kernel_classifier import KernelClassifier
from QCompute import Define
from QCompute.QPlatform import BackendName
```

我们利用内置的分类器 `KernelClassifier` 来进行数据分类。在 `KernelClassifier` 中，我们实现了一个上文中提到的量子核分类器，并且提供了易用的接口。并且，我们预设的后端是 `LocalBaiduSim2` ，读者可以在量易伏的网站上找到更多的后端。特别地，我们也提供真实量子设备的接口，读者可以通过 `CloudIoPCAS` 调用。

```python
backend = BackendName.LocalBaiduSim2
# 调用真机或者云服务器需要从量易伏网站个人中心里获取 token
# from QCompute import Define
# Define.hubToken = 'your token'
# backend = BackendName.CloudIoPCAS # 调用真机
# backend = BackendName.CloudBaiduSim2Water # 调用云服务器 CloudBaiduSim2Water
myKernelClassifier = KernelClassifier(backend=backend)
```

我们在本地准备了一些 MNIST 手写数字数据，读者可以直接载入该数据。

```python
print('Loading local MNIST data...')
fp = open('./data/MNIST_data.json', 'rt+')
mnist = json.load(fp)
fp.close()
```

原始 MNIST 的数据集中的图像像素为 28 x 28=784，为了将这些经典数据编码到有限数量的量子比特上，我们要先对图像进行一些预处理。通过 `get_data()` 函数，我们可以在原始 MNIST 数据集的基础上生成降维后的二分类问题数据。

```python
# 根据 MNIST 数据来生成一个降维后的二分类问题
def get_data(first_num=1,
             second_num=8,
             data_size=10,
             scale=2):
    # 选取两个标签所对应的数据并合并为一个新的数据集
    data_first_num = mnist['data'][mnist['target'] == str(first_num)]
    data_second_num = mnist['data'][mnist['target'] == str(second_num)]
    data_unsampled = np.vstack([data_first_num, data_second_num])
    label_unsampled = np.hstack([np.zeros(len(data_first_num)), np.ones(len(data_second_num))])

    # 从新的数据集中随机抽样 data_size 个样本作为新的数据集
    idx = np.random.randint(low=0, high=len(label_unsampled), size=data_size)
    data_uncompressed = data_unsampled[idx]
    label = label_unsampled[idx]

    # 将图像降采样为 scale x scale 个像素
    data_unnormalized = [resize(image.reshape([28, 28]), [scale, scale]).flatten() for image in data_uncompressed]

    # 重新归一化
    data = [(image / image.max()) * np.pi for image in data_unnormalized]

    return np.array(data), label
```

接下来，我们分别生成训练集和验证集。

```python
data_train, label_train = get_data(data_size=18, scale=3)
data_test, label_test = get_data(data_size=10, scale=3)
```

利用 `KernelClassifier.fit()` 以及 `KernelClassifier.predict()`，我们可以训练量子核分类器并进行预测。

```python
# 利用训练数据来训练分类器
myKernelClassifier.fit(data_train, label_train)

# 预测验证集数据的标签
predict_svm_qke_test = myKernelClassifier.predict(data_test)
```

最后，计算在验证集上的错误率。

```python
error_rate = sum(abs(predict_svm_qke_test - label_test)) / len(predict_svm_qke_test)
print('验证错误率为:', error_rate)
```
```
验证错误率为: 0.1
```
至此，我们就完成了利用量子核方法来分类手写数字的任务。在这个例子中，我们将 28 x 28 维的原始数据降采样至了 3 x 3 个像素，而量子核分类器依然可以得到很好的分类效果。我们有理由相信，随着真实量子设备上的量子比特数目的进一步增加，该分类器的能力也会进一步的增加。此外，我们也鼓励读者尝试更加多样的数据、以及更多的数据降维方法，并观察量子核分类器在这些数据上的表现。

---

## 参考文献

[1] Schuld, Maria. "Supervised quantum machine learning models are kernel methods." arXiv preprint [arXiv:2101.11020 (2021)](https://arxiv.org/abs/2101.11020).

[2] Havlíček, Vojtěch, et al. "Supervised learning with quantum-enhanced feature spaces." [Nature 567.7747 (2019): 209-212](https://arxiv.org/abs/1804.11326).

[3] Huang, Hsin-Yuan, et al. "Power of data in quantum machine learning." arXiv preprint [arXiv:2011.01938 (2020)](https://arxiv.org/abs/2011.01938).

[4] Schuld, Maria, and Nathan Killoran. "Quantum machine learning in feature Hilbert spaces." [Phys. Rev. Lett. 122.4 (2019): 040504](https://arxiv.org/abs/1803.07128).

[5] Hubregtsen, Thomas, et al. "Training Quantum Embedding Kernels on Near-Term Quantum Computers." arXiv preprint [arXiv:2105.02276(2021)](https://arxiv.org/abs/2105.02276).

[6] Glick, Jennifer R., et al. "Covariant quantum kernels for data with group structure." arXiv preprint [arXiv:2105.03406(2021)](https://arxiv.org/abs/2105.03406).

[7] Liu, Yunchao, Srinivasan Arunachalam, and Kristan Temme. "A rigorous and robust quantum speed-up in supervised machine learning." arXiv preprint [arXiv:2010.02174 (2020)](https://arxiv.org/abs/2010.02174).

[8] Schölkopf, Bernhard, and Alexander J. Smola"Learning with kernels: support vector machines, regularization, optimization, and beyond." [MIT Press(2002)](https://mitpress.mit.edu/books/learning-kernels).
