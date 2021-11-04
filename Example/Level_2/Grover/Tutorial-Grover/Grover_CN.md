# Grover 搜索算法

假设一个推销员想推销一种新出的量子计算机，他在世界各大城市间来回穿梭（纽约 $\rightarrow$ 上海 $\rightarrow\cdots$）并且最终回到位于起点的公司总部，他需要提前规划好一条可以经过所有大城市的最短路线以节省时间。这就是著名的[旅行推销员问题](https://en.wikipedia.org/wiki/Travelling_salesman_problem)，在本质上，它是一个**非结构化**的搜寻问题。利用数学的严格定义，大概就是如下
>在一个无序的搜寻问题中, 给定一个具有 $N$ 个元素的集合 $S = \{|0\rangle, |1\rangle, \cdots, |N-1\rangle \}$ 给定一个布尔函数 $f : S \rightarrow \{0, 1\}$, 目标是在 $S$ 中找到 **唯一一个** 元素 $|x^*\rangle$，使得$f(|x^∗\rangle) = 1$， 而剩下的元素满足 $f(|x\rangle) = 0$。

当我们提到"无序"，我们是指任何先验的关于这个数据集的知识都不存在，包括元素是如何排序 (大小、首字母排序等等) 都不清楚。最好的经典的方法也只能一个一个试，这个办法需要消耗 $\mathcal{O}(N)$ 搜寻步数，最坏的情况是需要搜寻完全部的元素！如果我们运气很差想要找的元素恰巧排到集合的最后，就需要搜索完整个集合。

有趣的是，在 1996 年, 计算机科学家 *Lov Grover* 发明了一种奇特的量子算法，它可以仅仅利用 $\mathcal{O}{(\sqrt{N})}$的搜索步骤找到无序集合中的那个我们想找到的元素，这是一个多项式级的加速 (而非量子运算中常见的指数加速，比如量子傅立叶变换)。尽管如此，假设你的集合有 100 万个元素，那么用 Grover 搜索算法只需要搜寻几千次左右。这也是很大的进步了！

---
## 算法部分

首先来看一个简单的例子，假设有一个集合 $S = \{ |0\rangle,\cdots, |6\rangle\}$， 而想要找到的那个元素靠近最末尾 $|x^*\rangle = |5\rangle$。

- **预处理**
  
  为了能让量子计算机正常工作，我们首先要将搜索空间扩大到2的指数，在讨论的例子中便是 $2^3=8$ 个元素。那么我们需要把集合拓展到 $S = \{ |0\rangle,\cdots, |7\rangle\}$（我们没必要关心 $|7\rangle$ 到底是什么，只要我们的量子黑箱能知道它不是正确解就行了）。在这个具体的问题中，我们的量子计算机需要3个量子比特来编码搜索空间里的元素，并且需要转化为相应的二进制数组 $S = \{ |000\rangle,\cdots, |111\rangle\}$。我们可以在 Quantum Leaf 上轻松的初始化输入态

  ```python
  # In this example we use 3 qubits in total
  qubit_num = 3
  # and set the shot number for each request
  shots = 1000
  
  
  def main():
      """
      main
      """
      # Create environment
      env = QEnv()
      # Choose backend Baidu Local Quantum Simulator-Sim2
      env.backend(BackendName.LocalBaiduSim2)
  
      # Initialize the three-qubit circuit
      q = env.Q.createList(qubit_num)
  ```

- **利用阿达玛门 $H$ 创造出量子叠加态**
  
  然后我们做的一步是，对电路中的每一个量子比特（它们的初始值都是 $|0\rangle$）施加阿达玛门来创造出叠加态
  $$
  |\psi\rangle = (H|0\rangle)^{\otimes n}= \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)\otimes \cdots \otimes \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)= \frac{1}{\sqrt{N}}\sum_{x=0}^{N-1} |x\rangle
  $$
  在我们的例子中，$n=3$，需要搜寻的元素数量是 $N = 2^3 = 8$，具体来说，就是
  $$
  |\psi\rangle = \frac{1}{\sqrt{8}}\bigg(|000\rangle + \cdots +|111\rangle \bigg) \rightarrow_{decimal} \frac{1}{\sqrt{8}}\bigg(|0 \rangle + \cdots +|7\rangle \bigg) = \frac{1}{\sqrt{8}}\sum_{x=0}^{7} |x \rangle 
  = \frac{1}{\sqrt{8}}
  \begin{pmatrix}
  1 \\
  1 \\
  \vdots\\
  1
  \end{pmatrix}_{8\times 1}
  $$
  电路模型可以写成

  ![avatar](./PIC/step2.png)

  在我们的量子模拟器中，阿达玛门可以这样调用：

  ```python
      # The first step of Grover's search algorithm, superposition
      H(q[0])
      H(q[1])
      H(q[2])
  ```

- **量子黑箱 $\hat{O}$**

  一旦我们实现了叠加态，就可以利用一个量子黑箱，这个量子黑箱会识别出正确的答案（注意：这个量子黑箱仅仅是当你把一个元素放到它面前时，识别它是否是正确的答案，但是它却不能告诉你答案是什么）。这里我们提供的是一个相对相位的旋转黑箱，它能把正确答案的相对相位旋转 180 度，而且它**不需要额外的 ancilla 参考量子比特**
  $$
  \hat{O} |x\rangle = (-1)^{f(x)} |x\rangle, \quad \text{其中}    \,
      f(x) = \bigg\{\begin{array}{lr}
          1, & \text{当} x =x^* \\
          0, & \text{其他} 
          \end{array}
  $$
  通过施加黑箱，我们得到以下量子态
  $$
  \hat{O} |\psi\rangle = \frac{1}{\sqrt{8}}\bigg(|0 \rangle + \cdots - |5 \rangle +|6 \rangle +|7\rangle \bigg) = |\phi \rangle
  = \frac{1}{\sqrt{8}}
  \begin{pmatrix}
  1 \\
  1 \\
  \vdots\\
  -1 \\
  1\\
  1
  \end{pmatrix}_{8\times 1}
  $$
  我们待会儿会讨论这个神秘的"量子黑箱"， 下图是一个黑箱 $\hat{O}$ 的电路表示：

  ![avatar](./PIC/oracle2.png)

  ```python
      # Enter the first Grover iteration, the oracle Uf for |101>
      X(q[1])
      H(q[2])
      CCX(q[0], q[1], q[2])
      X(q[1])
      H(q[2])
  ```

- **振幅纯化**

  下一步，我们增加一个"振幅扩大"电路，它可以增加正确答案的振幅，减小其他错误答案的振幅，这个电路也被称为"弥散算子" $\hat{D}$， 这个算子计算所有态平均振幅 $\mu$，并让所有振幅沿着 $\mu$ 镜面反射，通过这个操作来扩大正确答案的振幅，在给定的例子中，$n=3$

  $$
  \hat{D} = 2*|\psi\rangle\langle \psi| -I^{\otimes n} = 2*H^{\otimes n}|0^{\otimes n} \rangle\langle 0^{\otimes n}|H^{\otimes n} -I^{\otimes n} = \frac{1}{4}
  \begin{pmatrix}
  -3 &1 \cdots &1 \\
  1 &-3  &1 \\
  \vdots &\vdots &\vdots\\
  1 &1 \cdots &-3 
  \end{pmatrix}_{8\times 8}
  $$

  施加弥散算子代表的门

  $$
  \hat{D} |\phi\rangle 
  = \frac{1}{4}
  \begin{pmatrix}
  -3 &1 \cdots &1 \\
  1 &-3  &1 \\
  \vdots &\vdots &\vdots\\
  1 &1 \cdots &-3 
  \end{pmatrix}*
  \frac{1}{\sqrt{8}}  
  \begin{pmatrix}
  1 \\
  1 \\
  \vdots\\
  -1 \\
  1\\
  1
  \end{pmatrix}    
  = \frac{1}{4\sqrt{2}}
  \begin{pmatrix}
  1 \\
  1 \\
  \vdots\\
  5 \\
  1 \\
  1
  \end{pmatrix}
  $$

  这个向量代表了找到每一个元素的所代表的振幅。比如得到 $|101\rangle$ 的概率是 $|\frac{5}{4\sqrt{2}}|^2 = 78.125\%$，这比剩下的元素所对应的概率大的多，它们每一个所代表的概率为 $|\frac{1}{4\sqrt{2}}|^2 = 3.125\%$。我们一般会把量子黑箱和弥散算子结合到一起，并称之为 **Grover 旋转** $\hat{G}$
  $$
  \hat{G} = \hat{D}*\hat{O}
  $$
  接下来如果我们再次施加 Grover 旋转，会得到
  $$
  \hat{G}^2 |\psi\rangle = 
  \frac{1}{8\sqrt{2}}
  \begin{pmatrix}
  -1 \\
  -1 \\
  \vdots\\
  11 \\
  -1\\
  -1
  \end{pmatrix}
  $$
  这次我们会以 $94.53\%$ 的概率得到 $|x^*\rangle$， 非常好的结果！$\hat{D}$ 的电路模型是
  
  ![avatar](./PIC/diffusion2.png)

- **总结**

Grover 搜索算法包含以上几个步骤，可以总结为一张流程图。对于高维的问题，需要重复运算 Grover 旋转（下图中虚线框内）$\hat{G}$ 大约 $\sqrt{N}$ 次，就能得到很好的结果。
![avatar](./PIC/pipeline.png)

---
## 关于量子黑箱的一些见解

天下没有免费的午餐，Grover 量子搜索算法同样也是这样。平方加速带来的是量子黑箱构造的困难，让我们回到旅行商问题上，你会意识到其实并没有什么简单直接方法可以建造量子黑箱。甚至说可能构造这样一个量子黑箱就需要 $\mathcal{O}(\sqrt{N})$ 的复杂度，最后带来一个常数级加速。最坏的情况下，构造量子黑箱甚至可能比原问题更困难。

---
## 代码以及结果

我们现在来实战演练一下，如下是一个简单的展示
```python
import sys
sys.path.append('../../..')  # "from QCompute import *" requires this
from QCompute import *

matchSdkVersion('Python 2.0.2')

# In this example we use 3 qubits in total
qubit_num = 3
# and set the shot number for each request
shots = 1000

def main():
    """
    main
    """
    # Create environment
    env = QEnv()
    # Choose backend Baidu Local Quantum Simulator-Sim2
    env.backend(BackendName.LocalBaiduSim2)

     # Initialize the three-qubit circuit
    q = env.Q.createList(qubit_num)

    # The first step of Grover's search algorithm, superposition
    H(q[0])
    H(q[1])
    H(q[2])

    # Enter the first Grover iteration, the oracle Uf for |101>
    X(q[1])
    H(q[2])
    CCX(q[0], q[1], q[2])
    X(q[1])
    H(q[2])

    # The first layer of Hadamard gates in the first Grover iteration
    H(q[0])
    H(q[1])
    H(q[2])
    
    # The reflection gate 2|0><0| - I in the first Grover iteration, which is divided to three parts:
    # two layer of X gates and a decomposition for the gate CCZ between the above two
    X(q[0])
    X(q[1])
    X(q[2])

    H(q[2])
    CCX(q[0], q[1], q[2])
    H(q[2])

    X(q[0])
    X(q[1])
    X(q[2])
    
    # The second layer of Hadamard gates in the first Grover iteration
    H(q[0])
    H(q[1])
    H(q[2])

    # Measure with the computational basis;
    # if the user you want to increase the number of Grover iteration,
    # please repeat the code from the comment “Enter the first Grover iteration” to here,
    # and then measure
    MeasureZ(*env.Q.toListPair())
    taskResult = env.commit(shots, fetchMeasure=True)
    return taskResult['counts']


if __name__ == '__main__':
    main()
```

测量结果是：

```python
Shots 1000
Data {'101': 798, '100': 31, '110': 30, '001': 24, '111': 36, '010': 30, '011': 23, '000': 28}
```

如果我们再次施加一个 Grover 旋转，那么测量到的概率会变为：

```python
Shots 1000
Data {'101': 795, '010': 28, '100': 28, '000': 36, '110': 28, '001': 29, '111': 28, '011': 28}
```
**注意：** 返回的结果均为二进制。

---
## 参考文献
[Grover L.K.: A fast quantum mechanical algorithm for database search, Proceedings, 28th Annual ACM Symposium on the Theory of Computing, (May 1996) p. 212](https://dl.acm.org/doi/abs/10.1145/237814.237866)


