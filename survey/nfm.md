#  Neural Factorization Machines for Sparse Predictive Analytics

[原文链接](https://arxiv.org/pdf/1708.05027.pdf)
[开源代码](https://github.com/hexiangnan/)

## NFM 简介

对于拥有高维稀疏特征的预测问题而言， 如何有效地学习到交叉特征是重中之重，FM 可以学习到低阶交叉特征， Google 的 Wide&Deep 模型和 Microsoft 的 DeepCross 模型都是深度模型， 可以学习高高阶交叉特征， 但是难于训练； 


FM 模型是线性模型, 局限较大, 难以学习高阶/非线性交叉特征； 
其余的 DNN 模型, 在 embedding 之后, 都是简单的线性相加, 并没有太大作用；　

本文提出的 NFM 模型， 将 FM 和神经网络结合起来， 如果去掉隐藏层， NFM 就相当于是一个 FM 模型， 所以 FM 其实可以看做是 NFM 的一种特殊情况， NFM 在 FM 之后添加了几层隐藏层， 在原有 FM 的基础上大大增加了其特征表示能力， 而相对于 Wide&Deep 和 DeepCross 而言， NFM 的隐藏层很浅，所以降低了其计算难度。

单隐层的 NFM 比 LibFM 效果增强了 7.3%， 同样强于 3 层的 Wide&Deep 和 10 层的 DeepCross；

###  contributions

- 在神经网络建模过程中应用 Bi-Interaction pooling (Bilinear Interaction pooling)；
- 用神经网络深化 FM 以学习高阶/非线性交叉特征；
- 在两份真实数据上验证 Bi-Interaction pooling 和 NFM 模型； 

**其实最主要的优势就是将 FM 和 NN 结合到一起， 相对于 FM 多了隐含层， 相对于其它 NN 模型， 采用了 Bi-Interaction pooling， 而不是简单的加权求和；**

### 提取交叉特征的方法

在缺乏专业领域知识的情况下， 现今提取交叉特征的方法主要分为两类：

- FM一类的线性模型；
- 神经网络一类的非线性模型；

## NFM 结构

### 顶层结构

NFM 的结构可以用如下公式代表： 

$$
\hat{y}_{NFM}(x) = w_0 + \sum_{i=1}^{n}w_i x_i + f(x)
$$ 

这是一个总的结构， 第一项 \\(w_0\\) 是全局偏置量， 第二项是每个特征的加权求和项， 第三项才是本文的关键 NFM 的结构, \\(f{(x)}\\)的结构图如下图所示：

![](https://betterxys.github.io/styles/images/ctrfigs/NFM.png){height="500"}

### Embedding Layer

embedding layer 把原有的每个特征转换为新的 embedding 向量：

$$
V_x = \{x_1 v_1,...,x_n v_n\}
$$

计算时， 只需要包含非零项的 embedding 即可， 即：

$$
V_x = \{x_iv_i\}, where\  x_i \neq 0
$$

### Bi-Interaction Layer

Bi-Interaction 层的作用是将一个 embedding 向量集合转化为一个向量的 pooling 操作：

$$
f_{BI}(V_x) = \sum_{i=1}^n\sum_{j=i+1}^{n} x_i v_i \bigodot x_j v_j
$$

如上式所示， Bi-Interaction 其实是对 embedding 向量两两做 element product, 显而易见， Bi-Interaction layer 的输出是一个 k 维向量（embedding 向量的唯独为 k）；

同时，作者参照 FM 的做法， 将计算复杂度从 \\( O(kN^2)\\) 降低到 \\( O(kN) \\):

$$
f_{BI}(V_x) = \frac{1}{2} [ (\sum_{i=1}^n x_i v_i)^2 - (\sum_{i=1}^{n}(x_j v_j)^2]
$$


### Hidden Layer

全连接的隐藏层：

$$
z_1 = \sigma_1(W_1f_{BI}(V_x) + b_1)
$$

$$...$$

$$
z_L = \sigma_L(W_Lz_{L-1} + b_L)
$$

### Prediction Layer

$$
f(x) = h^T z_L
$$

其中， h 是预测神经元的参数


## NFM 特性

- NFM 去掉 hidden layer 等同于 FM， 所以 FM 可以视为NFM 的一种特殊情况； 

- 对 Bi-Interaction layer 应用 dropout 的效果， 要强于 FM 当中的 L2 正则化；

- 与 Wide&Deep 相比， 区别在于 Bi layer, 如果用 concatnation 替换到 Bi layer, NFM 可以复现 Wide & Deep

## NFM 训练技巧

- objective function
	- for regression: squared error
	- for classification:	logloss / hinge loss
	
- optimizer: mini-batch Adagrad

- Dropout
训练时，随机删除一部分节点，不参与调参，防止过拟合，本文建议在 Bi-Interaction Layer 采用 Dropout, 在实现过程中本文对每个 hidden layer 同样采用了 dropout;

- Batch Normalization
在训练过程中每层的输入都会随着上层参数的变化而发生变化，所以需要去调整参数适应上层这种变化，导致训练难度增加，BN 将每层的输入转换为一个零均值的高斯分布， 可以更快的收敛并取得更好的效果； 本文在BI层和所有hidden layer都有使用 BN；

- pre-training
用FM的参数进行初始化，可以加速收敛

## 实例验证

### 数据

- Frappe: 手机用户app使用记录，共28w条记录， 5k 特征；
- MovieLens: 电影推荐信息， 共200w记录， 9w维度；

### 验证方法

- 以 7:2:1 的方式拆分 训练集/验证集/测试集
- 将正例标签设置为1， 负例标签设置为 -1 ， 评估指标为 RMSE
- baseline: FM / higher-order FM / Wide & Deep / DeepCross

### 超参设置

- learning rate : [0.005, 0.01, 0.02, 0.05] -> 0.02

- L2 for FMs: [1e-6, 5e-6, 1e-5, ..., 1e-1]

- dropout: [0, 0.1, 0.2, ..., 0.9]  -> 0.3

- early stopping: 连续 4 个 epoch 的RMSE增加

- embedding size: 64

- optimization: FM: vanilla SGD; ohters: mini-batch Adagrad;


## future work

- **resorting to hashing techniques ** to make it more suitable for large-scale applications

**这个是啥意思？**

- extend the **objective function** with regularizers like the graph Laplacian

- exploring the BI-Interaction pooling for RNN for sequential data modelling