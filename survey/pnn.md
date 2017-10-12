# Product-based Neural Networks for User Response Prediction

## 主要解决的问题

- 高维稀疏特征学习： 离散型特征 one-hot-encoding 之后会转换为高维稀疏的二值化特征，导致难以训练学习；

- 交叉特征提取： 交叉特征对于模型性能有很大的影响，交叉特征提取的越充分，模型表现越好； 

***这里有个疑问：怎么算是提取到交叉特征了？本文采取inner/outter product两种方式，也就是说其实还存在其它的提取方式？***

## PNN 的优势

- 无需预训练
- product layer 应用于 embedded features 学习交叉特征
- 通过全连接的MLP进一步提取高阶交叉特征


## PNN 模型结构

![](https://betterxys.github.io/styles/images/ctrfigs/PNN.png){:height=500}


### output layer
输出预测结果

$$
\hat{y} = \sigma(W_3 l_2 +b_3)
$$

其中 \\( \sigma \\) 是 sigmoid 函数

### fully connected layers
学习高阶交叉特征

$$
l_2 = relu(W_2 l_1 + b_2)
$$

$$
l_1 = relu(W_{1z} \bigodot Z + W_{1p} \bigodot P + b_1)
$$

其中 \\(L_1\\) 层是分别对 product layer 输出的 Z 和 P 与相应的参数 \\( W_z , W_p \\) 做内积，内积可以定义为：

$$
A \bigodot B := \sum_{i,j} A_{i,j}B_{i,j}
$$

### product layer
学习交叉特征 P, 并保留原始特征 Z

$$
Z = (z_1, z_2, z_3, ..., z_N) := (f_1, f_2, f_3, ..., f_N) 
$$

对原始特征Z的求解中， f 代表的是各个特征对应的 embedding 向量，所以该式代表的含义是保留原始特征；

$$
P = {product({i, j})}, 其中 i=1...N, j=1...N
$$

对交叉特征P的求解过程可以采用不同的方式，product就是对两个特征进行 点乘/叉乘 运算以获取二阶交叉特征的函数。

***在求解P时，双重循环，意味着存在 AA, AB/BA 的情况， 会不会出现问题？***

### embedding layer
通过 emdding layer 学习离散特征的分布表示，从 sparse binary 特征转化为 dense real-value 特征

$$
f_i = W_0^i x[start_i : end_i]
$$

其中， \\( x[start_i : end_i] \\) 代表的是离散特征 i 进行 one hot encoding 之后的一组特征

## PNN 的训练

- loss function

PNN 采用 log-loss 进行训练

$$
L(y, \hat{y}) = -y log{\hat{y}} - (1-y) log{(1-\hat{y})}
$$

- PNN的复杂度优化

IPNN（采用 inner product 作为 product layer 的 product 函数）借鉴 FM 的思路来优化 内积 的计算；

OPNN（采用 outter product 作为 product layer 的 product 函数）借鉴 superposition 的思路优化 外积 的计算；

## PNN 的验证

### 验证数据集

- Criteo

从criteo的1T公开数据集中，选择了连续7天的数据作为训练，第8天的数据作为测试集；

由于数据量巨大且存在严重的数据倾斜，所以本文对负样本采取下采样的操作，令下采样的比例为w，ctr的预测值为p，则对ctr的预测值进行修正的到修正后的ctr预测值q为：
	$$
	q = \frac{p}{p + \frac{1-p}{w}}
	$$

下采样后共包含 79.38M 的样例，one-hot encoding 后共包含特征 1.64M；

- IPinYou

IPinYou提供了10天的广告点击日志记录，共计19.5M的样本，one-hot encoding过后共计937.67k个属性；以最后3天的数据作为测试集；

### 对比验证

本文作者用 tensorflow 实现了7种模型进行对比实验：
LR
FM: 10阶FM (同时所有神经网络模型都采用10阶embedding)
FNN :  1 * embedding layer + 3 * hidden layer
CCPM： 1 * embedding layer + 2 * convolution layer(max pooling) + 1 * hidden layer
IPNN : 1 * embedding layer + 1 * inner-product layer + 3 * hidden layer
OPNN: 1 * embedding layer + 1 * outter-product layer + 3 * hidden layer
PNN*: 1 * embedding layer + 1 * inner+outter product layer + 3 * hidden layer

- embedding 采用 10 阶(对比2/10/50/100，最终保留10)
- 采用随机梯度下降的方法进行训练
- 训练 FM / LR 时采用 L2 正则化 防止过拟合
- 训练神经网络时采用 dropout=0.5 防止过拟合

### 评估指标

- AUC
- RIG(relative information gain): RIG = 1 - NE(normalized cross entropy)
- LogLoss
- RMSE

### 验证结论

- 性能： LR <  FM < FM < NN < PNN
- PNN* 相对于 IPNN 和  OPNN 并没有太多提升
- 从 learning curve 来看， IPNN / OPNN 收敛较好

## Future work

- 设计新的 product layer

***这里有个问题，作者用了不少篇幅介绍其复杂度的优化措施，但是文章当中并没有比较PNN与其他NN模型的性能差异***