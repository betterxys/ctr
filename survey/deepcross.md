# Deep Crossing: Web-Scale Modeling without Manualy Crafted Combinatorial Features

[原文链接](http://www.kdd.org/kdd2016/papers/files/adf0975-shanA.pdf)

DeepCross 的目的是为了自动产生交叉特征， 避免大量的手动特征工程， 对密集型和稀 疏型的数据具有同样的效果； 

## DeepCross 结构

![](https://betterxys.github.io/styles/images/ctrfigs/deepcross.png){height="500"}

deepcross 模型的结果如上图所示，主要包括四层： embedding layer, stacking layer, residual units 和 scoring layer, 目标函数采用 logloss。 

### embedding layer

embedding layer 是一个单层的神经网络， 采用 ReLU 作为激活函数：

$$
x_j^O = max(0, W_j X_j^I + b_j)
$$

DeepCross 的 embedding 和其他模型的有些区别， 这里的 embedding layer 几乎都是借鉴的 FM， 采用的线性方式， 这里有 ReLU 的激活函数； 


### stacking layer

$$
X^O = [x_0^o, x_1^o, ..., x_n^o]
$$

stacking layer 是将上层 embedding layer 的输出 concat 为一个 n 维向量， 这里的 n 就是 Feature 的数量。

值得注意的是， 本文 embedding 的维度设置为 256 维， 并且有一个细节是，embedding 仅仅针对 one hot 过后维度大于 256 维的特征， 而保留不足256维的特征， 直接参与 stacking layer, 就像图中的 Feature #2; 

### residual layer

![](https://betterxys.github.io/styles/images/ctrfigs/residualunit.png){height="300"}

本文采用了简化版的 residual unit， 如上图所示， 输入向量经过两层 ReLU 单元后的输出， 再加上原始的输入， 组成了 residual unit 的输出值。

$$
X^O = F(X^I, \{ W_o, W_1 \}, \{ b_0, b_1 \}) + X^I
$$

从上式可以看出， F 拟合的是输出值与输出值的差， 这也是 residual 的含义；

### scoring layer

最终的 scoring layer 是一个 sigmoid 单元

## 实现

本文作者在论文中给出了 CNTK 版的实现：

```
ONELAYER(Dim_XO, Dim_XI, XI){
W = Parameter(Dim_XO, Dim_XI)
b = Parameter(Dim_XO)
XO = Plus(Times(W, XI), b)
}

ONELAYERSIG(Dim_XO, Dim_XI, XI){
t = ONELAYER(Dim_XO, Dim_XI, XI)
XO = Sigmoid(t)
}

ONELAYERRELU(Dim_XO, Dim_XI, XI){
t = ONELAYER(Dim_XO, Dim_XI, XI)
XO = ReLU(t)
}

RESIDUALUNIT(Dim_H, Dim_XI, XI){
l1 = ONELAYERRELU(Dim_H, Dim_XI, XI)
l2 = ONELAYER(Dim_XI, Dim_H, l1)
XO = ReLU(Plus(XI, l2))
}

## The Deep Crossing Model
## Step 1: Read in features, omitted
## Step 2A: Embedding
Q = ONELAYERRELU(Dim_E, Dim_Query, Query)
K = ONELAYERRELU(Dim_E, Dim_Keyword, Keyword)
T = ONELAYERRELU(Dim_E, Dim_Title, Title)
C = ONELAYERRELU(Dim_E, Dim_CampaignID, CampaignID)
## Step 2B: Stacking
# M = MatchType, CC = CampaignIDCount
Stack = RowStack(Q, K, T, C, M, CC)
## Step 3: Deep Residual Layers
r1 = RESIDUALUNIT(Dim_CROSS1, Dim_Stack, Stack)
r2 = RESIDUALUNIT(Dim_CROSS2, Dim_Stack, r1)
r3 = RESIDUALUNIT(Dim_CROSS3, Dim_Stack, r2)
r4 = RESIDUALUNIT(Dim_CROSS4, Dim_Stack, r3)
r5 = RESIDUALUNIT(Dim_CROSS5, Dim_Stack, r4)
## Step 4: Final Sigmoid Layer
Predict = ONELAYERSIG(Dim_L, Dim_Stack, r5)
## Step 5: Log Loss Objective
CE = LogLoss(Label, Predict)
CriteriaNodes = (CE)
```

其中模型参数设置：

Dim_E = 256,  # embedding dim
Dim_CROSS[1-5] = [512, 512, 256, 128, 64]  # residual size

## 后续思考

本文对离散特征的处理， 与其他论文有写区别， 作者仅保留离散特征的最频繁的10000个离散值， 其他的通过统计量来描述 (这块的具体实现方式一直没有看太明白， 详见论文4.1-Individual Featuers) ， 但是最后又通过实验表明， 不加这些统计量的时候模型性能表现， 要明显强于加上统计量， 本人对这块有点模糊， 抛却统计量的事情， 是不是应该对比一下仅保留 top 10000 和全部保留的性能区别？ 
