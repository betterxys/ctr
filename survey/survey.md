# CTR综述

[TOC]

---

## CTR的应用场景

CTR主要应用在广告系统和推荐系统当中，其目的在于预估出用户点击相应广告/推荐物品的概率。

CTR在推荐系统中的应用方式主要有两种：

1. 推荐列表依据 \\( CTR \\) 的值进行排序

- 推荐列表依据 \\( CTR*bid \\) 的值进行排序（ \\(bid\\)是指如果用户点击该推荐，商家可以获得的收益） 

CTR在广告系统中的应用也是类似，平台会根据广告收益和CTR的值对广告进行排序。


## CTR预测需要解决的问题

1. 难以发掘有效的**交叉特征**， 尤其是高阶交叉特征；

2. 传统 ML 方法难以适用于超**高维**度的**稀疏**数据；

## CTR的主流技术

### FM (Factorization machine)

FM 的出现主要想解决的问题是 SVM 在高维稀疏空间难以训练的问题， 现已成为稀疏数据下预测问题的经典算法。 FM 包括三部分：全局偏置量、 原始特征加权求和项、 二阶交叉项， 其中二阶交叉是 FM 的关键之处，每个特征对应有一个隐向量， 通过对所有特征的隐向量两两做内积， 在求和组成二阶交叉项。FM的模型可以表示为： 

$$
\hat{y}(x) := w_0 + 
\sum_{i=1}^n w_i x_i +
\sum_{i = 1}^{n} \sum_{j = i + 1}^{n} <V_i, V_j> x_{i} x_{j}
$$


[原文链接](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf) [参考代码](https://github.com/ibayer/fastFM) [阅读记录](http://htmlpreview.github.io/?https://github.com/betterxys/ctr/blob/master/survey/fm.html)



### FFM (Field-aware Factorization Machine)

FFM 是获得了 criteo 和 avazu 的冠军模型，FFM 本质是 FM 模型的一个变体， 对 FM 进行改进， FM 当中每个特征只有一个隐变量， 是与其余所有变量共享的， 而 FFM 模型通过引入 field 的概念， 每个特征对每个不同的 field 都有不同的隐变量对应， 有更强的灵活性和适应性。

$$
\phi_{FFM}(w,x) = \sum_{j_1=1}^{n} \sum_{j_2=j_1+1}^n
(w_{j_1, f_2} \cdot w_{j_2, f_1}) x_{j_1}x_{j_2}
$$

[原文链接](https://www.andrew.cmu.edu/user/yongzhua/conferences/ffm.pdf) [参考代码](https://github.com/guestwalk/libffm) [阅读记录](http://htmlpreview.github.io/?https://github.com/betterxys/ctr/blob/master/survey/fm.html)


### Wide & Deep model 
![](https://betterxys.github.io/styles/images/ctrfigs/wdm.png){:height="300", width="900"}
Wide & Deep 模型 是 2016 年 Google 开源于 TensorFlow 的一种混合网络结构 , Wide & Deep model 的出现是因为作者认为， 交叉特征的生成主要通过记忆式和生成式两种方式， 记忆式就是数据当中出现过的， 能够体现出来的特征交叉， 而生成式是指现有数据当中没有体现出来的隐式的交叉特征， 记忆式的特征提取较为简单， Wide & Deep 当中的 Wide 就是用于提取这部分交叉特征的， 而隐式的生成式特征交叉就只能通过深度学习深入挖掘， 所以 deep 部分的主要职能就是为了挖掘这部分生成式交叉特征。

[原文链接](https://arxiv.org/pdf/1606.07792.pdf) [参考代码](https://www.tensorflow.org/tutorials/wide_and_deep) [阅读记录](http://htmlpreview.github.io/?https://github.com/betterxys/ctr/blob/master/survey/wdm.html)


### FNN(Factorization-machine supported Neural Network)
![](https://betterxys.github.io/styles/images/ctrfigs/FNN.png){:height="300"}

FNN 拟解决的关键问题是高维稀疏特征难以通过神经网络进行训练，FNN 借鉴了 FM 的思路解决高维稀疏问题。FNN 在全连接的 hidden layer 之前添加了一个 Dense Real Layer 将原始的高维稀疏向量转化为低维连续向量， 训练时，预训练 FM , 然后以 FM 训练的 embedding 初始化 Dense Real Layer 的参数, 此外作者采用了 partial 连接的方式， 减少了参数的数量并建议将这种方法推广到更高层 。

### SNN(Sampling-based Neural Network)
![](https://betterxys.github.io/styles/images/ctrfigs/snn_all.png){:height="300"}

SNN与FNN发表于同一篇论文， 二者的基本结构相同，主要的区别有两点：

- SNN的第一层是全连接的，而FNN的第一层是基于Field进行b部分连接的;

- SNN的第一层权重初始化采用 Sampling-based RBM/DAE， 而FNN的第一层权重初始化需要进行FM的预训练;

此外，本文 SNN 所谓 Sampling based 是指，选中 one hot 的向量中非零的一项， 再随机选中 m 个值为零的项参与训练，如此一来可以减小训练复杂度;

[原文链接](https://arxiv.org/pdf/1601.02376.pdf) [参考代码](https://github.com/wnzhang/deep-ctr) [阅读记录](http://htmlpreview.github.io/?https://github.com/betterxys/ctr/blob/master/survey/fnn.html)


### PNN(Product-based Neural Network)

![](https://betterxys.github.io/styles/images/ctrfigs/PNN.png){:height="350"}
PNN的结构很清晰，针对CTR预测问题的两个难点： 特征向量高维稀疏和交叉特征难以提取，提出了解决方案： 

- 借鉴 FM 应对高维稀疏的方法， 在 input layer 之后添加一层 embedding layer， 将高维稀疏向量转化为低维连续向量解决高维稀疏带来的问题； 

- 在 embedding layer 之后添加 product layer， product layer 由 z 和 p 两部分组成， z 是原始特征和常数1做 product, 其实就是为了保留原始特征， 而 p 部分可以有选择的采用 inner product 或者 outter product 对 embedding 向量两两 product, 两部分同时作为 hidden layer 的输入， 通过 hidden layer 深入挖掘高阶交叉特征；

此外， 值得注意的是 PNN 采用了下采样的操作解决数据分布不均的问题。 

[原文链接](https://arxiv.org/pdf/1611.00144.pdf) [参考代码](https://github.com/Atomu2014/product-nets) [阅读记录](http://htmlpreview.github.io/?https://github.com/betterxys/ctr/blob/master/survey/pnn.html)



### DeepFM(Deep Factorization Machine)
![](https://betterxys.github.io/styles/images/ctrfigs/deepfm.png){:height="300"}
DeepFM 结合了 FM 和 DNN 模型，该模型主要分为 deep 和 wide 两部分，deep 模块使用 deep learning 学习高阶交叉特征，wide 模块使用 FM 提取低阶交叉特征，deep和wide两个模块共享输入和embeddings，最终通过output layer进行结合，输出CTR。

DeepFM 和 wide & deep 的区别在于:

- DeepFM 的 wide 部分由 FM 组成， 不需要人工参与特征构造；
- DeepFM 的 input 和 embedding 由 deep 和 wide 两部分共享；

[原文链接](https://arxiv.org/pdf/1703.04247.pdf) [参考代码](https://github.com/Leavingseason/OpenLearning4DeepRecsys) [阅读记录](http://htmlpreview.github.io/?https://github.com/betterxys/ctr/blob/master/survey/deepfm.html)


### Neural Factorization Machines(2017)
![](https://betterxys.github.io/styles/images/ctrfigs/NFM.png){:height="300"}

Neural FM 其实是 FM 的一个神经网络的实现版， 可以轻松复现 FM， 同时， 由于采用神经网络的结构实现了 FM， 所以可以在 FM 之后添加 hidden layer 来深化 FM 以学习非线性的高阶交叉特征。 Neural FM 的关键在于其 Bi-Interaction layer, 如果去掉后续的 hidden layer 它可以复现 FM 的效果， 而令 Neural FM 与其他神经网络不同的地方同样在于 Bi-Interaction layer, 其他的网络结构， 在输入数据进网络时， 一般都是简单的加权平均， 而 NFM 则是做了两两内积求和生成一个向量的 pooling 操作， 大大简化了后续 hidden layer 的训练复杂度。 


[原文链接](https://arxiv.org/pdf/1708.05027.pdf) [参考代码](https://github.com/hexiangnan/neural factorization machine) [阅读记录](http://htmlpreview.github.io/?https://github.com/betterxys/ctr/blob/master/survey/nfm.html)


### Attention Factorization Machines(2017)
![](https://betterxys.github.io/styles/images/ctrfigs/AFM.png){:height="300"}

AFM 和 NFM 来自于同一人，AFM 模型主要解决的问题在于， FM 模型在学习交叉特征的时候， 对所有的交叉特征项赋予了相同的权重， 而在实际情况当中， 不可能所有的特征对结果都有相同的影响， 所以， AFM 在原有 FM 的基础上， 添加 Attention 机制， 旨在赋予不同的交叉特征以不同的权重。

[原文链接](https://arxiv.org/pdf/1708.04617.pdf) [参考代码](https://github.com/hexiangnan/attentional factorization machine) [阅读记录](http://htmlpreview.github.io/?https://github.com/betterxys/ctr/blob/master/survey/afm.html)


### DeepCross

![](https://betterxys.github.io/styles/images/ctrfigs/deepcross.png){height="300"}

deepcross 模型的结果如上图所示，主要包括四层： embedding layer, stacking layer, residual units 和 scoring layer。embedding layer 将高维稀疏向量转换为低维连续向量， stacking layer 将所有 feature 的 embedding 向量 concat 到一起输入到 residual layers 学习交叉特征， 最终到 scoring layer 输出预测结果。 该模型最主要的贡献在于引入了 residual units 并取得了较好的效果。

[原文链接](http://www.kdd.org/kdd2016/papers/files/adf0975-shanA.pdf)  [阅读记录](http://htmlpreview.github.io/?https://github.com/betterxys/ctr/blob/master/survey/deepcross.html)


### others


- FTRL(Follow the Regularied Leader)
FTRL是一种基于逻辑回归的**在线学习算法**,能够学习出有效且稀疏的模型， FTRL 采用的依然是线性模型， 更倾向于工程方案。


- GBDT + LR
Facebook的论文尝试提出一种[解决特征组合问题的方案](https://github.com/neal668/LightGBM-GBDT-LR),基本思路是利用树模型的组合特性来自动做特征组合, 结合 GBDT 训练出一些组合特征, 将 GBDT 叶节点的输出结果传入 LR 进行分类,该方法取得极大成功,但也有很大程度的过拟合风险,所以必须要采取相应的防止过拟合的措施。


- RNN
更倾向于序列数据

- CNN
更倾向于由相邻特征之间的相互作用而产生的交叉特征.


## CT预测总体流程

通常， CTR预测包括如下流程：

- 特征预处理
	- 连续特征标准化
	- 连续特征离散化(离散特征更易于提取交叉特征)
	- 离散特征二值化
	
- embedding layer

	embedding layer 的作用是将高维稀疏特征转换为低维连续特征， 虽然大体的思路类似，但这里是否可以考虑引入 FFM 的 Field 的概念， 每个 feature 对应多个隐向量。
	
- feature interaction

	仅仅是原始特征很难达到好的效果，所以需要挖掘交叉特征，当前交叉特征的学习方式只有两类： product 和 hidden layer。
	
	- PNN 的作者尝试了 inner product 和 outter product， 但两者并没有表现出过大的差异， 所以 product 的方式基本可以锁定为 inner product， 可以考虑改进的方式是与 embedding layer 结合采取 FFM 的 field 的思路， 最好是改进为一个可以复现 FFM 的 NN 模型；
	
	- hidden layer 可以操作的空间较大， DeepCross 的 residual unit 可以应用到 hidden layer, 甚至 AFM 的 attention 机制也可以应用到 hidden layer 的输入层， 此外 dropout / bn / early stopping 应该都只算是该模型优化的一些小技巧；

	
- output layer

	output layer一般都是sigmoid function






## 参考文献

[^1]: Rendle, Steffen. "Factorization machines with libfm." ACM Transactions on Intelligent Systems and Technology (TIST) 3.3 (2012): 57.

[^2]: CHENG H-T, KOC L, HARMSEN J, et al. Wide & deep learning for recommender sys-tems[C] // Proceedings of the 1st Workshop on Deep Learning for Recommender Systems. 2016 : 7 – 10.

[^3]: Jahrer, Michael, et al. "Ensemble of collaborative filtering and feature engineered models for click through rate prediction." KDDCup Workshop. 2012.

[^4]: MCMAHAN H B, STREETER M. Adaptive bound optimization for online convex opti-mization[J]. arXiv preprint arXiv:1002.4908, 2010.

[^5]: MCMAHAN H B, HOLT G, SCULLEY D, et al. Ad click prediction: a view from thetrenches[C] // Proceedings of the 19th ACM SIGKDD international conference on Knowl-edge discovery and data mining. 2013 : 1222 – 1230.

[^6]: HE X, PAN J, JIN O, et al. Practical lessons from predicting clicks on ads at facebook[C]// Proceedings of the Eighth International Workshop on Data Mining for Online Advertising.2014 : 1 – 9.

[^7]: Zhang, Yuyu, et al. "Sequential Click Prediction for Sponsored Search with Recurrent Neural Networks." AAAI. 2014.

[^8]: Liu, Qiang, et al. "A convolutional click prediction model." Proceedings of the 24th ACM International on Conference on Information and Knowledge Management. ACM, 2015.

[^9]: ZHANG W, DU T, WANG J. Deep learning over multi-field categorical data[C] // Europeanconference on information retrieval. 2016 : 45 – 57.

[^10]: QU Y, CAI H, REN K, et al. Product-based neural networks for user response prediction[C]// Data Mining (ICDM), 2016 IEEE 16th International Conference on. 2016 : 1149 – 1154.


[^11]: GUO H, TANG R, YE Y, et al. DeepFM: A Factorization-Machine based Neural Networkfor CTR Prediction[J]. arXiv preprint arXiv:1703.04247, 2017.


[^12]: HE X, CHUA T-S. Neural Factorization Machines for Sparse Predictive Analytics[J], 2017.

[^13]: XIAO J, YE H, HE X, et al. Attentional factorization machines: Learning the weight of feature interactions via attention networks[J]. arXiv preprint arXiv:1708.04617, 2017.


[^14]: Steffen Rendle (2012): Factorization Machines with libFM, in ACM Trans. Intell. Syst. Technol., 3(3), May. 