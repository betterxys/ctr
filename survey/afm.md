# Attentional Factorization Machines

本文提出的 AFM 模型主要解决的问题在于， FM 模型在学习交叉特征的时候， 对所有的交叉特征项赋予了相同的权重， 而在实际情况当中， 不可能所有的特征对结果都有相同的影响， 所以， 本文在原有 FM 的基础上， 添加 Attention 机制， 旨在赋予不同的交叉特征以不同的权重。 

attention FM 的优势在于：

- 更好的性能表现；
- 更好的可解释性， 可以进一步分析具体的交叉特征对结果的影响；


## AFM 的结构

AFM 包括三部分， 全局偏置项、 原始特征加权求和项、 交叉特征的学习项（如下图所示）。

$$
\hat{y}_{AFM}(x) = w_0 + \sum_{i=1}^n w_i x_i +
p^T \sum_{i=1}^n \sum_{j=i+1}^{n} a_{ij} (v_i \bigodot v_j) x_i x_j
$$


![](https://betterxys.github.io/styles/images/ctrfigs/AFM.png){height="500"}


### Embedding Layer
AFM 的 input layer 和 embedding layer 与 FM / NFM 是一致的， 都是将原有特征转换为 embedding 向量再参与后续计算， 


### Pair-wise Interaction Layer

Pair-wise Interaction Layer 和 Neural FM 的 Bi-Interaction Layer 其实是一样的；

$$
f_{PI}(\varepsilon) = {(v_i \bigodot v_j) x_i x_j}
$$

$$
R_x = \{ (i, j) \}_{i \in X, j \in X, j>i}
$$

### Attention-based Pooling Layer

$$
f_{Att}(f_{PI}(\varepsilon)) = \sum_{(i,j)\in R_x} a_{ij} (v_i \bigodot v_j) x_i x_j
$$

其中 \\( a_{ij} \\) 是 特征 i 和特征 j 的交叉特征的 attention score， 本文的 attention net 是单层隐含层，但**后续可以考虑 Attention 部分采用 MLP 来学习 attention score**, 对从未出现过的交叉特征进行评分， 目前单层只能学习到数据中出现的交叉特征的权重；

本文 attention-based pooling 层的输出是一个 k 维向量， k 是 embedding 向量的维度， 这和 Neural FM 类似， NFM BI层的输出也是 k 维向量；

$$
a_{ij} = softmax(a'_{ij})
$$

$$
a'_{ij} = h^T ReLU(W(v_i \bigodot v_j) x_i x_j + b)
$$ 

- attention net 隐含层的节点数称为 attention factor， 对应的是 attention score 的个数;
- attention net 隐含层的激活函数采用 ReLU;
- attention score 通过 softmax 进行标准化处理;

## 训练技巧

- 对 Pair-wise Interaction Layer 使用 dropout

- attention net 不适用 dropout

- 目标函数加入 L2 正则化


## future research

- 探索能够有效获取高阶交叉特征的方法
- 在 Attention-based Pooling 后添加 MLP
- consider improving its learning efficiency, for example by using **learning to hash** and **data sampling techniques**.
- develop FM variants for semi-supervised and multi-view learning, for example by incorporating the widely used **graph Laplacian** and **co-regularization designs**.

### 需要继续学习的paper

以下 paper 都是 NFM 和 AFM 当中提到的性能相关的 future work， 优先级不高。

- data sampling

	- Meng Wang, Weijie Fu, Shijie Hao, Hengchang Liu, and Xindong Wu. Learning on big graph: Label inference and regularization with anchor hierarchy. IEEE TKDE, 2017.

- co-regularization

	- [Xiangnan He, Min-Yen Kan, Peichu Xie, and Xiao Chen. Comment-based multi-view clustering of web 2.0 items. In WWW, 2014.](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.463.2280&rep=rep1&type=pdf)
	
	- Yang Yang, Zhigang Ma, Yi Yang, Feiping Nie, and Heng Tao Shen. Multitask spectral clustering by exploring intertask correlation. IEEE TCYB, 2015.

- hashing tech

	- [Fumin Shen, Chunhua Shen, Wei Liu, and Heng Tao Shen. Supervised discrete hashing. In CVPR, 2015.](http://www.ee.columbia.edu/~wliu/CVPR15_SDH.pdf)
	
	- [Shen, Fumin, et al. "Classification by Retrieval: Binarizing Data and Classifier." (2017). ](http://www.ee.columbia.edu/~wliu/SIGIR17_binarizing.pdf)
	
	- [Zhang, Hanwang, et al. "Discrete collaborative filtering." Proceedings of the 39th International ACM SIGIR conference on Research and Development in Information Retrieval. ACM, 2016.](https://pdfs.semanticscholar.org/ef86/1dbbc8779e8d83bc588b23b0bd25fbaa9f36.pdf)


- graph Laplacian

	- [X. He, M. Gao, M.-Y. Kan, Y. Liu, and K. Sugiyama. Predicting the popularity of web 2.0 items based on user comments. In SIGIR, 2014.](https://www.comp.nus.edu.sg/~kanmy/papers/sigir2014_he.pdf)
	
	- [M. Wang, W. Fu, S. Hao, D. Tao, and X. Wu. Scalable semi-supervised learning by efficient anchor graph regularization. IEEE Transaction on Knowledge and
Data Engineering, 2016.](http://www.projectsgoal.com/download_projects/data-mining/data-mining-projects-GDM00056.pdf)

	- [Xiangnan He, Ming Gao, Min-Yen Kan, and Dingxian Wang. BiRank: Towards ranking on bipartite graphs. IEEE TKDE, 2017.](http://www.comp.nus.edu.sg/~xiangnan/papers/tkde16-birank-cr.pdf)
