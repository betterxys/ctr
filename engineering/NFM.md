[TOC]

## NFM / AFM 复现

### 数据集

Frappe 

- train/validation/test: 202,027 / 57,722 / 28, 860

- feature: 10(离散特征) -> 所有可取特征值共计 5382 

###  模型

本文复现模型包括： NFM， AFM， 并采用 FM 与之进行对比；

####  NFM

NFM 主要结构：

- embedding layer: 将每个 feature 转换为一个 k 维 embedding;

- Bi-interaction layer: 求解二阶交叉特征， 输出一个 k 维向量; 

- deep layer: 将 BI 层的 k 维输出，输入到 MLP 中，旨在求解高阶交叉项；

- output layer： 全局偏置量 + 原始特征加权求和 + MLP的输出；

#### AFM

AFM 的主要结构：

- embedding layer: 将每个 feature 转换为一个 k 维 embedding;

- Pair-wise Interaction Layer: 两两 element product, 输出 n(n-1)/2 个 k 维向量；

- Attention Based Pooling: 这一部分主要是通过 attention net 计算出 feature interaction 对应的 attention score， 该分数代表每个 feature interaction 对应的权重；

- prediction layer: 对全局偏置量、线性加权和、attention based pooling 求和；


chenglong 对 attention based pooling 中 attention score 的计算方式进行改进：

原文 attention net的输入是一个 feature interaction 的 embedding 向量， 输出的 attention score 就是该 interaction 对应的权重，但这种方式仅仅根据当前输入的 interaction 来确定其对应的 attention score , 并没有考虑其它 feature interaction 对该 interaction 的影响； chenglong 在此基础上， 对这部分的结构进行修改， 首先对每个 interaction 都进行一个 sum pooling 操作， 将 featuer interaction 的 embedding 转换为一个对应的 scalar， 再将所有的 scalar 作为 attention net 的输入， 综合考虑所有 feature interaction, 计算出所有 interaction 对应的 attention score.


### 实验


#### NFM 实验

- *hidden_factor*
    
epoch|batch_size|hidden_factor|keep|lr|optimizer| params |train|validation|test
-  |- | -| - | - |   -  |   -  |  - | - |-
20    | 128 | 64 | [0.8,0.5] | 0.05 | adagrad   | 354055 | 0.1601 | 0.338 | 0.3448
50    | 128 | 64 | [0.8,0.5] | 0.05 | adagrad   | 354055 | 0.1163 | 0.322 | 0.3286
100  | 128 | 64 | [0.8,0.5] | 0.05 | adagrad   | 354055 | 0.1023 | 0.3167 | 0.3233
20    | 128 | 128 | [0.8,0.5] | 0.05 | adagrad | 702599 | 0.1461 | 0.3299 | 0.3316
50    | 128 | 128 | [0.8,0.5] | 0.05 | adagrad | 702599 | 0.0858 | 0.3136 | 0.3176
100  | 128 | 128 | [0.8,0.5] | 0.05 | adagrad | 702599 | 0.0723 | 0.3086 | 0.3120

本实验对比了 NFM 在不同的 hidden_factor 下的运行结果， 结果显示 hidden_facor 越大， 收敛越快， 运行结果越好， 但同时意味着模型参数变得更多， 模型更复杂；

- *batch_size*

epoch|batch_size|hidden_factor|keep|lr|optimizer| params |train|validation|test
-  |- | -| - | - |   -  |   -  |  - | - |-
20    | 128 | 128 | [0.8,0.5] | 0.05 | adagrad | 702599 | 0.1461 | 0.3299 | 0.3316
50    | 128 | 128 | [0.8,0.5] | 0.05 | adagrad | 702599 | 0.0858 | 0.3136 | 0.3176
100  | 128 | 128 | [0.8,0.5] | 0.05 | adagrad | 702599 | 0.0723 | 0.3086 | 0.3120
20    | 4096 | 128 | [0.8,0.5] | 0.05 | adagrad | 702599 |  0.2077 | 0.3613 | 0.3625
50    | 4096 | 128 | [0.8,0.5] | 0.05 | adagrad | 702599 |  0.1344 | 0.3284 | 0.3312
100  | 4096 | 128 | [0.8,0.5] | 0.05 | adagrad | 702599 |  0.1060 | 0.3191 | 0.3209

本实验验证 batch_size 对 NFM 运行结果的影响，  可见更小的 batch_size, 收敛更快， 且效果更好；

- *learning rate*

epoch|batch_size|hidden_factor|keep|lr|optimizer| params |train|validation|test
-  |- | -| - | - |   -  |   -  |  - | - |-
20    | 128 | 128 | [0.8,0.5] | 0.05 | adagrad | 702599 | 0.1461 | 0.3299 | 0.3316
50    | 128 | 128 | [0.8,0.5] | 0.05 | adagrad | 702599 | 0.0858 | 0.3136 | 0.3176
100  | 128 | 128 | [0.8,0.5] | 0.05 | adagrad | 702599 | 0.0723 | 0.3086 | 0.3120
20    | 128 | 128 | [0.8,0.5] | 0.01 | adagrad | 702599 | 0.1555 | 0.3366 | 0.3412
50    | 128 | 128 | [0.8,0.5] | 0.01 | adagrad | 702599 | 0.1115 | 0.3202 | 0.324
100  | 128 | 128 | [0.8,0.5] | 0.01 | adagrad | 702599 | 0.0964 | 0.3143 | 0.3178

更小的学习率会有更慢的收敛速度

- *optimizer*

epoch|batch_size|hidden_factor|keep|lr|optimizer| params |train|validation|test
-  |- | -| - | - |   -  |   -  |  - | - |-
20    | 128 | 128 | [0.8,0.5] | 0.05 | adagrad | 702599 | 0.1461 | 0.3299 | 0.3316
50    | 128 | 128 | [0.8,0.5] | 0.05 | adagrad | 702599 | 0.0858 | 0.3136 | 0.3176
100  | 128 | 128 | [0.8,0.5] | 0.05 | adagrad | 702599 | 0.0723 | 0.3086 | 0.3120
20    | 128 | 128 | [0.8,0.5] | 0.05 | adam | 702599 | 0.4581 | 0.5092 | 0.5126
50    | 128 | 128 | [0.8,0.5] | 0.05 | adam | 702599 | 0.3853 | 0.4586 | 0.4604
100  | 128 | 128 | [0.8,0.5] | 0.05 | adam | 702599 | 0.4383 | 0.4998 | 0.4983

adam 的表现极差



####  AFM 模型

- *batch_size*

[hidden nodes, embedding] | attention | batch | lr | optimizer | params | #epoch | train | valid
 - | - | - | - | - | - | - | - | - 
 [16,128] | 1 | 4096 | 0.05 | Adagrad | 696487  | 20   | 0.2092 | 0.3748
 [16,128] | 1 | 4096 | 0.05 | Adagrad | 696487  | 50   | 0.149 | 0.3548
 [16,128] | 1 | 4096 | 0.05 | Adagrad | 696487  | 100 | 0.1213 | 0.3466
 [16,128] | 1 | 128   | 0.05 | Adagrad | 696487  | 20   | 0.1536 | 0.3541
 [16,128] | 1 | 128   | 0.05 | Adagrad | 696487  | 50   | 0.1243 | 0.3453
 [16,128] | 1 | 128   | 0.05 | Adagrad | 696487  | 100 | 0.1111 | 0.3415

 更小的 batch_size 可以更快的收敛

 - *hidden_layer*

[hidden nodes, embedding] | attention | batch | lr | optimizer | params | #epoch | train | valid
 - | - | - | - | - | - | - | - | - 
 [16,128]   | 1 | 128   | 0.05 | Adagrad | 696487  | 20   | 0.1536 | 0.3541
 [16,128]   | 1 | 128   | 0.05 | Adagrad | 696487  | 50   | 0.1243 | 0.3453
 [16,128]   | 1 | 128   | 0.05 | Adagrad | 696487  | 100 | 0.1111 | 0.3415
 [128,128] | 1 | 128   | 0.05 | Adagrad | 711047  | 20   | 0.1527 | 0.3541
 [128,128] | 1 | 128   | 0.05 | Adagrad | 711047  | 50   | 0.1238 | 0.3452
 [128,128] | 1 | 128   | 0.05 | Adagrad | 711047  | 100 | 0.1113 | 0.3416

hidden  layer 的节点数目对结果并无太大影响

- *embedding*

[hidden nodes, embedding] | attention | batch | lr | optimizer | params | #epoch | train | valid
 - | - | - | - | - | - | - | - | - 
 [16,128]   | 1 | 128   | 0.05 | Adagrad | 696487  | 20   | 0.1536 | 0.3541
 [16,128]   | 1 | 128   | 0.05 | Adagrad | 696487  | 50   | 0.1243 | 0.3453
 [16,128]   | 1 | 128   | 0.05 | Adagrad | 696487  | 100 | 0.1111 | 0.3415
 [16,64]     | 1 | 128   | 0.05 | Adagrad | 350951  | 20   | 0.2145 | 0.3743
 [16,64]     | 1 | 128   | 0.05 | Adagrad | 350951  | 50   | 0.1829 | 0.3649
 [16,64]     | 1 | 128   | 0.05 | Adagrad | 350951  | 100 | 0.1691 | 0.3619

embedding 越大， 收敛越快， 效果越好


#### learning rate

![](https://betterxys.github.io/styles/images/ctrfigs/afm_learning_rate.png)
显然，采用 adagrad 作为 optimizer,  lr = 0.05 时 远胜于 lr=0.001

    
#### AFM vs NFM vs FM

设置 hidden layer = 64, embedding = 128, batch = 128, lr = 0.05, optimizer = adagrad 对比 NFM 和 AFM 模型效果；

![](https://betterxys.github.io/styles/images/ctrfigs/afmVSnfm.png){:height=700, width=1000}


model |  attention | param | epoch | train | valid | test
 -  |   -  |  -  |  -  |  - | - | -
NFM     | -          |      702599      | 39      | 0.0927 | 0.3142 | 0.3215 
AFM     | paper          |      702727      | 100      | 0.1112 | 0.3408 | 0.3443
AFM     | chenglong          |      700231      | 44      | 0.0743 | 0.3217 | 0.3278
FM       | -          |      694279      | 100      | 0.2789 | 0.4080 | 0.4164

结论： 在数据集 frappe 上， 相同参数设置下， NFM 的效果要强于 AFM ， NFM 和 AFM 都要强过 FM； 而 chenglong 改进过的 AFM 在验证集上的表现最好；
