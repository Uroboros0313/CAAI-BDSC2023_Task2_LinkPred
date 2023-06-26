# CAAI-BDSC2023_Task2_LinkPred

## CAAI-BDSC2023_Task2 社交图谱动态链接预测

- **任务描述**
商品分享是用户在电商平台中主要的社交互动形式，商品分享行为包含了用户基于商品的社交互动偏好信息，用户的商品偏好和社交关系随时间变化，其基于商品的社交互动属性也随时间改变。本任务关注社交图谱动态链接预测问题，用四元组 (u, i, v, t) 表示用户在t (time) 时刻的商品分享互动行为，其中i (item) 标识特定商品，u (user) 表示发起商品链接分享的邀请用户，v (voter) 表示接收并点击该商品链接的回流用户。因此在本任务中链接预测指的是，在已知邀请用户u，商品i和时间t的情况下，预测对应回流用户v。

- **任务目标**
针对社交图谱动态链接预测，参赛队伍需要根据已有动态社交图谱四元组数据，对于给定u，i，t，对v进行预测。

## 方案

- **Rank**: 
  - 初赛: 37 / 805
  - 复赛: 20 / 805
### 模型
- 评价指标: Mean Reciprocal Rank(MRR)
- 模型选型(线下验证)

|Model|MRR|
|:-:|:-:|
|GCN(特征随机生成)|0.03|
|GCN(实体嵌入)|0.08|
|GCN(实体嵌入, 去ReLU和Linear)|0.10|
|训练集分享次数最多top5|0.09|
|TransE|0.14|
|SimpLE|0.18|
|DisMult|0.18|

```
- CompGCN由于算力不支持, 放弃
- Node2Vec、DeepWalk等GE模型线下分数很差,
- MF召回模型线下表现很差
```
- 模型优化

1. 节点和边id数量处于一个量级
2. 时间怎么处理
3. 用户和物品特征怎么融合

|提升方法|MRR|
|:-:|:-:|
|关系特征GES[1]|⬆ 0.04|
|用户特征GES|⬇⬇⬇|
|验证集加入训练|⬆ 0.05|
|Ranking Average Ensemble|⬆ 0.02|
|负采样时提高头节点替换概率|⬆0.00x|
|Margin Ranking Loss|⬆0.00x|
|时间加权损失函数|-|

- 遇到的问题
```
1. 时间特征没有用上，基本没有时间去探索更复杂的用户与时间特征的做法
2. EGES的效果相比GES要更差，也许是因为该数据集上线性层参数也比较难学习——使用Embedding作为输入时，线性层意义不大
3. u2x等召回基本方法可以尝试使用，但是没有时间深入挖掘
```


[1] [Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba](https://arxiv.org/pdf/1803.02349.pdf)

[2] [Embedding Entities and Relations For Learning And Inference In Knowledge Bases](https://arxiv.org/abs/1412.6575)

[3] [Translating Embeddings for Modeling
Multi-relational Data](https://proceedings.neurips.cc/paper_files/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html)