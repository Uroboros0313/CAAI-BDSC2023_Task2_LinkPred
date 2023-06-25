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
  - 提交训练集中分享次数最多的五个用户: 0.09
  - GCN + RandomInit Features：0.03
  - GCN + Entity Emebedding + 去线性层和ReLU: 0.10
  - TransE: 0.14
  - SimpLE/Dismult: 0.18——最终选型Dismult
  - CompGCN——算力不支持, 放弃
  - Node2Vec、DeepWalk等GE模型线下分数很差
  - Two Tower等召回模型线下表现很差
  
- 模型优化

模型优化主要考虑的是边和节点以及边类型的个数基本处于同一个量级时怎么处理。结合LightGCN的论文以及实验得出的结论是尽量取消线性层和激活函数, 让梯度可以直接反传到Embedding Table上。实际上我的观点是当输入是Embedding时只需要聚合函数, 不需要任何多余操作。

  - Relation Attribute: 利用**EGES**[1]中方法融合边特征, 测试发现EGES的融合方法效果较差, GES的SumPooling较好, 线下MRR提升0.04
  - 头节点替换概率提高, 尾节点替换概率下降, 少量提分
  - Margin Ranking Loss 代替 LogLoss, 少量提分
  - Validation as Input, 测试集上MRR提升0.05左右
  - 20个Dismult+GES进行超参扰动后排序平均融合, 线上提升0.02左右

## 总结

- 这场比赛和我看到的过去的知识图谱的内容不太一样, 原因是这场比赛几乎是一个纯粹的GraphEmbedding的任务。
- 第一次接触KGE模型的比赛, 这场比赛我几乎没有时间和算力去做, Base很早就搭好了, 到了最后两周才有空去学DGL, 不过最后也因为算力不充足放弃了。知识图谱内部也许有些我不了解的Trick, 希望后续有机会学习一下。

- 在GNN模型搭建里, 最初我认为PyG好用, DGL的文档等很不直观, 但是一旦涉及到自定义MessagePassing时DGL使用起来很方便。



[1] [Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba](https://arxiv.org/pdf/1803.02349.pdf)

[2] [Embedding Entities and Relations For Learning And Inference In Knowledge Bases](https://arxiv.org/abs/1412.6575)

[3] [Translating Embeddings for Modeling
Multi-relational Data](https://proceedings.neurips.cc/paper_files/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html)