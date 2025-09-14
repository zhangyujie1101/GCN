# GCN复现实验任务

---

## 1. GCN是如何工作的？

图卷积神经网络（GCN）的工作原理主要基于层间的传播规则，通过特征传递来实现对图结构数据的处理：

<div align="center">
  <img src="image/传播公式.png" alt="特征传递规则" />
  <p><em>图1：GCN中的特征传递规则公式</em></p>
</div>

---

## 2. 在CV领域，什么是CNN？CNN与MLP的区别在哪儿？

### 2.1 什么是CNN

卷积神经网络（CNN）是专为网格结构数据（如图像、语音）设计的深度学习模型，其核心流程包括：
- 卷积层提取局部特征
- 池化层进行降维处理
- 全连接层完成分类任务

通过这一流程，CNN能够逐步将底层像素特征抽象为高层语义特征（如边缘、纹理、物体部件），广泛应用于图像分类、目标检测、图像分割等计算机视觉（CV）任务。

### 2.2 CNN与MLP的区别

<div align="center">
  <img src="image/MLPvsCNN.png" alt="MLP与CNN的区别" />
  <p><em>表2：多层感知器（MLP）与卷积神经网络（CNN）的对比</em></p>
</div>

---

## 3. GCN与其他神经网络的区别

### 3.1 GCN与MLP的区别

<div align="center">
  <img src="image/MLPvsGCN.png" alt="MLP与GCN的区别" />
  <p><em>表3：多层感知器（MLP）与图卷积神经网络（GCN）的对比</em></p>
</div>

### 3.2 GCN与CNN的区别

<div align="center">
  <img src="image/CNNvsGCN.png" alt="GCN与CNN的区别" />
  <p><em>表4：图卷积神经网络（GCN）与卷积神经网络（CNN）的对比</em></p>
</div>

---

## 4. 数据集上的节点分类任务准确率

运行时提示找不到包，注意用 PYTHONPATH=. python gcn/train.py 指定数据集加入 --dataset  citeseer

官方代码中未放nell数据集，但是论文实验中有一个nell数据集

### 4.1 Cora数据集上的节点分类任务准确率

<div align="center">
  <img src="image/cora_hyperparameter_tuning_results.png" alt="Cora数据集" />
  <p><em>图2：Cora数据集上的节点分类任务准确率</em></p>
</div>

### 4.2 Citeseer数据集上的节点分类任务准确率

<div align="center">
  <img src="image/citeseer_hyperparameter_tuning_results.png" alt="Citeseer数据集" />
  <p><em>图3：Citeseer数据集上的节点分类任务准确率</em></p>
</div>

---

## 5. 中毒攻击

修改完数据后，重新训练模型，此为中毒攻击，中毒攻击的数据修改一般针对训练集

### 5.1 随机删除Cora数据集中的n个节点，可视化删除后模型分类效果

随机删除可以理解为，对应节点的特征置为0，同时邻接矩阵对应行列均置为0，该节点成为孤立节点：或者直接从特征矩阵与邻接矩阵中移除该节点对应的属性

<div align="center">
  <img src="image/Poison Attack/t-SNE Visualization of Node Embeddings (Isolated Nodes Highlighted).png" alt="中毒攻击删除节点" />
  <p><em>图4：Cora数据集删除节点效果</em></p>
</div>

### 5.2 往Cora数据集中随机增加n个节点，可视化增加后模型分类效果

随机增加可以理解为，往特征矩阵与邻接矩阵中添加额外的节点属性，添加节点的特征可以是随机生成的，也可以是平均特征等等；添加节点的邻居可以是以随机抽取m个节点作为其邻居，m在合理情况下应小于等于Cora数据集的平均度值

<div align="center">
  <img src="image/Poison Attack/t-SNE Visualization of Node Embeddings (Injected Nodes Highlighted).png" alt="中毒攻击增加节点" />
  <p><em>图5：Cora数据集增加节点效果</em></p>
</div>

### 5.3 往数据集中添加节点可以扰动模型的分类效果，这就是节点注入攻击

考虑单节点注入攻击，同时该节点可以拥有m个邻居，m为攻击预算，是超参数

#### 5.3.1 当m为随机抽取m个邻居时，可视化攻击后模型分类效果

<div align="center">
  <img src="image/Poison Attack/t-SNE Visualization of Node Embeddings (Injected Node Highlighted-Random).png" alt="中毒攻击单节点注入" />
  <p><em>图6：Cora数据集单节点注入效果</em></p>
</div>

#### 5.3.2 当m为抽取度值最大的m个邻居时，可视化攻击后模型分类效果

<div align="center">
  <img src="image/Poison Attack/t-SNE Visualization of Node Embeddings (Injected Node Highlighted-Max Degree).png" alt="中毒攻击单节点注入" />
  <p><em>图7：Cora数据集单节点注入效果</em></p>
</div>

---

## 6. 逃逸攻击

修改完数据后，不重新训练模型，而是直接将数据送入之前训练好的模型中，观察模型分类结果，此为逃逸攻击，逃逸攻击的数据修改一般针对测试集

### 6.1 随机删除Cora数据集中的n个节点，可视化删除后模型分类效果

<div align="center">
  <img src="image/Escape Attack/t-SNE Visualization of Node Embeddings (Isolated Nodes Highlighted).png" alt="逃逸攻击删除节点" />
  <p><em>图8：Cora数据集删除节点效果</em></p>
</div>

### 6.2 往Cora数据集中随机增加n个节点，可视化增加后模型分类效果

<div align="center">
  <img src="image/Escape Attack/t-SNE Visualization of Node Embeddings (Injected Nodes Highlighted).png" alt="逃逸攻击增加节点" />
  <p><em>图9：Cora数据集增加节点效果</em></p>
</div>

### 6.3 往数据集中添加节点可以扰动模型的分类效果，这就是节点注入攻击

#### 6.3.1 当m为随机抽取m个邻居时，可视化攻击后模型分类效果

<div align="center">
  <img src="image/Escape Attack/t-SNE Visualization of Node Embeddings (Injected Node Highlighted-Random).png" alt="逃逸攻击单节点注入" />
  <p><em>图10：Cora数据集单节点注入效果</em></p>
</div>

#### 6.3.2 当m为抽取度值最大的m个邻居时，可视化攻击后模型分类效果

<div align="center">
  <img src="image/Escape Attack/t-SNE Visualization of Node Embeddings (Injected Node Highlighted-Max Degree).png" alt="逃逸攻击单节点注入" />
  <p><em>图11：Cora数据集单节点注入效果</em></p>
</div>