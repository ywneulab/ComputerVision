---
title: R-FCN (NIPS, 2016)
sitemap: true
categories: 其它
date: 2018-10-24 14:58:30
tags:
- 计算机视觉
- 目标检测
- 论文解读
---

**文章:** R-FCN: Object Detection via Region-based Fully Convolutional Networks

# 核心亮点

- 全卷积网络怎分类任务上表现较好, 但是在目标检测任务往往精度不行, 这是因为在一般情况下, 分类任务具有平移不变性, 而检测任务却要求对目标的平移做出正确响应. 在Faster RCNN类的方法中RoI pooling之前都是卷积, 具有平移不变性, 但是一旦经过RoI pooling 之后, 后面的网络结果就不再具备平移不变性了. 因此, 本文了position sensitive score map来将目标位置的信息融合进RoI
- 对于Faster RCNN等基于感兴趣区域的检测方法来说, 实际上是分成了几个subnetwork, 第一个用来在整张图上做比较耗时的conv, 这些操作与region无关, 是计算共享的. 第二个subnetwork是用来产生候选区域(如RPN), 第三个subnetwork是用来分类或者进一步对box进行回归的, 这个subnetwork和region是有关系的, 衔接在这个subnetwork和前两个subnework中间的就是RoI pooling. 本文与FasterRCNN相比(前91层共享, RoI pooling之后, 后10层不共享)不同, 将ResNet所有的101层都放在的前面共享的subnetwork中, 最后用来进行prediction的卷积只有1层, 大大减少了计算量.
- 最终, 从实验结果来看, 本文提出的用于目标检测任务的全卷积网络在精度上仍然不如标准的Faster RCNN, 但是在test time上要好于FasterRCNN (因为有大部分层都变成参数共享的了).


# 论文细节

## 摘要

本文提出了一个用于精确高效进行物体检测的基于区域的全卷积网络. 在Fast/Faster RCNN中, 使用了计算成本很大的子网络来提取候选区域, 与之相比, 本文的基于区域的检测器是全卷积的, 因此几乎所有的计算都可以共享. 为了达到这个目标, 本文提出了一个位置敏感的score maps来解决图像分类问题中的平移不变性和物体检测中的平移可变性之间的鸿沟. 因此, 本文的方法可以很自然的使用全卷积图像分类网络, 比如ResNet. 本文的方法在检测阶段的速度大约为FasterRCNN的2.5~20倍.


## 介绍

最近的图像分类网络如 ResNet 和 GoogleNets 都是通过全卷积网络设计的.(只有最后一层是全连接的, 并且该层会在进行目标检测任务时被去掉).

我们讨论了之前提到的不自然的设计是由于图像分类的平移不变性和目标检测的平移可变性之间的鸿沟造成的. 一方面, 在图像级别上的分类任务倾向于平移不变性, 因此, 深度卷积网络的结构会尽可能是使得结果的检测具有一定的平移不变性. 另一方面, 物体检测任务需要坐标表示, 这是一种平移可变性的表现. 为了解决这个问题, ResNet的检测方法是插入了一个RoI pooling 层--该层可以用来打破神经网络原有的平移不变性, RoI后续的卷积层在对不同区域进行评价时, 实际上已经不再具有平移不变性(出现在天空位置时, 行人的预测概率较低).  但是, 这种RoI的设计模型牺牲了训练和测试的高效性, 因为它引入了大量额外的快计算.

在这篇文章中, 我们构建了一个用于物体检测任务的基于区域的全卷积网络的框架模型. 我们的网络模型由共享的全卷积网络组成. 为了与FCN的平移可变性相协调, 我们构建了一个位置敏感型的score maps. 每一个score maps都根据相对空间位置将位置信息进行了编码. 在FCN之上, 我们添加了一个位置敏感的RoI pooling layer 用于指导从score maps中获取的信息, 并且在其后没有再使用带权重的层(全连接or全卷积).

## Our approach

**Overview:** 和RCNN一样, 本文使用了 two-stage 的物体检测策略, 包括:1) 区域推荐, 2)区域分类. 本文通过RPN网络来提取候选框(RPN网络本身也是一个全卷积网络). 和Faster RCNN一样, 我们令RPN和R-FCN网络的权重参数共享. 下面的图2显示了这个系统的整体视图 (位于上方的RPN网络和位于下方的R-FCN网络使用的卷积图谱都是来自同一段卷积网络, 因此参数共享):

![](https://wx4.sinaimg.cn/large/d7b90c85ly1fwlogvettvj21ae0llapa.jpg)

当给定感兴趣区域(proposals regions)后, R-FCN网络会用于对RoIs分类. 在R-FCN中, 所有的可学习的参数都是卷及参数, 并且都是在同一张图片上计算得到的. 最后一层卷积层对于每一类都会输出 $k^2$ 个位置敏感的socre maps, 因此总共有对于具有 $C$ 个物体类别来说, 输出层具有 $k^2(C+1)$ 个通道. $k^2$ 主要是根据用于描述相关位置的 $k\times k$ 的空间网格来决定. 比如, 当 $k \times k = 3\times 3$ 时, 对于每一个物体类别都会有9个score maps 根据下列情况进行编码: {top-left, top-center, top-right, ... , bottom-right}.

R-FCN最后会有一个位置敏感的RoI pooling层, 这一层会将最后一层卷积层的输出全部整合, 并为每个RoI生成对应的score. 和之前的工作不同, 本文的位置敏感型的RoI层会执行selective pooling, 并且每一个$k\times k$ bin 都会整个仅一个score map. 下面的图展示了示例:

![](https://wx1.sinaimg.cn/large/d7b90c85ly1fwlpc7oinsj215c0lbtq5.jpg)

![](https://wx4.sinaimg.cn/large/d7b90c85ly1fwlpckwfuej215a0k3nej.jpg)

**Backbone architecture:** R-FCN的网络主体是基于ResNet-101的, 我们将平均池化层和fc层移除, 只使用前面的卷积网络来计算特征图谱. 由于卷积段输出的维度为2048, 因此我们使用 $1\times 1$ 的卷积层来进行降维, 使其维度变成1024.  然后我们使用通道数为 $k^2(C+1)$ 的卷积层来生成score maps.

**Position-sensitive score maps & Position-sentitive RoI pooling.** 为了具体的对每个RoI的位置信息进行编码, 我们将每一个RoI划分成 $k\times k$ 大小的普通网格, 然后, 最后一层卷积层对每一个类别都会生成 $k^2$ score maps, 在第 $(i,j)$ 个bin中 $(0 \leq i,j \leq k-1)$. 我们定义位置敏感的RoI pooling计算操作如下所示:

$$r_c(i,j | \theta) = \sum_{(x,y)\in bin(i,j)} z_{i,j,c}(x+x_0, y+y_0 | \theta)/n$$
