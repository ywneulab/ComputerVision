---
title: InceptionV4 and Inception-ResNet
sitemap: true
categories: 计算机视觉
date: 2018-11-20 15:05:08
tags:
- 计算机视觉
- 网络结构
- 论文解读
---

**文章:**
Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning
**作者:** Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi

<span id = "简述 InceptionV4 做了哪些改进">
# 简述 InceptionV4 做了哪些改进
<div style="width: 550px; margin: auto">![InceptionV4](https://wx2.sinaimg.cn/large/d7b90c85ly1g1g6on0w08j21hc0u0b29.jpg)

InceptionV4 使用了更复杂的结构重新设计了 Inception 模型中的每一个模块. 包括 Stem 模块, 三种不同的 Inception 模块以及两种不同的 Reduction 模块. 每一个模块的具体参数设置均不太一样, 但是整体来说都遵循的卷积分解和空间聚合的思想.

<span id = "简述 Inception-Resnet-v1 做了哪些改进">
# 简述 Inception-Resnet-v1 做了哪些改进

Inception ResNet v1 网络主要被用来与 Inception v3 模型性能进行比较, 因此它所用的 Inception 子网络的计算相对常规模块有所减少, 这是为了保证使得它的整体计算和内存消耗与 Inception v3 近似, 如此才能保证公平性. 具体来说, Inception ResNet v1 网络主要讲 ResNet 中的残差思想用到了 Inception 模块当中, 对于每一种不太的 Inception 模块, 都添加了一个短接连接来发挥残差模型的优势.

<div style="width: 550px; margin: auto">![InceptionResNetV1](https://wx1.sinaimg.cn/large/d7b90c85ly1g1g6orzjmmj21hc0u0b29.jpg)

<span id = "简述 Inception-ResNet-v2 做了哪些改进">
# 简述 Inception-ResNet-v2 做了哪些改进

Inception ResNet v2 主要被设计来探索残差模块用于 Inception 网络时所尽可能带来的性能提升. 因此它是论文给出的最终性能最高的网络设计方案, 它和 Inception ResNet v1 的不同主要有两点, 第一是使用了 InceptionV4 中的更复杂的 Stem 结构, 第二是对于每一个 Inception 模块, 其空间聚合的维度都有所提升. 其模型结构如下所示:

<div style="width: 550px; margin: auto">![InceptionResNetV2](https://wx1.sinaimg.cn/large/d7b90c85ly1g1g6ox6512j21hc0u0e81.jpg)
