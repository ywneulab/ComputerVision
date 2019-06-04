---
title: DenseNet (CVPR, 2017)
sitemap: true
date: 2018-08-14 21:16:47
categories: 计算机视觉
tags:
- 计算机视觉
- 论文解读
- 网络模型
---

**文章:** Densely Connected Convolutional Networks
**作者:** Gao Huang, Zhuang Liu, Laurens van der Maaten
**机构:** Cornell University

# 摘要

<span id = "简述 DenseNet 的原理">
# 简述 DenseNet 的原理

在训练特别深层的网络时, 随着深度的增加, 梯度消失的问题会越来越明显, 对此, ResNet 给出了一个很好的解决方案, 那就是在接近输入层的网络中添加一个短接路径到靠近输出层的网络. DenseNet 也延续了这个思路, 与 ResNet 不同的是, 他采用了一个更暴力的方式, 就是将所有网络层都连接起来, 具体来说, 就说每一层的输入会来自于 **前面所有层的输出**(这些层的特征图谱大小时相同的, 因此可以在 Channel 维度上进行叠加). 如果假设有网络的层数是 $L$, 那么 DenseNet 就会有 $\frac{L(L+1)}{2}$ 个短接路径. 在传统的卷积神经网络中, 通常会利用 Pooling 层或者卷积层来降低特征图谱的大小, 而 DenseNet 的密集连接需要特征图大小保持一致, 因此, 为了解决这个问题,  DenseNet 将网络分成了若干个 DenseBlock, DenseBlock 中的网络层都具有相同大小的特征图谱, 因此可以直接使用密集连接的方式进行连接. 在 DenseBlock 中, 每一层网络输出的特征图谱的通道数是通过一个超参数增长率 $k$ 来决定的, 这个 $k$ 可以设定的比较小, 比如32, 虽然每一层的输出图谱通道数较低, 但是每一层的输入是综合前面所有层的输出的, 因此具有特征重用的效果. 另外, 由于深层的网络层输入非常大, 因此 DenseBlock 内部会采用 bottleneck 来减少计算量, 主要是在原来的 $3 \times 3$ 卷积层之前添加 $1\times 1$ 的卷积层, 变成 **BN + ReLU + 1x1 Conv + BN + ReLU + 3x3 Conv** 的结构(DenseNet-B), $1\times 1$ 卷积会将 $l\times k$ 的通道数降低成 $4\times k$ 的通道数, 从而提升计算效率.
对于相邻的具有不同特征图谱大小的 DenseBlock, DenseNet 采用的 Transition Layers 进行连接, 它的结构是 **BN + ReLU + 1x1 Conv + 2x2 AvgPooling.** 它主要作用是降低特征图谱的尺寸大小, 另外还有一个作用就是压缩模型, 假定 Transition 层的前一个 DenseBlock 得到的特征图谱的数量为 $m$, 它会根据压缩系数 $\theta \in (0, 1]$ 的值来决定输出的特征图谱的数量(通过卷积层), 输出的图谱通道数量为 $\lfloor \theta m \floor$ 当 $\theta = 1$ 时, 相等于没有压缩, 文中使用 $\theta = 0.5$(DenseNet-C). DenseNet 的整体网络结构设计也是遵循的经典的五段式, 其中第一段是有传统 $7\times 7$ 卷积构成的 Stem, 后面四段是层数不同的 DenseBlock-BC, 最后是 GAP+FC+Softmax 的分类层.
