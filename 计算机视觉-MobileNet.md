---
title: MobileNet
sitemap: true
categories: 计算机视觉
date: 2018-09-22 16:54:24
tags:
- 计算机视觉
- 网络结构
- 论文解读
---

**文章:** MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
**作者:** Andrew G.Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam
**机构:** Google

# 摘要

我们提出了 **一类** 针对移动和嵌入式视觉应用的有效模型, 称之为 MobileNets. MobileNets 基于一种流线型结构(streamlined architecture), 并且使用 **深度可分离卷积(depthwise separable convolutions)** 来构建轻量级的深度神经网络. 我们引入了两个简单的全局超参数, 它们可以有效的在速度和精度之间权衡. 这些超参数允许模型构建器根据问题的约束为其应用程序选择适当大小的模型. 我们在资源分配和模型精度的权衡上进行了大量的实验, 并与其它流行的 ImageNet 分类模型相比, 表现出了较强的性能. 然后, 我们演示了 MobileNets 在各个领域的有效性, 包括目标检测, 细粒度分类, 人脸属性识别和大规模地理定位等.

<div style="width: 550px; margin: auto">![图1](https://wx3.sinaimg.cn/large/d7b90c85ly1g1iibxslymj214i0h9an9.jpg)

# 介绍

本文介绍了一种高效的网络架构和两个超参数, 以便构建非常小的, 低延迟的模型, 这些模型可以很容易的匹配移动和嵌入式视觉应用程序的设计需求.

# 前人工作
SqueezeNet, Factorized CNN, Quantized CNN.

MobileNets 的目的是减少模型大小的同时加速模型速度.(有些 papers 只关注模型大小)

# MobileNet Architecture

本节我们将介绍 MobileNet 中的核心网络层, 即深度可分离卷积层(depthwise separable). 然后会介绍 MobileNet 的网络结构和两个超参数: width multiplier, resolution multiplier.

## Depthwise Separable Convolution

深度可分离卷积实际上是一个卷积分解的形式, 它将标准的卷积分解为深度卷积(depthwise convolution)和 $1\times 1$ 卷积(也叫点卷积). 对于 MobileNets 来说, 深度卷积会对 **每个输入通道** 应用一个单独的过滤器(filter). 然后点卷积(pointwise convolution)会使用 $1\times 1$ 的卷积来融合深度卷积层的输出. 一个标准的卷积层会在使用 filters 的同时将输入融合启动, 并形成一个新的输出. 但是深度可分离卷积将这个过程分为了两层, 一层用于 filtering, 另一层用于 combining. 这种分解可以大大减少计算量和模型大小. 图2(a)代表标准卷积, (b)代表深度卷积, (c)代表点卷积.

<div style="width: 550px; margin: auto">![图2,图3,表2](https://wx4.sinaimg.cn/large/d7b90c85ly1g1iieb7cquj219m0to44t.jpg)

如果标准卷积接受一个尺度为 $D_F\times D_F\times M$ 的特征图谱 **F**, 输出一个尺度为 $D_G\times D_G\times N$ 的特征图谱 **G**, 那么 $M$ 就是输入图谱的通道数(input depth), $N$ 就是核的数量, 也是输出图谱的通道数. 该卷积层的参数量为: $D_K\times D_K\times M\times N$, $D_K$ 代表卷积核的大小. 标准卷积核的计算成本(乘法次数)为 $D_K\times D_K\times M\times N\times D_F\times D_F$, 即: $卷积乘法次数 = 核尺寸\times 输入通道数\times 输出通道数\times 输入图谱尺寸$.

深度可分离卷积由两个网络层组成: depthwise conv 和 pointwise conv. 前者对每个输入通道应用 filter, 后者用于组合前者的输出. MobileNets 在这两个网络层中都使用了 BN 和 ReLU.

Depthwise Conv 可以用公式表达如下(with one filter per input channel):

$$\hat G_{k, l, m} = \sum_{i, j}\hat K_{i,j,m} \cdot F_{k+i-1, l+j-1, m}$$

Depthwise Conv 的计算成本(乘法次数)为 $D_K\times D_K\times M\times D_F\times D_F$, 可以看出, 由于 Depthwise Conv 是对每个通道单独进行 filter, 所以其乘法次数无需计算输入图谱的通道数.

虽然 Depthwise Conv 可以大大节省参数量和计算成本, 但是它仅仅对每个通道进行的 filter 操作, 而没有将这些通道结合起来, 因此, 我们添加额外的一层 $1\times 1$ 的卷积网络来计算这些 Depthwise Conv 输出值的线性组合. Depthwise Conv 和 Pointwise Conv 的联合使用就被称为深度可分离卷积, 他们计算成本为 $D_K\cdot D_K\cdot M\cdot D_F\cdot D_F + M\cdot N\cdot D_F\cdot D_F$, 其中, $D_K$ 代表核的大小, $N, M$ 分别代表输入输出通道数, $D_F$ 代表输入图谱的大小.
最终, 计算成本的降低程度如下所示:

$$\frac{D_K\cdot D_K\cdot M\cdot D_F\cdot D_F + M\cdot N\cdot \cdot D_F\cdot D_F}{D_K\cdot D_K\cdot M\cdot N\cdot D_F\cdot D_F} = \frac{1}{N} + \frac{1}{D_K^2}$$

MobileNet 使用了 $3\times 3$ 的深度可分离卷积, 它的计算量比标准卷积低了 8~9 倍, 并且精度只降低了一点.
空间维度的卷积分解(InceptionV3 中使用的)并不能节省很多额外的计算, 因为只有很少的计算花费在深度卷积上.

## Network Structure and Training

MobileNet 除了第一层是一个完整的卷积之外, 其余层都是建立在深度可分离卷积层之上的. 通过这种简单的卷积结构, 我们可以很容易的探索网络的拓扑结构, 从而找到一个较好的网络模型. MobileNet 的网络结构如表1所示. **所有的网络层后面都具有 BatchNorm 层和 ReLU 层, 网络最后的分类层由 GAP + FC + Sofmax 组成.** 图3显示了标准卷积以及深度可分离卷积之间的区别. MobileNet 中的下采样操作同时通过步长为2的深度卷积(Conv dw/s2)完成的. 如果将 Depthwise Conv 和 Pointwise Conv 算作独立的层, 那么 MobileNet 的深度就为 28 层.

仅仅根据较少的乘法加法操作(Mult-Adds)来确定网络的结构是不够的, **同样重要的是确保这些操作能够有效的实现.** 举例来说, 除非具有非常高的稀疏程度, 否则非结构化的稀疏矩阵运算通常不会比密集矩阵的运算速度快. **我们的模型结构几乎将所有的计算都放在了密集的 $1\times 1$ 卷积当中.** 这可以通过高度优化的通用矩阵乘法(GEMM)函数来实现. **卷积操作通常由 GEMM 实现**, 但是需要在内存进行名为 im2col 的初始重新排序, 以便将其映射到 GEMM 当中. **而 $1\times 1$ 卷积不需要在内存中重新排序, 可以直接使用 GEMM 实现. 并且, MobileNets 95% 的计算时间都花费在 $1\times 1$ 卷积中, 同时其 75% 的参数都存在于 $1\times 1$ 卷积中, 具体如表2所示**

<div style="width: 550px; margin: auto">![表1-5](https://wx1.sinaimg.cn/large/d7b90c85ly1g1iigzgq6jj217p0osqda.jpg)

相比于训练大模型, 我们在训练小模型的时候会使用更少的正则化技术和数据增广技术, **因为小模型通常不太容易过拟合(参数少)**. 另外, 由于 Depthwise Conv 的参数非常少, 因此在 Depthwise Conv 上需要很少或者不添加 weight decay(L2), 这一点是非常重要的.

## Width Multiplier: Thinner Models

尽管 MobileNet 的网络结构已经非常小并且延迟性很低, 但是对于某些特殊的应用场景我们也许需要使用模型更小, 更快. 为此, 我们引入了一个非常简单的超参数 $\alpha$, 称之为 width multiplier. 它的作用是 **使网络在每一层均匀的变薄**. 当给定一个网络层和 width multiplier $\alpha$ 时, 该网络层的输入通道数 $M$ 会变成 $\alpha M$, 输出通道数 $N$ 会变成 $alpha N$. 如此一来, 深度可分离卷积的计算成本就变成了:

$$D_K\cdot D_K\cdot \alpha M\cdot D_F\cdot D_F + \alpha M\cdot \alpha N \cdot D_F\cdot D_F$$

当 $\alpha = 1$ 时, MobileNet 不发生变化, 当 $alpha < 1$ 时, MobileNet 的参数量和计算量会减少, $\alpha$ 的取值一般为 1, 0.75, 0.5 或 0.25. Width multiplier 可以使得网络的计算成本和参数量大约降低 $\alpha^2$.

## Resolution Multiplier: Reduced Representation

超参数 Resolution Multiplier($\rho$) 会用在 input image 和每一层的特征图谱上, 会将 resolution 降低相同的比例. 在实际使用中, 我们是通过设置输入图谱的分辨率来隐式的使用 $rho$ 的. 当同时使用 $alpha$ 和 $rho$ 以后, 深度可分解卷积的计算成本就变成了:

$$D_K\cdot D_K\cdot \alpha M\cdot \rho D_F\cdot \rho D_F + \alpha M\cdot \alpha N \cdot \rho D_F\cdot \rho D_F$$

Resolution multiplier 可以使计算量降低 $\rho^2$.

表3展示了标准的 $3\times 3$ 卷积层在 $14\times 14\times 512$ 的特征图谱上的乘法加法操作和参数量与深度可分离卷积之间的对比.

# Experiments

Model Choices
Model Shrinking Hyperparameters


<div style="width: 550px; margin: auto">![表6-14](https://wx4.sinaimg.cn/large/d7b90c85ly1g1iillqjl8j21r00u07op.jpg)
<div style="width: 550px; margin: auto">![图4,图5](https://wx1.sinaimg.cn/large/d7b90c85ly1g1iinaxqhfj21750g8q65.jpg)

<span id = "简述 MobileNet 的原理">
# 简述 MobileNet 的原理

MobileNet 的设计原则是要构建出模型 size 小, 并且执行速度的卷积网络. 它的整体结构框架和 AlexNet 以及 VGGNet 类似, 都是通过不断堆叠卷积层的方式来构建深层的卷积神经网络. 但与传统卷积网络不同的是, MobileNet 在初第一层使用了标准的卷积层之外, 其余的卷积层都是基于深度可分离卷积构建的. 具体来说, 标准的卷积层在进行操作时, 包括 filter 和 combining 两步, 而深度可分离卷积将这两步分成两个网络层执行, 第一层是 Depthwise Convolution, 它是对输入图谱的每一个通道分别进行操作, 因此它的计算量只与 **核的大小, 输入图谱的尺寸, 以及输出图谱的通道数有关, 与输入图谱的通道数无关.** 然后, 第二层是一个 $1\times 1$ 的卷积层, 它负责融合前一层输出的特征图谱, 由于它的核尺寸为 1, 因此它的计算量只与 **输入图谱的通道数, 输出图谱的尺寸和通道数有关.** 由于在卷积网络中, 通常 **特征图谱的通道数要远远大于特征图谱的尺寸, 因此, 深度可分离卷积的主要计算量(95%)都集中在 $1\times 1$ 的卷积层.** 并且当卷积层使用 im2col+ GEMM 的方式实现时, $1\times 1$ 的卷积层不需要在内存中重新排序, 因此它的实际执行速度很快. 最后, 原文还给出了两个超参数来进一步压缩模型大小, 分别 width multiplier $\alpha$ 和 resolution multiplier $rho$, 前者通过控制特征图谱的通道数来实现, 后者通过通过输入图片的尺寸来实现.

解释为何 $1\times 1$ 卷积层不需要重新排序: [卷积层底层是如何实现的](../深度学习-各种网络层/#卷积层底层是如何实现的)
