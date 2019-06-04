---
title: Pelee (NIPS, 2018)
sitemap: true
categories: 计算机视觉
date: 2018-12-04 14:44:50
tags:
- 计算机视觉
- 目标检测
- 论文解读
---

**文章:** Pelee: A Real-Time Object Detection System on Mobile Devices
**作者:** Robert J. Wang, Xiang Li, Charles X. Ling
**备注:** Department of Computer Science University of Western Ontario London, Ontario, Canada


# 核心亮点

# 摘要

近年来, 可在移动设备上运行的卷积神经网络的需求不断增长, 促进了相关高效模型的设计和研究. 目前已经有大量的研究成果发表, 如 MobileNet, ShuffleNet, 以及 MobileNetV2 等等. 但是, 所有的这些模型都严重依赖于深度可分离卷积(depthwise separable convolution), 但是这在大多数深度学习框架中都缺乏有效的实现. 在本文中, 我们提出了一个高效的结构, 命名为 PeleeNet, 它是通过 **传统的卷积** 构建的. 在 ImageNet ILSVRC 2012 数据集上, 本文提出的 PeleeNet 相对于 MobileNet 和 MobileNetV2 来说, 不仅具有更高的准确率, 同时还具有更快的速度(1.8倍). 同时, PeleeNet 的模型大小只有 MobileNet 的 66%. 我们还将 PeleeNet 和 SSD 方法结合起来, 提出了一个实时的目标检测系统, 并将其命名为 Pelee, 在 VOC2007 数据集上达到了 76.4% 的 mAP, 在 COCO 数据集上达到了 22.4 的 mAP, 在 iPone8 手机上达到了 23.6 FPS, 在 NVIDIA TX2 上达到了 125 FPS.

# 介绍

越来越多的研究开始关注在限制内存和计算成本的条件下, 如何构建可以高效运行的神经网络模型. 目前已经有很多创新的模型被提出, 如 MobileNets, ShuffleNet, NASNet-A, MobileNetV2, 但是所有的这些模型都严重依赖于深度可分离卷积(depthwise separable convolution), 但是这种卷积缺乏有效的实现. 同时, 有一些研究会将高效的模型和快速的目标检测算法结合起来(Speed/Accuracy trade-offs). 因此, 本文主要的研究就是设计一个用于图片分类和目标检测任何的高效 CNN 模型, 主要的贡献点有以下几点:

**(1)PeleeNet:**
我们提出了 DenseNet 的一种变体, 并将其命名为 PeleeNet, 它主要是为了在移动设备上运行而设计的. PeleeNet 延续了 DenseNet 的 connectivity pattern 和 一些关键的设计原则. 同时, 它的有些设计也是为了解决有限的内存和算力问题而存在的. 实现表明, PeleeNet 在 ImageNet ILSVRC 2012 上的 top-1 准确率为 72.1% (比 MobileNet 高 1.6%). 同时需要注意, PeleeNet 的模型大小只有 MobileNet 的 66%. PeleeNet 主要有以下几点关键特征:
- **Two-Way Dense Layer:** 受到 GooLeNet 的启发, 我们使用了 2-way dense layer 来获取不同尺寸的感受野. 其中一路使用了 $3\times 3$ 大小的卷积核. 另一路使用了两个 $3\times 3$ 大小的卷积核来学习更大物体的视觉特征. 具体的结构如下图1所示.
<div style="width: 550px; margin: auto">![图1](https://wx2.sinaimg.cn/large/d7b90c85ly1fxuqf7jqf7j213w0gpjtq.jpg)
- **Stem Block:** 受到 Inception-v4 和 DSOD 的启发, 我们在第一层 dense layer 之前设计了一个高效低成本(cost efficient)的 stem block. 该 stem block 的结构如图2所示. 它可以有效的提升特征表达能力, 同时不需要增减太大的计算成本, 要其他方法(增加第一个卷积层的通道数或者增加通道数的增长速度)要好.
<div style="width: 550px; margin: auto">![图2](https://wx4.sinaimg.cn/large/d7b90c85ly1fxuqfuzpnfj218e0ibtb5.jpg)
- **Dynamic Number of Channels in Bottleneck Layer:** 另一个亮点是 bottleneck 层的通道数是根据输入形状变化的, 而不是原始 DenseNet 中固定的 4 倍增长速度. 在 DenseNet 中, 我们观察到, 对于前一个 dense layers, bottlenec 层通道的数量远远大于其输入通道的数量, 这意味着对于这些层, 瓶颈层增加了计算成本, 而不是降低了成本. 为了保持体系结构的一致性, 我们仍然将 bottlenect 层添加到所有的 dense layers 当中, 但是数量是根据输入数据的 shape 动态调整的, 以 **确保通道的数量不超过输入通道的数量.** 实验显示, 和原始的 DenseNet 结构相比, 这个方法可以节省 28.5% 的算力耗费, 但是只会轻微降低最终的结果. 如图3所示
<div style="width: 550px; margin: auto">![图3](https://wx2.sinaimg.cn/large/d7b90c85ly1fxuqgdg1tqj21c70dcn1m.jpg)
- **Transition Layer without Compression:** 我们的实验表明, DenseNet 提出的压缩因子(compression factor)对于特征表达有一定的负面影响. 我们在 transition layers 当中总是保持输出通道的数量和输入通道的数量相同.
- **Composite Function:** 为了提高实际速度, 我们使用传统的 "后激活(conv+bn+relu)" 作为我们的复合函数, 而不是 DenseNet 中使用的预激活(这样会降低准确率). 对于后激活方法来说, 所有的 BN 层都可以与卷积层合并, 从而大大加快了推理的速度. 为了弥补这种变化对精度的负面影响, 我们使用了一种浅而宽的网络结果. 在最后一个 dense block 之后, 我们还添加了一个 $1\times 1$ 的卷积层, 以获得更强的表达能力.

**(2). 我们优化了 SSD 的结构, 使其速度更快, 然后将它与我们的 PeleeNet 相结合.**
我们将结合后的模型称为 Pelee, 该模型达到了 76.4% mAP on VOC 2007, 22.4 mAP on COCO. 为了平衡速度和准确度而提出的改善措施主要如下:
- **Feature Map Selection:** 我们以一种不同于原始 SSD 的方式构建了目标检测网络, 并精心选择了一组 5 个尺度的特征图谱(19, 10, 5, 3, 1). 为了降低计算的复杂度, 我们没有使用 $38\times 38$ 大小的 feature map.
- **Residual Prediction Block:** 我们令特征沿着网络进行传递. 对于每个用于检测的特征图, 我们构建一个残差块, 具体的结构如图4所示.
<div style="width: 550px; margin: auto">![图4](https://wx4.sinaimg.cn/large/d7b90c85ly1fxuqgrwiu3j21ei0jfjuy.jpg)
- **Small Convolutional Kernel for Prediction:** 残差预测块使得我们可以应用 $1\times 1$ 的卷积核来预测类别得分和框的偏移量. 实验表明, 使用 $1\times 1$ 核的模型精度与使用 $3\times 3$ 核的模型精度基本相同. 然而, $1\times 1$ 核的计算成本减少了 21.5%.

**(3).我们在 NVIDIA TX2 嵌入式平台上和 iPhone8 上为不同的高效分类模型和不同的单阶段目标检测方法提供了一个 benchmark test.**

# PeleeNet

An Efficient Feature Extraction Network

## Architecture

我们提出的 PeleeNet 的架构如表1所示. 整个网络由一个 stem block 和四个阶段的特征提取器构成(four stages of feature extractor). 除了最后一个阶段外, 每个阶段的最后一层是步长为2的平均池化层. 四阶段(不算 stem)结构是大型模型设计中常用的结构形式. ShuffleNet 使用了一个三阶段的结构, 并在每个阶段的开始将 feature map 的大小缩小. 虽然这可以有效的降低计算成本, 但我们认为, 早期阶段的特征对于视觉任务非常重要, 过早的减小特征图的大小会损害表征能力. 因此, 我们仍然保持四阶段结构. 前两个阶段的层数会专门控制在一个可介绍的范围内.

<div style="width: 550px; margin: auto">![表1](https://wx1.sinaimg.cn/large/d7b90c85ly1fxuqheoxfxj21qa0rstez.jpg)

## Ablation Study

**Dataset**

自定义了 Stanford Dogs 数据集用来进行消融实验(从 ILSVRC 2012 数据集的子集中创建)
- 类别数: 120
- 训练集图片数: 150466
- 验证集图片数: 6000

**Effects of Various Design Choices on the Performance:**

我们构建了一个类似于 DenseNet 的网络, 并将其命名为 DenseNet-41, 作为我们的 baseline 模型. 该模型和原始的 DenseNet 模型有两点不同. 第一, 首层 conv layer 参数不同, 其通道数设定为 24 而不是 64, 核的大小从 $7\times 7$ 改变到 $3\times 3$. 第二点不同是, 调整了每个 dense block 中的层的数量以满足算力限制.

我们在这部分的模型都是有 batch size 为 256 的 PyTorch 进行 120 epochs 的训练. 我们遵循了 ResNet 的大多数训练设置和超参数. 表2显示了各种设计选择对性能的影响. 可以看到, 在综合了所有这些设计选择以后, Peleenet 的准确率达到了 79.25%, 比 DenseNet-41 的准确率高 4.23%. 并且计算成本更低.

<div style="width: 550px; margin: auto">![表2](https://wx3.sinaimg.cn/large/d7b90c85ly1fxuqhyoagcj21b60h0dik.jpg)


## Results on ImageNet 2012

Cosine Learning Rate Annealing ($t \leq 120$)

$$0.5 \times lr \times (cos(\pi \times t / 120) + 1)$$

<div style="width: 550px; margin: auto">![表3](https://wx1.sinaimg.cn/large/d7b90c85ly1fxuqiekk67j215l0chwhc.jpg)

## Speed on Real Devices


<div style="width: 550px; margin: auto">![表4](https://wx4.sinaimg.cn/large/d7b90c85ly1fxuqira5vlj21cs0exq6v.jpg)

**使用 FP16 而不是 FP32 是一个常用的在 inference 阶段的加速方法.** 但是基于 depthwise separable convolution 的网络却很难从 TX2 的 half-precision(FP16)中获益, 如图5所示.

<div style="width: 550px; margin: auto">![图5](https://wx2.sinaimg.cn/large/d7b90c85ly1fxuqj67xn5j21p40oi43r.jpg)

<div style="width: 550px; margin: auto">![表5](https://wx1.sinaimg.cn/large/d7b90c85ly1fxuqjow0dvj21cc0c4ju4.jpg)

# Pelee

A Real-Time Object Detection System

## Overview

本小节介绍了我们的目标检测系统以及对 SSD 做出的一些优化. 我们的优化目的主要是在提升速度的同时还要保持一定的精确度. 除了我们上一节提到的特征提取网络以外, **我们还构建与原始 SSD 不同的目标检测网络, 并精心选择了一组 5 个尺度的特征图. 同时, 对于每一个用于检测的特征图谱, 我们在进行预测之前建立了一个残差块(如图4). 我们还使用小卷积核来预测对象类别和边界框位置, 以降低计算成本. 此外, 我们使用了非常不同的训练超参数.** 尽管这些贡献单独看起来影响很小, 但是我们注意到最终的模型在 PASCAL VOC 2007 上达到了 70.9% 的 mAP, 在 MS COCO 上实现了 22.4 的 mAP.

在我们的模型中我们使用了 5 种尺寸的特征图谱: 19, 10, 5, 3, 1. 我们没有使用 38 大小的特征图谱是为了平衡速度与精度. 19 大小的特征图谱使用了两种尺寸的 default boxes, 其他 4 个特征图谱使用了一种尺寸的 default box. Speed/Accuracy Trade-offs 论文中在使用 SSD 与 MobileNet 结合时, 也没有使用 38 尺寸的特征图谱. 但是, 他们额外添加了一个 $2\times 2$ 的特征图谱来保留6个尺寸的特征图谱进行预测, 这与我们的解决方案不多.

<div style="width: 550px; margin: auto">![表6](https://wx4.sinaimg.cn/large/d7b90c85ly1fxuqk52n9ij21bw0b00us.jpg)

## Results on VOC 2007

我们的目标检测模型是基于 SSD 的源码实现的(Caffe). batch-size 为 32, 初始的 learning rate 为 0.005, 然后在 80k 和 100k 次迭代时降低 10 倍. 总的迭代数是 120K.

**Effects of Various Design Choices**
表7显示了不同设计选择对性能的影响. 我们可以看到残差预测模块可以有效的提升准确率. 有残差预测模块的模型比无残差预测模块的模型精度高 2.2%. 使用 $1\times 1$ 卷积核进行预测的模型和使用 $3\times 3$ 的模型的精度几乎相同. 但是 $1\times 1$ 的内核减少了 21.5% 的计算成本和 33.9% 的模型大小.

<div style="width: 550px; margin: auto">![表7](https://wx2.sinaimg.cn/large/d7b90c85ly1fxuqknm2utj21as0ci0v5.jpg)

**Comparison with Other Frameworks**
表8显示了我们的模型与其他不同模型的对比

<div style="width: 550px; margin: auto">![表8](https://wx3.sinaimg.cn/large/d7b90c85ly1fxuqkyxbklj21eo0egtd3.jpg)

## Results on COCO
<div style="width: 550px; margin: auto">![表9](https://wx3.sinaimg.cn/large/d7b90c85ly1fxuqlg4xbhj219t0avdil.jpg)

## Speed on Real Devices
<div style="width: 550px; margin: auto">![表10](https://wx1.sinaimg.cn/large/d7b90c85ly1fxuqlvb4ktj21cj0b80v1.jpg)
