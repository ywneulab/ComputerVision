---
title: Speed Accuracy TradeOffs (CVPR, 2017)
sitemap: true
categories: 计算机视觉
date: 2018-11-10 16:27:03
tags:
- 计算机视觉
- 目标检测
- 论文解读
---

**文章:** Speed/accuracy trade-offs for modern convolutional object detectors
**作者:** Jonathan Huang, Vivek Rathod, Chen Sun, Menglong Zhu, Anoop Korattikara, Alireza Fathi, Ian Fischer, Zbigniew Wojna, Yang Song, Sergio Guadarrama, Kevin Murphy
**备注:** Google


# 核心亮点

**本文实现了一个灵活统一的目标检测框架, 并对三个主流的目标检测模型做了详细客观的分析和讨论**
通过该框架, 本文对目前主流的各个模型(Faster, R-FCN, SSD)影响精确度和速度的各个因素展开了详细的分析和讨论, 以此希望能够帮助从业者在面对真实应用场景时, 能够选择适当的模型来解决问题. 同时, 本文还发现了一些新的现象(减少 proposals box 的数量), 使得可以在保持精度的前提下, 提升模型的速度.

<div style="width: 600px; margin: auto">![](https://wx1.sinaimg.cn/large/d7b90c85ly1fx470hjw5yj213t0o40yg.jpg)

# 论文细节


## 摘要

本篇文章的目的主要是为在给定平台和应用环境下选择一个合适的目标检测模型提供指导, 即要达到合适的 speed/memory/accuracy 平衡. 为此, 我们研究了多种方法来平衡现代卷积网络检测模型的速度, 准确度, 内存消耗等. 近年来有大量成功的检测模型被提出, 但是, 我们很难直接将这些模型互相比较, 因为它们采用的特征提取器不同(eg, VGG, ResNet), 采用的图片尺寸不同, 硬件设备和软件平台也不同. 因此, 本文提供了一个基于 FasterRCNN, R-FCN, SSD 的统一实现版本, 我们称之为- meta-architectures, 然后通过使用不同的特征提取器, 不同的关键参数来跟踪这些模型之间的差异. 最终的模型有两个极端, 一是极度关注速度, 要求最终的模型可以运行在移动设备上, 二是极度关注准确度, 要求能够在 COCO 数据集上达到最高的分数.

## 介绍

目前有众多成功的模型, 但是却很难决定哪种场景下使用哪种模型, mAP 评价标准并不能反映所有问题, 还需要同时考虑模型的运行速度和内存消耗.
目前, 只有很少的模型讨论的运行速度(R-FCN, SSD, YOLO), 但是它们大多都只是声称它们达到了某个 fps, 并没有深入的展开关于速度和精度之间的讨论.
在这篇文章中, 我们希望以一种详尽而公平的方式来探讨这些模型的速度的精度之间的平衡关系. 在评估时, 我们不使用任何 tricks(ensembling, multi-crop, flipping等等), 仅仅评估单一模型的性能, 对于时间的测量, 我们仅关注预测阶段的运行时间, 而不在意训练时长.
本文的贡献主要有以下几点:
- 提供了一个关于现代卷积检测系统的具体调研报告, 并阐述了这些先进模型在设计模式上共有的通性.
- 用 TensorFlow 实现了一个灵活统一的检测框架 meta-architectures, 包含 FasterRCNN, R-FCN 和 SSD 三种模型
- 本文发现, 通过使用较少的候选区域框可以大大提高 FasterRCNN 的检测速度, 并且在精度上不会有太大损失. 同时, 我们还发现 SSDs 的性能表现在面对不同的特征提取器时, 不像 FasterRCNN 和 R-FCN 那么敏感. 并且我们在 acc/speed 曲线上确定了 sweet spots, 这些点上的 acc 只有在牺牲 speed 的情况下才能够提升.
- 我们尝试了一些以前从未出现过的 meta-architecture 和 feature-extractor 的结合方式, 并讨论了如何利用这些方式来训练 winning entry of the 2016 COCO object detection challenge.

## Meta-architectures

在我们的文章中, 我们主要讨论三种主流模型: SSD, FasterRCNN 和 R-FCN. 在这三个模型的原文中各自使用了特定的特征提取器(eg, VGG, ResNet). 现在我们将模型和特征提取器解耦, 重新审视这些模型结构.

<div style="width: 600px; margin: auto">![](https://wx2.sinaimg.cn/large/d7b90c85ly1fx494ceorlj21kw0di0yt.jpg)

### SSD

将画框和分类预测同时进行, 代表一众 one-stage 检测方法

### FasterRCNN

FasterRCNN 是自 2015 年以来最主流的 two-stage 目标检测模型, 它首先提出了 RPN 网络, 使得候选框推荐的步骤可以整合到神经网络中去, FasterRCNN 也衍生出了多种的版本, 代表着经典的 two-stage 模型.

### R-FCN
尽管 FasterRCNN 比 FastRCNN 快了一个数量级, 但是相对于 one-stage 方法, 它仍然很慢. 为此, 有人提出了 R-FCN 检测模型, 它和 FasterRCNN 模型类似, 但是将候选框的划取阶段移到了网络模型的最后一层, 使得有更多的卷积层可以共享计算结果, 同时还提出了 PSRoI(position-sensitive), 大大加快了模型的运算速度.

## Experimental setup

各个模型的实现在所用框架, 优化程度, 数据集等都有所不同, 因此, 单单比较 COCO 或 ImageNet 的 mAP 是不全面的. 因此, 为了更好地比较各个模型之间的差异, 我们用 TensorFLow 实现了一个目标检测框架, 从而可以让我们较为客观公平的进行对比.

### Architectural configuration

**Feature extractors:** VGG-16, ResNet-101, Inception v2, Inception v3, Inception ResNet v2, MobileNet.
对于 FasterRCNN 和 R-FCN 来说, 我们需要确定使用特征提取器的哪一层卷积特征图谱来预测候选区域框. 我本文的实验中, 我们尽可能的使用原文中的设置, 如果原文没有提到的, 我们则尽可能的选择相类似的设置.
在 SSD 中, 因为使用了多个不同尺度的特征图谱来预测 box 的位置和分类, 因此, 特征图谱的选择是至关重要的. 在 VGG 中, 原文使用了 conv4_3, fc7, 以及后续的几层卷积层, 与原文不同的是, 我们在每一个附加层之后都使用了 BN 层.

**Number of proposals**
FasterRCNN & R-FCN : 10~300 (trade-off)

**Output stride setting for Resnet and Inception ResNet**
采用stride 16, 将 conv5_1 的stride从2变为1, 并在后面使用 Atrous 卷积(Dilation 卷积) 来弥补缩小的感受野. 另外, 通过改变 conv4_1
 的stride, 还测试了 stride 8 的情况. 相比于 stride 16, stride 8 的 mAP 提升了 5%, 但是运行时间也变慢了 63%.

 **Matching**
 同样采用原文推荐的参数设置来将候选框与真实框匹配.

 **Box encoding:**
 与原文相同:

$$(b_a;a) = [10\cdot \frac{x_c}{w_a}, 10\cdot \frac{y_c}{h_a}, 5\cdot \log w,  5\cdot \log h]$$

 需要注意的是, 标量 10 和 5 在原文的代码实现中都有使用, 即使在原文中没有提及.

 **Location loss:** Smooth L1

 **Input size configuration:** M=300 / 600.

 **Training and hyperparameter tuning;** 对于 FasterRCNN 和 R-FCN, 我们用 TF 的 `crop_and_resize` 操作来代替 RoIPooling 和 PSRoIPooling, 该操作是利用双线性插值进行反向投影的, 其求导机制和原文中的类似.

**Benchmarking procedure:** 32GB RAM, Intex Xeon E5-1650 v2 processor, Nvidia GTX Titan X.

下面的表2总结了本文使用的特征提取器
<div style="width: 600px; margin: auto">![](https://wx3.sinaimg.cn/large/d7b90c85ly1fx470e6hkdj20rv0dqacy.jpg)

## Results

<div style="width: 600px; margin: auto">![](https://wx1.sinaimg.cn/large/d7b90c85ly1fx470hjw5yj213t0o40yg.jpg)

<div style="width: 600px; margin: auto">![](https://wx1.sinaimg.cn/large/d7b90c85ly1fx470lfttsj21e10a378c.jpg)

## 分析

通常来说, R-FCN 和 SSD 模型要比 Faster RCNN 模块快得多, 但是 Faster RCNN 的精确度更高. 但是, FasterRCNN 可以通过降低候选区域框的数量来提升速度.

**The effect of the feature extractor:** 整体来说, 越强的特征提取器与 mAP 分数成正比, 但是对于 SSD 来说, 这种提升并不明显 (为什么 Inception v2 的 FasterRCNN 和 R-FCN 的 mAP 值那么低?)

<div style="width: 600px; margin: auto">![](https://wx4.sinaimg.cn/large/d7b90c85ly1fx489ru3zcj213r0gm418.jpg)

**The effect of object size:**

<div style="width: 600px; margin: auto">![](https://wx4.sinaimg.cn/large/d7b90c85ly1fx489wwqs3j20xv0k8mz1.jpg)

**The effect of image size:** 当 image size 从 600 降到 300 时, 精度度平均会降低 15.88%, 同时 inference time 也会降低 27.4%.

<div style="width: 600px; margin: auto">![](https://wx1.sinaimg.cn/large/d7b90c85ly1fx48a2fj8sj210j0n2mzv.jpg)

**The effect of the number of proposals:** proposals 可以大幅度降低测试时间, 同时 mAP 值只会降低一点(Inception ResNet v2, 300 -> 10, 35.4% -> 29%). 我们找到的 sweet point 大约是 50 个候选区域框, 此时可以在保持 300 候选区域框精度的 96% 的前提下, 将测试速度提升 3 倍.

<div style="width: 600px; margin: auto">![](https://wx1.sinaimg.cn/large/d7b90c85ly1fx48a6iqsaj213p0o1aex.jpg)

**FLOPs analysis:**
FLOPs(multiply-adds)

<div style="width: 600px; margin: auto">![](https://wx3.sinaimg.cn/large/d7b90c85ly1fx48aap87jj20y50fh0u0.jpg)

<div style="width: 600px; margin: auto">![](https://wx1.sinaimg.cn/large/d7b90c85ly1fx48aeop46j21j90ko78f.jpg)

**Memory analysis:**

<div style="width: 600px; margin: auto">![](https://wx2.sinaimg.cn/large/d7b90c85ly1fx48aopxpqj21kw0p0gpx.jpg)

<div style="width: 600px; margin: auto">![](https://wx2.sinaimg.cn/large/d7b90c85ly1fx48c65691j21kw0l4wiw.jpg)

**Good localization at .75 IOU means good localization at all IOU thresholds:**

<div style="width: 600px; margin: auto">![](https://wx3.sinaimg.cn/large/d7b90c85ly1fx48cjsjjvj21kw0jpwja.jpg)


表4总结了我们模型的性能(融合了5个FasterRCNN), 并且突出了如何在 COCO 评价标准上提升性能.
在模型融合时, 我们选取了5个FasterRCNN模型, 每个模型都是基于ResNet 和 Inception Resnet的, 他们的 stride 不同, 并且使用了不同的损失函数, 以及不完全相同的训练数据. 最终使用ResNet论文中的附录A的方法融合这些模型的检测结果.

<div style="width: 600px; margin: auto">![](https://wx3.sinaimg.cn/large/d7b90c85ly1fx48l1cy5pj21kw08aq6d.jpg)

表5总结了最后选定的模型的性能表现. 模型融合以及 multi-crop inference 大约使模型的精度提升了7个百分点.

<div style="width: 600px; margin: auto">![](https://wx3.sinaimg.cn/large/d7b90c85ly1fx48l4n1ofj21kw0bxdj7.jpg)

表6比较了单个模型和模型融合之间的性能差异

<div style="width: 600px; margin: auto">![](https://wx2.sinaimg.cn/large/d7b90c85ly1fx48l887c6j21kw08incq.jpg)
