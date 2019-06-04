---
title: M2Det (AAAI, 2019)
sitemap: true
categories: 计算机视觉
date: 2019-01-10 14:44:50
tags:
- 计算机视觉
- 目标检测
- 论文解读
---

**文章:** M2Det: A Single-Shot Object Detector based on Multi-Level Feature PyramidNetwork
**作者:** Qijie Zhao, Tao Sheng, Yongtao Wang
**机构:** 北京大学, 阿里达摩院


# 摘要

Feature pyramids 在现阶段 SOTA 的 one-stage (DSSD, RetinaNet, RefineDet) 和 two-stage (Mask R-CNN, DetNet) 中都被用来解决目标实例的尺度变化问题. 尽管这些模型取得了令人鼓舞的结果. **但是他们都具有一些限制: 因为他们的特征金字塔都是简单的根据 backbone 固有的多尺度的金字塔结构建立起来的, 但是这些 backbone 实际上是针对物体分类任务而设计的**. 在本文中, 我们提出了 Multi-Level Feature Pyramid Network (MLFPN) 来构建更加有效的, 针对不同尺度下的目标检测问题的特征金字塔结构. 首先, 我们融合了从 backbone 中提取的 multi-level 的特征(即 multi-layers)作为 base feature. 然后, 将 base feature 送入一组交替连接的 Thinned U-shape Modules(简化U型模块) 和 Feature Fusion Modules(特征融合模块), 利用每个 U shape 模块的 decode layers 的输出作为特征用来检测物体. 最后, 将 decode layers 具有相同尺寸的特征组合起来, 构建用于目标检测的特征金字塔, 这个特征金字塔中的每一个 feature map 都包含不同 levels 当中的特征(layers). 为了评价 MLFPN 的有效性, 我们设计并训练了一个强有力的端到端的 one-stage 模型, 命名为 M2Det (将 MLFPN 应用到 SSD 上面), 并且取到了更好的检测性能. 具体来说, 在 MS-COCO 上取得了 41.0 mAP, 11.8 FPS 的成绩, 使用的是 sing-scale inference strategy. 如果使用 multi-scale inference strategy 的话, 可以达到 44.2 的 mAP.

# 介绍

物体实例之间的尺度变化是目标检测任务中主要挑战之一, 并且通常有两种策略来解决该问题. 其一是在 image pyramid (一系列输入图片的不同尺度的副本)上检测物体, **该方法仅仅可以在 testing 阶段使用**. 很明显, 该方法会大大增加计算的复杂度和内存的消耗, 因此检测器的效率会大大降低. 第二种方法是在从输入图片中提取的 feature pyramid 上检测物体, 该方法可以同时在 training 和 testing 阶段使用. 相对于使用 image pyramid, 第二种方法使用更少的内存和算力. 更重要的是, feature pyramid 的结构可以很容易的添加到 sota 的模型当中, 并且可以进行端到端的训练.

<div style="width: 550px; margin: auto">![图1](https://wx4.sinaimg.cn/large/d7b90c85ly1g14s1fmhu1j213x0jy43i.jpg)

尽管目标检测的 feature pyramids 取得了很大令人鼓舞的结果, 但是它依然有一些缺点(limitations), 因为这些特征金字塔的构建都是简单的从 backbone 固有的多尺度的, 金字塔式的结构中得到的, 而这些 backbone 主要是为目标分类任务设计的. 例如, 如图1所示, SSD 直接使用 backbone(VGG16) 的最后两个卷积段的卷积层和 4 个额外的步长为 2 的卷积层来构建特征金字塔. FPN 通过融合深层和浅层的特征图谱来构建特征金字塔. STDN 仅仅使用 DenseNet 的最后一个 dense block 通过 pooling 和 scale-transfer 操作来构建特征金字塔. 一般来说, **上面提到的方法都具有两点 limitations: 第一, 金字塔中的特征图谱不足以表示针对目标检测任务的特征, 因为它们仅仅是从针对分类任务的 backbone 的特征层中构建的. 第二, 金字塔中的每一层特征图谱大多都被用来检测对应尺度的物体, 即主要或者仅仅是从 backbone 中一个单一的 level 层中构建的.** 一般情况下, **深层网络中的高级特征对分类子任务的区分能力更强, 而层次较浅的低级特征有助于物体的 location regression 子任务**. 此外, **低级特征更适合描述具有简单外观的物体, 而高级特征适用于具有复杂外观的物体.** 例如, 一个交通灯和一个处于远处的人也许会具有相当的大小, 但是很明显人具有更加复杂的特征. 因此, 金字塔中的每一个特征图谱(用于检测固定范围大小)主要或者仅仅包含了单一 level 的特征, 这样就有可能无法生成最优的检测结果.
本文的目标是构建一个更加有效的 feature pyramid 来检测不同尺寸的物体, 同时可以避免现有方法存在的那些 limitations. 如图 2 所示, 为了达到该目标, 我们首先融合了从 backbone 中提取的 multi-level 的特征(即, multi layers), 然后将其送入到一个由 TUM 和 FFM 交替连接的组件中, 来提取 **表征能力更强, 的 multi-level multil-scale 的特征**. 值得注意的是, 在每一个 U-shape 模块中的 decoder layers 都共享中相似的 depth. 最后, 我们将具有相同尺寸的特征图谱聚集起来, 从而构建最终的用于目标检测的特征金字塔. 很明显, 组成特征金字塔的最后一层的 decoder layers 具有比 backbone 中的网络层更深的层次, 也就是说, 它们的表征能力更强. 不仅如此, 最终的特征金字塔的每一层特征图谱的 decoder layers 都是从不同的层级中获得, 也就是说, 它们的表征能力更强. 不仅如此, 最终的特征金字塔的每一层特征图谱的 decoder layers 都是从不同的层级中获得的. 因此, 我们称本文的 feature pyramid block 为 Multi-Level Feature Pyramid Network(MLFPN).

<div style="width: 550px; margin: auto">![图2](https://wx2.sinaimg.cn/large/d7b90c85ly1g14s1vjmz5j21en0ihdmy.jpg)

为了验证 MLFPN 的有效性, 我们设计并训练了一个强有力的端到端的 one-stage 目标检测器, 命名为 M2Det (M2 的意思是 multi-level multi-scale features). 该模型在 MS-COCO 上取得了 41.0 mAP, 11.8 FPS 的成绩, 使用的是 sing-scale inference strategy. 如果使用 multi-scale inference strategy 的话, 可以达到 44.2 的 mAP.

# 相关工作

featurizing image pyramid: 由于对内存和算力的消耗, 该策略在 real-time 任务中几乎不可用.

feature pyramid: MS-CNN, SSD, DSSD, FPN, YOLOv3, RetinaNet, RefineDet

# Proposed Method

M2Det 的整体结构如图2所示. M2Det 使用 backbone 和 Multi-Level Feature Pyramid Network(MLFPN) 从输入图片中提取特征, 然后类似于 SSD, 基于学习到的特征生成密集的 bounding boxes 和 category scores, 最终再使用 NMS 算法生成最终的结果. MLFPN 包含三个模块: 特征融合模块(Feature Fusion Module, FFM), 简化的 U-shape 模块(Thinned U-shape Module, TUM), 以及尺度特征聚合模块(Scale-wise Feature Aggregation Module, SFAM). FFMv1 通过融合 backbone 的特征图谱来丰富 base features 中的语义信息. 每一个 TUM 都会生成一组多尺度的特征(a group of multi-shape features), 然后会交替的连接 TUMs 和 FFMv2s 模块来提取 multi-level multi-scale features. 除此以外, SFAM 会收集这些特征, 并通过尺度特征连接操作和自适应的注意力机制(scale-wise feature concatenation operation and an adaptive attention mechanism)将它们送到不同层次的特征金字塔中. 下面我们将详细介绍这三个模块.

## Multi-level Features Pyramid Network.

如图2所示, MLFPN 包含三部分, **首先**, FFMv1 融合了浅层和深层的特征来生成 base feature, **具体来说就是 VGGNet 的 conv4_3 和 conv5_3.** 这为 MLFPN 提供了多层级(multi-level)语义信息. **其次**, 若干个 TUMs 和 FFMv2 交替连接. 具体的说, 每一个 TUM 都会生成多个不同尺度的 feature maps. FFMv2 融合了 base feature 和前一个 TUM 输出个最大的 feature map. 融合后的 feature maps 会被送到下一个 TUM 中. 注意到第一个 TUM 没有其他 TUMs 的先验知识, 因此它仅仅是从 base feature 中进行学习. 输出的 multi-level multi-scale features 的计算方式如下:

$$[x_1^l, x_2^l, ..., x_i^l] = \begin{cases} T_l(X_{base}), && l = 1 \\ T_l(F(X_{base}, x_i^l-1)), && l = 2, ..., L \end{cases} \tag 1$$

上式中, $X_{base}$ 表示 base feature, $x_i^l$ 表示第 $l$ 个 TUM 中的 第 $i$ 个尺寸(scale)的 feature, $L$ 代表 TUMs 的数量, $T_l$ 代表第 $l$ 个 TUM 的处理过程, $F$ 代表 FFMv1 的处理过程. **最终**, SFAM 会通过按照尺度的特征连接操作(scale-wise feature concatenation operation)和按照深度的注意力机制(channel-wise attention mechanism)来聚集 multi-level multi-scale features.

**FFMs:** FFMs 会从 M2Det 网络中的不同层级(different levels)融合特征, 这一点对于构建最终的 multi-level feature pyramid 来说至关重要. **FFMs 使用 $1\times 1$ 的卷积层来压缩 input features 的 channels, 同时使用连接操作(concatenation operation) 来聚集这些特征图谱**. 特别的, 由于 FFMv1 接受 backbone 中两个不同尺寸(scales)的 feature map 作为输入, 所以它将会采用上采样操作(upsample operation)来将深层次的 feature map 放大到和浅层 map 相同到尺寸, 然后才进行 concatenation 操作. 同时, FFMv2 接受 base feature 和前一个 TUM 输出的最大 feature map 作为输入, 这两个 feature map 本身就具有相同的尺寸(scale), 并且会生成融合后的特征, 用作下一个 TUM 的输入. FFMv1 和 FFMv2 的结构细节分别如图4(a)和(b)所示.

<div style="width: 550px; margin: auto">![图4](https://wx3.sinaimg.cn/large/d7b90c85ly1g14s31ygnyj21280ocjvp.jpg)

**TUMs:** 如图4所示, 和 FPN 以及 RetinaNet 不同, TUM 采用一个 "更薄" 的 U-shape 结构. **encoder 是一系列 stride 为 2 的 $3 \times 3$ 的卷积层组成, decoder 将这些卷积层的输出作为其特征映射的参考集, 而 FPN 的做法是将 ResNet backbone 中每一个卷积段(stage)的最后一层的输出作为参考集.** 除此以外, 我们在 decoder 的 upsample 和 element-wise sum operation 之后添加了 $1\times 1$ 的卷积层来增加学习能力, 同时保持特征的平滑度(smoothness). 每一个 TUM 的 decoder 的所有输出构成了当前 level 和 multi-scale features. 最终, 堆叠的 TUMs 的输出构成了 multi-level multi-scale features, 中间的 TUM 提供中间层级特征(medium-level features), 后面的 TUM 提供深层次的特征(deep-level features).

<div style="width: 550px; margin: auto">![图3](https://wx1.sinaimg.cn/large/d7b90c85ly1g14s2hzbonj21en0bsmzp.jpg)

**SFAM:** SFAM 的目的是聚集 TUMs 产生的 multi-level multi-scale features 到图3所示的 multi-level feature pyramid. SFAM 的第一阶段是沿着 channel dimension 将具有相同尺寸(scale) 的特征图片连接(concatenate)起来. 聚集后的特征金字塔可以表示成: $X = [X_1, X_2, ..., X_i]$, 这里 $X_i = Concat(x_i^1, x_i^2, ..., x_i^L) \in R^{W_i \times H_i \times C}$, 代表第 $i$ 大(scale)的特征. 因此, 在聚集后的特征金字塔中的每一个 scale 都包含 multi-level depths 的特征. 然而, **简单的连接操作(concatenation) 并没有足够的适应能力(not adaptive enough).** 在第二阶段, 我们引入了 channel-wise attention module 来丰富特征, 以便它能关注那些有益的 channels. Following SE block, 我们在 squeeze step 使用 global average pooling 来生成 channel-wise statistics $z\in R^C$. 同时为了完全捕获 channel-wise dependencies, 在之后的 excitation step 通过两个全连接层来学习注意力机制:

$$s = F_{ex}(z, W) = \sigma(W_2 \delta (W_1 z)), \tag 2$$

上式中 $\sigma$ 代表 ReLU, $\delta$ 代表 sigmoid, $W_1 \in R^{\frac{C}{r}\times C}$, $W_2 \in R^{ C\times \frac{C}{r}}$, $r$ 是 reduction ratio(在本文的实验中 $r=16$). 最终的输出是通过 reweighting the input X with activation s 得到的:

$$\tilde X_i^c = F_{scale}(X_i^c, s_c) = s_c \dot X_i^C, \tag 3$$

上式中, $\tilde X_i = [\tilde X_i^1, \tilde X_i^2, ..., \tilde X_i^C]$, 每一个特征都被 rescaling operation 增加或减弱(enhanced or weakened).

**二阶段的自适应调整过程实际上是先用 avg 得到 channel 上的统计数据, 然后用两个全连接层来学习 channel 之间的依赖, 最后根据此依赖对输入的 multi-level feature map 的权重进行 reweighting**

## Network Configurations

我们使用了两种类型的 backbones. 在训练整个网络之前, 先将 backbone 在 Image 2012 上进行预训练. MLFPN 的默认配置包含有 8 个 TUMs, 每一个 TUM 具有 5 个卷积层和 5 个上采样操作, 因此每个 TUM 将会输出 6 种 scales 的 feature maps. 为了降低参数的数量, 对于 TUM 的每种尺度的特征, 我们仅仅申请 256 维的通道, 因此网络可以容易的在 GPUs 上进行训练. 对于输入的尺寸大小, 我们采用和 SSD, RefineDet, 以及 RetinaNet 一样的设置, 即 320, 512, 和 800.
在检测阶段, 我们为 6 个金字塔特征都添加了两个卷积层, 分别用来获取位置回归和物体分类的预测结果. 检测使用的 default boxes 的尺度范围和 SSD 的规则相同. 并且当输入尺寸为 $800\times 800$ 的时候, 除了保持最大特征图的最小尺寸外, scale ranges 将会按比例增加. 在 特征金字塔的 **每一个像素点** 上, 我们设置了 6 个 anchors(具有3中宽高比). 然后, 我们利用 0.05 的阈值来过滤那些 socres 很低的 anchors (难负样例挖掘). 接着使用了 soft-NMS with linear kernel 来进行后处理.

# Experiments

## Implementation details

start training with warm-up strategy for 5 epochs.
initialize lr: $2\times 10^{-3}$, decrease it to $2\times 10^{-4}$ and $2\times 10{-5}$ at 90 epochs and 120 epochs, and stop at 150 epochs.
developed with PyTorch v0.4.0
input size: 320, 512
GPU: 4 NVIDIA Titan X GPUs
batch size: 32 (16 each for 2 Gpus, or 8 each for 4 GPUs)

<div style="width: 550px; margin: auto">![表1](https://wx4.sinaimg.cn/large/d7b90c85ly1g14s3laawbj212n0qwn7p.jpg)
<div style="width: 550px; margin: auto">![表2](https://wx1.sinaimg.cn/large/d7b90c85ly1g14s4b6xfjj20t20fcmze.jpg)
<div style="width: 550px; margin: auto">![表3](https://wx1.sinaimg.cn/large/d7b90c85ly1g14s4p71w4j20tf0f9go1.jpg)
<div style="width: 550px; margin: auto">![图5](https://wx2.sinaimg.cn/large/d7b90c85ly1g14s5kn8h2j20u20ghtb2.jpg)
<div style="width: 550px; margin: auto">![图6](https://wx2.sinaimg.cn/large/d7b90c85ly1g14s5v8vtpj20um0nrwxa.jpg)
