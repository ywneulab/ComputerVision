---
title: SqueezeNet (ICLR, 2017)
sitemap: true
categories: 计算机视觉
date: 2018-01-20 15:05:02
tags:
- 计算机视觉
- 网络结构
- 论文解读
---


**文章:** SqueezeNet: AlexNet-Level Accuracy With 50x Fewer Parameters And < 0.5 Mb Model Size
**作者:**
**备注:** UC Berkeley, Stanford University

# 摘要

在相同精度下, 小模型至少具有三点好处: (1) 在分布式训练时需要较少的服务器间的通信; (2) 导入到嵌入式设备上时需要更少的带宽; (3) 更适合部署到其他硬件上. 为了能够提供这些优势, 我们提出了一个小型的 CNN 框架, 名为 SqueezeNet. SqueezeNet 可以达到 AlexNet 的精度, 但是只需要 1/50 的参数量. 另外, 结合一些模型压缩技术, 我们可以将 SqueezeNet 压缩到不足 0.5Mb.

# Introduction and Motivation

在给定精度小, 小模型具有三点好处:
- More efficient distributed training
- Less overhead when exporting new models to clients
- Feasible FPGA and embedded deployment

为了获得以上好处, 我们提出了 SqueezeNet. 此外, 我们还尝试了一种更有规律的方法来搜索 CNN 网络结构的设计空间.

# Related Work

**Model Compression:** SVD, Network Pruning, Deep Compression, EIE.

**CNN MicroArchitecture:** 1x1, 3x3, 1x3, 3x3.

**CNN MacroArchitecture:** Inception, ResNet

**Neural Network Design Space Exploration:** Automated Search

# SqueezeNet: Preserving Accuracy With Few Parameters

首先介绍模型的整体架构, 然后介绍 Fire module, 最后介绍如果构建 SqueezeNet.

## Architectural Design Strategies

为了找到精度相当但是参数更少的模型, 我们采用了下面三种策略来设计 CNN 网络结构
- **Replace 3x3 filters with 1x1 filters.** 1x1 的卷积核参数量比 3x3 卷积核的参数量少9倍, 因此优先选择;
- **Decrease the number of input channels to 3x3 filters.** 由于 3x3 卷积核参数量和输入通道输入通道数有关, 因此, 我们优先减少 3x3 卷积核的输入通道数. 我们利用 **squeeze layers** 来完成减少通道数的任务.
- **Downsample late in the network so that convolution layers have large activation maps.** 每一个卷积层输出的特征图谱的尺寸通过两方面因素控制: (1) 输入图片的尺寸; (2) CNN 中的 downsample 网络层. 通常, downsample 可以通过 conv 和 pooling 来实现. 如果浅层的网络具有较大的下采样步长, 那么大多数网络层的特征图谱的尺寸都会比较小, 反之, 则比较大. **我们的 Intuition 是, 在其他条件相同的情况下, 较大的特征图谱(利用 delayed downsampling 实现)可以具有更高的分类精度.**

上面的前两条是在试图保持准确性的同时, 明智的减少 CNN 中的参数量, 第三条是在有限的资源条件下, 尽可能的提升检测的精度.

## The Fire Module

我们定义了 Fire Modules 作为组成 SqueezeNet 的基本 Block. 它包括: 一个 squeeze conv 层(由 1x1 卷积层组成), 一个 expand 层(由 1x1 和 3x3 卷积层组成), Fire Modules 的结构如图1所示. 我们在 Fire Modules 中展示了三个可以调节的维度(超参数): $s_{1x1}, e_{1x1}, e_{3x3}$. 其中, $s_{1x1}$ 代表了 squeeze layer 中的 filters 的数量, $e_{1x1}$ 代表了 expand layer 中 1x1 filters 的数量, $e_{3x3}$ 代表了 expand layer 中 3x3 filters 的数量. **当我们使用 Fire Modules 时, 我们令 $s_{1x1}$ 小于 $(e_{1x1} + e_{3x3})$, 这样一来 squeeze layers 可以起到降低通道维度的作用.**

<div style="width: 550px; margin: auto">![图1](https://wx4.sinaimg.cn/large/d7b90c85ly1g1iuaref4sj20yf0hw424.jpg)

## The SqueezeNet Architecture

<div style="width: 550px; margin: auto">![图2](https://wx1.sinaimg.cn/large/d7b90c85ly1g1iub0wfrbj20z20p2n26.jpg)

SqueezeNet 的结构如图2所示, 他的第一层是传统的卷积层, 之后由 8 个 Fire Modules 组成, conv10 也是传统的卷积层, 最后是由 GAP 和 Softmax 组成的分类层. 从网络的开始到结束, 我们会逐渐增加每个 Fire Module 模块的过滤器数量. SqueezeNet 会在 conv1, fire4, fire8, conv10 之后添加 max-pooling 层来进行下采样. 最终完整的 SqueezeNet 结构如表1所示.

<div style="width: 550px; margin: auto">![表1](https://wx4.sinaimg.cn/large/d7b90c85ly1g1iub8zx7fj20z80mkjxk.jpg)

其他的 SqueezeNet 细节如下所示:
- Padding: 为了使 1x1 和 3x3 滤波器的输出激活具有相同的高度和宽度, 我们在扩展模块的 3x3 滤波器的输入数据中添加了一个填充为零的1像素边框.
- ReLU
- Dropout: 在 fire9 使用, 0.5.
- 没有使用 FC 层
- lr: 初始值为 0.04, 之后不断衰减
- Caffe框架本身并不支持包含多个滤波器分辨率(例如 1x1 和 3x3 组成的 expand layer)的卷积层. 因此, 实际上我们是用了两个独立的卷积层来实现 expand layer. 我们将两个卷积层的输出结果在 channel 维度上连接起来, 这在数值上和 expand layer 是等价的.

# Evaluation of SqueezeNet


表2显示了不同的模型压缩方法对于 AlexNet 的压缩效果, 以及 SqueezeNet 的效果表现(Deep Compression 貌似很有用).
<div style="width: 550px; margin: auto">![表2](https://wx1.sinaimg.cn/large/d7b90c85ly1g1iubelnk8j20yt0ck42w.jpg)

# CNN MicroArchitecture Design Space Exploration

SqueezeNet 虽然已经达到了我们的预期目标, 但是还有许多未被探索的设计, 下面, 我们将分两部分进行介绍: MicroArchitecture(网络层的模块设计) 和 MacroArchitecture(顶层的整体架构设计).
请注意, 我们这里的目标不是在每个实验中都最大化精度, 而是理解 CNN 架构选择对模型大小和精度的影响.

定义了各种超参数来决定模型的大小(通过改变通道数实现): $base_e = 128, incr_e = 128, pct_{3x3} = 0.5, freq = 2, SR = 0.5$.

SR 对模型大小和精度的影响如图3(a)所示, 3x3 卷积核数量占比的多少对模型大小和精度的影响如图3(b)所示.

<div style="width: 550px; margin: auto">![图3](https://wx4.sinaimg.cn/large/d7b90c85ly1g1iubqq7h5j20y80gm41v.jpg)

# CNN MacroArchitecture Design Space Exploration

受到 ResNet 的启发, 我们对比了三种不同模型(结构如图2所示):
- 原始的 SqueezeNet
- 使用了简单的短接通路的 SqueezeNet
- 使用了复杂的短接通路的 SqueezeNet

simple bypass: 在 Fire Modules 3, 5, 7, 9 之间添加了 bypass 连接, 令这些模型学习输入输出之间的残差.

complex bypass: 当输入和输出的特征图谱的通道数不同时, 不能使用 simple bypass 连接, 因此, 我们利用 1x1 卷积核来实现 complex bypass 连接.

三种模型的精度和模型大小如表3所示.

<div style="width: 550px; margin: auto">![表3](https://wx4.sinaimg.cn/large/d7b90c85ly1g1iuc9qu5lj20yz08zgo3.jpg)

<span id = "简述 SqueezeNet 的原理">
# 简述 SqueezeNet 的原理

SqueezeNet 定义了 Fire Modules 作为其基本的组成部件, Fire Modules 由 squeeze layer 和 expand layer 组成, 其中是通过 1x1 的卷积层实现的, 后者是通过 1x1 和 3x3 的卷积层实现的(这两层的输入都是 sequeeze layer, 输出会将二者在通道维度叠加). SqueezeNet 的第一层是传统的 7x7 卷积层, 之后由 8 个 Fire Modules 组成,  conv10 也是传统的 1x1 卷积层, 最后是由 GAP 和 Softmax 组成的分类层. SqueezeNet 会在 conv1(自身也是下采样), fire4, fire8, conv10 之后添加 max-pooling 层来进行下采样(总步长为 32). 同时, SqueezeNet 受到 ResNet 的启发, 可以在 Fire Modules 3, 5, 7, 9 的输入和输出之间添加 bypass 连接, 令这些模型学习输入输出之间的残差(可以获得更高的精度). 对于通道数不同的其他 Modules, 可以通过 1x1 卷积层来建立 complex bypass).
