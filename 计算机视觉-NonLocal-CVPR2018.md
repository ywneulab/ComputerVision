---
title: Non-local Neural Networks (CVPR, 2018)
sitemap: true
categories: 计算机视觉
date: 2018-11-04 13:12:48
tags:
- 计算机视觉
- 目标检测
- 论文解读
---

**文章:** Non-local Neural Networks
**作者:** Xiaolong Wang, Ross Girshick, Abhinav Gupta, Kaiming He

# 核心亮点

**1) 提出了 non-local operations 来解决 CNN 网络中的 long-range dependencies 问题**
传统 CNN 的卷积操作由于输出神经元只会与输入图谱上的一部分区域有关系, 因此, 在面对那些 long-range dependencies 的时候, 往往不能捕获到足够的信息来表征数据, 为此, 作者提出了 non-locl operations, 其相当于构造了一个和特征图谱尺寸一样大的卷积核, 从而可以维持更多信息.

**2) non-local module 可以作为一种通用的模块应用在各项任务上**
作者通过实验证明, non-local 的有效性不仅仅局限于某一类特殊任务(如视频分类), 同时还可以轻易的整合到其他现有模型中, 如将其整合到 MaskRCNN 中, 可以当做是一种 trick 来提供 MaskRCNN 在目标检测/分割, 姿态识别等任务上的性能表现.

<div style="width: 600px; margin: auto">![](https://wx1.sinaimg.cn/large/d7b90c85ly1fx5hgur618j20o30klwgh.jpg)

# 论文细节

## 摘要

不论是卷积网络还是递归网络, 它们都是作用在某一块局部区域 (local neighborhood) 的operations. 在本文中, 我们提出了 non-local operations 作为一种通用的神经网络的 building blocks 来捕捉基于 long-range 的依赖关系. 受到经典的 non-local means 方法的启发, 本文的 non-local operation 会将某一位置的响应当做是一种从特征图谱所有位置的加权和来计算. 该 building block 可以插入到现在计算机视觉的许多模型当中, 进而可以提升分类, 检测, 分割等视觉任务的性能表现.

## 介绍

在深度神经网络中, 捕获 long-range dependencies 信息是十分重要的, 如面向序列数据的 LSTM, 面向图像数据的更大的感受野等. 但是不论是 convolutional 还是 recurrent 操作, 它们都是在一个 local neighborhood 上进行计算的. 因此, 只能通过不断的重复执行才能够捕获到足够的 long-range dependencies 信息(卷积计算之间的重叠区域). 这种 Repeating local operations 具有很多缺点. 第一, 计算的低效性; 第二, 会造成很多优化时的困难; 最后, 会产生多次反射, 这使得很难在相距较远两个位置的点传递反向和前向的计算结果.(???, 这三个缺点具体什么意思?)
在本篇文章中, 我们提出了一种用于捕获 long-range dependencies 信息的简单, 高效, 通用的神经网络构建模块, 称为 non-local. 本文的 non-local 是从经典的 non-local mean operation 泛化而来的. 直观来说, non-local operaions 会计算输入的特征图谱上所有点加权和的响应(如图1). 这些点既可以代表空间位置, 也可以代表时间, 时空等, 暗示着 non-local 可以应用于图片, 序列和视频相关任务.

<div style="width: 600px; margin: auto">![](https://wx4.sinaimg.cn/large/d7b90c85ly1fx5c7a6ovnj20t00h8kfn.jpg)

使用 non-local operaions 具有以下几点优点:
- 相比于 CNN 和 RNN 的逐步计算的劣势, non-local 操作 可以直接从任意两点中获取到 long-range dependencies.
- 根据实验结果可知, non-local operations 是十分高效的, 并且即使在只有几层网络层时, 也能取得很好的效果.
- 最后, 本文的 nocal operaions 会维持输入变量的尺寸大小, 并且可以很容易的与现有的其他 operations 结合使用.
我们用 video classification 任务来展示 non-local 的有效性. 在视频数据中, long-range interactions 不仅出现在 空间位置上的 distant pixels, 还会出现在时间维度上的 distant pixels. 通过一个单一的 non-local block (basic unit), 便可以捕获到这些 spacetime dependencies, 如果将多个 non-local block 组合起来形成 non-local neural networks, 便可以提高 video classification 任务的准确度(不加任何tricks). 另外, non-local 网络要比 3D 卷积网络的计算性价比更高. 为了说明 non-local 的一般性, 我们还在 COCO 数据集上进行了目标检测/分割, 姿态识别等任务的实验, 在基于 MaskRCNN 的网络基础上, 我们的 non-local blocks 可以用较少的计算开销进一步提升模型的精度.

## 相关工作

**Non-local image:** Non-local means 是一种经典的过滤算法, 它会计算整幅图片的像素值的加权平均和, 使得一些较远元素可以贡献一些位置上的响应.

**FeedForward modeling for sequences:** 近年来很多前馈网络被用于解决语音和自然语言处理, 它们通过更大的感受野来实现 long-range dependencies.

**Self-attention:** 本文的工作和机器翻译中的 self-attention 机制有关.

**Interaction networks**

**Video classification architectures**

## Non-local Neural Networks

下面首先给出 non-local operations 的一般性定义, 然后会给出几种特定的变体

### Formulation

根据 non-local mean operation, 我们可以在深度卷积网络中定义如下的一般化的 non-local operation:

$$y_i = \frac{1}{\zeta (x)} \sum_{\forall j}f(x_i, x_j) g(x_j) \tag 1$$

上式中, $i$ 代表了 output 的 position 响应, 而 $j$ 枚举了所有可能的 position. $x$ 是 input signal (一般为特征图谱), $y$ 是 output signal (与 $x$ 的 size 相同). $f$ 会返回一个标量, $g$ 也会返回一个标量, $\zeta (x)$ 的作用是对响应进行归一化. 该公式的 non-local 特性主要体现在考虑了所有可能的 position ($\forall j$), 而卷积网络只会考虑 output position 周围位置的像素点.
non-local 是一个非常灵活的模块, 它可以被添加到深度神经网络的浅层网络当中去(不像fc那样处于深层网络), 这使得我们可以构建更加丰富的模型结构来结合 non-local 和 local 信息.

### Instantiations

接下来, 我们举例说明几种常用的 $f$ 和 $g$. 有趣的是, 通过实验(表2a)发现, 本文的 non-local 模块对于 $f$ 和 $g$ 的选择并不是特别敏感, 这意味着 non-local 的通用性正是提升各个模型在不同任务上性能表现的主要原因.
为了简化, 我们考虑将 $g$ 写成线性形式: $g(x_j) = W_g x_j$, 这里的矩阵 $W_g$ 正是模型需要学习的参数, 在实现时, 通常会通过 1×1(或 1×1×1) 的卷积 来实现. 接下来, 我们来讨论 $f$ 的选择

**Gaussian:** 最容易想到的选择

$$f(x_i, x_j) = e^{x_i^T x_j}$$

在这里, $x_i^T x_j$ 是两个向量的点积, 则会返回一个标量, 有时候也可以使用欧几里得距离, 不过点积的实现更加容易. 归一化因子 $\zeta (x) = \sum_{\forall j} f(x_i, x_j)$

**Embedded Gaussian:** 这是高斯函数的一个简单扩展

$$f(x_i, x_j) = e^{\theta (x_i)^T} \phi(x_j)$$

在上式中, $\theta (x_i) = W_{\theta} x_i$ , $\phi(x_j) = W_{\phi} x_j$ , 分别为两种 embeddings. 同时, 归一化因子 $\zeta(x) = \sum_{\forall j} f(x_i, x_j)$.

**Dot product:** $f$ 也可以定义成点乘

$$f(x_i, x_j) = \theta(x_i)^T \phi(x_j)$$

这里我们采用了 embedded 版本, 并且令归一化因子 $\zeta = N$, $N$ 是 $x$ 中 positions 的数量, 而不是 $f$ 的和.

**Concatenation:** Concatenation 曾被用于 Relation Network 来解决 visual reasoning 问题, 形式如下

$$f(x_i, x_j) = ReLU(w^T_f[\theta (x_i), \phi (x_j)])$$

上式中, $[\cdot, \cdot]$ 代表这 concatenation 操作, $w_f$ 代表着将 concatenated vector 映射到标量的权重向量, 同样, 令 $\zeta(x) = N$.

### Non-local Block

我们将上面介绍的公式(1) (non-local operation)包装进一个 non-local block 中, 使其可以整合到许多现有的网络结构当中, 我们将 non-local 定义成如下格式:

$$z_i = W_z y_i + x_i$$

上式中, $y_i$ 即为公式(1)的返回结果, 而 $+x_i$ 代表着残差连接. 残差连接使得我们可以将一个全新的 non-local block 插入到任何预训练的模型中, 而不会坡缓其原始的行为(eg, $W_z$ 初始化为0). 一个关于 non-block 的实例如图2所示. 当将 non-local block 应用于一个 high-level 的特征图谱时, 其带来的计算成本是很低的, 如在图2中, 通常情况下, $T=4, H=W=14 or 7$.

<div style="width: 600px; margin: auto">![](https://wx2.sinaimg.cn/large/d7b90c85ly1fx5e0oc6krj20t40unq91.jpg)

**Implementation of Non-local Blocks:** 我们将 $W_g, W_{\theta}, W_{\phi}$ 的 channels 设置为 $x$ channels 的一半. 另一个 subsampling 的 trick 是将公式(1)改为: $y_i = \frac{1}{\zeta (\hat x)} \sum_{\forall j} f(x_i, \hat x_j) g(\hat x_j)$, 其中 $\hat x$ 是 $x$ 的 subsampled 版本. 这个 trick 可以使计算量减少 1/4, 并且不会改变 non-local 的行为, 仅仅只会令计算变得更加稀疏. 通过在图2中的 $\phi$ 和 $g$ 之后加一层 max pooling layer 即可实现.

## Video Classification Models

略

## Experiments on Video Classification

<div style="width: 600px; margin: auto">![](https://wx2.sinaimg.cn/large/d7b90c85ly1fx5enfe1sgj21kw0w97wi.jpg)

<div style="width: 600px; margin: auto">![](https://wx2.sinaimg.cn/large/d7b90c85ly1fx5er37xq0j21kw0ekx3w.jpg)

<div style="width: 600px; margin: auto">![](https://wx3.sinaimg.cn/large/d7b90c85ly1fx5ew2engqj20sm0crwq8.jpg)



## Extension: Experiments on COCO

**Object detection and instance segmentationn:** 我们修改了 MaskRCNN 的 backbone, 在其 res4 的后面添加了一个 non-local block. 与原文不同是, 我们使用了端到端的联合训练(原文是将 RPN 和 RoIAlign等分开训练), 这使得我们的 baseline 提高了.
表5显示了在 COCO 数据集上的 box 和 mask AP. 我们可以看到, 一个单独的 non-local block 可以提升所有 Res50/101 和 X152 的baseline. 另外, 下面的性能提升只需要付出很小的计算代价(不超过原模型的 5%), 我们同样尝试了使用更多的 non-local 模块, 但是发现这会降低模型的性能.

<div style="width: 600px; margin: auto">![](https://wx3.sinaimg.cn/large/d7b90c85ly1fx5ew9ewd2j20sp0d1qhh.jpg)

表6显示了 non-local 在姿态识别任务上的性能提升.

<div style="width: 600px; margin: auto">![](https://wx3.sinaimg.cn/large/d7b90c85ly1fx5ewok19fj20sj08hq4x.jpg)
