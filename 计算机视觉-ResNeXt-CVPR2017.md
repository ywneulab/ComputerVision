---
title: ResNeXt (CVPR, 2017)
sitemap: true
categories: 计算机视觉
date: 2018-11-15 15:08:33
tags:
- 计算机视觉
- 网络结构
- 论文解读
---

**文章:** Aggregated Residual Transformations for Deep Neural Networks
**作者:** Saining Xie, Ross Girshick, Piotr Dollar, Zhuowen Tu, Kaiming He
**备注:** UC San Diego, FAIR

# 核心亮点

本文提出了一种新的网络模型架构 ResNeXt, 通过利用多路分支的特征提取方法, 提出了一种新的基于 ResNet 残差模块的网络组成模块, 并且引入了一个新的维度 cardinality. **该网络模型可以在于对应的 ResNet 相同的复杂度下, 提升模型的精度**(相对于最新的 ResNet 和 Inception-ResNet).
同时, 还通过实验证明, **可以在不增加复杂度的同时, 通过增加维度 cardinality 来提升模型精度**, 比更深或者更宽的 ResNet 网络更加高效.

<div style="width: 550px; margin: auto">![](https://wx4.sinaimg.cn/large/d7b90c85ly1fxc7dy7cpgj216l0mfjvj.jpg)

# 摘要

本文提出了一个简单的, 高度模型化的针对图像分类问题的网络结构. 本文的网络是通过重复堆叠 building block 组成的, 这些 building block 整合了一系列具有相同拓扑结构的变体(transformations). 本文提出的简单的设计思路可以生成一种同质的, 多分支的结构. 这种方法产生了一个新的维度, 我们将其称为`基`(变体的数量, the size of the set of transformations). 在 ImageNet-1K 数据集上, 我们可以在保证模型复杂度的限制条件下, 通过提升基的大小来提高模型的准确率. 更重要的是, 相比于更深和更宽的网络, 提升基的大小更加有效. 我们将本文的模型命名为 ResNeXt, 本模型在 ILSVRC2016 上取得了第二名. 本文还在 ImageNet-5K 和 COCO 数据集上进行了实验, 结果均表明 ResNeXt 的性能比 ResNet 好.

# 介绍

目前, 计算机视觉任务已经慢慢从`特征工程`转向了`网络工程`, 但是, 随着网络深度的增加, 设计良好的网络结构, 变的异常困难. VGG-nets 保留了简单同时有效的网络结构, 它通过将多个卷积层堆叠的方式来组成神经网络, 这种堆叠式结构在 ResNet 中也得到了保留, 并在这种结构的基础上, 开发出了性能强劲的网络模型.
与 VGG-nets 不同的是, Inception 系列的模型通过精心的拓扑结构设计, 也取得了很好的模型准确度. Inception 模型的一个重要的属性就是 `split-transform-merge strategy`. 在 Inception 模块中, 输入数据会被划分成一些更低维度的 embeddings(通过1×1卷积), 然后通过一系列特定的卷积层(3×3, 5×5)进行转换, 最后通过 concatenation 融合起来. 这种方式所使用的空间复杂度是用单层卷积的空间复杂度的一个子集. 因此, Inception 模块可以利用更低的复杂度获取更高的特征表示能力. **尽管通过精心的布置和组合, Inception 模块组成的网络可以取得较好的性能表现, 但是, 当面对一个新的任务或数据集时, 往往无法很快的找到合适的模块组合和参数设置**.
本文提出了一种基于 VGG/ResNet 的 repeating layers 策略的模型, 同时还利用了 split-transform-merge 策略, 如图1所示(二者复杂度相同, 但是右边具有更高的精度).

本文的方法引入了一个新的维度 cardinality, 实验表明, 通过提升该维度, 可以更有效的提升模型的精度(相比于更深和更宽, ResNeXt 101 的精度高于 ResNet-200, 但是仅仅只有其一半的复杂度), 我们将模型命名为 ResNeXt (暗示 next dimension).

# 模板(Template)

本文提出的模型设计思路遵循 VGG/ResNets 的 repeating layers 策略. 首先包含一组堆叠的残差模块, 这些残差模块具有相同的拓扑结构, 并且服从两条基本规则: (1), 如果处理的特征图谱具有相同的尺寸大小, 这些这些 block 的超参数设置相同(filters); (2), 每次当特征图谱的尺寸缩小两倍时, 卷积核的深度也会放大两倍. 第二条规则使得每一个 block 的复杂度(FLOPs, floating-point operations)是差不多相同的. 根据这两条规则, 我们只需要设计出一个模板(template), 进而模板中所有的模块都会相应的确定(相比于 Inception 设计更加简单), 如表1所示.

<div style="width: 550px; margin: auto">![](https://wx3.sinaimg.cn/large/d7b90c85ly1fxc7ewvfr1j20u00wn0y7.jpg)

# 回顾简单神经元

最简单的神经元是通过内积计算的, 而实际上, 内积也可以被看做是一个 aggregating tansformation:

$$\sum_{i=1}^D w_i x_i \tag 1$$

该公式的计算操作会通过一个神经元输出, 如图2所示.

<div style="width: 550px; margin: auto">![](https://wx2.sinaimg.cn/large/d7b90c85ly1fxc7faltsij20w609l0tk.jpg)

上面的操作可以被重新定义成一组关于 splitting, transforming 以及 aggregating 的组合(conbination).
- Splitting: 向量 $\vec x$ 被划分成了低维度的 embedding, 每一个维度为 $x_i$
- Transforming: 低维度的相同表示通过权重 $w_i$ 进行 transform.
- Aggregating: 通过求和公式 $\sum_{i=1}^D$ 将 transformations 连接起来.

# 聚合变换(Aggregated Transformations)

根据上面的简单神经元的讨论, 下面我们考虑将 elementary trasformation($w_i x_i$) 用一个更加一般化的函数来替代, 这个函数本身也可以是一个神经网络, 如下所示:

$$F(x) = \sum_{i=1}^C T_i (x) \tag 2$$

上式中的 $T_i(x)$ 可以是任意形式的函数, 通常情况下,  $T_i(x)$ 会将 $x$ 映射到一个更低的维度上去, 形成一个 embedding. $C$ 代表了 Transformations 的个数, 我们将其定义为 cardinality.
在本文中, 我们使用了一种简单的方式来设计 transformation function: 所有的 $T_i$ 都具有相同的拓扑结构(图1右侧).
我们将(2)式的 aggregated transformation 写成残差函数的形式:

$$y = x + \sum_{i=1}^C T_i(x) \tag 3$$

图3展示了本文模型与 Inception-ResNet 之间的关系, 图3(a)中的一些操作和图3(b)很相似, 而图3(b)看起来很像是包含 branching 和 concatenating 的 Inception-ResNet block. 但是与 Inception 和Inception=ResNet 模块不同的是, 本文的模型在不同的 paths 之间共享同样的拓扑结构, 因此, 我们的模型在设计方面需要的成本更小.

<div style="width: 550px; margin: auto">![](https://wx1.sinaimg.cn/large/d7b90c85ly1fxc7gghlzjj21xt0l1n26.jpg)

图3(c)和图4展示了 group convolutions(Alex Net).

<div style="width: 550px; margin: auto">![](https://wx1.sinaimg.cn/large/d7b90c85ly1fxc7gu5otjj20we0df764.jpg)

# 模型容量(Model Capacity)

实验表明, 本文的模型可以在维持模型复杂度和参数量的前提下提升模型的准确率. 当我们在维持复杂度的前提下调节 cardinalities C 的时候, 我们希望尽可能的不去改动其他的超参数. 我们选择去调节 bottleneck 的宽度(如图1右侧中的4-d), 因为这可以独立于 block 的 input 和 output.
在图1左侧中, 原始的 ResNet bottleneck block 的参数量为 $256\cdot 64 + 3\cdot 3\cdot 64\cdot 64 + 64\cdot 256 \approx 70k$ 以及成比例的 FLOPs. 而我们的 template (图1右侧) 的参数量为:

$$C\cdot ( 256\cdot d + 3\cdot 3\cdot d\cdot d + d\cdot 256) \tag 4$$

当 $C=32, d=4$ 是, 上式约为 $70k$. 表2展示了 cardinality $C$ 和 bottleneck width $d$ 之间的关系.

<div style="width: 550px; margin: auto">![](https://wx3.sinaimg.cn/large/d7b90c85ly1fxc7haiwkej20wf0a4gno.jpg)

# 实现细节(Implementation details)

- input image: 224×224 randomly cropped from resized image
- resized image: scale and aspect ratio augmentation
- shortcuts: option B in ResNet
- 在conv3,4,5中的 downsampling 操作通过 stride 为2的 3×3 卷积完成
- SGD
- mini-batch size: 256 on 8 GPUs (32 per GPU)
- weight decay: 0.0001
- momentum: 0.9
- learning rate: 0.1, 会降低3次, 每次降低10倍(每次更新学习率都会使得错误率出现断崖式下跌)
- 权重初始化: Xavier
- module: 图3(c)
- BN: 在图3(c)中的卷积层之后
- ReLU: 在 block 的输出中, ReLU 在 shortcut 之后使用, 其情况下, 均在 BN 之后使用.
- 图3中的三种形式通过合理安排 BN 和 ReLU 的位置可以互相等价. 我们选择图3(c)是因为它更加简洁, 速度更快.

# 实验(Experiments)

# Experiments on ImageNet-1K

**Cardinality vs. Width:**

<div style="width: 550px; margin: auto">![](https://wx2.sinaimg.cn/large/d7b90c85ly1fxc7i6lo6tj20wn0magpy.jpg)

<div style="width: 550px; margin: auto">![](https://wx1.sinaimg.cn/large/d7b90c85ly1fxc7hqcjutj21xx0qiagu.jpg)

# Increasing Cardinality vs. Deeper/Wider

表4显示出提升 Cardinality 可以降低错误率, 但是加深或者加宽(channel 维度升高) ResNet 提升的精度幅度较小.

<div style="width: 550px; margin: auto">![](https://wx2.sinaimg.cn/large/d7b90c85ly1fxc7ijoms4j20wc0jbgpa.jpg)

下标展示了残差结构的 shortcut 分别在 ResNet 和 ResNeXt 中的影响:

<div style="width: 550px; margin: auto">![](https://wx1.sinaimg.cn/large/d7b90c85ly1fxcbsa88wcj20y605ldgk.jpg)

**Performance:**

ResNeXt: 0.95s per mini-batch
ResNet-101: 0.70s per mini-batch

**Comparisons with SOTA results:**

<div style="width: 550px; margin: auto">![](https://wx3.sinaimg.cn/large/d7b90c85ly1fxc7k1g9sxj20w00hin03.jpg)

# Experiments on ImageNet-5K

<div style="width: 550px; margin: auto">![](https://wx1.sinaimg.cn/large/d7b90c85ly1fxc7jgb1hnj20wm0sowhr.jpg)

<div style="width: 550px; margin: auto">![](https://wx2.sinaimg.cn/large/d7b90c85ly1fxc7jq7pikj20wb0gdwhx.jpg)

# Experiments on CIFAR

<div style="width: 550px; margin: auto">![](https://wx1.sinaimg.cn/large/d7b90c85ly1fxc7l5pg68j20vv0p7adq.jpg)

<div style="width: 550px; margin: auto">![](https://wx2.sinaimg.cn/large/d7b90c85ly1fxc7kon4fpj20x509lwg5.jpg)

# Experiments on COCO object detection

<div style="width: 550px; margin: auto">![](https://wx1.sinaimg.cn/large/d7b90c85ly1fxc7lt20wyj20wo0arq4u.jpg)


<span id = "ResNeXt 在 ResNet 上做了哪些改进">
# ResNeXt 在 ResNet 上做了哪些改进

ResNeXt 实际上是将 ResNet Block 当中的输入数据的通道划分到了不同的组, 每个组的计算过程相对独立, 最终将所有组的计算结果进行空间聚合, 作为最终的输出. ResNeXt 可以在不增加参数量的情况下进一步提高 ResNet 的特征提出能力, 从而表现出更好的网络性能. ResNeXt 的卷积方式实际上可以看做是通道分组卷积.
