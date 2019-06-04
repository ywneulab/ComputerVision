---
title: CornerNet (ECCV, 2018)
sitemap: true
categories: 计算机视觉
date: 2018-09-19 13:06:59
tags:
- 计算机视觉
- 目标检测
- 论文解读
---

**文章:** CornerNet: Detecting Objects as Paired Keypoints
**作者:** Hei Law, Jia Deng
**机构:** University of Michigan


# 摘要
本文提出了一种新的检测方法, 即利用一对关键点(左上角和右下角)来检测物体, 我们仅用了单一的神经网络来搭建该模型, 并命名为 CornerNet. 通过一对关键点来检测物体, 可以省去设计 anchor boxes 的步骤. 除此之外, 我们还引入了 corner pooling, 这是一个新型的池化层, 可以帮助网络更好的定位 corner. 实验表明 CornerNet 在 MS COCO 上取得了 42.1% AP, 超过了所有的 one-stage 模型.

# Introduction

现有的 sota 目标检测方法中的一个很常用的组成元素是 anchor boxes. anchor boxes 用在 one-stage 方法中时, 可以取得与 two-stage 相媲美的性能表现.
**但是使用 anchor boxes 具有两个缺点:**
**第一,** 通常来说, 我们会生成大量的 anchor boxes. (eg, more than 40k in DSSD). 这是因为检测器是被训练来决定每一个 anchor box 是否与 gt box 有足够大的交集的, 因此, 必须有足够多的 anchor boxes 才能保证尽可能的覆盖所有的 gt boxes. 最终, 仅仅只有一小部分的 anchor box 会有 gt box 的交并比大于一定值, 因此, 这会造成正样本和负样本之间的不均衡, 同时会降低训练速度.
**第二,** 使用 anchor boxes 会引入很多的超参数和设计选择. 包括需要多少个 boxes, 它们的 sizes 是多少, 以及它们的宽高比是多少等等. 这些选择很大程度上是通过特别的启发式规则做出的, 并且在结合 multiscale 结构时会变得更加复杂
在本文中, 我们提出了一个舍弃了 anchor boxes 设计的 one-stage 检测模型, 命名为 CornerNet. 我们利用一对关键点(左上角和右下角)来检测物体. 我们利用一个单一的卷积网络来预测同一对象类别的所有实例的左上角的热图(heatmap), 所有右下角的热图, 以及每个检测到的角的嵌入向量. 我们的灵感来自于人体关键点检测的一篇文章. 图1展示了本文方法的整体流程(overall pipeline)

<div style="width: 550px; margin: auto">![图1](https://wx3.sinaimg.cn/mw690/d7b90c85ly1g160x4drirj20uu0in43a.jpg)

<div style="width: 550px; margin: auto">![图2](https://wx2.sinaimg.cn/mw690/d7b90c85ly1g160xg3qtzj20um0c6k65.jpg)

CornerNet 中另一个新颖的组成要素就是 **corner pooling**, 它可以帮助卷积网络更好的定位 bbox 的 corner. bbox 的 corner 经常处于物体的外侧, 如图2所示. 在这种情况下, 不能基于局部特征对某个 corner 进行定位. 相反的, 为了决定某个像素位置上是否存在一个 top-left corner, 我们需要水平向右寻找物体的最高边界, 同时需要垂直向下寻找最左边界. 这种特点启发我们提出了 corner pooling layer: 它包含两个特征图, 在每一个像素位置上, 它对第一个特征图谱右侧的所有特征向量进行最大池化操作, 对第二个特征图谱下方的所有特征向量进行最大池化操作, 然后将两个池化后的结果相加. 示例如图3所示.

<div style="width: 550px; margin: auto">![图3](https://wx1.sinaimg.cn/mw690/d7b90c85ly1g160xuod4vj20v00h5ac3.jpg)

**我们假设了两个原因** 来解释为什么检测 corners 会比检测 bbox centers 或者 proposals 的效果好. **第一**, 定位 anchor box 更难一些, 因为它需要依赖物体的4个边界才能确定, 但是 corner 只需要物体的两个边界就可以确定, corner pooling 也是如此, 它编码了关于 corner 的一些明确的先验知识. **第二**, corners 提供了一种更加有效的方式来密集的对 boxes 的空间进行离散化: 我们只需要 $O(wh)$ 的 corners, 而 anchor boxes 需要的复杂度是 $O(w^2h^2)$.

我们通过实验证明了 CornerNet 的有效性, 同时通过消融实验发现, corner pooling 对于 CornerNet 的性能表现来说很重要.

# Related Works

Two-stage object detectors: R-CNN, SPP, RPN, R-FCN
One-stage object detectors: YOLO, SSD, DSSD, RON, YOLO9000, RetinaNet, RefineDet
DeNet
Multi-person pose estimation

# CornerNet

在 CornerNet 中, 我们利用一对关键点(左上角和右下角)来检测物体. 卷积网络会预测两组热图(heatmaps)来表示不同物体类别的 corners locations, 一组用于表示左上角, 一组用于表示右下角. 同时, 网络还会为每个检测到的 corner 预测一个 embedding vector, 其特点是同一个物体的两个角点(corners)的 embeddings vector 之间的距离会非常小. 为了产生更紧密的边界框, 网络还会预测偏移量, 以稍微调整焦点的位置. 得到预测的 heatmaps, embeddings 和 offsets 之后, 我们会使用简单的后处理算法来获取最终的 bboxes.

<div style="width: 550px; margin: auto">![图4](https://wx2.sinaimg.cn/mw690/d7b90c85ly1g160y3jwiej20vr0dqgqg.jpg)

图4展示了 CornerNet 的整体结构, 我们使用 **hourglass 网络作为 backbone 网络**, 之后接着是两个预测模块(prediction modules). 其中一个用来预测左上角点, 另一个预测右下角点. 每一个模块都有自己的 corner pooling 来池化从 hourglass 网络得到的特征, 然后再预测 heatmaps, embeddings, 和 offsets. **和其他目标检测器不同, 我们不使用不同尺度的特征来检测不同尺度的物体. 我们仅仅使用 hourglass network 输出的特征进行检测.**

## Detecting Corners

我们预测两组热图, 每一组热图都具有 $C$ 个通道, $C$ 代表了物体类别的数量, 热图的尺寸为 $H\times W$. **注意, 这里没有背景的通道(background channel)**. 每一个 channel 都是一个二值的 mask, 用于指示某个类别的角点位置.
对于每个角点来说, 都会有一个 gt positive location, 而其他的 locations 都将是负样本. 在训练阶段, 我们不会等价的惩罚负样本, 我们在正样本的一定半径内减少对负样本的惩罚. 这是因为对于一对假的角点检测来说, 如果它们接近各自的真实位置, 那么就仍然可以产生一个与真实框差不多的框., 如图5所示. 我们通过确保半径内的一对点将生成一个至少与真实框具有 t IoU ($t=0.7$)以上的边界框, 从而根据物体的大小来确定半径的值. 在给定半径以后, 减少惩罚的数量通过一个非规范的 2D 高斯分布给出: $e^{-\frac{x^2 + y^2}{2\sigma^2}}$, 其中心位于正样本的位置, $\sigma$ 的大小时半径的三分之一.

<div style="width: 550px; margin: auto">![图5](https://wx1.sinaimg.cn/mw690/d7b90c85ly1g160yaycnlj20va0h8anr.jpg)

$$L_{det} = \frac{-1}{N}\sum_{c=1}^{C} \sum_{i=1}^{H} \sum_{j=1}^{W} \begin{cases} (1-p_{cij})^\alpha log(p_{cij}) && y_{cij} = 1 \\ (1-y_{cij})^{beta} (p_{cij})^\alpha log(1 - p_{cij}) && otherwise \end{cases} \tag 1$$

// TODO

$$o_k = \bigg(\frac{x_k}{n} - \bigg\lfloor \frac{x_k}{n} \bigg\rfloor, \frac{y_k}{n} - \bigg\lfloor \frac{y_k}{n} \bigg\rfloor \bigg) \tag 2$$

$$L_{off} = \frac{1}{N} \sum_{k=1}^{N} SmoothL1Loss(o_k, \hat o_k) \tag 3$$


## Grouping Corners

//TODO

## Corner Pooling

如图2所示, 我们往往不能从局部的特征中推断出角点的位置. 为了确定一个像素点是不是角点, 我们需要水平和垂直的向右向下看才能确定, 因此我们提出了 corner pooling 来更好的确定角点.


<div style="width: 550px; margin: auto">![图6](https://wx4.sinaimg.cn/mw690/d7b90c85ly1g160yqhvzfj20v10d8wfx.jpg)
<div style="width: 550px; margin: auto">![图7](https://wx1.sinaimg.cn/mw690/d7b90c85ly1g160z1k8tlj20v80fi76j.jpg)


## Hourglass Network

<div style="width: 550px; margin: auto">![表1](https://wx4.sinaimg.cn/mw690/d7b90c85ly1g160zfplqij20v509tgmw.jpg)
<div style="width: 550px; margin: auto">![表2](https://wx2.sinaimg.cn/mw690/d7b90c85ly1g160zthzyij20v50al761.jpg)
<div style="width: 550px; margin: auto">![表3](https://wx1.sinaimg.cn/mw690/d7b90c85ly1g1610d9bi9j20vv0cfgnu.jpg)
<div style="width: 550px; margin: auto">![图8](https://wx3.sinaimg.cn/mw690/d7b90c85ly1g1610lcom4j20va0bfn80.jpg)
<div style="width: 550px; margin: auto">![表4](https://wx4.sinaimg.cn/mw690/d7b90c85ly1g1610qpil9j20uz0nvjzl.jpg)
