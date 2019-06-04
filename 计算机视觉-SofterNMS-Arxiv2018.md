---
title: Softer-NMS-Arvix 2018
sitemap: true
categories: 计算机视觉
date: 2018-11-05 15:32:41
tags:
- 计算机视觉
- 目标检测
- 论文解读
---

**作者:** Yihui He, Xiangyu Zhang, Kris Kitani, Marios Savvides
**发表:**
**机构:** CMU & Face++


# 核心亮点

**提出了一种新的边框回归损失函数和NMS算法**
作者提出了一种 **基于KL散度的边框回归损失函数**, 可以同时学习到边框的形变量和位置变化量. 最终产生的位置变化量会与位置的精确度有很强的联系, 然后将其使用在本文提出的 **新的NMS** 算法上, 以此提高准确度.

# 论文细节

## 摘要

在目前的目标检测模型中, NMS是十分重要的一步处理步骤. 但是, 有时候, 较精确的候选框可能并没有很高的socre, 这时候使用NMS就会导致物体位置的预测精度降低. 在这篇文章中, 作者提出了一种 **新的边框回归损失函数**, 可以同时学习到边框的形变量和位置变化量. 最终产生的位置变化量会与位置的精确度有很强的联系, 然后将其使用在本文提出的 **新的NMS** 算法上, 以此提高准确度. 在MS-COCO数据集上, 将 VGG-16 Faster RCNN的 AP 从 23.6 提高到了 29.1, 将 ResNet-50 FPN Fast RCNN 的 AP 从 36.8 提高到了 37.8 .


## 介绍

目前, 目标检测模型主要分为one-stage和two-stage两种, 本文主要关注two-stage模型. 本文主要关注候选区域框可能出现的以下两种问题: 第一, 当物体周围所有的dounding box都是不准确的, 如图1(a)所示. 第二, 较准确的box的score不高, 而不准确box的score较高, 如图1(b)所示. 上面两种问题都说明了box的位置和box的score不是强相关的.

<div style="width: 600px; margin: auto">![](https://wx2.sinaimg.cn/large/d7b90c85ly1fwx9g5daq9j20k50x51kx.jpg)

收到这两种问题的启发, 本文提出使用 **KL loss** 来作为物体边框回归loss. 具体来说, 首先将bounding box的预测值和真实值分别建模成高斯分布和Dirac delta function(狄拉克 $\delta$ 函数). 然后, 训练模型, 以期降低来自于这两个分布的KL散度边框回归损失函数. 最后, 提出一个基于权重平均的soft NMS算法, 简言之就是当box具有较高的confidence事, 它就会得到较大的权重, 而不管它的分类score是多少.

## 利用KL Loss来训练边框回归模型

<div style="width: 600px; margin: auto">![](https://wx3.sinaimg.cn/large/d7b90c85ly1fwy9fjaaa9j20t50rlq61.jpg)

本文的检测模型的头部结构如图2所示. 我们的目的是估计边框的位置置信度, 具体来说, 我们的网络将会预测下面的高斯分布而不仅仅是边框回归:

$$P_\theta (x) = \frac{1}{2\pi \sigma^2}e^{-\frac{(x-x_e)^2}{2\sigma^2}}$$

上式中, $x_e$ 代表预测的边框的位置期望, $\sigma$ 代表标准差. 这些值将从Fast RCNN的头部(fc7)产生, 注意, fc7使用的是绝对值激活函数(而不是ReLU), 主要目的是尽量避免大量的 $\sigma$ 值为0., 当 $\sigma \rightarrow 0$ 时, 说明网络对当前预测的边框位置期望持有很大的置信度.

同样, 真实边框也可以写成高斯函数的形式, 实际上就是如下Dirac delta 函数:

$$P_D(x) = \delta (x - x_g)$$

其中, $x_g$ 是真实边框的位置.
我们的目标时找到使得 $P_\theta (x)$ 和 $P_D(x)$ 之间KL散度最小的参数 $\hat\theta$, 即:

$$\hat\theta = \arg\min_{\theta} D_{KL}(P_D(x) || P_{\theta}(x))$$

综上, 本文的模型将使用KL散度作为边框回归损失函数 $L_{reg}$, 分类损失函数 $L_{cls}$ 维持不变(与其他模型一样)

$$L_{reg} = D_{KL}(P_D(x) || P_{\theta}(x)) = ... = \frac{(x_g - x_e)^2}{2\sigma^2} + \frac{1}{2}log(\sigma^2) + \frac{1}{2}log(2\pi) + H(P_D(x))$$

如图3所示, 当预测位置 $x_e$ 不准确时, 我们就希望方差 $\sigma^2$ 越小越好, 这样一来, 损失函数 $L_{reg}$ 就会变小.

<div style="width: 600px; margin: auto">![](https://wx4.sinaimg.cn/large/d7b90c85ly1fwyaiiqvu6j20sa0qfgor.jpg)

//TODO

## Softer-NMS

在获取到预测位置的标准偏差以后, 通过平均权重将bounding boxes融合, 如下面的算法流程所示, 主要使用两行代码来修改原始的NMS算法.

<div style="width: 600px; margin: auto">![](https://wx3.sinaimg.cn/large/d7b90c85ly1fwyan7cz2tj20t00sltsp.jpg)

首先, 使用标准的NMS或者soft NMS算法来候选框进行选择. 然后, 对于每一个box $M$, 我们计算它基于周围及自身box的权重均值的新的location. 举个例子, 对于第 $i$ 个box 的坐标 $x1$来说, 它的新坐标 $x1_i$ 计算如下:

$$x1_i = \frac{\sum_j x1_j / \sigma^2_x1,j}{\sum_j 1/ \sigma^2_x1,j}, \text{subject to } IoU(x1_j, x1_i) > N_t$$

当bounding box的iou大于一定阈值 $N_t$ 时, 就会被考虑加入到权重均值当追溯去. 在这里, 我们不需要设置分类score的阈值, 因为即使是较低的socre有时它的localization socre却较高. 图4展示了在应用softerNMS以后, 我们有时候可以避免文章开头提高的两种错误情况.

<div style="width: 600px; margin: auto">![](https://wx1.sinaimg.cn/large/d7b90c85ly1fwyb7a79nyj20ty0rbb29.jpg)
