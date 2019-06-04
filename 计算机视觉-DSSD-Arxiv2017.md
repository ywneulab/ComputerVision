---
title: DSSD-Deconvolutional Single Shot Detector
sitemap: true
categories: 计算机视觉
date: 2018-10-28 15:50:15
tags:
- 计算机视觉
- 目标检测
---


# 核心亮点

**(1) 利用反卷积模块向特征图谱中添加更多的上下文信息**
主要是对SSD的一点改进, SSD使用了不同阶段的卷积特征图谱进行目标检测, 而DSSD受到人体姿态识别任务的启发, 将这些不同阶段的卷积特征图谱通过反卷积模块连接起来, 然后再进行目标检测的预测任务.
<div style="width: 600px; margin: auto">![](https://wx4.sinaimg.cn/large/d7b90c85ly1fwtpkdiie6j21480trdn7.jpg)

<div style="width: 600px; margin: auto">![](https://wx3.sinaimg.cn/large/d7b90c85ly1fwtrhla3r4j20mm0jxgnc.jpg)
**(2), 预测模块采用Residual模块**
这个不算是亮点, 不过也是改动之一, 基本来说就说原始的SSD是直接在特征图谱预测结果并计算损失的, 而DSSD在预测之前会先经过一个Residual模块做进一步的特征提取, 然后在进行预测.
<div style="width: 600px; margin: auto">![](https://wx2.sinaimg.cn/large/d7b90c85ly1fwtqpmp2rvj21cf0kejv9.jpg)

# 论文细节

## 摘要
这篇文章最主要的贡献在于提出了一个可以将额外的上下文信息引入到现有大多数目标检测模型的方法. 为了达到这个目标, 本文首先将Resnet101 分类网络和SSD模型结合起来, 然后对SSD+Resnet101的模型进行了扩展, 添加了反卷积层, 以此引入了针对更大范围尺度的目标物上下文特征信息, 进而提高了准确率(特别是在小物体上面). **这种方法在进行高度阐述时, 很容易就能讲明白, 但是在真正实现时, 却很难成功.** 因此本文精心的添加了一些具有已经学习好的特征信息的阶段, 特别是在反卷积时的前向计算的连接上, 最终使得上面的方法得以起效.


<div style="width: 600px; margin: auto">![](https://wx4.sinaimg.cn/large/d7b90c85ly1fwtpkdiie6j21480trdn7.jpg)

## DSSD 模型

### base network

如图1上半部分所示, SSD模型是用一个经典网络做为基础网络, 然后后接几层额外的特征层构建的. 原始的SSD采用的是VGGnet, 但是大量的工作都是用ResNet获得了更好的效果, 因此, 本文也选用ResNet-101网络作为backbone. 并且将额外的卷积层(也换成Residual模块)接在 conv5_x block后面, 同时会分别从 conv3_x, conv5_x两个卷积段预测score和offsets. **个人觉得奇怪的一点是, 为什么单单把VGG换成ResNet并没有提高mAP?(VOC数据集, 前者77.5, 后者76.4) 而是在使用了其他辅助模块后才提高的**

### prediction module

<div style="width: 600px; margin: auto">![](https://wx2.sinaimg.cn/large/d7b90c85ly1fwtqpmp2rvj21cf0kejv9.jpg)

MS-CNN指出, 提升每个任务的子网络有助于提高准确率, 根据这个原则, 我们在每一个预测层都添加了一个residual block, 如图2(c)所示. 同时也对其他形式的predictin module进行了尝试(图2,a,b,c).

### Deconvolutional SSD

为了增加更多高级别的上下文信息, 文章将prediction module 移动到了一系列的反卷积层之后, 并与原始SSD的额外卷积层形成了一种非对称的沙漏结构(hourglass network, 灵感来自于一篇人体姿态检测论文), 如图1所示. 每一个卷积层的特征图谱会和之前的层一起经过一个反卷积模块(该模块细节在后面介绍), 这就相当于在特征图谱中加入了更多的上下文信息. 这里的反卷积只有很浅的几层, 作者这样设计的原因一是不想增加过多的计算时间, 二是由于迁移学习方法可以帮助更好更快的模型收敛, 同时可以获得更高的精度, 因此无需设计过深的网络. **反卷积层很重要的一个点在于计算成本的增加, 除了反卷积操作本身的计算, 还体现在从之前层中添加信息时**

### Deconvolution Module

为了帮助整合反卷积层和之前层的特征信息, 文章引入了一种反卷积模块, 如图3所示.

<div style="width: 600px; margin: auto">![](https://wx3.sinaimg.cn/large/d7b90c85ly1fwtrhla3r4j20mm0jxgnc.jpg)

图3中左下角是从原始SSD中得到的特征图谱, 左上角是论文`Learning to refine object segment`中提出的反卷积层. 在当前模块中的每一个卷积层, 都使用了BN层, 在放大特征图谱时, 我们使用了学习到的反卷积层, 而不是双线性插值法. 在测试了多种conbination方法后(element-wise sum, element-wise product), 根据实验结果决定采用对应为相乘的结合方式(VOC数据集, 前者78.4, 后者78.6, 提升了0.2的mAP).

### Training

除了一些参数设置的不同外, 训练策略基本遵循SSD.


## 实验

<div style="width: 600px; margin: auto">![](https://wx3.sinaimg.cn/large/d7b90c85ly1fwtsusimmaj20o70b1dhb.jpg)

<div style="width: 600px; margin: auto">![](https://wx2.sinaimg.cn/large/d7b90c85ly1fwtsvfcjkfj21kw0n911g.jpg)

<div style="width: 600px; margin: auto">![](https://wx3.sinaimg.cn/large/d7b90c85ly1fwtsvsst7yj20s60ildjb.jpg)

<div style="width: 600px; margin: auto">![](https://wx2.sinaimg.cn/large/d7b90c85ly1fwtsw8f5f3j21kw0gxn37.jpg)

<div style="width: 600px; margin: auto">![](https://wx3.sinaimg.cn/large/d7b90c85ly1fwtswrk1sjj21kw0ji7an.jpg)

<div style="width: 600px; margin: auto">![](https://wx4.sinaimg.cn/large/d7b90c85ly1fwtsx845hej21kw0q9aix.jpg)

从COCO数据集的结果来看, DSSD在小物体方面并没有提升, 而在大物体方面获得了很大的提升, 推测原因主要是因为ResNet-101在大物体的特征提取能力上, 要远强于VGGNet.
另外, 可以看出, 由于采用了更深的ResNet网络, 同时增加了反卷积过程, 使得FPS降低不少.(即使在测试阶段利用计算公式移除了BN层, 这个trick可以提升1.2~1.5倍的检测速度).
