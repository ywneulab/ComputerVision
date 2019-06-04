---
title: RFB Net (ECCV, 2018)
sitemap: true
categories: 计算机视觉
date: 2018-11-19 13:06:59
tags:
- 计算机视觉
- 目标检测
- 论文解读
---

**文章:** Receptive Field Block Net for Accurate and Fast Object Detection
**作者:** Songtao Liu, Di Huang, and Yunhong Wang
**备注:** Beihang University

# 核心亮点

**本文从感受野大小的角度出发, 提出了 RFB 模块, 可以融合多个感受野特征, 进而提升轻量级网络(SSD)的特征表达能力**
相比于不断增加模型复杂度(深度,宽度)来增强特征的表达能力, 本文通过一种人工设计的机制来增强轻量级模型的特征表达能力, 以期获得一种既快又好的检测模型.

<div style="width: 600px; margin: auto">![](https://wx2.sinaimg.cn/large/d7b90c85ly1fxdebnztlgj215z0tw4qp.jpg)


# 摘要

目前精确度最高的目标检测模型往往面临着巨大的计算开销, 而轻量级的目标检测模型在精度上却不够高. 本文通过利用人工设计来增强轻量级模型的特征, 以期获得一个既快又好的检测模型. 受到人类视觉系统感受野的启发, 文本提出了一个感受野模块(RF Block module), 它将 RFs 的 size 和 eccentricity 之间的关系考虑在内, 来增强特征的分辨能力和鲁棒性. 之后, 我们将 RFB 集成到了 SSD 之中, 建立了一个 RFB Net 检测器. 实验结果显示, 本文的 RFB Net 可以达到目前最高的性能表现.

# 介绍

通过讨论 two-stage 和 one-stage 模型各自的特点和优势, 本文发现, 相比于不断增加模型复杂度(深度,宽度)来增强特征的表达能力, 另一种可选的做法通过一种人工设计的机制来增强轻量级模型的特征表达能力, 以期获得一种既快又好的检测模型. 另一方面, 多项研究发现, 感受野的大小是视网膜图谱离心率的函数, 并且在不同的图谱上, 离心率会逐渐升高, 如图1所示.

<div style="width: 600px; margin: auto">![](https://wx3.sinaimg.cn/large/d7b90c85ly1fxdebh1h8ej20zo0u01kx.jpg)

目前的深度网络模型, 大多将不同层之间的感受野设置成相同大小, 这就有可能降低提取到的特征的表达能力. Inception 系列虽然融合了多个尺寸的感受野特征, 但是所有的卷积核仍然是在同一个中心进行采样的. Deformable CNN 尝试自适应的改变卷积核采样点, 尽管采样网格十分灵活, 但是依然没有考虑感受野的 eccentricity 属性, 这使得所有处于感受野当中的像素点都都输出响应具有同等的贡献度, 这会导致一些更重要的信息没有被突出出来.
本文根据人类视觉感受野的机制, 提出了 Receptive Field Block(RFB), 来增强轻量级的特征学习能力, 使得他们可以组建出更快更好的检测模型. 具体来说, RFB 通过使用不同大小的卷积核来实现多分支的 pooling 操作, 应用空洞卷积(dilated convolution)来控制它们的离心率(eccentricities), 并且进行 reshape 之后生成最终的特征表示, 如图2所示. 之后, 我们会将该模块集成到 SSD 网络之中, 形成一个更快更好的 one-stage 目标检测模型, 称为 RFB Net.

本文的贡献主要有以下三点:
- 提出了 RFB 模块, 用于提升轻量级 CNN 网络的特征表达能力
- 提出了基于 RFB Net 的目标检测模型, 并且通过实验证明了该模型可以在维持 one-stage 模型(SSD)复杂度的条件下增强模型的精度.
- 实验表明本文的 RFB Net 可以在实时运行的速度下在 VOC 和 COCO 数据集上达到 SOTA 的性能, 并且证明了 RFB 模型具有很好的泛化性能(可以连接到 MobileNet 之上).

# 相关工作

**Two-stage detector:** RCNN, Fast, Faster, R-FCN, FPN, Mask R-CNN
**One-stage detector:** YOLO, SSD, DSSD, RetinaNet
**Receptive filed:** Inception(多个感受野尺寸共同作用), ASPP, Deformable CNN. 图3展示了这三种方式与本文的 RFB 的区别.

<div style="width: 600px; margin: auto">![](https://wx3.sinaimg.cn/large/d7b90c85ly1fxdebsf4n4j21090omx50.jpg)

# Receptive Field Block

本文提出的 RFB 是一个多分支的卷积块, 其内部结构主要包含两个部分: 具有不同卷积核大小的多分支的卷积层, 以及紧跟其后的空洞池化或空洞卷积层. 前一部分和 Inception 相同, 复杂模拟不同尺寸的感受野, 后一部分生成 pRFs(population Receptive Fields) 尺寸和人类视觉离心率之间的关系.

**Multi-branch convolution layer:** 本文使用 Inception V4 和 Inception-ResNet V2 来构成多分支卷积层.
**Dilated pooling or convolution layer:** 也称为 astrous convolution layer.

图4展示了本文的 RFB 模块示意图.

<div style="width: 600px; margin: auto">![](https://wx2.sinaimg.cn/large/d7b90c85ly1fxdeco9ni3j21iv0ovgsf.jpg)

# RFB Net Detection Architecture

本文的 RFB Net 是基于 SSD 和 RFB 模块进行构建的, 其中, RFB 模块会嵌入到 SSD 网络中, 主要的改动在于将 SSD 顶层(head/top)的卷积层替换为 RFB 模块, 如图5所示.

<div style="width: 600px; margin: auto">![](https://wx3.sinaimg.cn/large/d7b90c85ly1fxded8xzv5j21540m0dml.jpg)

**Lightweight backbone:** 保持 SSD 的选择, 使用 VGG16 作为 backbone. (即使还有其他的选择, 但是为了与 SSD 形成对比, 决定选择 VGG16).

**RFB on multi-scale feature maps:** 在 SSD 中, 使用了多个不同大小的卷积特征图谱参与预测, 在本文的实现中, 会将较大的特征图谱后接的卷积层替换为 RFB 模块. 如图5所示.

# Training Settings

- framework: Pytorch
- strategies: follow SSD, 包括数据增广, 难负样例挖掘等等
- new conv-layers: MSRA initialization

# 实验

## Pascal VOC 2007

<div style="width: 600px; margin: auto">![](https://wx2.sinaimg.cn/large/d7b90c85ly1fxdedqo5m1j215a0nagrk.jpg)

## 消融实验(Ablation Study)

<div style="width: 600px; margin: auto">![](https://wx3.sinaimg.cn/large/d7b90c85ly1fxdeelkkfgj215c0fhtbb.jpg)

<div style="width: 600px; margin: auto">![](https://wx3.sinaimg.cn/large/d7b90c85ly1fxdef2wr70j216909w0v7.jpg)

## COCO

<div style="width: 600px; margin: auto">![](https://wx2.sinaimg.cn/large/d7b90c85ly1fxdeflviorj215f0qa46c.jpg)

<div style="width: 600px; margin: auto">![](https://wx1.sinaimg.cn/large/d7b90c85ly1fxdeg2ul5kj215m0rn0xc.jpg)

## Other BackBone
<div style="width: 600px; margin: auto">![](https://wx4.sinaimg.cn/large/d7b90c85ly1fxdegbnnovj219107gjtf.jpg)
