---
title: CoupleNet-Coupling Global Structure with Local Parts for Object Detection
sitemap: true
categories: 计算机视觉
date: 2018-11-03 13:07:59
tags:
- 计算机视觉
- 目标检测
- 论文解读
---

# 核心亮点

**在进行区域分类时, 同时使用了全局信息,上下文信息和局部信息综合判断**
提出了一个新颖的全卷积网络, 并称之为CoupleNet, 它可以在目标检测中结合使用全局和局部信息. 具体来说, CoupleNet会将由RPN网络产生的候选区域送入到coupling module中, 该模块包含两个分支. 第一条分支利用position-sensitive RoI pooling来捕获物体的局部信息, 另一条分支利用RoI pooling对全局上下文信息进行编码. 接着, 我们设计了不同的coupling策略和归一化方法来使用这些不同分支格子的优势.

# 论文细节

## 摘要
尽管R-FCN在保证检测准确度的同时, 取得了更快的检测速度, 但是position-sensitive score maps的设计依然忽略了图中的整体结构的全局信息. 为了充分利用并结合局部和全局信息, 本文提出了一个新颖的全卷积网络, 并称之为CoupleNet, 它可以在目标检测中结合使用全局和局部信息. 具体来说, CoupleNet会将由RPN网络产生的候选区域送入到coupling module中, 该模块包含两个分支. 第一条分支利用position-sensitive RoI pooling来捕获物体的局部信息, 另一条分支利用RoI pooling对全局上下文信息进行编码. 接着, 我们设计了不同的coupling策略和归一化方法来使用这些不同分支格子的优势. 最终, 本文的模型达到了SOTA.

## 介绍

典型的基于候选区域的检测算法如FasterRCNN使用了单独的网络来生成候选区域, 这使得检测速度很慢, 而R-FCN利用PSRoI pooling(position-sensitive RoI)对其进行了改进, 在保证精度的情况下获得了更快的检测速度. 但是, R-FCN网络依然没能利用到全局结构信息, 如图1所示, 当只利用局部信息时, 检测框内物体对沙发的预测概率只有0.08, 这显然是是不合理的, 而如果只利用全局信息, 也只能得到0.45的预测概率, 但是如果结合这两部分信息, 就能得到0.78的预测结果, 我们更乐意接受这个结果.

![](https://wx2.sinaimg.cn/large/d7b90c85ly1fwuuq0z3b6j20sn0nwk3a.jpg)

本文的主要贡献有以下三点:
1. 本文提出了一个统一的全卷积网络, 可以联合地学习到目标检测任务中的局部信息, 全局信息和相关的上下文信息
2. 本文设计了多个不同的归一化方法和coupling策略, 用以挖掘全局信息和局部信息之间的兼容性和互补性
3. 本文的模型在三个主流数据集(VOC07,VOC12,COCO)取得了SOTA

## CoupleNet

### 网络结构

![](https://wx4.sinaimg.cn/large/d7b90c85ly1fwuvmld7psj21kw0t4agk.jpg)

CounpleNet的网络结构如图2所示, 主要包含两条不同的分支:
1. 一个局部的part-sensitive全卷积网络, 用于学习特定物体的局部信息, 记为local FCN;
2. 一个全局的region-sensitive全卷积网络, 用于对物体整体结构的全局信息和上下文信息进行编码, 记为global FCN.
本文首先利用ResNet-101 (移除了最后的全局平均层和fc层)对图片进行卷积操作, 得到相应的特征图谱, 并利用RPN网络得到相应的候选区域, RPN网络与后续的CoupleNet共享特征图谱计算结果. 然后conv5上对应的候选区域会流向两个不同的分支: local FCN 和 global FC. 最后, 从 local FCN 和 gocal FCN 中得到的结果会被结合在一起, 作为最终的物体socre输出.

### Local FCN

![](https://wx4.sinaimg.cn/large/d7b90c85ly1fwvy0czkkbj20jd0prts4.jpg)

为了在local FCN高效的捕获特定区域的信息, 本文通过利用通道数为 $k^2(C+1)$ 的 $1\times 1$ 的卷积层构建了一系列的 part=sensitive socre map, 其中 $k$ 代表我们将物体划分成 $k\times k$ 个局部部位(local parts), $C+1$ 代表类别. 因此, 对于任意一个类别, 都会有 $k^2$ 个通道, 并且每一个通道会负责物体的一个特定局部部位. 最终的类别score将由这 $k^2$ 个结果投票产生. 这里, 我们使用了 R-FCN 的 position-sentitive RoI pooling 层来提取物体的特定部位, 并且是三简单的平均池化来进行投票. 如此一来, 我们就会得到一个 $C+1$ 维度的向量, 代表着当前候选区域属于每一类的概率. 这个过程相当于是把一个对物体类别的强分类器转换成了许多弱分类器, 如此便可以起到ensemble part models的作用, 使得分类更加准确. 如图3(a)所示, 对于一个被裁剪的人像来说, 神经网络对人的全局信息无法很高的响应, 但是如果仅从局部特征角度出发, 如人的鼻子, 眼睛等, local FCN可以十分高效的捕获到这些特定区域的特征. 因此, 我们认为 local FCN 更加关注物体的内部结构和组成要素等信息, 这些信息可以高效的反映出物体的局部属性, 特别是当物体被遮挡或者整体边界不完整的情况. 但是, 对于那些具有简单空间结构以及那些包括了相当多背景区域的物体来说(如, 餐桌), 单单只靠 local FCN 很难做出鲁棒性较高的预测结构. 因此有必要加入全局结构信息来增强网络的分辨能力.

### Global FCN

对于 global FCN, 本文通过使用整体的区域级别的特征来描述物体的全局信息. 首先, 我们将一个 1024 维度的 $1\times 1$ 卷积层接在ResNet101的最后一个残差模块之后, 用于降低维度. 由于候选区域的尺寸不唯一, 因此, 本文会插入一个 RoI pooling 层来提取固定长度的特征向量作为物体的全局结构信息. 第二, 本文使用了两个卷积层(分别为 $k\times k$ 和 $1\times 1$)来更进一步的抽象物体的全局信息. 最后, $1\times 1$ 的卷积结果会被送入分类器, 分类器的输出也是一个 $C+1$ 维度的向量(与local FCN一样).

此外, 上下文信息是视觉识别任务中最基本,最重要的元素之一. 例如, 船通常是在水里的, 而不太可能是在空中的. 尽管, 在深层网络中, 较深的网络的特征图谱具有更大的感受野, 可以相对获得更多的空间上下文信息, 但是实际中深层特征图谱所包含的上下文信息要理理论上少很多. 因此, 很有必要显式的去收集物体的周围信息, 以减少错分类的可能性. 为了增强 global FCN 的特征表达能力, 本文将上下文信息引入到网络中作为一种有效的附加信息. 具体来说, 本文将物体的RoI区域扩大为原来的2倍. 然后将这两种RoI(原始的和扩大后的)通过RoIpooling后再连接在一起(concatenated), 接着送入之后的子网络.(如图2后下部分所示, 实际上, global分支可以看做是一种特殊的FasterRCNN).

由于RoI pooling操作, global FCN可以将物体区域作为物体的整体特征进行描述, 因此, 它可以轻松的处理那些完整的物体, 如图3(b)所示.

### Coupling structure

为了让global FCN 和 local FCN 返回的结果在数量级上匹配, 本文对它们的输出使用了归一化操作. 主要利用了两种方案来进行归一化: L2归一化层或者 $1 \times 1$ 卷积层. 同时, 如何将local和global输出结合起来也是一个关键问题. 在本文中, 我们探究了三种不同的coupling方法: 对应位相加(element-wise sum), 对应位相乘(element-wise product), 以及对应位取最大(element-wise maximum). 实验结果表明, 使用 $1\times 1$ 卷积配合对应位相加可以取得最好的实验效果.

## 实验

使用L2归一化效果很差(甚至比不使用归一化的结果还要差), 推测原因可能是L2归一化以后会使得score之间的gap变小, 进而造成错分类. 而使用 $1\times 1$ 卷积进行归一化时, 网络会自动学习并调节local和global归一化以后的尺寸大小.

对于coupling策略的选择, 对应位相加是一种很有效的连接方式, 在ResNet也使用了这种连接, 而对应位相乘有时候会造成梯度的不稳定, 从而导致训练难以收敛. 对应位取最大则会丢失掉更多的信息, 同时也就丢失了结合局部和全局信息的优势.

正如我们之前讨论的那样, CoupleNet在面对遮挡, 截断以及包括大量背景的目标时(如沙发,人,桌子,椅子等等), 可以表现出很强的识别能力.

![](https://wx2.sinaimg.cn/large/d7b90c85ly1fwvy0gxc17j20s00fn0vy.jpg)

![](https://wx1.sinaimg.cn/large/d7b90c85ly1fwvy0kvkn0j21kw0g27ax.jpg)

![](https://wx4.sinaimg.cn/large/d7b90c85ly1fwvy0p2vmrj21kw0iphdt.jpg)

![](https://wx2.sinaimg.cn/large/d7b90c85ly1fwvy0rsfv2j20rt0e276x.jpg)

![](https://wx3.sinaimg.cn/large/d7b90c85ly1fwvy0vpsavj21kw0iphdt.jpg)

![](https://wx4.sinaimg.cn/large/d7b90c85ly1fwvy0yl1ujj21kw0hkn4k.jpg)
