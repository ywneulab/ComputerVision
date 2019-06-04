---
title: OHEM (CVPR, 2016)
sitemap: true
categories: 计算机视觉
date: 2018-10-22 16:54:25
tags:
- 计算机视觉
- 目标检测
---

**文章:** Training Region-based Object Detectors with Online Hard Example Mining
**作者:** Abhinav Shrivastava, Abhinav Gupta, Ross Girshick

# 核心亮点

**提出了一种在线的难样例挖掘算法:**
作者根据每个RoIs的loss的大小来决定哪些是难样例, 哪些试试简单样例, 通过这种方法, 可以更高效的训练网络, 并且可以使得网络获得更小的训练loss. 同时, OHEM还具有以下两个优点:
- **消除FastRCNN系列模型中的一些不必要这参数** , 这些参数大多都是为了解决难样例问题服务的, 在使用OHEM以后, 不仅无需在对这些超参数进行调优, 同时还能获得更好的性能表现.
- **OHEM算法可以与其他多种提升模型精度的trick相结合**, 对于大多数模型(RCNN系列), 在使用了OHEM以后, 都能够获得精度上的提高, 可以看做是一种普适性的提升精度的方法.

注: 在实现OHEM上, 作者为了提升速度和效率, 特意设计了两个RoI网络, 以减少无用的计算.

# 论文细节

## 摘要
目前, 基于区域选择的CNN目标检测方法已经取得了很大的成功, 但是他们的训练过程仍然包含着许多启发法[注]和超参数(调优过程成本很高). 本文提出了一种针对区域选择目标检测方法的一种十分简单但非常有效的 **在线的难样例挖掘(OHEM)算法**. 我们的动机来源于数据集中存在的海量的简单样本, 而只有一小部分困难样本. 如果能够自动的选择这些困难样本用于训练, 那么就会使得训练过程更加高效和有效. OHEM 是一种简单且直观的算法, 它可以消除多个启发过程和超参数. 更重要的是, 它可以稳定的提升检测模型的算法性能, 当数据集越来越大, 越来越复杂时, 它的提升效果就越大.

[注] 启发法: 启发式方法(试探法)是一种帮你寻找答案的技术, 但它给出的答案是具有偶然性的(subject to chance), 因为启发式方法仅仅告诉你该如何去找, 而没有告诉你要找什么, 它并不会告诉你该如何直接从A点到B点, 他甚至可能连A点和B点在哪里都不知道. 启发式算法的难点是建立符合实际问题的一些列启发式规则, 启发式算法的有点在于它比盲目型的搜索法要高效, 一个经过仔细设计的启发函数, 往往在很快的时间内就可以得到一个搜索问题的最优解.
## 介绍
基于区域的CNN目标检测法使用的数据集中带物体标签的区域和背景区域之前的样本比例存在着巨大失衡. 在DPM中, 这种比例达到了1:100000. 一些算法(如Selective Search)对此进行了处理但失衡比例仍然很大(1:70).

bootstrapping(现在多称为 hard negative mining)问题至少已经存在了20年. 并且Bootstrapping技术已经在目标检测领域内流行了十几年(尤其是在训练针对目标检测的SVMs时). 很多现代的基于深度学习的目标检测方法都是用了基于难样例挖掘的SVMs来帮助训练检测模型(RCNN, SPPnet).

但是奇怪的是后来的一些模型(FastRCNN, FasterRCNN等) 都没有使用bootstrapping技术. 一个潜在的原因可能是在深度神经网络中, 存在一些技术上的困难, 是的bootstrapping使用效果不佳. 传统的bootstrapping需要先用一个固定的模型来找到新的样本已准备训练数据, 然后再用一个固定的样本将检测模型激活并训练. 而在训练深度神经网络时, 其需要成千上万的大量样本用于训练, 因此, 我们需要一个 **纯粹在线** 的难样例选择算法.

在本文中, 我们提出了一个新型的bootstrapping技术称为 **online hard example mining(OHEM)**, 并将其应用到基于深度学习的当前最先进的目标检测模型上. 该算法实际上是对SGD做了一些小改动, 使得训练样本可以从一个非均匀分布都采样得到, 该分布是一个基于当前样本loss的非静态分布. 该方法利用了目标检测问题特殊的结构优势, 那就是每一个SGD的mini-batch中仅仅包含一张或两张图片, 但是会有上千个候选样本. 这样候选样本会被继续按照特定的分布(倾向于那些不同的, 可以造成很高loss的样本)进行采样, 由于采样后的样本只是一小部分样本子集, 因此梯度下降优化算法仍然高效. 将OHEM算法用于标准的Fast RCNN算法以后,  显示除了三点好处:
- 移除了一些在基于区域推荐的CNN目标检测算法中的启发方法和超参数
- 稳定且大幅度的提升了mAP
- 当training set变的更大更复杂时, OHEM的有效性会提升

不仅如此, 从OHEM中获得的性能提升是对最近目标检测领域其他提升性能方法的一种补充, 如multiscale testing和迭代式bounding box回归. 在使用OHEM的同时结合这些tricks, 可以更进一步的提升mAP.

## 相关工作

绝大多数目标检测模型都使用了结合bootstrapping算法的SVMs来作为检测的scoring function.
但是有一些特例, FastRCNN和FasterRCNN等没有使用结合bootstrapping的SVMs, 而是完全根据SGD进行训练, 这些模型通过引入一些在线的难样例挖掘算法来解决这个问题(送入minibatch的前后景比例1:3). 下面简单介绍一下 Hard example mining和CNN目标检测以及他们之间的关系

**Hard example mining:** 目前常用的有两大算法.
第一种是用于优化SVMs的: 训练算法会维护一个工作样本集, 并且该样本集会依据特定的规则不断在训练SVMshe更新样本集之间迭代. 这些规则会将简单样本(很容易分类正确)从样本集中移除, 同时会添加困难样本到样本集中.
第二种方法用于非SVMs模型, 如浅层神经网络和boosted决策树: 该算法会从正样例和一部分随机负样例组成的数据集开始训练, 在该数据集上训练至收敛以后, 会继续在一个更大的, 包含更多负样例的数据集上进行训练. 这个过程通常只会进行一次迭代, 并且没有任何的收敛性证明.

**ConvNet-bases detection:** 近年来CNN在目标检测领域迅速, 尤其是基于深度学习的目标检测方法(SPPNet, MRVNN Fast RCNN).

**Hard example selection in deep learning:** 目前已经有很多专门针对目标检测算法hard example mining方法. 这些算法的基本思路与我们相同, 但是我们的算法更关注基于区域推荐的目标检测算法的 **在线难样例挖掘**.

## Overview of Fast RCNN

在Fast RCNN中, 每一个minibatch包含N=2张图片, 每一张图片会采样B/N=128/2=64个候选框. RoI sampling过程使用了多个启发式规则, 如下所示:
- **Foreground RoIs:** 每一个RoI都会根据与真实框的IOU大小来分成前景框和后景框
- **Background RoIs:** 在[`bg_lo`,0.5) 区间的被认为是后景, `bg_lo`(FastRCNN为0.1)用来充当难样例挖掘的角色, 但是这样会忽视一些不频繁但是很重要的后景区域, OHEM移除了`bg_lo`阈值.
- **Balancing `fg-bg` RoIs:** 为了处理难样例问题, FastRCNN采取的策略是将minibatch中前后景的比例设置为 1:3 (25%为前景). 这是一个十分重要的启发式规则阈值, 在实验中, 将他移除或者改变都会引起mAP分数的大幅度下降(3 points). **但是利用OHEM, 就可以在不损失mAP的情况下, 移除这个超参数.**

## Our approach

本文提出的OHEM算法可以简化复杂的FastRCNN系列模型的训练及调参过程, 同时可以获得更好的训练结果(lower training loss)和更高的测试性能(higher mAP)

### Online hard example mining

在RCNN中使用SVMs,大致有两个阶段, 阶段 a) 首先会筛选并处理指定数目(10or100)个图片, 然后在在这些图片进进行训练直到收敛. 重复这两个阶段直到SVMs找到所有的支持向量为止. RCNN所采用的这个优化策略效率是很低的, 因为在对图片进行筛选和处理的时候, 没有任何模型会进行更新.


OHEM算法流程如下: 对于处于第t次SGD迭代的输入图片来说, 首先计算器卷积特征图谱. 然后令RoI网络使用这个特征图谱和 **所有** 的RoIs (不是minibatch子集, 而是所有), 进行前向传播(仅包含RoI pooling层和几层fc层). 由此计算出的loss代表了当前网络在每一个RoI上的表现好坏. Hard examples的选择方法是将每个RoI的loss进行排序, 然后选择loss最大的 B/N 个样本作为hard examples. 直观上可以看出, 这些样本对于当前网络来说, 是最难正确分类的样本. 同时, **由于RoI pooling层本身计算量不高, 且各个RoI之间的计算可以共享**(//TODO,怎么共享??), 因此, 额外的计算成本相对来说并不高. 此外, 参与反向计算过程的样本集合很很小的一个集合, 因此, 相比以前的方法并不会带来更多的计算成本.
但是, 这里有一个小警告: 如果两个RoI之间的overlap较大, 那么很有可能这两个RoI的loss直接会有一定关联程度. 也就是说, 这些重叠度较高的RoIs可以映射到特征图谱上的同一块区域中, 又因为分辨率之间的差异性, 就可能导致loss的重复计算. 为了解决这种冗余计算和关联区域, 我们使用NMS算法来降低重复性. 具体来说, 给定RoIs列表和它们对应的losses, NMS迭代的选择那些具有最高loss的RoI, 然后移除所有与高RoI重叠度较高(IOU>0.7)的其他RoI(这些RoI的loss更低).
可以看出, 上面的过程并不需要`fg-bg`比例这个超参数, 如果任何一类被忽视了, 那么这个类对应的loss就会一直升高, 知道这个类被采样的概率变大为止.  对于有些图片来说, 前景区域是简单样例, 此时网络就会只采用背景区域进行训练, 反之, 也有的图片会认为背景区域(草地, 天空)是简单样例.

### Implementatin details
实现OHEM算法的方法有很多, 它们之间有着不同的权衡和考量. 一个明显的方法是修改loss层, 使其完成难样例的选取工作. loss层可以计算所有RoIs的loss, 然后将它们进行排序,并且根据loss的值进行难样例的选择, 同时将所有非难样例的RoIs的loss设置为0 (表示梯度下降时不会考虑这些样例). 这种方式很直接, 但是却不够高效, 因为RoI网络仍然需要申请空间来存储这些RoIs, 并且还要对所有的RoIs进行反向传播计算(非难样例的loss虽然为0, 但也要参与计算流程).
为了克服上面的效率问题, 本文提出了一种结构如图2所示. 我们的实现包含两个RoI网络的副本, 其中一个是 **只读** 的. 这意味着我们的只读RoI网络只需要申请前向传播计算的内存, 而不用像经典RoI网络一样, 需要同时申请前向计算和反向传播时的内存. 对于一次SGD迭代来说, 给定卷积特征图谱, 只读RoI网络会对所有的RoIs执行前向计算并计算loss( $R$ , 图2中的绿色箭头所示). 然后难样例采样模块会按照之前说的采样策略来选择难样例, 然后会将这些难样例输入到一个常规的RoI网络中( $R_{hard-sel}$ 图2中的红色箭头所示). 于是, 这个网络仅仅只会对难样例同时进行前向计算和反向传播计算. 在实际中, 我们使用所有图片的所有RoI作为 $R$, 因此对于只读RoI网络最后的minibatch的size为 $|R|$. (//TODO 嘛意思?)
在实验中, N=2(意味着 $|R|\approx 4000$ ), B = 128.

<div style="width: 600px; margin: auto">![](http://wx4.sinaimg.cn/large/d7b90c85ly1fwi9ig4pthj213t0ls7fq.jpg)


## 实验与分析


### OHEM vs. heuristic sampling
如下表所示, 没有使用 `bg_lo` 超参数的OHEM在FasrRCNN上的mAP提高了4.8个点

<div style="width: 600px; margin: auto">![](https://wx2.sinaimg.cn/large/d7b90c85ly1fwj7f29xztj20st0kyadj.jpg)

### Robust gradient estimates
因为N=2, 所以选出来的框之间很有很大的关联性, 这会不会使得梯度不稳定从而难以收敛. 实验结果表明, OHEM同样对这种情况具有鲁棒性, 即使将N设置为1, mAP也不会降低多少(仅降低了0.2%). (当然, 在硬件条件允许的情况下, 选取越大的N, 效果一般越好)

### Why just hard examples, when u can use call?
如果使用所有的RoI loss参与权重更新(而不仅仅是hard examples)会怎么样呢? 在这种情况下, 那些简单样本将会拥有较小的loss, 从而不会对梯度的更新贡献太大, 整个训练过程会自动的专注于难样例的训练.(这种全训练的结果会使得相对于标准FastRCNN的mAP提升一点, 因为标准的FastRCNN并没有针对难样例, 而是随机选的minibatch), 但是这种全训练很是的训练时间大幅提高.

### Better optimization

利用OHEM可以获得更低的 mean loss per RoI  (取自各种方法的第20K次迭代)

<div style="width: 600px; margin: auto">![](https://wx4.sinaimg.cn/large/d7b90c85ly1fwj89dxiz3j20s40m0jvi.jpg)
### Computational cost

不论是耗时上还是内存占用上, 都变多了, 但总体来说, 是可以接受的

<div style="width: 600px; margin: auto">![](https://wx4.sinaimg.cn/large/d7b90c85ly1fwj89qxmy9j20s20c10un.jpg)


# PASXAL VOC and MS COCO results

**在训练阶段和测试阶段使用multi-scale和multi-stage bbox regression** 可以大幅度提高mAP

multi-scale: $s\in \{ 480, 576, 688, 864, 900 \}$ for training, $s\in \{480, 576,688,864,1000\}$ for testing. 这些scales和caps的选择主要受制于GPU的显存大小.
