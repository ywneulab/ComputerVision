---
title: RefineDet (CVPR, 2018)
sitemap: true
categories: 计算机视觉
date: 2018-11-29 14:43:10
tags:
- 计算机视觉
- 目标检测
- 论文解读
---

**文章:** Single-Shot Refinement Neural Network for Object Detection
**作者:** Shifeng Zhang, Longyin Wen, Xiao Bian, Zhen Lei, Stan Z.Li


# 论文亮点:

**结合了one-stage方法和two-stage方法各自的优势, 提出了一个基于single-shot的检测模型:**
模型主要包含两大模块, 分别是anchor精化模块和物体检测模块. 网络采用了类似FPN的思想, 通过 Transfer Connection Block 将特征图谱在两个模块之间传送, 不仅提升了的精度, 同时还在速度方面取得了与one-stage方案相媲美的表现

<div style="width: 600px; margin: auto">![](https://wx4.sinaimg.cn/large/d7b90c85ly1fwp44jaiyij213h0o2n1j.jpg)

# 论文细节

## 摘要

本文提出了一个基于single-shot的检测模型, 称为RefineDet, 它在精度上高于现有的two-stage方法, 同时, 可以和one-stage方法的速度相媲美. RefineDet包含两个内部连接的模块(如图1):
- anchor精化模块(anchor refinement module):
    1. 过滤掉负样本的anchors, 以减少分类器的搜索空间;
    2. 对anchors的位置和size进行粗糙的调整, 以便为后续的回归网络提供更好的初始化状态.
- 物体检测模块(object detection module):
    - 用refined anchors作为输入进行回归预测.
    - 同时, 设计一个`传送连接模块(transfer connection block)`, 将anchor refinement module里面的特征进行传送, 以此来预测框的位置, size 和 类别标签.

由于本文使用了多任务联合损失函数, 因此可以进行端到端的训练.

## 介绍

在作者看来, 目前的two-stage方法(Faster RCNN, R-FCN, FPN), 相比于One-Stage方法来说具有三个优势:
- 具有启发式规则来处理正负样本不均衡问题
- 具有两个级联的物体边框回归阶段(边框更加精确)
- 提取了更加丰富的物体特征(anchor使得提取过程更精细)

为了结合One-Stage和Two-Stage方法的优势, 同时克服他们的缺点, 本文的RefineDet设计了两个内部连接的模块: anchor refinement module(ARM) 和 object detection module(ODM).(如图1所示)

## 网络结构

网络结构的整体视图如图1所示, 和SSD类似, RefineDet会基于前向传播网络预测出固定数量的bounding box和对应的类别score. 本文的网络主要包含ARM和ODM两大部分.

**ARM:** 对经典网络结构(VGG-16, ResNet-101)进行改造, 去掉分类层, 并加上一些附属结构
**ODM:** ODM由TCBs (Transfer Connection Block)和预测层(3×3 卷积层)组成, 会输出物体的类别score和相对于refined anchor box的相对位置坐标.

**Transfer Connection Block:** 用于链接ARM和ODM, 引入TCBs的目的主要是为了将ARM中不同层的特征转换成ODM接受的形式, 这样一来,ODM和ARM就可以共享特征向量. 值得注意的是, 在ARM中, 本文仅仅对于anchor相关的特征图谱使用TCBs. 其实很像FPN的想法, 但是与它又不太一样. TCBs的结构如图2所示

<div style="width: 600px; margin: auto">![](https://wx2.sinaimg.cn/large/d7b90c85ly1fwp6b5h1hxj20ia0iemym.jpg)

**Two-Step Cascaded Regression:** 当前的one-stage方法仅仅依赖于一次边框回归过程, 其主要是基于在不同尺度的特征图谱上来预测不同size的物体的位置, 因此预测的精度较低, 尤其是在面对小物体时. 因此, 本文的模型采用了Two-Step Cascaded Regression. 首先, 利用ARM来调节anchor的位置和大小, 以便为后续的ODM的回归预测提供更好的anchor初始状态. 具体来说, 我们会从特征图谱中的每一个cell上得到n个anchor boxes, 最开始的时候, 每一个anchor box的位置相对于它的cell来说都是固定的. 在每一个特征图谱的cell上面, 我们都会预测4个相对坐标(refined网格相对于origin网格的位移坐标). 因此, 我们可以在每一个cell上面, 产生n个refined anchors.

在获得了refined anchor boxes以后, 我们将它们传送到对应的ODM中去(不同的feature map对应不同的ODM) 来预测物体类别,边框位置和大小等信息. 互相关联的ARM和ODM具有相同的维度, 对于每一个anchor box, ODM都会生成 $c+4$ 个输出( $c$ 为物体类别数 ).  这个预测过程与SSD很相似, 但是与之不同的是本文使用了Refined anchor boxes, 从而可以获得更精确的结果.

**注意:** 这里 RefineDet 和 SSD 一样, **没有使用 RoI Pooling**, 而是直接在 feature map 上中每个位置上, 都预定义了固定数量的 default boxes.

**SSD 与 RefineDet 的另一区别是:** SSD 仅仅使用了 one-stage 的 default box 的预测方案, 而 RefineDet 对 anchor 的调整是 two-stage 的, ARM 会先进行 objectness 的二分类预测和回归, 然后 ODM 会在基于 refined anchor 进行 object class 的多分类预测和回归.

**Negative Anchor Filtering:** 同以往检测模型一样, 本文不希望训练过多的(容易分类的)简单样本, 同时需要减轻样本不均衡问题, 因此, 本文设计了一个 negative anchor过滤机制. 具体来说, 就是在训练阶段, 对于一个refined anchor box, 如果它的负样本概率大于一个阈值(如0.99), 那么我们在训练 ODM 的时候就会忽略这个 refined anchor box, 具体代码实习时就是在匹配的时候, 将背景分类预测值大于 0.99 的直接置为 -1 即可. 这样一来, 网络只会训练 **难负样本(refined hard negative anchor boxes)** 以及 所有的 **正样本(refined positive anchor boxes)**.  同样, 在预测阶段, 对于大于阈值的负样本, 也会对其放弃检测.

## 训练和预测(Training and Inference)

**Data Augmentatin:** 采用了和SSD相同的数据增广方法

**Backbone Network:** 使用了 VGG-16 和 ResNet-101 作为骨架网络, 分别在两个网络后面多加了一些卷积层或者残差模块, 以提取更高level的特征.

**Anchors Design and Matching:** 为了处理不同的物体尺度问题, 本文选择了4个特征层, stride size分别为8, 16, 32 和 64.

**Hard Negative Mining:** 采用了和SSD类似的难样例挖掘算法.

**Loss Function:** RefineDet 的损失函数包含两部分, 即ARM的损失和ODM的损失. 对于ARM损失来说, 我们给每个anchor赋予一个二值标签(是或不是物体), 同时会对 anchor 的 size 和 scale 进行回归来得到 refined anchor. 之后, 我们会将 refined anchors 送入 ODM 模块, 来更进一步的预测物体的类别和更加精确的物体的 locations 和 size. 从这个定义可以知道, RefineNet 的 loss 通常都要比 SSD 的 loss 更大, 因为它比 SSD 的 loss 多了一个二分类和边框回归的 loss 计算. 具体的损失函数如下:

$$L(\{p_i\}, \{x_i\}, \{c_i\}, \{t_i\}) = \frac{1}{N_{ram}} (\sum_i L_b(p_i, [l_i^* \geq 1]) + \sum_i[l_i^* \geq 1] L_r(x_i, g_i^* )) $$
$$+ \frac{1}{N_{odm}}(\sum_i L_m (c_i, l_i^* ) + \sum_i[l_i^* \geq 1] L_r (t_t, g_i^* ))$$

上式中, $i$ 代表 mini-batch 中 anchor 的 index. $l_i^\*$ 是第 $i$ 个 anchor 所对应的真实类别的标签, $g_i^\*$ 是第 $i$ 个 anchor 所对应的真实边框的 location 和 size. $p_i$ 和 $x_i$ 是预测出来的第 $i$ 个 anchor 的二分类置信度和 refined anchor 的坐标. $c_i$ 和 $t_i$ 是 ODM 预测出的 object class 以及最终的 bbox 的坐标. $N_{arm}$ 和 $N_{odm}$ 分别是 ARM 和 ODM 中的 positive anchors 的数量. 二分类损失函数 $L_b$ 是二分类交叉熵损失, 多分类损失函数 $L_m$ 是 Softmax 多分类损失. $L_r$ 是 smooth L1 损失. $[]$ 为示性函数, 当框内表达式为真时, 输出 1, 否则输出 0. 当 $N_{arm}$ 或者 $N_{odm}$ 为 0 时, 则将对应损失置为0 .

**Optimization:** "xvavier"初始化用于额外增加的层(两层, conv6_1, conv6_2). batch size 设为 32, fine-tuned 使用 SGD + 0.9 momentum + 0.0005 weight decay + 0.001 initial learning rate.

**Inference:** 在测试阶段, ARM 首先过滤掉负样本置信度大于 $\theta$ (0.99) 的 anchors, 然后对于剩下的 anchors 进行 refine 操作. 之后, ODM 将这些 refined anchors 作为输入, 最终每张图片输出置信度 top 400 的边框, 然后使用 阈值为 0.45 的 NMS 算法去重, 最终将置信度 top 200 的边框作为最终的输出.


# RefineNet 使用了 two-stage 的边框回归过程, 为什么还说它是 one-stage 模型?

其实我个人觉得现在目标检测的很多模型中, one-stage 和 two-stage 的界限在慢慢变得模糊, 这也是一个正常的趋势, 因为我们希望得到的模型是不仅精度高, 速度也要快. 在 RefineDet, 这种界限就更模糊了, 我个人觉得 RefienDet 本质上还是属于 one-stage 模型, 因为它在 forward 计算的时候, 整体的流程还是和 SSD 很类似的, 是一步到底的走下来的, 只不过多走了一部分 anchor refine 的步骤. 而不像 Faster R-CNN 和 FPN 那样, proposals 的生成和最终 bbox 的预测有很明显的分隔. 因此, 我们还是倾向于认为它是 one-stage 模型.


# Reference

RefineDet(5)源码(1)CVPR2018: https://zhuanlan.zhihu.com/p/50917804
RefineDet 论文解析: https://zhuanlan.zhihu.com/p/39184173
尝试自己做一个refinedet的网络来训练数据: https://github.com/sfzhang15/RefineDet/issues/144
CVPR2018目标检测（objectdetection）算法总览: http://bbs.cvmart.net/articles/139/cvpr2018-mu-biao-jian-ce-object-detection-suan-fa-zong-lan
