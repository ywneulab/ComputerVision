---
title: Focal Loss (ICCV, 2017)
sitemap: true
categories: 计算机视觉
date: 2018-09-14 18:51:36
tags:
- 计算机视觉
- 目标检测
- 论文解读
---

**文章:** Focal Loss for Dense Object Detection
**作者:** Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár
**机构:** FAIR

# 核心亮点

**(1) 分析并指出了One Stage方法精度不高的原因:**
- **极度不平衡的正负样本比例:** anchor是一种类似sliding windows的选框方式, 这会使得正负样本的比例接近1000:1, 而且绝大部分负样本都是easy example.
- **梯度优化过程被easy example过度影响:** 这些easy example的loss虽然不高, 但由于数量众多, 最终合起来会对loss有很大的贡献, 从而导致优化的时候过度关注这些easy example, 这样会收敛到一个不够好的结果.

**(2) 提出了解决正负样本比例和easy example 问题的Focal loss:**

$$FL(p_t) = -(1-p_t)^{\gamma} log(p_t)$$

核心思想很简单, 就是在优化过程中逐渐减低那些easy example的权重, 这样会使得训练优化过程对更有意义的样本有更高的偏置.

**PS:**
注一: 为什么Focal Loss没有用在Two Stage方法上面? 这是因为以RCNN为代表的一系列Two Stage会在区域候选推荐阶段采用两个问题来降低正负样本比例和easy example问题带来的影响:
- 采用NMS算法将物体位置候选框降低到一到两千个，更重要的是，这一到两千个可能位置并不是随机选取的，它们移除了大量的负样本（背景框）
- 采用了biased-minibatch的采样策略, 比如，OHEM 或者保证正样本和负样本的比例为1：3进行训练（这其实相当于起到了 $\alpha$ 因子的作用

**Focal Loss 的两个重要性质:**
1. 当一个样本被分错的时候, $p_t$ 是很小的, 比如当 $y=1$ 时, $p_t$ 要小于0.5才算是错分类, 此时 $p_t$ 就比较小, 反之亦然, 因此调制系数就会趋近于1, 也就是说相比原来的loss没有太大的改变, 而当 $pt$ 趋近于1的时候, 说明此时分类正确而且是易分类样本, 调制系数就会趋近于0, 也就是该样本对总的loss的贡献度很小.
2. 当 $gamma=0$ 的时候, focal loss就是传统的交叉熵损失, 随着 $gamma$ 的增加, 调制系数的影响力也会增加

# 摘要

迄今为止, 精度最高的目标检测器是基于 R-CNN 推广的两阶段方法, 其中分类器应用于一组稀疏的候选框(这里的稀疏是指经过启发式规则采样后, 入选的anchor只有一小部分). 相比之下, one-stage 模型是对可能的目标位置进行常规且密集的采样, 他拥有更快的速度, 但是精度却不如 two-stage 模型. 在本文中, 我们将探讨为什么会出现这种情况. **我们发现, 在 one-stage 检测器训练过程中, 前后背景的样本数量会出现极度的不均衡, 这是造成精度低的主要原因. 因此, 我们提出通过对标准交叉熵损失函数重新构造来解决类别不平衡问题, 构造的原则是让它降低那些易区分样本(well-classified examples)对于 loss 的贡献度).** 我们新提出的 Focal Loss 会重点训练一组稀疏的 **难样例(hard examples)**, 并且会防止检测器在训练过程中被大量的 **简单负样例(easy negatives)** 所淹没. 为了评价 focal loss 的有效性, 我们设计并训练了一个简单的 one-stage 检测器, 称之为 RetinaNet. 我们的结果显示, 当使用 Focal Loss 进行训练时, RetinaNet 有能力同时保持精度和速度上的领先.

# 介绍

目前的 sota 目标检测模型都是基于 two-stage, proposal-driven 的. 类似于 R-CNN, 第一阶段会 **产生一组稀疏的候选位置**, 第二阶段 **会对每个候选位置进行分类**. 目前, two-stage 模型在精确度上遥遥领先.

暂且不考虑 two-stage 模型的成功性, 一个很自然的问题就是: **简单的 one-stage 模型能否达到类似的精度?** One-stage 模型通常会对物体的位置, 尺寸和宽高比进行密集的采样. 最近的 one-stage 模型, 如 YOLO 和 SSD, 显示出了很有希望的结果, 与 sota 的 two-stage 模型相比, 能产生精度较高且更快的模型(不如 two-stage 高).

本文更进一步的推进了这个概念: 我们首次提出了可以在精度上超过 two-stage 模型的 one-stage 模型. 为了实现这一目标, 我们认为训练过程中的类别不平衡问题是阻碍 one-stage 模型达到更高精度的主要障碍, 同时我们提出了一种新的损失函数来消除这一障碍.

**在 R-CNN 系列模型中, 类别不均衡问题是通过 two-stage 结构和启发式的采样算法来解决的.** 在第一阶段, 候选框生成算法(SS, EB, RPN)会大量的降低候选框的数量, 将大量的背景样本剔除. 在第二阶段, 会使用启发式的算法, 例如固定正负样本比例为 1:3, 或者使用难样例挖掘(OHEM), 以此来保证正负样本的平衡性.

相比之下, one-stage 检测器必须处理一组更大的候选框位置, 这些候选框是在图片上均匀采样生成的. 在实际中, 通常会枚举超过 100k 的位置, 这些位置密集的覆盖了物体可能的位置, 尺寸和宽高比. **虽然可以采用类似的启发式的抽样方法, 但是效率很低, 因为训练过程中的易分类负样本仍然占据主导地位. 这种低效率是目标检测中的一个经典问题, 通常利用 bootstrapping 或者 hard example mining 来解决.**

在本文中, 我们提出一种新的损失函数来作为一个更有效的方法解决样本不均衡问题. 如图1所示, 该损失函数一个可以动态缩放的交叉熵损失, 当对正确类别的置信度增加时($p^t$ 趋近于1), 比例因子($1-p^t$)将衰减为零. 从直觉上来说, **该比例因子能够自动降低训练过程中易分类样本的对损失函数的贡献度, 并能够快速的将模型集中在难样例上面.** 实验显示我们提出的 Focal Loss 可以训练出一个具有更高精确度的 one-stage 模型. 最后, 我们还注意到 Focal Loss 的确切形式并不是最重要的, 我们还展示了其他实例也可以获得类似的结果.

<div style="width: 550px; margin: auto">![图1](https://wx4.sinaimg.cn/large/d7b90c85ly1fw1pzf56s3j20jf0i177e.jpg)

为了证明 focal loss 的有效性, 我们设计并提出了一个 one-stage 的检测器称为 RetinaNet, 以其对输入图像中的目标位置的密集采样而命名. **该网络使用了高效的内部特征金字塔结构和 anchor boxes,** 借鉴了各种最新的目标检测观点. RetinaNet 不仅高效, 而且准确. 我们的最优模型使用了 ResNet-101-FPN 作为 backbone, 达到了 COCO 39.1 mAP, 5fps, 对比结果如图2所示

<div style="width: 550px; margin: auto">![图2](https://wx3.sinaimg.cn/large/d7b90c85ly1g1czavdenmj20nu0lugr5.jpg)


# Related Work

**Classic Object Detectors**
Sliding-window, HOG, DPMs,
主要是基于sliding-window paradigm的一类方法：HOG， DPMs等等。虽然滑动窗口类的方法在目标检测领域处于一线地位，但是随着deep learning的出现和研究，滑动窗口方法渐渐失去光芒。

**Two-stage Detectors**
two-stage方法的先驱是Selective Search work，它会首先提取出一个稀疏的候选框集合（稀疏是指只有很少一部分包含物体），然后对这些候选框进行分类，看是否包含物体，或包含哪种物体。
之后，RCNN的诞生标志着深度学习技术成功引入目标检测领域，利用cnn网络对特征的高度抽象和提取，rcnn在物体检测的准确率上大幅度提高，后期的RCNN系列又不断的提出新的方法来提升准确率和速度，到Faster RCNN时，提出了RPN网络，将候选框选取阶段和分类阶段都放在了统一个网络，使之可以进行端到端训练。后续还有更多的关于这一系列的工作继续被人们研究着。

**One-stage Detectors**
OverFeat算是首个现代的基于深度学习的one-stage检测方法，而最近的SSD和YOLO更是激起了人名对one-stage方法的研究热情，但是one-stage方法最令人诟病的地方就在于它们较低的准确率。
为此，本文的工作就是想要知道是否one-stage检测算法可以在精确度上匹敌two-stage检测算法，同时还要保持一定的检测速度。
于是，作者提出了Focal Loss，一种新的损失函数，利用这个损失函数，可以在保持现在模型大框架不变的基础上，达到最好的检测水平！

**Class Imbalance**
不管是传统的one-stage检测方法如boosted detectors， DMPs，还是最近的方法SSD，都会在训练阶段面临 $10^4\sim 10^5$ 个候选区域，这其中会包含大量的背景区域，也就是负样本，这种不平衡会造成两个问题：
- 在训练时，在大多数位置都是容易分类的负样本，这样只会贡献更多无用的信号
- 大量的易分类负样本会导致模型在一定程度上的退化
对于此问题，常用的解决方案是在训练阶段设计更复杂的样本抽取策略，但是这样速度就会受影响。而本文提出的Focal Loss，不仅解决了样本不均的问题，而且不需要增加额外的抽取策略，避免了易分类负样本淹没损失梯度.

**Robust Estimation**

有很多工作乐于设计健壮的损失函数. 相对于有的工作关注那些离异值的贡献. Focal Loss 实际上关注的是降低 inliers(easy examples) 的权重系数.

# Focal Loss

我们从交叉熵损失(CE)出发, 引入了 Focal Loss 进行二元分类(将 Focal Loss 扩展到多类情况是简单易行的, 我们在本文中主要关注二分类损失)

$$CE(p,y) = \begin{cases} -log(p) & \text {if y=1} \\ -log(1-p) & \text{otherwise}\end{cases} \tag 1$$

上式中, $y\in \{\pm 1\}$ 代表真实物体的类别, $p \in [0, 1]$ 是模型预测物体属于类别 $y=1$ 的概率. 为了便于表示，我们定义 $p_t$ 为

$$p_t = \begin{cases} p & \text{if y = 1} \\ 1-p & \text{otherwise} \end{cases} \tag 2$$

于是对上面的公式进行改写: $CE(p,y) = CE(p_t) = -log(p_t)$.

CE 损失可以看做是图1中的蓝色线条. 从图中可以看出 CE 损失的一个重要性质, 那就是即使是易分类样本($p_t \gg 0.5$), 也会产生较大的损失. 当对大量的易分类样本的损失求和时, 这些损失值就可能会淹没其他一些样本较少的类别.
当二分类问题中的样本分布不均时，数量多的负样本的损失值对最终函数的影响会淹没数量少的样本产生的影响。多分类问题也是如此。

<div style="width: 550px; margin: auto">![图1](https://wx4.sinaimg.cn/large/d7b90c85ly1fw1pzf56s3j20jf0i177e.jpg)


## Balanced Cross Entropy

一个常用的解决办法就是引入一个权重因子 $\alpha \in [0,1]$，然后分别令 $\alpha$ 和 $1 - \alpha$作为两个类别的权重，$\alpha$ 的取值可以是根据类别出现的频率决定，也可以作为超参数，利用交叉验证(cross validation)来选取较好的值。为了符号定义方便, 我们用类似于 $p_t$ 的方法来定义 $\alpha_t$, 我们给出 $\alpha$ CE 损失如下所示:

$$CE(p_t) = -\alpha log(p_t) \tag 3$$

这种损失是 CE 的一个简单的扩展, 我们将其作为 Focal Loss 的 baseline.

## Focal Loss Definition

本文的实验结果表明，类别分布不均衡会对交叉熵损失函数带来很大的影响。那些很容易被分类的负样本（背景等）贡献了大部分损失, 并且主导了 BP 中的梯度。**尽管 baseline 方法的 $\alpha$ 因子可以平衡正负样本之间的比例，但它仍然不能把握好简单样本和困难样本的比例（应该困难样本多一些，简单样本少一些，这样有利于模型的健壮性）**。于是，作者就提出了 Focal Loss 来降低易分类负样本的权重, 从而更多的关注难负样例.
具体来说, 我们向交叉熵损失中引入了一个 "调制因子(modulating factor)" $(1-p_t)^\gamma$ ，其中 $\gamma \geq 0$ ，我们定义 focal loss 损失函数形式如下:

$$FL(p_t) = -(1-p_t)^{\gamma}log(p_t) \tag 4$$

对于不同的聚焦参数 $\gamma \in [0, 5]$ 值下的 Focal Loss 的作用如图1所示. 我们注意到, focal loss 具有两个性质:
1. 当一个样本的 **$p_t$ 很小** 时(说明 **此样本被错分类, 并且错的很离谱, 如 $y=1$ 而 $p=0$, 或者 $y=0$ 而 $p=1$**), 调制因子的值趋近于1, 此时损失函数不会受到影响; 当一个样本的 **$p_t$ 很大** 时(说明 **此样本被正确分类, 并且置信度很高, 如 $y=1$ 且 $p=1$, 或者 $y=0$ 且 $p=0$**), 调制因子的值趋近于0, 此时损失函数的权重会被降低.
2. 聚焦参数 $\gamma$ 的值会平滑的调整易分类样本的权重降低比例. 当 $\gamma = 0$ 的时候, Focal Loss 就和普通的交叉熵相同, 当 $\gamma$ 升高的时候, 调制因子影响力就会增强. (在实验中, 我们发现 $\gamma=2$ 是一个不错的选择).

直观上来讲，**这个 "调制因子" 降低了易分类样本对于损失的贡献度, 并且扩展了样本接受较低损失的范围**。例如, 当 $\gamma = 2$ 时, 一个 $p_t = 0.9$ 的样本具有的损失比 CE 损失低 100 倍, 当 $p_t = 0.968$ 左右时, 比 CE 损失低 1000 倍. 这反过来又增加了纠正错误分类示例的重要性.
同时，还应注意到，Focal Loss的形式不是唯一固定的，在实际使用中, 我们使用了具有 $\alpha$ 因子的 Focal Loss 变体:

$$FL(p_t) = -\alpha_t(1-p_t)^\gamma log(p_t) \tag 5$$

后文的大部分实验都使用的是上面这个形式的Focal Loss, 因为它比不使用 $\alpha$ 因子的损失表现效果更好一点. 最后, 我们注意到损失层的实现将计算 $p$ 的 sigmoid 操作与损失计算相结合, 使得数值更加稳定.

## Class Imbalance and Model Initialization

二值分类模型在初始的时候，对两个类别的预测概率是均等的，在这种初始化条件下，如果某一个类别出现的次数过多，就会对损失函数产生较大的影响。为了解决这个问题，作者特意提出了“先入为主”的概念，也就是使得模型在开始的时候，对稀有类别（如前景类别）的预测概率的初始值设置的低一些，如0.01 。 经过实验表明，这样的方法可以提升模型训练的稳定性。

## Class Imbalance and Two-stage Detectors

Two-stage Detector 并没有使用类似 $\alpha$ 因此的方法来解决样本不均的问题。相反的，它们通过两个机制来降低这个问题带来的影响：（1）two-stage模式和（2）biased minibatch取样。首先，two-stage模式会在第一阶段就将近乎无限物体位置可能性降低到一到两千个，更重要的是，这一到两千个可能位置并不是随机选取的，它们移除了大量的易分类负样本（背景框）。第二，这些方法还设计了biased minibatch的取样策略，比如，保证正样本和负样本的比例为1：3进行训练（这其实相当于起到了 $\alpha$ 因子的作用。


## RetinaNet Detector

RetinaNet是一个单一的、统一的网络，它由一个backbone网络和两个task-specific子网络组成。backbone网络是现成的，主要负责计算卷积特征图谱。第一个子网络负责物体分类任务，第二个子网络负责bounding box回归任务，它们都是在backbone网络输出的卷积图谱上进行计算的。


**Feature Pyramid Network Backbone：**

采用了FPN作为backbone网络。

**Anchors:**

和FPN一样，对P3到P7使用了不同大小的anchors

**Classification Subnet：**

该子网络是一个较小的FCN，连接在FPN的每一层。

值得注意的一点是，该子网络并不与Box Regression Subnet共享参数，二者是互相独立的。

**Box Regresion Subnet：**

与分类子网络并行，该子网络也是一个FCN网络，连接在FPN的每一层上。目标是让anchor通过位移回归到gt box附近。

## Inference and Training

**Inference：** RetinaNet是有基于FPN和backbone和两个基于FCN的子网络组成的一个统一的单一网络，因此，在inference阶段，只需要简单的通过前向传播经过整个网络即可。为了提高速度，本文在每个FPN层级上，只会处理最多1000个box prediction。

**Focal Loss：** 使用了上文提到的Focal Loss。取 $\gamma=2$ 。在训练阶段，本文强调将focal loss应用到所有100k个anchors上，主要目的是为了与RPN和SSD等模型作对比。

从实验结果上看，当 $\gamma$ 的值取得较大时，$\alpha$ 的值就应该取消一些（for$\gamma=2$ , $\alpha = 0.25$ works best)。

**Initialization：** 本文分别实现了ResNet-50-FPN和ResNet-101-FPN。 对其中初始值可参见原文。

**Optimization：**

使用了SGD优化方法，在8个GPU上训练，每个minibatch有16张图片（一个GPU包含2张图片）。

损失函数为focal loss和标准smooth L1损失函数之和。

<div style="width: 550px; margin: auto">![图3](https://wx4.sinaimg.cn/large/d7b90c85ly1g1czbpsai0j218z0gw101.jpg)
<div style="width: 550px; margin: auto">![图4](https://wx1.sinaimg.cn/large/d7b90c85ly1fw1pzf7w44j213m0cr0wb.jpg)
<div style="width: 550px; margin: auto">![图5](https://wx1.sinaimg.cn/large/d7b90c85ly1g1czcwsuysj20nh0fxwg9.jpg)
<div style="width: 550px; margin: auto">![图6](https://wx1.sinaimg.cn/large/d7b90c85ly1g1czd58x5kj20nd0ewdh2.jpg)
<div style="width: 550px; margin: auto">![图7](https://wx1.sinaimg.cn/large/d7b90c85ly1g1czdghyejj20nz0fjmzb.jpg)
<div style="width: 550px; margin: auto">![表1](https://wx1.sinaimg.cn/large/d7b90c85ly1g1czc370pwj219c0rv7f1.jpg)
<div style="width: 550px; margin: auto">![表2](https://wx2.sinaimg.cn/large/d7b90c85ly1g1czco8067j21810ed0yd.jpg)
<div style="width: 550px; margin: auto">![表3](https://wx1.sinaimg.cn/large/d7b90c85ly1g1czdw6isoj20t209kdgu.jpg)
