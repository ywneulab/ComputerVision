---
title: YOLO v1
sitemap: true
date: 2018-09-11 19:27:16
categories: 计算机视觉
tags:
- 计算机视觉
- 目标检测
- 论文解读
---

**文章:** You Only Look Once: Unified, Real-Time Object Detection

**作者:** oseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi

# 核心亮点

**(1) 将检测问题看做是回归问题**
对于给定的输入图像, YOLO会使用一个单一的网络 **同时** 给出bounding box的预测结果和对应的类别概率.

**(2) 没有Region Proposal的过程**
YOLO采用 $S\times S$ 的网格划分来确定候选框, 如果某个物体的中心落在了某个cell里, 那么这个cell就负责该物体的检测.

**PS:**
注一: YOLO中采用 $S\times S$ 的网格划分来确定候选框, 这实际上是一种很粗糙的选框方式, 同时也导致了YOLO在面对小目标物以及群落目标物时, 性能较差.(因为YOLOv1的同一个cell无法预测多个目标)


# 背景介绍

YOLO将目标检测问题看作是一个回归问题，进而从整张图像中直接得到bounding boxes和对应的class probabilities。

之前的工作都是将检测任务看成是一个分类问题，如RCNN，通过区域提取，分类，区域修正，去重等等一系列工作得到检测结果，这样的模型十分复杂而且很难优化，因为区域提取和分类任务必须单独训练，麻烦且难以调试。

本文将目标检测问题看成是一个回归问题，直接从图片像素中得到bounding box坐标和class probabilities。

YOLO具有三大优点：
1. Fast。 由于不用按照复杂的pipeline进行运作，YOLO只需要一个卷积网络就可以同时预测出多个物体，因此十分快
2. YOLO在进行推理时，可以看到整幅图片，因此，可以隐式地对物体的周围像素进行分析。这使得YOLO不容易在背景中错误识别。反观Fast RCNN，经常会将背景中的非物体检测出来。
3. YOLO的泛化性更好，可以学到更一般的特征。在自然图像上训练后，YOLO在艺术图像上可以取得相比于RCNN更好的检测效果。

# 关键技术

YOLO没有提取候选区域的过程, 与之相对的, YOLO采用网格划分的方式来确定物体的候选区域框, 具体来说, YOLO会将图像按照 $S\times S$ 的大小划分成多个cell, 之后, 如果哪个物体的中心落在了某个cell里面, 那么这个cell就负责检测这个物体, 如下图中, 狗的中心落在了红色cell内, 则这个cell负责预测狗.

<div style="width: 550px; margin: auto">![](https://wx2.sinaimg.cn/mw690/d7b90c85ly1fw93fvg9xdj20gp0etqe3.jpg)

"物体落在哪个cell, 哪个cell就负责预测这个物体" 分为训练和测试两个阶段:
- 训练阶段. 在训练阶段, 如果物体中心落在这个cell, 那么就给这个cell打上这个物体的label, 让这个cell和该物体关联起来
- 测试阶段. cell会根据已经训练好的参数来决定自己负责预测哪个物体.

网络的整体架构如下图所示:

<div style="width: 550px; margin: auto">![](https://wx2.sinaimg.cn/mw1024/d7b90c85ly1fw93mm33q1j21kw0o80x2.jpg)

从图中可以看出, YOLO网络的输出网格是 7×7 大小的, 另外, 输出的channel数目30, 在每一个cell内, 前20个元素是每个类别的概率值, 然后2个元素对应2个边界框的置信度, 最后8个元素时2个边界框的 $(x,y,w,h)$.(每个cell会预测两个框, 最后选择IOU较大的来复杂物体的预测)

<div style="width: 550px; margin: auto">![](https://wx1.sinaimg.cn/mw690/d7b90c85ly1fw93lr5viuj20se0hunar.jpg)

根据网络的输出, 我们可以知道, YOLO的预测目标主要有三个: 类别预测, Confidence预测, Bounding box预测. 在训练阶段，该模型要优化下面的联合目标损失函数(第一行是bounding box预测, 接下来是confidence预测, 最后是类别预测)

$$\lambda_{coord} \sum_{i=0}^{S^2} \sum_{j=0}^B I_{ij}^{obj}[(x_i-\hat x_x)^2 + (y_i - \hat y_i)^2] \
+ \lambda_{coord}\sum_{i=0}^{S^2} \sum_{j=0}^B I_{ij}^{obj} [(\sqrt w_i - \sqrt{\hat w_i})^2 +(\sqrt h_i - \sqrt{\hat h_i})^2] $$
$$+ \sum_{i=0}^{S^2}\sum_{j=0}^B I_{ij}^{obj} (C_i - \hat C_i)^2  + \lambda_{noobj}\sum_{i=0}^{S^2}\sum_{j=0}^{B} I_{ij}^{noobj}(C_i-\hat C_i)^2$$
$$+ \sum_{i=0}^{S^2} I_i^{obj} \sum_{c\in \text{classes}} (p_i(c) - \hat p_i(c))^2
$$

需要注意的是, 网络并不会总是计算所有的loss项, 具体地说:
1. 对于有物体中心落入的cell, 需要计算分类loss, 两个confidenceloss, 但只计算IOU较大的bounding box loss
2. 对于没有物体中心落入的cell, 只需要计算confidence loss.

另外, 我们发现每一项的计算(即使是分类)都是 L2 loss, 从另一角度体现出YOLO把分类问题转化为了回归问题.

# 一体化检测

YOLO使用整幅图像的特征图谱进行预测，同时预测所有物体的所有bounding box。这样的设计思想，可以使得YOLO进行端到端的训练，并且能够进行实时检测。

系统将整张图片划分成 $S \times S$ 大小的网格。 如果某个物体落入了网格中的某一格，那么这个格子就负责检测该物体。

每个格子会预测B个bounding boxes和B个confidence scores。这些confidence scores反映了模型对这个box里面是否有物体，并且有多大的把握确定。 将confidence定义为 $Pr(Object)\times IOU_{pred}^{truth}$ 。 $IOU_{pred}^{truth}$ 代表真实框和预测框之间的IOU值。

每一个bounding box包含5个预测值：x，y，w，h，和confidence。

每一个grid cell预测C个conditional class probabilities，记为 $Pr(Class_i|Object)$ 。 C与B的个数之间没有直接关系。

在测试阶段，我们将conditional class probabilities和individual box confidence predictions相乘：

$$ Pr(Class_i|Object)\times Pr(Object)\times IOU_{pred}^{truth} = Pr(Class_i)\times IOU_{pred}^{truth} $$

由此可以得到针对每个box的特定class的confidence scores。这些scores代表着特定calss出现在box里面的概率，以及预测出来的box在多大程度上适应这个object。

最终预测的tensor维度： $S\times S\times (B\times 5+ C)$ 。

# 网络设计

YOLO：收到GoogleNet的启发，公有24层卷积层和2层全连接层

但是没有使用Inception模块，而是使用了 $3\times 3$ 的卷积层和一个 $1 \times 1$ 的reduction layers（减少depth）

fast YOLO：9个卷积层和2个全连接层。

# 训练

首先在ImageNet上进行了预训练。 预训练时，使用前20个卷积层，加上一个平均池化层，和一个全连接层。 使用了Darknet framework。

 Ren et al证明在预训练的网络上添加卷积层和全连接层可以提升性能。因此，本文添加了4个卷积层和2个全连接层，都赋予随机初始值。 模型的输入图像像素为448 。

 最后一层同时预测class probabities和bounding box coordinates。 我们将box的宽和高都归一化到占图片宽高值的比例，因此coordinates的值在0到1之间。coordiantes的x和y归一化到对特定cell的相对位移，所以它们的值也在0到1之间。

 本文最后一层使用线性激活函数，其他层均使用leaky rectified linear 激活函数，如下所示：

 $$ \phi(x) = \begin{cases} x & \text{if } x>0 \\ 0.1x& \text{otherwise} \end{cases} $$

本文的优化函数为平方和误差。 由于它对localization error的权重和对classification的权重是一样的，因此该函数并不能够很好的匹配我们的目标。为了解决问题，提升了bounding box coordinate predictions的loss，同时降低了confidence predictions的loss。 作者使用了 $\lambda_{coord} = 5$和 $\lambda_{noobj} = 0.5$ 来实现这一目标。 同时为了更好的针对小目标，本文对bounding box的宽和高都使用了平方跟。

YOLO对每个grid cell都会预测出多个bounding boxes，而在训练阶段，我们只需要一个bouding box 来对每个物体负责。选取的原则是与GT有最高的IOU值。

在训练阶段，本文优化下面的联合目标损失函数：

$$\lambda_{coord} \sum_{i=0}^{S^2} \sum_{j=0}^B I_{ij}^{obj}[(x_i-\hat x_x)^2 + (y_i - \hat y_i)^2] \
+ \lambda_{coord}\sum_{i=0}^{S^2} \sum_{j=0}^B I_{ij}^{obj} [(\sqrt w_i - \sqrt{\hat w_i})^2 +(\sqrt h_i - \sqrt{\hat h_i})^2] $$
$$+ \sum_{i=0}^{S^2}\sum_{j=0}^B I_{ij}^{obj} (C_i - \hat C_i)^2  + \lambda_{noobj}\sum_{i=0}^{S^2}\sum_{j=0}^{B} I_{ij}^{noobj}(C_i-\hat C_i)^2$$
$$+ \sum_{i=0}^{S^2} I_i^{obj} \sum_{c\in \text{classes}} (p_i(c) - \hat p_i(c))^2
$$

batch size为64，a momentum of 0.9 and a decay of 0.0005。

# 推理阶段

平均每张图片会得到98个bounding boxes。

虽然采用了非极大值抑制，但是提升的效果并不高，不如RCNN和DPM那么明显。

# YOLO的局限性

难以检测小物体和堆积在一起的物体，比如鸟群。

另外，YOLO对于不同大小的物体，其采取的损失函数是一样的，因此，在面对大物体时，细微的差别可能不会引起IOU的大幅变化，但是在面对小物体时，就会产生较大波动。YOLO的错误来源主要是由于定位错误。

# 和其他检测系统的比较

**Deformable parts models**：

DPM使用了滑动窗口的方法来做目标检测。它的检测是由分离的好几段过程完成的。 相比来说，作者的模型统一了所有这些过程，并且取得了更快更好的效果（基本来说就是把DPM吊打了。。。，不过毕竟DPM是2010年的产品，不吊打说不过去了。。）

**RCNN**：

RCNN没有使用滑动窗口的方法来获取bounding box，而是使用了Selective Search（之后也不用SS方法了，提出了RPN，继承到模型内部了）。同理，RCNN也是一种多阶段的方法，先画框，再检测，分两步走。YOLO在一定程度了也借鉴了RCNN及其变体的思想，但是YOLO是基于grid cell进行proposes bounding box的，所以最后只生成了98个框，而RCNN的框多大2000个，所以YOLO在速度上肯定是远超RCNN了，另外精度上也比RCNN高（不过RCNN只是region based检测方法的雏形，所以并不说明YOLO比RCNN整个系列都好）。

**Other Fast Detectors**： RCNN其他系列来了，作为后出生的Fast RCNN和Faster RCNN，当然视为自家的兄弟出了口气，在精度上爆了YOLO，但是速度还是不及YOLO（YOLO是真的快，真正意义上的实时监测系统）

**Deep MultiBox**

14年出来的，SPPNet使用了它进行选框

**OverFeat**

13年的一篇文章

**MultiGrasp**

这是Joseph自己的工作，在YOLO之间发的，解决的任务是检测一张的图片中某个包含物体的区域，比YOLO要解决的任务简单的多，没什么好说的

# 实验 Experiments

首先是在VOC2007上做了实验，然后专门针对YOLO和Fast RCNN进行比较，虽然整体mAP没有Fast高，但是在背景上的假正例比Fast少。接着，还给出2012VOC的实验结果。最后，还做了一个从自然图像训练，然后检测艺术作品的实验，提出YOLO可以学到更一般化的特征。

# Comparison to Other Real-Time Systems
