---
title: DCN-ICCV 2017
sitemap: true
categories: 计算机视觉
date: 2018-11-05 17:04:56
tags:
- 计算机视觉
---

**文章:** Deformable Convolutional Networks
**作者:** ifeng Dai, Haozhi Qi, Yuwen Xiong, Yi Li, Guodong Zhang, Han Hu, Yichen Wei
**机构:** Microsoft Research Asia

# 核心亮点

**1) 引入了可以自调节感受野大小的deformable convolution和deformable RoI 模块**
该模块通过额外学习一组采样偏移量来决定卷积操作和RoI pooling操作的采样位置, 通过这种方式, 是的网络模型可以根据输入的图谱自动调节感受野的大小的分布.

**2) 上面的两种deformable模块均可以无痛的添加到现有模型中**
由于deformable convolution和deformable RoI 模块并不会改变原始的输入输出大小, 因此可以很轻易的替换到现有网络中, 并且可以有其他多种提升精度的trick想叠加, 在多个视觉任务上(检测, 分割)都表现出色.

<div style="width: 550px; margin: auto">![](https://wx4.sinaimg.cn/large/d7b90c85ly1fx0rot8bcsj20r60fdgus.jpg)

# 论文细节

## 摘要

卷积神经网络由于其模型内部固定的几何结构, 使得它固有的受限于模型的几何变换. 在这片文章中, 作者引入了两种新的模块来增强CNN模型的transformation modeling能力, 分别为 **可变形的卷积模块** 和 **可变形的RoI池化模块(deformable convolution and deformable RoI pooling)**. 这两个模块都是基于增强模型中的空间采样点的思想提出的, 并且是通过 **在没有额外监督标签下, 从目标任务学习偏移量来增加空间采样位置的**. 这些新的模块可以 **轻易的替换掉CNN网络中的原始部分, 并且可以轻易地进行端到端的训练.** 下面, 我们将会展示在CNN网络中 **学习** 密集的空间形变, 有助于提升传统的视觉任务, 如物体检测, 语义分割等等.

## 介绍
在视觉识别任务中, 有一个很关键的点在于怎样才能最大程度的适应物体的几何变换, 如物体尺寸, 姿态, 观察角度, 局部形变等等. 在通常情况下, 具有两种做法, **第一** 是建立足够的数据集来包含这些可能的形变, 以便让网络能够学习到足够的形变知识. 这一类方法的典型应用就是数据增广. **第二种** 方法就是使用支持形变不变性的特征表示或算法, 如SIFT(CNN具有平移不变性, 但不具有形变不变性).
**上面的两种方法有一些明显的缺点**. **首先**, 就是要求对形变类型已知, 只有在这种假设成立的前提下, 才能有选择的应用最合适的数据增广方法. **第二**, 即使在知道形变可能类型时, 利用人工设计特征算子依然是一个不容易的工作, 尤其适当形变类型较为复杂时. CNN虽然取得了很大成功, 但是CNN仍然面临这这两问题. 它们对物体形变的适应能力实际上大多来自于海量数据集, 大型模型, 以及一些简单的人工设计模块的支持(如 max-pooling for small translation-invariance).
**总而言之, CNN 天生的局限于对大型的, 位置的转换任务进行建模**, 这种限制来自于CNN模型本身的结构模块(**不论是卷积层还是fc层, 层的参数及输出向量都是固定的**). 因此, 在CNN内部, 缺少相应的内部机制来处理几何形变的问题. 举个例子来说, 在同一层卷积层中所有激活单元感受野大小都是一样的, 而这并不是high level的卷积层所期望看到的, 因为正常来说, 不同位置与物体之间的联系紧密程度是不同的. 另外, 目前大多数方法都是基于主边框进行回归预测的, 这实际上是一种次优化的方法, 尤其是对于非规则物体来说.
在这篇文章中, 我们提出了两个新的模块可以增强CNN对物体集合形变的适应能力. 首先是 **可形变卷积模块**, 它将 **2D的偏移量** 添加到规则的网格采样位置中, 它可以使得采样网络能够自由变形, 如图1所示. 图中的不同的偏移量都是通过额外的卷积层从之前的特征图谱中学习到的. 因此, 可形变是以一种局部的, 密集的, 自适应的方法建立在输入特征之上的.

<div style="width: 550px; margin: auto">![](https://wx3.sinaimg.cn/large/d7b90c85ly1fwye3au996j20sb0kfdl0.jpg)

第二部分是 **可形变RoI pooling.** 它为前一个 RoI pooling 的常规 bin 划分中的每个 bin 位置都添加天一亮. 同样, RoI pooling的offset也是从前面的特征图谱和 RoIs 中学习得到的, 从而支持对具有不同形状的对象进行自适应的局部定位.
以上两种可形变模块都是轻量级的, 并且只引入了很少的参数和计算量, 可以很容易的替换掉标准的CNN网络中去. 产生的网络我们称之为可形变卷积网络(Deformable ConvNets).

## Deformable Convolutional Networks

特征图谱和卷积操作都是 3D 的. 可形变操作模块都是在 2D 空间领域进行操作的. 不同通道上的可形变操作是一样的, 因此, 为了简单起见, 这些模块都用 2D 进行描述, 并且可以轻易扩展到 3D.

### Deformable Convolution

一个2D的卷积操作由可以分为两步:
- 使用某个规则网格 $R$ 在输入的特征图谱上进行采样
- 将采样点的值与卷积核对应位置的权重相乘再求和, 既得输出图谱某点的值
网格 $R$ 定义了感受野的大小和 dilation 的选择, 例如下面的公式定义了一个 dilation 为 1 的 3× 3 大小的卷积核.
$$R=\{ (-1,-1), (-1,0), ..., (0,1), (1,1) \}$$

对应输出的 feature map 上的每个点的值, 可通过下式计算:

$$y(p_0) = \sum_{p_n\in R} w(p_n)\cdot x(p_0 + p_n) \tag 1$$

上式中, $p_0$ 对应的是输出特征图谱上的点坐标, $y(p_0)$ 对应的是该点的值, $p_n$ 对应的是常规偏移量集合 $R$ 中的坐标偏移量, $w(p_n)$ 代表该偏移量对应的权重值, $x(p_0 + p_n)$ 代表输入特征图谱在坐标 $p_0+p_n$ 上的值. 可以看出, 这里的 $R$ 实际上也代表了输出特征图谱 $p_0$ 点的感受野范围.

**在deformable 卷积中**, 常规坐标偏移量集合 $R$ 会与另一个额外的偏移量集合 $\{\Delta p_n | n =1,...N\}, 其中, N=|R|$ 共同决定卷积核感受野的采样坐标, 于是, 输出特征图谱上面的点的计算公式变成:

$$y(p_0) = \sum_{p_n\in R} w(p_n) \cdot x(p_0 + p_n + \Delta p_n) \tag 2$$

可以看到, 现在的采样点不再是一个规则的矩形了. 在实现中, 偏移量 $\Delta p_n$ 是浮点数类型, 我们将通过双线性插值来实现(将卷积的输出通过插值的方法计算):

$$x(p) = \sum_q G(q,p)\cdot x(q)$$

上式中, $p$ 代表任意一个浮点坐标值( $p=p_0+p_n+\Delta p_n$ ). $q$ 代表输入特征图谱 $x$ 中的所有整数点坐标(实际计算时, 只有 $p$ 周围的四个整数坐标点有用). $G(\cdot, \cdot)$ 代表双线性插值函数. 注意 $G$ 具有两个维度, 它被分成两个一维的核, 如下所示, 式子中 $g(a, b) = max(0, 1 - |a - b|)$

$$G(q, p) = g(q_x, p_x) \cdot g(q_y, p_y) \tag 4$$

如图2所示可形变卷积的过程, 我们通过在相同的输入特征图谱上使用卷积层来获得偏移量. 形变卷积核具有与当前卷积层相同的空间分辨率和 dilation. (如图2中的 $3\times 3$ with dilation 1). **最终输出的偏移量具有和输入特征图谱相同的空间分辨率**
在同一个卷积层当中, 卷积核的空间大小和dilation都是相同的. 图中的通道维数 $2N$ 对应着 $N$ 个二维偏移量. 在训练的时候, 会同时学习生成输出特征图谱和偏移量的卷积核. 在学习偏移量时, 通过公式(3)(4)用BP算法更新偏移量参数.

**注意, 同一层的多个卷积核各自持有一个offset**

<div style="width: 550px; margin: auto">![图2](https://wx1.sinaimg.cn/large/d7b90c85ly1fx0mt9twbyj20q60gctag.jpg)

### Deformable RoI Pooling

RoI pooling 目前被广泛的运用于各种目标检测模型当中, 它可以将不同尺度的矩形区域转换成固定尺寸的图谱.

**RoI Pooling:** 给定一个特征图谱 $x$ 和一个大小为 $w\times h$ , 左上角坐标为 $p_0$ 的RoI, 对其使用RoI pooling, 将其划分成 $k\times k$ 大小的网格, 并且输出一个 $k\times k$ 大小的特征图谱 $y$. 那么, 对与 $y$ 中坐标为 $(i,j), 0\leq i,j \leq k$ 的网格bins来说, 其输出的值为

$$y(i,j) = \sum_{p\in bin(i,j)} x(p_0 + p) / n_ij \tag 5$$

上式中, $n_ij$ 是bin中含有的像素点的个数. 同理, 我们可以将上面的标准RoI格式写成deformable RoI公式, 如下所示:

$$y(i,j) = \sum_{p\in bin(i,j)} x(p_0 + p + \Delta p_{ij}) / n_{ij} \tag 6$$.

上式同样包含浮点型坐标 $\Delta p_{ij}$, 因此也用双线性插值实现.

图3说明了如何获得offsets. 首先, 利用标准的RoI pooling生成池化后的特征图谱. 在特征图谱上, 会用一个 $fc$ 层来生成归一化的offsets $\Delta \hat p_{ij}$, 然后通过对应元素相乘(element-wise product) $\Delta p_{ij} = \gamma \cdot \Delta \hat p_{ij} \odot (w,h)$ 将其转换成上式中的offsets $\Delta p_{ij}$. 式中, $\gamma$ 是一个预先定义的标量(默认为0.1), 来控制offset的大小. **注意, 为了使偏移量学习不受 RoI 大小的影响, 需要对偏移量进行归一化.**

<div style="width: 550px; margin: auto">![图3](https://wx3.sinaimg.cn/large/d7b90c85ly1fx0mtq0hbfj20px0f6q4m.jpg)

**Position-Sensitive(PS) RoI Pooling** PS RoI pooling是全卷积的. 通过卷积层, 所有的输入特征图谱首先会被转换成 $k^2$ 大小的score maps.(对于每一类都有这样的一个maps, 因此共有 $C+1$ 个score maps), 如图4的底部分支所示. //TODO

<div style="width: 550px; margin: auto">![](https://wx1.sinaimg.cn/large/d7b90c85ly1fx0mu45kgzj20rd0kr0um.jpg)

### Deformable ConvNets

可以看出, 不论是Deformable convolution还是 Deformable RoI pooling, 它们的输出都和常规版本的卷积核RoI的输出大小相同. 因此, 我们可以很自然的用Deformable模块替换现有目标检测模型中的常规模块. 在训练时, 为了学习offsets参数而新增的卷积层和fc层都被初始化为0, 它们的学习率被设置为现有目标检测模型学习率的 $\beta$ 倍(通常为1, 在FasterRCNN中, fc的 $\beta$ 设为0.01). 它们通过双线性插值运算的反向传播算法进行训练, 最终生成的模型称为 deformable ConvNets.

为了将deformable convnets整合到现有的先进检测模型中, 我们可以将 two-stage 目标检测模型看成以下 **两部分**:第一部分是一个深层的全卷积网络, 用于生成整个图片的特征图谱. 第二部分, 是根据特征任务设计的具体的浅层网络, 它从feature maps中获取最终计算结果. 下面来详细说明一下这两个部分.
**Deformable Convolution for Feature Extraction:** 本文使用了两个先进的特征提取模型: ResNet-101 和 Inception-ResNet. 两个模型均在ImageNet数据集上预训练.
这两个模型都包含多个卷积块, 一个平均池化层和一个1000路的fc层用于分类. 我们将平均池化层和fc层移除. 然后添加一个 1×1 的卷积层(随机初始化)将通道维数降为 1024 维. 根据前人工作(R-FCN), 我们将最后一个卷积块的stride从32降为16, 以此来提升特征图谱的resolution, 具体来说, 就是在最后一个卷积块的最开始, 将stride从2变成1, 为了弥补这种改变, 同时会将这一个卷积块里面的卷积层的 dilation 从1变成2.
我们会将 deformable convolution 应用于最后的几层卷积层(kernel size > 1). 通过实验我们发现, 对3层的卷积层应用 deformable convolution 可以很好的权衡不同的任务, 如表1所示.

<div style="width: 550px; margin: auto">![表1](https://wx4.sinaimg.cn/large/d7b90c85ly1fx0qcabkyxj21kw0f70yr.jpg)

**Segmentation and Detection Networks**
在上述特征提取网络的基础上, 我们可以将 deformable convolution 应用于特定的任务. 下面中的 $C$ 代表目标物体的类别数量

- DeepLab: 实例分割任务的 sota 方法. 它在特征图上添加了一个 $1\times 1$ 的卷积层, 然后生成 $(C+1)$ 个表示每个像素分类 scores 的图谱. 然后, 后面的 Softmax 层会输出每个像素的概率.
- Category-Aware RPN: 和 Faster R-CNN 中的 RPN 差不多, 只不过用 $(C+1)$ 类别的分类卷积层替换掉了原始 RPN 中的二分类卷积层.
- Faster R-CNN: sota 的 two-stage 目标检测方法
- R-FCN: 另一个 sota 目标检测方法.

## Understanding Deformable ConvNets
该工作的思想是用额外的偏移量增加卷积和RoI池中的空间采样位置，并从目标任务中学习偏移量.

当可变性卷积叠加时, 复合变形产生的影响是很大的, 如图5所示, 在标准卷积中的感受野和采样点在顶层(深层)特征图谱上都是固定的, 但是在可形变卷积中, 它们会根据对象的尺度和形状进行自适应的调整.

<div style="width: 550px; margin: auto">![图5](https://wx2.sinaimg.cn/large/d7b90c85ly1fx0qi3arkgj20ty0uen9y.jpg)

更多的例子如图6所示.

<div style="width: 550px; margin: auto">![图6](https://wx2.sinaimg.cn/large/d7b90c85ly1fx0qmn6fbfj21kw0hl7wh.jpg)

表2提供了一些数值证明了Deformable ConvNets的有效性. 从中可以看出:
- Deformable filter的感受野大小是和物体大小相关的, 这说明卷积核的形变已经从图片中学到了很多的有效信息
- 背景区域的卷积核大小介于中等尺寸物体和大型物体之间, 这说明在识别背景区域时, 一个相对较大的感受野是有必要的.

<div style="width: 550px; margin: auto">![](https://wx4.sinaimg.cn/large/d7b90c85ly1fx0qopdjawj20rv0hq776.jpg)

Deformable RoI pooling的有效性如图7所示, 标准的 RoI pooling 中 bins 的网格结构不再成立. 取而代之的是, 部分区域的 bins 会偏移原来的位置, 移动到附近的目标对象的前景区域, 这增强了网络的定位能力, 尤其是对非刚性物体来说.

<div style="width: 550px; margin: auto">![](https://wx3.sinaimg.cn/large/d7b90c85ly1fx0qo5mw1gj21kw0hc4qp.jpg)

### In Context of Related Works

Spatial Transform Networks(STN),
Active Convolution,
Effective Receptive Field,
Atrous convolution,
Deformable Parts Models(DPM),
DeepID-Net,
Spatial manipulation in RoI pooling.
Transformation invariant features and their learning

## 实验

**Semantic Segmentation:** 使用 PASCAL VOC 和 CityScapes 数据集.

**Object Detectin:** 使用 PASCAL VOC 和 COCO 数据集.

默认的ResNet-101使用了 dilation为2, size为3×3的 **atrous convolution**. 我们还尝试了更多其它的可能参数, 如下表3所示. 表中数据说明:
- 当使用较大的dilatioin时, 所有任务的准确度都有所提升, 说明默认网络的感受野太小了(较大了dilation可以提供较大的感受野)
- 不同的任务和模型其最优的dilatioin参数不同, 但是deformable convolution总是能取得最高的准确度
- 对于具有RoI结构的网络来说, deformable RoI同样有效

<div style="width: 550px; margin: auto">![](https://wx3.sinaimg.cn/large/d7b90c85ly1fx0re07xzmj21gh0gw78j.jpg)

如下表4所示, 可以看出, deformable结构对于目标检测任务同样有效:

<div style="width: 550px; margin: auto">![](https://wx1.sinaimg.cn/large/d7b90c85ly1fx0re6gz2nj21kx0inaf9.jpg)

表5贴出了deformable模型的复杂度和运行时间, 可以看到, 模型增加的参数量和运行时间都是在可接受范围内的

<div style="width: 550px; margin: auto">![](https://wx3.sinaimg.cn/large/d7b90c85ly1fx0red9jj4j20tz0lon1o.jpg)
