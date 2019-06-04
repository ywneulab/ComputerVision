---
title: OverFeat (ICLR, 2014)
sitemap: true
categories: 计算机视觉
date: 2018-10-12 12:42:15
tags:
- 计算机视觉
- 目标检测
- 论文解读
---

**文章:** OverFeat: Integrated Recognition, Localization and Detectin using Convolutional Networks

**作者:**



# 核心亮点

**Multi-Scale Classification:**
在分类任务上, 虽然训练时采用和AlexNet相同的multi crop方法, 但是在预测阶段没有使用AlexNet的crop投票策略, 而是提出了Multi-Scale Classification方法, 一句话概括就是 **对整个图片以不同的尺寸, 并且对每一个location进行模型预测**

**利用了全卷积的思想代替全连接, 降低了滑动窗口的计算代价**

**可以用同一个模型完成分类, 定位, 检测任务:**
同一个模型, 只需要用回归层替换分类层, 即可完成目标定位任务, 同时利用了贪心策略来融合最终的定位结果

# 论文细节

## 背景介绍
提出了一种新的深度学习方法, 通过预测物体的边界来对目标物体进行定位.

本文的模型是ILSVRC2013的冠军

从表现最好的模型中开源了特征提取器, 并取名为OverFeat


作者为了避免大量的时间消耗, 没有在背景样例中训练, 但同时声称效果也不错?(持怀疑态度)

物体画框的时候, 因为画到了物体的一部分, 而没有画到整个物体, 更别说将框画到物体的正中心了. 因此, 希望可以对框进行预测回归, 使之画出来的框更加准确

与Krizhevsky等人之前的图像分类的paper做了个简单的比较

## 2. 视觉任务

本文研究了三种计算机视觉任务, 分别为: 分类, 定位和检测

## 3. 分类

本文的分类体系架构与Krizhevsky在12年使用的体系结构类似, 但是对模型中的一些细节进行了更多的探索

### 3.1 模型设计和训练

训练数据集: ImageNet 2012 (1.2 million 张图片, 共1000类)

输入尺寸: 训练阶段先将所有的图片downsample, 使其最小尺寸为256, 然后从每张图片(包括反转图片)中裁剪出5张子图(加上反转共10张), 尺寸为221*221, 这与AlexNet的策略相同, 但是在预测阶段, 使用了不同Multi-Scale Classificatioin方法, 具体见下文.

mini-batch: 128

模型权重初始化: 随机初始化 $(\mu, \sigma) = (0, 1\times 10^{-2})$

momentmu项为: 0.6

L2 权重衰减系数: $1\times 10^{-5}$

学习率: $5\times 10^{-2}$, 每经过(30,50,60,70,80)epochs时, 衰减1/2.

Dropout: 0.5(用于6,7层的全连接)

模型结果细节图:

<div style="width: 600px; margin: auto">![](https://wx4.sinaimg.cn/mw690/d7b90c85ly1fw5dt28gbnj20wi0e3aet.jpg)

<div style="width: 600px; margin: auto">![](https://wx2.sinaimg.cn/mw690/d7b90c85ly1fw5dx6jm9ej20vr0an408.jpg)

<div style="width: 600px; margin: auto">![](https://wx1.sinaimg.cn/mw690/d7b90c85ly1fw5e0t8ilxj20r006w3z5.jpg)

### 3.2 Feature Extractor

根据本文的方法, 开源了一个名为"OverFeat"的Feature Extractor, 有两个版本, 分别是精度优先和速度优先. 模型结构以及他们之间的比较都在上面两张图中显示.

### 3.3 Multi-Scale Classificaion

**在测试阶段**, 之前的Alex-Net中, 使用了multi-view投票来提升性能(一张图片crop出10个子图, 分别为四角和中心,以及他们的反转图片). 但是这个方法有一些问题: 忽视了图片中的很多区域, 并且计算存在冗余重复, 同时, 生成子图的操作都是在同一尺度上进行的, 这有时候不是最佳的尺度选择策略.

与此相反, **本文提出对整个图片以不同的尺寸, 并且对每一个location进行模型预测** , 虽然这种滑动窗口的方法在计算上是不可取的, 但是, 结合下面的3.5节中的策略, 可以缓解计算复杂问题.

**上面方法的好处:** 可以对同一张图片生成更多不同角度下的预测结果, 并将结果用于投票, 使得最终结果更鲁棒.

**但是存在问题:** 上面提到的subsampling方法的比例为 2×3×2×3, 也就是36. 这样一来, 模型在只能依赖36个像素点来生成分类向量. 这样粗粒度的分布使得模型的性能表现低于10-view方法.  主要原因是因为网络窗口没有和图片中的物体边界对齐. 于是本文提出了一种方法来克服这个问题, 对最后一个subsampling在不同的offset上进行pooling, 这样做可以缓解这一层的loss of resolution问题, 生成的总的subsampling比例为12, 而不是36. //TODO 这一段啥意思?, 36怎么来的, 12怎么来

下来解释具体是如何进行resolution augmentation的, 本文对于同一张图片, 使用了6种不同的尺寸, 如下图所示(C为类别):

<div style="width: 600px; margin: auto">![](https://wx2.sinaimg.cn/mw690/d7b90c85ly1fw5f5me3b3j20t608o0ub.jpg)

具体过程如下:
1. 对于一个给定的scale的图片,  我们从第5层, 还未pooling的特征图谱开始
2. 对于每一个未经过pooling的特征图谱, 进行 3×3 的pooling操作(非重叠池化), 重复 3×3 次($\Delta x, \Delta y)$ 以不同的offset {0,1,2}).  
3. 这样可以得到 3×3 个pooling后的特征图谱.
4. 分类器(6,7,8层)具有一个 5×5 的固定的输入大小, 同时对于每一个location的土整图谱都会生成 C 维的输出向量
5. 根据不同的offset, 可以得到不同的结果

下面的图以一维向量为例讲解了offset pooling的原理, 可以类推到3维结构中

<div style="width: 600px; margin: auto">![](https://wx2.sinaimg.cn/mw690/d7b90c85ly1fw5g21ruxgj20vu0g0gox.jpg)

上面的操作可以看做是以一个像素为单位, 在特征图谱上进行了位移操作, 进而得到了不同角度下的图谱结果用于分类, 与之前的crop方法相比, 这个方法没有对图片进行裁剪, 但同时又达到了获取同一张图片不同视角结果的目的. 这样做最大的优点在于: **对于一张图片, 可以宏观的将网络分成两个部分, 分别是特征提取层和分类层, 在特征提取部分, 同一张图片只会进行一次前向计算, 在这计算角度来说, 大大提高了计算效率, 减少了荣冗余计算.**

这种极致的pooling策略(exhaustive pooling scheme) 确保了我们可以在分类器和特征图的对象之间获得准确的边界对齐.

### 3.4 结果

实验结果如下表所示:

<div style="width: 600px; margin: auto">![](https://wx1.sinaimg.cn/mw690/d7b90c85ly1fw5h2ottzhj20vz0d0whr.jpg)

分类错误率在18个队伍中的排名为第5 (主要本篇文章的两点也不在分类, 还是在目标检测上)

### 3.5 convNets和滑动窗口效率

 首先, 要知道全卷积与全连接层之间的关系, 利用全卷积层代替全连接层以后, 可以接受不同图片尺寸的输入, 同时, 在计算时, 由于卷积操作本身的计算共享特性, 可以使得计算过程更加高效, 以此来缓解滑动窗口方法带来的计算问题.

<div style="width: 600px; margin: auto">![](https://wx4.sinaimg.cn/mw690/d7b90c85ly1fw5hncxz5yj20vz0lmtip.jpg)

// TODO  这块还是不是很懂

## 4. 定位 Localization

用回归网络代替分类网络, 同时训练网络使其在每一个空间location和scale下预测目标物体的bounding boxes.  然后将回归预测结果与每个位置的分类结果结合在一起.

### 4.1 生成预测

由于可以共享特征提取层的计算结果, 因此只需要重新计算最终的回归层即可. 因为每个位置的分类结果为所有的类别都赋予了一个置信度, 因此, 我们可以将这个置信度作为bounding box的置信度使用.

### 4.2 回归训练 Regressor Training

回归网络将第5层的池化后的特征图谱作为输入. 后面接入两个全连接层, 隐藏神经元个数分别为4096和1024. 最终的输出层具有4个神经元, 用于指定bounding box的坐标. 和分类网络一样, 也是用了基于offset的pooling策略.  该网络的结构如下图所示

<div style="width: 600px; margin: auto">![](https://wx2.sinaimg.cn/mw690/d7b90c85ly1fw5i3k7yjzj20ws0rp43x.jpg)

训练时, 固定前5层的特征提取层, 然后使用每个样本的真实边界与预测边界之间的L2损失(平方损失)来训练回归网络. 最后的这个回归层根据与特定类相关的, 每一个类都会有一个回归层(也就是说有1000个不同的回归层版本).

当输入图片与目标物体的IOU小于50%时, 则不会进行训练

同时使用了多尺度的输入进行训练(与第三节相同), 以便更好的进行不同尺度下的预测.

### 4.3 联合预测

通过对回归得到的bounding boxes使用贪心融合策略, 来得到单个目标物的预测结果. 具体算法过程如下:

1.
<div style="width: 600px; margin: auto">![](https://wx3.sinaimg.cn/mw690/d7b90c85ly1fw5if40q6lj20vw1fu163.jpg)

<div style="width: 600px; margin: auto">![](https://wx2.sinaimg.cn/mw690/d7b90c85ly1fw5ifyu74ij20x00ql47d.jpg)
