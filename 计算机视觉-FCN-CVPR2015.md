---
title: FCN-CVPR2015
sitemap: true
date: 2018-08-21 21:17:00
categories: 计算机视觉
tags:
- 计算机视觉
- 论文解读
- 实例分割
---

**文章:** Fully Convolutional Networks for Semantic Segmentation
**作者:** Jonathan Long, Evan Shelhamer, Trevor Darrell



# 核心亮点

~~全卷积网络: 只有最后一层是全连接层, 并且在针对 object detection 任务进行 fine-tuning 时, 会将该全连接层移除. (但是分类任务仍然需要这一层来输出最后的分类结果)~~

**(1) 利用FCN网络进行语义级别的图像分割**
与经典的CNN在卷积层之后使用全连接层得到固定长度的特征向量进行分类的方法不同, FCN将全连接层转化为卷积层, 使其可以接受任意尺寸的图像输入, 然后采用反卷积层对卷积层的 feature map 进行上采样, 使它恢复到输入图像相同的尺寸,, 从而可以对每个图像像素都产生一个预测, 同时保留了原始输入图像中的空间信息, 最后在上采样的特征图上进行逐像素的分类.

# 摘要

卷积神经网络是一种功能强大的视觉模型, 它可以生成具有层次结构的特征. 本文证明了卷积网络经过端到端, 像素到像素的训练, 可以在语义分割方面超过现有水平. 我们的核心观点是构建一个 **"全卷积"** 网络, 它可以接受任意大小的输入, 并通过高效的推理和学习产生相应大小的输出. 我们定义并详细描述了全卷积网络的空间, 解释了它们在 **空间密集** 预测任务中的应用, 并将它们与之前的模型联系起来. 我们将分类网络(AlexNet, VGGnet, GoogLeNet)应用到全卷积网络中, 并且通过 fine-tuning 将它们学习到的特征迁移到实例分割任务中. 然后, 我们定义了一种新的体系结构, 它结合了来自 **深层的, 较粗糙的语义信息和来自浅层的, 教精细的外观信息**, 从而可以产生精确和细致的分割结果.

<div style="width: 550px; margin: auto">![图1](https://wx1.sinaimg.cn/large/d7b90c85ly1g1cw0wotwtj20uh0gz0yk.jpg)

# 介绍

实例分割: 对于每一个像素点都要进行分类.

本文模型无需任何前处理或后处理操作, 如 superpixels, proposals, post-hoc.

本文还定义了一种新式的 "skip" 结构来结合深层粗糙的语义信息和浅层精化的外观信息.

# 相关工作

简单提了一下从分类到实例分割的相关论文. 然后从以下几个方面进行了介绍.

**Fully convolutional networks：** 介绍了全卷积网络的发展的现状, 从90年代开始, 就已经有人开始使用全卷积网络了, 但是全卷积网络相对研究成果还是较少.

**Dense prediction with convnets：** 目前已经有一些工作将convnets应用到了密集型预测任务. 这些方法都包含有以下几点共有特征：
- 模型较小：模型和容量和感受野都有一定限制.
- patchwise training：在预测指定像素时, 只将此像素和其周围像素作为输入送入模型里训练, 即每一个像素都会作为中心像素被训练来预测这个像素所属的分类. patch-wise的问题无疑非常明显, 第一：训练中用的patch会大量重叠, 非常低效. 第二：由于patch-wise大量的计算量, 预测的时候很慢.
- 后处理：超像素映射, 随机field正则化, 局部分类
- input shifting and output interlacing for dense output as introduced by OverFeat
- 多尺寸金字塔处理
- tanh非线性包含
- 融合（ensembles）

FCN则没有以上机制. 与现有模型不同, 本文使用image classification作为有监督的预训练, 同时fine-tune全卷积, 以期望从整张输入图片中快速且高效的学到相应的特征. 目前大多数的模型和方法都不是端到端的.

# Fully convolutional networks

在卷积网络中的每一层的数据, 都是一个 shape 为 $h\times w\times d$ 的三维数组, 其中 $h$ 和 $w$ 代表 spatial dimensions, $d$ 代表特征或通道的深度. 在第一层中, $h\times w$ 代表输入图片的尺寸, $d$ 代表图片的颜色三通道. 我们将更深网络层的某个 location 与图片中的像素连关联的部分, 称为感受野或者接受域(receptive fields).

卷积网络是建立在平移不变性之上的, 它的基本组成操作(卷积, 池化, 激活)都是作用在某一个局部区域上的, 并且仅仅依赖于相关联的一部分区域. 我们令 $x_{ij}$ 代表某一特定层上的特定位置 $(i,j)$ 的数据, 那么下一层的 $y_{ij}$ 的数据可通过下式计算:

$$y_{ij} = f_{ks}(\{x_{si+\delta i, sj+\delta j}\}_{0\leq \delta i, \delta j \leq k})$$

上式中, $k$ 为核的尺寸, $s$ 代表 stride 或者 subsampling factor, $f_{ks}$ 代表某种操作.

如果一个网络中只包含卷积层, 那么我们就将其称为全卷积网络, **全卷积网络可以接受任何尺寸的输入, 同时其输出与输入尺寸相关的结果.**

全卷积网络由于没有了全连接层的限制, 因此可以接受任意尺寸的输入, 并且生成相应的维度形状.

FCN的一个 real-valued 损失函数定义了一个任务：如果损失函数是最后一层中在 spatial dimensions 上的总和 $\ell(x;\theta) = \sum_{ij}\ell'(x_{ij};\theta)$ , 那么它的梯度就会是它所有 spatial components 的梯度的总和. 因此对于在整张图片上 $\ell$ 的 sgd 就是等于将最后一层所有感受野作为一个 minibatch 的$\ell'$的sgd.
当这些感受野重合度非常大时, layer-by-layer 方式的前向计算和反向计算相比于 patch-by-patch 的方式, 就变得十分高效.

下面我们将会解释如何将一个分类网络转换成一个生成粗糙图谱的全卷积网络. 对于像素级的预测任务来说, 我们需要将这些粗糙的图谱与图片中的每个像素对应. OverFeat 为此引入了一个 trick(后文介绍)来完成该任务. 我们将这个 trick 重新以网络的方式解释. 作为另一种有效且高效的替换, 我们引入了反卷积来完成 upsampling 操作.

## Adapting classifiers for dense prediction

大多数经典网络结构中都具有全连接层, 该层接受固定尺寸的输入, 同时会输出 non spatial outputs. 这使得全连接层的维度固定, 同时放弃了空间坐标信息(因为全连接都是一维的). 但是, 这些 **全连接层同样可以被看做是卷积核覆盖整个输入区域的卷积层**. 可以将它们全部强制转换成全卷积网络, 以便可以接受任意尺寸的输入, 同时输出对应的分类图谱(classifization maps). 该转换过程如图2所示

<div style="width: 550px; margin: auto">![图2](https://wx2.sinaimg.cn/large/d7b90c85ly1fwjeuyyn17j20s20kgn41.jpg)

<span id = "FCN 是如何降低计算量的?">
**FCN 是如何降低计算量的:**
面对 $384\times 384$ 的图像, 让含全连接层的初始卷积神经网络以 32 像素的步长独立对图像中的 $224\times 224$ 块进行多次评价, 其效果和使用全卷积网络进行一次评价时相同的. 后者通过权值共享起到了加速计算的作用.
首先, 对于输入 shape 为 $2\times 2\times 256$ 的特征图谱来说, 使用全连接的参数量为 $4096 \times 1$ (输入维度, 输出维度), 使用全卷积时, 核大小为 $2\times 2\times 256$, 核的数量为 $1$, 因此, 从参数量上看, 二者是相同的. **也就是说, 如果我们输入的特征图谱的大小和卷积核的大小相同时, 实际上当前的卷积层就可以看做是一个全连接层.** 但是, 全连接的输入尺寸是固定的, 因此它只能输出指定维度的结果, 而全卷积层的另一个好处就是不会限制输入的特征图谱的尺寸, 因此, 当输入的特征图谱的尺寸大于当前的核时, 全卷积网络就会输出多个结果, **而实际上, 这些结果对于的输入特征图谱上的区域具有大量的重叠, 因此, 起到了权重共享的作用, 从而可以减少重复计算, 降低模型复杂度.** 另一方面, 即使当输入的特征图谱和卷积核的大小一致时, 全卷积和全连接也不是完全等价的. 因为全连接的结构是固定的, 当我们训练时, 全连接上权重会发生变化, 但是连接关系不会改变. 而卷积层除了会学习权重外, 它还会学习节点之间的关系, 对于无用的关系它会直接弱化或过滤掉.

AlexNet 处理一张 227 尺寸的图片需要 1.2ms, FCN 版本的 AlexNet 在 500 尺寸的图片上输出 $10\times 10$ 的结果需要 22ms. 因此快了大约 5 倍.($1.2\times 100 = 120$, 大约是 22 的 5 倍).

这些卷积模型的空间输出的特征图谱使得它们称为语义分割之类的密集问题的自然选择. 由于每个输出单元都有 ground truth 可用, 因此前向和后向的传递过程都很简单, 并且都利用了卷积固有的计算效率(以及积极的优化).

AlexNet 反向传播过程需要 2.4 ms, FCN 版本在 $10\times 10$ 的输出上反向传播需要 37ms.

尽管我们的全卷积网络可以接受不同尺寸的输入, 但是特征图谱的输出维度是通过 subsampling 来降低的. 我们需要控制核的个数和大小, 使得模型的计算成本不会太高, 这就会使得全卷积网络的输出较为粗糙, 使其从输入图谱的大小减少了一定比例.

## Shift-and-stitch is filter rarefraction

做了很多研究和实验, 但是并没有使用该技术, 作者发现直接通过 upsampling 进行学习更加有效, 如下节描述.

## Upsampling is backwards strided convolution

另一种将粗糙的输出图谱连接到密集像素的方法是插值. 例如, 简单的双线性插值通过线性映射从最近的四个输入计算每个点的输出, 线性映射仅依赖于输入和输出单元的相对位置.

从某种意义上说, 因子为 $f$ 的上采样可以看成是步长为 $1/f$ 的卷积. 因此可以使用反卷积来实现(deconvolution). 该过程很容易实现, 因为主需要逆转正常卷积的计算过程即可. 注意到, 反卷积相对于双线性插值来说有一个好处就是可以进行学习, 而不是固定不变的. 将多个反卷积层和激活层堆叠起来甚至可以学习到非线性的上采样过程. 在实验中, 我们发现, 反卷积对于学习密集预测来说是快速有效点.

## Patchwise training is loss sampling

讨论在大的输入上通过FCN一次计算多个结果, 可以在一定程度上修正loss?

# Segmentation Architecture

我们利用 ImageNet 进行预训练. 之后, 我们通过构建一个 skip architecture 来结合深层和浅层信息.

我们使用逐像素的多项式逻辑回归损失进行训练, 并且使用评价像素 IoU 的标准度量进行验证.

## From classifier to dense FCN

使用 VGG16 和 GoogLeNet(没有使用最后的全局平均池化), 去掉了最后一层分类层, 并且将所有的全连接层转换成全卷积层, **我们添加了一个核大小为 $1\times 1$, 通道数为 21 的卷积层来预测深层粗糙特征图谱上每个位置的类别得分, 后面跟了一个反卷积层对粗糙的特征图谱进行上采样得到更密集的图谱.** 这样的设置在 VGG16 backbone 上已经取得了 sota 的效果. 如表1所示.

<div style="width: 550px; margin: auto">![表1](https://wx3.sinaimg.cn/large/d7b90c85ly1g1cw1k4qofj20us0mvjw5.jpg)

## Combining what and where

如图3所示, 我们定义了一种新的用于分割任务的全卷积网络, 它结合了不同网络层的特征, 并且精细化了输出的空间精度.

<div style="width: 550px; margin: auto">![图3](https://wx4.sinaimg.cn/large/d7b90c85ly1g1cw2p7nsvj21md0logr4.jpg)

尽管前面介绍的分割网络已经取得了不错的效果, 但是最终预测层的 32 像素的步长使得上采样输出的细节尺度有所限制. 我们通过添加链接来解决这一问题, 这些链接将最终的预测层与较浅的网络层结合在一起, 从而具有更小的步长. 我们可以用一个 DAG 来描述这个过程, 特征会从较浅的层跳到较深的层进行结合. 将精细层和粗糙层结合起来, 可以使得模型在做出局部预测的同时, 考虑全局结构.

我们首先通过一个步长为16像素的网络层上预测
我们在 pool4 上添加了一个 $1\times 1$ 的卷积层来生成额外的类别预测, 我们将 conv7(pool4上的预测层) 的输出和 pool2 上采样以后的输出进行融合, 并且将预测结果相加. 我们用双线性插值来初始化上采样, 但是允许上采样网络层学习参数. 最终, 步长为 16 的预测结果会上采样会原始图片, 我们将其称之为 FCN-16s.
通过这种 skip net 的形式, 我们可以将最终的 IU 性能提升 3 个百分点. 图4展示了精化后的结果的输出. 用同样的方法可以构建出 FCN-8s. 从表2可以看出, FCN-8s 的提升已经不是很明显了, 因此我们没有继续往下构建.
<div style="width: 550px; margin: auto">![图4](https://wx1.sinaimg.cn/large/d7b90c85ly1g1cw1so8yij20tp0hpgnw.jpg)
<div style="width: 550px; margin: auto">![表2](https://wx1.sinaimg.cn/large/d7b90c85ly1g1cw2i3lcrj20uq0eb774.jpg)


## Experimental framework

**Optimization:** SGD with momentum, lr = 0.001/0.0001/0.00001, momentum = 0.9, weight decay = 0.00005 / 0.0002.

**Fine-tuning:** fine-tune all layers.

**Patch Sampling:** 在同一张图片上选取 batch, 使得具有较大重叠, 加速训练. 图5 显示了这种形式的采样对收敛的影响.

<div style="width: 550px; margin: auto">![图5](https://wx3.sinaimg.cn/large/d7b90c85ly1g1cw3nj388j20up0jvq5t.jpg)

**Class Balancing:** unnecessary

**Dense Prediction:** 借助双线性插值(初始时)和反卷积(后续进行学习)进行上采样.

**Augmentation:** no noticeable improvement

**More Training Data:** 提升了 3.4 个点

**Implementation:** Caffe, NVIDIA Tesla K40c.

<div style="width: 550px; margin: auto">![图6](https://wx1.sinaimg.cn/large/d7b90c85ly1g1cw4sq58oj20u00wnqk7.jpg)
<div style="width: 550px; margin: auto">![表3](https://wx2.sinaimg.cn/large/d7b90c85ly1g1cw40ko2sj20t20dnjtp.jpg)
<div style="width: 550px; margin: auto">![表4](https://wx4.sinaimg.cn/large/d7b90c85ly1g1cw48pgxwj20r40ivwil.jpg)
<div style="width: 550px; margin: auto">![表5](https://wx2.sinaimg.cn/large/d7b90c85ly1g1cw4eo4spj20sk0mfjw7.jpg)
<div style="width: 550px; margin: auto">![表6](https://wx3.sinaimg.cn/large/d7b90c85ly1g1cw6ri96mj20sh0qwjvz.jpg)
