---
title: SPPNet (ECCV, 2014)
sitemap: true
categories: 计算机视觉
date: 2018-10-13 16:28:50
tags:
- 计算机视觉
- 目标检测
- 论文解读
---

**文章:** Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition
**作者:** Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun



# 核心亮点

**(1) 提出了一种新的池化方法---空间金字塔池化SPP**:
- 可以接受任意尺寸的输入图片,并生成固定长度的表征向量
- 可以进行多尺度的联合训练, 提升模型精度
- 这种池化方法是比较general的, 可以提升不同模型架构的性能(分类任务)

**(2) 将SPP用于目标检测, 并且提出了先求卷积特征图谱, 后取区域的的策略:**
- 大大提升了模型训练和预测的速度(在预测阶段, 比RCNN快24~102倍, 同时取得了更好的精度).

**注1: 在特征图谱上使用检测方法不是首次提出**, 而SPP的贡献可以结合了deep CNN结构强大的特征提取能力和SPP的灵活性, 使得精度和速度同时提高
注2: 相比于RCNN, SPPNet使用了EdgeBoxes( $0.2s/img$ )的方法来进行候选区域推荐, 而不是Selective Search( $1\sim 2s/img$ )
注3: SPPNet在ILSVRC2014的目标检测任务上取得第二名, 在图片分类任务上取得第三名

# 论文细节

## 背景介绍

指明固定图片输入尺寸的缺点: 需要剪裁或wrap图片, 无法包含整个图片信息, 导致信息丢失或引起形变

破除尺寸限制的做法: 在最后一个卷积层之后, 添加SPP层,  **SPP层可以将不同尺寸的feature map池化成固定长度的特征向量.**, 之后, 仍然与普通网络一样, 后街全连接层或者其他分类器

## SPP

本文的核心就在于SPPNet, 其主要用法是在卷积神经网络的最后一层卷积特征图谱上, 通过多尺度的网格划分, 接受任意尺寸的特征图谱输入, 同时输出固定长度的特征向量, 以此来实现网络模型接受任意尺寸图片输入的目的.

SPPNet实现原理如下图所示:

<div style="width: 600px; margin: auto">![](https://wx1.sinaimg.cn/mw690/d7b90c85ly1fw6muop69aj20mn0j7mzu.jpg)

首先, 设定好固定的网格划分方法, 以便得到spatial bins, 如上图, 有三种不同spatial bins, 网格划分力度分别为 4×4, 2×2 和 1×1, 因此, spatial bins的数量为:$4\times 4+2\times 2+ 1\times 1 = 21 = M$, 图中的256代表最后一层卷积层的filter的数量, 也就是特征图谱的depth = $k$. 因此, SPP层的输出为 $kM$ 维的一维向量.

注意: 这里最粗粒度的spatial bins 是对整张特征图谱上进行pooling, 这实际上是为了获得一些全局信息.(之前也很很多work集成了这种全局pooling方法, 貌似有助于提升精度, 同时由于是全局信息, 所以相当general, 可以一定程度上起到降低过拟合的作用)



SPP可以看做是Bag-of-Words(BoW)模型的一种扩展

SPP用于deep CNNs时, 具有一些瞩目的性质:
1. 相比于sliding window pooling的方式, SPP可以生成固定尺寸的特征向量
2. SPP使用了多尺寸的spatial bins, 而sliding window只是用了单一的window size. 这使得SPP对于物体的形变更加鲁棒
3. 由于输入尺寸的灵活性, SPP可以在不同尺度下提取图片特征(可以在训练时接受多尺寸土木, 降低过拟合风险)

SPP使用了MutiBoxes来进行候选区域推荐.

## Deep Networks With Spatial Pyramid Pooling

### 卷积层和特征图谱

卷积层可以接受任意尺寸的图片, 并且可以在特征图谱上得到相应的响应.

根据BoW的思想, 在 deep convolutional features 上也可以使用(与Bow)相似的pooling方式使得输出固定长度的特征向量

### Training the Network

虽然上面的work可以接受任意尺度的图片输入, 但在实际训练中, GPU相关组件如caffe或者cuda-convnet还是倾向于接受固定尺寸的输入. 因此作者介绍了他们的training solution, 以便在使用GPU高计算能力的同时, 保留SPP的pooling特点.

#### Single-size training

假设最后一层卷积层的特征图谱的size为 $a\times a$. pyramid level为 $n\times n$ bins. 那么实现该pyramid pooling的方式可以看做是 $win=\lceil  a / n\rceil , stride = \lfloor a/n \rfloor$ 的sliding window pooling. 对于不同级别的pyramid, 使用同样的方法, 最终将所有的bins的输出连接起来组成一个一维向量.

上面用一句话总结: 在实际实现中, 使用sliding window pooling的方式来实现spp的, 对于单一尺寸的输入, 可以提前计算好需要的windows size, 编码时直接常量定义相关pooling即可.

#### Multi-size training

对于不同的尺寸输入, 分别计算不同的windows, 然后定义相关的pooling. (需要重新定义网络的pooling参数)

为了减少训练不同size网络的间接成本, 作者会先训练一个网络, 在一个epoch完成后, 会转向训练另一个网络(两个网络共享参数, 注意pooling是没有参数的)

也就是说, 对于两个网络, spp的参数设置是根据图片的尺寸来调节的, 因为要得到固定长度的特征向量, 所以如果图片尺寸较大, 那么最后一层卷积层的特征图谱也就较大, 那么久需要比较大的sliding windows size 来实现spp. 作者为了能一次性训练不同尺寸图片, 采用了这种迭代训练的方式(反正 **pooling层是没有参数的** , 只不过是定义时需要指定window size).

进行多尺度训练的目的: 为了协同不同尺度下的图片特征提取, 同时利用已有的优化技术.

注意: 以上策略仅仅在training阶段使用.

## SPP-Net For Image Classification

### Experiments on ImageNet 2012 Classifization

训练策略:

images resize 使得图片的短边长度为256, 然后对其进行224×224 crop(四角+中心).

图像增强: horizontal flipping, color altering.

dropout: 用于2层全连接层

lr: 0.01, 两次衰减10分之1

#### Baseling Network Architectures

ZF-5, Convnet*-5, Overfeat-5/7

#### Multi-level Pooling Improves Accuracy

需要注意的是, 模型精度的不是因为更多参数(pooling没有参数), 但是因为考虑了这种spatial结构, 从而使得模型的精度提升.

#### Multi-size Training Improves Accuracy

顾名思义, 作者使用了多尺度的训练, 同样提升了精度

#### Full-image Representations Improve Accuracy

作者将Full-Image和crop(224×224, center)的策略进行了实验比较, 发现未经裁剪的Full-Image的精度更高, 说明维持原始图片的完整输入是很有必要的.

作者发现, **即使使用了很多视角下的crop结果进行投票, 额外的增加两个full-image(with flipping) 仍然可以提升模型 0.2%的精度**

#### Multi-view Testing on Feature Maps

受到目标检测算法的启发, 作者发现整合在feature maps上面进行不同视角的预测, 会使得模型精度有所提高. 实验结果证明, 在features maps上面进行10-view投票相比于直接在原始图上面进行10-view投票, 精度提高了0.1% (top-5 error)

作者使用了是不同image size, 结合了不同的view(18种, center+四角+四边中心+filp,), 总共有96种view(18*5 + 6(图片size只有224的时候)). 将top-5 error 从10.95% 降到 9.36%, 再结合two full image view, 降到了9.14%.

**Overfeat 也是从feature map中获取不同views的, 但它不能处理多尺度的图片**

### Experiments on VOC 2007 Classification

### Experiments on Caltech101

## SPP-Net For Object Detection

RCNN: 先从原始图像中选出大约2000个框, 然后将这些图像区域缩放到227×227大小, 然后对每个区域进行卷积分类, 使用了SVM分类器.  **特征提取存在大量重复计算, 占用了绝大部分的时间**

改进:

从feature maps提取对应框,只进行一次卷积计算:  之前的DPM在HOG特征图谱上选框, SS在SIFT特征图谱上选框, Overfeat在深度卷积特征图谱上选框, 但是Overfeat必须固定图片尺寸.

相比之前的这些方法, SPP具有在深度卷积特征图谱上的灵活性, 可以接受任意尺寸的输入,

### Detection Algorithm

使用 "fast" mode的SS来生成2000个候选区域框, 然后将image 进行resize ,使其 min(w,h) = s. 接着对整张图片计算卷积特征图谱. 然后在特征图谱上选框, 接着对这些框进行spp, 最后使用svm进行分类.

<div style="width: 600px; margin: auto">![](https://wx1.sinaimg.cn/mw690/d7b90c85ly1fw6pd0czr9j20n10ltn0b.jpg)

使用了strandard hard negative mining来训练svm.

### Detection Results

### Complexity and Running Time

改用了MultiBox算法, 处理每张图片只需要0.2左右, 原来的SS算法大约需要1~2s才可以.

### Model Combination for Detection

在ImageNet上训练另一个模型, 使用相同的网络结构, 但是随机初始化状态不同. 如此得到两个模型, 他们的性能表现差不多.

首先分别用这两个模型给所有的测试样例的候选框进行打分, 然后利用NMS消除这些候选框(包含两个模型的预测结果)中的重复框, 这样一来, 就会保留下置信度更高的框. mAP从 59.1%, 59.2% 提升到了 60.9%


### ILSVRC 2014 Detection


## Conclusion

## Appendix A

### Mean Subtraction

### Implementatioin of Pooling Bins

### Mapping a Window to Featrue Maps
