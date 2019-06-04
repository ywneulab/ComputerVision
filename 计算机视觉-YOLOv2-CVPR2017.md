---
title: YOLO9000
sitemap: true
date: 2018-09-13 21:13:35
categories: 计算机视觉
tags:
- 计算机视觉
- 论文解读
- 目标检测
---



# 核心亮点
https://zhuanlan.zhihu.com/p/35325884

# 关键技术

# 论文细节

# 背景介绍

https://zhuanlan.zhihu.com/p/40659490

<div style="width: 600px; margin: auto">![](https://wx1.sinaimg.cn/mw1024/d7b90c85ly1fw97m47l6uj21e70hogsn.jpg)

# 摘要

本文提出了一个新的，实时目标检测模型，YOLO9000。首先，作者使用了不同的提升技巧来优化YOLO模型，同时，利用多尺度的训练方法，YOLO可以方便的在speed和accuracy之间进行tradeoff。在67FPS时，YOLOv2可以在VOC2007上活得76.8的mAP，在40FPS时，YOLOv2可以或者78.6mAP，超过了Faster RCNN和SSD的性能表现。尽管只有200个类里面的44个类的数据，YOLO9000仍然可以在ImageNet上获得19.7的mAP，对于不在COCO里面的156个类，YOLO可以获得16.0的mAP。9000的含义是说YOLO-v2可以运行在超过9000个不同的物体类别，同时保持实时检测。

# 介绍

目前关于分类任务的数据集数量远远超过检测任务的数据集大小，而短期内，检测任务的数据集数量无法快速增长。对此，本文提出了一个新的方法来利用现有的分类任务的数据集，进而扩充当前目标检测系统的检测范围。

同时，本文还提出了一个联合训练方法，使得我们可以同时在检测数据集和分类数据集上训练检测器。（检测数据集提升定位能力，分类数据集提升类别容量和系统健壮性）

本文分两步：首先将YOLO升级到YOLOv2,然后利用本文提出的数据集联合方法来进行联合训练。

# Better

YOLO的主要缺点在于定位错误和较低的召回率。 本文在优化这些缺点时，并不是选择扩大网络的规模，而是将整个网络简化，使表征信息更容易学习。根据之前的工作，我们采用了很多方法来提升YOLO的性能。

## Batch Normalization：
在所有的卷积层之上加上BN，可以提升2%的mAP，并且可以去掉dropout层而不产生过拟合。

## 高分辨率分类器 High Resolution Classifier：
之前的YOLO是利用ImageNet的224大小的图像预训练的，然后在检测时，会将224的图像放大到448尺寸。在YOLOv2,首先在448的ImageNet图像上进行finetune 10 epochs。这给了一定时间让网络适应更大的尺寸大小，然后再在该网络进行物体检测的finetune。 这可以提升4%mAP。

## Convolutional With Anchor Boxes：
YOLO使用叠在卷积层之上的全连接层的特征提取器来直接预测bounding box的坐标。 相比于YOLO，Faster RCNN使用了精心挑选的方式来获得预测的boundign box，它在anchor box的基础上进行预测，并且其预测层是卷积层。为此，本文移除了YOLO的全连接层，改用anchor box来预测bouding box。

首先，移除了一个pool层，从而使网络卷积层的输出有更高的分辨率。另外，还将网络的输入图像的分辨率降到416,这么做的原因是作者希望在特征图谱上得到奇数个locations，这样一来，就由一个center cell。YOLO的结构可以使416的图像降到13×13的大小。

在使用anchor box时，我们将类别预测问题从位置标定问题中分离出来，然后为每个anchor box预测类别和是否有物体。和YOLO一样，预测是否有物体时会预测gt和proposed box的IOU，类别预测时会计算给定有物体的条件下给出属于每个class的条件概率。

原来的YOLO会对每张图片产生98个box，而使用anchor box后，每张图片会产生上千个box。 不用anchor box时，本文的模型可以达到69.5的mAP和81%的recall。而是用了anchor box后，可以到大69.2的mAP和88%的recall。虽然mAP变低了，但是recall的提升说明本模型还有很大的提升空间。

## Dimension Cluster：
在使用anchor box时，主要遇到了两个问题。

第一：anchor box的维度是手动标定的。 anchor值的选择会对最终结果有一定影响。为了解决这个问题，我们不采用手动标定的方法，而是对训练集的boudning boxes用k-means clustering来自动找到较好的anchor值。如果使用基于欧式距离的标准k-means，那么更大的box就会产生更多的error。为了不让box的大小对最终的anchor值有影响，我们使用下面的式子作为距离度量：

$$d(\text{box},\text{centroid}) = 1 - IOU(\text{box}, \text{centroid})$$

最终在模型复杂度和高召回率的权衡下，本文选择 $k=5$ 。


## 直接位置预测 Direct location prediction

使用anchor box的第二问题就是：模型不稳定，尤其是在早起迭代阶段。稳定性差的主要来源是box的坐标 $(x,y)$ ，在RPN网络中，网络会预测 $t_x$ 和 $t_y$ ，于是 $(x,y)$ 的值可以通过下面的公式计算得到：

$$x = (t_x*w_a) - x_a$$

$$y = (t_y*h_a) - y_a$$

本文不使用上面的方法，而是使用YOLO中的方法，预测相对于grid cell位置的相对坐标，这将gt限制在了0到1之间。这样的参数设置使得参数更容易学习，网络更加稳定。

## Fine-Grained Features

精细化的13×13的特征图谱对于标定大物体来说已经足够了，同时，由于特征更加细粒度，使得它在标定更小的物体时有一定提升。 Faster RCNN和SSD都在不同的特征图谱上执行，因此，它们得到的是一个区间的图像分辨率大小。  本文采用一个更简单的测率，直接添加一个passthrough层，使得从更早的26×26的层中得到特征。

**这个passthrough层将高分辨率的特征和低分辨率的特征连接起来，通过将相邻特征堆叠到不同的channes？** 这将26×26×512的特征图谱变成了一个13×13×2048的特征图谱。

## Multi-Scale Training

为了使模型更加健壮，使用了不同尺度的图片来训练模型。在训练时，每经过一段迭代次数后，都会改变接受输入图片的size。由于本文的模型输出的尺寸会变成原来1/32,因此选择以下尺寸：{320,352,...,608}。

这样一来，一个网络可以在不同的分辨率下进行目标检测，可以取得更好的效果。

# Faster

大多数目标检测网络使用了VGG16作为基础网络，但是VGG16需要30.69billion浮点运算，十分复杂。

而本文使用基于GoogleNet的自定义网络，只需要8.52billion浮点运算。（但是精确性低于VGGnet）

## Darknet

最终网络起名为Darknet-19。 具有19个卷积层和5个最大池化层。

## Training for classification
将网络在标准Imagenet 1000上进行训练。

SDG的初始学习率为0.1, 递减指数为0.4,权重递减为0.0005,momentum为0.9。

在训练时，使用了标准的数据增广方法：random crops，ratations，hue，saturation，exposure shifts等。

## Training for detection
将上面训练好的网络的最后一层卷积层移除，然后加上三个具有1024个filter的3×3卷积层，并在其中跟一个1×1的卷积层，使输出是我们需要的结果。 比如，对于VOC，需要有5个box，每个box有5个coordinates和20个class，所以需要125个filetes。 同时使用了passthrough层，以便模型可以使用fine grain features。

# Stronger

本文提出了一种可以联合训练分类数据和检测数据的机制。本文的方法使用检测数据的图像标签来学习物体位置信息，使用分类数据的标签来扩充可以检测的物体的类别。

在训练阶段，我们了检测数据和分类数据混合。当网络模型看到一个带有检测标签的图片时，就会对YOLOv2的整个损失函数进行BP求导，当看到分类图片时，则只会对分类部分的损失函数进行BP求导。

上面的方法具有一些难点：
