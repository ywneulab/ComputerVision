---
title: YOLOv3
sitemap: true
date: 2018-09-14 18:46:50
categories:
tags:
- 计算机视觉
- 目标检测
- 论文解读
---

# 摘要

作者对YOLOv2进行了一些改进，使之在保持实时检测的同时，准确率又有所提升了。

# 介绍

作者说他这一年（18年）基本没干啥，就是打打电话，玩玩推特，偶尔还帮别人干点活。。

然后因为只对YOLO做了一些改进，但是并没什么特别的地方，因此就写了这一篇技术报告,而没有选择发表成论文形式。

# The Deal

作者说了，他们大部分的工作都是从别人那里吸取好的点子，同时训练了一个新的分类器网络（比别人的好，恩。。）

## Bounding Box Prediction

和YOLO9000一样，在预测bounding box时使用了dimension clusters和anchor boxes。

YOLOv3在预测每个bouding box的objectness score时，使用的是logistic regression。

与faster rcnn不同的是，我们的系统只会给每个gt object指派一个bounding box。如果没有指派的话，就说明没有对象的box坐标，只有objectness。

## Class Prediction

每个box使用了多标签分类，我们不选择softmax是因为发现它很难取得好的效果，因此，改用一个单独的logistic classifiers。在训练阶段，使用binary cross-entropy loss来进行类别预测。

## Predictions Across Scales

YOLOv3在三种不同的scales下进行预测。

## Feature Extractor

作者使用了一个新的网络模型来提取特征，主要是在Darknet-19中引入了residual network stuff，最终模型的卷积层数达到53层，也就是Darknet-53。

## Training

仍然使用不带hard negative mining的图片训练。同时使用了multi-scale training，data augmentation，batch normalization，以及其他的一些标准程序。


# How We Do

根据不同的评价标准，YOLO的性能差异较大，总的来说主要是因为YOLO虽然能标出物体的大致位置，但是画出的框并不是“完美”，使得在IOU要求高的评价标准上，YOLO的得分很低。

另外， 之前的YOLO在检测小物体上往往有很多瓶颈，而目前的YOLO已经在慢慢克服这方面的缺陷

# Things We Tried That Didn't Work

**Anchor box $x,y$ offset predictions**


**Linear $x,y$ predictions instread of logistic**


**Focal loss**



**Dual IOU thresholds and truth assignment**

# What This All means

最后，作者说了为什么要选择其他的评价标准。

对于人类来说，很难直接区分出IOU0.3和IOU0.5之间的差别，那么我们要求计算机这样做是否合理呢（我认为是合理的。。。）

最后作者说出了对计算机视觉未来发展的一些“愿景”。（作者反对隐私泄漏和军事用途）
