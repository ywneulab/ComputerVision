---
title: SSD-Single Shot MultiBox Detector
sitemap: true
categories: 计算机视觉
date: 2018-10-17 15:39:04
tags:
- 计算机视觉
- 目标检测
- 论文解读
---

**文章:** SSD: Single Shot MultiBox Detector
**作者:** Wei Liu, Dragomir Anguelov, Dumitru Erhan

# 核心亮点

**(1) 在不同层的feature map上使用网格法获取候选区域:**

某种程度上SSD可以看做是YOLOv1和FasterRCNN的结合, 在不同layer的输出的不同尺度的feature map上划格子, 在格子上利用`anchor`思想. 因此, . (YOLOv2也使用了Anchor的思想)

**(2) 使用了数据增广, 难样例挖掘, atrous算法等trick大幅度提升精度和速度**
这个其实算不上亮点, 只不过作者确实使用这些技术提升性能不少

**(3) 相对于那些需要object proposals的两阶段模型, SSD方法全完取消了 proposals generation, pixel resampling 或者 feature resampling 这些阶段**

# 论文细节

## 背景介绍

目前, 虽然Faster RCNN取得了很好的检测精度, 但是对于普通设备来说, Faster RCNN过于庞大, 计算时间太久, 不足以达到实时监测. YOLO虽然速度很快, 但是精度太低了, 达不到基本要求.

SSD的出现正是为了解决这些问题, 确定在不丢失过度精度的前提下, 提升检测的速度.

## Single Shot Detector(SSD)

<div style="width: 550px; margin: auto">![](https://wx1.sinaimg.cn/mw1024/d7b90c85ly1fwb7fnbsu4j20oj09bgq8.jpg)

根据上图, 简单说一下SSD的关键要素
- 输入: 图像以及每个物体对应的ground truth boxes
- 多特征图谱的anchor思想: 在不同尺度的特征图谱上(如上图的4×4和8×8), 对每个位置上设置多个具有不同大小和长宽比的boxes, 称之为 **default boxes**.
- 输出: 对于每一个default box, 都会输出4个相对位移用于边框回归, 同时会输出所有类别的预测概率
- 匹配: 在预测阶段, 需要将这些 **defaults boxes** 与 **gt boxes** 匹配. 在上图中, 最终有两个框(蓝色)与猫所在框匹配, 有一个框(红色)与狗所在框匹配. 这三个框被标记为正样本, 其余剩下的框都被标记为负样本. (可见负样本数量远远大于正样本数量)
- 损失函数: 边框回归损失(Smooth L1) 和 类别置信度损失(softmax 交叉熵损失) 的权重和.

## Model
SSD 会产生固定数量的bounding box, 以及每个bounding box的各个类别的预测概率, 最后会使用NMS算法生成罪最终的检测结果.

**多尺度feature map:** 在卷积网络的不同卷积层后面添加convolutional feature layers, 并且在每一层中都会进行检测任务.

**Convolutional predictors for detection:** 每一个添加的特征层(或者在基础网络里的特征层), 都可以使用一系列的卷积核产生固定大小的predictions, 如图2所示. 对于一个大小为 $m\times n$, 具有 $p$ 通道的特征层, 使用的卷积核就是 $3\times 3\times p$ , 之后会产生相对于 default box 的预测坐标, 已经每一类的预测置信度. 对于特征图上 $m\times n$个位置, 在对每个位置使用卷积核之后, 都会产生一个输出值. (YOLO架构则是用一个全连接层来代替这里的卷积层)

**Default boxes and aspect ratios:** 每一个feature map的cell都会与一系列 default bounding box 相关联. 对于每一个cell来说, 模型会预测与之关联的 default bounding box 相对于该cell的偏移量, 同时会预测这些boxes对应的每一类的出现概率. 具体来说, 对于一个位置上的 $k$ 个boxes中的每一个box, 都会计算出这个box的4个相对位移值 $c$ 个类别的score. 因此, 对于一个feature map来说, 总共需要 $(c+4)\times k$ 个卷积核, 最终该对于大小为 $m\times n$ 的 feature map来说, 其输出结果数量为: $(c+4)\times k\times m\times n$.

<div style="width: 550px; margin: auto">![](https://wx2.sinaimg.cn/mw1024/d7b90c85ly1fwbhhr03m4j216z0rtdmo.jpg)

**空洞卷积(Dilation Conv)**

采用VGG16做基础模型，首先VGG16是在ILSVRC CLS-LOC数据集预训练。然后借鉴了DeepLab-LargeFOV，分别将VGG16的全连接层fc6和fc7转换成 $3 \times 3$卷积层 conv6和 $1 \times 1$ 卷积层conv7，同时将池化层 pool5 由原来的 stride=2 的 $2\times 2$ 变成 stride=1 的(猜想是不想reduce特征图大小)，为了配合这种变化，采用了一种 Atrous Algorithm，其实就是conv6采用扩展卷积或带孔卷积（Dilation Conv），其在不增加参数与模型复杂度的条件下指数级扩大卷积的视野，其使用扩张率(dilation rate)参数，来表示扩张的大小，如下图所示，(a)是普通的 $3 \times 3$ 卷积，其视野就是 $3\times 3$ ，(b)是扩张率为 1，此时视野变成 $7\times 7$ ，(c)扩张率为3时，视野扩大为 $15 \times 15$ ，但是视野的特征更稀疏了。Conv6采用 $3\times 3$ 大小但dilation rate=6的扩展卷积。

https://wx4.sinaimg.cn/mw690/d7b90c85ly1g178tkhu8kj20lc07lgq1.jpg

## Training

SSD与二阶段检测方法在训练时的区别: SSD训练图像中的GT信息需要赋予到那些固定输出的boxes上面(也就是说不仅要使用bounding box的坐标, 还有把类别标签也与每一个box绑定, 这种方法在YOLO, FasterRCNN(只分前后景), MultiBOx中都有用到).

**Matching strategy:**

只要预测框与gt box之间的 jaccard overlap(就是交并比) 大于一个阈值(0.5), 就认为是配对成功, 反之, 认为是背景.

**Training objective:**

SSD的损失函数源自于MultiBox的损失函数, 但是SSD对其进行拓展, 使其可以处理多个目标类别.  用 $x_{ij}^p={1,0}$ 表示第 $i$ 个default box 与类别 $p$ 的第 $j$ 个gt box匹配, 否则若不匹配的话, 则 $x_{ij}^p = 0$.

根据上面的策略, 一定会有 $\sum_i x_{ij}^p >1 $ 的情况出现, 意味着对于第 $j$ 个gt box, 很有可能有多个default box与之匹配.

总的目标函数是由Localization loss(loc) 和 confidence loss(conf) 的加权求和得到的:

$$L(x,c,l,g) = \frac{1}{N} (L_{conf}(x,c) + \alpha L_{loc}(x,l,g))$$

式中:
- $N$是与ground truth box 相匹配的 default boxes个数(如果N为0, 则将该项loss设为0)
- localization loss(loc) 是 Fast RCNN 中的Smooth L1 loss, 用于对bounding box进行回归, 与Faster RCNN一样, 我们会将真实的gt box坐标值转换成相对于default box( $d$ )中心 $(cx,cy)$ 的偏移量和缩放度, 预测的时候也是对偏移量和缩放度进行预测:

$$L_{loc}(x,l,g) = \sum_{i\in Pos}^N \sum_{m\in\{cx,cy,w,h\}} x_{ij}^k smooth_{L_1}(l_i^m - \hat g_j^m)$$

- confidence loss(conf) 是 Softmax 交叉熵loss

$$L_{conf}(x,c) = -\sum_{i\in Pos}^N x_{ij}^p log(\hat c_i^p) - \sum_{i\in Neg} log(\hat c_i^0), 其中, \hat c_i^p = \frac{exp(c_i^p)}{\sum_p exp(c_i^p)}$$

- 权重项 \alpha, 默认设置为1.

**Choosing scales and aspect ratios for default boxes:**

大部分CNN网络在越深的层, feature map的尺寸会越来越小, 这样做不仅仅是为了减少计算与内存的需求, 还有个好处就是, feature map往往具有一定程度的平移和尺度不变性.
为了处理不同尺度的物体, OverFeat和SPPNet都是通过在feature map上进行不同尺度的pooling, 然后再将这些pooling综合进行获取固定长度的特征向量输出.

SSD采用的策略是使用同一个网络中不同层的feature map, 这些feature map也是不同尺度的, 同时也具有共享参数的好处. 本文使用了 8×8 和 4×4 大小的feature map.

假如feature maps数量为 $m$, 那么每一个feature map中的default box的尺寸大小计算如下:

$$s_k = s_{min} + \frac{s_{max} - s_{min}}{m-1}(k-1), k\in [1,m]$$

上式中, $s_{min} = 0.2 , s_{max} = 0.9$. 对于原文中的设置 $m=6 (4, 6, 6, 6, 4, 4)$, 因此就有 $s = \{0.2, 0.34, 0.48, 0.62, 0.76, 0.9\}$
然后, 几个不同的aspect ratio, 用 $a_r$ 表示: $a_r = {1,2,3,1/2,1/3}$, 则每一个default boxes 的width 和height就可以得到( $w_k^a h_k^a=a_r$ ):

$$w_k^a = s_k \sqrt{a_r}$$

$$h_k^a = \frac{s_k}{\sqrt {a_r}}$$

对于宽高比为1的 default box, 我们额外添加了一个 scale 为 $s_k' = \sqrt{s_k s_{k+1}}$ 的 box, 因此 feature map 上的每一个像素点都对应着6个 default boxes (**per feature map localtion**).
每一个default box的中心, 设置为: $(\frac{i+0.5}{|f_k|}, \frac{j+0.5}{f_k})$, 其中, $|f_k|$ 是第 $k$ 个feature map的大小 $i,j$ 对应了 feature map 上所有可能的像素点.
**在实际使用中, 可以自己根据数据集的特点来安排不同的 default boxes 参数组合**

这种多尺度的feature maps策略, 可以适应不同大小的物体的检测, 如下图, 对于体积较小的猫来说, 它会与 8×8 feature map 的default box匹配, 而对于体积较大的狗来说, 它会与 4×4 feature map 的default box 匹配, 而其余的都会被看做是负样本.

**Hard negative mining:**

在一系列的matching之后, 大多数default boxes都会被认为是负样本. 这会导致正负样本不均衡问题. 对于, SSD会将所有的负样本按照scores loss的高低进行排序(损失高的优先级在前), 然后每次只选择顶端的负样本进行训练, 负样本与正样本之间的比例为 3:1.

**Data augmentation:**

为了增强对物体多尺度和形状的鲁棒性, 对于每一张训练数据, 会随机使用下列数据增广技术:
- 使用整张图片
- 采样一个patch (这里的patch我认为就是裁剪子区域的感觉), 使其与物体之间的最小交并比为0.1, 0.3, 0.5, 0.7, 0.9
- 随机采样一个patch

采样的patch与原始图像大小的比例为[0.1, 1], aspect ratio在 0.5 到 2 之间. 当gt box的中心出现在采样的patch中时, 我们保留重叠部分. 在这些采样步骤之后, 每一个采样的patch被resize到固定的大小, 并且以0.5的概率被水平翻转.

## 实验

- hole filling algorithm??//TODO
- SSD模型对于bounding box的size非常敏感, 也就是说, SSD对于小物体目标较为敏感, 在检测小物体目标上表明交差, 主要也是因为对于小目标而言, 经过多层卷积之后, 剩余信息过少导致的.
- 数据增广对于结果的提升非常明显
- 多feature map对结果的提升是有很大帮助的
- 使用较多的default boxes, 效果较好(但也不能太多)
- atrous: atrous是精度有一定提高, 同时对速度有很大提高(约20%)



**Inference time:**

SSD会产生大量的bounding boxes, 使用NMS算法只留下top200 (这一步SSD300在VOC20类的每张图像上, 需耗时2.2msec)

<div style="width: 550px; margin: auto">![](https://wx1.sinaimg.cn/mw1024/d7b90c85ly1fwbl46szs0j213m0ewag3.jpg)

上图看起来SSD300比YOLO还要快, 但实际上YOLO的网络层数是24层, 而SSD的层数是12层, 这样比起来有点不太公平( 但是层数多居然精度没SSD高, 这也值得吐槽, 但是个人觉得这是因为YOLOv1的设计比较粗糙, 很多trick没有使用导致的, 所以看看YOLOv2, 和YOLOv3的版本, 结果还是挺不错的)

# SSD 中如何计算 default box 的大小

假如feature maps数量为 $m$, 那么每一个feature map中的default box的尺寸大小计算如下:

$$s_k = s_{min} + \frac{s_{max} - s_{min}}{m-1}(k-1), k\in [1,m]$$

上式中, $s_{min} = 0.2 , s_{max} = 0.9$. 对于原文中的设置 $m=6 (4, 6, 6, 6, 4, 4)$, 因此就有 $s = \{0.2, 0.34, 0.48, 0.62, 0.76, 0.9\}$
然后, 几个不同的aspect ratio, 用 $a_r$ 表示: $a_r = {1,2,3,1/2,1/3}$, 则每一个default boxes 的width 和height就可以得到( $w_k^a h_k^a=a_r$ ):

$$w_k^a = s_k \sqrt{a_r}$$

$$h_k^a = \frac{s_k}{\sqrt {a_r}}$$

对于宽高比为1的 default box, 我们额外添加了一个 scale 为 $s_k' = \sqrt{s_k s_{k+1}}$ 的 box, 因此 feature map 上的每一个像素点都对应着6个 default boxes (**per feature map localtion**).

# SSD 使用了哪些数据增广方法?

水平翻转, 随机裁剪+颜色扭曲(random crop & color distortion), 随机采集区域块(randomly sample a patch, 目标是为了获取小目标训练样本)

# 为什么SSD不直接使用浅层的特征图谱, 而非要额外增加卷积层, 这样不是增加模型的复杂度了吗?

FPN: 理想情况下, SSD 的特征金字塔是从多个卷积层输出的特征图谱得到的, 因此它的计算成本几乎为零. 但是为了避免使用到那些表征能力不强的低阶特征图谱(浅层), SSD 只使用了深层的特征图谱(conv4_3), 同时在 backbone 网络的后面又添加了几层卷积层来提取高表征能力的特征图谱. 但是这样就是的 SSD 错过了那些低阶特征的信息, 这些低阶特征中往往包含了高阶特征不具有的信息, 如小物体的特征信息, 这也是为什么 SSD 对小物体不敏感的原因之一.
