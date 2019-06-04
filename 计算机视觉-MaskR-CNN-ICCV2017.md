---
title: Mask R-CNN (ICCV, 2017)
sitemap: true
date: 2018-04-09 19:27:03
categories: 计算机视觉
tags:
- 计算机视觉
- 目标检测
- 实例分割
- 论文解读
---
**文章:** MaskRCNN
**作者:** Kaiming He, Georgia Gkioxari, Piotr Dollar, Ross Girshick
**备注:** FAIR, ICCV best paper

# 核心亮点

**1) 提出了一个简单,灵活,通用的实例分割模型框架**
MaskRCNN 在 FasterRCNN 的基础上进行改进, 在模型的head部分引入了一个新的mask预测分支, 在训练阶段, 该分支会与其他分支并行执行, 在测试阶段, 虽然不是并行执行, 但是利用 NMS 减少了需要计算的候选框个数, 因此 MaskRCNN 模型整体增加的额外开销较小.

**2) 提出了RoI Align来解决 RoI 与 pooling 后的特征图谱之间的不对齐问题**
Fast/FasterRCNN 原始的 RoIPool 操作在进行池化时, 会进行两次粗糙的量化操作, 这使得池化后的特征图谱与 RoI 中的信息不能很好的对齐, 对于像素级任务实例分割来说, 这种非对齐会使得模型性能大大降低, 因此 MaskRCNN 提出用基于双线性插值法的 RoI Align 代替 RoI Pool, 以此来解决非对齐问题.

![图1](https://wx2.sinaimg.cn/large/d7b90c85ly1fx1toyw0xkj20kc0a5wkw.jpg)

# 摘要
本文提出了一个简单, 灵活, 通用的目标实例分割框架. 本文的方法可以有效的检测图片中的物体, 同时为每个实例生成一个高质量的掩膜, 我们将其称为 Mask RCNN, 它是从 Faster RCNN扩展而来的, 添加了一个新的分支来并行预测物体掩膜. MaskRCNN可以轻易的进行训练, 并且相对于FasterRCNN只增加了很少的开销, 可以在5fps下运行. 不仅如此, MaskRCNN可以轻易的泛化到其他任务中, 如人体姿态识别. 本文的模型在COCO的三个系列任务都, 都取得了最好的效果.

# 背景介绍

本文意在提出一种通用的实例分割模型框架-MarkRCNN, 该模型扩展自FasterRCNN, 在FasterRCNN模型中的每一个RoI上, 添加一个与检测分支平行运行的掩膜预测分支, 如图1所示. 掩膜分支(mask branch) 是一个小型的FCN网络, 它应用在每一个RoI上, 以pixel-to-pixel的方式来预测一个分割掩膜. Mask RCNN易于实现, 且增加的额外开销很小, 并且具有很大的灵活性, 是一个通用的实例分割模型框架. 在FasterRCNN中使用的RoI pooling是一种针对目标检测任务的粗糙的pooling方法, 会造成一定程度上的不对齐结果, 为了克服这一点, 本文提出了RoIAlign, 用于保留准确的空间位置, RoIAlign可以将掩膜的精确度才提高10%~50%. 另外, 本文发现, 将二值掩膜预测和类别预测任务分开独立进行是非常重要的一步, 因此, 我们为每一个类别都会单独进行mask预测, 避免了不同类别之间的冲突. 这一点与FCN不同, FCN会对一个像素点进行多类别的分类.
我们的模型在GPU上的运行速度大约为200ms/frame, 在8-GPU的单机上训练时, 需要1到2天的时间.

# Faster RCNN
简单回顾一下Faster RCNN, 它包含两个阶段, 第一个阶段, 是RPN结构, 用于生成候选框集合. 第二个阶段, 本质上就是一个Fast RCNN, 利用RoI pooling从RoI中提出固定尺寸的特征, 然后进行分类任务和边框回归任务. 这两个阶段使用的特征图谱是共享的, 都来自backbone网络.

# Mask RCNN
Mask RCNN在概念上来说非常简单: FasterRCNN对于每个候选框来说都有两个输出分支, 一个class label和一个bounding-box offset, 对此在MaskRCNN中我们添加了第三个分支用于输出掩膜. 虽然这是一个看起来很自然的想法, 但是额外增加的掩膜分支和class, box分支并不相同, 它需要物体更加精确的空间位置. 因此, 我们还引入了一个在MaskRCNN中非常关键的元素:RoIAlign, 用于进行像素对其. 来弥补FasterRCNN RoI pooling的粗糙映射造成的位置偏移问题.
Mask R-CNN 使用了与 Faster R-CNN相同的two-stage结构, 第一阶段使用了相同的RPN网络, 第二阶段, 在执行class分类和box回归任务的 **同时(并行)**, MaskRCNN会为每一个RoI生成一个二值掩膜. **这一点与许多现有系统不同, 这些系统都是在mask预测结果的基础上进行分类任务的.** 我们的灵感来自于Fast RCNN中class任务和box任务的并行执行(这种并行执行方式很大程度上简化了原始的 R-CNN 的 multi-stage pipeline).
Formally, 在训练阶段, 我们在每一个采样的RoI上定义一个multi-task loss如: $L = L_{cls}+L_{box}+L_{mask}$. 前两个loss和Fast RCNN相同, 第三个分支对于每一个RoI的输出维度为 $Km^2$, 代表这分辨率 $m\times m$ 下的 $K$ 个二值掩膜, 每一个掩膜对应了一个类别(共 $K$ 个类别). 为此, 我们使用了 **per-pixed sigmoid**, 并且将 $L_{mask}$ 定义为平均二值交叉熵(average binary cross-entropy loss). 对于一个与真实类别 $k$ 相关联的RoI, $L_{mask}$ 只在第 $k$ 个mask上有定义(其他mask不计入loss).
我们对 $L_{mask}$ 的定义使得网络模型可以为每个class生成mask, 避免了不同类别之间的竞争冲突. 并且利用分类分支的结果标签来选择对应的mask进行计算. 这样做可以使得mask预测任务和class预测任务 **decouple**, 这与许多现有应用FCNs进行实例分割任务的模型不同, 这些模型通常会使用一个 per-pixel **softmax** 和一个 **multinomial** cross-entropy loss(多分类), 在这种模型中, 掩膜通常会引起类别之间的竞争. 而在本文的模型中, 使用的是 per-pixel **sigmoid** 和一个 **binary** loss (二分类). 我们将会通过实验来展示这将对于呈现出好的实例分割结果来说十分关键!

# Mask Representation
一个mask代表了一个物体的空间布局(spatial layout). 因此, 和 class labels 或者 box offsets 不可避免的要通过 FC layer 降到较小维度输出不同, 提取mask的空间结构时可以很自然的通过像素到像素(卷积层正好提供了这种相关关系)的方式解决.
具体来说, 我们使用FCN从每个RoI中预测出一个 $m\times m$ 的mask. 这使得mask分支中的每一层都维护着 $m\times m$ 的物体空间布局矩阵, 而不用将其转换成低纬度的向量形式(缺少空间信息). 和之前的工作(依赖于fc层预测mask)不同的是, 本文的全卷积表征需要的参数更少, 同时通过实验证明, 更加精确. 这种 pixel to pixel 的机制需要我们的RoI feature (通常都是些很小的特征图谱)能够较好与真实物体对齐, 保持明确的逐像素间的对应关系. 为此, 本文提出了 **RoIAlign** 来代替 **RoiPool**.(这个替换很重要)

# RoIAlign
RoIPool可以从每个RoI中提取到一个较小的固定的feature map(如, 7×7). RoIPool首先会将浮点型的 RoI (因为 RoI 是模型预测出来的 bounding box 的坐标, 不是严格个 feature map 的整数像素点对应的) **离散量化** 到 feature map 的像素粒度上, 然后量化后的 RoI (整数) 又会细分到 RoIPool 的各个 bin 中去(这又是一次量化), 最终的 feature values 是通过每个 bin 整合信息得到的(通常为 max pooling). 例如, 首先, 第一次量化会在 (x / 16) 上以四舍五入的方式(rounding)进行, 这里, 16 是特征图谱相对于原图片的步长, 然后, 同样的, 量化又会在划分到 bins 中时被执行(因为不可能总是刚好整数划分). 这些量化操作(两次)会引入 RoI 和提取到的特征之间的 misalignments, 这种misalignments对于分类任务来说或许影响不大, 但是对于predicting pixel-accurate mask任务来说就会造成很大的负面影响.

为了解决这个问题, 我们提出了RoIAlign层, 移除了RoIPool粗糙的量化计算, 将提取到的的 features 与输入的RoI对齐. RoIAlign的原理很简单: 避免在 RoI 边界上或者 bins 中执行任何量化计算(即, 我们使用 $x/16$, 而不是 $[x/16]$). 我们利用双线性插值法来计算每个 RoI bin 中各个位置上具体的像素值, 并且对计算结果整合(max或者average). 具体计算细节如图3所示. 我们注意到, 在不使用任何量化计算以后, 计算结果对于具体的采样点位置和采样数量的多少都不再那么敏感了.

在4.2节中可以看到 RoIAlign 会带来很大的提升. 同时我们还与 RoIWarp 操作进行了比较. 和 RoIAlign 不同, RoIWarp 忽视了 alignment 问题, 并且同样采用了想 RoIPool 一样的量化 RoI 的操作. 因此, 即使 RoIWarp 也采用了二线性插值法的 resampling 操作, 它的表现在实验中也没有取得很大的提升, 这也正说明了 alignment 的关键性作用.

RoI Pooling存在两次量化过程:
- 将候选框边界量化为整数坐标值
- 将量化后的边界区域分割成 $k\times k$ 个bins, 并对每一个单元的边界量化

可以看出, 上面的量化操作是一种很粗糙的Pooling方式, 由于feature map可以看做是对原图特征的高度概括信息, 所以feature map上的细微差别映射回原图时, 往往会导致产生很大的像素位移差. 故此, 提出了RoI Align的解决思路: 取消量化操作, 使用双线性内插的方法获得坐标为浮点数的像素点上的图像数值, 从而将整个特征聚集过程转换为一个连续的操作. 其具体流程如下:
- 遍历每一个候选区域, 保持浮点数边界不做量化
- 将候选区域分割成 $k\times k$ 个bins, 每个bins的边界也不做量化
- 在每个bins中计算固定四个采样点的位置, 使用双线性插值的方法计算出这四个位置的值, 然后进行最大化操作. 如下图所示.

![](https://wx4.sinaimg.cn/large/d7b90c85ly1fx1w5spug0j20s10am4az.jpg)

# 网络结构(Network Architecture)
为了说明本文方法的通用性, 我们利用多个模型结构初始化Mask RCNN. For clarity, 我们做出以下区分: 1) 用于提取整个图片特征的卷积结构称为 backbone, 2) 用于执行bounding-box recognition(cls,reg)任务和 mask prediction 任务的部分称为 network head, 它会被分别应用在每一个 RoI 上面.
我们将基于网络深度特性的命名方法来表示 backbone network(主干网络). 我们测试了 50和101层的 ResNet, ResNeXt 网络. 原始的 FasterRCNN 在使用ReNets时, 利用的是第4段卷积块的最后一层卷积层的输出, 我们称为 C4. 这个backbone, 我们将其记为 ResNet-50-C4 , 它是很多模型中的一个常用的选择.
我们同时还使用了另一个更有效backbone: Feature Pyramid Network(FPN). FPN 使用了一种 top-down 的横向连接结构, 使得可以从单一的图片输入建立起网络内部的特征金字塔结构(in-network feature pyramid). 用 FPN 当做 backbone 的 Faster R-CNN 根据 RoI 的尺寸从特征金字塔的不同层级中提取 RoIs features, 但是其他的流程基本和普通 ResNet 相同. 使用 ResNet-FPN 作为 backbone 用于特征提取的 Mask R-CNN 可以获得极大的性能提升(both acc and speed).
对于 network head, 我们几乎是按照之前的 Faster R-CNN 的框架设计的, 我们在其基础上添加了一个全卷积的mask prediction 分支. Specifically, 我们对现有的 ResNet 和 FPN 的论文中 Faster R-CNN 的 box heads 进行扩展. 具体细节如图4所示. ResNet-C4 的 head 包含 ResNet 的第5个卷积块(即 9层的res5, 计算密集型). 对于FPN来说, 它的backbone就已经包含了res5, 因此可以允许有 filters 更少的有效头部(efficient head).
我们注意到本文的mask分支具有一个很简单的结构. 其他更加复杂设计也许可以更进一步的提升性能, 这将会在以后的工作的进一步讨论, 并非本文的焦点.

![](https://wx2.sinaimg.cn/large/d7b90c85ly1fx1wztcgzhj20yf0tob29.jpg)

# 实现细节(Implementation Details)

我们使用Faster RCNN原文提供的超参数.(无需调参也说明了MaskRCNN的鲁棒性很高)

**Training:**
和FastRCNN一样, 如果RoI与真实框的IOU大于0.5, 就会被认为是正样本, 否则认为是负样本(这里与FasterRCNN不同). $L_{mask}$ 只会计算正样本上的掩膜损失. mask target 是 RoI 和 真实mask之间的交集(注意不是直接根据真实mask计算损失). 我们使用 image-cengric (FastRCNN), 图片会被resized成scale(shorter edge) 为800 像素. minibatch为 2 img/GPU, 每个img含有N个sampled RoIs. 正负样本比例为 **1:3**. 对于 C4 backbone来说, N为 64, 对于FPN来说, N为512.(因为FPN效率更高, 所以可以在一张图片上计算更多的RoIs). 在8PGUs上训练(即有效minibatch为16). lr为0.02, 每120k 迭代(iteration)会缩小10倍. weight decay为0.0001, momentum为0.9. 在使用ResNeXt时, one img/GPU, lr初始为0.02, 其他相同.
RPN的anchor具有 5 scales 和 3 aspect ratios. 为了方便进行消融实验, RPN是被单独训练的, 并且没有与Mask RCNN共享了卷积特征(可共享, 只是为了方便没有共享). 对于本文中的每一项, RPN和MaskRCNN都具有相同的backbones.

**Inference:**
在预测阶段, proposal的数量为 300 for C4 backbone, 1000 for FPN. 我们会在这些 proposals 上面进行 box prediction branch, 然后使用NMS选择了最高score的100个boxes, 并对这100个 boxes 应用 mask branch, 虽然这里和训练时采用的并行计算不同, 但是它仍然可以加速预测速度, 同时能够提高精度(因为只使用了更少但是更精确的100个box). mask branch 可以对每个RoI预测出 $K$ 个masks, 但是我们只会使用第 $k$ 个mask, 这里的小写 $k$ 代表着物体的预测分支上确定的类别(注意不一定是真实类别). 我们会将 $m\times m$ 的浮点类型的mask放缩到RoI的尺寸大小, 然后依据一个阈值(0.5)对mask的像素值进行二值化操作. 注意到, 由于我们只对top 100 的 score box执行mask branch , 因此在模型预测时, MaskRCNN相比于FasterRCNN, 只增加了很小的开销.(增加了20%)

# 实验: 实例分割(Instance Segmentation)

使用了 COCO 数据集, 采用 AP(averaged over IoU thresholds) 评价标准, $\text{AP}_{50}$, $\text{AP}_{75}$, $\text{AP}_{S}$, $\text{AP}_{M}$, $\text{AP}_{L}$.

如表1所示, MarkRCNN 的性能超过 COCO2015 和 COCO2016的实例分割冠军 MNC 和 FCIS.(并且 MaskRCNN 没有使用 multi-scale train/test, horizontal flip test, OHEM 等 trick, 言外之意 MaskRCNN 的性能可以进一步利用这些 trick 提高)

![](https://wx3.sinaimg.cn/large/d7b90c85ly1fx2wm74k7ij21kw0d3njk.jpg)

Table2 显示了对 MaskRCNN 的消融实验分析结果.

![](https://wx1.sinaimg.cn/large/d7b90c85ly1fx2wsrjbd5j21kw0mwqv5.jpg)

表3 显示了 MaskRCNN 与当前的 state of art 的目标检测方法在 COCO 数据集上的表现.

![表3](https://wx1.sinaimg.cn/large/d7b90c85ly1g0wvcbvcjjj214i0b578q.jpg)

**MaskRCNN for Human Pose Estimation:**

![](https://wx1.sinaimg.cn/large/d7b90c85ly1fx2yy9ki30j20sw0ht7pe.jpg)

![](https://wx3.sinaimg.cn/large/d7b90c85ly1fx2yydz8o1j20sb0dgqgp.jpg)

![](https://wx1.sinaimg.cn/large/d7b90c85ly1fx2yyhmr1nj20ru07g3zz.jpg)

# 附录A: Experiments on Cityscapes

![](https://wx1.sinaimg.cn/large/d7b90c85ly1fx2yz18aouj21i30bbdzk.jpg)

# 附录B: Enhanced Results on COCO

下表显示了 MarkRCNN 被各种 trick 增益后的性能表现

![](https://wx2.sinaimg.cn/large/d7b90c85ly1fx2zdx07m9j20s80dq19y.jpg)

![](https://wx2.sinaimg.cn/large/d7b90c85ly1fx314i2opdj20s80dq19y.jpg)
