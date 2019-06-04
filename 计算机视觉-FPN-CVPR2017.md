---
title: FPN (CVPR, 2017)
sitemap: true
categories: 计算机视觉
date: 2018-10-18 14:36:46
tags:
- 计算机视觉
- 目标检测
- 论文解读
---

**文章:** Feature Pyramid Networks for Object Detectin
**作者:** Tsung-Yi Lin, Piotr Dollar, Ross Girshick, Kaiming He, Bharath Hariharan, and Serge Belongie

# 核心亮点

**提出了多尺度的特征金字塔结构**
将最后一层特征图谱进行不断进行上采样, 并与每一个金字塔阶级的特征图谱进行加法合并操作, 得到新的表征能力更强的不同金字塔层次的特征图谱, 然后将RoI按照尺寸分别映射到这些特征图谱上, 再在每个特征图谱上进行类别和位置预测. 可以直观感受到, 这种多尺度的特征图谱在面对不同尺寸的物体时, 具有更好的鲁棒性, 尤其是在面对小型物体时. 同时, 这种特征金字塔结构是一种通用的特征提取结构, 可以应用到不同的网络框架中, 显著提高(5~8%)模型的召回率(因为提出了更多不同尺度, 不同特征信息的anchor box), 并且可以广泛提高(2~3%)模型的mAP.


# 背景介绍

在面对不同尺度的物体检测问题时, 特征金字塔结构是一个非常基本的组成部分, 但是最近的检测模型都舍弃了这一结构(Fast RCNN, Faster RCNN, YOLOv2等), 其一部分原因是因为这个结构对计算和内存的要求较高. 本文在控制资源消耗的情况下, 建立了一个跨所有层的特征金字塔结构, 我们将其称为 Feature Pyramid Network(FPN), 将 FPN 应用在基本的 Faster R-CNN 网络中, 取得了当下的STOA性能.

传统的特征金字塔结构对于计算资源和内存资源的依赖较为严重, 同时深度卷积网络在不同阶段的卷积层, 虽然较好的传递了特征, 但是因为每一层的输出通道数不同, 会导致层与层之间形成一种潜在的语义鸿沟. 较高的分辨率往往具有更多的低级信息(在深层会被过滤掉), 但是多余的信息也会对降低泛化能力, 较低的分辨率则具有权重更高的重要信息, 但是这样也会使得小目标物体难以检测.

# SSD与FPN中多尺度特征图谱融合的区别
SSD算是首批结合多尺度特征金字塔的检测系统, 但是SSD为了避免用到过多的低级特征(高层卷积图谱上的特征), 放弃使用以及计算好的特征特普, 而是从网络的最后一层卷积层开始, 添加新的卷积层, 并在这些新添加的卷积层上进行特征金字塔融合. 这样做一个很直观的结果就是, 它会错过很多高分辨率特征图谱上的特征信息, 而这些特征信息在面对小物体检测时是十分有用的.(这也是SSD对小物体检测较为敏感的原因之一).

# 介绍

![图1](https://wx1.sinaimg.cn/large/d7b90c85ly1fy9h88eypoj21xi0sx1kx.jpg)

在不同尺寸(different scales)上对物体进行识别是计算机视觉领域中一个具有挑战性的基本问题(fundamental challenge). 在图片金字塔(image pyramids)上建立特征金字塔(feature pyramids)的方法(我们称之为 featurized image pyramids)是一个标准的解决方案, 如图1(a)所示. 这些金字塔结构具有尺寸不变性(scale-invariant), 因为当物体的scale发生变化时, 我们可以通过在pyramid的levels之间移动(shifting)来适应. Intuitively, 这个特性可以通过让模型同时在位置和pyramid levels上进行扫描(scanning), 从而可以在很大的尺度范围内检测物体.
这种方法(featurized image pyramids)在人工设计图片特征的时代十分受用. 而随着深度卷积网络的兴起, 我们可以从一张单一的特征图谱上获取到大量的抽象特征, 抽了提取特征之外, ConvNets在尺寸变化上也表现出了很强的鲁棒性, 这使得我们可以利用一张图片就能获得足够的信息来做出预测, 如图(b)所示. 但是即使拥有这种鲁棒性, 我们仍然需要 pyramid 结构来获取更加准确的结果. 几乎所有的模型在 ImageNet 和 COCO 的目标检测比赛上都使用了在 featurized image pyramids 上进行 multi-scale testing 的方法. 在不同 level 的 image pyramid 上进行预测的一个主要好处是, 它可以产生一个多尺度的特征表示(multi-scale), 其中所有层在语义上都是强的, 包括高分辨率层.
但是, 在 image pyramids 的每一层上提取特征具有很明显的缺点, 那就是会使得 Inference time 显著提升. 更进一步的, 会使得在 image pyramid 上训练端到端的深层网络变的不可行, 也因此, 我们仅仅在测试阶段才会使用 image pyramids, 这会在训练和测试阶段之间产生不一致问题. **基于以上原因, Fast 和 Faster R-CNN 选择在默认设置下不使用 featurized image pyramid(尽管它有助于提高精度)s**
然而, image pyramids 并不是唯一的计算多尺度特征表示(multi-scale feature representation)的方法. 一个深层的 ConvNet 会一层层的计算出一个特征层级(feature hierarchy), 同时配合 subsampling 层, 使得 feature hierarchy 具有了固有的多尺度金字塔结构(inherent multi-scale, pyramidal shape). SSD 是首个在这些特征图谱上使用特征金字塔的模型, 它看起来就像是 featurized image pyramid(实际上不是), 如图(c)所示. 理想情况下, SSD 的特征金字塔是从多个卷积层输出的特征图谱得到的, 因此它的计算成本几乎为零. 但是为了避免使用到那些表征能力不强的低阶特征图谱(浅层), SSD 只使用了深层的特征图谱(conv4_3), 同时在 backbone 网络的后面又添加了几层卷积层来提取高表征能力的特征图谱. 但是这样就是的 SSD 错过了那些低阶特征的信息(高分辨率的特征图谱), 这些低阶特征中往往包含了高阶特征(低分辨率特征图谱)不具有的信息, 如小物体的特征信息, 这也是为什么 SSD 对小物体不敏感的原因之一.
本文的目标时可以自然的利用 ConvNet's feature hierarchy 的金字塔结构的同时创建一个在所有尺度(all scales)上都具有强语义信息的特征金字塔. 为了达到这个目标, 我们通过 top-down pathway(top指的就是深层的小卷积图谱) 和 lateral connections 建立了一个可以结合低阶特征(high-resolution, semantically weak features)和高阶特征(low-resolution, semantically strong features)的特征金字塔结构, 如图1(d)所示. 由图易知, 这个特征金字塔在所有尺度的特征图谱上都具有很强的语义特征, 同时可以从一张单一尺寸的图片出发, 利用很低的计算成本建立起来. 换句话说就是, 我们可以在不牺牲模型特征表达能力, 速度, 以及内存消耗的情况下, 创建一个网络模型内部的特征金字塔结构, 从而可以替换掉 featurized image pyramids.
和本文的特征金字塔比较相似的结构是利用 top-down 和 skip-connection 建立的网络, 如图2(top)所示, 该网络的主要目前是产生一个 **单一的** 具有很强表征能力和适度大小(fine resolution)的特征图谱, 然后在这张特征图谱上进行预测. 与此相反的是, 我们的方法是利用特征图谱的结构, 将其作为一个特征金字塔, 然后在金字塔的每一层中都进行独立的预测(predictions are independently made on each level). 如图2(down)所示.

![图2](https://wx3.sinaimg.cn/large/d7b90c85ly1fy9gwgg1dhj21k50ltgzy.jpg)

通过对比实验和消融实验, 我们发现 FPN 可以大幅的题目 bounding box proposals 的召回率, 也可以提升检测和分割任务的性能, 并且很容易应用到现有的模型当中.
除此以外, 我们的 pyramid 结构可以被端到端(with all scales)的训练, 并且在 train/test 阶段可以保证一致性.

# 相关工作

**Hand-engineered features and early neural networks:** SIFT, HOG, shallow networks

**Deep ConvNet object detectors:** OverFeat, R-CNN, SPPnet, Fast R-CNN, Faster R-CNN

**Methods using multiple layers:** FCN, HyperNet, ParseNet, ION, SSD, MS-CNN

# Feature Pyramid Networks

我们的目标是利用 ConvNet 的金字塔式的特征层级, 这种特征层级拥有从低到高的不同级别的语义信息(浅层的语义信息少, 深层的多), 同时建立一个具有具有整体高级语义信息的特征金字塔. 本文我们主要关注 FPN 的建立, 以及在 RPN(sliding window proposers) 中和 Fast R-CNN 中的使用, 同时会在第6节给出在实例分割proposals上的扩展.

输入: 任意尺寸的单张图片(不进行尺度缩放)
输出: 以全卷积的方式, 输出多层次的按比例大小的对应特征图谱映射
上面的过程独立于骨干卷积体系结构(backbone convolution architectures), 本文的网络我们使用 ResNets. 构建 FPN 的步骤依次为 bottom-up pathway, top-down pathway 和 lateral connections, 介绍如下:

**自底向上的路径(bottom-up pathway):**
该步骤是根据 backbone 卷积网络的前馈计算过程进行的. 在前向计算中, 卷积网络会以两倍的缩放系数计算出不同尺寸的 feature maps, 最终形成我们所说的 feature hierarchy. 通常情况下, 会有很多的网络层输出的特征图谱具有相同的尺寸, 我们称这些网络层都处于同一个网络阶段(same network stage). 对于我们的特征金字塔来说, 我们在每一个"network stage"上都定义一个金字塔级别. 然后选择每个阶段的最后一层作为特征图的参考集合(因为每一个 stage 的最深层理应具有最强的特征表示), 我们会丰富(enrich)这个特征图谱来创建金字塔.
具体来说, 对于 ResNets, 我们使用了每一个阶段的最后一个残差结构的激活层输出的特征, 将这些残差模块conv2, conv3, conv4, conv5 的输出表示为 $\{C_2, C_3, C_4, C_5\}$, 并且注意到他们相对于输入图像具有 $\{4,8,16,32\}$ 像素的步长(原始图片与特征图谱宽或高的比例). 由于 conv1 的特征图谱占用内存较多, 因此我们没有将它包括在金字塔中.

**自顶向下的路径以及横向连接(Top-down pathway and lateral connections):**
自顶向下的路径是通过在较粗糙, 但是语义信息较强的高层特征图(深层)上进行上采样来产生(hallucinates)更高分辨率的图谱. 然后将这些上采样之后的 features 与自底向上(自浅而深)的 features 通过横向连接(lateral connections)的方式拼接在一起.(横向连接的feature map的size是一样大的). 每一次横向连接都会将两个 pathway 上的具有相同大小的 feature maps 融合在一起.
那些 bottom-up frature maps 具有较为低级的语义信息(低级是指抽象程度低), 但是这些图谱的特征激活信息(重要特征才会被激活)的位置精度更高, 因为它们经过的下采样次数更少.
下面的图3显示了构建 top-down feature map 的模块. 对于一个分辨率较粗糙的特征图谱, 首先将其上采样至2倍(为了简单起见, 直接使用最近邻上采样法), 然后将上采样后的特征图谱与对应的自底向上的图谱进行按元素相加合并(element-wise addition, 由于二者通道数不同, 因此合并前自底向上的图谱会将经过1×1卷积降低通道数). 这个过程会一直迭代进行, 直到最浅的卷积图谱也被合并为止. 为了开始迭代, 在最开始的时候, 我们直接用 $1\tiems 1$ 的卷积层作用在 $C_5% 上, 来产生分辨率最粗糙的特征图谱. 最后, 我们会用3×3的卷积层作用在每一个合并后的特征图谱上, 以此来得到最终的特征图谱, 这是为了消除上采样的混叠效应(aliasing effect of unsampling??). 我们将这些最后得到的特征图谱记为 $\{P_2, P_3, P_4, P_5\}$, 他们与$\{C_2, C_3, C_4, C_5\}$ 相对应并且具有相同的大小.

![图3](https://wx4.sinaimg.cn/large/d7b90c85ly1fy9guhouhvj217x0hrn4b.jpg)

由于金字塔的每一层都使用了共享的分类器和回归器(就像传统的 featurized image pyramid 一样), 因此我们固定了每一层特征图谱的深度 $d$(numbers of channels), 本文中, 我们令 $d=256$, 也就是说所有额外添加的卷积层的输出通道数都是 256. **在这些额外添加的卷积层中, 我们没有使用非线性结构, 我们通过实验发现, 这会对最终的结果有轻微的影响**.
Simplicity 是本文设计的核心, 同时我们也发现我们的模型对于很多设计选择都具有很强的鲁棒性. 我们实验了更多复杂的模块(eg using multilayer residual blocks as connections)并且也发现了略微更好的结果. 同时也存在其他更好的连接设计, 但本文的主要目的是探讨FPN的有效性, 因此没要尝试过多的连接组合.

# 应用

FPN是一种用于在卷积网络内部建立特征金字塔的一般化的解决方. 在下文中, 我们在 RPN 中使用我们的方法来进行 bounding box proposal generation, 同时在 Fast R-CNN 中使用 FPN 来进行 object detion. 为了证明我们方法的简单性和有效性, 我们仅仅对原始的系统做很小的改变.

## Feature Pyramid Networks for RPN

RPN 是一个滑动窗口式的物体检测器(类别不可知, class-agnostic). 在原始的 RPN 设计中, 一个小的子网络会在密集的 $3\times 3$ 大小的滑动窗口上进行 evaluate, 这是在一个 **单一尺寸的卷积特征图谱** 进行的二分类和边框回归操作. 这个过程是通过一个 $3\times 3$ 的卷积层, 后接两个并行的 $1\times 1$ 的卷积层, 这两个卷积层分别进行分类和回归任务, 我们将其称为 **network head**(这三个网络层统称为 network head).
我们通过将原本的 **单一尺寸的特征图谱** 替换成 **FPN 的金字塔特征图谱** 来在 FPN 中应用 RPN. 我们在特征金字塔的每一层都添加与 RPN 相同的 head 结构( $3\times 3$ conv and two sibling $1\times 1$ convs). 由于 head 会密集地在金字塔中的所有层级的所有 locations 上滑动(sliding), 因此就 **没有必要在每一个层级上使用多尺度的 anchors 了**. 相对应的, 我们会给每一个 level 赋予单一的尺寸. 具体来说, 对于特征图谱 $\{P_2, P_3, P_4, P_5, P_6\}$ (这里的 $P_6$ 仅仅为了覆盖更大的尺寸 $512^2$, 它是通过在 $P_5$ 上进行 stride 为 2 的降采样得到的, 在 Fast R-CNN 中并没有使用 $P_6$) 来说, 其每一个特征图谱对应的 anchors 的大小分别为 ${32^2, 64^2, 128^2, 256^2, 512^2}$, 每一层anchors的宽高比例为{1:2, 1:1, 2:1}, 因此, 总共具有15个 anchors (对于每一个 location 而言有15个 anchors, 对比 Faster R-CNN, 每一个 location 具有默认具有 3×3=9 个 anchors)
训练时的标签赋值策略和FasterRCNN是一样的, 都会根据 IoU 的大小为每一个 anchors 贴上正负样本的浅标签, 或者不贴.
注意, heads 的参数在所有的特征金字塔层级中都是共享的, 我也测试了不共享的情况, 结果显示二者差不多. 共享参数的良好性能表现说明了我们的特征金字塔在所有的层级上都共享着差不多的语义信息. 这个特性带来的好处和使用 featurized image pyamid 差不多, 因为我们可以使用一个通用的 head classifier 在任意图片尺寸中进行预测.
按照上述的策略, 我们可以很自然的在 RPN 中使用 FPN 网络, 更具体的细节会在实验部分给出.

## Feature Pyramid Networks for Fast RCNN

FastRCNN 通常只作用在单一尺寸的特征图谱上, 将FPN用于FastRCNN时, 我们需要在不同的金字塔层次上赋予不同尺度的 RoI 大小(因为 RoI pooling 是根据特征图谱和原图的尺寸关系决定的).
我们将特征金字塔看做是从图片金字塔产生的特征图谱, 因此我们可以采用和 Fast R-CNN 相同的分配策略来完成 RoI pooling, 具体来说, 我们将宽度为 $w$ ,高度为 $h$ 的 RoI (这里的宽和高是针对原始图片而言的)通过如下公式分配到特征金字塔的 $P_k$ 等级上:

$$ k = \lfloor k_0 + log_2(\frac{\sqrt{wh}}{224})$$

这里 224 是标准的ImageNet预训练的大小, 而 $K_0$ 是则大小为 $w\times h = 224^2$ 的RoI应该映射到的目标级别. 类似于基于 ResNet 的 Faster RCNN 使用 $C_4$ 作为单尺度特征映射, 我们将 $k_0$ 设置为4 (也就是说, 与图片一样大的RoI会映射到 $P_4$ 的特征图谱上). 上式表明, 如果RoI的尺度变的更小(如224的0.5倍), 那么该RoI就应该映射到分辨率更高的金字塔图谱上(如 $k=3$ ).(也就是说不同大小的RoI会映射到不同金字塔层级的特征图谱上, 总的来说, 越小的RoI, 会映射到更浅层的特征图谱上, 因为太深的图谱可能已经将小物体信息过滤掉了)

文章将预测器(分类和坐标回归)应用到所有金字塔层级的RoI上面. 需要注意, **预测器(heads)在所有层级上的权重都是共享的.** . 在ResNet中, 会在conv4的特征图谱上在加上一个conv5, 但是本章已经将conv5用于构建特征金字塔. 所以和ResNet不同, 文章很直接的利用RoI pooling来获取 $7\times 7$ 的特征 (注意不是基于滑动窗口的检测器, 这一点和YOLO差不多), 并且会使用2层1024维的 fc 隐藏层(后接 ReLU), 然后才会送入最终的预测器层当中(分类和位置回归). **注意到, 和标准的 conv5 head 相比, 我们的 2-fc MLP head 具有更少的权重参数, 运行速度也更快**

# 实验

用COCO 80ktrain和35k val进行实验. 所有的网络均在ImageNet1k上预训练.

## Region Proposal with RPN

![表1](https://wx2.sinaimg.cn/large/d7b90c85ly1fy9gxelwczj21m80f2n1v.jpg)

**实验细节:**
表1中的所有模型都是以端到端的方式训练的, 输入图片的尺寸被resize, 其最短边长为800像素. 8块GPU同步训练, 每个GPU的minibatch为两张图片, 每张图片的anchors为256. weight decay为0.0001, momentum为0.9. learning rate 开始的30k图片是0.02, 之后是0.002. 训练时包含了那些处于image之外的anchor boxes(Faster选择忽略).

### 消融实验

**Comparisons with baselines:** 为了公平比较, 我们使用了两种 baseline 方法, 一个采用 C4 特征图谱, 一个采用 C5 特征图谱, 并且他们的 scales 都是 $\{ 32^2, 64^2, 128^2, 256^2, 512^2\}$, 表1(b)相对于表1(a)并没有提高, 说明单一的更高层级的特征图谱不足以输出更好的结果, 因此当层级更高时, 虽然表征能力更强, 但同时特征图谱也变的更加粗糙. 而使用 FPN 则可以大大提高目标的准确率和召回率(AR), 如表1(c)所示. 尤其是在面对小物体和中等物体等多尺度物体时, 会显著提高AR指标.

**top-down** 表1(d)显示了不带 top-down pathway 的 FPN 实验结果, 与表1(b)相比, 可以看出, 带有 top-down pathway 的特征图谱加强可以使得特征图谱具有很强的语义特征信息和更好的分辨率.(原始的特征图谱之间的语义鸿沟更大, 层与层之间的联系比较简单粗糙)

**lateral connections:** 虽然top-down方式的特征图谱具有很强的语义特征信息和更好的分辨率效果, 但是由于经过不断的降采样和上采样过程, 该特征图谱的位置信息可能会变得不够精确. **lateral connections** 同时结合具有精确位置信息的特征图谱和具有强语义信息的图谱, 进而达到更好的效果. 当没有 **lateral connections** 时(也就是不使用 down-top 的特征图谱), 效果明显下降, 如表1(e)所示.

**Pyramid结构的重要性:** 如果我们只在特征图谱 $P_2$ 上进行检测, 这就像是Faster RCNN的单尺度方法一样, 所有的anchors都在最后一层图谱上, 这种变体比Faster RCNN好但是比FPN差, 如表1(f)所示. 直观上来说, 在所有特征层上进行检测, 对不同尺度的物体的鲁棒性要更好. 同时我们也注意到, 由于 $P_2$ 的分辨率较高, 因此它会产生更多的 anchors, 但是最终的精度并没有提供太过, 这说明生成大量的 anchors 并不足以提升模型的精度.

## 利用Fast/Faster RCNN进行目标检测

接下来我们在基于区域的检测器上来使用验证 FPN 的有效性, 我们利用AP(Average Precision)指标对FPN进行验证.

**实验细节:**
图片被缩放到最短边长为 800 pixels; 在 8 个 GPU 上利用同步 SGD 来训练模型, 每一个 GPU 上的 mini-batch 中包含 2 张图片, 每张图片有 512 个 RoIs. weight decay 为 0.0001, momentum 为 0.9; lr 在前 60k mini-batches 为 0.02, 在后 20k 为 0.002, 在训练时每张图片会产生 2000 个 RoIs, 在测试时会产生 1000 个 RoIs. 在 COCO 数据集上训练带有 FPN 的 Fast R-CNN 需要 10 个小时.

### Fast R-CNN

![表2](https://wx1.sinaimg.cn/large/d7b90c85ly1fy9gxx88aaj21lh0dmaey.jpg)


### Faster R-CNN(on consistent proposals)

![表3](https://wx3.sinaimg.cn/large/d7b90c85ly1fy9gycey9dj21lh0a7n10.jpg)


### Comparing with COCO Competition Winners

![表4](https://wx2.sinaimg.cn/large/d7b90c85ly1fy9gyue4j2j21lr0h6n3j.jpg)

![表5](https://wx2.sinaimg.cn/large/d7b90c85ly1fy9gzlarcnj20xb09qgn8.jpg)


在Faster RCNN上使用FPN, mAP提高了 2%, 其中小物体的mAP提高了2.1%.(固定的候选区域集合)


在面对consistent proposals时(因为RPN和Fast RCNN要共享权重,所以会不断迭代训练), FPN比Faster RCNN的AP高 **2.3** 点, 比AP@0.5高 **3.8** 点.

FasterRCNN中RPN和FastRCNN的权重共享大约可以提升mAP值0.5左右(0.2~0.7), 同时, 权重共享也可以降低预测时间(0.148 vs 0.172, ResNet50, M40 GPU因为不用计算两个不同的权重参数, RPN与Fast RCNN用的是一个权重参数).

**FPN没有使用很多流行的提升精度的方法, 如迭代回归, 难样例挖掘, 上下文建模, 数据增强等等. 但是FPN仍然在目标检测, 实例分割, 关键点检测等多项任务上刷新了最高分. 如果使用这些trick, 理论上会获得更高的精度.**

FPN是一种通用的特征提取方法, 他同样也适用于实例分割任务, 并且可以取得很好的效果.

# Extensions: Segmentation Proposals

![图4](https://wx4.sinaimg.cn/large/d7b90c85ly1fy9h11hrm5j215d0ie4ff.jpg)

![表6](https://wx2.sinaimg.cn/large/d7b90c85ly1fy9h1zwjgaj21ed0lqdzh.jpg)
