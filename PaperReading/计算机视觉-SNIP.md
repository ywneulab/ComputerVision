---
title: SNIP (CVPR, 2018)
sitemap: true
categories: 计算机视觉
date: 2018-09-16 15:05:02
tags:
- 计算机视觉
- 网络结构
- 论文解读
---


**文章:** An Analysis of Scale Invariance in Object Detection - SNIP
**作者:** Bharat Singh, Larry S.Davis
**备注:** Maryland



# 摘要

本文分析了在极端尺度变化下的目标识别和检测的各种技术, 通过对不同输入数据配置的检测器进行训练, 比较了尺度固定(scale specific)和尺度不变(scale invariant)检测器的不同设计. 通过对不同网络结构在 ImageNet 上对小目标分类的性能评估, **我们发现 CNN 对尺度变化并不具有足够的鲁棒性.** 在此基础上, 本文提出了在图像金字塔(image pyramid)的相同尺度上训练和测试检测器. **由于小物体和大物体分别在更小的尺度和更大的尺度上难以识别, 因此, 我们提出了一种新的图像金字塔尺度归一化(Scale Normalization for Image Pyramids, SNIP)训练方案, 该方案可以根据图像尺寸的大小, 有选择的反向传播不同大小对象实例的梯度.** 在 COCO 数据集上, 我们的单模型表现为 45.7%, 三个模型的融合表型为 48.3%. 我们使用了现成的(off-the-shelf) ImageNet-1000 预训练模型, 并且只在边框监督下训练.

# Introduction

相比于图像分类, 为什么目标检测任务更难? 其中一个重要的影响因素就是, 跨实例的大幅尺度变化, 尤其是检测非常小的物体时所带来的挑战. 但有趣的是, ImageNet 中的中型物体相对于图片的尺寸大约有 55.4%, 而 COCO 中只有 10.6%. 因此, COCO 中的大多数对象实例都小于图像区域的 1%! 更糟糕的是, COCO 中最小和最大的 10% 个对象实例占图片的比例分别是 0.024 和 0.472, 如图 1 所示. 因此, detectors 需要处理的尺度变化是非常巨大的, 这对卷积神经网络的尺度不变性提出了极大的挑战. 此外, ImageNet 和 COCO 数据集在对象实例尺度上的差异也会导致从预训练 ImageNet 模型中进行 fine-tuning 时出现较大的 **domain-shift**. 在本文中, 我们首先提供了证明这些问题存在性的证据, 然后提出了一种称为图像金字塔尺度归一化(Scale Normalization for Image Pyramids)的训练方法, 从而在 COCO 上实现了 sota detectors.

<div style="width: 550px; margin: auto">![图1](https://wx4.sinaimg.cn/large/d7b90c85ly1g1nf6w9o6vj20tm0giaco.jpg)

为了解决尺度变化和小物体检测的问题, 有很多方案被提出: 特征金字塔, 空洞(dilated)卷积, 可形变(deformable)卷积, ..., 尽管这些 architectural innovations 极大的提升了目标检测的性能, 但是和 training 相关的许多问题都还没有得到解决:
- 为了获得较好的目标检测结果, 对图像进行上采样是至关重要的吗? 既然检测数据集中图像的典型大小时 480x640, 那么为什么将它们的样本提高到 800x1200 是一种常见的做法? 我们能否再 ImageNet 的低分辨率图像上以较小的步长对 CNN 进行预训练, 然后在检测数据集中对它们进行 fine-tuning, 以检测小物体吗?
- 当从预训练的分类模型中对目标检测任务进行 fine-tuning 时, 我们应该将图片进行适当放缩以使得参与训练的物体尺寸被限制在一个范围内(64x64 ~ 256x256), 还是应该令所有尺寸(16x16 ~ 800x1000)的物体都参与训练?

我们在 ImageNet 和 COCO 上设计了相应的实验来寻找这些问题的答案. 在第三节中, 我们通过在现有网络上测试不同尺度的图像作为输入时的 ImageNet 的分类性能, 研究了 scale variation 的影响. 我们还对 CNN 的结构进行了小的修改, 以便对不同尺度的图像进行分类. **这些实验揭示了上采样操作对小目标检测的重要性.**
Section 5: 分析尺度变化对目标检测的影响
Section 6: SNIP

# Related Work

Scale Space Theory: 学习具有尺度不变性的特征, 在传统检测领域非常流行

Dilated/Atrous Conv

Deformable Conv

Up-sampling: 图片在训练阶段放大 1.5~2 倍, 在测试阶段放大 4 倍. (可以提升特征图谱的尺度)

特征金字塔: 浅层的特征图谱和深层的特征图谱具有互相补充的信息.(smaller objects are trained on higher resolution layers, while larger objects are trained on lower resolution layers). 但是, **尽管特征金字塔能够有效地利用网络中所有层的特征, 但是对于检测非常小/大的物体来说, 它们并不是替代图像金字塔的一个好选择.**
<div style="width: 550px; margin: auto">![图2-3](https://wx3.sinaimg.cn/large/d7b90c85ly1g1nf7eqy7gj21ji0mhx5b.jpg)

# Image Classification at Multiple Scales

本节讨论 domain shift 的影响(training 阶段和 inference 阶段的图片分辨率不同).

**Naive Multi-Scale Inference:** 首先, 对 ImageNet 的图像进行 down-sampling, 或者 48, 64, 80, 96, 128 等不同分辨率的图像. 然后将这些图像全部 up-sampling 到 224 分辨率, 并将其作为输入提供给一个针对 224x224 大小图像进行训练的 CNN 结构, 如图 3 中的 CNN-B 所示. 图 4(a) 中显示了具有 ResNet-101 backbone 的 CNN-B 的 top-1 精度. 我们观察到, 随着 training 和 testing 图像分辨率的差异逐渐增大, CNN 的性能也逐渐下降. 对没有经过训练的分辨率进行 testing 是次优的, 至少对于图像分类来说如此.

**Resolution Specific Classifiers:** 基于上述观察, 提高小目标探测器性能的一个简单的解决方案是在 ImageNet 上对不同步长的分类网络进行预训练. 毕竟, 在 CIFAR10 上获得最佳性能的网络架构不同于 Imagenet. 在 ImageNet 分类网络中, 第一个卷积层的步长为 2, 接着是最大池化层的步长为 2x2, 两次下采样可能会消除小对象中存在的大部分图像信号. 因此, 我们在训练输入为 48x48 的图像时, 将第一层的卷积层核改为 3x3, 步长改为 1, 如图 3 所示. 同样, 同于 96x96 的图像, 将核改为 5x5, 步长改为 2. 同时还是用了一些常用的数据增广技术. 如图 4 所示, 这些网络(CNN-S)的表现均强于 CNN-B. 因此, 对接收低分辨率图像为输入的不同网络结构进行预训练, 然后将其应用于低分辨率目标的检测是一个更好的选择.

**Fine-tuning High-Resolution Classifiers:** 另一个小目标检测的简单解决方案是对上采样的低分辨率图像在 CNN-B 上进行 finetuning, 从而得到 CNN-B-FT(如图 3 所示). 在上采样的低分辨率图像上, CNN-B-FT 的性能优于 CNN-S, 如图 4 所示. 实验结果表明, 在高分辨率图像上学习的 filters 也可以用于低分辨率图像的识别, 因此, 与其将步长减少 2, 不如将图像上采样 2 倍, 然后对使用高分辨率图像预训练的网络进行 fine-tuning.

<div style="width: 550px; margin: auto">![图4-5](https://wx3.sinaimg.cn/large/d7b90c85ly1g1nf8hkdmoj21hd0u0hdt.jpg)

在训练 object detectors 时, 我们可以使用不太的网络结构对不同分辨率的对象进行训练, 也可以令所有的分辨率都使用同一体系结构. 由于在 ImageNet 上进行训练是有益的, 并且在较大的目标实例上学习的 filters 有助于对较小的目标实例进行分类, 因此对图像进行上采样并使用在高分辨率图像上进行预训练的网络应该比专门用于较小对象进行分类的网络更好. Fortunately, 现有的 object detectors 通过对图像进行上采样来检测较小的对象, 而不是使用不同的网络结构. 本文的分析支持这种做法, 并将其与其他的替代方法进行比较, 以强调差异.

# Background

Deformable-RFCN

# Data Variation or Correct Scale?

第三节的研究证明, 在 training 和 testing 阶段之间的使用的分辨率的差异性可以导致性能显著下降. 不幸的是, 这种分辨率上的差异性是当前的 GPU 内存限制造成的, 训练时的分辨率(800x1200)低于测试时(1400x2000)的分辨率. 本节分析了图像分辨率, 目标实例的尺寸和数据变化对 detectors 的影响. 我们在不同的 settings 下对 detectors 进行训练, 并在 1400x2000 的图像上对小物体(低于 32x32 像素)进行检测. 实验结果如表 1 所示.

<div style="width: 550px; margin: auto">![表1-3](https://wx1.sinaimg.cn/large/d7b90c85ly1g1nfajpr1pj21is0intf8.jpg)

**Training at different resolutions:** 我们首先训练两种不同分辨率(800x1400, 1400x2000)下的所有对象实例训练 detectors, 如图 5(1) 所示. 实验结果和预期一样, 1400 的表现强于 800, 因为前者在训练和推理时采用了相同的分辨率. 但是, 这种提升很少(only marginal), 这是为什么呢, 为了回答这个问题, 我们思考了在用大分辨率的图像进行训练时, 中型目标和大型目标会受到什么影响? 很明显, 这些物体变的太大而无法正确的分类了(图像变大的另一个影响就是感受野变小了), 在高分辨率下的训练可以将小物体放大以获得更好的分类, 但是这样会降低中型物体和大型物体的检测效果.

**Scale specific detectors:** 我们用 1400x2000 的分辨率训练了另一个 detectors, 同时忽略了所有中型和大型物体(原始像素大于 80), 以此消除大型物体的有害影响, 如图 5(2) 所示. 但是不幸的是, 实验结果甚至比 800 像素下的训练还要差, 产生这种现象的原因是因为由于忽略了中型物体和大型物体(站物体总数约 30%), 我们失去了外观和姿态变化的一个重要来源, 而这种忽略带来的性能损害要远大于其带来的好处.

**Multi-Scale Training(MST):** 最后, 我们评估了在训练过程使用随机分辨率来提升 detectors 尺度不变性的常见方法, 称为多尺度训练(MST), 如图 5(3) 所示. 它可以使用一个参与训练的对象实例在不同的尺度下被观察到, 但是同时它也会被极大或者极小的物体影响性能. 它的性能表现和 800 像素下的训练结果相似. 由此, 我们的结论是, 在训练时不仅要在一个合适的尺度下训练, 同时还应该能够捕捉物体上尽可能多的变化. 下一节我们将介绍本文的训练方法, 它取得了很大的性能提升.

# Object Detection on an Image Pyramid

我们的目标是将上面结论中的两个关键因素结合起来, 即将训练图片的尺寸控制在合理范围内的同时, 使参与训练的物体在外观和形态上具有更多的多样性. 我们还讨论了在当前 GPU 内存限制下, 在图像金字塔上训练 detectors 的细节.

## Scale Normalization for Image Pyramids

SNIP 是 MST 的修改版本, 其中只有分辨率接近 ImageNet(224) 尺寸的物体实例才会被用来训练 detector. 在 MST 中, 每一张图片都会在不同的分辨率下被观察到, 因此, 在高分辨率(1400x2000)下大型物体很难被检测识别, 在低分辨率(480x800)下小型物体很难被检测识别. 幸运的是, 每个对象实例都会在多个分辨率下进行训练, 而其中正好有一些处在我们期望的分辨率范围内. **为了消除过大或者过小的极端尺度对象, 我们只对处于所需尺度范围内的对象进行训练, 对于其他的对象则不会进行反向传播计算.** 为了更有效的训练, SNIP 在训练过程中使用了所有的对象实例(但不是都会参与 BP 计算), 这有助于捕获外观和姿态的所有变化, 同时减少了预训练网络在 scale-space 中的 domain-shift. 最终结果如图 1 所示, it outperforms all the other approaches. 实验结果证明了 SNIP 在检测小物体上的有效性. 接下来我们将会讨论 SNIP 的实现细节.

在训练分类器时, 所有的 gt boxes 都会被用来给 proposals 分配标签. **在训练期间, 我们不会选择超出特定分辨率范围的 proposals 和 gt boxes.** 对于一个特定的分辨率 $i$, 如果 RoI 的面积 $ar(r)$ 落在了 $[s_i^c, e_i^c]$ 区间内, 我们就将其标记为有效样本, 否则将其标记为无效样本. 同样, RPN 在训练时也使用所有的 gt boxes 来为 anchor 分配标签. 最后, 那些与 invalid gt box 的交并比大于 0.3 的 anchors 都会在训练中剔除(即, 他们的梯度设置为 0). 在 Inference 阶段, 我们在每个分辨率下都使用 RPN 来生成 proposals, 并在每个分辨率下进行单独分类, 如图 6 所示. 和训练阶段相同, 我们在每个分辨率下没有选择落在特定范围外的检测结果(是 detectin, 不是 proposals). 在分类和回归操作完成后, 我们使用 soft-NMS 来融合多个分辨率下的检测结果, 以便获取最终的检测结果, 如图 6 所示.

<div style="width: 550px; margin: auto">![图6](https://wx1.sinaimg.cn/large/d7b90c85ly1g1nf97t0s9j21jf0oyhat.jpg)

池化后的 RoI 的分辨率和预训练的网络相匹配, 因此在 fine-tuning 期间网络更容易学习. 对于像 R-FCN 这样将 RoI 分成子部分并使用 position sensitive filters 的方法, 这边的更加重要. 例如, 如果 RoI 的大小为 48 个像素(在 conv5 中就会变成 3 像素), 并且每个轴上都有 7 个 filters, 那么特征和 filters 之间的位置对应关系就会丢失.

## Sampling Sub-Images

用深度卷积网络训练高分辨率的图像需要更多的 GPU 显存. 因此, 我们对图片进行剪裁, 以使得他们可以适应 GPU 的显存大小. 我们的目标是生成最小数量的 1000x1000 尺寸的 chips(sub-images), 并让这些 chips 覆盖图像中的所有小目标. 这有助于加速训练, 因为在没有小目标物体的地方不需要进行计算. 为此, 我们为每张图像生成 50 个随机定位的芯片, 大小为 1000x1000. 我们将覆盖最多目标对象的 chips 添加到我们的训练集中, 在覆盖图片中的所有目标物体之前, 我们队其余对象重复这个采样和选择过程. 由于 chips 是随机生成的, 而 proposals boxes 通常在图像边界上有一个边, 因此为了加速采样过程, 我们将芯片对于到图像的边界. 我们发现, 平均情况下, 对于 1400x2000 的图像, 可以生成 1.7 个尺寸为 1000x1000 的 chips. 在图片分辨率为 800x1200 或者 480x640 , 或者图片中不包含小物体时, 这个采样步骤是不需要的. 随机剪裁并不是 detectors 性能提高的原因. 为了验证这一点, 我们使用了非剪裁的高分辨率图像(1400x2000)来训练 ResNet-50, 并且没有发现 mAP 的变化.

# Datasets and Evaluation

small objects: less than 32
medium objects: from 32 to 96
large objects: greater than 96

## Training Details

在三个不同的分辨率上训练 Deformable-RFCN: 480x800, 800x1200, 1400x2000, 第一个值代表 shorter side 的长度, 第二个值代表边长的最大值;
training: 7 epochs for classifier while RPN is trained for 6 epochs.
warmup learning rate: 0.0005 for 1000 iterations, then 0.005. step down is performed at 4.33 epochs for RPN and 5.33 epochs otherwise.
valid range: $[0, 80], [40, 160], [120, \infity]$

**SNIP doubles the training time**

## Improving RPN

## Experiments

表 2 展示了 single scale model, multi-scale testing, multi-scale training + multi-scale testing 的性能

表 3 展示了 Average Recall 的性能表现, 使用更强的分类网络(DPN-92)也能够提升 AR.

<div style="width: 550px; margin: auto">![表1-3](https://wx1.sinaimg.cn/large/d7b90c85ly1g1nfajpr1pj21is0intf8.jpg)

表 4 展示了各个模型与 SNIP 的消融对比实验

<div style="width: 550px; margin: auto">![表4](https://wx4.sinaimg.cn/large/d7b90c85ly1g1nfawjkj3j21j40jaagf.jpg)
