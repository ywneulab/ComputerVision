---
title: Bag of Freebies (Arxiv, 2019)
sitemap: true
categories: 计算机视觉
date: 2019-02-20 14:57:22
tags:
- 计算机视觉
- 目标检测
---
**文章:** Bag of Freebies for Training Object Detection Neural Networks
**作者:** Zhi Zhang, Tong He, Hang Zhang, Zhongyuan Zhang, Junyuan Xie, Mu Li
**机构:** Amazon Web Services


# 摘要
与针对更好的图像分类模型的大量研究成果相比, 应用于目标检测器训练的研究在普及度和广泛性上相对较差. 由于越来越复杂的网络结构和优化目标, 各种训练策略和流程(pipeline)都是针对特定的检测算法而设计的. 在本文中, 我们将会探索通用的调整方法, 来帮助提高现有目标检测模型的性能, 并且确保不降低 inference 速度. 我们的实验表明, 本文提出的这些方法可以使得现有模型的绝对精度提高5%, 因此, 我们建议大家在一定程度上将其应用于目标检测训练当中.

<div style="width: 550px; margin: auto">![图1](https://wx1.sinaimg.cn/mw690/d7b90c85ly1g19b68lzyfj20wg0n10vf.jpg)

# Introduction
目标检测无疑是计算机视觉领域的前沿应用之一, 受到了各个领域研究者的关注. 最新的检测模型都是基于分类的 backbone 网络(VGG, ResNet, Inception, MobileNet)建立的.
但是, 由于模型容量和训练复杂度相对较高, 目标检测任务从最近的 training tweaks 的研究中获益较少. 更糟糕的是, 不同的检测网络在没有进行显式初始化, 数据预处理和优化分析的情况下, 只使用自己的训练步骤, 这使得在采用最新技术来提高图像分类能力时造成了巨大的混乱.
在本文中, 我们的重点是探索有效的方法, 使得可以在提高主流目标检测网络性能的同时, 维持原有的计算成本. 我们首先探讨了将混合技术(mixup: Beyond empirical risk minimization)应用在目标检测上. 和 mixup 原文不同的是, 我们注意到多目标检测任务倾向于保持空间变换(favors spatial preserving transforms)的特殊性质, 因此提出了一种 **用于多目标检测任务的视觉相干图像混合方法(visually coherent image mixup methods for object detection tasks).** 然后, 我们讨论了训练过程的许多细节问题, 包括学习率调度(learning rate scheduling), 权重衰减(weight decay)和同步 BatchNorm (synchronized BatchNorm)等. 最后, 我们讨论了本文提出的训练策略优化(training tweaks)的有效性, 并通过增量叠加这些优化来训练 one-stage 和 multiple-stage 的目标检测网络. 我们的主要贡献可以总结如下:
1. 我们首次系统的评估了各种应用于不同目标检测流程的启发式训练规则, 为未来的研究提供了有价值的实践指导.
2. 我们提出了一种用于训练目标检测网络的视觉相干图像混合方法, 并证明了该方法在提高模型泛化能力方法的有效性.
3. 我们在不改变现有模型网络结构和损失函数的情况下, 实现了 5% ~ 30% 的绝对精度提高. 我们所做的一切都没有增加额外的推理成本.
4. 我们扩展了目标检测中数据增广领域的研究深度, 显著的增强了模型的泛化能力, 并有减少了过拟合问题. 实验还展示了一些很好的技术, 可以在不同的网络结构中 **一致的** 提高目标检测结果.

# Related Work

## Scattering tricks from Image Classification

Learning rate warm up: 用于克服超大的 mini-batch size 带来的负面影响(Accurate, large minibatch sgd: training imagenet in 1 hour, FAIR 铁三角). 有趣的是, 尽管在典型的目标检测训练中使用的 mini-batch size 和图像分类中的规模(10k, 30k)相差甚远, **大量的 anchor (30k) 有会隐式的增加 mini-batch 的大小. 在我们的实验中, 我们发现热身机制(gradual warmup heuristic)对于 YOLOv3 来说是直观重要的.** 有一系列的方法试图解决深层神经网络的脆弱性. **Inceptionv3 介绍了标签平滑性(Label smoothing), 对交叉熵损失中的难真实标签(hard ground truth labeling)进行的改进.** Zhang 等人 **提出了 mixup 方法来缓解对抗性扰动**. 针对传统的步长策略, Sgdr 提出了学习率衰减的 **余弦退火策略**. He Tao等人通过探索大量 tricks, 在训练精度上取得了显著的提高(Bag of tricks for image classification). 在本文中, 我们将会深入研究这些由分类任务引入的启发式技术在目标检测任务中的应用.

## Deep Object Detection Pipelines

SSD, YOLO, Faster R-CNN 等

由于 one-stage 缺少空间上的多样性, 因此, 对于 one-stage 模型来说, 空间数据增广对于性能的提升非常重要.

# Technique Details

## Visually Coherent Image Mixup for Object Detection

由 Zhang Hongyi 等人引入的 mix-up 被证明在减轻分类网络中的对抗性扰动方面是成功的. 原文中的 mix-up 算法的混合比例(blending ratio)是通过 beta 分布得到的($a = 0.2, b = 0.2$). 大部分的混合结构几乎都是 beta 分布的噪声. 受到 Rosenfeld 等人启发式实验(The elephant in the room)的影响, 我们更关注那些在目标检测任务中起重要作用的自然共现目标的特征表示. 半对抗性的目标移植方法并不是传统的攻击方式. 通过应用更复杂的空间变换, 我们引入了在自然图像表示中常见的遮挡, 空间信号扰动.
在我们的经验实验中, 当我们持续增加 mixup 中的混合比(blending ratio)时, 得到的图像帧中的目标物体变的更加具有活力和连贯的自然表现, 就像是在低 FPS 电影中的过渡帧一样. 图2和图3分别展示了图像分类和这种高混合比例融合的视觉对比. 特别的, 我们使用几何保留对其(geometry preserved alignment)的图像融合, 以避免在最初中步骤中扭曲图像. 我们同时还选择了一个更具视觉一致性的 beta 分布($a\geq 1, b\geq 1$), 而不是采用和图像分类中同样的融合方法, 如图4所示.

<div style="width: 550px; margin: auto">![图2](https://wx2.sinaimg.cn/mw690/d7b90c85ly1g19b6nohbtj214w0bf7gq.jpg)
<div style="width: 550px; margin: auto">![图3](https://wx2.sinaimg.cn/mw690/d7b90c85ly1g19b6yy0xnj21ky0me4qp.jpg)
<div style="width: 550px; margin: auto">![图4](https://wx4.sinaimg.cn/mw690/d7b90c85ly1g19b78abu3j20vh0p476c.jpg)

我们还在 Pascal VOC 数据集上用 YOLOv3 对经验混合比的分布进行了实验测试. 表1显示了使用 detection mixup 后实际的提升情况. 可以看出, 使用 $\alpha = 1.5, \beta = 1.5$ 的 beta 分布稍微比使用 1.0 的 beta 分布(相当于是均匀分布, uniform distribution)好一点, 同时也比一半一半的混合方法好.
<div style="width: 550px; margin: auto">![表1](https://wx4.sinaimg.cn/mw690/d7b90c85ly1g19b7z2y70j20vg0f5q5q.jpg)

为了验证视觉相干混合(visually coherent mixup)的有效性, 我们进行了和 "The elephant in the room" 论文中相同的实验, 我们将大象的图片在一张室内的图片上滑动通过. 我们在 COCO 2017 数据集上训练了两个 YOLOv3 模型, 除了模型 mix 使用了我们的 mixup 方法以外, 其他的设置都是相同的. 我们在图5中给出了一些令人惊讶的发现. 如图5所示, 原始模型(vanilla model)没有采用我们的 mix 方法, 它在面对 "房间里的大象" 时往往很难正确检测, 这是由于当缺乏足够的上下文时, 原始的模型不足以检测到我们希望的目标, 实际上, 当我们检查了常用的数据集以后, 也没有发现有这样的训练图像. **相比之下, 使用 mix 方法的模型在训练后会变得更加健壮, 这多亏了随机生成的具有视觉欺骗性的训练图像**. 除此以外, 我们还注意到使用了 mix 方法训练后的模型会变得更加 "不自信", 它在给目标物体进行置信度打分的时候, 往往会给出较低的分数. 但是, 这种行为并不影响我们将在实验部分中展示的评估结果.

<div style="width: 550px; margin: auto">![图5](https://wx1.sinaimg.cn/mw690/d7b90c85ly1g19b7nmvonj21ms0u0b2a.jpg)

## Classification Head Label Smoothing

对于每一个物体, 检测网络经常会在所有类别上使用 softmax 函数来计算概率分布:

$$p_i = \frac{e^{z_i}}{\sum_j e^{z_j}} \tag 1$$

上式中, $z_i$ 是直接从最后一个用于分类预测的线性层中获得的非归一化对数. 对于训练阶段中的目标检测任务来说, 我们只通过比较输出分布 $p$ 和真实分布 $q$ 的交叉熵来修正分类损失:

$$L = \sum_i q_i log p_i \tag 2$$

$q$ 通常是 one-hot 编码, 正确类别位置为1, 其他位0. 但是, 对于 softmax 函数来说, 只有当 $z_i >> z_j, \forall j \neq i$ 时才能逼近公式(2)分布, 但是永远不会完全等于它. 这使得模型对自己的预测过于自信, 容易过度拟合(因为模型会趋势预测结果想着 $z_i >> z_j, \forall j \neq i$ 的方向更新).
**标签平滑化(Label smoothing)** 是由 Szegedy (Inception v3)等人提出的一种正则化形式(regularization). 我们用下面的公式来对真实分布进行平滑化处理:

$$q^{'}_i = (1 - \epsilon) q_i + \frac{\epsilon}{K} \tag 3$$

上式中, $K$ 代表类别的数量, $\epsilon$ 是一个很小的常数. 这个技巧可以降低模型的自信度, 由最大对数和最小对数之间的差值来衡量.
在 YOLOv3 中, 当 sigmoid 的输出为 0 到 1 时, 通过修正目标范围的上下限可以让 label smoothing 过程变的更加简单.

## Data Pre-processing

和图像分类领域中网络对几何变换具有极强的容忍度不同. 实际上, 我们鼓励这样做来提高模型的泛化精度. **但是, 对于目标检测任务的图像预处理来说, 由于检测网络对于这种转换更加敏感, 因此我们需要格外的谨慎**. 我们通过实验回顾了以下数据增强的方法:
- 随机几何变换(Random geometry transformation). 包括随机剪裁(random cropping with constraints), 随机扩展(random expansion), 随机水平翻转(random horizontal flip), 随机缩放(random resize with random interpolation)
- 随机颜色抖动(Random color jittering). 包括亮度(brightness), 色调(hue), 饱和度(saturation), 和对比度(contrast).
就检测网络的类型而言, 有两种类型. 第一种是 one-stage 检测网络, **其最后的输出结果是从特征图谱上的每一个像素点生成的.** 如 SSD 和 YOLO. 第二种是 multi-stage 检测网络, 如 Fast-RCNN 等, 它的检测结果是通过在特征图谱上进行滑动窗口式的遍历生成的 proposals 进行回归和分类得到的
由于基于 proposals 的方法会产生大量的候选框, 因此它在一定程度上梯段了随机检测输入图像的操作, 所以这些网络在训练阶段不需要大量的几何增强.

## Training Scheduler Revamping(改进)

在训练过程中, 学习率往往从一个相对较大的数字开始, 然后在整个训练过程中逐渐变小. 例如, step schedule 是使用最广泛的学习率调整策略. 在 step schedule 中, 在达到预定义的时间点或者迭代次数之后, 学习率会乘以一个小于 1 的常数. 例如 Faster R-CNN 和 YOLO 都采用的这种学习率调整策略. **Step schedule 方法的学习速率转换机制过于突然, 这可能导致优化器在接下来的几个迭代中重新稳定学习势头(learning momentum).** 相比之下, Loshchilov 提出了一种更平滑的余弦学习率调整策略(cosine learning rate adjustment). cosine schedule 根据 cosine 函数在 0 到 $\pi$ 上的值来调整学习率. 它从缓慢的降低大的学习率开始, 然后在中途快速降低学习速度, 最后以微小的斜率降低小的学习率直至达到0.
**Warm up learning rate 是另一种常见的策略, 它可以在初始训练迭代过程中发生梯度爆炸.** Warm-up learning rate 策略对于一些目标检测模型来说至关重要, 例如 YOLOv3, which has a dominant gradient from negative examples in the very beginning iterations where sigmoid classification score is initialized around 0.5 and biased towards 0 for the majority predictions.
训练时使用 cosine schedule 和适当的 warmup 可以得到更好的验证集检测结果, 如图6所示. 在训练过程中, 使用 cosine schedule 的验证集 mAP 始终优于 step schedule. 优于学习率的调整频率较高, 也较少出现 step schedule 的 plateau 现象(由于当前学习率下的训练以及近乎收敛, 所以验证集性能会在学习率降低前卡住(stuck)一段时间).
<div style="width: 550px; margin: auto">![图6](https://wx1.sinaimg.cn/mw690/d7b90c85ly1g19b8fb260j20vy0r0goz.jpg)

## Synchronized Batch Normalization

近来来, 大量的计算需求迫使训练环境必须装备多个设备(GPUs)来加速训练. 尽管在训练过程中需要处理不同的超参数来应对更大的 mini batch 大小, BN 仍然由于其实现细节而引起了多设备用户的注意. 尽管在多设备上工作的 Batch Norm 的典型实现非常快(没有通信开销), 但它不可避免的会减少 mini batch 的大小, 并在计算期间导致统计数据略有不同, 这可能会降低性能. 在一些标准的视觉任务中, 例如 ImageNet 分类, 这不是一个重要的问题(**因为每个设备的 mini batch 大小通常足够大, 可以获得良好的统计数据**). 但是, 在某些任务中, mini batch 的设置通常会非常小, 这可能会影响性能. 最近, Peng 等人通过 Megdet(A large mini-batch object detector) 分析证明了同步(synchronized) Batch Norm 在目标检测中的重要性. 在本文中, 我们重新审视了 YOLOv3 中 Synchronized Batch Normalization 的重要性, 以评估相对较小的 batch-size 对每个 GPU 的影响. 因为目标检测任务中的图片大小远远大于分类任务中的图片, 因此, 无法使用较大的 mini-batch.

## Random shapes training for single-stage object detection networks

自然训练图像具有多种性状. 为了使用内存限制和使用更简单的 batching, 许多 one-stage 目标检测网络采用固定的形状进行训练. 为了降低过拟合的风险, 同时提高模型预测结果的泛化性, **我们采用随机尺度训练的方法(正如 YOLOv3 中一样).** 更具体的说, 我们将一个 mini-batch 中的 $N$ 张训练图片 resized 到 $N\times 3\times H\times W$ 的大小, 这里的 $H$ 和 $W$ 通过乘以一定的系数 $D = randint(1, k)$ 来获得. 例如, 在 YOLOv3 的训练中, $H = W \in \{320, 352, 384, 416, 480, 512, 544, 576, 608\}$

# Experiments

<div style="width: 550px; margin: auto">![表2](https://wx2.sinaimg.cn/mw690/d7b90c85ly1g19b90gsknj20vi0fk773.jpg)
<div style="width: 550px; margin: auto">![表3](https://wx3.sinaimg.cn/mw690/d7b90c85ly1g19b97yusej20vl0cstat.jpg)

为了比较所有的 tweaks 对目标检测结果的增量改进, 我们使用了 YOLOv3 和 Faster R-CNN 分别作为 one-stage 和 two-stage 的代表模型. 为了适应大规模的训练任务, 我们使用 VOC 进行精细级的技巧评估(trick evaluation), 使用 COCO 数据集进行整体性能增兴和泛化能力的验证.

**Pascal VOC.** 按照 Fast-RCNN 和 SSD 中常用的设置, 我们使用 Pascal VOC 2007 trainval 和 Pascal VOC 2012 进行训练, 并使用 2007 test 进行验证. 最终的结果根据 VOC 中定义的 mAP(mean average precision) 进行评估. 对于 YOLOv3 模型, 我们始终在 $416\times 416$ 的分辨率下测量评价精度(mAP). 如果使用随机尺度训练, YOLOv3 模型的随机分辨率将从 320 增加到 608, 增量大小为 32, 否则将固定在 416 尺寸进行训练. **Faster R-CNN 模型采用任意的输入分辨率**. 为了约束训练时的内存消耗, 输入图片的较短边会被 resized 到 600, 同时要确保较长边不超过 1000 pixels. Faster R-CNN 的训练和验证过程遵循相同的预处理步骤, **只不过训练图像将有 0.5 的概率进行水平翻转来进行数据增广**. YOLOv3 和 Faster-RCNN 的实验细节如表2和表3所示. 实验结果表明, 通过叠加这些技巧, YOLOv3 模型可以获得高达3.5%的性能增益, 而 Faster-RCNN 模型的技巧增益效果较差, 但获得了类型的性能提升. **从实验结果我们还发现, 数据增广对于 two-stage 的 Faster R-CNN 模型的增益效果一般, 但是对于 YOLOv3 模型来说却构成了非常重要的影响, 其原因是由于 YOLOv3 模型缺乏空间突变操作(spatial mutational operations)**

**MS COCO.** COCO 2017 比 VOC 数据集大 10 倍, 同时包含了更多的小物体. 我们使用 MS COCO 来验证本文中 ticks 的泛化性. 我们使用了和 VOC 数据集上差不多的训练和验证设置, 只不过 Faster R-CNN 使用了 $800 \times 1300$ 的大小来适应更小的目标物. 最终的结果如表4所示. 总的来说, 我们的 ticks 可以令 ResNet50 和 ResNet101 的 Faster-RCNN 模型的 mAP 分别提升 1.1% 和 1.7%. 同时可以将 YOLOv3 模型的 mAP 提升 4.0%. 注意, 所有这些结果都是通过在完全兼容的 inference model 中生成更好的权重得到的, 所以这些提升可以完美的应用在 inference 中.
<div style="width: 550px; margin: auto">![表4](https://wx3.sinaimg.cn/mw690/d7b90c85ly1g19bamtirlj21l50ejwig.jpg)

**Impact of mixup on different phases of training detection network**
Mixup 可以在目标检测模型中的两个阶段中使用: (1) 在预训练的 backbone 网络中使用 mixup; (2) 利用提出的针对目标检测任务的视觉相干混合(visually coherent image mixup)来训练目标检测网络. 我们比较了使用 Darknet-53 的 YOLOv3 和 ResNet101 的 Faster R-CNN 来进行比较, 结果如表5和表6所示. 结果表明, 不论在那个阶段使用 mixup 都可以取得一致的性能提升. **同样值得注意的是, 在这两个阶段都使用 mixup 可以产生更加显著的增益, 即 $1+1 > 2$**.

<div style="width: 550px; margin: auto">![表5](https://wx2.sinaimg.cn/mw690/d7b90c85ly1g19bav6yhlj20tx08y75k.jpg)
<div style="width: 550px; margin: auto">![表6](https://wx3.sinaimg.cn/mw690/d7b90c85ly1g19bb6bbj3j20u909rq43.jpg)
<div style="width: 550px; margin: auto">![图7](https://wx1.sinaimg.cn/mw690/d7b90c85ly1g19baam4rgj21680u0qv5.jpg)
