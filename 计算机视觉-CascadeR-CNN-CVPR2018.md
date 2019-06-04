---
title: Cascade R-CNN (CVPR, 2018)
sitemap: true
categories: 计算机视觉
date: 2018-11-13 14:19:10
tags:
- 计算机视觉
- 目标检测
---
**文章:** Cascade R-CNN: Delving into High Quality Object Detection
**作者:** Zhaowei Cai, Nuno Vasconcelos
**机构:** UC San Diego

# 核心亮点

**本文针对检测问题中区分正负样本的 IoU 阈值选择问题提出了一种新的目标检测框架, Cascade R-CNN**
周所周知, 在 two-stage 的目标检测模型当中, 需要设置 IoU 阈值来区分正样本和负样本, 通常, 阈值选的越高, 正样本的框就与真实框越接近, 但是这样就会使得正样本的数量大大降低, 训练时容易产生过拟合问题, 而如果阈值选的较低, 就会产生大量的假正例样本. 根据经验和实验证明可知, **当输入的 proposals 和真实框的 IoU 的值, 与训练器训练时采用的 IoU 的阈值比较接近的时候, 训练器的性能会比较好**, 为此, 作者提出了一种级联式的阈值训练方法, 先在较低的阈值上训练检测器, 得到具有更高 IoU 的候选框输出, 然后在此基础上进行训练, 不断提升 IoU 的阈值, 这样一来, 最终生成的候选框质量会变得更高 (与真实框的 IoU 更大). **作者提出这种框架的启发来自于图1(c), 整体来说, 输入的 proposals 的 IoU 在经过检测器边框回归以后, 其输出的边框与真实框会有更大的 IoU, 因此可以将这个具有更大 IoU 的框作为下一个检测器的输入, 同时调高训练时的 IoU, 进而得到质量更高的框**

<div style="width: 550px; margin: auto">![](https://wx2.sinaimg.cn/large/d7b90c85ly1fx6k6fv7t4j20sz0dcdi0.jpg)


# 摘要

在目标检测任务中, 交并比 (IoU, Intersection over Union) 常常被用来区分正样本和负样本. 如果一个检测器在训练时, IoU 阈值设置的比较低时(如0.5), 通常会导致产生更多的噪声检测. 但是, 如果一味的提高 IoU 阈值, 检测器的性能也会有所下降. 产生这一现象的因素主要有两点: 1), 由于阈值升高后, 大量样本都会无法被当做正样本训练, 使得正样本的数量大大降低, 从而容易在训练时产生过拟合; 2, 在 inference 阶段,检测器优化的(人为设定的) IoU 阈值和假设的(真实的) IoU 阈值之间的不匹配. 为了解决上述问题, 本文提出了一个 multi-stage 的目标检测模型架构, 称为 Cascade R-CNN. 它由一系列经过训练的检测器组成, 这些检测器的 IoU 阈值不断增加, 从而对假阳性样本更加具有选择性. 这些检测器会被阶段性 (stage by stage) 的训练, 而如果当前检测器的输出是一个好的分布, 就会用于训练下一个阶段的检测器, 从而得到更好的检测器. 对逐渐改进的假设进行重采样, 保证所有的检测器都具有一组同等大小的正样本集合, 从而缓解了过拟合问题. 相同的 cascade 流程会继续应用在 inference 阶段, 确保可以令每一个 stage 的检测器质量与假设之间更加匹配. Cascade R-CNN 的一个简单实现在 COCO 数据集上胜过了其他所有的单模型检测器. 并且, 实验表明, Cascade R-CNN 可以应用在多种模型架构之中, 并且无论 baseline detector 的性能是强还是弱, Cascade R-CNN 总是能够进一步提升模型的性能 (consistent gains).

# 介绍

目标检测任务需要同时解决分类和定位两个问题, 这两个问题都不太好解决, 因为往往检测器会面对许多 close false positives, 简单来说就是指许多密集重复的无用框 (很接近正样本的假正例). 检测器必须要找到真正的正样本, 同时还要抑制这些假正例. 在 two-stage 检测模型中, 我们将分类和候选框生成任务分为两个阶段执行, 此时需要设置一个 IoU 阈值来标记候选框的正负, 通常情况下, 我们设置一个固定的 IoU 阈值 ($\mu$) 进行训练和预测, 但是, 这种设置(如, $\mu=0.5$)实际上建立了一个相等宽松的正样本要求. 结果就是当阈值设置较低时, 往往会产生出很多的噪声候选框, 如图1(a)所示. 大多数人的假设都会使得假正例通过 $IoU \geq 0.5$ 的限制. 虽然在 $\mu = 0.5$ 的准则下获得的样本更加丰富多样, 但是它们也会使得检测器难以有效的拒绝这些假正例.

<div style="width: 550px; margin: auto">![图1](https://wx2.sinaimg.cn/large/d7b90c85ly1fx6guu76wej20z80w1e81.jpg)

在本文中, **我们将假设质量(quality of an hypothesis)定义为 proposals 与真实框之间的 IoU, 而将检测器质量(quality of the detector)定义为训练时使用的 IoU 阈值 $\mu$**. 我们的目标是研究一个目前为止较少研究的问题, **即找到高质量的检测器 IoU 阈值, 该检测器会输出更少的 close false positives (注意, 我们是希望得到更少的假正例, 而不是希望负样本的数量变低, 这两个是有区别的, 不要搞混), 如图1(b)所示.** 本文的基本思想是, 如果只使用一个单一的模型, 那么我们就只能在一个单一的级别上优化检测器的 IoU 阈值. 这是著名的 cost-sensitive 学习迭代. 我们与之不同的地方在于, 我们是基于 IoU 阈值来进行优化的, 而不是基于假正例样本率.

核心思想如图1(c)和(d)所示, 图中展示了三种 IoU 阈值下的检测器的定位和检测性能. 定位性能是关于输入的候选框 IoU 的函数, 检测性能是关于 IoU 阈值的函数(COCO 数据集). 从图1(c)可以看出, 当输入的 proposals 的 IoU 在0.5~0.6之间时, 训练时回归器采用 $\mu=0.5$ 可以获得最大的 IoU 输出(预测结果的框与真实框的 IoU 越大, 说明这些框的质量越好), 而当输入的 proposals 的 IoU 在 0.6~0.75 之间时, 训练时采用 $\mu=0.6$ 时的性能最好, 再之后就是 $\mu=0.7$ 时的性能最好. 可以得出, **只有 proposal 自身的阈值和训练器训练时用的阈值较为接近时, 训练器的性能才更好**. 从图1(d)中可以看出, 使用 $\mu = 0.5$ 的检测器相比于 $\mu = 0.6$ 的检测器, 在面对 IoU 较低的 proposals 样本时, $\mu = 0.5$ 的检测器性能较好, 在面对 IoU 较高的 proposals 样本时, $\mu = 0.6$ 的检测器性能较好. **一般来说, 在一个 IoU 水平上优化的检测器在其他水平上不一定是最优的.** 这些观察表明, 更高质量的检测结构要求检测器和它处理的假设之间具有更紧密的质量匹配(closer quality match). **一般来说, 检测器只有在面对高质量的 proposals(与gt有高iou值) 时, 才能生成高质量的检测结果.**

但是, 为了生成一个高质量的检测器, 仅仅在训练阶段提升 $\mu$ 的值是不够的. 实际上, 如图1(d)中的 $\mu = 0.7$ 的检测器所示, 一味的升高 $\mu$ 会降低检测性能(全程低于 $\mu = 0.5$ 和 $\mu = 0.6$ 的检测器). 其问题是因为从 proposal detector 中得到的假设分布通常对于 low quality 的 proposals 严重失衡. 一般来说, **当我们设置更大的 IoU 阈值以后, 就会使得参与训练的正样本的数量减少, 这对于神经网络来说很容易导致过拟合现象的发生.** 另一种难点就在于 **检测器的质量与 inference 阶段时的假设质量不匹配**. 如图1所示, 高质量的检测器仅仅只对高质量的假设是最优的. 当它们在面对其他质量水平的假设时, 检测可能不是最优的.

在这片文章中, 为了解决上述问题, 我们提出了一种新的检测框架: Cascade R-CNN. 它是 R-CNN 的一种 multi-stage 扩展, 在该模型中, **处于级联网络更深处的检测器对于难分辨的假正例(close false positive)具有更强的选择性**. Cascade R-CNN 是 **按阶段训练的 (stage by stage)**, 它会用一个 stage 的输出来训练下一个检测器. 这是通过观察图1(c)中, 每一个检测器输出的 IoU 总是比输入的 IoU 更好而受到的启发. Cascade R-CNN 的流程很像 **boostrapping** 方法. 但是不同之处在于 Cascade R-CNN 的冲采样过程不是为了挖掘难反例. 相反, 通过调节 bounding boxes, 每一个 stage 都会去寻找更好(容量更少)的 close false positives 集合来训练下一个 stage. 在具体操作时, 一系列的检测器会在一组递增的 IoU 阈值集合上训练, 以此避免过拟合问题(直接在大的 IoU 上训练会导致过拟合). 在测试阶段, 会执行同样的流程. 这种逐步改进的假设与每个阶段的检测器质量的匹配度会更好.

**举例说明:**
有三个串联起来的用0.5/0.6/0.7的阈值训练出来的detector，有一个 IoU 约为0.55的proposal，经过0.5的detector，输出的物体框的 IoU 变为0.75；将此框再经过 0.6 的detector，输出的 IoU 变为 0.82；再经过 0.7 的detector，最终IoU变为 0.87. 这比任何一个单独的detector的结果都要好。同时，因为每经过一个 detector，其输出的 proposal 的 IoU 都更高，样本质量更好了，那么即使我下一个 detector 阈值设置得比较高，也不会有太多的样本被刷掉，这样就可以保证样本数量避免过拟合问题。

Cascade R-CNN **很容易实现并且可以端到端的训练.** 可以使用任何基于 R-CNN 的 two-stage 目标检测模型进行搭建. 可以获得 consistent gains(2~4 points). 并且可以和其他各种 trick 叠加.

# Related Work

R-CNN, SPP-Net, Fast R-CNN, Faster R-CNN, RPN, R-FCN, MS-CNN, YOLO, SSD, RetinaNet

# Object Detection

在本文章, 我们将 Faster R-CNN 模型进行扩展, 如图3(a)所示, 第一阶段是一个 proposal sub-network (H0), 将其作用于整个图片, 会生成 **主要的检测假设(即 anchor box proposals)**. 在第二个阶段, 这些生成的候选框 (hypotheses) 会被一个 roi detection sub-network (H1) 处理, 我们将其记为 detection head. 最终, 分类 score (C) 和 bounding box (B) 会被分配到每一个候选区域框 (hypothesis)上. 本文主要是构建一种 multi-stage 的模型框架, 如图3(d)所示.

<div style="width: 550px; margin: auto">![图3](https://wx4.sinaimg.cn/large/d7b90c85ly1fx6jmz4bhtj21kw0egtdb.jpg)

## Bounding Box Regression

bounding box $b = (b_x, b_y, b_w, b_h)$ 包含了某个图像区域块 $x$ 的四个坐标. 边框回归(bbox regression)的任务就是利用回归器 $f(x, b)$ 将一个 box $b$ 回归对目标 box $g$ 上. 这是通过训练样本 $(g_i, b_i)$ 学习得到的, 以使得 L1 损失函数 $L_{loc} (f(x_i, b_i), g_i)$ 最小化. 为了保证尺寸不变性和位置不变性, $L_{loc}$ 通常是通过学习偏移量而不是直接学习坐标. **由于 bounding box 的偏移量都是归一化的, 所以数值都比较小, 因此, regression loss 通常会远小于 classification loss.**

可以从图1中看到, 在得到了 anchor boxes 后, Faster R-CNN 只进行了一次 box regression. 因此, 有一些工作认为单次的 box regression 是不够的, 故而提出了 **iterative bounding box regression**, 记为 **iterative BBox**, 用该方法作为后处理步骤来对 bbox 进行精化.

$$f'(x, b) = f\circ f\circ ... \circ f(x,b)$$

它的实现结构如图3(b)所示(因为是后处理操作, 所以只在 inference 阶段执行), **该方法中使用的所有的 head 都相同**. 这种方法, 忽略了两个问题: 第一, 对于具有更高 IoU 的输入来说, 较低的阈值 ($\mu=0.5$) 往往是一种次优解 (如图1所示); 第二, 如图2所示, bounding box 的分布在每次迭代后都会发生很大的变化, 但是固定的阈值使得模型每次更新时都是以初始分布为目标的. 基于这些问题, iterative BBox 需要大量的人工工程设计, 以 proposals 累积, box 投票等形式存在, 其收益也不太可靠. 通常, 应用两次 $f$ 以上并不会获得更多的收益.

<div style="width: 550px; margin: auto">![图2](https://wx1.sinaimg.cn/large/d7b90c85ly1fx6k6bghwsj20t10o2keq.jpg)

## Detection Quality

分类器 $h(x)$ 会给一个图片区域块(image patch) $x$ 分配一个类别(M+1 类, 0 代表背景). 给定训练集 $(x_i, y_i)$, 它将会通过最小化分类交叉熵 $L_{cls}(h(x_i), y_i)$ 来进行学习, 式中 $y_i$ 代表了 $x_i$ 的类别.
因为一个 bounding box 通常会包含一个目标物体和一些背景, 因此很难决定一个检测框是 positive 还是 negative 的. 通常我们利用 IoU 来解决这个问题. 当给定 IoU 一个阈值 $\mu$, 如果图片区域块(image patch)和真实框的 IoU 大于该阈值, 就认为他是正样本. 因此, 候选区域框(hypothesis)的标签可以看做是 $\mu$ 的函数.

$$y = \begin{cases} g_y, && IoU(x, g) \geq \mu \\ 0, && otherwise \end{cases}$$

上式中, $g_y$ 是真实目标 $g$ 的分类标签. 这个 IoU 阈值 $\mu$ 就定义了检测器的质量(quality of a detector).

目标检测问题具有挑战性的原因之一就是无论设置什么样的阈值, 检测设置都是具有高度对抗性的. 具体来说, 当 $\mu$ 较高时, 正样本会包含较少的背景, 但是这样又难以收集到足够的正样本进行训练. 当 $\mu$ 较低时, 可以获得更加丰富多样正样本训练集合, 但是检测器却难以拒绝不好分辨的假正例样本. 通常情况下, 我们很难要求单个分类器在所有 IoU 级别上都能很好的执行. 在 inference 阶段, 由于 proposal 检测器(RPN)生成的大多数候选框(hypotheses)具有较低的质量, 因此, 检测器必须对低质量的 hypotheses 具有更强的鉴别能力. 在这些相互冲突的需求之间, 一个标准的折衷方案是令 $\mu = 0.5$. 但是, 这相对来说是一个比较低的阈值, 会导致生成很多假正例检测结果, 如图1(a)所示.
一个简单解决方案就是将多个分类器集成, 如图3(c)所示. 但是该方法会使得分类器的性能过强而陷入过拟合状态


# Cascaded R-CNN

下面我们将介绍 Cascade R-CNN 目标检测模型, 如图3(d)所示.

## Cascaded Bounding Box Regression
如图1(c)所示, 我们很难令一个单一的回归器在所有的 quality levels(输入的框的 IoU 级别) 上获得完美的表现. **我们可以将较难的回归任务分解成一系列较小的步骤**, 在 Cascaded R-CNN 中, 我们将其组织成如图3(d)中的结构, 其依赖于一系列专门的回归函数, 写成公式表达如下:

$$f(x,b) = f_T \circ f_{T-1} \circ ... \circ$$

上式中, $T$ 是 cascade stages 的数量.
该式和图3(b)所示的 iterative BBox 方法有很多不同之处. 第一, iterative BBox 是一种用来提升 bounding boxes 的后处理步骤, 而 cascaded regression 是一种用于在不同 stages 改变假设分布的重采样过程. 第二, 由于会同时在训练和预测阶段使用 cascade 策略, 因此训练和预测阶段之间的分布没有差异性. 第三, 在不同的 stages 上, 会对重新采样后的分布会训练不同的回归器 $\{f_T, f_{T-1}, ..., f_1 \}$. 这些特点使得我们的模型可以产生更加精确的 BBox, 而不需要过多的 human engineering.

## Cascaded Detection

如图4所示, RPN 网络最初生成的假设分布更多的集中在 low quality 的box上, 这不可避免的会导致对高质量分类器的无效学习. Cascade R-CNN 通过将级联回归用作重采样机制来解决这个问题. 这是受到图1(c)的启发, 即在给定阈值下训练的回归器会生成更高质量的 bbox. 因此, 从 $(x_i, b_i)$ 开始, 级联回归会先后的进行重采样来获得更高的 IoU proposals. 这样一来, **即使检测器的质量(IoU阈值)升高了, 我们也可以将连续阶段中的正样本比例保持在大致恒定的大小**. 如图4所示, 在每一次重采样之后, 分布都会倾向于高质量(高IoU)的样本. 这会导致两个现象. **第一**, 因此在所有级别的 IoU 上都有大量的正样本, 因此不会出现过拟合. **第二**, 对较深阶段的检测器进行了优化, 使用具有较高的 IoU 阈值. 注意到, 通过增加 IoU 阈值, 一些异常的值将被逐渐删除, 如图2所示, 这样可以实现更好的训练有素的专用检测器序列.

在每一个阶段 $t$ 中, R-CNN 都包含一个分类器 $h_t$ 和一个回归器 $f_t$ (针对 $u^t$ 进行优化, $u^t > u^{t-1}$), 通过最小化下面的 loss 进行学习

$$L(x^t, g) = L_{cls} (h_t(x^t), y^t) + \lambda [y^t \geq 1] L_{loc}(f_t(x^t, b^t), g)$$

上式中, $b^t = f_{t-1} (x^{t-1}, b^{t-1})$, $g$ 是 $x^t$ 的真实物体标签, $\lambda = 1$ 是平衡系数(trade-off coefficient).

在 Inference 阶段, hypotheses 的质量会逐渐提高, 通过使用相同的 cascade procedure 以后, 更高质量的检测只需要作用在更高质量的 hypotheses 之上即可.

<div style="width: 550px; margin: auto">![图4](https://wx2.sinaimg.cn/large/d7b90c85ly1fx6k6fv7t4j20sz0dcdi0.jpg)

# Experimental Results

**实现细节:**
所有的 cascade detection stages 具有相同的结构(the head of the baseline network).
Cascade R-CNN 总共具有 **四个 stages**, 一个 RPN 以及三个 U = {0.5, 0.6, 0.7} 的 detection network.
第一阶段的 sampling 策略和 Faster R-CNN 中相同, 后面阶段的重采样使用的是前一阶段输出的结果.
除了标准的水平翻转以外, 没有使用其他的 data augmentation 技术.
Inference 是在 single image scale 上进行的.
End-to-end training

**Baseline Networks:** Faster R-CNN, R-FCN, FPN.
- Faster R-CNN: 该网络具有两层 fc 层. 为了降低参数量, 我们使用 "attend refine repeat" 来剪去不重要的链接. lr: 起始 0.002, 在 60k 和 90k 次迭代的时候降低 10 倍, 在 100k 次迭代的时候停止, 2 个 synchronized GPUs, 每一个持有 4 张图片, 每张图片提供 128 个 RoIs.
- R-FCN: 没有使用 OHEM. lr: 起始 0.003, 160k 和 240k 次迭代的时候降低 10 倍, 280k 次迭代的时候定值, 4 个 synchronized GPUs, 每一个持有一张图片, 每张图片提供 256 个 RoIs.
- FPN: 使用了 RoIAlign. lr: 0.005 用于最开始的 120k 次迭代, 0.0005 用于后面的 60k 次迭代, 8 个 synchronized GPUs, 每一个持有一张图片, 每张图片提供 256 RoIs.


## Quality Mismatch

下图5(a), 可以看出, 虚线始终位于对应实线的上方, 说明应用了 Cascade R-CNN 以后, 模型产生的框的 mAP 变高了. 从图(b)中可以看出, 当输入的框的 IoU 较大时(通过不断添加真实框来增大输入的 IoU 的大小), $\mu$ 值较大的检测器可以获得更高的 IoU.

<div style="width: 550px; margin: auto">![](https://wx3.sinaimg.cn/large/d7b90c85ly1fx7k7kyrtwj20u10j6dwq.jpg)

图6显示了所有的 Cascade R-CNN 检测器在所有 stage 上面的 mAP值, 从图6可以看出, 经过 Cascade R-CNN 以后, 输入的框的 IoU提升了, 是的阈值为 $\mu = 0.7$ 的检测器的 mAP 提升了, 不仅如此, 我们还可以看到, 在经过 Cascade R-CNN 以后, 即使是对具有更高 IoU 的输入, $\mu =0.5$ 的检测器也比 stage-1 阶段的 mAP值高, 这说明本文提出的 Cascade R-CNN 框架可以有效的提升检测器的性能.

<div style="width: 550px; margin: auto">![](https://wx3.sinaimg.cn/large/d7b90c85ly1fx7k902012j213k0ezan4.jpg)

在图7(a)中, 我们将本文的 Cascade R-CNN 与 Iterative Box 进行了对比, 在图1中, 我们可以看出, 使用单个回归器不断迭代的方式会降低输出的 IoU 大小. 相反, 使用本文的 Cascade R-CNN 方法, 可以在新的 stage 中生成更高的 IoU.
在图7(b)中, 使用同一个检测器, 但是赋予不同的 $\mu$ 值时, 当 $\mu=0.6$ 时 mAP 最高, 当 $\mu=0.7$ 时 mAP最高, 而融合吼的模型结果也没有获得较大的提升.

<div style="width: 550px; margin: auto">![](https://wx4.sinaimg.cn/large/d7b90c85ly1fx7kbcth3sj212d0kwne0.jpg)

从表1可以看出, Iterative BBox 和 intergral loss 检测器相对于 baseline 方法都可以提升模型的精度, 但是本文的 Cascade R-CNN 具有最好的精度表现.

<div style="width: 550px; margin: auto">![](https://wx2.sinaimg.cn/large/d7b90c85ly1fx7kbg8eifj21i40i5jve.jpg)

**消融实验**

**Stage-wise Comparison:** 表2总结了每个 stage 的性能表现, 注意到, stage-1 已经超过了 baseline detector, 这是因为经过 multi-stage 学习后, stage-1 的检测能力也得到了一定的提升. 总体趋势显示越深的 cascade stage 具有越高的 quality localization. (但是考虑到模型复杂度和训练难度问题, 也不能叠加太多 stage, 一般2,3层差不多)

<div style="width: 550px; margin: auto">![](https://wx1.sinaimg.cn/large/d7b90c85ly1fx7ktgiye6j20so0elwhn.jpg)

**IoU Thresholds:** 表3前两行显示, 相对于每一个阶段使用固定的 IoU 阈值 (如0.5), 采用递增式的 IoU 阈值可以获得更好的效果 (对于 close false positives 更具有选择性). 但是同样的, 即使使用相同的 IoU 阈值来训练每一个阶段, 也比 baseline 的 mAP 高.

**Regression Statistics:** 表3第一行和第三行对了使用和不使用 sequential regression statictics 时的模型性能差异.

<div style="width: 550px; margin: auto">![](https://wx2.sinaimg.cn/large/d7b90c85ly1fx7ktleytvj20sc0afjtj.jpg)

表4总结了 stages 的个数对模型的影响. 添加两层 stages 可以大幅度提升 baseline 的精度, 第三层可以小幅度的的提升模型精度, 但是当叠加到第4层时, 模型精度就会收到一定影响并有略微下降.(尽管如此, 具有4个 stages 的检测器在较高的 IoU (AP90)下可以取得最好的精度表现)

<div style="width: 550px; margin: auto">![](https://wx2.sinaimg.cn/large/d7b90c85ly1fx7ktozgt2j20ru0ad0uw.jpg)

**Comparison with the state-of-the-art:**

表5显示了本文的 Cascade R-CNN 模型与现有模型之间的性能对比.

<div style="width: 550px; margin: auto">![](https://wx3.sinaimg.cn/large/d7b90c85ly1fx7ll0o1z5j21kw0ga45p.jpg)

表6显示了在不同的 baseline 模型上应用 Cascade R-CNN 之后的性能表现

<div style="width: 550px; margin: auto">![](https://wx4.sinaimg.cn/large/d7b90c85ly1fx7lsmvzv0j21kw0i8dnv.jpg)

表7显示在不同的 backbone 网络中, Cascade R-CNN 仍然能够大幅度提升模型的 mAP

<div style="width: 550px; margin: auto">![](https://wx2.sinaimg.cn/large/d7b90c85ly1fx7lt5meqlj21060dwacy.jpg)
