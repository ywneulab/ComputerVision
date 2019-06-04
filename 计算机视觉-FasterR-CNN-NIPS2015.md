---
title: Faster R-CNN (NIPS, 2015)
sitemap: true
date: 2018-04-19 19:27:03
categories: 计算机视觉
tags:
- 计算机视觉
- 目标检测
- 论文解读
---
**文章:** Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks
**作者:** Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun

# 背景介绍

# 核心亮点

# 摘要

在本文中, 我们提出了一个候选区域框推荐网络(Region Proposal Network, RPN), 该网络可以与检测模型的网络共享图片的卷积计算结果. PRN 是一个可以在每一个位置上同时预测物体边界(object bounds)以及前后景置信度(objectness scores)的全卷积网络. RPN 可以通过端到端(Faster R-CNN 由 RPN 和 Fast R-CNN 组成)的训练来生成高质量的候选框区域, 而这些区域可以提供给 Fast R-CNN 网络来进行目标检测任务. 更进一步地, 本文通过共享卷积特征的方法将 RPN 和 Fast R-CNN 网络融合成了一个单独的网络(RPN 的作用类似于一种注意了机制, 即告诉 Fast R-CNN 应该对哪里进行检测). 当使用深度卷积网络 VGG-16 时, 我们的检测模型可以在 GPU 上达到 5fps 的检测帧率, 同时在 VOC 2007, 2012 和 MS COCO 数据集上达到了 state-of-the-art 的精确度.

# 介绍

Fast R-CNN 目标检测模型取得了很大的成功, 但是它使用的候选区域框仍然是通过 Selective Search 或者 EdgeBoxes 算法来获得的, 这种方式是在 CPU 上运行的, 相对来说比较慢. 因此, 一个明显的思路就是利用 GPU 来加速候选区域框的生成. 在本文中, 我们提出了一个新的深度网络模型, 可以代替 Selective Search 算法来为 Fast R-CNN 提供候选区域框, 并且与 Fast R-CNN 通过共享卷积计算结果, RPN 可以节省大量的计算成本.
我们观察到, 用于检测目标的卷积特征图谱, 同样可以用来生成候选区域框, 因此, 我们在 backbone 卷积网络之上额外添加一些卷积层来构成 RPN 网络, RPN 网络和 Fast R-CNN 网络的分类器和回归器是并列关系.

**注意, RPN 网络和 Fast R-CNN 网络各自可以进行端到端的训练, 但是 Faster R-CNN 网络并非是端到端的, 它需要结合 RPN 和 Fast R-CNN 网络才可以工作***

![图1](https://wx3.sinaimg.cn/large/d7b90c85ly1fxseevwsqkj21n50hctlj.jpg)

如图1(a)和(b)所示, 目前主流的候选框生成算法都是通过 "图片金字塔" 或者 "过滤器金字塔" 来生成候选框的, 而 RPN 网络通过引入一种新颖的 "anchor" boxes 来达到生成不同尺寸和比例的候选框. 为了融合 RPNs 和 Fast R-CNN 物体检测模型, 我们提出了交替 fine-tuning 候选框生成任务和目标检测任务的训练策略, 在训练其中一个网络时, 将另一个网络固定.

# Related Work

**Object Proposals:** Selective Search, EdgeBoxes.

**Deep Networks for Object Detection:** R-CNN, OverFeat, MultiBox

# Faster R-CNN

本文将提升的模型命名为 Faster R-CNN, 它主要由两部分组成. 其一是用于生成候选区域框的深度全卷积网络, 其二是 Fast R-CNN 检测模型. 整体的检测系统是一个统一的目标检测网络, 如图2所示.

![图2](https://wx1.sinaimg.cn/large/d7b90c85ly1fxsf9g1udsj21q00s3152.jpg)

# Region Proposals Networks

RPN 网络会将一张任意尺寸的图片作为输入, 同时会输出一系列的矩形候选框, 每一个候选框都带有一个前后景置信度(objectness score). 因为我们希望 RPN 网络和 Fast R-CNN 网络能够共享卷积计算结果, 因此我们假设两个网络中的卷积层是共享的. 在我们的实验中, 我们研究分析了 ZF Net(拥有5个可共享的卷积层)和 VGG-16 Net(拥有13层可共享的卷积层).
为了生成候选区域框, 我们在最后一层共享卷积层的特征图谱上添加了一个小型的网络. 这个小型网络会接受 $n\times n$ 大小的特征图谱上的窗格. 每一个滑动的窗口都可以映射到更低维度的特征(256-d for ZF, 512-d for VGG, with ReLU, following). 这个特征会被送入到两个并行的全连接层: 一个边框回归层(reg)和一个分类层(cls). 本文中我们使用 $n=3$, 注意到在输入图片上的有效感受野非常大(171 and 228 pixels for ZF and VGG, respectively). 这个小型网络在图3中的左侧单独展示. 请注意, 因为 mini-network 是以滑动窗口的形式进行工作的, 因此两个全连接层可以在所有位置上共享. 上面的小型网络可以很自然的用一个 $n\times n$ 的卷积层后接两个并行的 $1\times 1$ 的卷积层(for reg and cls, respectively)实现

实际上, RPN网络由两部分构成: 一个卷积层, 一对全连接层分别输出分类结果(cls layer)以及坐标回归结果(reg layer). 利用了一个卷积核大小为 $n\times n$ 的卷积层, 后接两个 $1\times 1$ 的卷积层(分别用于回归和分类)实现.

![图3](https://wx1.sinaimg.cn/large/d7b90c85ly1fxseqf6a69j21fa0jpqpj.jpg)

我们以 ZF model(具有5层卷积层) 为例对RPN网络进行讲解, ZF model的最后一层卷积层Conv5具有256个卷积核, 也就是说它生成的特征图谱的shape为 $W\times H\times 256$, RPN网络的第一层可以认为是一个卷积层, 卷积核的尺寸为 $n\times n$ (文中使用的是 3×3), 卷积核的个数为 256 (维持通道数不变). 利用该层的卷积层对 Conv5 的特征图谱操作, 最终生成的特征图谱大小仍然为 $W\times H\times 256$ (只不过此时,图谱中的每个点都会与原图谱中的 $k\times k$ 个点相关联). 对于这个图谱中的每个点, 我们认为它是一个 anchor, 可以将它看做是一个元素个数为256的1维向量 (因为通道数为256).
然后, 对于每一个anchor, 都会分配 $k$ 个anchor boxes. 每个anchor box都要分前景后景, 也就是分是否包含物体, 同时, 每一个anchor box还要输出预测物体相对于anchor的偏移坐标量.

训练的时候会选择128个postive anchor boxes 和128个negative anchor boxes.

![](https://wx2.sinaimg.cn/mw1024/d7b90c85ly1fw9tvpr5doj20xz0lk1kx.jpg)
## Anchors

在每一个 sliding-window 的 location 上, 我们会同时预测多个 region proposals, 每个 location 上可以预测的 proposals 的最大数量为 $k$. 因此 reg layer 将会有 $4k$ 个输出, 表示 $k$ 个 box 的坐标, 同时 cls layer 将会有 $2k$ 个 scores, 用来估计每个 proposal 中是否含有物体的概率(为简单起见, cls layer 使用二分类的 softmax layer 实现, 也可以选择用 logistic regression 实现). $k$ 个 proposals 的具体参数是和 $k$ 参考框有关的, 我们称这些参考框为 anchors.  anchor 处于滑动窗口的中心位置, 并且会带有一个放缩比例(scale)和宽高比(aspect ratio), 如图3左侧所示. 在默认情况下, 我们使用3个 scales 和3个 aspect ratios, 这样, 每个 sliding position 总共会生成9个anchors. 因此, 对于一张 $W\times H$ 大小的特征图谱, 总共会生成 $WHk$ 个 anchors.

**Translation-Invariant Anchors**
本文方法的一个重要性质就是它具有 **平移不变性(translation invariant)**, 不论是在 anchors 方面还是在根据 anchors 计算 proposals 方面都具有此特性. 无论物体被移动到图片中的哪里, 我们的方法都可以生成与之对应的相同的 proposals.
这种平移不变性还可以降低模型的大小. MultiBox 的全连接层维度为 $(4+1)\times 800$, 而我们的维度为 $(4+2)\times 9$. 最终, 我们输出层的参数量为 $512\times (4+2)\times 9 = 2.8 \times 10^4$ (for VGG-16), 比 MultiBox 的参数量低了两个数量级 $1536 \times (4+1) \times 800 = 6.1\times 10^6$. 如果考虑到 feature projection layers, 我们的 proposal layers 仍然具有更少的参数量, 因此, 理论上我们的模型在训练过程中发生过度拟合的风险更低.

**Multi-Scale Anchors as Regression References**

anchors 的设计提出了一种新的解决目标多尺度问题的方法. 如图1所示, 目前有两中主流的多尺度目标预测方法. 第一种方法是基于 image/feature pyramid, 如 DPM 和基于 CNN (SPPNet, Fast R-CNN, Overfeat)的方法. 图片会被放缩到不同的尺度, feature map 会根据特定尺度进行计算(如图1(a)所示). 这种方法很有用, 但是很耗时. 第二种方法是在 feature map 上使用多尺度的 sliding windows. 举例来说, 就是对于不同的尺度, 我们会使用不同的 filter sizes 来分别进行训练, 这也可以看做是 pyramid of filters (如图1(b)所示).第二种方法经常会结合第一种方法使用.
与上面两种方法相比, 我们的基于 anchor 的方法是建立在 pyramid of anchors 之上的, 这种方法的成本收益较高. 我们方法的分类器和回归器都和多尺度(scales, aspect ratios)的 anchor boxes 相关. 它仅仅依赖于单一尺度的图片和特征图谱, 同时也只是用单一大小的 filters (sliding windows on the feature map). 我们在表8中展示了我们的模型在解决多尺度问题上的有效性.
由于这种多尺度解决方案是建立在 anchors 之上的, 因此我们可以简单的使用卷积层来计算单一尺度的图片, 正如 Fast R-CNN 中的一样. 这种多尺度 anchors 的设计对于共享特征(sharing features without extra cost for addressing scales)来说是非常关键的一点.


综上, RPN网络做的事情就是，如果一个Region的 $p\geq 0.5$ ，则认为这个Region中可能是80个类别中的某一类，具体是哪一类现在还不清楚。到此为止，Network只需要把这些可能含有物体的区域选取出来就可以了，这些被选取出来的Region又叫做ROI （Region of Interests），即感兴趣的区域。当然了，RPN 同时也会在 feature map 上框定这些ROI感兴趣区域的大致位置，即输出Bounding-box。

好的，到此为止，RPN网络的工作就完成了，即我们现在得到的有：在输入RPN网络的 feature map 上，所有可能包含80类物体的Region区域的信息，其他Region（非常多）我们可以直接不考虑了（不用输入后续网络）。接下来的工作就很简单了，假设输入RPN网络的feature map大小为 $64\times 64$，那么我们提取的ROI的尺寸一定小于 $64 \times 64$ ，因为原始图像某一块的物体在feature map上也以同样的比例存在。我们只需要把这些Region从feature map上抠出来，由于每个Region的尺寸可能不一样，因为原始图像上物体大小不一样，所以我们需要将这些抠出来的Region想办法resize到相同的尺寸，这一步方法很多（Pooling或者Interpolation，一般采用Pooling，因为反向传播时求导方便）。假设这些抠出来的ROI Region被我们resize到了 $14\times 14$ 或者 $7\times 7$，那我们接下来将这些Region输入普通的分类网络，即第一张Faster R-CNN的结构图中最上面的部分，即可得到整个网络最终的输出classification，这里的class（车、人、狗。。）才真正对应了COCO数据集80类中的具体类别。同时，**由于我们之前RPN确定的box\region坐标比较粗略**，即大概框出了感兴趣的区域，所以这里我们再来一次精确的微调，根据每个box中的具体内容微微调整一下这个box的坐标，即输出第一张图中右上方的Bounding-box regression。我们网络输出的两个Bounding-box regression，都是输出的坐标偏移量，也就是在初始锚点的基础上做的偏移修正和缩放，并非输出一个原图上的绝对坐标。

## Region Proposal有什么作用？
1、COCO数据集上总共只有80类物体，如果不进行Region Proposal，即网络最后的classification是对所有anchor框定的Region进行识别分类，会严重拖累网络的分类性能，难以收敛。原因在于，存在过多的不包含任何有用的类别（80类之外的，例如各种各样的天空、草地、水泥墙、玻璃反射等等）的Region输入分类网络，而这些无用的Region占了所有Region的很大比例。换句话说，这些Region数量庞大，却并不能为softmax分类器带来有用的性能提升（因为无论怎么预测，其类别都是背景，对于主体的80类没有贡献）。2、大量无用的Region都需要单独进入分类网络，而分类网络由几层卷积层和最后一层全连接层组成，参数众多，十分耗费计算时间，Faster R-CNN本来就不能做到实时，这下更慢了

## 损失函数

为了训练 RPNs, 我们给每个 anchor 都赋予了一个 binary class label. 我们会给两种类型的 anchors 赋予正样本标签: (1), 和真实框具有最大交并比(不一定大于0.7)的 anchor/anchors; (2), 和某一个真实框的交并比大于0.7. **注意, 一个真实框可以被赋值给多个 anchors**. 通常情况下, 第二种情况就已经足够用来决定正样本了, 我们利用第一种情况的原因是在某些极端情况下有可能第二种情况找不到正样本. 如果某一个 anchor 和所有的真实框的交并比都低于0.3, 那么我们就对其赋予负样本标签. 剩下就不是正样本也不是负样本的 anchors 不参与训练过程.
通过以上的定义, 我们可以根据 Fast R-CNN 中的多任务损失函数确定训练函数如下:

$$L(\{p_i\}, \{t_i\}) = \frac{1}{N_{cls}} \sum_i L_{cls}(p_i, p_i^* ) + \lambda \frac{1}{N_{reg}} \sum_i p_i^* L_{reg}(t_i, t_i^* )$$

上式中, $i$ 代表 mini-batch 中 anchor 的下标, $p_i$ 代表预测 anchor $i$ 是一个物体的可能性大小. 如果 anchor 是正样本, 则真实标签 $p^\*_i$ 为1, 反之, 如果是负样本, 则为0. $t_i$ 是一个 vector, 用来表示参数化以后的边框坐标, $t^\*_i$ 是正样本 anchor 对应的真实框的参数化坐标. 分类损失函数 **$L_{cls}$ 是一个二分类 log 损失**. 对于回归损失, 我们使用 $L_{reg}(t_i, t^\*_i) = R(t_i - t^\*_i)$, 这里 $R$ 代表 robust loss function(smooth L1). $p^\*_i L_{reg}$ 代表回归损失仅仅会被正样本的 anchor 所激活.
$N_{cls}$ 和 $N_{reg}$ 是两个归一项, 同时 $lambda$ 会调节每一项的权重. 在本文的发行版代码中, $N_{cls}$ 设定为 mini-batch 的大小(256), $N_{reg}$ 设定为 anchor locations 的大小(约2400). 同时 $lambda$ 默认为10, 这样, $cls$ 和 $reg$ 所占的比重大致相等. 我们在表9中给出了 $lambda$ 值在很大范围内对结果的影响并不敏感(这是比较好的, 说明我们不需要过度调整该超参的值). 同时, 我们发现归一化也不是必须的, 可以被简化.

## 训练 RPNs
 RPN 可以利用反向传播和 SGD 进行端到端的训练(注意这里是指 RPN, 而不是 Faster R-CNN). 我们依照 "image-centric" 的简单策略来训练网络. 每一个 mini-batch 都从一张单一的图片中得到, 其中包含了许多正样本和负样本. 我们可以对所有的 anchors 进行训练, 但是这样就会导致预测结果偏向于负样本, 因为负样本的数量占据绝对地位. 因此, 我们选择从 **一张图片** 中随机的挑选 256 个 anchors 来组成一个 mini-batch, 其中 **正负样本的比例为 1:1**, 如果图片中的正样本数量不足128, 那么就用负样本补足.
 对于新添加的所有网络层, 我们都是用均值方差为(0, 0.01)的高斯分布来初始化, 其它层使用在 ImageNet 上预训练后的权重进行初始化.

# RPN 和 Fast R-CNN共享卷积参数

到目前为止我们已经讨论过如何训练网络来进行 region proposal generation, 但是还没有考虑如何让 region-bases object detection CNN 来使用这些 proposals. 对于目标检测网络, 我们使用 Fast R-CNN 进行预测. 接下来我们将会介绍由 RPN 和 Fast 组成的参数共享的统一网络的算法训练流程.
如果 RPN 和 Fast R-CNN 单独训练的话, 那么就会以不同的方式改变他们的卷积层参数. 因此, 我们需要提出一种新的训练策略使得可以在这两个网络之间的参数共享. 为了使RPN和FastRCNN共享卷积参数, 我们讨论了三种不同的训练策略:
- Alternating training: 先训练RPN, 然后用 RPN 产生的候选区域来训练Fast RCNN, 之后, FastRCNN 更新参数以后, 继续用来训练RPN, 这个过程是迭代进行的. 该策略是本篇 paper 中所有实验使用的训练方法
- Approximate joint training: 在该策略中, RPN 网络和 Fast R-CNN 网络在训练阶段会被合并到一个网络中去(如图2所示). 每一次 SGD 迭代中, 前向计算过程都会产生 region proposals, 这些 proposals 在训练 Fast R-CNN detector 时被看做是固定的, 预计算好的. 在反向传播过程中, 会将 RPN loss 和 Fast R-CNN 的损失结合计算. 这种策略实现起来很容易, **但是这种策略忽略了对于 proposals boxes 坐标的导数**, 因此这只是一种粗略的联合训练方式. 在我们的实验中, 我们发现这种方法可以取得与策略一接近的结果, 但是可以 **令训练时间降低25%~50%**. 该策略被包含在我们发布的Python版本的发行版代码中.
- Non-approximate joint training: 正如上面讨论的, RPN 预测的 bounding boxes 的坐标同样与输入数据之间存在联系. 在 Fast R-CNN 的 RoI pooling Layer 中会接受卷积特征, 同时也会将预测的 bounding boxes 作为输入, 因此理论上来说, 一个有效的优化器(backpropagation solver)应该包含相对于 box coordinate 的梯度. 因此, 我们需要 RoI pooling layer 相对于 box coordinates 是可导的. 这是一个 nontrivial 的问题, 并且它的解决方案在 RoI warping Layer (何恺明的另一篇paper, Instance-aware semantic segmentation via multi-task network cascades)给出, 但是这一部分超出了本论文想说明的事情, 故不做详细描述.

**4-Step Alternating Training**
在本篇文章中, 我们采用了一种实用的四步训练算法通过 alternating optimization 来学习共享的特征. 第一步, 我们按照前文提到的训练 RPN 的方法对其进行训练. 这个网络使用 ImageNet 预训练的模型进行初始化并且端到端的对 region proposal 任务进行 fine-tune. 第二步, 我们使用第一步生成的 proposals 来训练一个单独的 Fast R-CNN 网络. 这个检测网络同样也是用 ImageNet 预训练的模型进行初始化. **到这里为止, 这两个网络还没有共享卷积层.** 第三步, 我们使用检测网络来对 RPN 网络的训练过程进行初始化, 但是此时 **我们固定住共享的卷积层, 仅仅 fine-tuning 属于 RPN 独有的那些网络层**.  到这里, 两个网络已经共享的卷积层的参数. 第四步, **同样固定住共享的卷积层, 仅仅 fine-tuning 属于 Fast R-CNN 独有的网络层**. 这样一来, 两个网络都共享了同样的卷积层, 并且组成一个统一的网络. 类似的交替训练可以用于更多的迭代.


# Implementation Details

训练和预测 RPN 和 detection networks 都是在单一的图片尺寸下进行的. 我们将图片进行放缩, 使其较短边长度为 $s = 600$ pixels. Multi-scale feature extraction(图片金字塔)也许可以提升精度, 但是无法保证检测速度. 在放缩后的图片上, ZFNet 和 VGG net 到最后一层卷积层的总共的 stride 为 16. 这样大的步长依然可以取得较好的效果, 如果使用较小的 stride, 也许精度更高.
对于 anchors, 我们使用三种 scales: $128^2$, $256^2$, $512^2$, 和三种 aspect ratios: 1:1, 1:2, 2:1. 这种设定并非是精心设计的, 我们的实验证明了我们方法在很大范围都有效. 表1给出了 ZF Net 学习到的每个 anchor 的平均 proposal size.

![表1](https://wx1.sinaimg.cn/large/d7b90c85ly1fxserec80rj219w0700tt.jpg)

那些超出图片边界的 anchor 需要特别处理, 在训练阶段, 我们忽略了所有越界的 anchors, 因此它们不会参与训练. 对于一张经典的 $1000\times 600$ 大小的图片来说, 总共将会有大约 20000($60\times 40 \times 9$) 的 anchors, 忽略掉越界的 anchors 以后, 大约有 6000 个 anchors 可以参与训练. 如果越界的 anchors 没有被忽略, 它们将会带来大量的, 难以纠正的错误项, 这会使得训练过程难以收敛. 而在预测阶段, 我们会将全卷积的 RPN 应用到整张图片中, 这也许会产生越界的 proposal boxes, 但是我们会对其进行剪裁处理, 而不是忽略.
有一些 RPN proposals 之间会有很高的 overlap. 为了减少冗余, 我们会对 proposal regions 采用基于 scores 的 NMS 算法, 最终会留下大约 2000 个proposal regions. 在 NMS 之后, 我们将会使用 top-N proposal regions 进行 detection.

# Experiments

![表2](https://wx4.sinaimg.cn/large/d7b90c85ly1g0uk9mqgcvj21890hxjwo.jpg)

![表3](https://wx3.sinaimg.cn/large/d7b90c85ly1fxses5eo8xj21hx0dpq6y.jpg)

![表4](https://wx2.sinaimg.cn/large/d7b90c85ly1fxsese9ppjj21h00bz0wn.jpg)

![表5](https://wx1.sinaimg.cn/large/d7b90c85ly1fxseso62vaj21gg08rjtv.jpg)

![表6](https://wx2.sinaimg.cn/large/d7b90c85ly1fxseszi56sj21fp0bhad9.jpg)

![表7](https://wx4.sinaimg.cn/large/d7b90c85ly1fxsetia0l7j21fi0ahacv.jpg)

![表8](https://wx4.sinaimg.cn/large/d7b90c85ly1fxseupadmlj20xz0g117h.jpg)

![表9](https://wx3.sinaimg.cn/large/d7b90c85ly1fxsevorhqej20v70907cs.jpg)

![图4](https://wx3.sinaimg.cn/large/d7b90c85ly1fxsevx06smj217m0cpq5n.jpg)

![表10](https://wx4.sinaimg.cn/large/d7b90c85ly1fxsewgfnp5j21h008rtbh.jpg)

![表11](https://wx1.sinaimg.cn/large/d7b90c85ly1fxsewwg162j21eh09j76v.jpg)

![表12](https://wx3.sinaimg.cn/large/d7b90c85ly1fxsexjkme6j20ug0ewgxe.jpg)


# 如果anchor box有两个物体重叠了?  怎么处理????
