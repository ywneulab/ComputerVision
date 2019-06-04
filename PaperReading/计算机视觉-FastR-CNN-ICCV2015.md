---
title: Fast R-CNN (ICCV, 2015)
sitemap: true
date: 2018-04-09 19:27:03
categories: 计算机视觉
tags:
- 计算机视觉
- 目标检测
- 论文解读
---

**文章:** Fast R-CNN
**作者:** Ross Girshick

# 核心亮点

# 摘要

本篇文章提出了一个用于解决目标检测问题的 Fast Region-based Convolutional Network (Fast R-CNN). Fast R-CNN 是基于之前的 R-CNN 提出的. 相比于之前的工作, Fast R-CNN 利用一些新颖的方法来提升训练和推演时的速度, 同时还提高了模型的准确率. 使用 VGG16 的 Fast R-CNN 在训练时的速度是 R-CNN 的9倍, 是 SPP-Net 的3倍, 在推演时的速度是 R-CNN 的213倍, 是 SPP-Net 的10多倍.

# 介绍

R-CNN 是近年来较为成功的目标检测模型之一, 但是, 它却存在着一些明显的缺点:
- **训练过程是分阶段的(Training is a multi-stage pipeline):** 在物体候选框上训练 CNN 网络, 然后再在 ConvNet features 上面训练 SVMs 分类器, 最后在训练一个边框回归器来对候选框的位置修正
- **Training is expensive in space and time:** 对于 SVM 和 bounding-box 回归器训练来说, 我们需要分别计算每个候选框的 feature maps, 这造成了大量的重复计算, 会消耗掉大量的存储空间.
- **目标检测速度太慢(Object detection is slow):** 在推演阶段, 也会对图片上的每个候选框分别计算其 feature maps, 大量重复计算是的网络模型的计算效率很低.

从上面的分析可以看出, R-CNN 速度慢的原因在于它需要在每个候选区域框上进行卷积计算, 这会造成大量的重复计算. SPP Net 通过共享卷积计算图谱的方式提高了 R-CNN 的速度. 但是 SPP Net 同样也具有很多缺点:
- **训练过程是分阶段的(Training is a multi-stage pipeline):** 训练的 pipeline 和 R-CNN 差不多, 都是多阶段的
- **无法 Fine-Tuning 金字塔池化层之前的卷积层:** 由于 SPPNet 提出的 fine-tuning 算法不能更新 spatial pyramid pooling 层之前的卷积层, 这使得它的准确率有待提高. (不能更新的原因是 SPPNet 的 mini-batch 选择策略和 R-CNN 是相同的, 这使得计算的复杂度很高, 详细解释可看下文).

本文主要的贡献点有以下几点:
- 更高的检测准确率(mAP)
- 整个训练过程更加统一(利用多目标损失函数)
- 训练时可以对所有网络层参数进行更新(相比于SPPNet)
- 无需在硬盘上额外存储 feature.(相比于 R-CNN, 因为共享卷积计算结果, 使得feature的体积大大降低)


# Fast R-CNN architecture and training

![](https://wx2.sinaimg.cn/large/d7b90c85ly1fxq312xytnj20wp0k8tj7.jpg)

如图1所示为 Fast R-CNN 的结构. Fast R-CNN 网络的输入是一张图片和一系列的物体候选框. 网络首先会对整张图片进行卷积计算来得到卷积特征图谱(conv feature map). 然后, 对于每一个物体候选框, 都会通过 RoI pooling 层从卷积特征图谱上提取固定长度的 feature vector. 将这些还有物体区域特征的 feature vector 送入一系列的全连接层, 最终会分别送到两个不同的分支, 一个用于生成类别的置信度, 另一个用于计算每个物体边框区域的坐标.

# RoI Pooling Layer

RoI Pooling Layer 使用 max pooling 来将 **任意尺寸** 的有效感兴趣区域中的特征转换成一个具有 **固定尺寸** $H\times W (e.g., 7\times 7)$的较小的 feature map, 这里的 $H$ 和 $W$ 是超参数. 在本文中, 一个 RoI(感兴趣区域)就是 feature map 上面的一个矩形窗口. 每一个 RoI 都通过四元组 $(r,c,h,w)$ 来表示(top-left corner, and its height and width).

RoI Pooling的前向传播过程如下:

![](https://wx1.sinaimg.cn/large/d7b90c85ly1fw9pzrhzpjj20ny078747.jpg)

对于任意给定尺寸为 $h\times w$ 的feature map的 RoI 窗口, 将其划分成 $W\times H$ 的网格大小(上图中的示例为 $W\times H= 3\times 3$ ), 这样, 每一个网格 cell 中的尺寸大约为 $h/H \times w/W$, 然后我们在网格 cell 中执行max pooling操作. 和标准的 max pooling 相同, RoI pooling 在卷积图谱上的各个通道之间是独立计算的. 这样, 对于任意size的输入, 都可以获得固定长度的输出. 可以看出, **RoI layer 实际上是 spatial pyramid pooling layer 中的一个特例, 即只有一个 pyramid level.** (但是相比于金字塔池化, RoI 池化可以确定固定大小的 pooling 窗口, 这使得我们可以更新池化层之前的网络层参数, 进而提高准确率)

# 利用预训练网络初始化参数

本文使用了三种不同的在 ImageNet 上预训练的网络结构, 每一种都具有5层池化层, 以及5~13层的卷积层. 在利用预训练模型初始化 Fast R-CNN 网络时, 网络结构会发生以下变化:
1. 首先用 RoI pooling 替代网络的最后一层 max pooling. 需要设置超参数 $H$ 和 $W$ (e.g. $H=W=7$ for VGG16)
2. 网络的最后两层全连接层和 softmax 层会被两个并列的网络层替代, 即分类层(fc,  K+1softmax)和 bounding-box 回归层
3. 网络的输入除了接收图片外, 还要接收每张图片中含有的一系列 RoIs(特征图谱上的感兴趣区域).

# Fine-tuning for detection

Fast R-CNN 的一个重要特性就是可以对模型中的所有参数进行更新. 首先, 我们来说明一下为什么 SPPNet 不能更新 spatial pyramid pooling layer 之间的网络权重.
最根本的原因是当每一个 RoI 训练样本来自于不同的图片时, 在 SPP layer 上进行反向传播时的效率非常低(highly inefficient), 而这恰恰是 R-CNN 和 SPPnet 网络的训练方式(即从不同图片获取 RoI).  这种低效性源自于每一个 RoI 都可能具有非常大的感受野(receptive field), 通常会覆盖整个输入图片(often spanning the entire input image). 因此前向传播时必须处理整个感受野, 因此训练时的输入会非常大(通常回事整张图片).
本文提出了一个更加高效的训练方式, 它可以在训练阶段有效利用特征共享的优势. 在 Fast R-CNN 中, SGD 的 mini-batches 是分层次采样的(sampled hierarchically), 首先会采样出 N 个图片, 然后会在每张图片中采样 R/N 个 RoIs. 关键性的一点是, 从同一张图片中得到的 RoIs 会在前向传播和反向传播的过程中共享卷积计算结果和内存. 如果我们令 N 很小, 就可以降低 mini-batch 的计算复杂度. 例如, 我们令 $N=2, R=128$, 此时的训练策略就会比使用 128 张不同的图片(R-CNN 和 SPPNet 的训练策略)计算速度快 64 倍.
这种加速策略的一点顾虑就是由于同一张图片中的不同 RoIs 之间是有关联性的, 因此这会降低模型训练的收敛速度, 但是, 通过实验表明, 这种顾虑并不会在实际使用中出现, 并且我们利用 $N=2, R=128$ 的参数设置取得了更快的收敛效果(比 R-CNN 的收敛迭代次数少).
除了使用层次采样外, Fast R-CNN 还使用了流水线式的训练过程(streamlined training process), 通过联合训练 softmax 分类器和 bounding box 回归器, FastRCNN 可以更加统一的进行训练(多目标联合训练).
接下来我们会对以上这些关键部分进行详细介绍

# Multi-task loss

Fast R-CNN 网络拥有两个并列的输出层. 第一个输出层是离散的概率分布预测层(per RoI, over K+1 categories), $p=(p_0, ..., p_K)$. 第二个输出层用于回归预测 bounding-box 的坐标偏移量, $t^k = (t_x^k, t_y^k, t_w^k, t_h^k)$, 每一个物体都具有 $k$ 个边框预测结果, 对应着 $K$ 个物体类别.
每一个训练样本 RoI 都会用真实类别标签 $u$ 和真实 bounding-box 回归目标 $v$ 标记. 我们会在每一个标记好的 RoI 样本上计算联合任务损失函数 $L$ 如下所示:

$$L(p, u, t_u, v) = L_{cls}(p,u) + \lambda [u \geq 1] L_{loc}(t^u, v) \tag 1$$

上式中, $L_{cls}(p,u) = - log p_u$, 即对于真实类别 $u$ 的 log 损失.
$L_{loc}$ 被定义为相对于真实类别 $u$ 的边框回归损失, 即不计算其他类别的边框损失, 也不计算背景的边框损失. 对于 bounding-box 归回, 我们使用下面的损失:

$$L_{loc}(t^u, v) = \sum_{i\in {x,y,w,h}} smooth_{L_1}(t_i^u - v_i) \tag 2$$

上式中, smooth L1 损失被定义为:

$$smooth_{L_1}(x) = \begin{cases} 0.5x^2 && |x|<1 \\ |x| - 0.5 && otherwise \end{cases} \tag 3$$

smooth L1 损失是一种鲁棒性较强的 L1 损失, 相比于 R-CNN 和 SPPNet 中使用的 L2损失, 它对离异点的敏感度更低. 当回归目标趋于无限时, L2 损失需要很小心的处理学习率的设置以避免发生梯度爆炸, 而 smooth L1 损失则会消除这种敏感情况.
相比于 $L_2$ 损失, $L_1$ 损失对于离异值更加鲁棒, 当预测值与目标值相差很大时, 梯度很容易爆炸, 因为梯度里面包含了 $(t_i^u - v_i)$ 这一项, 而smooth L1 在值相差很大是, 其梯度为 $\pm 1$ ( $L_1$ 在 $x$ 绝对值较大时, 是线性的, 而 $L_2$ 是指数的, 很容易爆炸).
公式(1)中的超参数 $\lambda$ 用于平衡两种损失之间的影响力. 默认情况下 $\lambda = 1$.

# Mini-batch Sampling

在 fine-tuning 阶段, 每一个 SGD mini-batch 由 $N=2$ 张图片组成(均匀采样), 我们将 mini-batch 的大小设置为 $R=128$, 也就是说从每张图片中采样 64 个 RoIs. 和 R-CNN 一样, 我们令 IOU 大于 0.5 的 RoI 的 $u$ 大于等于 1, 代表该 RoI 中含有物体, 令 IOU 处于 [0.1, 0.5) 的 RoI 代表背景, 前景和背景的比例为 1:3. **没有使用其他的数据增广方法.**

# RoI 反向传播算法

下面我们来介绍一下 RoI pooling 层是如何进行反向传播的. 为了简单起见, 我们假设每一个 mini-batch 仅包含一张图片, 即 $N=1$. 我们令 $x_i \in R$ 表示 RoI pooling layer 的第 $i$ 个激活输入, 令 $y_{rj}$ 表示第 $r$ 个 RoI 的 第 $j$ 个输出. 则 RoI pooling layer 的计算公式为 $y_{rj} = x_{i^{\*}(r,j)}$, 其中 $i^{\*}(r,j) = \arg\max x_{i' \in R(r,j)} x_{i'}$. $R(r,j)$ 为输出单元 $y_{rj}$ 对应的窗口内的下标集合. 每个 $x_i$ 都可以被赋值到不同的 $y_{rj}$ 输出上. RoI Pooling 在反向传播计算梯度时, 可以看做是分别对每个候选区域框计算max pooling 的梯度, 然后将所有候选区域框的梯度累加, 其过程及公式如下:

$$\frac{\partial L}{\partial x_i} = \sum_r \sum_j [i = i^*(r,j)]\frac{\partial L}{\partial y_{rj}}$$

式中, $x_i$ 代表RoI Pooling前特征图上的像素点, $y_{rj}$ 代表pooling后的第 $r$ 个候选区域的第 $j$ 个点, $i^\*(r,j)$ 代表点 $y_{rj}$ 像素值的来源(最大池化的时候选出的最大像素值所在点的坐标). 由此可以看出, 只有当池化后某个点的像素值在池化过程中采用了当前点 $x_i$ 的像素值 (即满足 $i = i^\* (r,j)$ ) 时, 才在 $x_i$ 处回传梯度. 注意, 每个 $x_i$ 都有可能被赋值到不同的 $y_{rj}$ 输出上, 正如下图所示.

![](https://wx3.sinaimg.cn/large/d7b90c85ly1fw9pzwf08rj208h05u0si.jpg)


**实际上 RoI pooling max 的反向传播算法与 max pooling 的反向传播算法很类似, 区别仅在于后者每个输出点的来源是一个固定的子窗口, 而前者每个点的来源窗口的大小不固定**

# Fast R-CNN detection

Fast R-CNN 网络接收一张图片作为输入, 在推演阶段, $R$ 的值大约为 2000, 对于每一个 RoI $r$, 前向传播过程都会输出一个类别概率分布 $p$, 以及 $K$ 个相关的 box-bounding 的偏移量. 我们利用估计概率来为每一个 $r$ 赋予检测置信度 $Pr(class = k | k) \triangleq p_k$. 然后我们对每一类都是单独的使用 NMS 算法.

# Truncated SVD 截断式奇异矩阵分解

对于整张图片的分类问题来说, 花费在全连接上的计算时间相比于在卷积层上的计算时间来说, 要小很多. 但是, 与之相反的, 对于目标检测问题来说, 需要处理的RoI数量很大, 并且前向计算的时间有几乎一半都花费在了全连接层的计算上, 因此, 使用truncated SVD技术来进行加速.

对于一个权重矩阵为 $u\times v$ 的全连接层来说, 该矩阵可以被近似的因式分解为:

$$W \approx U \Sigma_t V^T$$

式中, $U$ 是一个 $u\times t$ 的矩阵, $\Sigma_t$ 是一个 $t\times t$ 的对角矩阵, 包含着矩阵 $W$ 的值最大的 $t$ 个奇异值, $V$ 是一个 $v\times t$ 的矩阵. 可以看到, 奇异值分解将矩阵 $W$ 的参数量从 $uv$ 降低到了 $u+v$, 如果此时 $t$ 远远小于 $\min(u,v)$, 那么就会大大节省总的参数量. 为了压缩网络, 单个的全连接网络层的权重矩阵 $W$ 会被两层全连接层所替代, 注意在这两层全连接层中间没有非线性激活函数. 第一个全连接层使用的权重矩阵为 $\Sigma_t V^T$ (没有偏置项), 第二个权重矩阵为 $U$ (带有原始矩阵 $W$ 的偏置项). 这一压缩步骤在 RoIs 的数量很大时可以起到不错的加速效果. 关于奇异值分解更详细的介绍可以看 [奇异值分解解析](../深度学习-奇异值分解)

# Main results

本文的实验结果体现出了三点贡献:
1. 在 VOC07, 2010 和 2012 上取得了 state-of-the-art 的准确率
2. 相对于 R-CNN, SPPNet, Fast R-CNN 的训练和推演的速度更快
3. 通过 fine-tuning VGG16 中的网络层进一步提升了 mAP.



表1

![](https://wx2.sinaimg.cn/large/d7b90c85ly1fxq31qnyjmj21n30c1n1m.jpg)

表2

![](https://wx3.sinaimg.cn/large/d7b90c85ly1fxq31wpj8wj21nx0df0xh.jpg)

表3

![](https://wx4.sinaimg.cn/large/d7b90c85ly1fxq325f6t9j21nj0c978h.jpg)

表4

![](https://wx2.sinaimg.cn/large/d7b90c85ly1fxq32n4ql0j20wm0m3jvf.jpg)

表5

![](https://wx4.sinaimg.cn/large/d7b90c85ly1fxq333iyftj20xk0cfacc.jpg)

图2

![](https://wx4.sinaimg.cn/large/d7b90c85ly1fxq33jkuzoj20wy0hi424.jpg)

表6

![](https://wx4.sinaimg.cn/large/d7b90c85ly1fxq33zqt16j21fu0ababp.jpg)

表7

![](https://wx3.sinaimg.cn/large/d7b90c85ly1fxq348vetuj20x10cmq58.jpg)

表8

![](https://wx1.sinaimg.cn/large/d7b90c85ly1fxq34ncol8j20x4097wg4.jpg)

图3

![](https://wx1.sinaimg.cn/large/d7b90c85ly1fxq353y1txj20xa0jc41o.jpg)

<span id = "简述 RoI Pooling 的原理和反向传播公式">
# 简述 RoI Pooling 的原理和反向传播公式
对于任意给定尺寸为 $h\times w$ 的feature map的 RoI 窗口, 将其划分成 $W\times H$ 的网格大小(上图中的示例为 $W\times H= 3\times 3$ ), 这样, 每一个网格 cell 中的尺寸大约为 $h/H \times w/W$, 然后我们在网格 cell 中执行max pooling操作. 和标准的 max pooling 相同, RoI pooling 在卷积图谱上的各个通道之间是独立计算的. 这样, 对于任意size的输入, 都可以获得固定长度的输出. 可以看出, **RoI layer 实际上是 spatial pyramid pooling layer 中的一个特例, 即只有一个 pyramid level.** (但是相比于金字塔池化, RoI 池化可以确定固定大小的 pooling 窗口, 这使得我们可以更新池化层之前的网络层参数, 进而提高准确率)

<span id = "简述 SVD 奇异值分解的原理">
# 简述 SVD 奇异值分解的原理

对于一个权重矩阵为 $u\times v$ 的全连接层来说, 该矩阵可以被近似的因式分解为:

$$W \approx U \Sigma_t V^T$$

式中, $U$ 是一个 $u\times t$ 的矩阵, $\Sigma_t$ 是一个 $t\times t$ 的对角矩阵, 包含着矩阵 $W$ 的值最大的 $t$ 个奇异值, $V$ 是一个 $v\times t$ 的矩阵. 可以看到, 奇异值分解将矩阵 $W$ 的参数量从 $uv$ 降低到了 $ut+tv$, 这个 $t$ 就是奇异矩阵中的奇异值数量, 奇异值有一个非常重要的性质, 就是它的下降速度很快, 在很多情况下, 前 10% 甚至 1% 的奇异值的和就站了全部奇异值之和的 99% 以上的比例. 也就是说, 我们可以用最大的 $k$ 个奇异值来近似描述矩阵. 由于 $k$ 远远小于 $\min(u,v)$, 因此可以大大节省参数量. 在实现上, 将单个的全连接网络层的权重矩阵 $W$ 用两层全连接层所替代, 注意在这两层全连接层中间没有非线性激活函数. 第一个全连接层使用的权重矩阵为 $\Sigma_t V^T$ (没有偏置项), 第二个权重矩阵为 $U$ (带有原始矩阵 $W$ 的偏置项).
关于奇异值分解更详细的介绍可以看 [奇异值分解解析](../深度学习-奇异值分解)

<span id = "为什么 RoI Pooling 比 SPP 效果好">
# 为什么 RoI Pooling 比 SPP 效果好

SPP的Pooling方式是组合不同划分粒度下feature map的max pooling. 它也具有和 RoI Pooling 类似的效果, 可以接受任意尺度的特征图谱, 并将其提取成固定长度的特征向量, 但是 SPPNet 和 R-CNN 在选取候选框时的采样策略是相同的, 都会在不同的图片上进行采样, 这样的话就会使得反向传播过程非常缓慢, 因此为了 SPPNet 没有对浅层的网络进行 fine-tuning, 而是直接在最后两个全连接层上进行fine-tune, 虽然最后也取得了不错的成果, 但是Roos认为, 虽然离输入层较近的前几层卷积层是比较generic和task independent的, 但是靠近输出层的卷积层还是很有必要进行fine-tune的, 他也通过实验证实了这种必要性, 于是他简化了SPP的Pooling策略, 用一种更简单粗暴的Pooling方式来获得固定长度的输出向量, 同时也设计了相应的RoI Pooling的反向传播规则, 并对前面的基层卷积层进行了fine-tune, 最终取得了不错的效果.
