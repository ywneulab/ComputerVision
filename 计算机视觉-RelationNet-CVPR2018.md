---
title: Relation-Network (CVPR, 2018)
sitemap: true
categories: 计算机视觉
date: 2018-11-22 16:34:24
tags:
- 计算机视觉
- 目标检测
- 论文解读
---

**文章:** Relation Networks for Object Detection
**作者:** Han Hu1, Jiayuan Gu, Zheng Zhang, Jifeng Dai1, Yichen Wei
**备注:** MSRA, Peking University, Oral Paper



# 摘要

尽管长久以来, 人们都任务对图像中物体之间的关系进行建模可以提升目标检测模型的精度, 但是, 就目前而言, **所有先进的** 目标检测模型都是将物体作为一个独立的个体进行检测的, 并没有在学习的过程中利用到它们之间的关系(Relations).
本文提出了一个物体关系模型, 它会通过物体之间的特征和纹理来 **同时** 处理一系列物体之间的关系, 因此允许对这些关系进行建模. 本文提出的方法是轻量级的, 无需额外的监督信息, 并且可以很容易嵌入到现有的模型当中. 在现代的目标检测流水线当中, 本文的方法对于目标识别和去除重复步骤的改进是有效的, 验证了在基于CNN的检测模型中对物体关系建模的有效性. **本文的模型是收个完全的端到端的目标检测模型.**

# 介绍

近年来, 基于深度学习的目标检测取得了很多成功, 但是, 依然没能够利用到物体之间的相关关系信息, 其中一个难点在于目前的检测模型都比较简单, 无法对复杂度物体关系进行建模.
本文的方法受到了自然语言处理中 attention 模型的启发, attention 模型中一个元素可以被某个集合的元素的累积权重所影响. 这个累积权重是在模型学习的过程当中学习到的, 近年来, attention 模型在图像描述领域也得到了许多成功应用.
在本文中, 我们首次提出了适用于目标检测任务的自适应的 attention 模型. 该模型建立与一个基本的 attention 模型, 不同之处在于其主要元素不再是 words, 而是 objects. 物体(objects)具有2D的空间布局, 并且具有不同的 scale 和 aspect ratio, 这些信息相对于一维的 words 来说更加复杂. 因此, 本文提出的模型会将原来的 attention 权重扩展成两部分: 原始的权重和一个新的几何(geometric)权重. 后者会对物体间的空间关系进行建模, 并且仅仅考虑它们之间的相对几何关系, 使得整个模型具有平移不变性---这正是物体识别期望的一个性质( **物体检测希望的是平移可变性, 这里会不会有些问题?** ). 通过实验证明, 这里新添加的几何权重(geometric weights) 是非常重要的.
本文的模型称为 `object relation module`, 它和 attention 模型具有相同的优点. 它会接受多个输入, 并且以并行的方式进行处理(相对于序列模型的串行), 并且使可导和 in-place(输入输出的维度相同) 的, 因此, 本文提出的模型可以作为一个基本的 building block 灵活的嵌入到现有的各个模型当中去.
如图1所示, 我们可以将 relation module 嵌入到现有的模型当中去, 来提升 instance recognition step, 同时 duplicate removal step.

![](https://wx2.sinaimg.cn/mw690/d7b90c85ly1fxi1bla0fnj20zm0l4n4x.jpg)

从原理上来说, 我们的方法与目前的大部分检测方法都不相同, 并且可以弥补目标的许多检测方法. 本文的方法采用了一个新的维度: 一系列的物体在被处理时, 会同时对其他物体的识别产生影响, 而不是将每个物体单独识别.

# 相关工作

**Object Relation in post-processing:** 这些方法在 pre-DP 时代取得了不错的效果, 但是在 deep ConvNets 时代却没能表现出其有效性.

**Sequential relation modeling:** LSTM. 在目标检测任务中, 有方法建议令先找到的物体会帮助寻找下一个物体, 但是没能证明该方法的有效性.

**Human centered scenarios:** 关注与人相关的关系检测, 但是需要额外的监督标签.

**Duplicate removal:** 去重, NMS, GossipNet(learn duplicate removal),

**Attention modules in NLP and physical system modeling:** Attention Module.

# 物体关系模型(Object Relation Module)

我们首先回顾一个简单的 Attention 模型, 名为 Scaled Dot-Product Attention. 假设输入为 $d_k$ 维的 queries 和 keys, 并且具有 $d_v$ 维的values. 点积会在 query 和所有的 keys 之间进行, 以获取它们之间的相似度, 我们利用 SoftMax 函数来获取 values 的维度. 具体来说, 给定一个 query q, keys(packed into matrices K), values(packed into V), 则输出如下:

$$v^{out} = softmax( \frac{qK^t}{\sqrt{d_k}}) V \tag 1$$

接下来我们描述一下物体关系的计算. 令 $f_G$ 代表物体的几何信息, 即边框的四个坐标, $f_A$ 代表物体的特征信息, 具体形式视任务而定. 当给定 $N$ 个物体 $\{ f_A^n, f_G^n \}, n=1, ..., N$ 时, 第 $n$ 个物体和其他所有物体之间的关系特征 $f_R(n)$ 为:

$$f_R(n) = \sum_{m} \omega^{mn} \cdot (W_V \cdot f_A^m) \tag 2$$

上面的输出是第 $n$ 个物体与所有所有物体的特征信息关系的权重和, 通过 $W_V$ 进行线性转换, 关系权重 $\omega^{mn}$ 表明了对其他物体对当前物体的影响程度, 计算公式如下:

$$\omega^{mn} = \frac{\omega_G^{mn}\cdot exp(w_A^{mn})}{\sum_k \omega_G^{kn}\cdot exp(\omega_A^{kn})} \tag 3$$

物体特征的权重 $\omega_A^{mn}$ 计算公式如下:

$$\omega_A^{mn}  = \frac{dot(W_k f_A^m, W_Q f_A^n)}{\sqrt(d_k)} \tag 4$$

这里的 $W_K$ 和 $W_Q$ 都是评价标准(matrices), 它们会将原始的图片特征 $f_A^m$ 和 $f_A^n$ 投影到一个子空间中去, 并在此空间可以描述这些特征的好坏, 投影之后的维度是 $d_k$.
几何权重(Geometry weight)计算如下:

$$\omega_G^{mn} = max{0, W_G \cdot \xi_G(f_G^m, f_G^n)} \tag 5$$

这里有两步, 首先, 两个物体的几何特征会被嵌入到一个更高的维度 $\omega_G$ 上, 然后, 为了保证平移不变性和尺寸不变性, 我们会利用一个4个的相对几何信息来代替:

$$(log(\frac{|x_m - x_n}{w_m}), log(\frac{|y_m - y_n}{h_m}), log(\frac{w_n}{w_m}), log(\frac{h_n}{h_m}))

然后, 嵌入后的相对坐标会通过一个权重矩阵 $W_G$ 转化成一个标量, 并且用ReLU 来激活. 几何特征的attention有效性正如表1(a)所示.
一个物体关系模型总共会累积 $N_r$ 个关系特征, 并且通过加上下面的项来增加输入的图片特征:

$$f_A^n = f_A^n + Concat[f_R^1(n), ..., f_R^{N_r}(n)], \text{for all} n \tag 6$$

上式的流程可以总结出算法1, 如下所示:

![](https://wx1.sinaimg.cn/mw690/d7b90c85ly1fxi1c5q4a2j20s40jfgqs.jpg)

上式算法可以通过基本的操作实现, 如图2所示

![](https://wx2.sinaimg.cn/mw690/d7b90c85ly1fxi1cievh3j20rw0i4jtt.jpg)

空间复杂度和计算复杂度如下所示:

$$O(Space) = N_r (2d_f d_k + d_g) + d_f^2$$

$$O(Comp) = N d_f(2N_r d_k + d_f) + N^2 N_r(d_g + d_k + d_f/N_r + 1)$$

通常情况下: $N_r = 16, d_k = 64, d_g = 64$. 关系模型具有相同的输入和输出, 使得可以更容易作为 building block 添加到现有模型当中.

# 目标检测关系模型(Relation Networks For Objects Detection)

目标检测的流程可以分布以下四步:
- 在整张图片上生成特征图谱
- 生成候选区域框
- 执行实例识别(instance recognition)
- 去重(duplicate removal): NMS

本文提出的物体关系模型主要作用于后两步, 即令起提升 instance recognition 的能力以及具有学习去重的能力.

**Our implementation of different architectures:**
用 ResNet 作为backbone, 用 RPN 来生成候选区域框, 并对以下几种模型进测试.
- Faster RCNN
- FPN
- DCN
抛去以上三者整体结构的区别不说, 它们都是用了同样的 head 网络, 即利用 RoI pooling 后接两个全连接层来生成用于分类和回归的最终的特征图谱.

**Relation for Instance Recognition:**


$$ {RoIFeat}_n$$
