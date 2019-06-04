---
title: Inception V3
sitemap: true
categories: 计算机视觉
date: 2018-11-20 15:05:05
tags:
- 计算机视觉
- 网络结构
- 论文解读
---

**文章:** Rethinking the Inception Architecture for Computer Vision
**作者:** Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna
**备注:** Google, Inception V3


# 核心


# 摘要
近年来, 越来越深的网络模型使得各个任务的 benchmark 都提升了不少, 但是, 在很多情况下, 我们还需要考虑模型计算效率和参数量. 本文我们将通过适当地使用 factorized convolutions(卷积分解) 和 aggressive regularization 来尽可能的是增加计算效率.

# 介绍

AlexNet, VGGNet 虽然很成功, 但是他们的计算成本太大, 模型参数量较多, 与之相比, Inception 模型不仅参数量小, 特征提取能力也较强. 但是, Inception 模型较为复杂, 这使得我们很难对模型进行修改. 在本文中, 我们会先描述一个一般化的原则和优化想法, 使得可以高效的扩大卷积网络的大小.

# General Design Principles

接下来我们会叙述几条基于大规模多结构的神经网络的设计原则

1. **避免使用 representational bottlenecks, 尤其是在网络的较浅层.** 前馈神经网络可以被表示成一个有向无环图, 这定义了一个非常明确的信息流. 对于任何输入, 都可以获得很大的信息. 因此, 我们应避免使用 bottlenecks 过度压缩信息. 在通常情况下, 特征的尺寸应该从输入到输出平缓降低. 理论上来说, 降维后的信息不能完全提供足够的信息内容给后续的结构, 而仅仅只是对主要信息的一种粗略的概括.
2. **高维表示更容易在网络本地进行处理.**
3. **空间聚合可以在降低的维度 embedding 中完成, 而不需要太多或任何表征能力的损失.** 比如, 在执行一个更大的卷积操作(如3×3)之前, 我们可以在 spatial aggregation 之间先降低维度, 而这不会带来严重的负面影响. 我们检测其原因是因为相邻单元之间的相关性很强, 所以导致在降维的时候损失较小. 降维有助于加速训练.
4. **权衡网络模型深度的宽度.** 提升模型的宽度和深度都可以提升模型的性能, 但是, 最好的方式是结合这两种方式, 以便使得模型的复杂度可以均衡的分布在网络的深度和宽度中.

上面的原则不建议直接使用, 更好的办法是在你不确定如何提升模型性能时进行权衡和尝试.

# Factorizing Convolutions with Large Filter.

GooLeNet 的成功原因之一得益于广泛的使用降维. 这可以看做是 **factorizing convolutions**(对卷积进行因式分解) 的一种特殊情况. 在一个视觉网络中, 某点的输出与它相邻的其他点的响应之间有很高的相关性. 因此, 我们可以在聚合之前将这些响应进行降维, 在这理论上, 应该能够产生相同的局部特征表示. 由于 Inception 网络是全卷积的, 每一个权重都会与多处响应相关联, 计算成本的降低会带来参数量的降低. 这意味着 **通过恰当的因式分解, 我们可以得到更多解耦的参数, 从而可以带来更快的训练速度.**


## 分解成更小的卷积(Factorization into smaller convolutions)
较大的卷积核尺寸(如5×5, 7×7)往往意味着很高的计算成本. 例如, 5×5 的计算成本为 3×3 卷积核的 25/9 = 2.78 倍. 但是, 如果直接替换为 3×3 的卷积核, 那么就会在特征表达能力上造成一些损失. 幸运的是, 我们可以通过多层小卷积核添加的方式来替换大卷积核, 如图1所示, 他们的感受野是相当的, 但是前者的参数只有后者的 $\frac{9+9}{25} = 28 %$.

<div style="width: 550px; margin: auto">![](https://wx3.sinaimg.cn/large/d7b90c85ly1fxfpsaihoej20pa0i2tdo.jpg)

图4, 图5展示了用两个3×3来替换 5×5 卷积核的示意图.

<div style="width: 550px; margin: auto">![图4图5](https://wx4.sinaimg.cn/large/d7b90c85ly1g0j0zt2p2yj214z0hnju2.jpg)

但是这种替换会引出两个问题, 其一为是否为造成特征表达能力的降低, 其二是如果我们的主要目标时分解计算的线性部分, 那么是否还应该在第一层保持线性激活? 即是否在第一层使用非线性激活函数? 对此, 通过实验证明, 使用线性激活比使用非线性激活的效果差一些, 如图2所示.

<div style="width: 550px; margin: auto">![](https://wx4.sinaimg.cn/large/d7b90c85ly1fxfpsmbuv0j21290rhaee.jpg)

## 空间分解为不对称卷积(Spatial Factorization into Asymmetric Convolutions)

上面的结果说明大于 3×3 的卷积核通常都可以分解为一系列 3×3 卷积核堆叠. 那么如果继续分解, 我们可以将 3×3 的卷积核分解为 3×1 的卷积核和 1×3 的卷积核, 这样一来, 参数量就变成了6, 降低了 33%, 如图3所示(将 3×3 分解成两个 2×2 的卷积核, 只能降低 11% 的参数量).

<div style="width: 550px; margin: auto">![](https://wx2.sinaimg.cn/large/d7b90c85ly1fxfpt3f686j20sp0ne425.jpg)

理论上, 我们可以将任何 $n\times n$ 的卷积核用一个 $n\times 1$ 和一个 $1\times n$ 的卷积核替代, 如图6所示. 在实际使用中, 我们发现这种分解方式在网络的浅层并不能很好的工作, 但是在网络的中层可以取得很好的效果(特征图谱大小在 12~20 之间的网络层).

<div style="width: 550px; margin: auto">![图6](https://wx3.sinaimg.cn/large/d7b90c85ly1g0j145ly6tj21dw0q3dm5.jpg)

# 辅助分类器的效用(Utility of Auxiliary Classifiers)

Inception V1 首次引入辅助分类器来提升深度网络的收敛性, 其最初动机是为了可以及时利用那些浅层网络中有用的梯度来帮助模型快速收敛, 从而缓解深度神经网络中的梯度消失问题. 有趣的是, 我们发现这个辅助分类器并不会加快训练初期的收敛速度: 对于带有辅助分类器和不带辅助分类器的两个网络, 在模型达到较高精度以前, 他们的性能看起来是差不多的. 但是 **当到了训练后期, 带有辅助分支的网络开始超越没有任何辅助分支的网络, 进而达到更高的精度**.
并且, 在 Inception V1 中使用了两个辅助分支, 我们发现, 将浅层的辅助分支去除并不会对最终的模型质量产生任何不利影响.

# 有效缩小网格尺寸(Efficient Grid Size Reduction)

传统的卷积网络通过池化操作来降低特征图谱的网格尺寸, 但是为了避免降低特征表达能力, 对于一个 $d\times d\times k$ 的特征图谱, 我们通常会先利用一个卷积层使它的通道数增加到 $2k$, 然后再利用池化层来降低它的图谱尺寸到 $\frac{d}{2}$, 因此, 这一步需要的计算量为 $2d^2 k^2$. 我们可以将卷积层和池化层的位置互换, 这样一来, 计算量就会降为 $2(\frac{d}{2})^2 k^2$, 但是, 这样会导致网络的特征表达能力下降, 造成 representational bottlenecks, 如图9所示.

<div style="width: 550px; margin: auto">![](https://wx3.sinaimg.cn/large/d7b90c85ly1fxfpvtkg0nj21310px78o.jpg)

因此, 我们推荐另一种降低计算量的方式, 如图10所示, 我们可以利用两个并行的 block P 和 block C 来达到目的, 其中 P 代表池化, C 代表卷积.

<div style="width: 550px; margin: auto">![](https://wx1.sinaimg.cn/large/d7b90c85ly1fxfpwdgzpzj21250t5jwt.jpg)

# Inception-v2

表1展示了本文网络的整体布局

<div style="width: 550px; margin: auto">![](https://wx2.sinaimg.cn/large/d7b90c85ly1fxfpwzn61yj20sr0x2q9d.jpg)

注意到我们将原来的 7×7 卷积转换成了3个 3×3 卷积.

<div style="width: 550px; margin: auto">![图7](https://wx4.sinaimg.cn/large/d7b90c85ly1fxfput5ylzj20rz0w9gpr.jpg)

图8
<div style="width: 550px; margin: auto">![](https://wx1.sinaimg.cn/large/d7b90c85ly1fxfpvbtkctj211n0n6gpn.jpg)

# Model Regularization via Label Smoothing

表3 展示了 ILSVRC 2012 的测试结果

<div style="width: 550px; margin: auto">![](https://wx1.sinaimg.cn/large/d7b90c85ly1fxfpxzneorj20rt0u1grl.jpg)

表4
<div style="width: 550px; margin: auto">![](https://wx1.sinaimg.cn/large/d7b90c85ly1fxfpyj1ykqj20t90j3adv.jpg)

# Training Methodology

# Performance on Lower Resolution Input

表2

<div style="width: 550px; margin: auto">![](https://wx2.sinaimg.cn/large/d7b90c85ly1fxfpxiwfcoj20vf0a5q4f.jpg)

# Experimental Results and Comparisons

表5
<div style="width: 550px; margin: auto">![](https://wx4.sinaimg.cn/large/d7b90c85ly1fxfpzae2exj20v40hvq6n.jpg)



<span id = "InceptionV2 相比于 GoogLeNet 有什么区别">
# InceptionV2 相比于 GoogLeNet 有什么区别

InceptionV2 改进的主要有两点. 一方面加入了 BN 层, 减少了 Internal Covariate Shift 问题(内部网络层的数据分布发生变化), 另一方面参考了 VGGNet 用两个 $3\times 3$ 的卷积核替代了原来 Inception 模块中的 $5\times 5$ 卷积核, 可以在降低参数量的同时加速计算.

<span id = "InceptionV3 相比于 GoogLeNet 有什么区别">
# InceptionV3 相比于 GoogLeNet 有什么区别

<div style="width: 550px; margin: auto">![Inception](https://wx2.sinaimg.cn/large/d7b90c85ly1g1g5nupi5fj21c80u0doq.jpg)

InceptionV3 最重要的改进是分解(Factorization), 这样做的好处是既可以加速计算(多余的算力可以用来加深网络), 有可以将一个卷积层拆分成多个卷积层, 进一步加深网络深度, 增加神经网络的非线性拟合能力, 还有值得注意的地方是网络输入从 $224\times 224$ 变成了 $299\times 299$, 更加精细设计了 $35\times 35$, $17\times 17$, $8\times 8$ 特征图谱上的 Inception 模块.
具体来说, 首先将第一个卷积段的 $7\times 7$ 大小的卷积核分解成了 3 个 $3\times 3$ 大小的卷积核. 在第二个卷积段也由 3 个 $3\times 3$ 大小的卷积核组成. 第三个卷积段使用了 3 个 Inception 模块, 同时将模块中的 $5\times 5$ 卷积分解成了两个 $3\times 3$ 大小的卷积. 在第四个卷积段中, 使用了 5 个分解程度更高的 Inception 模块, 具体来说, 是将 $n\times n$ 大小的卷积核分解成 $1\times n$ 和 $n\times 1$ 大小的卷积核, 在论文中, 对于 $17\times 17$ 大小的特征图谱, 使用了 $n = 7$ 的卷积分解形式. 在第五个卷积段中, 面对 $8\times 8$ 大小的特征图谱, 使用了两个设计更加精细的 Inception 模块. 它将 $3\times 3$ 大小的卷积层分解成 $1\times 3$ 和 $3\times 1$ 的卷积层, 这两个卷积层不是之前的串联关系, 而是并联关系.

<span id = "Inception 模块的设计和使用原则是什么">
# Inception 模块的设计和使用原则是什么
1. 在网络的浅层要避免过度的压缩特征信息, 特征图谱的尺寸应该温和的降低;
2. 高维的特征信息更适合在本地进行处理, 在网络中逐渐增加非线性激活层, 这样可以使得网络参数减少, 训练速度更快;
3. 低维信息的空间聚合不会导致网络表达能力的降低, 因此, 当进行大尺寸的卷积之前, 可以先对输入进行进行降维处理, 然后再进行空间聚合操作;
4. 网络的深度和宽度需要反复权衡, 通过平衡网络中每层滤波器的个数和网络的层数使用网络达到最大性能.
