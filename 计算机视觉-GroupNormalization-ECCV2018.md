---
title: Group Normalization
sitemap: true
categories: 计算机视觉
date: 2018-10-24 15:59:22
tags:
- 计算机视觉
- 目标检测
- 论文解读
---

**作者:** Yuxin Wu and Kaiming He
**发表:** ECCV 2018, Best Paper Honorable Mention


# 核心亮点

**针对BN对batch size的依赖问题, 提出了一种新的通用型归一化方法**
提出了一个用于替代BN的简单算法, 称之为GN(Group Normalization). GN将输入图谱的通道分成不同的组, 并且计算每一组的mean和variance, 然后将其进行归一化. GN的计算复杂度与batch size 的大小是相互独立的, 并且它的准确度在不同范围内的batch size下仍然是稳定的. 并且在整体表现和不同任务上的效果均强于其他类型的归一化方法(LN,IN等)

<div style="width: 550px; margin: auto">![](https://wx3.sinaimg.cn/large/d7b90c85ly1fww0h6ugadj20s80kjacr.jpg)

# 摘要

BN是深度学习中的一项里程碑式的技术, 它可以让不同网络进行训练. 但是按照batch进行归一化的过程会引入一些问题: 当batch的size比较小时, 由于batch的统计期望十分不精确, 会使得BN的误差大幅度增加. 这一点限制了BN在训练大型模型时的使用(如目标检测, 分割等等, 由于GPU内存的限制往往batch的值很小). 因此, 本文提出了一个用于替代BN的简单算法, 称之为GN(Group Normalization). GN将输入图谱的通道分成不同的组, 并且计算每一组的mean和variance, 然后将其进行归一化. GN的计算复杂度与batch size 的大小是相互独立的, 并且它的准确度在不同范围内的batch size下仍然是稳定的. 在ResNet50中, 当batch size为2时,  GN的错误率比BN要低10.6%, 当batch size为经典值时(32,64等), GN的表现也可以媲美BN, 同时, 也超越了其他归一化方法(Layer, Instance, Weight Normalization等). 不仅如此, GN还可以很自然的从预训练的模型中进行fine-tuning. GN在多个任务上都表现出了很好的效果, 可以用其替换掉BN. (只需数行代码即可实现GN).

# 介绍

BN在多项任务中都起到了很好的实验效果, 但是BN的有效性必须建立在足够大的batch size之上(如32/GPU). 当batch size 比较小时, BN往往无法取得好的效果, 并且还会提升模型的误差, 如图1所示.

由于硬件GPU显存大小的设置, 在使用较大的模型训练分辨率较高的图片时, 往往不能设置很高的batch size. 因此, 本文提出了GN算法来作为BN的代替.

# 相关工作

**Normalization**: BN在很多任务上都取得了很好的效果(BN通常会在每一层都执行), 但是, BN依赖于batch的平均值和方差, 这使得batch size的大小对BN的效果有较大的影响, 同时, 在测试阶段, 单个的图片是没有均值和方差的, 所以只能用整个数据集的均值和方差来代替, 通常会使用滑动平均来维护这两个变量, 也就是说, 在使用BN时, 如果数据集改变了, 则均值和方差就会有较大改变, 这就造成了训练阶段和测试阶段的不一致性, 由此也会带来一些问题.
另一些Normalization方法, 尝试避开batch size, 如Layer Normalization(LN)和Instance Normalization等, 但他们通常是针对RNN/LSTM模型的, 并且在大多数的视觉任务上表现不如BN好, 还有Weight Normalization, 它不是对网络层的输入进行归一化, 而是对网络层的参数进行归一化, 同样, 在大多数的视觉任务上, 表现都不如BN.

**Addressing small batches:** Ioffe在NIPS2017上提出Batch Renormalization(BR)尝试解决BN的小batch size问题. 它引入了两个额外的参数来限制BN的估计期望和方差, 减少它们在小batch size时的不稳定问题. 但是BR本质上还是依赖于batch的, 当batch较小时, 其性能也会相应下降(不过下降较少).
也有的工作尝试避免使用小batch size. CVPR2018的一篇文章通过同步BN的方法, 使得模型可以在多GPU上计算平均值和方差, 但是, 这个方法并没有才本质上解决BN的问题, 相反的, 它的方法更像是一种工程和硬件上的解决方案, 希望以此来达到BN的要求. 不仅如此, 这种同步BN方案也使得模型在工业中的大规模分布式训练下无法利用异步的优化方法, 如ASGD.

**Group-wise computation:** Group convolution计算在多个模型中都有提及. 但是本文的GP并不需要组卷积计算, 它是一个一般化的网络层(generic layer).

# Group Normalization

特征图谱中的各个channels之前也并不是完全互相独立的.

## 公式

典型的归一化公式形式(BN,LN,IN,GN)为:

$$\hat x_i = \frac{x_i - \mu_i}{\sigma_i}$$

上式中, $x$ 代表某一层的特征, $i$ 代表下标, 在2D图像中, $i=(i_N, i_C, i_H, i_W)$ , 是一个4D的向量. $\mu$ 和 $\sigma$ 分别为期望和标准差, 通过下式计算得到:

$$\mu_i = \frac{\sum_{k\in S_i} x_k}{m}, \sigma_i = \sqrt{\frac{sum_{k\in S_i} (x_k-\mu_i)^2}{m} + \epsilon}$$

大多数的Normalization方法的不同之处就在于集合 $S_i$ 的不同(如图2所示):
- BN: $S_i = \{k | k_C = i_C \}$, 代表BN是在求每一个channel的均值和方差. 也即 $C$ 不变, 求固定 $C$ 时 $(N,H,W)$ 的均值和方差
- LN: $S_i = \{k | k_N = i_N \}$, 代表 $N$ 不变, 求固定 $N$ 以后 $(C,H,W)$ 的均值和方差(可以看出, 此时的均值和方差已经不受batch size的影响)
- IN: $S_i = \{k | k_N = i_N, k_C = i_C\}$, 代表 $N$ 和 $C$ 都固定时 , $(H,W)$ 的均值和方差. 也就是说是对单个特征中, 单个通道上的均值和方差.

<div style="width: 550px; margin: auto">![图2](https://wx2.sinaimg.cn/large/d7b90c85ly1fww4hqied3j21kw0esjwt.jpg)

上面所有的Normalization方法(BN,LN,IN)都使用了线性偏移来补偿可能引起的数据分布表征丢失问题($\gamma$ 和 $\beta$ 都通过 $i_C$ 作为下标, 也就是对于每个通道都有不同的 $\gamma$ 和 $beta$):

$$y_i = \gamma x_i + \beta$$

**Group Norm:** 在Group Norm中, $\mu$, $\sigma$ 以及集合 $S_i$ 的定义如下:

$$S_i = \{ k | k_N = i_N, \lfloor \frac{k_C}{C/G} \rfloor = \lfloor \frac{i_C}{C/G} \rfloor \}$$

上式中, $G$ 是group的数量, 是一个超参数(默认为32), $C/G$ 是每一个group中的channels的数量, $\lfloor \frac{k_C}{C/G} \rfloor = \lfloor \frac{i_C}{C/G} \rfloor \}$ 意味着下标 $i$ 和 $k$ 在同一个group内. GN计算的 $\mu$ 和 $\sigma$ 是处于同一个group的所有通道上的 $(H,W)$ 的均值和方差, 具体的计算方式如图2最右侧所示, 在该图实例中, G=2, 并且每一个group具有3个channels. GN同样会在每一个channel中使用参数 $\gamma$ 和 $\beta$ 对数据执行线性偏移. 具体来说, 处于同一个 group 当中的所有像素都会被都一个 $\mu$ 和 $\simga$ 归一化, GN 会在每个通道上都学习参数 $\gamma$ 和 $\beta$. (注意是通道, 不是 Group)

**Relation to Prior Work:**  很明显, LN和IN实际上可以看做是GN的一种特例情况(如图2). 当 $G=1$ 时, GN就变成LN, 当 $G=C$ 时, GP就变成了IN.

## 实现

就和GN的思想一样, GN实现起来也十分简单, 下图是GN基于python和tensorflow的实现代码, 从代码中可以看到, $\gamma$ 和 $\beta$ 的shape为 [1,C,1,1], 也就是说, $\gamma$ 和 $\beta$ 在同一个group中的不同channel来说, 值是不一样的.

<div style="width: 550px; margin: auto">![](https://wx3.sinaimg.cn/large/d7b90c85ly1fwwerxxn0kj20s70haad2.jpg)


# 实验

<div style="width: 550px; margin: auto">![](https://wx3.sinaimg.cn/large/d7b90c85ly1fwwes3nh5zj21kw0j5dyd.jpg)

<div style="width: 550px; margin: auto">![](https://wx2.sinaimg.cn/large/d7b90c85ly1fwwes7p05wj21kw0huwuz.jpg)

<div style="width: 550px; margin: auto">![](https://wx4.sinaimg.cn/large/d7b90c85ly1fwwesbnq0mj20s809qmyz.jpg)

<div style="width: 550px; margin: auto">![](https://wx2.sinaimg.cn/large/d7b90c85ly1fwwesfiq43j20sd0cajtw.jpg)

<div style="width: 550px; margin: auto">![](https://wx4.sinaimg.cn/large/d7b90c85ly1fwwesjygdnj21kw0gpat8.jpg)

<div style="width: 550px; margin: auto">![](https://wx2.sinaimg.cn/large/d7b90c85ly1fwwesnf0lvj20t20j6tcb.jpg)

<div style="width: 550px; margin: auto">![](https://wx2.sinaimg.cn/large/d7b90c85ly1fwwesrox6zj20sl0jwtdr.jpg)

<div style="width: 550px; margin: auto">![](https://wx4.sinaimg.cn/large/d7b90c85ly1fwwesvxyoej20sv0jkwyp.jpg)

<div style="width: 550px; margin: auto">![](https://wx4.sinaimg.cn/large/d7b90c85ly1fwwet23060j21kw0nwag1.jpg)

<div style="width: 550px; margin: auto">![](https://wx2.sinaimg.cn/large/d7b90c85ly1fwwet5sufej20sa0f5aoe.jpg)

<div style="width: 550px; margin: auto">![](https://wx3.sinaimg.cn/large/d7b90c85ly1fwwet9mkjbj20s209s762.jpg)

<span id = "简述 GN 的原理">
# 简述 GN 的原理
BN 在很多任务上都取得了很好的效果(BN通常会在每一层都执行), 但是, BN 依赖于 batch 的平均值和方差, 这使得 batch size 的大小对BN的效果有较大的影响, 同时, 在测试阶段, 单个的图片无法提供良好的均值和方差进行归一化, 所以只能用整个数据集的均值和方差来代替, 通常会使用滑动平均来维护这两个变量, 也就是说, 在使用 BN 时, 如果数据集改变了, 则均值和方差就会有较大改变, 这就造成了训练阶段和测试阶段的不一致性, 由此也会带来一些问题. 因此, GN 为了解决 BN 对 batch 大小的依赖问题, 转而从另一个角度来进行归一化, GN 更像是介于 LN 和 IN 中间的一种归一化方法, 它会将通道分成不同的组, 同时在固定下标 N 的同时, 求取当前组内的均值和方差来进行归一化. 通过实验分析和论证, GN 可以取得不错的效果, 避免了对 batch 的依赖问题.

<span id = "为什么 GN 效果好">
# 为什么 GN 效果好
GN 是从 LN 和 IN 中变化来的, 组的划分实际上可以看做是一种对数据分布的假设, 以 LN 为例, 它实际上假设了每张图片所有通道的特征都是同分布的, 而 GN 则是假设每个组的分布不同, 条件没有那么苛刻, 因此 GN 的表现力和包容性会更强, 而 IN 只依赖与独立的某一维, 没有探究不同通道之间特征的关联性. 相对于 BN 来说, 当 batch 的大小足够时, BN 的性能表现依然很不错, 因此, GN 充当的角色更像是当 batch 较小, 无法使用 BN 时的一种替代措施.

<span id = "简述 BN, LN, IN, GN 的区别">
# 简述 BN, LN, IN, GN 的区别

<div style="width: 550px; margin: auto">![图2](https://wx2.sinaimg.cn/large/d7b90c85ly1fww4hqied3j21kw0esjwt.jpg)
这些 Norm 方法的不同之处就在于计算均值和方差时使用的像素集合不同(如图2所示), 假设输入的 tensor 的 shape 为 $(N, C, H, W)$:
- BN 是固定 $C$ 不变, 求固定 $C$ 时所有 $(N,H,W)$ 像素点的均值和方差, 这个均值和方差会用来归一化所有处于当前通道 $C$ 上的像素.
- LN 是固定 $N$ 不变, 求固定 $N$ 时所有 $(C,H,W)$ 像素点的均值和方差, 这个均值和方差会用来归一化所有处于当前 $N$ 上的像素. 可以看出, 这里 LN 在求取均值和方差时, 由于固定了 $N$, 所以与 batch 的大小无关.
- IN 是同时固定 $N$ 和 $C$ 不变, 求固定 $N$ 和 $C$ 时所示 $(H,W)$ 像素点的均值和方差.
- GN 是介于 LN 和 IN 中的一种 Norm 方法, 它首先也是固定 $N$ 不变, 然后会将 $C$ 分成若干个 Group, 然后分别求取每个 Group 的均值和方差, 并对 Group 中的像素进行归一化

**注意, 无论是哪种 Norm 方法, 它们使用的线性偏移的参数个数都等于通道 $C$ 的大小.**

<span id = "GN 中线性偏移的参数个数怎么计算的">
# GN 中线性偏移的参数个数怎么计算的

对于 GN 层来说, 如果它的输入 shape 均为为 $(N, C, H, W)$, 则其输出 shape 也为 $(N, C, H, W)$, **即保持输入输出 shape 不变.** GN 中的线性偏移参数 $\gamma$ 和 $beta$ 的个数 **与输入 shape 的通道数相同, 均为 $C$**. GN 除了需要确定输入层的通道数以外, 还需要确定 Goup 的数量. 下面给 PyTorch 中 GN 的声明.
```py
torch.nn.GroupNorm(num_groups, num_channels, eps=1e-05, affine=True)
```
