---
title: ResNet (CVPR, 2016)
sitemap: true
categories: 计算机视觉
date: 2018-09-27 14:44:25
tags:
- 计算机视觉
- 网络结构
- 论文解读
---

https://blog.csdn.net/malefactor/article/details/67637785

**文章:** Deep Residual Learning for Image Recognition
**作者:** Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun
**备注:** MSRA, Best Paper

# 核心亮点

**本文突破了传统的卷积神经网络结构, 首次提出了残差网络, 并成功的将网络的深度提升到了一个很高的层级上, 同时解决了深层网络的模型退化问题, 对整个深度学习领域产生了重大影响.**

<div style="width: 550px; margin: auto">![](https://wx1.sinaimg.cn/large/d7b90c85ly1fx8tjak1d4j21ul0nxtgb.jpg)

# 提出动机

首先文章提出了一个假设:
有一个L层的深度神经网络, 如果我们在上面加入一层, 直观来讲得到的L+1层深度神经网络的效果应该至少不会比L层的差. 因为可以简单的学习出最后一层为前一层的恒等映射, 并且其它层参数设置不变.(说明是这种更深的网络是存在是的性能不下降的解的)
但是, 通过实验发现, 当网络层数加深时, 网络的性能会下降(说明后面几层网络层没有学习到恒等映射这个解), 也就是所谓的"模型退化"问题, 如图1所示.

<div style="width: 550px; margin: auto">![](https://wx1.sinaimg.cn/large/d7b90c85ly1fx8slpn3qlj211y0jq0xb.jpg)

观察上述现象后, 作者认为产生模型退化的根本原因很大程度上也许不在于过拟合, 而在于梯度消失问题. 为了解决模型退化问题, 作者基于以上假设, 提出了`深度残差学习框架`, 没有直接堆叠网络层来 fit 期望的映射函数, 而是选择让这些网络层来 fit 一个残差映射. 也就是说, 如果我们期望得到的映射函数为 $H(x)$, 那么我们不是通过堆叠网络来直接学习这个映射函数, 而是学习对应的残差函数: $F(x):=H(x)-x$. 那么, 原始的映射函数就可以通过 $F(x)+x$ 得到(如图2所示). 我们假设这个残差映射比原始的映射函数更容易学习和优化. 极端情况下, 如果一个恒等映射是最优的, 那么相对于使得网络层学习到 $H(x)=x$ 这个映射关系, 它应该更加容易使得残差部分 $F(x) \rightarrow 0$.(原因可以看后文)


# 残差学习(Residual Learning)

首先, 我们假设 $H(x)$ 就是几层网络层堆叠后希望学习到的映射函数(underlying mapping), 而 $x$ 代表了这几层网络的输入. 在神经网络中, 我们通常假设几层非线性的网络相堆叠可以渐进的拟合一个复杂函数, 那我们也可以等价的假设这些网络层可以渐进的拟合对应的残差函数: $H(x) - x$(姑且假设 $H(x)$ 和 $x$ 维度相同). 因此, 我们不需要令网络层来近似函数 $H(x)$, 相反, 我们希望这些网络层能够近似函数 $F(x):=H(x) - x$. 原始的映射函数也可以通过 $F(x)+x$ 得到.
这种残差定义方式是收到了图1中的违反直觉的现象的启发而得出的. 正如我们之前所说的, 如果我们仅仅在模型中添加了一些恒等连接层, 那么得到的新的更深的模型的精度应该至少不会比之前的差, 但是模型还是出现了退化问题, **这说明很有可能是模型在学习的时候, 很难直接通过多层的非线性网络层学习到这种恒等映射.** 然而, 通过本文的残差学习定义, 如果恒等连接层是最优的, 那么模型在学习时可以简单的令非线性的网络层函数 $F(x)$ 为0, 以此来使模型学习到恒等映射.
在实际情况中, 往往不太可能使恒等映射是最优的, 但是本文提出的残差方法可以帮助模型提前为解决问题提供便利(precondition the problem). 核心思想为: 如果最优的映射函数相对于 zero mapping 更接近恒等映射, 那么相对于学习一个新的映射函数, 应该更容易的找到与恒等映射相关的扰动. 在实验中(图7), 我们发现这些学习后的残差函数大多具有很小的响应(标准差 standard deviations), 这说明本文的恒等映射提供了一种合理的先验条件(preconditioning).

# 恒等短接(identity shortcut connection)

ResNet提出的恒等短接用于直接跳过一个或多个层, 以便让离输入层近的网络更加靠近输出层, 残差块的结构如下图所示:

<div style="width: 550px; margin: auto">![](https://wx4.sinaimg.cn/large/d7b90c85ly1fx8tg62esbj20we0drgmx.jpg)

在本文中, 我们可以将一个 building block 定义成下面的形式:

$$y = F(x, \{W_i\}) + x \tag 1$$

上式中, $x,y$ 分别代表着这个 block 的输入和输出, 而函数 $F(x, \{W_i\})$ 代表着需要学习的残差映射. 以图2为例, 该残差块具有两个网络层, 因此 $F=W_2 \sigma (W_1 x)$, 其中, $\sigma$ 代表 ReLU, 同时为了简化表示, 忽略了偏向量. 操作 $F + x$ 是通过 element-wise addition 的短接实现的. 我们采用 $\sigma(y)$ 作为本残差模块的输出. 可以看出, 恒等短接的方式有一个好处就是既不会引入任何额外参数, 也不会带来计算成本. 这样一来我们就可以很公平的与其他卷积网络模型在各种参数上进行对比.
**注意, 公式(1)中 $x$ 和 $F$ 的维度必须相同**. 如果不同的话(即 changing the input/output channels), 我们可以利用一个线性投影矩阵 $W_s$ 来让维度匹配:

$$y = F(x, \{W_i\}) + W_s x \tag 2$$

我们也在可以在公式(1)用添加方阵 $W_s$(不改变维度). 但是通过实验我们发现, 当维度相同时, 直接相加就已经足够了, 因此我们只会在维度不同时才使用矩阵 $W_s$.
函数 $F$ 的形式是灵活的, 本文中包含了两种形式(如图5, 分别为两层和三层). 如果只使用一层的残差模块, 这近乎于是普通的线性层了, 貌似并不能获得什么提升.
上面的讨论为了方便我们使用的是全连接层, 但是残差模块同样可以用于卷积层, 在两个 feature maps 之间 channel by channel 的执行 element-wise addition.

# 网络结构(Network Architectures)

我们通过实验验证了多种不同的 plain/residual 网络, 并且观察到了相同的现象, 下面我们介绍两种网络以供讨论.

## Plain Network

如图3中间所示, 我们对 VGG-19 进行扩展, 得到了 plain baseline. 图中的卷积层大多为 3×3 大小, 并且遵守两条设计规则: 1), 对于输入和输出的 feature map 具有相同的 size 时, 卷积层也和设定为相同数量的卷积核(即当输出不改变尺寸时, 也不应改变通道数); 2), 如果 feature map size 减半, 那么卷积核的数量会变为双倍, 以此来保持每一层的时间复杂度. 我们在执行 downsampling 时, 是通过利用 stride=2 的卷积层实现的(即没有用 max pooling 层). 网络的最后会接一个全局平均池化层和一个1000路的 softmax 全连接层. 图3中的网络总共的层数为34层.
**值得注意的是: ResNet 虽然比 VGGNet 的层深更深, 但是却拥有更低的复杂度, VGG-19 的 FLOPs (multiply-adds) 次数约为 19.6 billion, ResNet 的 FLOPs 如表1所示.(复杂度低的原因主要是去掉了两次全连接层)**

<div style="width: 550px; margin: auto">![](https://wx2.sinaimg.cn/large/d7b90c85ly1fx8pr34i9hj20rc1y0kjl.jpg)


## 残差网络(Residual Network)

基于上面的 Plain 网络, 我们向其中添加 shortcut connections(如图3右侧所示), 如此, 便可以将网络转换成残差网络(residual network). 当残差模块的输入和输出的维度相同时(实线), 就是可直接使用公式(1)来建立短接. 当输入和输出的维度不同时(虚线), 我们考虑了两中方法: (A), shorcut 仍然通过恒等连接来实现, 对于升高的那些维度, 直接用0填充, 这个方法不会引入额外的参数; (B), 利用公式(2)的矩阵来使维度匹配(利用1×1卷积实现). 另外, 对于输入输出的特征图谱 size 不同的情况, 我们通过将卷积层的 stride 设置为2来实现.

## 实现细节(Implementation)

**training:**
- scale augmentation(image 的最短边被随机放缩到256或480)
- horizaontal flip
- 224 random crop
- per-pixel mean subtracted
- 标准color augmentation
- **在每一个卷积层之后, 激活层之前, 都是用了BN**
- 使用了msra初始化方法. 训练时没有使用预训练模型(train from scratch)
- SGD
- batch size = 256
- lr 从0.1开始,每当 error 停滞(plateaus)时, 缩小1/10
- 总训练迭代次数为 $60\times 10^4$
- weight decay 为 0.0001
- momentum 为 0.9
- **没有使用dropout**

**testing:**
- 10-crop
- multi-scales: {224, 256, 384, 480, 640}


# 实验(Experiments)

## 图像分类(Image Classification)

不同层的模型结构和参数如表1所示(both plain and residual).

<div style="width: 550px; margin: auto">![](https://wx1.sinaimg.cn/large/d7b90c85ly1fx8tjak1d4j21ul0nxtgb.jpg)

表2的数据显示出较深的34层的 plain 网络相比于它的 residual 版本, 具有更高的错误率.

<div style="width: 550px; margin: auto">![](https://wx3.sinaimg.cn/large/d7b90c85ly1fx9x8s7jzmj210b0a4aby.jpg)

### Plain Network

为了揭示其中的原因, 我们比较了这两个网络在训练/验证过程中错误率如图4所示. 我们观察到, 对于 plain 版本的网络, 18层的网络的解空间只是34层网络的解空间的一个子集, 但是更深的34层网络却发生了模型退化的问题.

<div style="width: 550px; margin: auto">![](https://wx4.sinaimg.cn/large/d7b90c85ly1fx9yjufe80j21ub0khtec.jpg)

**我们认为造成优化困难的原因不太可能是因为梯度消失问题而产生的.** 因为这些 plain network 在每一个卷积层之后的应用了 BN, 这就保证了在前向传播过程中的信号具有非零的方差(which ensures forward propagated signals to have non-zero variances). 同时, 我们也验证了在反向传播过程中梯度值保持着健康的归一化. **所以不论是前向传播过程还是反向传播过程, 都没有出现信号消失的现象.**
**因此, 具体是什么原因导致了 plain network 难以优化还有待讨论(The reason for such opti- mization difficulties will be studied in the future).**


### Residual Network

接下来我们评估了 ResNet-18 和 ResNet-34 两个网络, 基本的网络结构和对应的 Plain-18 和 Plain-34 相同, 只是在每一对 3×3 的卷积层之间添加了 shortcut connection. 在第一次对比当中(表2和图4右侧), 我们采用了恒等连接和零填充的短接方式, 因此相对于 plain 版本的网络并没有引入新的参数.
从表2和图4中, 我们观察到了三个结论:
1. ResNet-34 比 ResNet-18 的错误率更低(说明找到了解空间中另一个更优的解, 而此时 ResNet-34 的解空间和 Plain-34 的解空间是完全相同的). 更重要的是, ResNet-34 不仅在训练数据集上错误率更低, 在验证集上的错误率也更低, 说明确实找到了一个泛化能力更好的解, 而不是因为过拟合.
2. 图4中的 ResNet 相比于 PlainNet, 错误率更低, 说明了 ResNet 的有效性.
3. 对于错误率相当的 Plain-18 和 ResNet-18, ResNet 的收敛速度更快, 说明残差模块的存在确实可以加快模型的训练速度.

### 恒等连接与映射连接(Identity / Projection Shortcuts)

上面我们讨论了一种 parameter-free 的恒等短接的方式, 接下来我们将研究一下引入参数的映射短接(Projection Shortcuts). 在表3中我们给出了三个选项:
- (A). 使用恒等映射, 如果需要改变输出维度时, 对增加的维度用0来填充, 不会增加任何参数.(这种就是之前讨论的 parameter-free 的恒等短接)
- (B). 在输入输出维度一致时使用恒等映射, 不一致时使用矩阵映射以保证维度一致, 增加部分参数.
- (C). 对所有的block均使用矩阵映射, 大量增加参数

如表3所示, 这三种方式相比于对应的 PlainNet 都可以取得较大的精度提升, 我们可以发现, 在效果上 C>B>A，我们认为这是因为在 A 中的 zero-padded dimensions 实际上并没有进行残差学习. 由于 A/B/C 之间的差距比较小，而线性变换需要引进额外的参数, 因此这是一个可以根据实际问题进行权衡的事情.(通常不要用C, 因为增加的参数较多, 且性能提升并不是很明显).

<div style="width: 550px; margin: auto">![](https://wx2.sinaimg.cn/large/d7b90c85ly1fx9ytn38nzj20wq0kzgpx.jpg)

## Deeper Bottleneck Architectures

<div style="width: 550px; margin: auto">![](https://wx1.sinaimg.cn/large/d7b90c85ly1fx8tjn73sij21e50iw77t.jpg)

接下来, 我们讨论一下本文在 ImageNet 中使用的更深的网络. 为了取得更快的训练速度, 我们将残差网络的 building block 修改成了 bottleneck building block.(如图5所示, 左右两种 block 的复杂度相同). 其中, 1×1 的卷积层负责降维和升维, 使得 3×3 的卷积层需要处理维度更小.
对于 bottleneck 结构来说, parameter-free 的恒等短接尤其重要. 如果用矩阵映射替换了 bottleneck 中的恒等短接, 那么因为 shortcuts 需要处理的维度很高, 使得模型的 size 和时间复杂度都会加倍. 因此, 对于 bottlenect 来说, 选择恒等短接可以大大降低模型复杂度.

**ResNet-50:**
把 ResNet-34 中的每一个2层的 building block 换成3层的 bottlenect block.

**ResNet-101/152:**
在 conv4 阶段使用更多的 bottleneck block.(ResNet-152 在 conv3 也使用了更多的 bottleneck block).

在表4和表5中, 我们将本文的 ResNet 与目前最好的模型进行了对比. 结果显示本文的 ResNet 具有更高的精度.

<div style="width: 550px; margin: auto">![](https://wx2.sinaimg.cn/large/d7b90c85ly1fxa0pfl51fj20wb0izq6u.jpg)

<div style="width: 550px; margin: auto">![](https://wx4.sinaimg.cn/large/d7b90c85ly1fxa0rqbs0fj20um0d9mzt.jpg)


## CIFAR-10 and Analysis

表6

<div style="width: 550px; margin: auto">![](https://wx3.sinaimg.cn/large/d7b90c85ly1fxa10l3h7hj20wx0px43v.jpg)

图6

<div style="width: 550px; margin: auto">![](https://wx2.sinaimg.cn/large/d7b90c85ly1fxa10q1dw8j21md0erh7h.jpg)


图7

<div style="width: 550px; margin: auto">![](https://wx2.sinaimg.cn/large/d7b90c85ly1fxa10t8wnwj20w90j917z.jpg)

表7

<div style="width: 550px; margin: auto">![](https://wx3.sinaimg.cn/large/d7b90c85ly1fxa10wme4cj21230bo76w.jpg)

表8

<div style="width: 550px; margin: auto">![](https://wx1.sinaimg.cn/large/d7b90c85ly1fxa110cw5nj210x08ntao.jpg)

表9

<div style="width: 550px; margin: auto">![](https://wx2.sinaimg.cn/large/d7b90c85ly1fxa114pzqwj21a60estck.jpg)

表10

<div style="width: 550px; margin: auto">![](https://wx3.sinaimg.cn/large/d7b90c85ly1fxa118hr75j21oq08rwhy.jpg)

表11

<div style="width: 550px; margin: auto">![](https://wx2.sinaimg.cn/large/d7b90c85ly1fxa11cauxpj21pp0a278n.jpg)

表12

<div style="width: 550px; margin: auto">![](https://wx2.sinaimg.cn/large/d7b90c85ly1fxa11fzegtj20w90ba0va.jpg)

表13

<div style="width: 550px; margin: auto">![](https://wx1.sinaimg.cn/large/d7b90c85ly1fxa11krz8mj20wq0gedk5.jpg)

表14

<div style="width: 550px; margin: auto">![](https://wx2.sinaimg.cn/large/d7b90c85ly1fxa11o0ga1j20y40ca76v.jpg)


<span id = "简述 ResNet 的原理">
# 简述 ResNet 的原理

首先, ResNet 提出了一个直觉上比较合理的假设, 那就是对于一个深度为 $L$ 的神经网络, 如果我们在上面加入一层, 那么新得到的 $L+1$ 层深度的神经网络的性能至少不应该比 $L$ 层的神经网络差. 因为我们可以简单的将新加入的网络层设为前一层的拷贝(通过恒等映射即可实现), 而其他层维持原来的参数即可. 也就是说, $L+1$ 层的神经网络至少存在一个解可以达到 $L$ 层神经网络的性能. 但是, 在实际训练过程中, 我们发现有时候深层的神经网络反而具有更大的训练误差, 根据反向传播原理, 我们有理由认为这种误差是因为深度神经网络的梯度消失问题造成的. ResNet 从这个角度出发, 提出了残差模块作为网络的基本结构. 该模块首先是两层神经网络的简单叠加, 这时输入数据 $x$ 会经过这两个网络层的变换得到 $H(x)$, 在进行反向传播时, 较浅层的网络层相比于较深层的网络层更容易发生梯度消失现象(因为连乘更多). ResNet 的想法也非常直接, 它认为既然较浅层的网络层较难训练, 那么我们就直接将它短接到更深网络层, 这样, 中间被跳过的两层网络需要拟合的目标就不再是最终输出的 $H(x)$, 而是最终输出和输入之间的残差 $F(x)$, 即 $F(x) = H(x) - x$. 这样一来, 如果某一层的输出已经较好的拟合了期望结果, 那么它们的梯度就会被直接传送到两层网络之前, 从而减少了深度神经网络中由于连乘问题导致的梯度消失现象, 进而使得网络有可能拟合到更好的结果上. ResNet 的残差模块分为基本的 ResNet Block 和经过卷积分解的 Bottleneck 两种形式. 对于层数较浅的 ResNet-18 和 ResNet-34 来说, 使用的是基本的 ResNet Block 作为网络的基本单元, 而对于较深的 ResNet-50, 101, 152等, 使用的是经过卷积分解的 Bottleneck 作为网络的基本单元. ResNet 网络的整体结构还是遵循经典的五段式结构, 具体来, 第一段为 Stem 段, 使用了 $7\times 7$ 的传统卷积核, 后面四段是残差模块组成的卷积段, 每一段使用的残差模块的数量都不同一样, 深层残差网络的残差模块主要在导数第二个卷积段大量堆叠.

<div style="width: 550px; margin: auto">![](https://wx1.sinaimg.cn/large/d7b90c85ly1fx8tjak1d4j21ul0nxtgb.jpg)

<div style="width: 550px; margin: auto">![](https://wx1.sinaimg.cn/large/d7b90c85ly1fx8tjn73sij21e50iw77t.jpg)

<span id = "ResNet 中可以使用哪些短接方式">
# ResNet 中可以使用哪些短接方式

基本来说, 有三中选项可以选择
- (A). 使用恒等映射, 如果需要改变输出维度时, 对增加的维度用0来填充, 不会增加任何参数.(这种就是之前讨论的 parameter-free 的恒等短接)
- (B). 在输入输出维度一致时使用恒等映射, 不一致时使用矩阵映射以保证维度一致, 增加部分参数.
- (C). 对所有的block均使用矩阵映射, 大量增加参数

在效果上, 通常 C>B>A, 我们认为这是因为在 A 中的 zero-padded dimensions 实际上并没有进行残差学习. 但是由于 A/B/C 之间的差距比较小, 而线性变换需要引进额外的参数, 因此这是一个可以根据实际问题进行权衡的事情(通常不要用C, 因为增加的参数较多, 且性能提升并不是很明显).
对于 bottleneck 结构来说, parameter-free 的恒等短接尤其重要. 如果用矩阵映射替换了 bottleneck 中的恒等短接, 那么因为 shortcuts 需要处理的维度很高, 使得模型的 size 和时间复杂度都会加倍. 因此, 对于 bottlenect 来说, 选择恒等短接可以大大降低模型复杂度.

<span id = "如何理解所谓的残差比原始目标更容易优化">
# 如何理解所谓的残差 $F(x)$ 比原始目标 $H(x)$ 更容易优化

假设我们要学习一种从输入x到输出H(x)的mapping, 最简单的例子, 假设解空间里的函数只有两个，就是在这两个可能的mapping 函数里面选择一个更好的。
如果是非resnet的情况，那么给定 $H(5)＝5.1$ 和 $H(5)＝5.2$ 这两个函数映射, 其对应权重参数分别是 $H(x) = wx = \frac{5.1}{5} x$  和 $H(x) =w  x = \frac{5.2}{5} x$ ，这两个函数的w近似的都近似等于1, 或者说一个 $w$ 是另一个 $w$ 的1.04/1.02＝1.0196倍. 也就是说，如果用sgd来选择参数 $w$ 的话，是容易认为两个 $w$ 很像的(对数据不敏感, 导致训练慢，学错)。
但是resnet就不同了，在resnet下，原输入输出数据相当于变成了 $H(5)=0.1$ 和 $H(5)=0.2$, 这两个对应的潜在函数变成了 $F(x)= wx = \frac{0.1}{5} x$ 和 $H(x) = wx = \frac{0.2}{5} x$ , 两个 $w$ 的关系变成了一个 $w$ 是另一个 $w$ 的0.2／0.1 ＝ 2倍，所以 $w$ 的选取对于数据集非常敏感了。 这是基于这个原因，resnet里面的参数 $w$ 会更加"准确"反映数据的细微变化。(因此也更容易学到不同数据的特征)

~~另一方面, 由于恒等连接的存在, 当我们令学得的 $F(x)=0$ 时, 那么就有 $H(x)=x$, 所以如果我们将残差模块拼接在普通的 vgg 网络之后, 最终的模型性能也不会比 vgg 差, 因为后面几层相当于是一种恒等短接, 也可以认为是为模型的性能做到了一种保底措施.~~

<span id = "为什么恒等映射x之前的系数是1,而不是其他的值, 比如0.5">
# 为什么恒等映射x之前的系数是1,而不是其他的值, 比如0.5
关于为什么是 $x$　而不是 $\lambda_i x$,
主要是因为如果是 $\lambda_i x$ 的话,梯度里面  就会有一项 $\lambda_i$ 的连乘 $\prod_{i=1}^{L-1}{\lambda_i}$，就是从输出到当前层之间经过的 shortcut上的所有$\lambda_i$相乘，假如$\lambda_i$都大于 1 那经过多层之后就会爆炸，都小于1就会趋向0而引发梯度消失.

具体公式分析可见下面关于"用简单缩放来替代恒等连接"的讨论

<span id = "ResNet 到底解决了一个什么问题">
# ResNet 到底解决了一个什么问题
既然可以通过初试化和归一化（BN层）解决梯度弥散或爆炸的问题，那Resnet提出的那条通路是在解决什么问题呢？
在He的原文中有提到是解决深层网络的一种模型退化问题，但并未明确说明是什么问题！

今年2月份有篇文章，正好跟这个问题一样。The Shattered Gradients Problem: If resnets are the answer, then what is the question?大意是神经网络越来越深的时候，反传回来的梯度之间的相关性会越来越差，最后接近白噪声。因为我们知道图像是具备局部相关性的，那其实可以认为梯度也应该具备类似的相关性，这样更新的梯度才有意义，如果梯度接近白噪声，那梯度更新可能根本就是在做随机扰动。有了梯度相关性这个指标之后，作者分析了一系列的结构和激活函数，发现resnet在保持梯度相关性方面很优秀（相关性衰减从  到了  ）。这一点其实也很好理解，从梯度流来看，有一路梯度是保持原样不动地往回传，这部分的相关性是非常强的。

<span id = "ResNet 残差模块中激活层应该如何放置">
# ResNet 残差模块中激活层应该如何放置

推荐采用预激活的方式来放置激活层: BN+ReLU+Conv

<div style="width: 550px; margin: auto">![残差模块不同激活方式](https://wx2.sinaimg.cn/large/d7b90c85ly1g1g9k25d0ej20uu0s0gqf.jpg)
