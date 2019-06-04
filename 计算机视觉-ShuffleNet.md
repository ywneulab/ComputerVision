---
title: ShuffleNet
sitemap: true
categories: 计算机视觉
date: 2018-08-20 15:05:02
tags:
- 计算机视觉
- 网络结构
- 论文解读
---

**文章:** ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices
**作者:** Xiangyu Zhang, Xinyu Zhou, Mengxiao Lin, Jian Sun
**备注:** Megvii Inc

# 摘要

我们介绍了一种计算效率极高的 CNN 结构, 名为 ShuffleNet, 它是专门为计算能力非常有限的移动设备设计的. 这个新的网络结构采用了两种新的计算操作: 点态组卷积(pointwise group convolution) 和 通道洗牌(channel shuffle). 实验显示, 效果很好.

# Introduction

我们注意到, 最先进的架构, 如 Xception 和 ResNeXt, 在非常小的网络中, 由于成本较高的密集 1x1 卷积, 使得计算效率变的较低. 因此, 本文提出使用点态组卷积(pointwise group convolution)来降低 1x1 卷积的计算复杂度. 为了克服 group convolution 带来的副作用, 我们提出了一种新的 channel shuffle 操作来帮助信息在特征信道间流动. 基于这两种技术, 我们构建了一个名为 ShuffleNet 的高效结构.

# Related Work

**Efficient Model Designs:** GoogLeNet, SqueezeNet, ResNet, SENet, NASNet

**Group Convolution:** AlexNet 首先提供 group conv 的概念, ResNeXt, Xception, MobileNet.

**Channel Shuffle Operation:** Interleaved group convolutions for deep neural networks (paper)

**Model Acceleration(在 Inference 阶段加速以训练好的模型):** Pruning network connections, Channels Reduce, Quantization, Factorization. **利用 FFT 或者其他方法优化卷积计算**, 大模型到小模型的知识迁移.

# Approach

## Channel Shuffle for Group Convolutions

现代的卷积神经网络通常由相同结构的 building blocks 组成. 其中, Xception 和 ResNeXt 在 building blocks 中引入了高效的深度可分离卷积(Depthwise Separable Conv)和组卷积(Group Conv), 从而在表征能力和计算成本之间取得了很好的平衡. **但是, 我们注意到这两种设计没有完全对 1x1 卷积进行分析, 而实际上该卷积也需要较大的计算量.** 例如, 在 ResNeXt 中, 只有 3x3 卷积层配置了 Group Conv, 因此, 对于 ResNeXt 中的每一个残差单元来说, Pointwise Conv 都要占用 93.4% 的乘法加法操作. 在微型网络中, 成本高的 Pointwise Conv 会利用有限的通道数来满足复杂度的约束, 但是这样可能会损害精度.


为了解决这个问题, 一个简单的解决方案是利用通道稀疏连接(channel sparse connections). 通过确保每个卷积只对对应的输入通道进行运算, Group Conv 显著降低了计算成本. 但是, 如果将多个 Group 叠加在一起, 就会产生一个副作用: 来自某个通道的输出只来自于一小部分的输入通道. 图1(a)展示了两个 Group Conv 叠加的情况, 很明显, 某个 Group 的输出至于 Group 内的输入有关, 这种性质会阻塞通道之间的信息流, 降低表征能力.

<div style="width: 550px; margin: auto">![图1](https://wx3.sinaimg.cn/large/d7b90c85ly1g1jpz4z41nj217i0ijn0e.jpg)

如果我们允许 Group Conv 获取来自不同 groups 的输入数据, 如图1(b)所示, 那么输入和输出通道将会完全相关. 具体来说, 对于前一个 group layer 生成的 feature map, 我们可以先将每个 group 内部的通道划分成若干个 subgroups, 然后给下一个 group layer 的中每一个 group 提供不同的 subgroups. **这可以利用通道洗牌(channel shuffle)操作高效且优雅的实现, 如图1(c)所示:** 假设一个卷积层具有 $g$ 个 groups, 它的输出通道数为 $g\times n$. 我们首先将输出的通道数 reshape 成 $(g, n)$, 然后利用 transposing 进行转置, 最后利用 flatten 将其开展, 并作为下一层的输入. 注意到, 即使两个卷积层的组数不同, 该操作依然有效. 此外, channel shuffle 也是可微的, 这意味着它可以嵌入到网络结构中进行端到端的训练. 信道洗牌操作使得构建具有多组卷积层的更强大的结构成为可能。在下一小节中，我们将介绍一种具有信道洗牌和组卷积的高效网络单元

## ShuffleNet Unit

利用 channel shuffle 的优势, 我们提出了一种专为小型网络设计的 ShuffleNet unit. 我们从 ResNet 的 bottleneck 的设计原则开始. 首先, 我们将残差分支的 3x3 卷积换成计算效益更高的 Depthwise Conv, 如图2(a)所示. 然后, 我们将第一个 1x1 卷积层换成 Group Conv(Pointwise) 和 Channel Shuffle, 如图2(b)所示, 第二个 Group Conv(Pointwise) 的目的是恢复通道数以匹配 shortcut 连接, 为了简单起见, 我们没有这里使用额外的 Channel Shuffle 操作, 因为这已经可以产生不错的效果了. BN 和 ReLU 会用在每个卷积层之后, 只不过根据 Xception 的建议, **我们没有在 Depthwise Conv 之后使用 ReLU.** 对于需要下采样(with stride)的情况, 我们做了两点修改, 如图2(c)所示: 首先, 在 shortcut path 上添加一个 3x3 的 avg pooling; 然后, 用 channel concatenation 操作替换 element-wise addition 操作, 这使得在不增加额外计算成本的情况下, 可以方便的扩大通道尺寸.

<div style="width: 550px; margin: auto">![图2](https://wx4.sinaimg.cn/large/d7b90c85ly1g1jpzhxydej217o0jvgoi.jpg)

得益于 Pointwise Group Conv 和 Channel Shuffle 的结合, ShuffleNet 中的所有组件都可以高效的进行计算. 和 ResNet 与 ResNeXt 相比, 我们的结构在相同的设定下具有更少的复杂度. 例如, 当给定输入尺寸为 $c\times h\times w$, 而 bottleneck 的通道数为 $m$ 时, ResNet unit 需要 $hw(2cm + 9m^2)$ FLOPs, ResNeXt 需要 $hw(2cm + 9m^2/g)$ FLOPs, 而 ShuffleNet 只需要 $hw(2cm/g + 9m)$ FLOPs. 也就说, 当给定计算资源限制后, ShuffleNet 可以使用更大的特征图谱, 这对于小型网络来说非常重要, 因为小型网络通畅没有足够的通道来处理信息.

此外, ShuffleNet 中的 Depthwise Conv 只在 bottleneck 的特征图谱上执行. 虽然深度卷积通常具有非常低的理论复杂度, 但我们发现在低功耗移动设备上很难有效地实现, 这可能是因为与其他密集操作相比, Depthwise Conv 的计算/内存访问率更低.

## Network Architecture

基于 ShuffleNet Units, 我们在表1中给出了整个 ShuffleNet 的网络结构. 该网络主要有一系列的 ShuffleNet units 组成, 并且可以划分为三个阶段. **每个阶段的第一个 building block 的步长为2(stride=2).** 每个 Stage 内的其他超参数都相同, 并且当进入下一个 Stage 时, Channels 的输出数量都翻倍. 和 ResNet 类似, 我们设置 bottleneck 的 channels 的数量为最终输出通道数量的 1/4. **我们的目标是提供一个尽可能简单的参考设计, 尽管我们发现更进一步的超参数调优可能会产生更好的结果.**

<div style="width: 550px; margin: auto">![表1-2](https://wx2.sinaimg.cn/large/d7b90c85ly1g1jpzzw2hwj21780q9dmy.jpg)

在 ShuffleNet units 中, Group 的数量 $g$ 控制了 Pointwise Conv 的连接稀疏性. 表1研究了不同组数带来的影响, 同时我们会适当调节输出通道数, 以确保总体的计算成本不变. 很明显, 在给定计算资源限制的情况下, 更多的组数可以具有更多的输出通道数, 这样可以对更多的信息编码, 但是由于相应的输入通道有限, 因此这也可能导致单个卷积 filter 的性能下降.

要将网络自定义为所需的复杂性, 我们可以简单地在通道的数量上应用一个比例因子 $s$. 例如, 我们将表1中的网络表示为 "ShuffleNet 1x", 那么 "ShuffleNet sx" 表示将 ShuffleNet 1x 中的 filters 数量乘以 $s$, 因此总体复杂度大约是 ShuffleNet 1x 的 $s^2$ 倍.

# Experiments

weight decay: 4e-5.
lr: linear-decay learning rate policy.
batch size: 1024

Pointwise Group Convolutions: 如表2所示

Channel Shuffle: 如表3所示, 当 group 越大时, Channel Shuffle 带来的效益越高

<div style="width: 550px; margin: auto">![表3](https://wx4.sinaimg.cn/large/d7b90c85ly1g1jq09fei8j20xf0atacn.jpg)

Comparison with Other Structure Units: 如表4所示, 在小模型上, ShuffleNet 的效果很好. 没有使用 Inception 进行比较, 因为 Inception 包括了太多的超参数, 很难将其模型缩小化. 表6 展示了在达到相同精度时, 不同模型需要的 MFLOPs.

<div style="width: 550px; margin: auto">![表4-5](https://wx1.sinaimg.cn/large/d7b90c85ly1g1jq0lbp91j21790m3q97.jpg)
<div style="width: 550px; margin: auto">![表6-8](https://wx4.sinaimg.cn/large/d7b90c85ly1g1jq0xl5qrj21810u07e3.jpg)

Comparison with MobileNets and Other Frameworks: 表5显示了 ShuffleNet 与 MobileNet 之间的性能对比.

Generalization Ability: 测试了 ShuffleNet 在目标检测上的通用性, 结果如表7所示.

Actual Speedup Evaluation: 表8显示了在 ARM 平台上的实际 Inference 速度.

<span id = "简述 ShuffleNet 的原理">
# 简述 ShuffleNet 的原理

ShuffleNet 从深度卷积模型的计算效率角度出发, 对 ResNet 中的 bottleneck 模块进行改进, 首先利用 Pointwise Group Conv 来替换 ResNet 中计算成本较高 1x1 卷积, 但是 Group Conv 操作本身会阻塞特征图谱通道之间的信息流动, 因此, ShuffleNet 提出使用 Channel Shuffle 操作来缓解这个问题, 在实际使用中 Channel Shuffle 可以利用 reshape + 转置 + flatten 的方法快速实现(为了简单, 只在第一个 1x1 卷积后使用了 Channel Shuffle). 然后, ShuffleNet 用 Depthwise Conv 替换了原本的 3x3 卷积, 进一步降低计算量. 在进行 Downsample 时, 会将 shortcutpath 上添加步长为2的 avg pooling 层, 同时会将残差分支的 Depthwise Conv 的步长置为2. 用改进后的残差模块作为基本单元, ShuffleNet 的网络结构最开始由 3x3 的卷积层和最大池化层组成, 二者的步长均为2, 也就说这里的 Downsample 总步长为 4. 然后是三个不同的 Stage(3,7,3), 每个 Stage 最开始第一个 building block 的步长为2(stride=2). 当进入下一个 Stage 时, Channels 的输出数量都翻倍. 最后是由 GAP+FC+Softmax 组成分类层. ShuffleNet 最主要的特点是可以在较少的计算资源限制下达到更高的精度和计算效率. 举例来说, 当给定输入尺寸为 $c\times h\times w$, 而 bottleneck 的通道数为 $m$ 时, ResNet unit 需要 $hw(2cm + 9m^2)$ FLOPs, ResNeXt 需要 $hw(2cm + 9m^2/g)$ FLOPs, 而 ShuffleNet 只需要 $hw(2cm/g + 9m)$ FLOPs. 也就说, 当给定计算资源限制后, ShuffleNet 可以使用更大的特征图谱, 这对于小型网络来说非常重要, 因为小型网络通畅没有足够的通道来处理信息.

<span id = "简述 ShuffleNet 和 MobileNet 的区别">
# 简述 ShuffleNet 和 MobileNet 的区别

ShuffleNet 使用的是 Pointwise Group Conv, 而 MobileNet 使用的是 Pointwise Conv.
在 3x3 的卷积层上, 二者相同, 都使用的是 Depthwise Conv(不改变通道数).
