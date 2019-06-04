---
title: RCNN(CVPR, 2014)
sitemap: true
date: 2018-04-09 19:27:03
categories: 计算机视觉
tags:
- 计算机视觉
- 目标检测
- 论文解读
---
**文章:** Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation
**作者:** Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik


# 训练流程
1. SS 提取出候选区域框;
2. 根据候选区域框与真实框的交并比决定正负样本标签(此时不关心框内物体的类别);
2. 送入到CNN中提取特征;
3. 将提取到的特征送入到 SVM 分类器中进行分类, 每一个类别都单独训练了一个 SVM 分类器;
4. 对每一个框进行边框回归, 学习特征图谱候选区域框到真实框的转换, 调整框的位置.

# 检测流程
1. SS 提取出候选区域框;
2. 送入到CNN中提取特征;
3. 对于每一个类别的 SVM, 计算这些框的得分, 并在每一个类别上使用 NMS 算法, 最终根据每一个框在不同类别上的得分决定框预测类别;
4. 根据类别和得分情况输出检测结果.

# Region Proposals
Selective Search.
基本思路如下:
1. 使用一个分割手段, 将图像分割成小区域
2. 查看现有小区域, 合并可能性最高的两个区域, 重复直到整张图像合并成一个区域位置. 优先合并以下区域:
    - 颜色(颜色直方图)相近的
    - 纹理(梯度直方图)相近的
    - 合并后总面积小的
    - 合并后, 总面积在其BBox中所占比例大的
3. 输出所有存在过的区域, 即所谓的候选区域

**Feature Extraction**: AlexNet (5层卷积, 2层FC, 最终特征向量的维度为 4096).

**输入图片大小**: $227\times 227$.

**正负样本划分**: 与 gt-box 的 IoU 大于 0.5 的认为是正样本, 反之认为是负样本. 训练时, mini-batch 中正样本采样数为32(over all classes), 负样本的采样数为 96. 负样本数量多是因为在真实情况下, 背景的区域数量远大于物体数量.

**分类器**: 为每个类别训练了一个 SVM.

# Appendix

# A. Object proposal transformations**

图片大小归一化.

# B. Positive vs. negative examples and softmax**

在 fine-tuning CNN 时和训练 SVM 时采用的正负样本的定义是不同的
- find-tuning: 根据候选框与真实框的交并比决定, 无视框的类别, 我们只在乎前景和后景的区别
- SVM: 当训练某一个类别的 SVM 时, 只认为与当前类别的真实框的交并比大于一定阈值的为正样本, 其他的均为负样本.

为什么要使用 SVM 而不用更加方便的 Softmax 分类器?
- 作者尝试过但是 mAP 从 54.2% 降到了 50.9%
- 下降的原因是多因素造成的, 比如对正负样本的定义, 再比如在训练 Softmax 时使用的负样本是随机采样的, 而训练 SVM 时的负样本更像是 "hard negatives" 的子集, 导致训练精度更高等等.
- 后续的 Fast RCNN 使用 Softmax 也达到了和 SVM 差不多的准确率, 训练过程更加简单.

# C. Bounding-box regression
在利用 SVM 对每个候选框预测完得分以后, 我们会用一个 class-specific bounding-box regressor 来对候选框的位置进行调整.
输入为 $N$ 个训练数据对: $\{(P^i, G^i)\}_{i=1,...,N}$, 其中, $P^i = (P^i_x, P^i_y, P^i_w, P^i_h)$ 代表预测框的中心坐标和宽高(后面为了简洁, 会在不必要的时候省略上标 $i$). 真实框的记法也相同: $G = (G_x, G_y, G_w, G_h)$. 注意, 这里的 $P$ 代表的是特征图谱上的候选框, 它与真实框之间存在一个转换关系. 为了让神经网络更加方便的学习, 我们让网络直接学习这 4 个特定的转换函数: $d_x(P), d_y(P), d_w(P), d_h(P)$, 我们可以通过这四个函数通过下面的公式将 $P$ 转换成真实框 $\hat G$ 的值. 假设我们具有一个特征图谱上的候选区域框, 用 $P=(P_x, P_y, P_w, P_h)$ 表示, 它对应的真实框用 $G=(G_x, G_y, G_w, G_h)$ 表示, 那么, 我们的目标是希望回归器能够学习到一个从 $P$ 到 $G$ 的转化(transformation), 如下所示, $\hat G$ 就是我们经过转换后得到的预测框在原始图片上的坐标和宽高.

$$\hat G_x = P_w d_x(P) + P_x$$
$$\hat G_y = P_h d_y(P) + P_y$$
$$\hat G_w = P_w exp(d_w(P))$$
$$\hat G_h = P_h exp(d_h(P))$$

上式中的 $d_x(P), d_y(P), d_w(P), d_h(P)$ 就是我们要学习的参数, 因此我们有 $d_\*(P) = w^T_\* \Phi_5(P)$, 这里的 $w_\*$ 就是神经网络中的可学习参数. 我们通过优化下面的最小二乘目标函数(ridge regression)来学习参数:

$$w_* = \arg\min_{\hat w_*} \sum^N_i (t^i_* - \hat w^T_* \Phi_5(P^i))^2 + \lambda \|\hat w_* \|^2$$

那么我们的学习目标就是使得这些参数可以满足 $\hat G = G$,  也就是说, 我们的学习目标就是令参数 $d_x(P), d_y(P), d_w(P), d_h(P)$ 无限近似于下面的 $(t_x, t_y, t_w, t_h)$:

$$t_x = (G_x - P_x) / P_w$$
$$t_y = (G_y - P_y) / P_h$$
$$t_w = log(G_w / P_w)$$
$$t_h = log(G_h / P_h)$$

![](https://wx2.sinaimg.cn/large/d7b90c85ly1g113krcrqdj20xy0exq38.jpg)

以上图举例来说, 红色的框 $P$ 代表的是原始的 proposal, 绿色的框 $G$ 代表的是真实的 Ground Truth, 我们的目标时寻找一种函数关系使得 $P$ 经过映射后可以得到一个更接近 $G$ 的回归窗口 $\hat G$. 也就是说, 边框回归的目的既是：给定一个 box 坐标 $(P_x,P_y,P_w,P_h)$, 我们要寻找一种映射关系 $f$, 使得 $f(P) = \hat G$, 其中, $\hat G \approx G$.

<span id = "简述 Selective Search 的原理">
# 简述 Selective Search 的原理

首先, 首先利用分割算法(Graph-Based Image Segmentation, 2004, IJCV, 贪心)得到一些初始化的区域, 然后计算每个相邻区域的相似性, 相似性的计算依赖于颜色相似性和纹理相似性, 同时给较小的区域赋予更多的权重, 也就是优先合并小区域(否则大区域有可能会不断吞并周围区域, 使得多尺度之应用了在局部区域, 而不是在每个位置都具有多尺度), 接着找出相似性最大的区域, 将它们合并, 并计算新合并的区域与其他相邻区域的相似性, 重复这个过程, 直到所有的区域被合并完为止.

<span id = "简述 Bounding Box 的回归方式">
# 简述 Bounding Box 的回归方式

在 R-CNN 的边框回归中, 我们不是直接学习真实框的坐标, 而是学习从 Proposals 到 真实框的一个偏移变换函数, 具体来说, 对于中心点, 需要学习的是 proposal 和 真实框相对位移, 这个位移会用 proposal 的宽和高进行归一化, 对于宽和高, 需要学习的是真实框相对于 proposal 的 log 缩放度.

$$t_x = (G_x - P_x) / P_w$$
$$t_y = (G_y - P_y) / P_h$$
$$t_w = log(G_w / P_w)$$
$$t_h = log(G_h / P_h)$$

<span id = "Bounding box 回归的时候, 为什么不直接对坐标回归, 而是采用偏移量和缩放度">
# Bounding box 回归的时候, 为什么不直接对坐标回归, 而是采用偏移量和缩放度

最主要是为了获得对物体的尺度不变性和唯一不变性, 一般来说, 对于大物体和小物体, 在没用使用归一化时, 对于相同的偏移量, 大物体可能只偏移了一点, 而小物体可能会偏移很多. 具体来说, 对于两个不同尺度但是完全一样的物体, 我们得到的特征应该是相似的, 因此, 我们最终学习出来的结果也应该是相似的, 但是如果采用直接学习坐标的方式, 那么由于此时大物体和小物体虽然在相对位移上是相同的, 但是绝对位移和坐标值是不同的, 因此, 我们最终学习出来的结果就会不一样, 这就相当于给了一个函数相同的输入, 但是却得到了不同的结果, 这样就很难让网络学习.

边框回归(BoundingBoxRegression)详解
https://blog.csdn.net/zijin0802034/article/details/77685438

# 为什么当 Region Proposals 和 Ground Truth 较接近时的 IoU 较大时, 可以认为是边框回归函数是线性变换?

当输入的 Proposal 与 Ground Truth 相差较小时(RCNN 设置的是 IoU>0.6)， 可以认为这种变换是一种线性变换， 那么我们就可以用线性回归来建模对窗口进行微调， 否则会导致训练的回归模型不 work (当 Proposal跟 GT 离得较远，就是复杂的非线性问题了，此时用线性回归建模显然不合理). 对于这一段的话解释如下:

首先, Log 函数肯定不满足线性函数的定义, 但是根据极限的相关定义, 我们如下面的等式成立:

$$lim_{x\rightarrow0}log(1+x) = x$$

根据上面的公式, 我们可以对公式 $t_w$ 作如下推导:

$$t_w = log(G_w / P_w) = log(\frac{G_w + P_w - P_w}{P_w}) = log(1 + \frac{G_w - P_w}{P_w})$

从上式我们可以看出, 当 $G_w - P_w = 0$ 的时候, 回归函数 $t_w$ 可以看做是线性函数.

这里还有一点疑问: 从公式来说, $t_x$ 和 $t_y$ 本身就已经是线性函数, 而 $t_w$ 和 $t_h$ 只需要 Proposals 和 Ground Truth 的宽高相似即可满足线性回归条件. 那么为什么必须要 IoU 较大才可以? 不是只要宽高相似就可以吗?

边框回归(BoundingBoxRegression)详解
https://blog.csdn.net/zijin0802034/article/details/77685438
