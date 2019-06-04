---
title: IdentityMappings (ECCV, 2016)
sitemap: true
categories: 计算机视觉
date: 2018-11-15 15:10:48
tags:
- 计算机视觉
- 网络结构
- 论文解读
---

**文章:** Identity Mappings in Deep Residual Networks
**作者:** Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun
**备注:** MSRA

# 核心亮点


# 摘要


# Identity Mappings

在第一篇ResNet论文中提到, 1202层的ResNet出现了性能退化的问题.  本文主要是对residual block的一个改进, 也就是将BN和ReLU放到计算权重之前进行, 称为"预激活" , 如下图所示:

![](https://wx3.sinaimg.cn/large/d7b90c85ly1fvoavon0q5j20rp0dvac3.jpg)

# 关于Deep ResNet的分析


https://blog.csdn.net/wspba/article/details/60750007


# 用简单缩放来代替恒等连接

设计一个简单的缩放: $h(x_l) = \lambda_l x_l$ 来代替恒等连接:

$$x_{l+1} = \lambda_l x_l + F(x_l, W_l)$$

于是,继续通过递归我们可以得到:

$$x_L =(\prod_{i=l}^{L-1}) x_l + \sum_{i=l}^{L-1}{\hat F(x_i, W_i)}$$

对上面的式子求导, 可以得到:

![](https://wx2.sinaimg.cn/large/d7b90c85ly1fvod04no6qj20da035weh.jpg)

可以看到, 在该式子中, 由于 $\lambda$ 连乘项的存在, 可能会使这个因子变的很大或者消失, 从而阻断从短接反向传来的信号, 进而对优化造成困难


# 关于Skip Connections的其他实验

## Constant scaling

考虑对 $F$ 的缩放, 训练结果显式优化变的更加困难,  因此不建议缩放

因为 $F$ 对应的是连加项, 不会出现连乘项, 所以不能说因子很指数增长或消失

## Exclusive gating

## Shortcut-only gating

## 1×1 卷积shortcut

在ResNet34的时候, 使用了1×1的卷积(即方案C),  并且取得了较好的结果,  表明1×1卷尺短接还是有效果的.  但是当残差单元变多时, 并不能起到很好的效果

值得注意的是1××\times1的卷积捷径连接引入了更多的参数，本应该比恒等捷径连接具有更加强大的表达能力。事实上，shortcut-only gating 和1××\times1的卷积涵盖了恒等捷径连接的解空间(即，他们能够以恒等捷径连接的形式进行优化)。然而，它们的训练误差比恒等捷径连接的训练误差要高得多，这表明了这些模型退化问题的原因是优化问题，而不是表达能力的问题

## Dropout shortcut
这个在统计学上相当于给短接强加了一个 $\lambda=0.5$ 的缩放, 这和constant scaling很类似, 同样阻碍了信号的传播

## 激活函数的使用

通过重新安排激活函数(ReLU和/或BN)来使得 $f$ 成为一个恒等映射. 最原始的残差连接如下图a所示, b~e展示了其他形式. 图中所有单元的组成成分相同, 只是顺序不同, e形式取得了最后的结果, 也就是full pre-activation形式

![](https://wx3.sinaimg.cn/large/d7b90c85ly1fvodw0dpnmj20sr0p2433.jpg)

对以上形式讨论如下:

**BN after addition:** 图b, 此种做法正好反其道而行之, 此时 $f$ 不仅包含了 ReLU, 还包含了BN, 最终导致的结果就是阻碍了信息的传递, 是性能下降

**ReLU before addition:**  图c, 这是一种很直接的做法, 也很天真,  直接将ReLU移动到加法之前, 这导致了F的输出非负, 然我们我们希望残差函数的值是在政府无穷区间内的

**Post-activation or Pre-activation:** 如图c和d, 图d通过一种非对称的转换, 使得当前块的激活函数成为一个块的预激活项, 具体转换如下图所示:

![](https://wx4.sinaimg.cn/large/d7b90c85ly1fvoe4wy57wj20su0iatbc.jpg)

对上图的解释就是, 在原始的设计中, 激活函数会对两条路径的下一个残差单元造成影响:

$$y_{l+1} = f(y_l) + F(f(y_l), W_{l+1})$$

而通过这种非对称的转换, 能够让激活函数 $\hat f$ 对于任意的　$l$ , 都只对$F$ 路径造成影响:

$$y_{l+1} = y_l + F(\hat f(y_l), W_{l+1})$$

于是, 新的激活函数就变成了一个恒等映射.

后激活与预激活的区别是有元素级加法的存在而造成的,一个含有N层的平铺网络，包含有N−1个激活层(BN/ReLU)，而我们如何考虑它们是否是后激活或者预激活都不要紧。但是对附加的分支层来说，激活函数的位置就变得很重要了。只使用ReLU预激活的结果与原始ResNet-110/164的已经很接近。

**只是用ReLU的预激活vs完全预激活**

从图d中, 我们可以看到, ReLU层不与BN层连接使用，因此无法共享BN所带来的好处, 因此, 很自然的,我们将BN层移到ReLU的前面, 最终, 性能获得了较大的提升, 超过了原始ResNet-110/164

### 分析

文章发现预激活的影响具有两个方面:
- 由于$f$变成了恒等映射,优化变的更加简单
- 在预激活中使用BN能提高模型的正则化能力
