---
title: Soft-NMS(ICCV, 2017)
sitemap: true
categories: 计算机视觉
date: 2018-10-24 14:24:58
tags:
- 计算机视觉
- 论文解读
- 目标检测
---

**作者:** Navaneeth Bodla, Bharat Singh, Rama Chellappa, Larry S.Davis
**发表:** ICCV2017
**机构:** Center For Automation Research

# 核心亮点

**提出了一种NMS的变体, 通过利用该变体, 基本上可以提升任何模型的检测准确率**
作者们提出了一种新式的NMS算法, 并且利用该算法, 可以普遍提高当前现有模型的召回率(尤其是面对重叠程度大的物体), 同时, 由于可以不增加复杂度的情况下直接用该算法替换传统NMS算法,  因此, 在替换SoftNMS时, 无需更改模型的任何参数, 也无需重新训练模型, 就可以达到提升召回率的作用. (对mAP的提升大约为1%左右)

<div style="width: 550px; margin: auto">![](https://wx4.sinaimg.cn/large/d7b90c85ly1fwskrv3tdtj20rt0p61kx.jpg)

# 论文细节

**传统NMS:** 先将score倒序排列, 然后取socres值最大的box并将其置于final box列表中, 计算所有剩余box与该box的重叠度, 大于某一阈值的就将其删除, 然后迭代的使用此方法, 直到final box数量达到要求或者没有多的box了.(在FasterRCNN中, 生成候选框时会使用一次NMS, 预测还会分别对每一个类别都使用一次NMS)



**传统NMS存在的问题:** 由上面的描述可知, 如果两个物体本身的重叠度过大, 那么其中一个物体的框就会被删除(score被置为0), 从而导致漏解.

**Soft-NMS:** 在将具有最大score的box置于 picked box 之后, 计算所有剩余 box 与该 box 的重叠度, 对于那些重叠度大于一定阈值的 box, 我们并不将其删除, 而仅仅只是根据重叠程度来降低那些 box 的 socre, 这样一来, 这些 box 仍旧处于 box 列表中, 只是 socre 的值变低了. 具体来说, 如果 box 的重叠程度高, 那么 score 的值就会变得很低, 如果重叠程度小, 那么 box 的 score 值就只会降低一点, Soft-NMS 算法伪代码如下图所示:

<div style="width: 550px; margin: auto">![](https://wx2.sinaimg.cn/large/d7b90c85ly1g0mczrbnkfj20xe0ipdi2.jpg)

<div style="width: 550px; margin: auto">![](https://wx2.sinaimg.cn/large/d7b90c85ly1fwsmozvvpsj20oh0omkb4.jpg)

设 $s_i$ 为第 $i$ 个 box 的 score, 则在应用 SoftNMS 时各个 box score 的计算公式如下:

$$s_i = \begin{cases} s_i, & iou(M, b_i) < N_t \\ s_i(1-iou(M, b_i)), & iou(M, b_i) \geq N_t \end{cases}$$

上式过于简单直接, 为了函数的连续性, 文章改用了高斯惩罚系数(与上面的线性截断惩罚不同的是, 高斯惩罚会对其他所有的 box 作用):

$$s_i = s_i e^{\frac{iou(M, b_i)^2}{\sigma}} \forall b_i\notin D$$
