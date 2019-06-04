---
title: Image-Generation-from-Scene-Graphs
sitemap: true
date: 2018-07-12 14:01:35
categories: 计算机视觉
tags:
- 计算机视觉
- 论文解读
---

本篇论文解读的排版主要参见原文的格式，针对原文中的每一个小节进行展开，有的是对原文的一个提炼和简单概括，有的是对原文中一些要点的补充和说明。

# 摘要

&emsp;&emsp;近几年来（至2018），针对某些特定目标（花，鸟等）的图片生成已经取得了令人激动的研究成果，但是当文本描述中包含多个物体和物体之间的关系时，仍然具有一些困难。 为了克服这一点，本文提出了从场景图来生成图片的方法，该方法可以推理出具体的物体和物体之间的关系。 本文的模型使用“图卷积层”（graph convolution）来处理输入的“图”（graph），通过预测物体的bounding boxes和segmentation masks来计算“场景布局”（scene layout），然后利用“级联精细化网络”（cascaded refinement network）将“场景布局”转换成一张图片输出。在训练网络时，通过对抗式的训练一对儿discriminators来确保生成图片的真实感。 本文在Visual Genome和COCO-Stuff数据集是验证了以上模型的有效性，结合高质量的生成图片、消融实验和人工调研的方法证明了本文提出的模型可以生成含有多个物体的复杂图片。

# 介绍——Introduction

What I cannot create，I do not understand —— Richar Feynman

&emsp;&emsp;要想让计算机生成图片，就需要令其对图片有更深刻的理解。

&emsp;&emsp;为了达到以上目标，目前在text to image synthesis领域已经有许多的工作成果。这些模型可以在limited domains内产生十分惊人的结果，但是当文本信息变得复杂起来时，其生成出来的图片就不尽人意了。

![图片显示失败，请联系博主进行维护](https://s1.ax1x.com/2018/07/15/PMTISI.jpg "图1")

&emsp;&emsp;句子通常都是由一个单词接一个单词组成的线性结构，但是，一个复杂的句子，其内部携带的信息，通常需要由基于物体的“场景图”在具体表示，“场景图”中包含物体和物体之间的关系。“场景图”作为表征图片和文本的强有力的工具，常常被用于语义图片检测、提高和评价图片描述领域中。也有关于将自然语言或图片转换成“场景图”的研究。

22,1，31,47,32,36，57,58

&emsp;&emsp;本篇文章的主要研究目的是在“场景图”约束条件下，生成带有多个物体和物体之关系的复杂图片。这一任务也带来了许多新的挑战。首先，必须要找到可以处理场景图的输入方法，对此本文使用了“graph convalution network”，它可以沿着“场景图”的“边”将信息传递。处理完“图”以后，还必须建立图结构的输入与二维图片的输出之间的联系。为此 ，本文通过预测图中所有物体的bounding boxes和segmentation masks构建了“场景布局（scene layout）”。得到“布局”以后，就需要生成图片，本文使用了“cascaded refinement network（CRN）”，它可以不断增大空间尺寸，生成图片。 最后，我们必须确保生成的图片具有一定的真实感并且包含多个可辨识的物体，为此我们针对image patches和generated objects训练了一对儿discriminator网络。另外，所有的模型可以进行端到端的联合训练。

&emsp;&emsp;我们在2个数据集上进行了实验：Visual Genome（提供人工标注的场景图）和COCO-stuff（从真实的物体位置生成场景图）。两个数据集的生成结果都证明了本文提出的方法的可以生成包含多个物体并且反映它们之间关系的复杂图片。同时，还进行了综合的消融实验来验证本文提出的模型中每一部分的有效性。

# 相关工作——Related Work

&emsp;&emsp;**生成图片模型 Generative Image Models** 目前，生成模型主要可分为三类：Generative Adversarial Networks（GANs）、Variational Autoencoders（VAE）和基于像素的似然法autoregressive approaches。

[12,40,24,38,53]

&emsp;&emsp;**条件图片生成 Conditional Image Synthesis** 通过在生成图片时向GAN网络添加条件的方式来控制最终输出图片的结果。由两种不同方法：一是将条件作为附加信息同时送入到generator和discriminator中，二是强制让discriminator去预测图片的label。本文选择后者。

[10,35,37,42,59,41,43,6,9,21,4,5,20,27,28,55,56,22]

&emsp;&emsp;**场景图 Scene Graph** 表现为有向图，它的结点是物体，边是物体之间的关系。场景图多倍用于图片检索、图片描述评价等，有些工作也场景从文本或图片中生成场景图

[1,47,32,36,57,58,26]

&emsp;&emsp;**针对图的深度学习 Deep Learning on Graphs** 有的工作是对图进行embedding学习，类似于word2vec，但这与本文的方法不同，因为本文在进行一次前向计算时，传过来的图都是新的。与本文方法更相关的是Graph Neural Networks，它可以对任意的图进行处理。

[39,51,14,34,11,13,46,8,49,48,7,19,29,54,2,15,25]

# 方法——Methond

&emsp;&emsp;我们的目标是得到一个模型，该模型的输入描述物体和它们之间关系的“场景图”，输出是基于该场景图的图片。主要的挑战和困难有三个方面：一、必须找到可以处理“场景图”输入的方法;二、确保生成的图片可以真实反映出场景图中缩描述的物体;三、确保生成的图片具有真实感。

&emsp;&emsp;如图2所示，本文通过“image generation network f”将场景图转换成图片。该网络的inputs是场景图 $G$ 噪声变量 $z$ ，ouputs是 $\hat I = f(G,z)$ 。

<center><div style="color:orange;
border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
">图2</div>
</center>

![图片显示失败，请联系博主进行维护](https://s1.ax1x.com/2018/07/15/PMT7Of.jpg "图2")



&emsp;&emsp;场景图经过“图卷积网络”后，会得到每个物体的embedding vectors，如图2和图3所示，每一层“图卷积层”都会沿着图的边将信息混合在一起。

<center><div style="color:orange;
border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
">图3</div>
</center>

![图片显示失败，请联系博主进行维护](https://s1.ax1x.com/2018/07/15/PMT4fA.jpg "图3")



&emsp;&emsp;本文利用从图卷积网络中得到的object embedding vectors来预测每个物体的bounding boxes和segmentation masks。将它们结合起来形成一个“场景图 scene layout”，如图2中心所示，场景布局相当于是场景图和图片中间媒介。

&emsp;&emsp;最终将布局送入到CRN中，生成图片，如图2右边所示，CRN中会不断将布局的尺寸放大，指定生成新的图片为止。本文训练的生成器是CRN网络 $f$ 和两个分辨器 $D_{img}$ 和 $D_{obj}$ ，它们可以确保图片的真实感以及图片中物体的可识别力。关于这部分的详细介绍可以查看后文以及附加材料中的内容。

&emsp;&emsp;**场景图 Scene Graphs** 给定一个物体类别集合 $C$ 和一个关系集合 $R$ 。一个场景图可以用一个元组 $(O,E)$ 表示，其中 $O \subseteq C$ ， $E \subseteq O \times R \times O$ 。在处理的第一阶段，使用学习好的embedding layer将场景图中的结点和边转换成一个dense vector，就像语言模型中的那样。

&emsp;&emsp;**图卷积网络 Graph Convolution Network** 为了实现端到端的处理，本文需要一个可以对场景图进行处理的神经网络模型，为此，采用了由若干图卷积层构成的图卷积网络。 本文的图卷积网络与传统的卷积网络的工作方式类似：给定一个input graph，它每个结点和边的vecotrs维度为 $D_{in}$ ，然后经过一层图卷积层以后，就会生成一个新的vector，其维度为 $D_{out}$ 。（输出结点的值是关输入结点周围像素的函数）。 具体来说，对于所有的 $o_i \in O , (o_i,r,o_j) \in E$ ，给定输入向量 $v_i,v_r \in R^{D_{in}}$ 都会计算出输出向量 $v_i^{'} , v_r^{'} \in R^{D_{out}}$ 。 对于所有的结点和边，都会使用3个函数： $g_s , g_p , g_o$ ，其接受的输入为一个向量的三元组 $(v_i, v_r, v_j)$。&emsp;计算边的输出向量时，直接使用 $v_r^{'} = g_p(v_i, v_r, v_j)$ 。而更新结点的值时较为复杂，因为结点往往连接了很多条边。对于每条始于 $o_i$ 的结点，都利用 $g_s$ 去计算候选向量（candidate vector），收集到所有的候选向量以后，将其放置于集合 $V_i^s$ 中。用 $g_o$ 以同样的方式处理止于 $o_i$ 的边。公式表示如下：
$$ V_i^s = {g_s(v_i, v_r, v_j) : (o_i, r, o_j) \in E} $$
$$ V_i^o = {g_o(v_j, v_r, v_i) : (o_j, r, o_i) \in E} $$
然后再利用公式 $v_i^{'} = h(V_i^s \cup V_i^o)$ 计算得到物体 $o_i$ 的输出向量 $v_i^{'}$ （ $h$ 为池化操作）。有关计算的例子可以看图3。在本文中，函数 $g_s , g_p , g_o$ 的实现采用了一个单一网络，该网络会将输入向量连接起来，然后送到一个多层感知机（MLP）当中。pooling 函数 $h$ 会将输入结果进行平均值池化，然后送到MLP当中。


<center><div style="color:orange;
border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
">图4</div>
</center>

![图片显示失败，请联系博主进行维护](https://s1.ax1x.com/2018/07/15/PMTT6P.md.jpg "图4")

&emsp;&emsp;**场景布局 Scene Layout** 为了生成图片，本文利用object embedding vectors去计算场景布局，该布局给出了要生成的图片的2D结构。本文利用图4中的object layout network来预测每个物体的bounding boxes和 segmentation masks，进而生成场景布局。  object layout networks接受形状为 $D$ 的embedding vector $v_i$ ，并把它送入到一个 mask regression network中去预测形状为 $M \times M$ 的soft binary mask $\hat m_i$ ，同时也送到一个 box regression network中去预测bounding box $\hat b_i = (x_0, y_0, x_1, y_1)$ 。 我们将 embedding vectors $v_i$ 和 mask $\hat m_i$ 逐个元素相乘，得到一个masked embedding ， 其shape为 $D \times M \times M$ ，然后，再利用双线性插值法结合物体的bounding box得到一个object layout。将所有的object layout相加，最终得到scene layout。在训练阶段，我们使用ground-truth bounding boxes来计算scene layout，在测试阶段我们使用预先预测好的bounding boxes进行计算。

&emsp;&emsp;**级联精细化网络 Cascaded Refinement Network** 在给定场景布局以后，本文使用CRN来根据场景布局生成图片。一个CRN网络包含了一系列的convolutional refinement modules，modules之间的spatial resolutoin会不断变大（double），最终达到预定义的图片大小。 每个module都以scene layout（downsampling到当前module接受的大小）和前一层module的输出结果。 这两部分输入沿着channel连接在一起,送到2层3×3的卷积层里，然后利用最近邻插值对结果进行upsampling，之后继续传送到下一个module中。第一个module以scene layout和高斯噪声 $z \sim p_z$ 作为输入。把从最后一个module得到的结果再送到两个final convolution layers中去，生成最终的图片。

&emsp;&emsp;**分辨器 Discriminators** 本文训练了两个分辨器 $D_{img}$ 和 $D_{obj}$
- patch-based image discriminators $D_{img}$ ：确保生成图片的overall appearance是realistic的。利用全卷积网络实现。
- object discriminator $D_{obj}$ ：确保图片中的每个物体都是recognizable并且realistic的。分别利用辅助分类器 auxiliary classifier和全卷积网络实现。

&emsp;&emsp;**训练 Training** 本文将generation network $f$ 和 $D_{img} , D_{obj}$ 联合训练。generation network的训练目标是minimize下面的6个损失函数的权重和：
- $Box \ loss\ \ L_{box} = \sum_{i=1}^n ||b_i - \hat b_i||_1$ ：计算真实box和预测box之间的L1范式
- $Mask\ loss\ \ L_{mask}$ ：计算真实mask和预测mask之间基于像素的交叉熵
- $Pixel\ loss\ \ L_{pix} = ||I - \hat I||_1$ ：真实图片和生成图片之间的L范式
- $Image\ adversarial\ loss\ \ L_{GAN}^{img}$ ：针对 $D_{img}$ 的损失函数
- $Object\ adversarial\ loss\ \ L_{GAN}^{obj}$ ：针对 $D_{obj}$ 的损失函数，确保物体的realistic
- $Auxiliarly\ classifier\ loss\ \ L_{AC}^{obj}$ ：针对 $D_{obj}$的损失函数，确保物体的recognizable

&emsp;&emsp;**实现细节 Implementation Details** 本文对所有的scene graphs都进行了数据增强，并且添加了特殊的图片间的relationships，可以把每个真实物体与图片物体进行连接，确保所有的scene graphs都是连通的。我们使用Adam训练所有的模型，学习率设置为 $10^{-4}$ ， batch size 设置为32, 迭代次数为一百万次，使用单个Tesla P100训练了3天。 对于每一次minibatch，我们首先更新 $f$ ，而后更新 $D_{img}$ 和 $D_{obj}$ 。对于所有的graph convolution 本文使用ReLU作为激活函数，对于CRN和discriminators 使用Leaky ReLU作为激活函数，同时使用了batch normalization技术。

# 实验

&emsp;&emsp;在实验中，我们将证明本文提出的方法可以生成复杂的图片，并且正确反应场景图中的物体和物体之间的关系。

## 数据集

&emsp;&emsp;**COCO** 使用2017 COCO-Stuff 数据集，该数据集共有80个物体类别，40K的训练集和5K的验证集，所有的图片标注都具有bounding boxes和segmentation masks 。利用这些标注，本文建立了2D平面上的场景图，总共包含6中人工设定的关系：左边，右边，上边，下边，里面，外面。我们忽略了图片中占图片比例小于2%的物体，使用的图片包含3～8个物体。将COCO val分为val和test两部分。最终，我们得到了24972张训练图片，1024张val图片，2048张test图片。

&emsp;&emsp;**Visual Genome** 本文使用VG 1.4数据集，它包含108077张图片，并且具有标注好的场景图。将其中的80%用作训练集，10%分别用作val和test，本文仅仅使用在训练集中出现次数大于2000次的物体和大于500次的关系，最终，我们得到的训练集具有178种物体和45种关系类型。我们忽略图片中的小物体，并且使用图片中具有3～30个物体和至少一种关系类型的图片，最终我们得到了62565张图片作训练集，5506张val和5088张test，平均每张图片包含10个物体和5种关系类型。由于VG数据集没有提供segmentation masks标注，所以在使用VG数据集时，我们忽略mask loss 。

## 定性结果——Qualitative Results

&emsp;&emsp;由本文提出的模型生成的图片示例如图5,6所示

<center><div style="color:orange;
border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
">图5</div>
</center>

![图片显示失败，请联系博主进行维护](https://s1.ax1x.com/2018/07/15/PMThYd.jpg "图5")

<center><div style="color:orange;
border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
">图6</div>
</center>

![图片显示失败，请联系博主进行维护](https://s1.ax1x.com/2018/07/15/PMTolt.jpg "图6")

## 消融实验

&emsp;&emsp;在消融实验中，如表1所示，我们验证了模型每一部分对最终图片质量的重要性和必要性。文本使用 $inception\ score^2$作为衡量生成图片好坏的标准。

<center><div style="color:orange;
display: inline-block;
color: #999;
">表1</div>
</center>

<center><div style="color:orange;
border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
">表1</div>
</center>

![图片显示失败，请联系博主进行维护](https://s1.ax1x.com/2018/07/15/PMTbm8.jpg "表1")

我们测试了以下几种不同的消融模型：

&emsp;&emsp;**无图卷积 no gconv** ：去掉图卷积层，因此boxes和masks会直接从原始的object embedding vectors预测而来。

&emsp;&emsp;**无关系 no relationships** ：使用图卷积层，但是忽视场景图中的所有“边”，即关系信息。

&emsp;&emsp;**无分辨器 no discriminators** ：去掉分辨器 $D_{img}$ 和 $D_{pix}$ ，依靠像素回归损失函数 $L_{pix}$ 来引导图片的生成。

&emsp;&emsp;**去掉一个分辨器 omit one of the Discriminators** ：仅去掉其中一个分辨器

&emsp;&emsp;**GT Layout** ：除了消融实验外，本文还使用了GT layout来代替 $L_{box} 和 $L_{mask}$ 损失函数。

## 物体定位 Object Localization

&emsp;&emsp;除了关注生成图片的质量外，我们还对本文模型预测到的bounding boxes进行了分析。在表2中，我们展示了object的召回率和2种交并比的分值。另一个评价标准就是多样性。

<center><div style="color:orange;
border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
">表2</div>
</center>

![图片显示失败，请联系博主进行维护](https://s1.ax1x.com/2018/07/15/PMTXkQ.jpg "表2")

## 用户调研 User Studies

&emsp;&emsp;作者找来了数名志愿者，让他们根据以下两个评价标准对本文模型的生成结果和StackGAN的结果进行评价。
- Caption Matching
- Object Recall
