---
title: SSD 源码实现 (PyTorch)
sitemap: true
categories: PyTorch
date: 2018-12-28 16:57:02
tags:
- PyTorch
- SSD
- 目标检测
- 源码实现
---

# 概览
SSD 和 YOLO 都是非常主流的 one-stage 目标检测模型, 并且相对于 two-stage 的 RCNN 系列来说, SSD 的实现更加的简明易懂, 接下来我将从以下几个方面展开对 SSD 模型的源码实现讲解:
- [模型结构定义](#模型结构定义)
- [DefaultBox 生成候选框](#DefaultBox)
- [解析预测结果](#解析预测结果)
- [MultiBox 损失函数](#MultiBox)
- [Augmentations Trick](#Augmentations Trick)
- [模型训练](#模型训练)
- [模型预测](#模型预测)
- [模型验证](#模型验证)
- [其他辅助代码](#其他辅助代码)

<div style="width: 600px; margin: auto">![](https://wx1.sinaimg.cn/large/d7b90c85ly1fx6o47y0w0j215y0r2jz0.jpg)

可以看出, 虽然 SSD 模型本身并不复杂, 但是也正是由于 one-stage 模型较简单的原因, 其检测的准确率相对于 two-stage 模型较低, 因此, 通常需要借助许多训练和检测时的 Tricks 来提升模型的精确度, 这些代码我们会放在第三部分讲解. 下面, 我们按照顺序首先对 SSD 模型结构定义的源码进行解析.(项目地址: https://github.com/amdegroot/ssd.pytorch)

<span id="模型结构定义">
# 模型结构定义
本部分代码主要位于 `ssd.py` 文件里面, 在本文件中, 定义了SSD的模型结构. 主要包含以下类和函数, 整体概览如下:
```py
# ssd.py
class SSD(nn.Module): # 自定义SSD网络
    def __init__(self, phase, size, base, extras, head, num_classes):
        # ... SSD 模型初始化
    def forward(self, x):
        # ... 定义forward函数, 将设计好的layers和ops应用到输入图片 x 上

    def load_weights(self, base_file):
        # ... 加载参数权重值
def vgg(cfg, i, batch_norm=False):
    # ... 搭建vgg网络
def add_extras(cfg, i, batch_norm=False):
    # ... 向VGG网络中添加额外的层用于feature scaling
def multibox(vgg, extra_layers, cfg, num_classes):
    # ... 构建multibox结构
base = {...} # vgg 网络结构参数
extras = {...} # extras 层参数
mbox = {...} # multibox 相关参数
def build_ssd(phase, size=300, num_classes=21):
    # ... 构建模型函数, 调用上面的函数进行构建
```

为了方便理解, 我们不按照文件中的定义顺序解析, 而是根据文件中函数的调用关系来从外而内, 从上而下的进行解析, 解析顺序如下:
- [build_ssd(...) 函数](#build_ssd)
- [vgg(...) 函数](#vgg)
- [add_extras(...) 函数](#add_extras)
- [multibox(...) 函数](#multibox)
- [SSD(nn.Module) 类](#SSD)

<span id="build_ssd">
## build_ssd(...) 函数

在其他文件通常利用`build_ssd(phase, size=300, num_classes=21)`函数来创建模型, 下面先看看该函数的具体实现:

```py
# ssd.py
class SSD(nn.Module): # 自定义SSD网络
    def __init__(self, phase, size, base, extras, head, num_classes):
        # ...
    def forward(self, x):
        # ...
    def load_weights(self, base_file):
        # ...
def vgg(cfg, i, batch_norm=False):
    # ... 搭建vgg网络
def add_extras(cfg, i, batch_norm=False):
    # ... 向VGG网络中添加额外的层用于feature scaling
def multibox(vgg, extra_layers, cfg, num_classes):
    # ... 构建multibox结构
base = { # vgg 网络结构参数
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512],
    '500': []
}
extras = { # extras 层参数
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '500': []
}
mbox = { # multibox 相关参数
    '300': [4, 6, 6, 6, 4, 4],
    '500': []
}
def build_ssd(phase, size=300, num_classes=21):
    # 构建模型函数, 调用上面的函数进行构建
    if phase != "test" and phase != "train": # 只能是训练或者预测阶段
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 300:
        print("ERROR: You specified size " + repr(size) + ". However, "+
                "currently only SSD300 is supported!") # 仅仅支持300size的SSD
        return
    base_, extras_, head_ = multibox(vgg(base[str(size)], 3),
            add_extras(extras[str(size), 1024),
            mbox[str(size)], num_classes )
    return SSD(phase, size, base_, extras_, head_, num_classes)
```

可以看到, `build_ssd(...)`函数主要使用了`multibox(...)`函数来获取`base_, extras_, head_`, 在调用`multibox(...)`函数的同时, 还分别调用了`vgg(...)`函数, `add_extras(...)`函数, 并将其返回值作为参数. 之后, 利用这些信息初始化了SSD网络. 那么下面, 我们就先查看一下这些函数定义和作用

<span id="vgg">
## vgg(...) 函数
我们以调用顺序为依据, 先对`multibox(...)`函数的内部实现进行解析, 但是在查看`multibox(...)`函数之前, 我们首先需要看看其参数的由来, 首先是`vgg(...)`函数, 因为 SSD 是以 VGG 网络作为 backbone 的, 因此该函数主要定义了 VGG 网络的结果, 根据调用语句`vgg(base[str(size)], 3)`可以看出, 调用`vgg`时向其传入了两个参数, 分别为`base[str(size)]` 和`3`, 对应的就是`base['300']`和3.

```py
# ssd.py

def vgg(cfg, i, batch_norm = False):
    # cfg = base['300'] = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512],
    # i = 3
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        if v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels=in_channels, out_channels=v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
        pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        conv7 = nn.Con2d(1024, 1024, kernel_size=1)
        layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
        return layers
```

上面的写法是 `ssd.pytorch` 代码中的原始写法, 代码风格体现了 PyTorch 灵活的编程特性, 但是这种写法不是那么直观, 需要很详细的解读才能看出来这个网络的整个结构是什么样的. 建议大家结合 VGG 网络的整个结构来解读这部分代码, 核心思想就是通过预定义的 `cfg=base={...}` 里面的参数来设置 vgg 网络卷积层和池化层的参数设置, 由于 vgg 网络的模型结构很经典, 有很多文章都写的很详细, 这里就不再啰嗦了, 我们主要来看一下 SSD 网络中比较重要的点, 也就是下面的 `extras_layers`.


<span id="add_extras">
## add_extras(...) 函数

想必了解 SSD 模型的朋友都知道, SSD 模型中是利用多个不同层级上的 feature map 来进行同时进行边框回归和物体分类任务的, 除了使用 vgg 最深层的卷积层以外, SSD 还添加了几个卷积层, 专门用于执行回归和分类任务(如文章开头图2所示), 因此, 我们在定义完 VGG 网络以后, 需要额外定义这些新添加的卷积层. 接下来, 我们根据论文中的参数设置, 来看一下 `add_extras(...)` 的内部实现, 根据调用语句`add_extras(extras[str(size)], 1024)` 可知, 该函数中参数`cfg = extras['300']`, `i=1024`.
```py
# ssd.py
def add_extras(cfg, i, batch_norm=False):
    # cfg = [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256]
    # i = 1024
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S': # (1,3)[True] = 3, (1,3)[False] = 1
                layers += [nn.Conv2d(in_channels=in_channels, out_channels=cfg[k+1],
                                    kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels=in_channels, out_channels=v,
                                    kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers
```

**注意, 在`extras`中, 卷积层之间并没有使用 BatchNorm 和 ReLU, 实际上, ReLU 的使用放在了`forward`函数中**

同样的问题, 上面的定义不是很直观, 因此我将上面的代码用 PyTorch 重写了, 重写后的代码更容易看出网络的结构信息, 同时可读性也较强, 代码如下所示(与上面的代码完全等价):

```py
def add_extras():
    exts1_1 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1)
    exts1_2 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)
    exts2_1 = nn.Conv2d(512, 128, 1, 1, 0)
    exts2_2 = nn.Conv2d(128, 256, 3, 2, 1)
    exts3_1 = nn.Conv2d(256, 128, 1, 1, 0)
    exts3_2 = nn.Conv2d(128, 256, 3, 1, 0)
    exts4_1 = nn.Conv2d(256, 128, 1, 1, 0)
    exts4_2 = nn.Conv2d(128, 256, 3, 1, 0)

    return [exts1_1, exts1_2, exts2_1, exts2_2, exts3_1, exts3_2, exts4_1, exts4_2]
```
在定义完整个的网络结构以后, 我们就需要定义最后的 head 层, 也就是特定的任务层, 因为 SSD 是 one-stage 模型, 因此它是同时在特征图谱上产生预测边框和预测分类的, 我们根据类别的数量来设置相应的网络预测层参数, 注意需要用到多个特征图谱, 也就是说要有多个预测层(原文中用了6个卷积特征图谱, 其中2个来自于 vgg 网络, 4个来自于 extras 层), 代码实现如下:

<span id="multibox">
## multibox(...) 函数
`multibox(...)` 总共有4个参数, 现在我们已经得到了两个参数, 分别是`vgg(...)`函数返回的`layers`, 以及`add_extras(...)`函数返回的`layers`, 后面两个参数根据调用语句可知分别为`mbox[str(size)]`(`mbox['300']`)和`num_classes`(默认为21). 下面, 看一下`multibox(...)`函数的具体内部实现:

```py
# ssd.py
def multibox(vgg, extra_layers, cfg, num_classes):
    # cfg = [4, 6, 6, 6, 4, 4]
    # num_classes = 21
    # ssd总共会选择6个卷积特征图谱进行预测, 分别为, vggnet的conv4_3, 以及extras_layers的5段卷积的输出(每段由两个卷积层组成, 具体可看extras_layers的实现).
    # 也就是说, loc_layers 和 conf_layers 分别具有6个预测层.
    loc_layers = []
    conf_layers = []
    vgg_source = [21, -2]
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels, cfg[k]*4, kernel_size=3, padding=1]
        conf_layers += [nn.Conv2d(vgg[v].out_channels, cfg[k]*num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]*4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]*num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)
```

同样, 我们可以将上面的代码写成可读性更强的形式:
```py
# ssd.py
def multibox(vgg, extras, num_classes):
    loc_layers = []
    conf_layers = []
    #vgg_source=[21, -2] # 21 denote conv4_3, -2 denote conv7

    # 定义6个坐标预测层, 输出的通道数就是每个像素点上会产生的 default box 的数量
    loc1 = nn.Conv2d(vgg[21].out_channels, 4*4, 3, 1, 1) # 利用conv4_3的特征图谱, 也就是 vgg 网络 List 中的第 21 个元素的输出(注意不是第21层, 因为这中间还包含了不带参数的池化层).
    loc2 = nn.Conv2d(vgg[-2].out_channels, 6*4, 3, 1, 1) # Conv7
    loc3 = nn.Conv2d(vgg[1].out_channels, 6*4, 3, 1, 1) # exts1_2
    loc4 = nn.Conv2d(extras[3].out_channels, 6*4, 3, 1, 1) # exts2_2
    loc5 = nn.Conv2d(extras[5].out_channels, 4*4, 3, 1, 1) # exts3_2
    loc6 = nn.Conv2d(extras[7].out_channels, 4*4, 3, 1, 1) # exts4_2
    loc_layers = [loc1, loc2, loc3, loc4, loc5, loc6]

    # 定义分类层, 和定位层差不多, 只不过输出的通道数不一样, 因为对于每一个像素点上的每一个default box,
    # 都需要预测出属于任意一个类的概率, 因此通道数为 default box 的数量乘以类别数.
    conf1 = nn.Conv2d(vgg[21].out_channels, 4*num_classes, 3, 1, 1)
    conf2 = nn.Conv2d(vgg[-2].out_channels, 6*num_classes, 3, 1, 1)
    conf3 = nn.Conv2d(extras[1].out_channels, 6*num_classes, 3, 1, 1)
    conf4 = nn.Conv2d(extras[3].out_channels, 6*num_classes, 3, 1, 1)
    conf5 = nn.Conv2d(extras[5].out_channels, 4*num_classes, 3, 1, 1)
    conf6 = nn.Conv2d(extras[7].out_channels, 4*num_classes, 3, 1, 1)
    conf_layers = [conf1, conf2, conf3, conf4, conf5, conf6]

    # loc_layers: [b×w1×h1×4*4, b×w2×h2×6*4, b×w3×h3×6*4, b×w4×h4×6*4, b×w5×h5×4*4, b×w6×h6×4*4]
    # conf_layers: [b×w1×h1×4*C, b×w2×h2×6*C, b×w3×h3×6*C, b×w4×h4×6*C, b×w5×h5×4*C, b×w6×h6×4*C] C为num_classes
    # 注意pytorch中卷积层的输入输出维度是:[N×C×H×W], 上面的顺序有点错误, 不过改起来太麻烦
    return loc_layers, conf_layers
```

定义完网络中所有层的关键结构以后, 我们就可以利用这些结构来定义 SSD 网络了, 下面就介绍一下 SSD 类的实现.

<span id="SSD">
## SSD(nn.Module) 类

在 `build_ssd(...)` 函数的最后, 利用语句`return SSD(phase, size, base_, extras_, head_, num_classes)`调用的返回了一个`SSD`类的对象, 下面, 我们就来看一下看类的内部细节(这也是SSD模型的主要框架实现)

```py
# ssd.py
class SSD(nn.Module):
    # SSD网络是由 VGG 网络后街 multibox 卷积层 组成的, 每一个 multibox 层会有如下分支:
    # - 用于class conf scores的卷积层
    # - 用于localization predictions的卷积层
    # - 与priorbox layer相关联, 产生默认的bounding box

    # 参数:
    # phase: test/train
    # size: 输入图片的尺寸
    # base: VGG16的层
    # extras: 将输出结果送到multibox loc和conf layers的额外的层
    # head: "multibox head", 包含一系列的loc和conf卷积层.

    def __init__(self, phase, size, base, extras, head, num_classes):
        # super(SSD, self) 首先找到 SSD 的父类, 然后把类SSD的对象转换为父类的对象
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = (coco, voc)[num_classes == 21]
        self.priorbox = PriorBox(self.cfg) # layers/functions/prior_box.py class PriorBox(object)
        self.priors = Variable(self.priorbox.forward(), volatile=True) # from torch.autograd import Variable
        self.size = size

        self.vgg = nn.ModuleList(base)
        self.L2Norm = L2Norm(512,20)  # layers/modules/l2norm.py class L2Norm(nn.Module)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0]) # head = (loc_layers, conf_layers)
        self.conf = nn.ModuleList(head[1])

        if phase = "test":
            self.softmax = nn.Softmax(dim=-1) # 用于囧穿概率
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45) #  layers/functions/detection.py class Detect
            # 用于将预测结果转换成对应的坐标和类别编号形式, 方便可视化.
    def forward(self, x):
        # 定义forward函数, 将设计好的layers和ops应用到输入图片 x 上

        # 参数: x, 输入的batch 图片, Shape: [batch, 3, 300, 300]

        # 返回值: 取决于不同阶段
        # test: 预测的类别标签, confidence score, 以及相关的location.
        #       Shape: [batch, topk, 7]
        # train: 关于以下输出的元素组成的列表
        #       1: confidence layers, Shape: [batch*num_priors, num_classes]
        #       2: localization layers, Shape: [batch, num_priors*4]
        #       3: priorbox layers, Shape: [2, num_priors*4]
        sources = list() # 这个列表存储的是参与预测的卷积层的输出, 也就是原文中那6个指定的卷积层
        loc = list() # 用于存储预测的边框信息
        conf = list() # 用于存储预测的类别信息

        # 计算vgg直到conv4_3的relu
        for k in range(23):
            x = self.vgg[k](x)

        s = self.L2Norm(x)
        sources.append(s) # 将 conv4_3 的特征层输出添加到 sources 中, 后面会根据 sources 中的元素进行预测

        # 将vgg应用到fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x) # 同理, 添加到 sources 列表中

        # 计算extras layers, 并且将结果存储到sources列表中
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True) # import torch.nn.functional as F
            if k % 2 = 1: # 在extras_layers中, 第1,3,5,7,9(从第0开始)的卷积层的输出会用于预测box位置和类别, 因此, 将其添加到 sources列表中
                sources.append(x)

        # 应用multibox到source layers上, source layers中的元素均为各个用于预测的特征图谱
        # apply multibox to source layers

        # 注意pytorch中卷积层的输入输出维度是:[N×C×H×W]
        for (x, l, c) in zip(sources, self.loc, self.conf):
            # permute重新排列维度顺序, PyTorch维度的默认排列顺序为 (N, C, H, W),
            # 因此, 这里的排列是将其改为 $(N, H, W, C)$.
            # contiguous返回内存连续的tensor, 由于在执行permute或者transpose等操作之后, tensor的内存地址可能不是连续的,
            # 然后 view 操作是基于连续地址的, 因此, 需要调用contiguous语句.
            loc.append(l(x).permute(0,2,3,1).contiguous())
            conf.append(c(x).permute(0,2,3,1).contiguous())
            # loc: [b×w1×h1×4*4, b×w2×h2×6*4, b×w3×h3×6*4, b×w4×h4×6*4, b×w5×h5×4*4, b×w6×h6×4*4]
            # conf: [b×w1×h1×4*C, b×w2×h2×6*C, b×w3×h3×6*C, b×w4×h4×6*C, b×w5×h5×4*C, b×w6×h6×4*C] C为num_classes
        # cat 是 concatenate 的缩写, view返回一个新的tensor, 具有相同的数据但是不同的size, 类似于numpy的reshape
        # 在调用view之前, 需要先调用contiguous
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        # 将除batch以外的其他维度合并, 因此, 对于边框坐标来说, 最终的shape为(两维):[batch, num_boxes*4]
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        # 同理, 最终的shape为(两维):[batch, num_boxes*num_classes]

        if self.phase == "test":
            # 这里用到了 detect 对象, 该对象主要由于接预测出来的结果进行解析, 以获得方便可视化的边框坐标和类别编号, 具体实现会在后文讨论.
            output = self.detect(
                loc.view(loc.size(0), -1, 4), #  又将shape转换成: [batch, num_boxes, 4], 即[1, 8732, 4]
                self.softmax(conf.view(conf.size(0), -1, self.num_classes)), # 同理,  shape 为[batch, num_boxes, num_classes], 即 [1, 8732, 21]
                self.priors.type(type(x.data))
                # 利用 PriorBox对象获取特征图谱上的 default box, 该参数的shape为: [8732,4]. 关于生成 default box 的方法实际上很简单, 类似于 anchor box, 详细的代码实现会在后文解析.
                # 这里的 self.priors.type(type(x.data)) 与 self.priors 就结果而言完全等价(自己试验过了), 但是为什么?
            )
        if self.phase == "train": # 如果是训练阶段, 则无需解析预测结果, 直接返回然后求损失.
            output = (
                loc.view(loc.size(0), -1, 4), conf.view(conf.size(0), -1, self.num_classes), self.priors
            )
        return output
    def load_weights(self, base_file): # 加载权重文件
        other, ext = os.path.splitext(base_file)
        if ext == ".pkl" or ".pth":
            print("Loading weights into state dict...")
            self.load_state_dict(torch.load(base_file, map_location=lambda storage, loc: storage))
            print("Finished!")
        else:
            print("Sorry only .pth and .pkl files supported")
```

在上面的模型定义中, 我们可以看到使用其他几个类, 分别是
- `layers/functions/prior_box.py class` 的  `PriorBox(object)`,
- `layers/modules/l2norm.py` 的 `class L2Norm(nn.Module)`
- `layers/functions/detection.py` 的 `class Detect`

基本上从他们的名字就可以看出他们的用途, 其中, 最简单的是 l2norm 类, 该类实际上就是实现了 L2归一化(也可以利用 PyTorch API 提供的归一化接口实现). 这一块没什么好讨论的, 朋友们可以自己去源码去查看实现方法, 基本看一遍就能明白了.下面我们着重看一下用于生成 Default box(也可以看成是 anchor box) 的 `PriorBox` 类, 以及用于解析预测结果, 并将其转换成边框坐标和类别编号的 `Detect`类. 首先来看如何利用卷积图谱来生成 default box.

<span id="DefaultBox">
# DefaultBox 生成候选框

根据 SSD 的原理, 需要在选定的特征图谱上输出 Default Box, 然后根据这些 Default Box 进行边框回归任务. 首先梳理一下生成 Default Box 的思路. 假如feature maps数量为 $m$, 那么每一个feature map中的default box的尺寸大小计算如下:

$$s_k = s_{min} + \frac{s_{max} - s_{min}}{m-1}(k-1), k\in [1,m]$$

上式中, $s_{min} = 0.2 , s_{max} = 0.9$. 对于原文中的设置 $m=6 (4, 6, 6, 6, 4, 4)$, 因此就有 $s = \{0.2, 0.34, 0.48, 0.62, 0.76, 0.9\}$
然后, 几个不同的aspect ratio, 用 $a_r$ 表示: $a_r = {1,2,3,1/2,1/3}$, 则每一个default boxes 的width 和height就可以得到( $w_k^a h_k^a=a_r$ ):

$$w_k^a = s_k \sqrt{a_r}$$

$$h_k^a = \frac{s_k}{\sqrt {a_r}}$$

对于宽高比为1的 default box, 我们额外添加了一个 scale 为 $s_k' = \sqrt{s_k s_{k+1}}$ 的 box, 因此 feature map 上的每一个像素点都对应着6个 default boxes (**per feature map localtion**).
每一个default box的中心, 设置为: $(\frac{i+0.5}{|f_k|}, \frac{j+0.5}{f_k})$, 其中, $|f_k|$ 是第 $k$ 个feature map的大小 $i,j$ 对应了 feature map 上所有可能的像素点.
**在实际使用中, 可以自己根据数据集的特点来安排不同的 default boxes 参数组合**

了解原理以后, 就来看一下怎么实现, 输出 Default Box 的代码定义在 `layers/functions/prior_box.py` 文件中. 代码如下所示:

```py
# `layers/functions/prior_box.py`

class PriorBox(object):
    # 所谓priorbox实际上就是网格中每一个cell推荐的box
    def __init__(self, cfg):
        # 在SSD的init中, cfg=(coco, voc)[num_classes=21]
        # coco, voc的相关配置都来自于data/cfg.py 文件
        super(PriorBox, self).__init__()
        self.image_size = cfg["min_dim"]
        self.num_priors = len(cfg["aspect_ratios"])
        self.variance = cfg["variance"] or [0.1]
        self.min_sizes = cfg["min_sizes"]
        self.max_sizes = cfg["max_sizes"]
        self.steps = cfg["steps"]
        self.aspect_ratios = cfg["aspect_ratios"]
        self.clip = cfg["clip"]
        self.version = cfg["name"]
        for v in self.variance:
            if v <= 0:
                raise ValueError("Variances must be greater than 0")

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps): # 存放的是feature map的尺寸:38,19,10,5,3,1
            # from itertools import product as product
            for i, j in product(range(f), repeat=2):
                # 这里实际上可以用最普通的for循环嵌套来代替, 主要目的是产生anchor的坐标(i,j)

                f_k = self.image_size / self.steps[k] # steps=[8,16,32,64,100,300]. f_k大约为feature map的尺寸
                # 求得center的坐标, 浮点类型. 实际上, 这里也可以直接使用整数类型的 `f`, 计算上没太大差别
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k # 这里一定要特别注意 i,j 和cx, cy的对应关系, 因为cy对应的是行, 所以应该零cy与i对应.

                # aspect_ratios 为1时对应的box
                s_k = self.min_sizes[k]/self.image_size
                mean += [cx, cy, s_k, s_k]

                # 根据原文, 当 aspect_ratios 为1时, 会有一个额外的 box, 如下:
                # rel size: sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # 其余(2, 或 2,3)的宽高比(aspect ratio)
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
                # 综上, 每个卷积特征图谱上每个像素点最终产生的 box 数量要么为4, 要么为6, 根据不同情况可自行修改.
        output = torch.Tensor(mean).view(-1,4)
        if self.clip:
            output.clamp_(max=1, min=0) # clamp_ 是clamp的原地执行版本
        return output # 输出default box坐标(可以理解为anchor box)
```

最终, 输出的ouput就是一张图片中所有的default box的坐标, 对于论文中的默认设置来说产生的box数量为:
$$38^2 \times 4+19^2 \times 6+ 10^2 \times 6+5^2 \times 6+3^2 \times 4+1^2 \times 4 = 8732$$

<span id="解析预测结果">
# 解析预测结果

在模型中, 我们为了加快训练速度, 促使模型收敛, 因此会将相应的 box 的坐标转换成与图片size成比例的小数形式, 因此, 无法直接将模型产生的预测结果可视化. 下面, 我们首先会通过接受 `Detect` 类来说明如何解析预测结果, 同时, 还会根据源码中提过的 `demo` 文件来接受如何将对应的结果可视化出来, 首先, 来看一下 `Detect` 类的定义和实现:

```py
# ./layers/
class Detect(Function):
    # 测试阶段的最后一层, 负责解码预测结果, 应用nms选出合适的框和对应类别的置信度.
    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.variance = voc_config["variance"]

    def forward(self, loc_data, conf_data, prior_data):
        # loc_data: [batch, num_priors, 4], [batch, 8732, 4]
        # conf_data: [batch, num_priors, 21], [batch, 8732, 21]
        # prior_data: [num_priors, 4], [8732, 4]

        num = loc_data.size(0) # batch_size
        num_priors = prior_data.size(0)
        output = torch.zeros(num, self.num_classes, self.top_k, 5) # output:[b, 21, k, 5]
        conf_preds = conf_data.view(num, num_priors, self.num_classes).transpose(2,1) # 维度调换

        # 将预测结果解码
        for i in range(num): # 对每一个image进行解码
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)#获取第i个图片的box坐标
            conf_scores = conf_preds[i].clone() # 复制第i个image置信度预测结果

            for cl in range(1, self.num_classes): # num_classes=21, 所以 cl 的值为 1~20
                c_mask = conf_scores[cl].gt(self.conf_thresh) # 返回由0,1组成的数组, 0代表小于thresh, 1代表大于thresh
                scores = conf_scores[cl][c_mask] # 返回值为1的对应下标的元素值(即返回conf_scores中大于thresh的元素集合)

                if scores.size(0) == 0:
                    continue # 没有置信度, 说明没有框
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes) # 获取对应box的二值矩阵
                boxes = decoded_boxes[l_mask].view(-1,4) # 获取置信度大于thresh的box的左上角和右下角坐标

                # 返回每个类别的最高的score 的下标, 并且除去那些与该box有较大交并比的box
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k) # 从这些box里面选出top_k个, count<=top_k
                # count<=top_k
                output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1), boxes[:count]), 1)
        flt = output.contiguous().view(num,-1,5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
        # 注意, view共享tensor, 因此, 对flt的修改也会反应到output上面
        return output
```

在这里, 用到了两个关键的函数 `decode()` 和 `nms()`, 这两个函数定义在`./layers/box_utils.py`文件中, 代码如下所示:
```py
def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes
def nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """

    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count
```

<span id="MultiBox">
# MultiBox 损失函数

在`layers/modules/multibox_loss.py` 中定义了SSD模型的损失函数, 在SSD论文中, 损失函数具体定义如下:
$$L_{loc}(x,l,g) = \sum_{i\in Pos}^N \sum_{m\in\{cx,cy,w,h\}} x_{ij}^k smooth_{L_1}(l_i^m - \hat g_j^m)$$


$$L_{conf}(x,c) = -\sum_{i\in Pos}^N x_{ij}^p log(\hat c_i^p) - \sum_{i\in Neg} log(\hat c_i^0), 其中, \hat c_i^p = \frac{exp(c_i^p)}{\sum_p exp(c_i^p)}$$


## 损失函数定义
根据上面的公式, 我们可以定义下面的损失函数类, 该类继承了 `nn.Module`, 因此可以当做是一个 `Module` 用在训练函数中.
```py
# layers/modules/multibox_loss.py

class MultiBoxLoss(nn.Module):
    # 计算目标:
    # 输出那些与真实框的iou大于一定阈值的框的下标.
    # 根据与真实框的偏移量输出localization目标
    # 用难样例挖掘算法去除大量负样本(默认正负样本比例为1:3)
    # 目标损失:
    # L(x,c,l,g) = (Lconf(x,c) + αLloc(x,l,g)) / N
    # 参数:
    # c: 类别置信度(class confidences)
    # l: 预测的框(predicted boxes)
    # g: 真实框(ground truth boxes)
    # N: 匹配到的框的数量(number of matched default boxes)

    def __init__(self, num_classes, overlap_thresh, prior_for_matching, bkg_label, neg_mining, neg_pos, neg_overlap, encode_target, use_gpu=True):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes= num_classes # 列表数
        self.threshold = overlap_thresh # 交并比阈值, 0.5
        self.background_label = bkg_label # 背景标签, 0
        self.use_prior_for_matching = prior_for_matching # True 没卵用
        self.do_neg_mining = neg_mining # True, 没卵用
        self.negpos_ratio = neg_pos # 负样本和正样本的比例, 3:1
        self.neg_overlap = neg_overlap # 0.5 判定负样本的阈值.
        self.encode_target = encode_target # False 没卵用
        self.variance = cfg["variance"]

    def forward(self, predictions, targets):
        loc_data, conf_data, priors = predictions
        # loc_data: [batch_size, 8732, 4]
        # conf_data: [batch_size, 8732, 21]
        # priors: [8732, 4]  default box 对于任意的图片, 都是相同的, 因此无需带有 batch 维度
        num = loc_data.size(0) # num = batch_size
        priors = priors[:loc_data.size(1), :] # loc_data.size(1) = 8732, 因此 priors 维持不变
        num_priors = (priors.size(0)) # num_priors = 8732
        num_classes = self.num_classes # num_classes = 21 (默认为voc数据集)

        # 将priors(default boxes)和ground truth boxes匹配
        loc_t = torch.Tensor(num, num_priors, 4) # shape:[batch_size, 8732, 4]
        conf_t = torch.LongTensor(num, num_priors) # shape:[batch_size, 8732]
        for idx in range(num):
            # targets是列表, 列表的长度为batch_size, 列表中每个元素为一个 tensor,
            # 其 shape 为 [num_objs, 5], 其中 num_objs 为当前图片中物体的数量, 第二维前4个元素为边框坐标, 最后一个元素为类别编号(1~20)
            truths = targets[idx][:, :-1].data # [num_objs, 4]
            labels = targets[idx][:, -1].data # [num_objs] 使用的是 -1, 而不是 -1:, 因此, 返回的维度变少了
            defaults = priors.data # [8732, 4]
            # from ..box_utils import match
            # 关键函数, 实现候选框与真实框之间的匹配, 注意是候选框而不是预测结果框! 这个函数实现较为复杂, 会在后面着重讲解
            match(self.threshold, truths, defaults, self.variance, labels, loc_t, conf_t, idx) # 注意! 要清楚 Python 中的参数传递机制, 此处在函数内部会改变 loc_t, conf_t 的值, 关于 match 的详细讲解可以看后面的代码解析
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        # 用Variable封装loc_t, 新版本的 PyTorch 无需这么做, 只需要将 requires_grad 属性设置为 True 就行了
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)

        pos = conf_t > 0 # 筛选出 >0 的box下标(大部分都是=0的)
        num_pos = pos.sum(dim=1, keepdim=True) # 求和, 取得满足条件的box的数量, [batch_size, num_gt_threshold]

        # 位置(localization)损失函数, 使用 Smooth L1 函数求损失
        # loc_data:[batch, num_priors, 4]
        # pos: [batch, num_priors]
        # pos_idx: [batch, num_priors, 4], 复制下标成坐标格式, 以便获取坐标值
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)# 获取预测结果值
        loc_t = loc_t[pos_idx].view(-1, 4) # 获取gt值
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False) # 计算损失

        # 计算最大的置信度, 以进行难负样本挖掘
        # conf_data: [batch, num_priors, num_classes]
        # batch_conf: [batch, num_priors, num_classes]
        batch_conf = conf_data.view(-1, self.num_classes) # reshape

        # conf_t: [batch, num_priors]
        # loss_c: [batch*num_priors, 1], 计算每个priorbox预测后的损失
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1,1))

        # 难负样本挖掘, 按照loss进行排序, 取loss最大的负样本参与更新
        loss_c[pos.view(-1, 1)] = 0 # 将所有的pos下标的box的loss置为0(pos指示的是正样本的下标)
        # 将 loss_c 的shape 从 [batch*num_priors, 1] 转换成 [batch, num_priors]
        loss_c = loss_c.view(num, -1) # reshape
        # 进行降序排序, 并获取到排序的下标
        _, loss_idx = loss_c.sort(1, descending=True)
        # 将下标进行升序排序, 并获取到下标的下标
        _, idx_rank = loss_idx.sort(1)
        # num_pos: [batch, 1], 统计每个样本中的obj个数
        num_pos = pos.long().sum(1, keepdim=True)
        # 根据obj的个数, 确定负样本的个数(正样本的3倍)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        # 获取到负样本的下标
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # 计算包括正样本和负样本的置信度损失
        # pos: [batch, num_priors]
        # pos_idx: [batch, num_priors, num_classes]
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        # neg: [batch, num_priors]
        # neg_idx: [batch, num_priors, num_classes]
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        # 按照pos_idx和neg_idx指示的下标筛选参与计算损失的预测数据
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        # 按照pos_idx和neg_idx筛选目标数据
        targets_weighted = conf_t[(pos+neg).gt(0)]
        # 计算二者的交叉熵
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        # 将损失函数归一化后返回
        N = num_pos.data.sum()
        loss_l = loss_l / N
        loss_c = loss_c / N
        return loss_l, loss_c
```

## GT box 与default box 的匹配

在上面的代码中, 有一个很重要的函数, 即 `match()` 函数, 因为我们知道, 当根据特征图谱求出这些 prior box(default box, 8732个)以后, 我们仅仅知道这些 box 的 scale 和 aspect_ratios 信息, 但是如果要计算损失函数, 我们就必须知道与每个 prior box 相对应的 ground truth box 是哪一个, 因此, 我们需要根据交并比来求得这些 box 之间的匹配关系. 匹配算法的核心思想如下:
1. 首先将找到与每个 gtbox 交并比最高的 defaultbox, 记录其下标
2. 然后找到与每个 defaultbox 交并比最高的 gtbox. 注意, 这两步不是一个相互的过程, 假想一种极端情况, 所有的priorbox与某个gtbox(标记为G)的交并比为1, 而其他gtbox分别有一个交并比最高的priorbox, 但是肯定小于1(因为其他的gtbox与G的交并比肯定小于1), 这样一来, 就会使得所有的priorbox都与G匹配.
3. 为了防止上面的情况, 我们将那些对于gtbox来说, 交并比最高的priorbox, 强制进行互相匹配, 即令 `best_truth_idx[best_prior_idx[j]] = j`, 详细见下面的for循环.
4. 根据下标获取每个priorbox对应的gtbox的坐标, 然后对坐标进行相应编码, 并存储起来, 同时将gt类别也存储起来, 到此, 匹配完成.


根据上面的求解思想, 我们可以实现相应的匹配代码, 主要用到了以下几个函数:
- `point_form(boxes)`: 将 boxes 的坐标信息转换成左上角和右下角的形式
- `intersect(box_a, box_b)`: 返回 box_a 与 box_b 集合中元素的交集
- `jaccard(box_a, box_b)`: 返回 box_a 与 box_b 集合中元素的交并比
- `encode(matched, priors, variances)`: 将 box 信息编码成小数形式, 方便网络训练
- `match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx)`: 匹配算法, 通过调用上述函数实现匹配功能


完整代码及解析如下所示(位于 `./layers/box_utils.py` 文件中):
```py
# ./layers/box_utils.py
def point_form(boxes):
    # 将(cx, cy, w, h) 形式的box坐标转换成 (xmin, ymin, xmax, ymax) 形式
    return torch.cat( (boxes[:2] - boxes[2:]/2), # xmin, ymin
                    (boxes[:2] + boxes[2:]/2), 1) # xmax, ymax


def intersect(box_a, box_b):
    # box_a: (truths), (tensor:[num_obj, 4])
    # box_b: (priors), (tensor:[num_priors, 4], 即[8732, 4])
    # return: (tensor:[num_obj, num_priors]) box_a 与 box_b 两个集合中任意两个 box 的交集, 其中res[i][j]代表box_a中第i个box与box_b中第j个box的交集.(非对称矩阵)
    # 思路: 先将两个box的维度扩展至相同维度: [num_obj, num_priors, 4], 然后计算面积的交集
    # 两个box的交集可以看成是一个新的box, 该box的左上角坐标是box_a和box_b左上角坐标的较大值, 右下角坐标是box_a和box_b的右下角坐标的较小值
    A = box_a.size(0)
    B = box_b.size(0)
    # box_a 左上角/右下角坐标 expand以后, 维度会变成(A,B,2), 其中, 具体可看 expand 的相关原理. box_b也是同理, 这样做是为了得到a中某个box与b中某个box的左上角(min_xy)的较大者(max)
    # unsqueeze 为增加维度的数量, expand 为扩展维度的大小
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A,B,2),
                        box_b[:, :2].unsqueeze(0).expand(A,B,2)) # 在box_a的 A 和 2 之间增加一个维度, 并将维度扩展到 B. box_b 同理
    # 求右下角(max_xy)的较小者(min)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A,B,2),
                        box_b[:, 2:].unsqueeze(0).expand(A,B,2))
    inter = torch.clamp((max_xy, min_xy), min=0) # 右下角减去左上角, 如果为负值, 说明没有交集, 置为0
    return inter[:, :, 0] * inter[:, :, 0] # 高×宽, 返回交集的面积, shape 刚好为 [A, B]


def jaccard(box_a, box_b):
    # A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    # box_a: (truths), (tensor:[num_obj, 4])
    # box_b: (priors), (tensor:[num_priors, 4], 即[8732, 4])
    # return: (tensor:[num_obj, num_priors]), 代表了 box_a 和 box_b 两个集合中任意两个 box之间的交并比
    inter = intersect(box_a, box_b) # 求任意两个box的交集面积, shape为[A, B], 即[num_obj, num_priors]
    area_a = ((box_a[:,2]-box_a[:,0]) * (box_a[:,3]-box_a[:,1])).unsqueeze(1).expand_as(inter) # [A,B]
    area_b = ((box_b[:,2]-box_b[:,0]) * (box_b[:,3]-box_b[:,1])).unsqueeze(0).expand_as(inter) # [A,B], 这里会将A中的元素复制B次
    union = area_a + area_b - inter
    return inter / union # [A, B], 返回任意两个box之间的交并比, res[i][j] 代表box_a中的第i个box与box_b中的第j个box之间的交并比.

def encode(matched, priors, variances):
    # 对边框坐标进行编码, 需要宽度方差和高度方差两个参数, 具体公式可以参见原文公式(2)
    # matched: [num_priors,4] 存储的是与priorbox匹配的gtbox的坐标. 形式为(xmin, ymin, xmax, ymax)
    # priors: [num_priors, 4] 存储的是priorbox的坐标. 形式为(cx, cy, w, h)
    # return : encoded boxes: [num_priors, 4]
    g_cxy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2] # 用互相匹配的gtbox的中心坐标减去priorbox的中心坐标, 获得中心坐标的偏移量
    g_cxy /= (variances[0]*priors[:, 2:]) # 令中心坐标分别除以 d_i^w 和 d_i^h, 正如原文公式所示
    #variances[0]为0.1, 令其分别乘以w和h, 得到d_i^w 和 d_i^h
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:] # 令互相匹配的gtbox的宽高除以priorbox的宽高.
    g_wh = torch.log(g_wh) / variances[1] # 这里这个variances[1]=0.2 不太懂是为什么.
    return torch.cat([g_cxy, g_wh], 1) # 将编码后的中心坐标和宽高``连接起来, 返回 [num_priors, 4]

def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    # threshold: (float) 确定是否匹配的交并比阈值
    # truths: (tensor: [num_obj, 4]) 存储真实 box 的边框坐标
    # priors: (tensor: [num_priors, 4], 即[8732, 4]), 存储推荐框的坐标, 注意, 此时的框是 default box, 而不是 SSD 网络预测出来的框的坐标, 预测的结果存储在 loc_data中, 其 shape 为[num_obj, 8732, 4].
    # variances: cfg['variance'], [0.1, 0.2], 用于将坐标转换成方便训练的形式(参考RCNN系列对边框坐标的处理)
    # labels: (tensor: [num_obj]), 代表了每个真实 box 对应的类别的编号
    # loc_t: (tensor: [batches, 8732, 4]),
    # conf_t: (tensor: [batches, 8732]),
    # idx: batches 中图片的序号, 标识当前正在处理的 image 在 batches 中的序号
    overlaps = jaccard(truths, point_form(priors)) # [A, B], 返回任意两个box之间的交并比, overlaps[i][j] 代表box_a中的第i个box与box_b中的第j个box之间的交并比.

    # 二部图匹配(Bipartite Matching)
    # [num_objs,1], 得到对于每个 gt box 来说的匹配度最高的 prior box, 前者存储交并比, 后者存储prior box在num_priors中的位置
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True) # keepdim=True, 因此shape为[num_objs,1]
    # [1, num_priors], 即[1,8732], 同理, 得到对于每个 prior box 来说的匹配度最高的 gt box
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_prior_idx.squeeze_(1) # 上面特意保留了维度(keepdim=True), 这里又都把维度 squeeze/reduce 了, 实际上只需用默认的 keepdim=False 就可以自动 squeeze/reduce 维度.
    best_prior_overlap.squeeze_(1)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)

    best_truth_overlap.index_fill_(0, best_prior_idx, 2)
    # 维度压缩后变为[num_priors], best_prior_idx 维度为[num_objs],
    # 该语句会将与gt box匹配度最好的prior box 的交并比置为 2, 确保其最大, 以免防止某些 gtbox 没有匹配的 priorbox.

    # 假想一种极端情况, 所有的priorbox与某个gtbox(标记为G)的交并比为1, 而其他gtbox分别有一个交并比
    # 最高的priorbox, 但是肯定小于1(因为其他的gtbox与G的交并比肯定小于1), 这样一来, 就会使得所有
    # 的priorbox都与G匹配, 为了防止这种情况, 我们将那些对gtbox来说, 具有最高交并比的priorbox,
    # 强制进行互相匹配, 即令best_truth_idx[best_prior_idx[j]] = j, 详细见下面的for循环

    # 注意!!: 因为 gt box 的数量要远远少于 prior box 的数量, 因此, 同一个 gt box 会与多个 prior box 匹配.
    for j in range(best_prior_idx.size(0)): # range:0~num_obj-1
        best_truth_idx[best_prior_idx[j]] = j
        # best_prior_idx[j] 代表与box_a的第j个box交并比最高的 prior box 的下标, 将与该 gtbox
        # 匹配度最好的 prior box 的下标改为j, 由此,完成了该 gtbox 与第j个 prior box 的匹配.
        # 这里的循环只会进行num_obj次, 剩余的匹配为 best_truth_idx 中原本的值.
        # 这里处理的情况是, priorbox中第i个box与gtbox中第k个box的交并比最高,
        # 即 best_truth_idx[i]= k
        # 但是对于best_prior_idx[k]来说, 它却与priorbox的第l个box有着最高的交并比,
        # 即best_prior_idx[k]=l
        # 而对于gtbox的另一个边框gtbox[j]来说, 它与priorbox[i]的交并比最大,
        # 即但是对于best_prior_idx[j] = i.
        # 那么, 此时, 我们就应该将best_truth_idx[i]= k 修改成 best_truth_idx[i]= j.
        # 即令 priorbox[i] 与 gtbox[j]对应.
        # 这样做的原因: 防止某个gtbox没有匹配的 prior box.
    mathes = truths[best_truth_idx]
    # truths 的shape 为[num_objs, 4], 而best_truth_idx是一个指示下标的列表, 列表长度为 8732,
    # 列表中的下标范围为0~num_objs-1, 代表的是与每个priorbox匹配的gtbox的下标
    # 上面的表达式会返回一个shape为 [num_priors, 4], 即 [8732, 4] 的tensor, 代表的就是与每个priorbox匹配的gtbox的坐标值.
    conf = labels[best_truth_idx]+1 # 与上面的语句道理差不多, 这里得到的是每个prior box匹配的类别编号, shape 为[8732]
    conf[best_truth_overlap < threshold] = 0 # 将与gtbox的交并比小于阈值的置为0 , 即认为是非物体框
    loc = encode(matches, priors, variances) # 返回编码后的中心坐标和宽高.
    loc_t[idx] = loc # 设置第idx张图片的gt编码坐标信息
    conf_t[idx] = conf # 设置第idx张图片的编号信息.(大于0即为物体编号, 认为有物体, 小于0认为是背景)
```

<span id="模型训练">
# 模型训练
在定义了模型结构和相应的随时函数以后, 接下来就是训练阶段, 训练代码位于`train.py`文件中, 下面对该文件代码进行解读:

```py
# train.py

def str2bool(v):
    return v.lower() in ("yes", "true", "t", 1)

import argparse
parser = argparse.ArgumentParser(description="Single Shot MultiBox Detection")
#...
parser.add_argument("--cuda", default=True, type=str2bool,
                    help="Use CUDA to train model")
#...
args = parser.parse_args()

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        torch.set_default_tensor_type("torch.FloatTensor")
else:
    torch.set_default_tensor_type("torch.FloatTensor")


def train():
# 该文件中中主要的函数, 在main()中, 仅调用了该函数
    if args.dataset == "COCO":
        if args.dataset_root == VOC_ROOT:
            # ...
        cfg = coco # coco位于config.py文件中
        # COCODetection类 位于coco.py文件中
        # SSDAugmentation类 位于utils/augmentations.py文件中
        dataset = COCODetection(root=args.dataset_root,
                                transform=SSDAugmentation(cfg["min_dim"], MEANS))
    elif args.dataset == "VOC":
        if args.dataset_root == COCO_ROOT:
            #...
        cfg = voc
        dataset = VOCDetection(root=args.dataset_root,
                               transform=SSDAugmentation(cfg["min_dim"], MEANS))

    if args.visdom:
        import visdom
        viz = visdom.Visdom()
    # from ssd import build_ssd
    ssd_net = build_ssd("train", cfg["min_dim"], cfg["num_classes"])
    net = ssd_net

    if args.cuda:
        net = torch.nn.DataParallel(ssd_net)
        # import torch.backends.cudnn as cudnn
        cudnn.benchmark = True # 大部分情况下, 这个flag可以让内置的cuDNN的auto-tuner自动寻找最适合当前配置的算法.

    if args.resume: # resume 类型为 str, 值为checkpoint state_dict file
        ssd_net.load_weights(args.resume)
    else:
        vgg_weights = torch.load(args.save_folder + args.basenet)
        ssd_net.load_state_dict(vgg_weights)

    if args.cuda:
        net = net.cuda() # 将所有的参数都移送到GPU内存中

    if not args.resume:
        ssd_net.extras.apply(weights_init) # 本文件的函数: def weights_init(), 对网络参数执行Xavier初始化.
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    # import torch.optim as optim
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # MultiBoxLoss类 位于layers/modules/multibox_loss.py文件中
    criterion = MultiBoxLoss(cfg["num_classes"], 0.5, True, 0, True, 3, 0.5, False, args.cuda)

    net.train()
    # loss计数器
    loc_loss = 0
    conf_loss = 0
    epoch = 0

    epoch_size = len(dataset) // args.batch_size

    step_index = 0

    if args.visdom:
        #...

    # import torch.utils.data as data
    data_loader = data.DataLoader(dataset, args.batch_size, num_workers=args.num_workers, shuffle=True, collate_fn=detection_collate, pin_memory=True)

    # 创建batch迭代器
    batch_iterator = iter(data_loader)
    for iteration in range(args.start_iter, cfg["max_iter"]):
        if args.visdom and iteration != 0 and (iteration % epoch_size==0):
            update_vis_plot(epoch, loc_loss, conf_loss, epoch_plot, None, "append", epoch_size)
            loc_loss = 0
            conf_loss = 0
            epoch += 1

        if iteration in cfg["lr_steps"]:
            step_index += 1
            # 每经过固定迭代次数, 就将lr衰减1/10
            adjust_learning_rate(optimizer, args.gamma, step_index)

        # load train data
        images, targets = next(batch_iterator)

        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
        else:
            images = Variable(images)
            targets = [Variable(ann, valotile=True) for ann in targets]

        # forward
        t0 = time.time()
        out = net(images)
        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets) # criterion = MultiBoxLoss(...)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()
        t1 = time.time()
        loc_loss += loss_l.data[0]
        conf_loss += loss_c.data[0]

        if iteratioin % 10 == 0:
            # print(...) 每隔10次迭代就输出一次训练状态信息

        if args.visdom:
            # update_vis_plot(...)

        if iteration != 0 and iteration % 5000 ==0:
            # save model

```

<span id="模型验证">
# 模型验证

下面是模型验证的相关代码, 存在于`./test.py`文件中, 代码没有太多特殊的处理, 和`./train.py`文件略有相似.

```py

def test_net(save_folder, net, cuda, testset, transform, thresh):

    filename = save_folder+"test1.txt"
    num_images = len(testset)
    for i in range(num_images):
        print("Testing image {:d}/{:d}...".format(i+1, num_images))
        img = testset.pull_image(i)
        img_id, annotation = testset.pull_anno(i)
        x = torch.from_numpy(transform(img)[0]).permute(2,0,1)
        x = Variable(x.unsqueeze(0))

        with open(filename, mode='a') as f:
            f.write('\n GROUND TRUTH FOR: ' + img_id + '\n')
            for box in annotation:
                f.write("label"+" || ".join(str(b) for b in box) + "\n")
        if cuda:
            x = x.cuda()
        y = net(x)
        detections = y.data
        # 将检测结果返回到图片上
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        pred_num = 0
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.6:
                if pred_num == 0:
                    with open(filename, mode='a') as f:
                        f.write('PREDICTIONS' + '\n')
                score = detections[0, i, j, 0]
                label_name = labelmap[i-1]
                pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
                coords = (pt[0], pt[1], pt[2], pt[3])
                pred_num += 1
                with open(filename, mode='a') as f:
                    f.write(str(pred_num)+' label:' + label_name + ' score' + str(socre) + ' '+ ' || '.join(str(c) for c in coords) + '\n')
                j += 1

def test_voc():
    # 加载网络
    num_classes = len(VOC_CLASSES) + 1 # 1 为背景
    net = build_ssd("test", 300, num_classes)
    net.load_state_dict(torch.load(args.trained_model))
    net.eval() # 将网络只与eval状态, 主要会影响 dropout 和 BN 等网络层
    print("Finished loading model!")
    # 加载数据
    testset = VOCDetection(args.voc_root, [("2007", "test")], None, VOCAnnotationTransform())
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    test_net(args.save_folder, net, args.cuda, testset, BaseTransform(net.size, (104, 117, 123)), thresh=args.visual_threshold)

if __name__ == '__main__':
    test_voc()
```

# 其他辅助代码

## 学习率衰减

```py

def adjust_learning_rate(optimizer, gamma, step):
    lr = args.lr * (gamma ** (step)) ## **为幂乘
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

```

## Xavier 初始化

```py
# tran.py

def xavier(param):
    init.xavier_uniform(param) # import torch.nn.init as init

def weights_init(m):
    if isinstance(m, nn.Conv2d): # 只对卷积层初始化
        xavier(m.weight.data)
        m.bias.data.zero_()
```
