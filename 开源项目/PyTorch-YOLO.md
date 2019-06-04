---
title: 利用PyTorch自己动手从零实现YOLOv3
sitemap: true
categories: PyTorch
date: 2018-11-22 20:28:09
tags:
- PyTorch
- YOLO
- 源码实现
---
学习一个算法最好的方式就是自己尝试着去实现它! 因此, 在这片博文里面, 我会为大家讲解如何用PyTorch从零开始实现一个YOLOv3目标检测模型, 参考源码请在[这里](https://github.com/eriklindernoren/PyTorch-YOLOv3#train)下载.

在正式介绍 YOLOv3 之前, 我们先将其和 YOLO 的其他版本做一个简单的比较, 它们的网络结构对比如下所示:

这里我们假设大家对YOLOv3的各个细节都比较熟悉, 因此就不对YOLOv3做过多介绍, 如果对YOLOv3不太懂的话, 可以再看看原文, 或者看看我写的[YOLOv3解析](../计算机视觉-YOLOv3-Arxiv2018).

模型实现总共会分为以下六部分:
- (一) 配置文件以及解析
- (二) 搭建YOLO模型框架
- (三) 实现自定义网络层的前向和反向传播过程
- (四) 数据类的设计与实现
- (五) 训练/测试/检测脚本的实现
- (六) 辅助函数及算法实现(目标函数, NMS算法等)


# (一) 配置文件以及解析

## 配置文件

官方代码使用了配置文件来创建网络, `cfg` 文件中描述了网络的整体结构, 它相当于 caffe 中的 `.protxt` 文件一样. 我们也将使用官方的 `cfg` 文件来创建我们的网络, 点击这里下载并它放在 `config/` 文件夹中, 即 `config/yolov3.cfg`.

```py
mkdir config
cd config
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
```

打开该文件, 将会看到类似于下面的信息:
```py
[convolutional]
batch_normalize=1
filters=64
size=3
stride=2
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=32
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear
...
```

### convolutional 和 shortcut

上面的信息中显示了4个 block, 其中 3 个是卷积网络层, 最后一个是 shortcut 网络层, shortcut 网络层是一种 skip connection, 就像 ResNet 中的一样, 其中的 `from` 参数为 `-3` 表示该层的输出是从往前数倒数第三层的图谱 **直接相加** 得到的.

### upsample
`cfg`文件中的 `upsample` 参数代表了双线性插值时使用的 `stride` 参数
```py
[upsample]
stride=2
```

### route
`route` 参数拥有 `layers` 属性, 它的值可以是一个, 也可以是两个, 如下所示. 当 `layers` 属性只含有一个值时, 它会输出指定的网络层的特征图谱, 在下面的例子中, `layers=-4`, 因此, 当前的 `route` 网络层会输出前面的倒数第 4 个网络层的特征图谱. 当 `layers` 属性含有两个值时, 它会输出两个网络层的特征图谱连接(concatenated)后的特征图谱, 在下面的例子中, 当前的 `route` 网络层会将前一层(-1)和第 61 层的特征图片沿着深度维度(depth dimension)进行连接(concatenated), 然后输出连接后的特征图谱.
```py
[route]
layers = -4

[route]
layers = -1, 61
```

### net
`cfg` 文件中的另一种 block 类型是 `net`, 它提供了网络的训练信息, 如下所示:
```py
[net]
# Testing
batch=1
subdivisions=1
# Training
# batch=64
# subdivisions=16
width= 320
height = 320
channels=3
momentum=0.9
decay=0.0005
angle=0
saturation = 1.5
exposure = 1.5
hue=.1
```

## 解析配置文件

我们定义了一个名为 `parse_config.py` 的文件, 其内部的 `parse_model_config()` 函数的参数是指定的 `cfg` 的文件路径, 它的功能是将 `cfg` 文件中的信息加载到模型中, 并且用 **元素为字典的列表** 的形式进行存储, 如下所示:
```py
# ./utils/parse_config.py

def parse_model_config(path):
    f = open(path, 'r') #读取文件
    module_defs = [] # 创建列表, 列表中的元素为字典
    for line in f.readlines(): # 逐行读取
        line = line.strip() # 消除行头尾的空白符(空格, 回车等)
        if not line or line.startswith('#'): # 如果遇到空行或者注释行, 则跳过
            continue
        if line.startswith('['):# 遇到模块的起始, 在列表后添加新的字典
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].strip() # 根据参数值为字典赋值
            if(module_defs[-1]['type']=="convolutional"):
                module_defs[-1]["batch_normalize"] = 0
        else:
            key, value = line.split('=')# 根据参数值为字典赋值, 注意要去除空白符
            module_defs[-1][key.strip()] = value.strip()
    return module_defs
```

调用该函数后, 会返回一个列表, 列表中的每个元素都是一个字典, 代表了配置文件中的以 `[...]` 开头的一个 block, 下面是列表中的部分元素示例:
```py
model_config = parse_model_config("../config/yolov3-tiny.cfg")
print(model_config[0])
print(model_config[1])
```
输出如下:
```
{'channels': '3', 'hue': '.1', 'batch': '1', 'steps': '400000,450000', 'burn_in': '1000', 'max_batches': '500200', 'learning_rate': '0.001', 'exposure': '1.5', 'policy': 'steps', 'height': '416', 'width': '416', 'subdivisions': '1', 'angle': '0', 'type': 'net', 'scales': '.1,.1', 'momentum': '0.9', 'decay': '0.0005', 'saturation': '1.5'}

{'stride': '1', 'activation': 'leaky', 'type': 'convolutional', 'filters': '16', 'pad': '1', 'size': '3', 'batch_normalize': '1'}
```

# (二) 数据类的设计与实现

在搭建 YOLO 模型之前, 我们需要先创建处理数据输入的类, 在 PyTorch 中, 通常是通过集成 `torch.utils.data.Dataset` 类来实现的, 我们需要实现该类的 `__getitem__()` 和 `__len__()` 方法, 实现后, 会将子类的实例作为 `DataLoader` 的参数, 来构建生成 batch 的实例对象. 下面, 先只给出有关数据集类的实现, 具体的加载过程在后续的脚本解析中给出.

## class ImageFolder(Dataset) 类

这里我们起名为 `ImageFolder`, 主要是因为原作者使用了这个名字, 实际上我不太建议使用这个名字, 因为会与 PyTorch 中 `ImageFolder` 类的名字冲突, 容易引起误会, 这里注意一下, 我们这里实现的 `ImageFolder` 类与 PyTorch 中的同名类并没有任何联系. 代码解析如下:
```py
# ./utils/datasets.py

class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        # 获取文件夹下的所有图片路径, glob是一个用于获取路径的通配符模块
        self.files = sorted(glob.glob('%s/*.*' % folder_path))
        #　设置数据集的图片大小属性, 所有的图片都会被放缩到该尺寸
        self.img_shape = (img_size, img_size)

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)] # 根据index获取图片路径
        # Extract image
        img = np.array(Image.open(img_path)) # 利用PIL Image读取图片, 然后转换成numpy数组
        h, w, _ = img.shape # 获取图片的高和宽
        dim_diff = np.abs(h - w) # 计算高宽差的绝对值
        # 根据高宽差计算应该填补的像素数量（填补至高和宽相等）
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # 确定填补位置(填补到边长较短的一边)
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # 调用 np.pad 函数进行填补
        input_img = np.pad(img, pad, 'constant', constant_values=127.5) / 255.
        # 将图片放缩至数据集规定的尺寸, 同时进行归一化操作
        input_img = resize(input_img, (*self.img_shape, 3), mode='reflect')
        # 将通道维度放置在首位(C,H,W)
        input_img = np.transpose(input_img, (2, 0, 1))
        # 将numpy数组转换成tenosr, 数据类型为 float32
        input_img = torch.from_numpy(input_img).float()
        # 返回图片路径和图片 tensor
        return img_path, input_img

    def __len__(self):
        return len(self.files)
```

## class ListDataset(Dataset) 类

`ListDataset` 类定义了训练时所需的数据集和标签, 该类的 `__getitem__()` 方法会返回三个变量, 分别是: 图片路径, 经过放缩处理后的图片(尺寸大小为指定尺寸), 以及经过处理后的 box 坐标信息. 其中, 图片的存储形式为: $(C\times H\times W)$, 标签的存储形式为: $(50 \times 5)$, 这 50 条数据不一定每一条都具有意义, 对于无意义的数据, 其值为 0, 训练时直接跳过即可, 对于有意义的数据, 每一条数据的形式为: $(class_id, x, y, w, h)$, 其中, $class_id$ 是每个 box 对应的目标类别编号,  $x, y, w, h$ 是每个 box 的中心点坐标和宽高, 它们都是以小数形式表示的, 也就是相对于图片宽高的比例.
```py
# ./utils/datasets.py

class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416):
        # list_path: data_config 文件中的 trian 或 val 指定的文件: trainvalno5k.txt 或者 5k.txt
        # 该文件中存放了用于训练或者测试的.jpg图片的路径, 同时根据此路径可以得到对应的 labels 文件
        with open(list_path, 'r') as file:
            self.img_files = file.readlines()
        # 根据图片的路径得到 label 的路径, label 的存储格式为一个图片对应一个.txt文件
        # 文件的每一行代表了该图片的 box 信息, 其内容为: class_id, x, y, w, h (xywh都是用小数形式存储的)
        self.label_files = [path.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt') for path in self.img_files]
        self.img_shape = (img_size, img_size) # 获取图片目标大小, 之后会将图片放缩到此大小, 并相应调整box的数据
        self.max_objects = 50 # 定义每一张图片最多含有的 box 数量

    def __getitem__(self, index):
        # 根据index获取对应的图片路径
        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img = np.array(Image.open(img_path))

        # 如果当前获取到的图片的通道数不为3, 则跳过当前图片, 直到获取到通道数为3的图片
        while len(img.shape)!=3:
            index += 1
            img_path = self.img_files[(index) % len(self.img_files)].rstrip()
            img = np.array(Image.open(img_path))

        # 获取图片的高和宽, 并根据它们的差异对图片执行 padding 操作, 使图片宽高比为1
        h, w, _ = img.shape
        dim_diff = np.abs(h - w)        
        pad1, pad2 = dim_diff//2, dim_diff - dim_diff//2
        pad = ((pad1, pad2), (0,0), (0,0)) if h<=w else ((0,0), (pad1, pad2), (0,0))        
        input_img = np.pad(img, pad, 'constant', constant_values=128) / 255.

        # 暂存padding后的图片的宽和高
        padded_h, padded_w, _ = input_img.shape

        # 将图片大小放缩到指定的存储, 并将通道数放置到高和宽之前
        input_img = resize(input_img, (*self.img_shape, 3), mode='reflect')
        input_img = np.transpose(input_img, (2,0,1))

        # 将图片转化成 tensor
        input_img = torch.from_numpy(input_img).float()

        # 获取图片对应的 label 文件的路径
        label_path = self.label_files[index % len(self.img_files)].rstrip()

        labels = None
        # 根据图片 padding 之后的存储, 对 label 文件中的 box 坐标按比例进行缩放
        if os.path.exists(label_path):
            labels = np.loadtxt(label_path).reshape(-1, 5)

            x1 = w * (labels[:, 1] - labels[:, 3] / 2) # 先获取box左上角和右下角的像素坐标
            y1 = h * (labels[:, 2] - labels[:, 4] / 2)
            x2 = w * (labels[:, 1] + labels[:, 3] / 2)
            y2 = h * (labels[:, 2] + labels[:, 4] / 2)

            # 根据 padding 的大小, 更新这些坐标的值
            x1 += pad[1][0]
            y1 += pad[0][0]
            x2 += pad[1][0]
            y2 += pad[0][0]

            # 重新将坐标转化成小数模式(相对应padding后的宽高的比例)
            labels[:, 1] = ((x1+x2)/2) / padded_w
            labels[:, 2] = ((y1+y2)/2) / padded_h
            labels[:, 3] *= w / padded_w
            labels[:, 4] *= h / padded_h
        filled_labels = np.zeros((self.max_objects, 5)) # 创建50×5的占位空间
        if labels is not None: # 将更新后的box坐标填充到刚刚申请的占位空间中
            filled_labels[range(len(labels))[:self.max_objects]] = labels[:self.max_objects]
        # 将 label 转化成 tensor
        filled_labels =torch.from_numpy(filled_labels)

        # 返回图片路径, 图片tensor, label tensor
        return img_path, input_img, filled_labels

    def __len__(self):
        return len(self.img_files)
```

# (三) 搭建YOLO模型框架

在 `models.py` 文件中, 定义了 YOLO 的模型框架, 文件概览及类之间的调用关系如下:

```py
# ./models.py

import torch
#...

def create_modules(module_defs):
    # 根据配置文件的列表字典创建模型
    # ...
    EmptyLayer()
    YOLOLayer()

class EmptyLayer(nn.Module):
    # 'route' 和 'shortcut' 网络层的占位符(placeholder)
    # ...

class YOLOLayer(nn.Module):
    # Detection Layer
    # ...

class Darknet(nn.Module):
    # YOLOv3 object detection model
    def __init__(self, config_path, img_size=416):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_config(config_path)
        # 这里调用了 create_modules 函数来根据配置文件的信息创建对应的网络
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        # ...
    # ...
```

## create_modules() 函数

下面我们先来看看模型创建函数 `create_modules` 的代码解析:

```py
# ./models.py

def create_modules(module_defs):
    # 参数 module_defs 是根据配置文件生成的列表字典, 列表中的每一个字典都代表一个网络层模块

    hyperparams = module_defs.pop(0) # 第0个字段是 [net] 模块, 存储了模型的一些超参数

    output_filters = [int(hyperparams["channels"])] # 获取输入层的输出通道数,
    # out_filers 是一个列表, 后续还会添加其他网络层的输出通道数

    module_list = nn.ModuleList()

    for i, module_def in enumerate(module_defs): # 遍历配置文件中的每一个模块([net]模块已被弹出)
        modules = nn.Sequential() # 存储一个模块, 一个模块可能包含多个层, 如 卷积+激活

        if module_def["type"] == "convolutional":
            bn = int(module_def["batch_normalize"])
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            pad = (kernel_size-1) // 2 if int(module_def["pad"]) else 0 # 维持卷积前后图片大小不变

            modules.add_module(
                "conv_%d" % i, # 名字
                nn.Conv2d(
                    in_channels=output_filters[-1], # 前一层的输出就是这一层的输入
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"])
                    padding=pad,
                    bias=not bn,# 当带有 BN 层时, 会抵消掉前一层的偏置项(可通过数学计算证明)
                ),
            )

            if bn: # 添加 BatchNorm 网络层
                modules.add_module("batch_norm_%d" % i, nn.BatchNorm2d(filters))

            if module_def["activation"] == "leaky": # 添加激活层
                modules.add_module("leaky_%d" % i, nn.LeakyReLU(0.1))

        elif module_def["type"] == "maxpool":
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])

            if kernel_size == 2 and stride == 1:
                padding = nn.ZeroPad2d((0,1,0,1)) # 在右边和下边添加 zero padding
                modules.add_module("_debug_padding_%d" % i, padding)

            # 定义 max_pool 网络层, 注意 maxpool 没有 filter 参数
            maxpool = nn.MaxPool2d(
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size-1) // 2,
            )

            modules.add_module("maxpool_%d" % i, maxpool)

        elif module_def["type"] == "upsample": # 根据 stride 扩大特征图谱的宽和高

            # 目前, 新版本的 PyTorch 已经逐渐启用 Upsample, 而推荐使用更加一般化的 nn.functional.interpolate
            upsample = nn.Upsample(scale_factor=int(module_def["stride"]), mode="nearest")

            modules.add_module("upsample_%d" % i, upsample)

        elif module_def["type"] == "route":
            layers = [int(x) for x in module_def["laers"].split(',')]
            filters = sum([output_filters[layer_i] for layer_i in layers])

            module.add_module("route_%d" % i, EmptyLayer())

        elif module_def["type"] == "shortcut":
            filters = output_filters[module_def["from"]]

            modules.add_module("shortcut_%d" % i, EmptyLayer())

        elif module_def["type"] == "yolo":

            anchor_idxs = [int(x) for x in module_def["mask"].split(',')]

            # 提取 anchors
            anchors = [int(x) for x in module_def["anchors"].split(',')]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]

            anchors = [anchors[i] for i in anchor_idxs]

            num_classes = module_def["classes"]
            img_height = hyperparams["height"]

            # 定义 Detection Layer
            yolo_layer = YOLOLayer(anchors, num_classes, img_height)
            modules.add_module("yolo_%d" % i, yolo_layer)

        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list
```

## class EmptyLayer(nn.Module)

在上面的代码中, 对于 `route` 和 `shortcut` 使用了自定义的 `class EmptyLayer(nn.Module)`, 该类主要起到一个占位符(placeholder)的作用, 其内部实现会根据模块的类型不同而有所区别, 下面是该类的定义:

```py
# ./models.py

class EmptyLayer(nn.Module):

    def __init__(self):
        super(EmptyLayer, self).__init__()
```

## class YOLOLayer(nn.Module)

接着, 对于 `yolo` 模块, 使用了 `class YOLOLayer(nn.Module)` , 该类的定义如下:

```py
# ./models.py

class YOLOLayer(nn.Module):

    def __init__(self, anchors, num_classes, img_dim):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors # anchors = [(116,90),(156,198),(373,326)]
        self.num_anchors = len(anchors) # 3
        self.num_classes = num_classes # 80
        self.bbox_attrs = 5 + num_classes #
        self.image_dim = img_dim # 416
        self.ignore_thres = 0.5
        self.lambda_coord = 1

        self.mse_loss = nn.MSELoss(size_average=True)
        self.bce_loss = nn.BCELoss(size_average=True)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, x, targets=None):
        # x: [1, 255, 13, 13]
        # targets: [50, 5]
        nA = self.num_anchors # 3
        nB = x.size(0) # batch_size
        nG = x.size(2) # W = 13
        stride = self.image_dim / nG # 416 / W = 416 / 13 = 32

        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor
        # (batch, anchors, 5+num_classes, x.size(2), x.size(2)), 调换顺序
        # [1, 3, 13, 13, 85]
        prediction = x.view(nB, nA, self.bbox_attrs, nG, nG).permute(0,1,3,4,2).contiguous()

        x = torch.sigmoid(prediction[..., 0]) # center x: [1, 3, 13, 13]
        y = torch.sigmoid(prediction[..., 1]) # center y: [1, 3, 13, 13]
        w = prediction[..., 2] # width: [1, 3, 13, 13]
        h = prediction[..., 3] # height: [1, 3, 13, 13]
        pred_conf = torch.sigmoid(prediction[..., 4]) # [1, 3, 13, 13]
        pred_cls = torch.sigmoid(prediction[..., 5:]) # [1, 3, 13, 13, 80]

        # grid_x的shape为[1,1,nG,nG], 每一行的元素为:[0,1,2,3,...,nG-1]
        grid_x = torch.arange(nG).repeat(nG, 1).view([1,1,nG,nG]).type(FloatTensor)
        # grid_y的shape为[1,1,nG,nG], 每一列元素为: [0,1,2,3, ...,nG-1]
        grid_y = torch.arange(nG).repeat(nG, 1).t().view(1,1,nG,nG).type(FloatTensor)

        # scaled_anchors 是将原图上的 box 大小根据当前特征图谱的大小转换成相应的特征图谱上的 box
        # shape: [3, 2]
        scaled_anchors = FloatTensor([(a_w / stride, a_h / stride) for a_w, a_h in self.anchors])

        # 分别获取其 w 和 h, 并将shape形状变为: [1,3,1,1]
        anchor_w = scaled_anchors[:, 0:1].view((1, nA, 1, 1))
        anchor_h = scaled_anchors[:, 1:2].view((1, nA, 1, 1))

        # shape: [1, 3, 13, 13, 4], 给 anchors 添加 offset 和 scale
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 0] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

        if targets is not None:# 如果提供了 targets 标签, 则说明是处于训练阶段

            if x.is_cuda:
                self.mse_loss = self.mse_loss.cuda()
                self.bce_loss = self.bce_loss.cuda()
                self.ce_loss = self.ce_loss.cuda()

            # 调用 utils.py 文件中的 build_targets 函数, 将真实的 box 数据转化成训练用的数据格式
            # nGT = int 真实box的数量
            # nCorrect = int 预测正确的数量
            # mask: torch.Size([1, 3, 13, 13])
            # conf_mask: torch.Size([1, 3, 13, 13])
            # tx: torch.Size([1, 3, 13, 13])
            # ty: torch.Size([1, 3, 13, 13])
            # tw: torch.Size([1, 3, 13, 13])
            # th: torch.Size([1, 3, 13, 13])
            # tconf: torch.Size([1, 3, 13, 13])
            # tcls: torch.Size([1, 3, 13, 13, 80])
            nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, cls = build_targets(
                pred_boxes = pred_boxes.cpu().data,
                pred_conf=pred_cls.cpu().data,
                target=targets.cpu().data,
                anchors=scaled_anchors.cpu().data,
                num_anchors=nA,
                num_classes=self.num_classes,
                grid_size=nG,
                ignore_thres=self.ignore_thres,
                img_dim=self.image_dim,
            )

            nProposals = int((pred_conf > 0.5).sum().item()) # 计算置信度大于0.5的预测box数量
            recall = float(nCorrect / nGT) if nGT else 1 # 计算召回率
            precision = float(nCorrect / nProposals)

            # 处理 masks
            mask = Variable(mask.type(ByteTensor))
            conf_mask = Variable(conf_mask.type(ByteTensor))

            # 处理 target Variables
            tx = Variable(tx.type(FloatTensor), requires_grad=False)
            ty = Variable(ty.type(FloatTensor), requires_grad=False)
            tw = Variable(tw.type(FloatTensor), requires_grad=False)
            th = Variable(th.type(FloatTensor), requires_grad=False)
            tconf = Variable(tconf.type(FloatTensor), requires_grad=False)
            tcls = Variable(tcls.type(LongTensor), requires_grad=False)

            # 获取表明gt和非gt的conf_mask
            # 这里 conf_mask_true 指的是具有最佳匹配度的anchor box
            # conf_mask_false 指的是iou小于0.5的anchor box, 其余的anchor box都被忽略了
            conf_mask_true = mask # mask 只有best_n对应位为1, 其余都为0
            conf_mask_false = conf_mask-mask # conf_mask中iou大于ignore_thres的为0, 其余为1, best_n也为1

            # 忽略 non-existing objects, 计算相应的loss
            loss_x = self.mse_loss(x[mask], tx[mask])
            loss_y = self.mse_loss(y[mask], ty[mask])
            loss_w = self.mse_loss(w[mask], tw[mask])
            loss_h = self.mse_loss(h[mask], th[mask])

            # 这里 conf_mask_true 指的是具有最佳匹配度的anchor box
            # conf_mask_false 指的是iou小于0.5的anchor box, 其余的anchor box都被忽略了
            loss_conf = self.bce_loss(
                pred_conf[conf_mask_false], tconf[conf_mask_false]
            ) + self.bce_loss(
                pred_conf[conf_mask_true], tconf[conf_mask_true]
            )

            # pred_cls[mask]的shape为: [7,80], torch.argmax(tcls[mask], 1)的shape为[7]
            # CrossEntropyLoss对象的输入为(x,class), 其中x为预测的每个类的概率, class为gt的类别下标
            loss_cls = (1 / nB) * self.ce_loss(pred_cls[mask], torch.argmax(tcls[mask], 1))

            loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            return (
                loss,
                loss_x.item(),
                loss_y.item(),
                loss_w.item(),
                loss_h.item(),
                loss_conf.item(),
                loss_cls.item(),
                recall,
                precision,
            )
        else:
            # 非训练阶段则直接返回准确率, output的shape为: [nB, -1, 85]
            output = torch.cat(
                (
                    pred_boxes.view(nB, -1, 4) * stride,
                    pred_conf.view(nB, -1, 1),
                    pred_cls.view(nB, -1, self.num_classes),
                ),
                -1,
            )

            return output
```

上面 `YOLOLayer` 类的 `forward()` 函数使用了 `build_targets()` 函数来将真实的标签数据转化成训练用的格式, 关于该函数的解析可以看 utils.py 文件解析中的 [build_target()函数](#build_target)

## class Darknet(nn.Module)



```py
# ./models.py

class Darknet(nn.Module):
    # yolo v3 检测模型
    def __init__(self, config_path, img_size=416):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_config(config_path)
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        self.img_size = img_size
        self.seen = 0
        self.header_info = np.array([0,0,0,self.seen,0])
        self.loss_names = ["x", "y", "w", "h", "conf", "cls", "recall", "precision"]

    def forward(self, x, targets=None):
        is_training = targets is not None # 如果targets不为None, 则将is_training设为true
        output = []
        self.losses = defaultdict(float)
        layer_outputs = []
        print("input: ", x.shape)

        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                # 如果是内置的网络层类型, 则直接调用其 forward 函数即可
                x = module(x)
            elif module_def["type"] == "route":
                layer_i = [int(x) for x in module_def["layers"].split(",")]
                # x的shape为:[N,C,W,H], 因此, dim=1代表在深度维度上进行连接
                x = torch.cat([layer_outputs[i] for i in layer_i], 1)
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                # 注意这里可看出route的shortcut的区别, 前者相当于短路(不在乎前一层的输出),
                # 后者相当于res模块(需要加上前一层的输出)
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def["type"] == "yolo":
                if is_training:
                    # yolo module的输出为(tuple):
                    # ( loss,loss_x.item(),loss_y.item(),loss_w.item(),loss_h.item(),
                    # loss_conf.item(),loss_cls.item(),recall,precision )
                    # 令 x = loss, losses=(剩余的元素组成的tuple)
                    x, *losses = module[0](x, targets)
                    for name, loss in zip(self.loss_names, losses): #将losses根据名字加入字典
                        self.losses[name] += loss
                else:# 如果是非training阶段, 则直接计算结果, 无需记录loss
                    x = module(x)

                # 记录yolo层的预测结果
                output.append(x)

            #记录每一层的输出
            layer_outputs.append(x)

        self.losses["recall"] /= 3
        self.losses["precision"] /= 3

        # 如果是训练阶段, 则计算和, 否则, 沿着深度维度将不同yolo层的预测结果连接起来并返回
        return sum(output) if is_training else torch.cat(output, 1)
```


# (四) 实现自定义网络层的前向和反向传播过程

# (五) 训练/测试/检测脚本的实现

## detect.py

该函数定义了模型的检测逻辑, 调用该函数, 会将图片送入模型中去运算, 并且会返回相应的预测结果, 然后, 需要对预测结果执行 NMS 算法, 消除重叠的框, 最后, 将预测结果以`.png`的格式进行可视化存储.

```py
# ./detect.py

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator


parser = argparse.ArgumentParser()
parser.add_argument('--image_folder', type=str, default='data/samples', help='path to dataset')
parser.add_argument('--config_path', type=str, default='config/yolov3.cfg', help='path to model config file')
parser.add_argument('--weights_path', type=str, default='weights/yolov3.weights', help='path to weights file')
parser.add_argument('--class_path', type=str, default='data/coco.names', help='path to class label file')
parser.add_argument('--conf_thres', type=float, default=0.8, help='object confidence threshold')
parser.add_argument('--nms_thres', type=float, default=0.4, help='iou thresshold for non-maximum suppression')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_size', type=int, default=416, help='size of each image dimension')
parser.add_argument('--use_cuda', type=bool, default=True, help='whether to use cuda if available')
opt = parser.parse_args([])
print(opt)


# 指示当前cuda是否可用
cuda = torch.cuda.is_available() and opt.use_cuda

# 创建模型并加载权重
model = Darknet(opt.config_path, img_size=opt.img_size)
model.load_weights(opt.weights_path)

# 如果cuda可用, 则将model移至cuda
if cuda:
    model.cuda()

model.eval() # 将模型的状态置为eval状态(会改变月一些内置网络层的行为)

img_datasets = ImageFolder(opt.image_folder, img_size=opt.img_size)
dataloader = DataLoader(img_datasets,
                        batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu
                       )

# 调用utils/utils.py文件中的load_classes()函数加载类别的名字(['person','car',...,'toothbrush'])
classes = load_classes(opt.class_path)

Tensor = torch.cuda.FloatTensor if cuda else torch.FLoatTnesor

imgs = [] # 存储图片路径
img_detections = [] # 存储每张图片的检测结果

data_size = len(img_datasets) # 图片的数量
epoch_size = len(dataloader) # epoch的数量: data_size / batch_size

print ('\nPerforming object detection: {} images, {} epoches'.format(data_size, epoch_size))

prev_time = time.time()
for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
    # 配置输入
    input_imgs = Variable(input_imgs.type(Tensor)) # Tensor: FloatTensor

    # 获取预测结果
    with torch.no_grad():
        # detections的shape为: [1,10647,85], 其中, 1为batch_size
        # 因为对于尺寸为416的图片来说:(13*13+26*26+52*52) * 3 = 10647
        # 如果图片尺寸为608(必须为32的倍数), 那么就为:(19*19+38*38+76*76) * 3 = 22743
        detections = model(input_imgs)

        # nms: 对于每一类(不同类之间的box不执行nms), 先选出具有最大score的box,
        # 然后删除与该box交并比较大的同类box, 接着继续选下一个最大socre的box, 直至同类box为空
        # 注意yolo与faster rcnn在执行nms算法时的不同, 前者是在多类上执行的, 后者是在两类上执行的
        # 执行nms后, 这里的detections是一个列表, 列表中的每个元素代表着一张图片nms后的box集合
        # 每一张图片的shape为:[m, 7], m代表box的数量, 7代表:(x1,y1,x2,y2,obj_conf,class_conf,class_pred)
        detections = non_max_suppression(detections, 80, opt.conf_thres, opt.nms_thres)


        #break

    # 记录当前时间
    current_time = time.time()
    # 计算detect花费的时间(一张图片)
    inference_time = datetime.timedelta(seconds=current_time - prev_time)

    # 更新prev_time
    prev_time = current_time

    # 打印到屏幕
    print ('\t+ Batch %d, Inference Time: %s' % (batch_i, inference_time))

    # 记录图片的路径和检测结果, 以便后面进行可视化
    imgs.extend(img_paths)
    img_detections.extend(detections)


# 检测完成后, 根据 imgs 和 img_detections 的值进行可视化(以.png图片形式存储在磁盘上)

# 设值边框的颜色
cmap = plt.get_cmap('tab20b')
colors = [cmap(i) for i in np.linspace(0, 1, 20)]

print('\nSaving image:')

# 遍历所有的imgs 和 img_detections, 对检测结果进行可视化
for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):
    print ("(%d) Image: '%s'" % (img_i, path))

    # 创建plot
    img = np.array(Image.open(path))
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(img) # 将img添加到当前的plot中

    # 计算给当前图片添加的padding的像素数
    pad_x = max(img.shape[0] - img.shape[1], 0) * (opt.img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (opt.img_size / max(img.shape))

    # 获取移除padding之后的图片的宽和高, 注意这个宽和高不同图片的原始大小, 而是放缩后的大小(长边为opt.img_size)
    unpad_h = opt.img_size - pad_y
    unpad_w = opt.img_size - pad_x

    # 在图片上画相应的box的边框和标签
    if detections is not None:
        # 获取当前图片中出现过的标签
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels) # 获取出现过的标签的数量
        bbox_colors = random.sample(colors, n_cls_preds) # 为每个类别标签随机分配颜色

        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            # 输出当前box的标签和相应的概率
            print ('\t+ Label: %s, Conf: %.5f' % (classes[int(cls_pred)], cls_conf.item()))

            # 将坐标转换到原始图片上的像素坐标
            box_h = ((y2-y1) / unpad_h) * img.shape[0]
            box_w = ((x2-x1) / unpad_w) * img.shape[1]            
            y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
            x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]

            # 获取当前类别的颜色
            color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]

            # 创建矩形
            bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor='none')

            # 将创建好的矩形添加到当前的plot中(会加载在图片的上面)
            ax.add_patch(bbox)
            # 添加标签
            plt.text(x1, y1, s=classes[int(cls_pred)], color='white', verticalalignment='top', bbox={'color':color, 'pad':0})

    # 将图片保存在磁盘上
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    plt.savefig('output/%d.png' % (img_i), bbox_inches='tight', pad_inches=0.0)
    plt.close()

```

## train.py 训练脚本

```py
# ./train.py

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=30, help="number of epochs")
parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
parser.add_argument("--batch_size", type=int, default=16, help="size of each image batch")
parser.add_argument("--model_config_path", type=str, default="config/yolov3.cfg", help="path to model config file")
parser.add_argument("--data_config_path", type=str, default="config/coco.data", help="path to data config file")
parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
parser.add_argument(
    "--checkpoint_dir", type=str, default="checkpoints", help="directory where model checkpoints are saved"
)
parser.add_argument("--use_cuda", type=bool, default=True, help="whether to use cuda if available")
opt = parser.parse_args([])
print(opt)

cuda = torch.cuda.is_available() and opt.use_cuda

os.makedirs("output", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# 加载各个类的名字
classes = load_classes(opt.class_path)

# 加载数据集相关配置(主要是路径)
data_config = parse_data_config(opt.data_config_path)
train_path = data_config["train"]

# 获取模型超参数
hyperparams = parse_model_config(opt.model_config_path)[0]
learning_rate = float(hyperparams["learning_rate"])
momentum = float(hyperparams["momentum"])
decay = float(hyperparams["decay"])
burn_in = int(hyperparams["burn_in"])

# 初始化创建模型结构
model = Darknet(opt.model_config_path)

# 随机初始化权重, weights_init_normal是定义在utils.py文件中函数, 会对模型进行高斯随机初始化
model.apply(weights_init_normal)

if cuda:
    model = model.cuda()

model.train() # 将模型置于训练模式

# ListDataset是用于训练时使用的数据集类, 它会返回以下三个变量:
# 图片路径(str), 图片(3,416,416), 以及图片的box标签信息(50,5)
dataloader = torch.utils.data.DataLoader(
    ListDataset(train_path), batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu
)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# 下式的lambda函数等价于: Adam(p for p in model.parameters() if p.requires_grad== True)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

for epoch in range(opt.epochs):
    for batch_i, (_, imgs, targets) in enumerate(dataloader):
        # imgs: [16, 3, 416, 416]
        # targets: [16, 50, 5]
        imgs = Variable(imgs.type(Tensor))
        targets = Variable(targets.type(Tensor), requires_grad=False)

        # 清空优化器中的缓存梯度
        optimizer.zero_grad()

        loss = model(imgs, targets)

        loss.backward() # 执行反向传播算法
        optimizer.step() # 根据梯度对参数进行更新

        # 打印当前训练状态的各项损失值
        print(
            "[Epoch %d/%d, Batch %d/%d] [Losses: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f, recall: %.5f, precision: %.5f]"
            % (
                epoch,
                opt.epochs,
                batch_i,
                len(dataloader),
                model.losses["x"],
                model.losses["y"],
                model.losses["w"],
                model.losses["h"],
                model.losses["conf"],
                model.losses["cls"],
                loss.item(),
                model.losses["recall"],
                model.losses["precision"],
            )
        )

        # 记录当前处理过的图片的总数
        model.seen += imgs.size(0) # 16
    if epoch % opt.checkpoint_interval == 0:
        # 调用 ./models.py 文件中的 save_weights 函数, 将训练好的参数权重进行存储
        model.save_weights("%s/%d.weights" % (opt.checkpoint_dir, epoch))
```

## test.py 测试脚本

```py
# ./test.py

from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=16, help="size of each image batch")
parser.add_argument("--model_config_path", type=str, default="config/yolov3.cfg", help="path to model config file")
parser.add_argument("--data_config_path", type=str, default="config/coco.data", help="path to data config file")
parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.45, help="iou thresshold for non-maximum suppression")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
parser.add_argument("--use_cuda", type=bool, default=True, help="whether to use cuda if available")
opt = parser.parse_args([])
print(opt)

cuda = torch.cuda.is_available() and opt.use_cuda

# 获取数据集配置(路径)
data_config = parse_data_config(opt.data_config_path)
test_path = data_config["valid"]
num_classes = int(data_config["classes"])

# 初始化网络模型结构
model = Darknet(opt.model_config_path)

# 调用 ./models.py 文件中的 load_weights 函数加载模型的预训练权重
model.load_weights(opt.weights_path)

if cuda:
    model = model.cuda()

model.eval() # 将模型置于推演模式eval

# 获取数据集加载器, 这里需要根据数据的标签计算准确率, 因此需要使用ListDataset
dataset = ListDataset(test_path)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

print("Compute mAP...")

all_detections = []
all_annotations = []

for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
    imgs = Variable(imgs.type(Tensor))

    with torch.no_grad(): # 禁止计算梯度, 加快模型运算速度
        outputs = model(imgs)
        # 对计算结果执行 NMS 算法
        # outputs的shape为:[batch_size, m, 7]
        outputs = non_max_suppression(outputs, 80, conf_thres=opt.conf_thres, nms_thres=opt.nms_thres)

    for output, annotations in zip(outputs, targets): #targets的shape为:[batch_size, n, 5]
        # 根据类别的数量创建占位空间, all_detections为一个列表, 列表中只有一个元素,
        # 该元素还是一个列表, 该列表中有80个np元素
        all_detections.append([np.array([]) for _ in range(num_classes)])

        if output is not None:
            # 获取预测结果的相应值
            pred_boxes = output[:, :5].cpu().numpy() # 坐标和包含物体的概率obj_conf
            scores = output[:, 4].cpu().numpy() # 置信度
            pred_labels = output[:, -1].cput().numpy() # 类别编号

            # 按照置信度对预测的box进行排序
            sort_i = np.argsort(scores)
            pred_labels = pred_labels[sort_i]
            pred_boxes = pred_boxes[sort_i]

            for label in range(num_classes):
                # all_detections是只有一个元素的列表, 因此这里用-1,
                # 获取所有预测类别为label的预测box, 可以将all_detections的shape看作为[1,1,80]
                all_detections[-1][label] = pred_boxes[pred_labels == label]

        # [1,1,80]
        all_annotations.append([np.array([]) for _ in range(num_classes)])

        if any(annotations[:, -1] > 0):
            annotations_labels = annotations[annotations[:, -1] > 0, 0].numpy() # 获取类别编号
            _annotation_boxes = annotations[annotations[:, -1] > 0, 1:].numpy() # 获取box坐标

            # 将box的格式转换成x1,y1,x2,y2的形式, 同时将图片放缩至opt.img_size大小
            annotation_boxes = np.empty_like(_annotation_boxes)
            annotation_boxes[:, 0] = _annotation_boxes[:, 0] - _annotation_boxes[:, 2] / 2
            annotation_boxes[:, 1] = _annotation_boxes[:, 1] - _annotation_boxes[:, 3] / 2
            annotation_boxes[:, 2] = _annotation_boxes[:, 0] + _annotation_boxes[:, 2] / 2
            annotation_boxes[:, 3] = _annotation_boxes[:, 1] + _annotation_boxes[:, 3] / 2
            # 因为原始的标签数据是以小数形式存储的, 所以可以直接利用乘法进行放缩
            annotation_boxes *= opt.img_size

            for label in range(num_classes):
                all_annotations[-1][label] = annotation_boxes[annotation_labels == label, :]

# 以字典形式记录每一类的mAP值
average_precisions = {}
for label in range(num_classes):
    true_positives = []
    scores = []
    num_annotations = 0

    for i in tqdm.tqdm(range(len(all_annotations)), desc="Computing AP for class '{}'".format(label)):

        # 获取同类的预测结果和标签信息, i代表当前图片在batch中的位置
        detections = all_detections[i][label]
        annotations = all_annotations[i][label]

        num_annotations += annotations.shape[0]
        detected_annotations = []

        for *bbox, score in detections:
            scores.append(score)

            if annotations.shape[0] == 0:
                true_positives.addpend(0) # 当前box并非真正例
                continue

            # 利用./utils/utils.py文件中的bbox_iou_numpy函数获取交并比矩阵(都是同类的box)
            overlaps = bbox_iou_numpy(np.expand_dims(bbox, axis=0), annotations)
            assigned_annotation = np.argmax(overlaps, axis=1) # 获取最大交并比的下标
            max_overlap = overlaps[0, assigned_annotation] # 获取最大交并比

            if max_overlap >= opt.iou_thres and assigned_annotation not in detected_annotations:
                true_positives.append(1)
                detected_annotations.append(assigned_annotation)
            else:
                true_positives.append(0)

    # 如果当前类没有出现在该图片中, 在当前类的 AP 为 0
    if num_annotations == 0:
        average_precisions[label] = 0
        continue

    true_positives = np.array(true_positives) # 将列表转化成numpy数组
    false_positives = np.ones_like(true_positives) - true_positives

    #按照socre进行排序
    indices = np.argsort(-np.array(scores))
    false_positives = false_positives[indices]
    true_positives = true_positives[indices]

    # 统计假正例和真正例
    false_positives = np.cumsum(false_positives)
    true_positives = np.cumsum(true_positives)

    # 计算召回率和准确率
    recall = true_positives / num_annotations
    precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

    # 调用utils.py文件中的compute_ap函数计算average precision
    average_precision = compute_ap(recall, precision)
    average_precisions[label] = average_precision

print("Average Precisions:")
for c, ap in average_precisions.items():
    print("+ Class '{}' - AP: {}".format(c, ap))

mAP = np.mean(list(average_precisions.values()))
print("mAP: {}".format(mAP))
```

# (六) 辅助函数及算法实现(目标函数, NMS算法等)

## utils.py

### load_classes()

### weights_init_normal()

### compute_ap()

### bbox_iou()

在 `build_targets` 函数中, 使用了 `bbox_iou()` 函数来计算两组 box 之间的 iou 大小, 代码实现逻辑如下所示:

```py
#./utils/utils.py

def bbox_iou(box1, box2, x1y1x2y2=True):
    # 返回 box1 和 box2 的 iou, box1 和 box2 的 shape 要么相同, 要么其中一个为[1,4]
    if not x1y1x2y2:
        # 获取 box1 和 box2 的左上角和右下角坐标
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # 获取 box1 和 box2 的左上角和右下角坐标
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # 获取相交矩形的左上角和右下角坐标
    # 注意, torch.max 函数要求输入的两个参数要么 shape 相同, 此时在相同位置上进行比较并取最大值
    # 要么其中一个 shape 的第一维为 1, 此时会自动将该为元素与另一个 box 的所有元素做比较, 这里使用的就是该用法.
    # 具体来说, b1_x1 为 [1, 1], b2_x1 为 [3, 1], 此时会有 b1_x1 中的一条数据分别与 b2_x1 中的三条数据做比较并取最大值
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # 计算相交矩形的面积
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # 分别求 box1 矩形和 box2 矩形的面积.
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    # 计算 iou 并将其返回
    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou
```

### bbox_iou_numpy()

### non_max_suppression()

对预测的结果执行 NMS 算法, 传入的预测结果shape为: [1,10647,85], 最终会返回一个列表, 列表中的每个元素是每张图片的box组成的tensor, box的shape为: (x1, y1, x2, y2, object_conf, class_score, class_pred).
在 YOLO 中, 是对每一个类别(如80类)执行 NMS 算法. 而在 Faster R-CNN 中, 是对两个类进行 NMS 算法, 因此, 在 Faster R-CNN 中, 对于不同的类的 box, 如果它们的重叠度较高, 那么就会删除其中的一个.

```py
# ./utils/utils.py

# nms: 对于每一类(不同类之间的box不执行nms), 先选出具有最大score的box, 删除与该box交并比较大的同类box,
# 接着继续选下一个最大socre的box, 直至同类box为空, 然后对下一类执行nms
# 注意yolo与faster rcnn在执行nms算法时的不同, 前者是在多类上执行的, 后者是在两类上执行的
def non_max_suppression(prediction, num_classes, conf_thres=0.5, nms_thres=0.4):
    # prediction的shape为: [1,10647,85], 其中, 1为batch_size, 10647是尺寸为416的图片的anchor box的总数
    # num_classes: 80
    # 移除那些置信度低于conf_thres的boxes, 同时在剩余的boxes上执行NMS算法
    # 返回值中box的shape为: (x1, y1, x2, y2, object_conf, class_score, class_pred)

    # 获取box的(x1,x2,y1,y2)坐标
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    # len(prediction)为Batch_size, 这里申请了占位空间, 大小为batch_size
    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # 先清除所有置信度小于conf_thres的box, conf_mask的shape为:[n], n为置信度大于阈值的box数量
        conf_mask = (image_pred[:, 4] >= conf_thres).squeeze() # 这里的squeeze()可加可不加
        image_pred = image_pred[conf_mask] # image_pred的shape为[n, 85]

        if not image_pred.size(0):
            continue # 如果所有的box的置信度都小于阈值, 那么就跳过当前的图片, 对下一张进行操作

        # 获取每个box的类别的预测结果和编号(0~79), 使用了keepdim, 否则shape维数会减一(dim指定的维度会消失)
        # class_conf的shape为[n, 1], 代表n个box的score
        # class_pred的shape为[n, 1], 代表n个box的类别编号
        class_conf, class_pred = torch.max(image_pred[:, 5 : 5 + num_classes], 1, keepdim=True)

        # 对以上结果进行汇总, shape为[n,7]: (x1,y1,x2,y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)

        # 获取当前image中出现过的类别号, 然后分别对每一类执行NMS算法
        unique_labels = detections[:, -1].cpu().unique()

        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()

        # 分别对每一类执行NMS算法, 注意这点与faster rcnn不同, 后者只对两类执行nms算法, 也就是是否出现物体
        # faster rcnn的nms算法会有一个问题, 那就是当两个不同物体重复度较高时, fasterrcnn会忽略置信度较低的一个
        for c in unique_labels:
            # 获取指定类别的所有box
            detections_class = detections[detections[:, -1] == c] # detections的最后一维指示类别编号

            # 按照每个box的置信度进行排序(第5维代表置信度 score)
            _, conf_sort_index = torch.sort(detections_class[:, 4], descending=True)
            detections_class = detections_class[conf_sort_index]

            # 执行NMS算法, 核心思想是先将具有最大socre的box放置在max_detections列表当中,
            # 然后令该box与剩余的所有同类box计算交并比, 接着删除那些具有较大交并比的box(大于阈值)
            # 重复对detections_class执行上面两步操作, 知道detections_class中只剩下一个box为止
            max_detections = []
            while detections_class.size(0):
                # 将具有最大score的box添加到max_detections列表中,
                # 注意要将box的shape扩展成:[1,7], 方便后续max的连接(cat)
                max_detections.append(detections_class[0].unsqueeze(0))

                # 当只剩下一个box时, 当前类的nms过程终止
                if len(detections_class) == 1:
                    break

                # 获取当前最大socre的box与其余同类box的iou, 调用了本文件的bbox_iou()函数
                ious = bbox_iou(max_detections[-1], detections_class[1:])

                # 移除那些交并比大于阈值的box(也即只保留交并比小于阈值的box)
                detections_class = detections_class[1:][ious < nms_thres]

            # 将执行nms后的剩余的同类box连接起来, 最终shape为[m, 7], m为nms后同类box的数量
            max_detections = torch.cat(max_detections).data

            # 将计算结果添加到output返回值当中, output是一个列表, 列表中的每个元素代表这一张图片的nms后的box
            # 注意, 此时同一张图片的不同类的box也会连接到一起, box的最后一维会存储类别编号(4+1+1+1).
            output[image_i] = (
                max_detections if output[image_i] is None else torch.cat(
                    (output[image_i], max_detections)
                )
            )

    return output
```


<span id="build_targets">
### build_targets() 函数

该函数会根据 targets, anchors 以及预测的 box 来创建训练模型时使用的数据形式, 在 YOLO 中, 我们的训练目标不是直接的 box 坐标, 而是对其进行相应的编码, 然后在进行训练, 编码的方式如下所示, 数据的标注信息为 $(b_x, b_y, b_w, b_h)$, 而我们的训练目标是 $(t_x, t_y, t_w, t_h)$, 这两组数据可以互相转换.

![](https://wx1.sinaimg.cn/large/d7b90c85ly1fyrcrzvpfjj216g0izn52.jpg)

```py
# ./utils/utils.py

def build_targets(
    pred_boxes, pred_conf, pred_cls, target, anchors, num_anchors, num_classes, grid_size, ignore_thres, img_dim
):
    # 参数:
    # pred_boxes: [1, 3, 13, 13, 4]
    # pred_conf: [1, 3, 13, 13]
    # pred_cls: [1, 3, 13, 13, 80]
    # target: [1, 50, 5]
    # anchors: [3, 2]
    # num_anchors: 3
    # num_classes: 80
    # grid_size: 13(特征图谱的尺寸)
    # ignore_thres: 0.5
    # img_dim: 图片尺寸
    nB = target.size(0) # batch_size
    nA = num_anchors # 3
    nC = num_classes # 80
    nG = grid_size # 特征图谱的尺寸(eg: 13)
    mask = torch.zeros(nB, nA, nG, nG) # eg: [1, 3, 13, 13], 代表每个特征图谱上的 anchors 下标(每个 location 都有 3 个 anchors)
    conf_mask = torch.ones(nB, nA, nG, nG) # eg: [1, 3, 13, 13] 代表每个 anchor 的置信度.
    tx = torch.zeros(nB, nA, nG, nG) # 申请占位空间, 存放每个 anchor 的中心坐标
    ty = torch.zeros(nB, nA, nG, nG) # 申请占位空间, 存放每个 anchor 的中心坐标
    tw = torch.zeros(nB, nA, nG, nG) # 申请占位空间, 存放每个 anchor 的宽
    th = torch.zeros(nB, nA, nG, nG) # 申请占位空间, 存放每个 anchor 的高
    tconf = torch.ByteTensor(nB, nA, nG, nG).fill_(0) # 占位空间, 存放置信度, eg: [1, 3, 13, 13]
    tcls = torch.ByteTensor(nB, nA, nG, nG, nC).fill_(0) # 占位空间, 存放分类预测值, eg:[1, 3, 13, 13, 80]

    nGT = 0
    nCorrect = 0
    for b in range(nB):
        for t in range(target.shape[1]):
            if target[b, t].sum() == 0: # b指定的batch中的某图片, t指定了图片中的某 box(按顺序)
                continue # 如果 box 的5个值(从标签到坐标)都为0, 那么就跳过当前的 box
            nGT += 1 # 每找到一个非零的 box, 则真实box的数量就加一

            # Convert to position relative to box
            # 由于我们在存储box的坐标时, 就是按照其相对于图片的宽和高的比例存储的
            # 因此, 当想要获取特征图谱上的对应 box 的坐标时, 直接令其与特征图谱的尺寸相乘即可.
            gx = target[b, t, 1] * nG
            gy = target[b, t, 2] * nG
            gw = target[b, t, 3] * nG
            gh = target[b, t, 4] * nG

            # Get grid box indices
            # 获取在特征图谱上的整数坐标
            gi = int(gx)
            gj = int(gy)

            # Get shape of gt box, 根据 box 的大小获取 shape: [1,4]
            gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)

            # Get shape of anchor box
            # 相似的方法得到anchor的shape: [3, 4] , 3 代表3个anchor
            anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((len(anchors), 2)), np.array(anchors)), 1))


            # 调用本文件的 bbox_iou 函数计算gt_box和anchors之间的交并比
            # 注意这里仅仅计算的是 shape 的交并比, 此处没有考虑位置关系.
            # gt_box 为 [1,4], anchors 为 [3, 4],
            # 最终返回的值为[3], 代表了 gt_box 与每个 anchor 的交并比大小
            anch_ious = bbox_iou(gt_box, anchor_shapes)

            # 将交并比大于阈值的部分设置conf_mask的对应位为0(ignore)
            conf_mask[b, anch_ious > ignore_thres, gj, gi] = 0

            # 找到匹配度最高的 anchor box, 返回下标: 0,1,2 中的一个
            best_n = np.argmax(anch_ious)

            # 获取相应的 ground truth box, unsqueeze用于扩充维度, 使[4]变成[1,4], 以便后面的计算
            gt_box = torch.FloatTensor(np.array([gx, gy, gw, gh])).unsqueeze(0)

            # 获取最佳的预测 box, pred_boxes的shape为: [1,3,13,13,4]
            # pred_box经过unsqueeze扩充后的shape为: [1,4]
            pred_box = pred_boxes[b, best_n, gj, gi].unsqueeze(0)

            # 设置 mask 和 conf_mask
            mask[b, best_n, gj, gi] = 1
            # 注意, 刚刚将所有大于阈值的 conf_mask对应为都设置为了0,
            # 然后这里将具有最大交并比的anchor设置为1, 如此确保一个真实框只对应一个 anchor.
            # 由于 conf_mask 的默认值为1, 因此, 剩余的box可看做是负样本
            conf_mask[b, best_n, gj, gi] = 1

            # 设置中心坐标, 该坐标是相对于 cell的左上角而言的, 所以是一个小于1的数
            tx[b, best_n, gj, gi] = gx - gi
            ty[b, best_n, gj, gi] = gy - gj
            # 设置宽和高, 注意, 这里会转化成训练时使用的宽高值
            tw[b, best_n, gj, gi] = math.log(gw / anchors[best_n][0] + 1e-16)
            th[b, best_n, gj, gi] = math.log(gh / anchors[best_n][1] + 1e-16)


            # 获取当前 box 的 标签
            target_label = int(target[b, t, 0])
            # tcls: [1,3,13,13,80]
            # 将当前true box对应的 anchor 的正确类别设置为1
            tcls[b, best_n, gj, gi, target_label] = 1
            # 将置信度设置为 1
            tconf[b, best_n, gj, gi] = 1

            # 调用 bbox_iou 函数计算 ground truth 和最佳匹配的预测box之间的 iou
            # 注意, 此时的 gt_box为 [gx,gy,gw,gh], 不是 [tx,ty,tw,th]
            # gt_box的shape为[1,4], pred_box为最佳匹配的预测 box, 其shape也为[1,4]
            iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False)
            # pred_cls的shape为[1,3,13,13,80], 获取最佳匹配anchor box的最大概率类别的下标
            pred_label = torch.argmax(pred_cls[b, best_n, gj, gi])
            # pred_conf的shape为[1,3,13,13], 获取最佳匹配anchor box的置信度
            score = pred_conf[b, best_n, gj, gi]
            if iou > 0.5 and pred_label == target_label and score > 0.5:
                nCorrect += 1 # 如果 iou 和 score 大于阈值, 并且标签预测正确, 则正确项增1

    # 将所有需要的信息都返回, 从这里可以看出, 每一个 YOLO 层都会执行一次预测.
    return nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls
```

# 参考文献

https://github.com/eriklindernoren/PyTorch-YOLOv3#train
