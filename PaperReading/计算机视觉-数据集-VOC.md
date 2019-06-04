---
title: PASCAL VOC数据集
sitemap: true
categories: 计算机视觉
date: 2018-10-11 18:29:44
tags:
- 计算机视觉
- 目标检测
---

# 评价标准

# COCO数据集介绍

COCO数据集具有5种标签类型, 分别为: 目标检测, 关键点检测, 物体分割, 多边形分割 以及 图像描述. 这些标注数据使用`JSON`格式存储. 所有文件除了标签项以外都共享同样的数据结构, 如下所示:

<div style="width: 600px; margin: auto">![](https://wx1.sinaimg.cn/mw1024/d7b90c85ly1fwcngpaj5nj20s60lv405.jpg)

标签结构各有不同, 如下所示:

## Object Detection

每一个目标实例的标签都具有一系列条目, 包括该目标的类别id以及分割掩膜(segmentation mask). 分割掩膜的格式取决于实例表示的是一个单一的目标(iscrowd=0, 使用polygons标注)还是一群目标(iscrowd=1, 使用RLE标注). 注意, 一个单一的物体(iscrowd=0)在被遮挡的情况下, 可能会需要多个多边形来表示. Crowd标签用来标注一大群目标(如一群人). 另外, 每一个物体都会提供一个闭合的bounding box( box的坐标系从图左上角开始,0-indexed). 最后, 标注结构的类别条目存储着从cat id 到类别的映射以及超类别的名字.

## 关键点检测

## Stuff Segmentation
Stuff Segmentation的格式和object detection的格式几乎一模一样, 但是stuff segmentation无需`iscrowd`条目, 因为该条默认置为0. 为了方便访问, coco提供了json和png两种标注格式. 在 json 格式中, **每一个出现在图片中的类别都会单独用一个RLE标签条目编码**(也就是说同类的会被放到同一个RLE编码里面). category_id 则代表当前的物体类别的id.

## Panoptic Segmentation


<div style="width: 600px; margin: auto">![](https://wx4.sinaimg.cn/mw1024/d7b90c85ly1fwcnkqsfkxj20rs0au0tk.jpg)

## 数据集信息

## 标注格式


# COCO-API 使用方法及源码解析

最常用的是 `pycocotools/coco.py` 文件中的`COCO`类, 其内部实现如下:
```py
# pycocotools/coco.py

# 首先, 是一大串的注释
# COCO API提供了一系列的辅助函数来帮助载入,解析以及可视化COCO数据集的annotations
# 该文件定义了如下API 函数:
# COCO        - COCO api 类, 用于载入coco的annotation 文件, 同时负责准备对应数据结构来存储
# decodeMask  - 通过rle编码规范, 来对二值mask M进行解码
# encodeMask  - 使用rle编码规范来对二值mak M进行编码
# getAnnIds   - 获得满足给定过滤条件的ann ids(标注)
# getCatIds   - 获得满足给定过滤条件的cat ids(类别)
# getImgIds   - 获得满足给定过滤条件的img ids(图片)
# loadAnns    - 根据指定的ids加载anns
# loadCats    - 根据指定的ids加载cats
# loadImgs    - 根据指定的ids加载imgs
# annToMask   - 将annotation里面的segmentation信息转换成二值mask
# showAnns    - 可视化指定的annotations
# loadRes     - 加载算法结果同时创建API以便访问它们
# download    - 从Mscoco.org.server上下载COCO数据集


# 接下来, 具体看一下COCO类的实现
class COCO:
    def __init__(self, annotation_file=None):
        # 构造函数
        # 参数annotation_file: 指定了annotation文件的位置

        # dataset, anns, cats, imgs均为字典类型数据
        self.dataset, self.anns, self.cats, self.imgs = dict(), dict(), dict(), dict()

        # imgToAnns, catToImgs均为defaultdict数据类型(带有默认值的字典)
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)

        if not annotation_file == None:
            #...

            # 以只读方式加载json标注文件
            dataset = json.load(open(annotation_file, 'r'))
            assert type(dataset)==dict # 确保读出来的是字典类型
            #...

            # 正式将读取的字典数据赋给该类的成员变量
            self.dataset = dataset

            # 创建索引
            self.createIndex()

    def createIndex(self):
        # 创建索引

        # 三个都是字典数据类型
        anns,cats,imgs={},{},{}
        # 两个defaultdict数据类型
        imgToAnns, catToImgs = defaultdict(list), defaultdict(list)
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
```
