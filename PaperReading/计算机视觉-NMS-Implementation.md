---
title: NMS 算法源码实现
sitemap: true
categories: 计算机视觉
date: 2018-10-24 14:22:43
tags:
- 计算机视觉
- 目标检测
- 源码实现
- Python
- Cpp
---

【链接】非极大值抑制算法(NMS)及python实现
https://blog.csdn.net/Blateyang/article/details/79113030

【链接】NMS-非极大值抑制-Python实现
https://blog.csdn.net/u011436429/article/details/80107042

<span id = "简述 NMS 的原理">
# 简述 NMS 的原理

非极大值抑制(Non-Maximum Suppression, NMS), 顾名思义就是抑制那些不是极大值的元素, 可以理解为局部最大值搜索. 对于目标检测来说, 非极大值抑制的含义就是对于重叠度较高的一部分同类候选框来说, 去掉那些置信度较低的框, 只保留置信度最大的那一个进行后面的流程, 这里的重叠度高低与否是通过 NMS 阈值来判断的.

计算两个边框 IoU 的公式如下所示:
$$x1 = \max (box1_{x1}, box2_{x1}), y1 = \max (box1_{y1}, box2_{y1})$$
$$x2 = \min (box1_{x2}, box2_{x2}), y2 = \min (box1_{y2}, box2_{y2})$$
$$intersection = (x2 - x1 + 1) \times (y2 - y1 + 1)$$
$$IoU = \frac{intersection}{area1+area2-intersection}$$

<span id = "NMS 算法源码实现">
# NMS 算法源码实现

**算法逻辑:**
输入: $n$ 行 $4$ 列的候选框数组, 以及对应的 $n$ 行 $1$ 列的置信度数组.
输出: $m$ 行 $4$ 列的候选框数组, 以及对应的 $m$ 行 $1$ 列的置信度数组, $m$ 对应的是去重后的候选框数量
算法流程:
1. 计算 $n$ 个候选框的面积大小
2. 对置信度进行排序, 获取排序后的下标序号, 即采用`argsort`
3. 将当前置信度最大的框加入返回值列表中
4. 获取当前置信度最大的候选框与其他任意候选框的相交面积
5. 利用相交的面积和两个框自身的面积计算框的交并比, 将交并比大于阈值的框删除.
6. 对剩余的框重复以上过程

**Python 实现:**
```py
import cv2
import numpy as np

def nms(bounding_boxes, confidence_score, threshold):
    if len(bounding_boxes) == 0:
        return [], []
    bboxes = np.array(bounding_boxes)
    score = np.array(confidence_score)

    # 计算 n 个候选框的面积大小
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    areas =(x2 - x1 + 1) * (y2 - y1 + 1)

    # 对置信度进行排序, 获取排序后的下标序号, argsort 默认从小到大排序
    order = np.argsort(score)

    picked_boxes = [] # 返回值
    picked_score = [] # 返回值
    while order.size > 0:
        # 将当前置信度最大的框加入返回值列表中
        index = order[-1]
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])

        # 获取当前置信度最大的候选框与其他任意候选框的相交面积
        x11 = np.maximum(x1[index], x1[order[:-1]])
        y11 = np.maximum(y1[index], y1[order[:-1]])
        x22 = np.minimum(x2[index], x2[order[:-1]])
        y22 = np.minimum(y2[index], y2[order[:-1]])
        w = np.maximum(0.0, x22 - x11 + 1)
        h = np.maximum(0.0, y22 - y11 + 1)
        intersection = w * h

        # 利用相交的面积和两个框自身的面积计算框的交并比, 将交并比大于阈值的框删除
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)
        left = np.where(ratio < threshold)
        order = order[left]

    return picked_boxes, picked_score
```

**C++ 实现**

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

struct Bbox {
    int x1;
    int y1;
    int x2;
    int y2;
    float score;
    Bbox(int x1_, int y1_, int x2_, int y2_, float s):
	x1(x1_), y1(y1_), x2(x2_), y2(y2_), score(s) {};
};

float iou(Bbox box1, Bbox box2) {
    float area1 = (box1.x2 - box1.x1 + 1) * (box1.y2 - box1.y1 + 1);
    float area2 = (box2.x2 - box2.x1 + 1) * (box2.y2 - box2.y1 + 1);
    int x11 = std::max(box1.x1, box2.x1);
    int y11 = std::max(box1.y1, box2.y1);
    int x22 = std::min(box1.x2, box2.x2);
    int y22 = std::min(box1.y2, box2.y2);
    float intersection = (x22 - x11 + 1) * (y22 - y11 + 1);
    return intersection / (area1 + area2 - intersection);
}

std::vector<Bbox> nms(std::vector<Bbox> &vecBbox, float threshold) {
    auto cmpScore = [](Bbox box1, Bbox box2) {
	return box1.score < box2.score; // 升序排列, 令score最大的box在vector末端
    };
    std::sort(vecBbox.begin(), vecBbox.end(), cmpScore);

    std::vector<Bbox> pickedBbox;
    while (vecBbox.size() > 0) {
        pickedBbox.emplace_back(vecBbox.back());
        vecBbox.pop_back();
        for (size_t i = 0; i < vecBbox.size(); i++) {
            if (iou(pickedBbox.back(), vecBbox[i]) >= threshold) {
                vecBbox.erase(vecBbox.begin() + i);
            }
        }
    }
    return pickedBbox;
}
```

**CUDA 实现:**

<span id = "简述 Soft-NMS 的原理">
# 简述 Soft-NMS 的原理

在 Soft-NMS 中, 对于那些重叠度大于一定阈值的 box, 我们并不将其删除, 而仅仅只是根据重叠程度来降低那些 box 的 socre, 这样一来, 这些 box 仍旧处于 box 列表中, 只是 socre 的值变低了. 具体来说, 如果 box 的重叠程度高, 那么 score 的值就会变得很低, 如果重叠程度小, 那么 box 的 score 值就只会降低一点, Soft-NMS 算法伪代码如下图所示:

<span id = "Soft-NMS 算法源码实现">
# Soft-NMS 算法源码实现

**算法逻辑:**
输入:
- bboxes: 坐标矩阵, 每个边框表示为 [x1, y1, x2, y2]
- scores: 每个 box 对应的分数, 在 Soft-NMS 中, scores 会发生变化(**对外部变量也有影响**)
- iou_thresh:   交并比的最低阈值
- sigma2: 使用 gaussian 函数的方差, sigma2 代表 $\sigma^2$
- score_thresh: 最终分数的最低阈值
- method: 使用的惩罚方法, 1 代表线性惩罚, 2 代表高斯惩罚, 其他情况代表默认的 NMS

返回值: 最终留下的 boxes 的 index, 同时, scores 值也已经被改变.
算法流程:
1. 在 bboxes 之后添加对于的下标[0, 1, 2...], 最终 bboxes 的 shape 为 [n, 5], 前四个为坐标, 后一个为下标
2. 计算每个 box 自身的面积
3. **对于每一个下标 $i$**, 找出 i 后面的最大 score 及其下标, 如果当前 i 的得分小于后面的最大 score, 则与之交换, 确保 i 上的 score 最大.
4. 计算 IoU
5. 根据用户选定的方法更新 scores 的值
6. 以上过程循环 $N$ 次后($N$ 为总边框的数量), 将最终得分大于最低阈值的下标返回, 根据下标获取最终存留的 Boxes, **注意, 此时, 外部 scores 的值已经完成更新, 无需借助下标来获取.**

```py
import numpy as np

def soft_nms(bboxes, scores, iou_thresh=0.3, sigma2=0.5, score_thresh=0.001, method=2):
    # 在 bboxes 之后添加对于的下标[0, 1, 2...], 最终 bboxes 的 shape 为 [n, 5], 前四个为坐标, 后一个为下标
    N = bboxes.shape[0] # 总的 box 的数量
    indexes = np.array([np.arange(N)])  # 下标: 0, 1, 2, ..., n-1
    bboxes = np.concatenate((bboxes, indexes.T), axis=1) # concatenate 之后, bboxes 的操作不会对外部变量产生影响

    # 计算每个 box 的面积
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    for i in range(N):
        # 找出 i 后面的最大 score 及其下标
        pos = i + 1
        if i != N-1:
            maxscore = np.max(scores[pos:], axis=0)
            maxpos = np.argmax(scores[pos:], axis=0)
        else:
            maxscore = scores[-1]
            maxpos = 0

        # 如果当前 i 的得分小于后面的最大 score, 则与之交换, 确保 i 上的 score 最大
        if scores[i] < maxscore:
            bboxes[[i, maxpos + i + 1]] = bboxes[[maxpos + i + 1, i]]
            scores[[i, maxpos + i + 1]] = scores[[maxpos + i + 1, i]]
            areas[[i, maxpos + i + 1]] = areas[[maxpos + i + 1, i]]

        # IoU calculate
        xx1 = np.maximum(bboxes[i, 0], bboxes[pos:, 0])
        yy1 = np.maximum(bboxes[i, 1], bboxes[pos:, 1])
        xx2 = np.minimum(bboxes[i, 2], bboxes[pos:, 2])
        yy2 = np.minimum(bboxes[i, 3], bboxes[pos:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        intersection = w * h
        iou = intersection / (areas[i] + areas[pos:] - intersection)

        # Three methods: 1.linear 2.gaussian 3.original NMS
        if method == 1:  # linear
            weight = np.ones(iou.shape)
            weight[iou > iou_thresh] = weight[iou > iou_thresh] - iou[iou > iou_thresh]
        elif method == 2:  # gaussian
            weight = np.exp(-(iou * iou) / sigma2)
        else:  # original NMS
            weight = np.ones(iou.shape)
            weight[iou > iou_thresh] = 0

        scores[pos:] = weight * scores[pos:]

    # select the boxes and keep the corresponding indexes
    inds = bboxes[:, 4][scores > score_thresh]
    keep = inds.astype(int)
    return keep

# boxes and scores
boxes = np.array([[200, 200, 400, 400], [220, 220, 420, 420],
                  [240, 200, 440, 400], [200, 240, 400, 440],
                  [1, 1, 2, 2]], dtype=np.float32)
boxscores = np.array([0.9, 0.8, 0.7, 0.6, 0.5], dtype=np.float32)
index = soft_nms(boxes, boxscores, method=2)
print(index) # 按照 scores 的排序指明了对应的 box 的下标
print(boxes[index])
print(boxscores) # 注意, scores 不需要用 index 获取, scores 已经是更新过的排序 scores
```

[Soft-NMS 论文解读](../计算机视觉-SoftNMS-ICCV2017)

<span id = "介绍一下其他的 NMS 算法">
# 介绍一下其他的 NMS 算法

[Face++ 论文: SofterNMS-Arxiv2018](../计算机视觉-SofterNMS-Arxiv2018)


# 完整实现

**Python 实现:**
```py
import cv2
import numpy as np

def nms(bounding_boxes, confidence_score, threshold):
    if len(bounding_boxes) == 0:
        return [], []

    bboxes = np.array(bounding_boxes)

    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]

    score = np.array(confidence_score)

    picked_boxes = []
    picked_score = []

    areas =(x2 - x1 + 1) * (y2 - y1 + 1)

    order = np.argsort(score)

    while order.size > 0:
        index = order[-1]

        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])

        x11 = np.maximum(x1[index], x1[order[:-1]])
        y11 = np.maximum(y1[index], y1[order[:-1]])
        x22 = np.minimum(x2[index], x2[order[:-1]])
        y22 = np.minimum(y2[index], y2[order[:-1]])

        w = np.maximum(0.0, x22 - x11 + 1)
        h = np.maximum(0.0, y22 - y11 + 1)
        intersection = w * h

        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < threshold)
        order = order[left]

    return picked_boxes, picked_score

def main():
    img_path = "/home/zerozone/Pictures/testimg/face.jpg"
    bounding_boxes = [(187, 82, 337, 317), (150, 67, 305, 282),
            (246, 121, 368, 304)]
    confidence_score = [0.9, 0.75, 0.8]
    image = cv2.imread(img_path)
    orig = image.copy()

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    threshold = 0.5

    # Draw bboxes and score
    for (x1, y1, x2, y2), confidence in zip(bounding_boxes, confidence_score):
        (w, h), baseline = cv2.getTextSize(
                str(confidence), font, font_scale, thickness)
        cv2.rectangle(orig, (x1, y1 - (2 * baseline + 5)),
                (x1 + w, y1), (0, 255, 255), -1)
        cv2.rectangle(orig, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(orig, str(confidence), (x1, y1),
                font, font_scale, (0, 0, 0, thickness))

    # Draw bboxes and score after NMS
    picked_boxes, picked_score = nms(bounding_boxes, confidence_score, threshold)

    for (x1, y1, x2, y2), confidence in zip(picked_boxes, picked_score):
        (w, h), baseline = cv2.getTextSize(
                str(confidence), font, font_scale, thickness)
        cv2.rectangle(image, (x1, y1 - (2 * baseline + 5)),
                (x1 + w, y1), (0, 255, 255), -1)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(image, str(confidence), (x1, y1),
                font, font_scale, (0, 0, 0, thickness))
    cv2.imshow("origin", orig)
    cv2.imshow("NMS", image)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
```

**C++ 实现**

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

struct Bbox {
    int x1;
    int y1;
    int x2;
    int y2;
    float score;
    Bbox(int x1_, int y1_, int x2_, int y2_, float s):
	x1(x1_), y1(y1_), x2(x2_), y2(y2_), score(s) {};
};

float iou(Bbox box1, Bbox box2) {
    float area1 = (box1.x2 - box1.x1 + 1) * (box1.y2 - box1.y1 + 1);
    float area2 = (box2.x2 - box2.x1 + 1) * (box2.y2 - box2.y1 + 1);

    int x11 = std::max(box1.x1, box2.x1);
    int y11 = std::max(box1.y1, box2.y1);
    int x22 = std::min(box1.x2, box2.x2);
    int y22 = std::min(box1.y2, box2.y2);
    float intersection = (x22 - x11 + 1) * (y22 - y11 + 1);

    return intersection / (area1 + area2 - intersection);
}

std::vector<Bbox> nms(std::vector<Bbox> &vecBbox, float threshold) {
    auto cmpScore = [](Bbox box1, Bbox box2) {
	return box1.score < box2.score; // 升序排列, 令score最大的box在vector末端
    };
    std::sort(vecBbox.begin(), vecBbox.end(), cmpScore);

    std::vector<Bbox> pickedBbox;
    while (vecBbox.size() > 0) {
        pickedBbox.emplace_back(vecBbox.back());
        vecBbox.pop_back();
        for (size_t i = 0; i < vecBbox.size(); i++) {
            if (iou(pickedBbox.back(), vecBbox[i]) >= threshold) {
                vecBbox.erase(vecBbox.begin() + i);
            }
        }
    }
    return pickedBbox;
}

int main() {
    std::vector<Bbox> vecBbox;
    vecBbox.emplace_back(Bbox(187, 82, 337, 317, 0.9));
    vecBbox.emplace_back(Bbox(150, 67, 305, 282, 0.75));
    vecBbox.emplace_back(Bbox(246, 121, 368, 304, 0.8));

    auto pickedBbox = nms(vecBbox, 0.5);

    for (auto box : pickedBbox) {
	std::cout << box.x1 << ", " <<
		box.y1 << ", " <<
		box.x2 << ", " <<
		box.y2 << ", " <<
		box.score << std::endl;
    }
    return 0;
}
```
