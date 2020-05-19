

# Task01 - 赛题理解

#### 任务：以计算机视觉为中心，要求预测街道字符编码，这是一个典型的字符识别问题

##### 赛题数据：公开数据集SVHN

数据展示：

![数据集样本展示](D:\software\jupyter\cv_practice\0_学习规划\IMG\赛事简介\数据集样本展示.png)

训练集包括3W张图片，验证集包括1W张图片，每张图片包含颜色图像和对应的编码类别以及具体位置

#### 数据标签

| Field  | Description |      |
| ------ | ----------- | ---- |
| top    | 左上角坐标X |      |
| height | 字符高度    |      |
| left   | 左上角最表Y |      |
| width  | 字符宽度    |      |
| label  | 字符编码    |      |

###### 字符坐标含义：

![字符坐标](D:\software\jupyter\cv_practice\0_学习规划\IMG\Task01\字符坐标.png)

在比赛数据（训练集、测试集和验证集）中，同一张图片中可能包括一个或者多个字符，因此在比赛数据的JSON标注中，会有两个字符的边框信息：      
 |原始图片|图片JSON标注|
 |----|-----|

![原始图片](D:\software\jupyter\cv_practice\0_学习规划\IMG\Task01\原始图片.png)





![原始图片标注](D:\software\jupyter\cv_practice\0_学习规划\IMG\Task01\原始图片标注.png)

#### 评测指标：

结果与实际图片的编码进行对比，以编码整体识别准确率为评价指标。任何一个字符错误都为错误，最终评测指标结果越大越好，具体计算公式如下：     
                                              Score=编码识别正确的数量/测试集图片数量  

#### 读取数据：

```python
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import os
%matplotlib inline
```

```python
train_json = json.load(open('./train.json'))
```

```python
#数据标注处理
def parse_json(d):
    arr = np.array([
        d['top'], d['height'], d['left'], d['width'], d['label']
    ])
    arr = arr.astype(int)
    return arr
```

```python
img = cv2.imread('./train/000000.png')
arr = parse_json(train_json['000000.png'])  #json文件包含每张图片数字的位置信息
arr.shape
```

#### subplot(numRows, numCols, plotNum)

图表的整个绘图区域被分成numRows行和numCols列，plotNum参数指定创建的Axes对象所在的区域，如何理解呢？如果numRows ＝ 3，numCols ＝  2，那整个绘制图表样式为3X2的图片区域，用坐标表示为（1，1），（1，2），（1，3），（2，1），（2，2），（2，3）。这时，当plotNum ＝ 1时，表示的坐标为（1，3），即第一行第一列的子图.

```python
plt.figure(figsize=(10, 10))
plt.subplot(1, arr.shape[1]+1, 1)  #画面分成一行三列，图像为第一个图
plt.imshow(img)
plt.xticks([]);plt.yticks([])

for idx in range(arr.shape[1]):
    plt.subplot(1, arr.shape[1]+1, idx+2)
    plt.imshow(img[arr[0, idx]:arr[0, idx]+arr[1, idx],arr[2, idx]:arr[2, idx]+arr[3, idx]])
    plt.title(arr[4, idx])
    plt.xticks([]);plt.yticks([])
```

![image-20200519110207368](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20200519110207368.png)



```python
def show_img(img):
    plt.figure(figsize=(10, 10))
    plt.subplot(1, arr.shape[1]+1, 1)  #画面分成一行三列，图像为第一个图
    plt.imshow(img)
    plt.xticks([]);plt.yticks([])

    for idx in range(arr.shape[1]):
        plt.subplot(1, arr.shape[1]+1, idx+2)
        plt.imshow(img[arr[0, idx]:arr[0, idx]+arr[1, idx],arr[2, idx]:arr[2, idx]+arr[3, idx]])
        plt.title(arr[4, idx])
        plt.xticks([]);plt.yticks([])
```

#### 将图像序列号保存

```python
def read_template(directory_name):
    img_list = os.listdir(r"./"+directory_name)
    img_list.sort(key=lambda x: int(x.split('.')[0]))  #按照数字顺序排列
    return img_list
```

#### 按顺序读取需要的图片并显示出来

```python
def read_img(directory_name, nums, img_list):    
    for num in range(nums):
        img_name = img_list[num]
        img = cv2.imread(directory_name + '/' + img_name)
        cv2.imshow('image',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
```

```python
directory_name = "train"
img_list = read_template(directory_name)
read_img = read_img(directory_name, 10, img_list)
```

