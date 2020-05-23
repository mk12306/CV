数据读取与数据扩增

#### 数据读取

在计算机视觉中读取图像数据一般会用一下两种库：

#### 1.Pillow

pillow是python图像处理函式库PIL的一个分支

常规操作：

#导入Pillow库

```python
from PIL import Image
```

#读取图片

```python
img = Image.open("C:\\Users\\Administrator\\Pictures\\mk\\ironman.jpg")
img.show()
```

![image-20200524012803632](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20200524012803632.png)



#滤镜功能

```python
img2 = img.filter(ImageFilter.BLUR)
img2.show
```

![image-20200524013109923](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20200524013109923.png)

#### 2.opencv

OpenCV是一个跨平台的计算机视觉库，功能比Pillow更加强大

```python
import cv2
```

转换为灰度图

```python
img = cv2.imread("C:\\Users\\Administrator\\Pictures\\mk\\ironman.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("test_img", img)
cv2.waitKey(0)
```

![image-20200524013622398](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20200524013622398.png)

#### 数据扩增

当数据集不够时，需要通过数据扩增的方式增加数据集

数据扩增方法有很多：从颜色空间、尺度空间到样本空间，同时根据不同任务数据扩增都有相应的区别。        
对于图像分类，数据扩增一般不会改变标签；对于物体检测，数据扩增会改变物体坐标位置；对于图像分割，数据扩增会改变像素标签。

在常见的数据扩增方法中，一般会从图像颜色、尺寸、形态、空间和像素等角度进行变换。当然不同的数据扩增方法可以自由进行组合，得到更加丰富的数据扩增方法。         

以torchvision为例，常见的数据扩增方法包括：

- transforms.CenterCrop      对图片中心进行裁剪      
- transforms.ColorJitter      对图像颜色的对比度、饱和度和零度进行变换      
- transforms.FiveCrop     对图像四个角和中心进行裁剪得到五分图像     
- transforms.Grayscale      对图像进行灰度变换    
- transforms.Pad        使用固定值进行像素填充     
- transforms.RandomAffine      随机仿射变换    
- transforms.RandomCrop      随机区域裁剪     
- transforms.RandomHorizontalFlip      随机水平翻转     
- transforms.RandomRotation     随机旋转     
- transforms.RandomVerticalFlip     随机垂直翻转  
- ![数据扩增示例](D:\software\jupyter\cv_practice\0_学习规划\IMG\Task02\数据扩增示例.png)