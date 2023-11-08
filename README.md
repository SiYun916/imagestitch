## 图像拼接

### 功能

实现了对两张或多张图像的拼接算法，不存在由于单应变换造成的缺失问题。高分辨率用的是ORB算法，低分辨率图像用的是SIFT算法，想换成相应的算法可在detectAndDescribe函数中进行修改。测试数据方面（存放在sourcedata里面），针对实际分辨率为640\*512的红外图像及可见光窗口图像，各准备了5组测试数据进行测试；实际分辨率为2048\*1536和4096\*3072的可见光窗口图像分别准备了4组和3组数据进行测试。

### 参数及使用

这个参数是source最初版本的参数，另外两个的参数具体看源码。如果真能写成，再把参数更新了。

| 参数          | 说明                          |
|:-----------:|:---------------------------:|
| --list      | 待拼接的图片路径，至少有两张图片            |
| --transpath | 中间结果的保存路径，默认是tansformed.png |
| --savepath  | 最终结果保存路径，默认是processed.png   |

示例

```powershell
.\main.exe  --list .\image1.png .\image2.png --transpath tmp.png --savepath result.png
```

这里是用pyinstaller生成了exe可执行文件，如果用源码执行的话参数同上。

### 环境

source最初版本：
1. python == 3.9
2. opencv-python == 4.8.0.76
3. opencv-contrib-python == 4.8.0.76
4. numpy == 1.26.0

refactorCode和reSampleCode：
1. python == 3.7
2. opencv-python == 3.4.2.17
3. opencv-contrib-python ==3.4.2.17
4. numpy == 1.21.6

这里提醒一下，SURF算法在3.4.2.17版本上可用，但在更高的opencv版本上就用不了，说是让自己去编译。SIFT算法写法在opencv3和4上的写法不一样，有报错对着报错改一下就行。而且一定要install contrib那个包，否则会提醒找不到方法。

### 存在的问题

为了解决单应变换造成的缺失问题，我先将第一张图进行了单应变换并且进行了平移。然后将变换后的图再去和第二张图进行一次拼接。由于用了两次特征点寻找和匹配的算法，所以在高分辨率图像拼接的时候速度会慢，并且多张图像拼接的时候速度也会变慢。速度和CPU的性能也有很大关系，CPU越好速度也会越快。低分辨率图像拼接速度还行。

两张图像拼接基本没有问题，就是没有进行图像融合，边界部分有一条白线。多张图像拼接除了速度问题外，还有一个拼接顺序问题没完全解决，由于我这个是逐步拼接的，多张图片拼接的时候，如果有两张图片没有交集，无法进行拼接，需要找出一个顺序来。我写了一个判断过程，但没完全写，因为多张高分辨率图像特征点太多，用我这个代码会卡死。所以高分辨率多张拼接就没判断顺序，低分辨率的图像可以是可以，但速度也会降低，而且测试图像基本都是按顺序有交集的，顺序基本也没变，所以基本没啥用。

### 代码重构

对代码进行了重构，删减了一些东西，同时把注释也加了上去，代码放在refactor中，具体参数见代码。还未解决的问题也写了一个文档放在其中。

### 重构后简化

今天思考一天（感觉自己智商不高），发现没必要加第二次的特征点提取和匹配来确定两个的位置。直接用correct\_H方法返回的单应变换后坐标x,y最小值对B进行平移即可（经验证计算出的最小值，就是需要平移的像素点值），具体见里面的pdf。

简化后，速度变快了不少。默认用的特征点提取算法是SURF，速度肯定是比SIFT快的，但是多张图像拼接时，可以肉眼看出不如SIFT精确。所以用的时候看情况选算法了，如果是4K的图，建议用ORB，ORB速度和精度都还行，不然用SIFT精度是高但速度是真慢。

代码放在reSampleCode中，还有一些没有解决的问题，比如先对图像进行去畸变和亮度处理等。还是先解决边缘融合的问题吧，之后更新打算在这个里面去添加新参数和新功能。

10月30日更新，新增了自适应颜色直方图均衡、gamma变换、畸变矫正（没完全改完，需要数据测试）、直方图均衡（有代码，但没用，实际效果不如自适应的好）功能，去除了拼接的缝隙（实际就是拷贝warp图的3列or行像素到最终结果的对应缝隙那里）根据代码和1030那张图片理解一下。更新的参数还是一样具体看代码，这里提醒一下畸变矩阵顺序是k1、k2、p1、p2、k3，另外一个相机内参矩阵是00、11、02、12位置的值。

下一步更新计划（不知道什么时候能搞好）：有遮光罩拍摄出的图片进行亮度调整（试一试把论文里面的算法加上去）；畸变图像边缘损失像素去除；图像融合（不是像我这样的裁剪粘贴）；拼接顺序（可能先解决这个比较简单）。

11月7日更新，应要求新增一个功能，在图片左上角添加一行字显示拼接好的图片。11月8日更新，修复左上角字会超出边界的bug，改为自适应五分之三图片的宽度。


