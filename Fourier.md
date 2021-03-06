# 傅里叶变换
### 频域
* 这个世界就像是一个超大的并行运算系统，时域表现的就是这个系统输出的变化的表象。

* 而频域更像是在一个瞬间对这个系统的组成元素进行分解，查看各个基本的部件。

* 频域从一开始描绘的就是一个永恒的世界，根据过去到现在的信息，预测整个未来。

### 傅里叶级数

* 任何周期函数，都可以看作是有限个不同振幅，不同相位正弦波的叠加。

![1](http://i2.muimg.com/524586/3f08a36d65f7497fs.jpg)

![2](http://i4.buimg.com/524586/9150a52df8f1248as.gif)

![3](http://i4.buimg.com/524586/7987e5634f226fbas.jpg)

***

### FFT的分辨率问题
![4](http://i4.buimg.com/524586/9f47ee23a7e8364bs.png)

上图中采用的是128点FFT，如果假设128次采用总用时1s，那么sin(x)的周期就是1s，
频率为1Hz，sin(4x)的周期就是0.25s，频率为4Hz。

128点FFT每一点的时间间隔为1/128s，所以采样频率为128Hz，根据奈奎斯特准则，
它的最大采样频率则为64Hz。

根据图中的对应关系，FFT的分辨率等于采样总时间的倒数。