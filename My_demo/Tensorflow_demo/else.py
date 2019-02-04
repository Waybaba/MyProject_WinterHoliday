"""
# 变量储存
save_path = saver.save(sess, "my_net/save_net.ckpt")
print("Save to path: ", save_path)

#变量提取
先要建立零时的W 和 b容器. 找到文件目录, 并用saver.restore()我们放在这个目录的变量.
saver = tf.train.Saver()
saver.restore(sess, "my_net/save_net.ckpt")
print("weights:", sess.run(W))
print("biases:", sess.run(b))

# 卷积层
步长（stride）
边界扩充(pad)可以为0
扫描生成的下一层神经元矩阵 称为 一个feature map (特征映射图)
激励层主要对卷积层的输出进行一个非线性映射，因为卷积层的计算还是一种线性计算。使用的激励函数一般为ReLu函数：

# 池化层
池化层也有一个“池化视野（filter）”来对feature map矩阵进行扫描，对“池化视野”中的矩阵值进行计算，一般有两种计算方式：
Max pooling：取“池化视野”矩阵中的最大值
Average pooling：取“池化视野”矩阵中的平均值

# 归一化层
Batch Normalization（批量归一化）实现了在神经网络层的中间进行预处理的操作，即在上一层的输入归一化处理后再进入网络的下一层，这样可有效地防止“梯度弥散”，加速网络训练。
切分层：融合层

# 卷积层：
tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None)
    filter: 4维格式的数据：shape表示为：[height,width,in_channels, out_channels]，分别表示卷积核的高、宽、深度（与输入的in_channels应相同）、输出 feature map的个数（即卷积核的个数）。
    strite: 表示步长：一个长度为4的一维列表，每个元素跟data_format互相对应，表示在data_format每一维上的移动步长。当输入的默认格式为：“NHWC”，则 strides = [batch , in_height , in_width, in_channels]。其中 batch 和 in_channels 要求一定为1，即只能在一个样本的一个通道上的特征图上进行移动，in_height , in_width表示卷积核在特征图的高度和宽度上移动的布长，即 strideheight和 stridewidth
    padding：表示填充方式：“SAME”表示采用填充的方式，简单地理解为以0填充边缘，当stride为1时，输入和输出的维度相同；“VALID”表示采用不填充的方式，多余地进行丢弃。

# 池化层：
tf.nn.max_pool( value, ksize,strides,padding,data_format=’NHWC’,name=None)
    大多参数和conv相同
    value：表示池化的输入
    ksize：表示池化窗口的大小：一个长度为4的一维列表，一般为[1, height, width, 1]，因不想在batch和channels上做池化，则将其值设为1。


"""