# View more python tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784])/255.   # 28x28
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 28, 28, 1])
# print(x_image.shape)  # [n_samples, 28,28,1]





def add_my_dense(inputs, in_size, out_size, activation_function=None,dropout=False,dropout_rate=0.5):
    Weights = weight_variable([in_size, out_size])#下面那个初始化会导致没有办法收敛，但是不知道为什么，暂时先保留把
    biases = bias_variable([1, out_size])
    # Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    # biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    # dropout选择
    if dropout :
        Wx_plus_b = tf.nn.dropout(Wx_plus_b, dropout_rate)
    # activation选择
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

def add_my_cnn(input, filter_shape=[3,3,1,1], strides=[1,1,1,1], padding="SAME"):
    """
    :param input: 输入数据
    :param filter:[height,width,in_channels, out_channels]，分别表示卷积核的高、宽、深度（与输入的in_channels应相同）、输出 feature map的个数（即卷积核的个数）。
    :param strides:表示步长：一个长度为4的一维列表，每个元素跟data_format互相对应，表示在data_format每一维上的移动步长。当输入的默认格式为：“NHWC”，则 strides = [batch , in_height , in_width, in_channels]。其中 batch 和 in_channels 要求一定为1，即只能在一个样本的一个通道上的特征图上进行移动，in_height , in_width表示卷积核在特征图的高度和宽度上移动的布长，即 strideheight和 stridewidth
    :param padding:"SAME"（填充边界）或者"VALID"（不填充边界）
    :return:建好的层的输出
    """
    W_conv = weight_variable(filter_shape)  # patch 5x5, in size 1, out size 32
    b_conv = bias_variable([filter_shape[3]])
    tem = tf.nn.relu(tf.nn.conv2d(input=input,filter=W_conv,strides=strides,padding=padding)+b_conv)
    return tem
def add_my_max_pool(input,ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME'):
    # stride [1, x_movement, y_movement, 1]

    return tf.nn.max_pool(input, ksize=ksize, strides=strides, padding=padding)

## conv1 layer ##
h_conv1 = add_my_cnn(input=x_image,filter_shape=[5,5,1,32],strides=[1,1,1,1])
h_pool1 = add_my_max_pool(h_conv1)                                     # output size 14x14x32

## conv2 layer ##
h_conv2 =  add_my_cnn(input=h_pool1,filter_shape=[5,5,32,64],strides=[1,1,1,1]) # output size 14x14x64
h_pool2 = add_my_max_pool(h_conv2)                                       # output size 7x7x64

## flatten ##
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

## fc1 layer ##
h_fc1_drop = add_my_dense(inputs=h_pool2_flat,in_size=7*7*64,out_size=1024,dropout=True,dropout_rate=keep_prob,activation_function=tf.nn.relu)

## fc2 layer ##
final_out = add_my_dense(inputs=h_fc1_drop,in_size=1024,out_size=10,dropout=False)
prediction = tf.nn.softmax(final_out)











# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))       # loss
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()
# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12

init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 50 == 0:
        print(compute_accuracy(
            mnist.test.images[:1000], mnist.test.labels[:1000]))

