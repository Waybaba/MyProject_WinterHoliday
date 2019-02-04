"""
BUILD NETWARK TEMPLATE 网络构筑模板
1: 初始化import
2: Create data
3: Create graph
4: 创建输入容器，placeholder
5: 添加layer（包括了变量和运算关系，返回输出值）
6: 定义loss-损失量化表征
7: 定义Optimizer-优化方式
8: 然后创建step，这个确定了优化方向minimize，是以后训练要调用的参数，另外这句话把op和loss链接了
9: 初始化变量等
10: 输入数据（这里用了循环，实际上只有一句加粗的是核心的）好像会自己判断用到的参数
11: Write logs in specific file
    tensorboard --logdir /Users/Waybaba/PycharmProjects/Machine_learning/MyProject/Tensorflow_Demo/tensorflow_cookbook-master/logs
    http://WaybabadeMacBook.local:6006
"""
# 1: 初始化import
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import for_import.ntu_date_preprocess_2 as ntu
import matplotlib.pyplot as plt
import for_import.ntu_skeleton_visualization as ntusee
import tensorflow as tf

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#把创建变量，写graph关系的步骤都封装在一个函数里面
def add_my_dense(inputs, in_size, out_size, activation_function=None,dropout=False,dropout_rate=0.5):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
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
def reshape(h_pool2):
    tf.reshape(h_pool2, [-1, 7 * 7 * 64])



# 2: Create data
input_train,lable_train,input_test,lable_test =ntu.load_date("4_actions")

input_train = input_train.reshape((-1,50*75))#数据整形，然后输入
input_test = input_test.reshape((-1,50*75))

# ntusee.show_gif(input_train[0])
lable_train = ntu.change_into_muti_dim(lable_train)
lable_test = ntu.change_into_muti_dim(lable_test)
lable_train = lable_train.astype(np.float32)#不知道为什么不转成32会导致交叉墒出错
lable_test = lable_test.astype(np.float32)

print('input_train shape:', input_train.shape)
print('input_test shape:', input_test.shape)

# 3: Create graph
sess = tf.Session()

# 4: 创建输入点，一个数据，一个标签/或者说是拟合值
with tf.name_scope('INPUT'):
    xs = tf.placeholder(tf.float32, [None, 3750])
    ys = tf.placeholder(tf.float32, [None, 4])

# 5: 添加layer（包括了变量和运算关系，返回输出值）
with tf.name_scope('NETWOARS'):
    l1 = add_my_dense(inputs=xs,in_size=3750, out_size=10, activation_function=tf.nn.relu,dropout=True)
    prediction = add_my_dense(inputs=l1,in_size=10,out_size=4, activation_function=tf.nn.softmax)

# 6: 定义loss-损失量化表征
# def get_accuracy():


with tf.name_scope('LOSS'):
    # loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))  # mean是平均
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ys, logits=prediction))
    # loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction,labels=ys))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(ys,1), tf.argmax(prediction,1)),tf.float32))

# 7: 定义Optimizer-优化方式
with tf.name_scope('optimizer'):
    my_opt = tf.train.GradientDescentOptimizer(0.1)  # 这个确定了优化方式
# 8: 然后创建step，这个确定了优化方向minimize，是以后训练要调用的参数，另外这句话把op和loss链接了
with tf.name_scope('TRAIN'):
    train_step = my_opt.minimize(loss)

# 9: 初始化变量等
init = tf.global_variables_initializer()
sess.run(init)

# 10: 输入数据（这里用了循环，实际上只有一句加粗的是核心的）
for i in range(2000):
    # training
    sess.run(train_step, feed_dict={xs: input_train, ys: lable_train})
    if i % 50 == 0:
        # to see the step improvement
        print("Loss: ",sess.run(loss, feed_dict={xs: input_test, ys: lable_test}),end="   ")
        print("Accuracy: ",sess.run(accuracy, feed_dict={xs: input_test, ys: lable_test}))

# 打印一些输出
if 1 :
    val_test = sess.run(prediction,feed_dict={xs:input_test[0:100]})
    right_count = 0.0
    for index in range(len(val_test)):
        argmax_val = np.argmax(val_test[index])
        argmax_lable = np.argmax(lable_test[index])
        if argmax_lable == argmax_val :
            right_count+=1
            print("^ O ^ ",end="")
        else:
            print("* x * ",end="")
        print("Prediction: ",argmax_val,end="  ")
        print("Real_lable: ",argmax_lable)
    print("Right Rate : ",right_count/float(len(val_test)))

# 11: Write logs in specific file
log_writing_path = '/Users/Waybaba/PycharmProjects/Machine_learning/MyProject/Tensorflow_Demo/tensorflow_cookbook-master/logs/'
my_writer = tf.summary.FileWriter(log_writing_path, sess.graph)


