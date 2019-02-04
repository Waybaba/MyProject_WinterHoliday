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

#把创建变量，写graph关系的步骤都封装在一个函数里面
def add_my_layer(inputs, in_size, out_size, activation_function=None,dropout=False,dropout_rate=0.5):
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

# 1: 初始化import
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 2: Create data
x_data = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise

# 3: Create graph
sess = tf.Session()

# 4: 创建输入点，一个数据，一个标签/或者说是拟合值
with tf.name_scope('INPUT'):
    xs = tf.placeholder(tf.float32, [None, 1])
    ys = tf.placeholder(tf.float32, [None, 1])

# 5: 添加layer（包括了变量和运算关系，返回输出值）
with tf.name_scope('NETWOARS'):
    l1 = add_my_layer(inputs=xs,in_size=1, out_size=10, activation_function=tf.nn.relu,dropout=True)
    prediction = add_my_layer(inputs=l1,in_size=10,out_size=1, activation_function=None)

# 6: 定义loss-损失量化表征
with tf.name_scope('LOSS'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))  # mean是平均

# 7: 定义Optimizer-优化方式
with tf.name_scope('optimizer'):
    my_opt = tf.train.GradientDescentOptimizer(0.1)  # 这个确定了优化方式
# 8: 然后创建step，这个确定了优化方向minimize，是以后训练要调用的参数，另外这句话把op和loss链接了
with tf.name_scope('TRAIN'):
    train_step = my_opt.minimize(loss)

# 9: 初始化变量等
init = tf.global_variables_initializer()
sess.run(init)

# 10: 输入数据（这里用了循环，实际上只有一句加粗的是核心的）好像会自己判断用到的参数
for i in range(1000):
    # training
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        # to see the step improvement
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))

# 11: Write logs in specific file
log_writing_path = '/Users/Waybaba/PycharmProjects/Machine_learning/MyProject/Tensorflow_Demo/tensorflow_cookbook-master/logs/'
my_writer = tf.summary.FileWriter(log_writing_path, sess.graph)


