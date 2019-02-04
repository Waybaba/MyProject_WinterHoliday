# coding=UTF-8
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
import is_for_import.ntu_skeleton_visualization as ntusee
import is_for_import.ntu_date_preprocess_2 as ntu
import matplotlib.pyplot as plt

import tensorflow as tf


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
# def add_my_rnn(input):
#     input=tf.transpose(input,[1,0,2])#step,batch,features
#     inputs=tf.unstack(input)
#     outputs=[]
#     for i in range(len(inputs)):
#         outputs.append(add_my_layer(inputs[i],in_size=75,out_size=5,activation_function=tf.nn.relu,dropout=True))
#     outputs=tf.stack(outputs)
#     outputs = tf.transpose(outputs, [1, 0, 2])
#     cell=tf.nn.rnn_cell.BasicRNNCell(num_units=5)
#     output,state=tf.nn.dynamic_rnn(cell,outputs,dtype=tf.float32)
#     output=tf.nn.dropout(output,0.5)
#     output=tf.transpose(output,[1,0,2])
#     last = tf.gather(output,int(output.shape[0])-1)
#     print("output shape: ",last.get_shape())
#     return last



# 2: Create data

def add_my_rnn(input):
    # 一个动作流里面有50组，把各组分别送入数据中提取，因为这个算数元是可以共享的。（之后可以尝试段送入提取，有重叠部分的那种）（v2 可以考虑程序自由更改长度）
    # 先把75个特征综合一下到16个
    input=tf.reshape(input,(-1,75))
    outputs=add_my_layer(inputs=input,in_size=75,out_size=32,activation_function=tf.nn.relu,dropout=True)
    outputs = add_my_layer(inputs=outputs, in_size=32, out_size=16, activation_function=tf.nn.relu, dropout=True)
    input=tf.reshape(outputs,(-1,50,16))
    print("Intense output; ",outputs.get_shape())
    #送到rnn里面
    cell=tf.nn.rnn_cell.BasicRNNCell(num_units=16)
    print("in1 shape: ", input.get_shape())
    output,state=tf.nn.dynamic_rnn(cell,input,dtype=tf.float32)
    output=tf.nn.dropout(output,0.5)
    print("out1 state shape: ", state.get_shape())
    # output=tf.transpose(output,[1,0,2])
    # last = tf.gather(output,output.shape[0]-1)
    # last = tf.reshape(output, [-1,50*128])
    print("rnn output shape: ",state.get_shape())
    return state

input_train,lable_train,input_test,lable_test =ntu.load_date("2_actions")
ntusee.show_gif(input_train[0])

input_train = input_train.reshape((-1,50,75))#数据整形，然后输入
input_test = input_test.reshape((-1,50,75))
# ntusee.show_gif(input_test[0].reshape((50,25,3)))
# ntusee.show_gif(input_train[0]) ????????????????????????显示不鸟，不知道为什么
lable_train = ntu.change_into_muti_dim(lable_train)
lable_test = ntu.change_into_muti_dim(lable_test)
lable_train = lable_train.astype(np.float32)#不知道为什么不转成32会导致交叉墒出错
lable_test = lable_test.astype(np.float32)
# sample
input_train=input_train[0:500]
input_test=input_test[0:100]
lable_train=lable_train[0:500]
lable_test=lable_test[0:100]

print('input_train shape:', input_train.shape)
print('input_test shape:', input_test.shape)

# 3: Create graph
sess = tf.Session()

# 4: 创建输入点，一个数据，一个标签/或者说是拟合值
with tf.name_scope('INPUT'):
    xs = tf.placeholder(tf.float32, [None, 50,75])
    ys = tf.placeholder(tf.float32, [None, 2])

# 5: 添加layer（包括了变量和运算关系，返回输出值）
# ##############################################################################################################################################################################################################################################################################################
with tf.name_scope('NETWOARS'):
    l0=add_my_rnn(input=xs)
    l1 = add_my_layer(inputs=l0,in_size=16, out_size=64, activation_function=tf.nn.relu,dropout=True)
    l2 = add_my_layer(inputs=l1, in_size=64, out_size=10, activation_function=tf.nn.relu, dropout=True)
    prediction = add_my_layer(inputs=l2,in_size=10,out_size=2, activation_function=tf.nn.softmax)
################################################################################################################################################################################################################################################################################################

# 6: 定义loss-损失量化表征
# def get_accuracy():


with tf.name_scope('LOSS'):
    # loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))  # mean是平均
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ys, logits=prediction))
    # loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction,labels=ys))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(ys,1), tf.argmax(prediction,1)),tf.float32))

# 7: 定义Optimizer-优化方式
with tf.name_scope('optimizer'):
    my_opt = tf.train.GradientDescentOptimizer(0.0001)  # 这个确定了优化方式
# 8: 然后创建step，这个确定了优化方向minimize，是以后训练要调用的参数，另外这句话把op和loss链接了
with tf.name_scope('TRAIN'):
    train_step = my_opt.minimize(loss)

# 9: 初始化变量等
init = tf.global_variables_initializer()
sess.run(init)

# 11: Write logs in specific file
log_writing_path = '/Users/Waybaba/PycharmProjects/Machine_learning/MyProject/Tensorflow_Demo/tensorflow_cookbook-master/logs/'
my_writer = tf.summary.FileWriter(log_writing_path, sess.graph)


# 10: 输入数据（这里用了循环，实际上只有一句加粗的是核心的）
for i in range(20):
    # training
    sess.run(train_step, feed_dict={xs: input_train, ys: lable_train})
    if i % 20 == 0:
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


