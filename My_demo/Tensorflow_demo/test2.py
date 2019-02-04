"""
Implementing a one-layer Neural Network

We will illustrate how to create a one hidden layer NN

We will use the iris data for this exercise

We will build a one-hidden layer neural network
 to predict the fourth attribute, Petal Width from
 the other three (Sepal length, Sepal width, Petal length).
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
from tensorflow.python.framework import ops
import pickle
import os
import Basic_demo.ntu_date_preprocess_2 as ntu
import Basic_demo.ntu_skeleton_visualization as ntusee



# load date from the disk
def load_date(save_name):
    #先写一个两个提取的
    #再写一个3个提取的
    #之后再考虑要不要把所有的单独提出来
    save_path = '/Users/Waybaba/PycharmProjects/Machine_learning/Date_and_Else/variables/'
    if not os.path.exists(save_path + save_name):
        os.makedirs(save_path + save_name)
        input_train, y_train, input_test, y_test = ntu.get_date(["A007", "A006"])
        # open file and sava
        f = open(save_path + save_name +'/'+"input_train.txt", 'wb')
        pickle.dump(input_train, f)
        f.close()
        f = open(save_path + save_name + '/' + "y_train.txt", 'wb')
        pickle.dump(y_train, f)
        f.close()
        f = open(save_path + save_name + '/' + "input_test.txt", 'wb')
        pickle.dump(input_test, f)
        f.close()
        f = open(save_path + save_name + '/' + "y_test.txt", 'wb')
        pickle.dump(y_test, f)
        f.close()
    else:
        f = open(save_path + save_name + '/' + "input_train.txt", 'rb')
        input_train = pickle.load(f)
        f.close()
        f = open(save_path + save_name + '/' + "y_train.txt", 'rb')
        y_train = pickle.load(f)
        f.close()
        f = open(save_path + save_name + '/' + "input_test.txt", 'rb')
        input_test = pickle.load(f)
        f.close()
        f = open(save_path + save_name + '/' + "y_test.txt", 'rb')
        y_test = pickle.load(f)
        f.close()
    return input_train,y_train,input_test,y_test

input_train,y_train,input_test,y_test = load_date("2_actions")
input_train = input_train.reshape((-1,50*75))#数据整形，然后输入
input_test = input_test.reshape((-1,50*75))
#change the format
new_y_train = np.zeros(shape=[len(y_train),2],dtype=float)
new_y_test = np.zeros(shape=[len(y_test),2],dtype=float)
for index in range(len(y_train)):
    if y_train[index] == 0.0 :
        new_y_train[index][0] = 1.0
    else:
        new_y_train[index][1] = 1.0
for index in range(len(y_test)):
    if y_test[index] == 0.0 :
        new_y_test[index][0] = 1.0
    else:
        new_y_test[index][1] = 1.0
# ntusee.show_gif(input_train[0])
print('input_train shape:', input_train.shape)# input_train shape: (1464, 3750)
print('input_test shape:', input_test.shape)# input_test shape: (366, 3750)
x_vals_train = input_train.astype(np.float32)
y_vals_train = new_y_train.astype(np.float32)
x_vals_test = input_test.astype(np.float32)
y_vals_test = new_y_test.astype(np.float32)


# Create graph session
ops.reset_default_graph()
sess = tf.Session()

# make results reproducible
seed = 3
tf.set_random_seed(seed)
np.random.seed(seed)

# Declare batch size
batch_size = 200

# Initialize placeholders
x_data = tf.placeholder(shape=[None, 3750], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 2], dtype=tf.float32)

# Create variables for both NN layers
hidden_layer_nodes = 50
with tf.name_scope('variables_all'):
    A1 = tf.Variable(tf.random_normal(shape=[3750, hidden_layer_nodes]), name='muli_1')  # inputs -> hidden nodes
    b1 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes]), name='bias_1')  # one biases for each hidden node
    A_out = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes, 2]), name='muti_2')  # hidden inputs -> 1 output
    b_out = tf.Variable(tf.random_normal(shape=[1]), name='bias_2')  # 1 bias for the output
    A2 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes, hidden_layer_nodes]), name='muli_1')  # inputs -> hidden nodes
    b2 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes]), name='bias_1')  # one biases for each hidden node

# Declare model operations
with tf.name_scope('hidden_layer_1'):
    hidden_output1 = tf.nn.relu(tf.add(tf.matmul(x_data, A1), b1))
# with tf.name_scope('hidden_layer_2'):
#     hidden_output2 = tf.nn.relu(tf.add(tf.matmul(hidden_output1, A2), b2))
with tf.name_scope('output_layer'):
    final_output = tf.nn.relu(tf.add(tf.matmul(hidden_output1, A_out), b_out))

# Declare loss function (MSE)
with tf.name_scope('LOSS'):

    # loss = tf.reduce_mean(tf.square(y_target - final_output))
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=final_output,labels=y_target))
    loss = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits=final_output,labels=y_target) )
# Declare optimizer

with tf.name_scope('Train'):
    my_opt = tf.train.GradientDescentOptimizer(0.9)
    # my_opt = tf.train.MomentumOptimizer(0.005 ,0.9)
    train_step = my_opt.minimize(loss)

# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# Writelogsinspecificfile
log_writing_path = '/Users/Waybaba/PycharmProjects/Machine_learning/MyProject/Tensorflow_Demo/tensorflow_cookbook-master/logs/'
my_writer = tf.summary.FileWriter(log_writing_path, sess.graph)

# Training loop
loss_vec = []
test_loss = []
for i in range(10000):
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    rand_x = x_vals_train[rand_index]
    rand_y = y_vals_train[rand_index]
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    loss_vec.append(np.sqrt(temp_loss))

    test_temp_loss = sess.run(loss, feed_dict={x_data: x_vals_test, y_target: y_vals_test})
    test_loss.append(np.sqrt(test_temp_loss))

    if (i + 1) % 50 == 0:
        print('Generation: ' + str(i + 1) + '. Loss = ' + str(temp_loss))

# Plot loss (MSE) over time
plt.plot(loss_vec, 'k-', label='Train Loss')
plt.plot(test_loss, 'r--', label='Test Loss')
plt.title('Loss (MSE) per Generation')
plt.legend(loc='upper right')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()
