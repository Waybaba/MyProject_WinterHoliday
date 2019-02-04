# View more python tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from My_demo.Tensorflow_demo.for_import import my_basic_layer as my
# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result


# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784])/255.   # 28x28
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 28, 28, 1])
print("Image shape: ",x_image.shape)  # [n_samples, 28,28,1]





## conv1 layer ##
h_conv1 = my.add_my_cnn(input=x_image,filter_shape=[5,5,1,32],strides=[1,1,1,1])
h_pool1 = my.add_my_max_pool(h_conv1)  # output size 14x14x32

## conv2 layer ##
h_conv2 =  my.add_my_cnn(input=h_pool1,filter_shape=[5,5,32,64],strides=[1,1,1,1]) # output size 14x14x64
h_pool2 = my.add_my_max_pool(h_conv2)   # output size 7x7x64

## flatten ##
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

## fc1 layer ##
h_fc1_drop = my.add_my_dense(input=h_pool2_flat,in_size=7*7*64,out_size=1024,dropout=True,dropout_rate=keep_prob,activation_function=tf.nn.relu)

## fc2 layer ##
final_out = my.add_my_dense(input=h_fc1_drop,in_size=1024,out_size=10,dropout=False)
prediction = tf.nn.softmax(final_out)









# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))       # loss
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()

# important step
init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 50 == 0:
        print("Accuracy: ",compute_accuracy(mnist.test.images[:1000], mnist.test.labels[:1000]))

