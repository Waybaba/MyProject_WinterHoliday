# Operations on a Computational Graph
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Create graph
sess = tf.Session()

# Create tensors

# Create data to feed in
x_vals = np.array([1., 3., 5., 7., 9.])
x_data = tf.placeholder(tf.float32)
m_const = tf.constant(3.)

# Multiplication
my_product = tf.multiply(x_data, m_const)
your_product = tf.multiply(my_product,m_const)

for x_val in x_vals:
    print(sess.run(my_product, feed_dict={x_data: x_val}))

# View the tensorboard graph by running the following code and then
#    going to the terminal and typing:
#    $ tensorboard --logdir=tensorboard_logs
merged = tf.summary.merge_all()
if not os.path.exists('tensorboard_logs/'):
    os.makedirs('tensorboard_logs/')

# Write logs in specific file
log_writing_path = '/Users/Waybaba/PycharmProjects/Machine_learning/MyProject/Tensorflow_Demo/tensorflow_cookbook-master/logs/'
my_writer = tf.summary.FileWriter(log_writing_path, sess.graph)
