import tensorflow as tf


def add_my_dense(input, in_size, out_size, activation_function=None,dropout=False,dropout_rate=0.5):
    """
    :param input:
    :param in_size: 因为是dense层，所以直接输入一个数字就行，代表有多少个神经元
    :param out_size: 同上
    :param activation_function: eg.tf.nn.relu，如果不填就没有
    :param dropout: 默认没有，填了True才有效
    :param dropout_rate: 默认0.5
    :return:
    """

    # init parameters
    Weights = weight_variable([in_size, out_size])#下面那个初始化会导致没有办法收敛，但是不知道为什么，暂时先保留把
    biases = bias_variable([1, out_size])
    # Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    # biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)

    #main
    Wx_plus_b = tf.matmul(input, Weights) + biases

    # dropout选择
    if dropout :
        Wx_plus_b = tf.nn.dropout(Wx_plus_b, dropout_rate)

    # activation选择
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)

    # print output shape
    print("Dense layer output shape: ",outputs.get_shape())

    return outputs

def add_my_cnn(input, filter_shape=[3,3,1,1], strides=[1,1,1,1], padding="SAME"):
    """
    :param input:
    :param filter:会根据这个生成filiter矩阵，[height,width,in_channels, out_channels]，分别表示卷积核的高、宽、深度（与输入的in_channels应相同）、输出 feature map的个数（即卷积核的个数）=output_channels。
    :param strides:表示步长：一个长度为4的一维列表，每个元素跟data_format互相对应，表示在data_format每一维上的移动步长。当输入的默认格式为：“NHWC”，则 strides = [batch , in_height , in_width, in_channels]。其中 batch 和 in_channels 要求一定为1，即只能在一个样本的一个通道上的特征图上进行移动，in_height , in_width表示卷积核在特征图的高度和宽度上移动的布长，即 strideheight和 stridewidth
    :param padding:"SAME"（填充边界，默认）或者"VALID"（不填充边界）
    :return:
    """
    # init patameters
    W_conv = weight_variable(filter_shape)  # patch 5x5, in size 1, out size 32
    b_conv = bias_variable([filter_shape[3]])

    # activation function
    output = tf.nn.relu(tf.nn.conv2d(input=input,filter=W_conv,strides=strides,padding=padding)+b_conv)

    # print output shape
    print("CNN layer output shape: ", output.get_shape())

    return output

def add_my_max_pool(input,ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME'):
    # stride [1, x_movement, y_movement, 1]
    # 默认情况下就是四格变一格
    return tf.nn.max_pool(input, ksize=ksize, strides=strides, padding=padding)

def add_my_rnn(X, weights, biases):
    # hidden layer for input to cell
    ########################################

    # transpose the inputs shape from
    # X ==> (128 batch * 28 steps, 28 inputs)
    X = tf.reshape(X, [-1, n_inputs])

    # into hidden
    # X_in = (128 batch * 28 steps, 128 hidden)
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # X_in ==> (128 batch, 28 steps, 128 hidden)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    # cell
    ##########################################

    # basic LSTM Cell.
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    else:
        cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)
    # lstm cell is divided into two parts (c_state, h_state)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)

    # You have 2 options for following step.
    # 1: tf.nn.rnn(cell, inputs);
    # 2: tf.nn.dynamic_rnn(cell, inputs).
    # If use option 1, you have to modified the shape of X_in, go and check out this:
    # https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py
    # In here, we go for option 2.
    # dynamic_rnn receive Tensor (batch, steps, inputs) or (steps, batch, inputs) as X_in.
    # Make sure the time_major is changed accordingly.
    outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)

    # hidden layer for output as the final results
    #############################################
    # results = tf.matmul(final_state[1], weights['out']) + biases['out']

    # # or
    # unpack to list [(batch, outputs)..] * steps
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        outputs = tf.unpack(tf.transpose(outputs, [1, 0, 2]))    # states is the last outputs
    else:
        outputs = tf.unstack(tf.transpose(outputs, [1,0,2])) # begin:[batch,step,features]=>[step,batch,features],然后吧最后一级取出来
    #取最后一级输出
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']    # shape = (128, 10)

    return results

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1) # stddev是标准差，还可以给一个mean均值 先初始化一个常量，再把这个常量的值作为变量的初始值，返回变量
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape) # 先初始化一个常量，再把这个常量的值作为变量的初始值，返回变量，这两个有什么区别呢？？？
    return tf.Variable(initial)


"""
#这句话可以用来展平，相当于keras里面的flatten层
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])#这个-1代表的应该不是数据数量吧？

cast可以用来改数据格式

"""