
from keras import models
from keras import layers
import pickle
from keras import optimizers
from keras import losses
from keras import metrics
import for_import.ntu_date_preprocess_2 as ntu
import matplotlib.pyplot as plt
import for_import.ntu_skeleton_visualization as ntusee
import tensorflow as tf
import numpy as np

# preparate date
input_train ,y_train ,input_test ,y_test =ntu.load_date("4_actions")
input_train = input_train.reshape((-1, 50, 75))  # 数据整形，然后输入
input_test = input_test.reshape((-1, 50, 75))
# ntusee.show_gif(input_train[0])
y_train = ntu.change_into_muti_dim(y_train)
y_test = ntu.change_into_muti_dim(y_test)
print('input_train shape:', input_train.shape)
print('input_test shape:', input_test.shape)


# load_model
load_model_name = "second_flatten4"
model = models.load_model('/Users/Waybaba/PycharmProjects/Machine_learning/Date_and_Else/models_backup/'+load_model_name+'.h5')

# Writelogsinspecificfile
sess = tf.Session()
log_writing_path = '/Users/Waybaba/else/tensorboard_logs/'
my_writer = tf.summary.FileWriter(log_writing_path, sess.graph)

# 开始

history = model.predict_classes(input_test, batch_size=None, verbose=1)
count_right = 0
count_wrong = 0
for index in range(len(input_test)):
    max_position = history[index]
    print("The predict lable is : ",max_position,end="")
    print(" , real lable is ",y_test[index].argmax(),".",end="")
    if y_test[index].argmax() == max_position :
        count_right=count_right+1
        print(" ^ O ^ ")
    else:
        count_wrong=count_wrong+1
        print("   X   ")
print("Ringht:",count_right,end="   ")
print("Wrong:",count_wrong)
print("Accuracy: ",float(count_right)/(float(count_right)+float(count_wrong)))





