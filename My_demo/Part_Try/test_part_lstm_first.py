# encoding: utf-8
from keras import models
from keras import layers
from keras.utils import plot_model
from keras import optimizers
from keras import losses
from keras import metrics
from keras.layers import Input,Dense,Convolution3D,Flatten
from keras import layers
from keras.models import Model
from keras.layers import TimeDistributed
import is_for_import.ntu_date_preprocess_2 as ntu
import is_for_import.ntu_skeleton_visualization as ntusee
import matplotlib.pyplot as plt
import numpy as np
import os,time
os.environ['PATH']=os.environ['PATH']+":/Users/Waybaba/anaconda3/envs/winter2/bin"  #修改环境变量，因为绘图的时候要调用一个底层的命令，而那个命令因为一些错误没有装在系统命令下，所以在这里提前把路径加上，这是在winter2的conda环境下面，如果删除环境，也会导致出错

# input_train,lable_train,input_test,lable_test = ntu.get_date(["A007","A006"])
#
# input_train = input_train.reshape((-1,50,75))#数据整形，然后输入
# input_test = input_test.reshape((-1,50,75))
input_train,lable_train,input_test,lable_test =ntu.load_date("40_actions")
# ntusee.show_gif(input_train[1])

input_train = input_train.transpose((0,2,1,3))
input_test = input_test.transpose((0,2,1,3))
lable_train = ntu.change_into_muti_dim(lable_train)
lable_test = ntu.change_into_muti_dim(lable_test)
print('input_train shape:', input_train.shape)
print('input_test shape:', input_test.shape)

#构建模型/网络

# lstm for sigle point
lstm_unit = 25
inputs = Input(shape=(50,3,))
x = layers.LSTM(units=lstm_unit)(inputs)
sigle_model = Model(inputs=inputs,outputs=x) # lstm for single point
sigle_model.summary()


def slice(x, index):
    return x[index,:]



plot_model(sigle_model,to_file="sigle_model_test_png",show_layer_names=True,show_shapes=True)
input_all_point = Input(shape=(25,50,3,))
x = TimeDistributed(sigle_model)(input_all_point)

# slice into 5 parts
x_0 = layers.Lambda(slice, output_shape=(lstm_unit,), arguments={'index': 0})(x)
x_1 = layers.Lambda(slice, output_shape=(lstm_unit,), arguments={'index': 1})(x)
x_2 = layers.Lambda(slice, output_shape=(lstm_unit,), arguments={'index': 2})(x)
x_3 = layers.Lambda(slice, output_shape=(lstm_unit,), arguments={'index': 3})(x)
x_4 = layers.Lambda(slice, output_shape=(lstm_unit,), arguments={'index': 4})(x)
x_5 = layers.Lambda(slice, output_shape=(lstm_unit,), arguments={'index': 5})(x)
x_6 = layers.Lambda(slice, output_shape=(lstm_unit,), arguments={'index': 6})(x)
x_7 = layers.Lambda(slice, output_shape=(lstm_unit,), arguments={'index': 7})(x)
x_8 = layers.Lambda(slice, output_shape=(lstm_unit,), arguments={'index': 8})(x)
x_9 = layers.Lambda(slice, output_shape=(lstm_unit,), arguments={'index': 9})(x)
x_10 = layers.Lambda(slice, output_shape=(lstm_unit,), arguments={'index': 10})(x)
x_11 = layers.Lambda(slice, output_shape=(lstm_unit,), arguments={'index': 11})(x)
x_12 = layers.Lambda(slice, output_shape=(lstm_unit,), arguments={'index': 12})(x)
x_13 = layers.Lambda(slice, output_shape=(lstm_unit,), arguments={'index': 13})(x)
x_14 = layers.Lambda(slice, output_shape=(lstm_unit,), arguments={'index': 14})(x)
x_15 = layers.Lambda(slice, output_shape=(lstm_unit,), arguments={'index': 15})(x)
x_16 = layers.Lambda(slice, output_shape=(lstm_unit,), arguments={'index': 16})(x)
x_17 = layers.Lambda(slice, output_shape=(lstm_unit,), arguments={'index': 17})(x)
x_18 = layers.Lambda(slice, output_shape=(lstm_unit,), arguments={'index': 18})(x)
x_19 = layers.Lambda(slice, output_shape=(lstm_unit,), arguments={'index': 19})(x)
x_20 = layers.Lambda(slice, output_shape=(lstm_unit,), arguments={'index': 20})(x)
x_21 = layers.Lambda(slice, output_shape=(lstm_unit,), arguments={'index': 21})(x)
x_22 = layers.Lambda(slice, output_shape=(lstm_unit,), arguments={'index': 22})(x)
x_23 = layers.Lambda(slice, output_shape=(lstm_unit,), arguments={'index': 23})(x)
x_24 = layers.Lambda(slice, output_shape=(lstm_unit,), arguments={'index': 24})(x)

# merge
x_main = layers.concatenate(inputs=[x_3,x_2,x_1,x_0])
x_lefthand = layers.concatenate(inputs=[x_4,x_5,x_6])
x_righthand = layers.concatenate(inputs=[x_8,x_9,x_10])
x_leftleg = layers.concatenate(inputs=[x_12,x_13,x_14])
x_rightleg = layers.concatenate(inputs=[x_16,x_17,x_18])

x_main = layers.Dense(units=10,activation='relu')(x_main)
x_lefthand = layers.Dense(units=50,activation='relu')(x_lefthand)
x_righthand = layers.Dense(units=50,activation='relu')(x_righthand)
x_leftleg = layers.Dense(units=30,activation='relu')(x_leftleg)
x_rightleg = layers.Dense(units=30,activation='relu')(x_rightleg)

x = layers.concatenate([x_main, x_lefthand, x_righthand, x_leftleg, x_rightleg], axis=-1)

# x = layers.Flatten()(x)

predictions = Dense(40,activation='softmax')(x)
model = Model(inputs=input_all_point, outputs=predictions)

model.summary()
plot_model(model, to_file='model.png',show_layer_names=True,show_shapes=True)
#编译步骤
model.compile(optimizer='rmsprop',
                # loss='binary_crossentropy',
              loss='categorical_crossentropy',
                metrics=['accuracy'])
#开始
history=model.fit(
    input_train,lable_train,
    epochs=5,
    batch_size=25,
    validation_split=0.2,
    # validation_data=(input_test,lable_test)/
)


localtime = time.asctime( time.localtime(time.time()) )
localtime=localtime.replace(" ","_")
model.save("model_backup/"+localtime+".h5")

"""----------绘图----------"""
#从训练返回值里面提取数据'
#history字典里面的acc,loss分别是训练集的准确度、失误率，
# val_开头的对应的是test集的
history_dict = history.history
acc_values=history_dict['acc']
val_acc_values=history_dict['val_acc']
echochs=range(1,len(acc_values)+1)

#画曲线
plt.plot(echochs,acc_values,'bo',label='Train Accuracy')
plt.plot(echochs,val_acc_values,'b',label='Validation/Test Accuracy')

#图标签设置
plt.title('Accuracy-Epochs Figure')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()#??


plt.show()