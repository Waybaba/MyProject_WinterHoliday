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

def change_data_to_3D(data_25x3_list):
    data_25x3_list = np.reshape(data_25x3_list,newshape=(-1,25,3))
    data_num = data_25x3_list.shape[0]
    data_3D_list = np.zeros(shape=(data_num,3,5,5,3))
    for i in range(data_num):
        data_25x3 = data_25x3_list[i]
        data_3D = np.zeros(shape=(3,5,5,3))
        data_3D[0][0][2] = data_25x3[3]
        data_3D[0][2][2] = data_25x3[20]
        data_3D[0][3][2] = data_25x3[1]
        data_3D[0][4][2] = data_25x3[0]
        data_3D[0][2][1] = data_25x3[4]
        data_3D[0][2][3] = data_25x3[8]
        data_3D[0][4][1] = data_25x3[12]
        data_3D[0][4][3] = data_25x3[16]
        data_3D[1][2][1] = data_25x3[5]
        data_3D[1][2][3] = data_25x3[9]
        data_3D[1][4][1] = data_25x3[12]
        data_3D[1][4][3] = data_25x3[16]
        data_3D[2][2][1] = data_25x3[6]
        data_3D[2][2][3] = data_25x3[10]
        data_3D[2][4][1] = data_25x3[14]
        data_3D[2][4][3] = data_25x3[18]
        data_3D_list[i] = data_3D
    data_3D_list = np.reshape(data_3D_list,newshape=(-1,50,3,5,5,3))
    return data_3D_list

input_train = change_data_to_3D(input_train)
input_test = change_data_to_3D(input_test)
lable_train = ntu.change_into_muti_dim(lable_train)
lable_test = ntu.change_into_muti_dim(lable_test)
print('input_train shape:', input_train.shape)
print('input_test shape:', input_test.shape)

#构建模型/网络
inputs = Input(shape=(3,5,5,3,))
x = Convolution3D(4,(2,2,2),border_mode='same',input_shape=(3,5,5,3,),data_format="channels_last")(inputs)
# x = layers.MaxPool3D(pool_size=(2,2,2),strides=1,padding="same",data_format="channels_last")(x)
x = Convolution3D(4,(2,2,2),border_mode='same',input_shape=(3,5,5,3,),data_format="channels_last")(x)
# x = layers.MaxPool3D(pool_size=(2,2,2),strides=2,padding="same",data_format="channels_last")(x)
x = Convolution3D(4,(2,2,2),border_mode='same',input_shape=(3,5,5,3,),data_format="channels_last")(x)
# x = layers.MaxPool3D(pool_size=(2,2,2),strides=2,padding="same",data_format="channels_last")(x)
x = Convolution3D(4,(1,2,2),border_mode='same',input_shape=(3,5,5,3,),data_format="channels_last")(x)
# x = layers.MaxPool3D(pool_size=(1,2,2),strides=1,data_format="channels_last")(x)
# x = Convolution3D(4,(2,2,2),border_mode='same',input_shape=(3,5,5,3,),data_format="channels_last")(x)
# x = layers.MaxPool3D(pool_size=(2,2,2),strides=2,padding="same",data_format="channels_last")(x)
# x = Convolution3D(2,(2,2,2),border_mode='same',input_shape=(3,5,5,3,),data_format="channels_last")(x)
# x = layers.MaxPool3D(pool_size=(2,2,2),strides=2,padding="same",data_format="channels_last")(x)
sigle_model = Model(inputs=inputs,outputs=x)
sigle_model.summary()
plot_model(sigle_model,to_file="sigle_model_test_png",show_layer_names=True,show_shapes=True)
input_sequence = Input(shape=(50,3,5,5,3,))
x = TimeDistributed(sigle_model)(input_sequence)
x = layers.Reshape(target_shape=(50,-1))(x)
# x = Convolution3D(1,3,3,3,strides=2,border_mode='same',input_shape=(3,5,5,3,),data_format="channels_last")(x)
x = layers.LSTM(units=128)(x)
# x = Flatten()(x)
# x = Dense(8,activation='relu')(x)
predictions = Dense(40,activation='softmax')(x)
model = Model(inputs=input_sequence, outputs=predictions)

model.summary()
# plot_model(model, to_file='model.png')
#编译步骤
model.compile(optimizer='rmsprop',
                # loss='binary_crossentropy',
              loss='categorical_crossentropy',
                metrics=['accuracy'])
#开始
history=model.fit(
    input_train,lable_train,
    epochs=10,
    batch_size=16,
    validation_split=0.2,
    validation_data=(input_test,lable_test)
)


plot_model(model,to_file="model_test_png",show_layer_names=True,show_shapes=True)

#开始
history=model.fit(
    input_train,lable_train,
    epochs=20,
    batch_size=16,
    validation_split=0.2,
    validation_data=(input_test,lable_test)
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