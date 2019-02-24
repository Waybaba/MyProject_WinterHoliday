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
lstm_unit = 30
inputs = Input(shape=(50,3,))
x = layers.LSTM(units=lstm_unit)(inputs)
sigle_model = Model(inputs=inputs,outputs=x) # lstm for single point
sigle_model.summary()


def slice(x, index):
    return x[index,:]



plot_model(sigle_model,to_file="sigle_model_test_png",show_layer_names=True,show_shapes=True)
input_all_point = Input(shape=(25,50,3,))
x = TimeDistributed(sigle_model)(input_all_point)


x = layers.Flatten()(x)

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