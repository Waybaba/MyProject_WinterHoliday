# encoding: utf-8
from keras import models
from keras import layers
from keras.utils import plot_model
from keras import optimizers
from keras import losses
from keras import metrics
from keras.layers import Input,Dense,Flatten,Lambda
from keras.models import Model
from keras.layers import TimeDistributed
import is_for_import.ntu_date_preprocess_2 as ntu
import is_for_import.ntu_skeleton_visualization as ntusee
import matplotlib.pyplot as plt
import os,time
import numpy as np
os.environ['PATH']=os.environ['PATH']+":/Users/Waybaba/anaconda3/envs/winter2/bin"  #修改环境变量，因为绘图的时候要调用一个底层的命令，而那个命令因为一些错误没有装在系统命令下，所以在这里提前把路径加上，这是在winter2的conda环境下面，如果删除环境，也会导致出错

# input_train,lable_train,input_test,lable_test = ntu.get_date(["A007","A006"])
#
# input_train = input_train.reshape((-1,50,75))#数据整形，然后输入
# input_test = input_test.reshape((-1,50,75))
input_train,lable_train,input_test,lable_test =ntu.load_date("4_actions")
# ntusee.show_gif(input_train[1])

input_train = input_train.reshape((-1,50,75))#数据整形，然后输
input_test = input_test.reshape((-1,50,75))
lable_train = ntu.change_into_muti_dim(lable_train)
lable_test = ntu.change_into_muti_dim(lable_test)



print('input_train shape:', input_train.shape)
print('input_test shape:', input_test.shape)

#构建模型/网络
#-------------------------------------------------------------------------------------------------------------------#
input_whole = Input(shape=(50,75,))
tensor_1 = Flatten()(input_whole)
tensor_2 = Dense(50*16,activation='relu')(tensor_1)
tensor_3 = Dense(50*4,activation='relu')(tensor_2)
tensor_4 = Dense(50*1,activation='relu')(tensor_3)
# choose_index_model = Model(inputs=input_whole,outputs=tensor_4)


def lambda_fuc_1(inputx):
    input_whole_sequences = inputx[0]
    input_mask = inputx[1]
    # whole_number = 50
    # output_number = 30
    # index = np.argsort(input_mask)[-output_number:] # 获取到前30的索引列表
    # output = np.array([])
    # for i in range(whole_number):
    #     if np.sum(index == i) != 0: # if exist
    #         output = np.append(output, input_whole_sequences[i])
    output = input_whole_sequences[:,0:29,:]
    return input_whole_sequences
# choose_index = choose_index_model(input_whole)

tensor_5 = Lambda(function=lambda_fuc_1,output_shape=(30,75),arguments=None)([input_whole,tensor_4])

# choose_model = Model(inputs=[Input(shape=(50,75,)), Input(shape=(50,))],outputs=tensor_5,name="choose_to_30")

# single process
inputs = Input(shape=(75,))
x = Dense(128,activation='relu')(inputs)
x = Dense(128,activation='relu')(x)
predictions = Dense(128,activation='relu')(x)
single_model = Model(inputs=inputs, outputs=predictions)

single_time_input = Input(shape=(75,))
single_time_output = single_model(single_time_input)

# sequence process
sequence_input = tensor_5
processed_sequence = TimeDistributed(single_model)(sequence_input)

# pre_model = Model(inputs=sequence_input, outputs=processed_sequence,name="pre_process")

tensor_6 = Flatten()(processed_sequence)
tensor_7 = Dense(4,activation='softmax')(tensor_6)

model = Model(inputs=input_whole,outputs=tensor_7)
# model = models.Sequential()
#
#
# # model.add(layers.SimpleRNN(
# #     # unroll=True,
# #     units=16,
# #     batch_input_shape=(None, 50, 75),#75相当于特征数量，或者也可以叫维度，这里有3*25个坐标点，要提前整形
# #     return_sequences=0
# # ))
# model.add(choose_model)
# model.add(pre_model)
# model.add(layers.LSTM(
#     units=128,
#     batch_input_shape=(None, 30, 75),  # 75相当于特征数量，或者也可以叫维度，这里有3*25个坐标点，要提前整形
#     return_sequences=0  # 是否返回整个列表，用于再嵌套LSTM
# ))
#
# # model.add(layers.Dense(16, activation='relu'))  # 这个shape好像是抛去了数据的第一维度，认为这个
# # model.add(layers.Dense(16, activation='relu'))
# # model.add(layers.Dense(8, activation='relu'))
# # model.add(layers.Dense(4, activation='relu'))
# model.add(layers.Dense(4,activation='softmax'))
model.summary()
#-------------------------------------------------------------------------------------------------------------------#
plot_model(model,to_file="choose_frame_dense.png",show_layer_names=True,show_shapes=True)
#编译步骤
model.compile(optimizer='rmsprop',
                # loss='binary_crossentropy',
              loss='categorical_crossentropy',
                metrics=['accuracy'])
#开始
history=model.fit(
    input_train,lable_train,
    epochs=5,
    batch_size=16,
    validation_split=0.2,
    validation_data=(input_test,lable_test)
)


plot_model(model,to_file="model_test_png",show_layer_names=True,show_shapes=True)
pre_model.trainable = False
model.compile(optimizer='rmsprop',
                # loss='binary_crossentropy',
              loss='categorical_crossentropy',
                metrics=['accuracy'])
model.summary()
#开始
history=model.fit(
    input_train,lable_train,
    epochs=10,
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