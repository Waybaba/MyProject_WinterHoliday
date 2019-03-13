# encoding: utf-8
from keras import models
from keras import layers
from keras.utils import plot_model
from keras import optimizers
from keras import losses
from keras import metrics
from keras.layers import Input, Dense
from keras.models import Model
from keras.layers import TimeDistributed
import is_for_import.ntu_date_preprocess_2 as ntu
import is_for_import.ntu_skeleton_visualization as ntusee
import matplotlib.pyplot as plt
import numpy as np
import os, time, math

os.environ['PATH'] = os.environ[
                         'PATH'] + ":/Users/Waybaba/anaconda3/envs/winter2/bin"  # 修改环境变量，因为绘图的时候要调用一个底层的命令，而那个命令因为一些错误没有装在系统命令下，所以在这里提前把路径加上，这是在winter2的conda环境下面，如果删除环境，也会导致出错

# input_train,lable_train,input_test,lable_test = ntu.get_date(["A007","A006"])
#
# input_train = input_train.reshape((-1,50,75))#数据整形，然后输入
# input_test = input_test.reshape((-1,50,75))
input_train, lable_train, input_test, lable_test = ntu.load_date("40_actions")
# ntusee.show_gif(input_train[1])

input_train = input_train.reshape((-1, 50, 75))  # 数据整形，然后输
input_test = input_test.reshape((-1, 50, 75))
lable_train = ntu.change_into_muti_dim(lable_train)
lable_test = ntu.change_into_muti_dim(lable_test)
# choose_index_50_to_30 = np.arange(30)
# for i in np.arange(30):
#     choose_index_50_to_30[i] = math.floor(choose_index_50_to_30[i] * 5/3)
# input_train = input_train[:,choose_index_50_to_30,:]
# input_test = input_test[:,choose_index_50_to_30,:]


print('input_train shape:', input_train.shape)
print('input_test shape:', input_test.shape)

# 构建模型/网络

model = models.Sequential()

model.add(layers.LSTM(
    units=128,
    batch_input_shape=(None, 30, 75),  # 75相当于特征数量，或者也可以叫维度，这里有3*25个坐标点，要提前整形
    return_sequences=0  # 是否返回整个列表，用于再嵌套LSTM
))

# model.add(layers.Dense(16, activation='relu'))  # 这个shape好像是抛去了数据的第一维度，认为这个
# model.add(layers.Dense(16, activation='relu'))
# model.add(layers.Dense(8, activation='relu'))
# model.add(layers.Dense(40, activation='relu'))
model.add(layers.Dense(40, activation='softmax'))
model.summary()
plot_model(model, to_file='model_backup/predict_model.png',show_layer_names=True, show_shapes=True)
# 编译步骤
model.compile(optimizer='rmsprop',
              # loss='binary_crossentropy',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# 开始
for train_eposide in range(5):
    input_train_choose = np.zeros(shape=(len(input_train), 30, 75))
    print("choose datas ...")
    for date_count in range(len(input_train)):
        current_date = input_train[date_count]
        index = np.random.choice(a=np.arange(50), size=30, replace=False)
        index = np.sort(index, axis=-1)
        input_train_choose[date_count] = current_date[index]

    history = model.fit(
        input_train_choose, lable_train,
        epochs=1,
        batch_size=16,
        validation_split=0.2,
    )

plot_model(model, to_file="model_backup/predict_model.png", show_layer_names=True, show_shapes=True)


# 存
localtime = time.asctime(time.localtime(time.time()))
localtime = localtime.replace(" ", "_")
model.save("model_backup/" + "30_predict_model" + ".h5")
print("Save Successfully")

"""----------绘图----------"""
# 从训练返回值里面提取数据'
# history字典里面的acc,loss分别是训练集的准确度、失误率，
# val_开头的对应的是test集的
history_dict = history.history
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']
echochs = range(1, len(acc_values) + 1)

# 画曲线
plt.plot(echochs, acc_values, 'bo', label='Train Accuracy')
plt.plot(echochs, val_acc_values, 'b', label='Validation/Test Accuracy')

# 图标签设置
plt.title('Accuracy-Epochs Figure')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()  # ??

plt.show()
