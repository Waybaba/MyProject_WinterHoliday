

from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics
import Basic_demo.ntu_date_preprocess_2 as ntu
import matplotlib.pyplot as plt


input_train,y_train,input_test,y_test = ntu.get_date(["A007","A006"])

input_train = input_train.reshape((-1,50,75))#数据整形，然后输入
input_test = input_test.reshape((-1,50,75))



print('input_train shape:', input_train.shape)
print('input_test shape:', input_test.shape)

#构建模型/网络
model = models.Sequential()

model.add(layers.SimpleRNN(
    # unroll=True,
    units=10,
    batch_input_shape=(None, 50, 75),#75相当于特征数量，或者也可以叫维度，这里有3*25个坐标点，要提前整形
    return_sequences=0
))
# model.add(layers.LSTM(
#     units=1,
#     batch_input_shape=(None, 50, 75),#75相当于特征数量，或者也可以叫维度，这里有3*25个坐标点，要提前整形
#     return_sequences=0#是否返回整个列表，用于再嵌套LSTM
# ))

model.add(layers.Dense(16, activation='relu'))  # 这个shape好像是抛去了数据的第一维度，认为这个
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

#编译步骤
model.compile(optimizer='rmsprop',
                loss='binary_crossentropy',
                metrics=['accuracy'])


#开始
history=model.fit(
    input_train,y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.2,
    validation_data=(input_test, y_test)
)



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