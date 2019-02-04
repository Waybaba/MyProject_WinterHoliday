from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras import models
from keras import layers
from keras.utils import to_categorical

"""----------数据准备----------"""
(train_images,train_lables),(test_images,test_lables)=mnist.load_data()
#取出一个数据
digit=train_images[4]
plt.imshow(digit,cmap=plt.cm.binary)#绘图
plt.show()#确认显示
#数据整形
train_images=train_images.reshape((60000,28,28,1))#注意这里的reshape形状和第二章不一样
train_images=train_images.astype('float32')/255
test_images=test_images.reshape((10000,28,28,1))
test_images=test_images.astype('float32')/255
#准备标签
train_lables=to_categorical(train_lables)
test_lables=to_categorical(test_lables)
#打印输出数据信息
print("Information about the dataset:")
print('train_images shape:', train_images.shape)
print('test_images shape:', test_images.shape)

"""----------网络准备----------"""
#构建网络
model=models.Sequential()#初始化

#卷积层
model.add(
    layers.Conv2D(32,(3,3),activation="relu",input_shape=(28,28,1))
                  )
model.add(layers.MaxPool2D(2,2))
model.add(
    layers.Conv2D(64,(3,3),activation="relu")
                  )
model.add(layers.MaxPool2D(2,2))
model.add(
    layers.Conv2D(64,(3,3),activation="relu")
                  )

model.add(layers.Flatten())#线性一维展开，为之后提供便利

#Dense层
model.add(
    layers.Dense(64,activation='relu')
)
model.add(
    layers.Dense(10,activation='softmax')
)
#编译网络
model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])


"""----------运行----------"""
#进行训练，返回值包含了每次训练后的精确度
history=model.fit(train_images,train_lables,
                    epochs=3,#迭代次数：针对数据的训练次数
                    batch_size=64,#数据小组大小，接受多少数据后开始调参数
                    validation_data=(test_images,test_lables)#用于验证的test数据集，可以缺省，写上返回值里面会有每次迭代周期后的测试集表现
                    )

"""----------输出打印----------"""
test_loss,test_acc=model.evaluate(test_images,test_lables)
print(test_acc)

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

# plt.clf()
# history_dict = history.history
# acc_values=history_dict['acc']
# val_acc_values=history_dict['val_acc']
#
# echochs=range(1,len(acc_values)+1)
#
# plt.plot(echochs,acc_values,'bo',label='Train acc')
# plt.plot(echochs,val_acc_values,'b',label='Validation/Test acc')
#
# plt.title('Acc Figure')
# plt.xlabel('Epochs')
# plt.ylabel('acc')
# plt.legend()#??
#
# plt.show()

# history_dict = history.history
# loss_values=history_dict['loss']
# val_loss_values=history_dict['val_loss']
#
# echochs=range(1,len(loss_values)+1)
#
# plt.plot(echochs,loss_values,'bo',label='Train Loss')
# plt.plot(echochs,val_loss_values,'b',label='Validation/Test Loss')
#
# plt.title('Loss Figure')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()#??
#
# plt.show()









