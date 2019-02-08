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
train_images=train_images.reshape((60000,28*28))
train_images=train_images.astype('float32')/255
test_images=test_images.reshape((10000,28*28))
test_images=test_images.astype('float32')/255
#准备标签
train_lables=to_categorical(train_lables)
test_lables=to_categorical(test_lables)

"""----------网络准备----------"""
#构建网络
model=models.Sequential()
model.add(layers.Dense(512,activation='relu',input_shape=(28*28,)))
model.add(layers.Dense(10,activation='softmax'))

#编译步骤
model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

"""----------运行----------"""
history=model.fit(train_images,train_lables,
                    epochs=2,
                    batch_size=128,
                    validation_data=(test_images,test_lables)
                    )


"""----------绘图----------"""
plt.clf()
history_dict = history.history
acc_values=history_dict['acc']
val_acc_values=history_dict['val_acc']

echochs=range(1,len(acc_values)+1)

plt.plot(echochs,acc_values,'bo',label='Train acc')
plt.plot(echochs,val_acc_values,'b',label='Validation/Test acc')

plt.title('Acc Figure')
plt.xlabel('Epochs')
plt.ylabel('acc')
plt.legend()#??

plt.show()

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









