from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics

构建模型/网络
model = models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(10000,0)) )
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

#编译步骤
network.compile(optimizer='rmsprop',
                loss='binary_crossentropy',
                metrics=['accuracy'])

### 自定义optimizer,loss,metric的方法
#
# model.compile(optimizer=optimizers.RMSprop(lr=0.001),#通过这里给optimizer传参数
#               loss=loss.binary_crossentropy,
#               metrics=['accuracy']
#               )



network=models.Sequential()
network.add(layers.Dense(512,activation='relu',input_shape=(28*28,)))
network.add(layers.Dense(10,activation='softmax'))



#开始
network.fit(train_images,train_lables,epochs=2,batch_size=128)