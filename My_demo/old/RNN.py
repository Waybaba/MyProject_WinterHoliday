"""
用RNN对评论进行二值分类的问题

先获取25000条评论数据，这里库里面就进行了整数对应，把每个单词映射到一个整数上
因为序列一般长度都不一样，所以要先进行一个sequence.pad_sequences(input_train, maxlen=maxlen)，把长的去掉，短的补充
然后就送入训练层了
第一层是一个embedding
然后是rnn
最后接一个dense
问题：32的batch是什么，和后面的batch没有冲突吗？
"""


from keras.datasets import imdb

import matplotlib.pyplot as plt

from keras import models
from keras import layers
from keras.utils import to_categorical

from keras.preprocessing.sequence import sequence #这是一个sequence数据处理的库，用来对数据进行一些操作

"""----------数据准备----------"""
max_features = 10000
maxlen = 500
batch_size = 32
print('Loading data...')

#获取到数据
(input_train, y_train), (input_test, y_test) = imdb.load_data( num_words=max_features)
print(len(input_train), 'train sequences')#打印数据个数，一共25000调评论，单词都化成数字了，是一个25000xn的列表，其中n长度不定
print(len(input_test), 'test sequences')
print('Pad sequences (samples x time)')
#数据整形
input_train =sequence.pad_sequences(input_train, maxlen=maxlen)#这里截取的是25000条评论各自的长度，而不是取25000里面500个。？所以说sequence截断的二级维度上的
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)
print('input_train shape:', input_train.shape)
print('input_test shape:', input_test.shape)

"""----------网络准备----------"""
"embedding layer是一个可变的字典映射，将单词（已经化成整数）映射到一个空间里，使得能够保留之间的关系，10000表示这个空间只接受前10000个常用的，这个层好像只适合用于文本类的信息处理"
model=models.Sequential()
model.add(layers.Embedding(max_features,32))#max_fetures是10000个常用的单词，这里的32不知道是什么，可能是数据组数，那后面的batch_size有什么用呢
model.add(layers.SimpleRNN(32))
model.add(layers.Dense(1,activation='sigmoid'))
model.summary()#打印一下模型

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc']
              )

"""----------运行----------"""
#进行训练，返回值包含了每次训练后的精确度
history=model.fit(
    input_train,y_train,
    epochs=3,
    batch_size=128,
    validation_split=0.2
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




