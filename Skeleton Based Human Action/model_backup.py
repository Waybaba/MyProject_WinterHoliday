
from keras import models
from keras import layers
import pickle
from keras import optimizers
from keras import losses
from keras import metrics
import Basic_demo.ntu_date_preprocess_2 as ntu
import matplotlib.pyplot as plt
import Basic_demo.ntu_skeleton_visualization as ntusee
import tensorflow as tf

input_train ,y_train ,input_test ,y_test =ntu.load_date("4_actions")

input_train = input_train.reshape((-1, 50, 75))  # 数据整形，然后输入
input_test = input_test.reshape((-1, 50, 75))
# ntusee.show_gif(input_train[0])
y_train = ntu.change_into_muti_dim(y_train)
y_test = ntu.change_into_muti_dim(y_test)

print('input_train shape:', input_train.shape)
print('input_test shape:', input_test.shape)


# load_model
load_model_name = "second_flatten3"
model = models.load_model('/Users/Waybaba/PycharmProjects/Machine_learning/Date_and_Else/models_backup/'+load_model_name+'.h5')

# Writelogsinspecificfile
sess = tf.Session()
log_writing_path = '/Users/Waybaba/else/tensorboard_logs/'
my_writer = tf.summary.FileWriter(log_writing_path, sess.graph)

# 开始

history=model.fit(
    input_train,y_train,
    epochs=100,
    batch_size=16,
    # validation_split=0.2,
    validation_data=(input_test, y_test)
)
# score = model.evaluate(
#     input_train, y_train,
#     batch_size=128)
#保存模型到first.h5
model.save(filepath='/Users/Waybaba/PycharmProjects/Machine_learning/Date_and_Else/models_backup/'+load_model_name+'.h5',overwrite=True, include_optimizer=True )


"""----------绘图----------"""
# 从训练返回值里面提取数据'
# history字典里面的acc,loss分别是训练集的准确度、失误率，
# val_开头的对应的是test集的
history_dict = history.history
# history_dict = score.history
acc_values = history_dict['acc']#train accuracy
val_acc_values = history_dict['val_acc']#test accuracy
echochs = range(1, len(acc_values) + 1)

# 画曲线
plt.plot(echochs,acc_values,'bo',label='Train Accuracy')
plt.plot(echochs,val_acc_values,'ro',label='Test Accuracy')

#图标签设置
plt.title('Accuracy-Epochs Figure')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()#??


plt.show()

