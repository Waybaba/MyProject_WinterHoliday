# encoding: utf-8
import is_for_import.ntu_date_preprocess_2 as ntu
import os,time
import numpy as np
import matplotlib.pyplot as plt
from keras import layers
from keras import models
from keras.utils import plot_model
os.environ['PATH']=os.environ['PATH']+":/Users/Waybaba/anaconda3/envs/winter2/bin"  #修改环境变量，因为绘图的时候要调用一个底层的命令，而那个命令因为一些错误没有装在系统命令下，所以在这里提前把路径加上，这是在winter2的conda环境下面，如果删除环境，也会导致出错



def change_into_predict_data(video_data,frames_length=3):
    total_video_num = video_data.shape[0]
    result_frames_list = []
    result_label_list = []
    for video_index in range(total_video_num):
        current_video = video_data[video_index]
        for frame_index in range(50 - frames_length):
            result_frames_list.append(current_video[frame_index:frame_index + frames_length])
            result_label_list.append(current_video[frame_index + frames_length])
    result_frames_list = np.array(result_frames_list)
    result_label_list = np.array(result_label_list)
    print(result_frames_list.shape)
    print(result_label_list.shape)
    return result_frames_list, result_label_list


frames_length = 3
if __name__ == "__main__":
    # load_date
    input_train,lable_train,input_test,lable_test =ntu.load_date("40_actions")
    # ntusee.show_gif(input_train[1])
    input_train = input_train.reshape((-1,50,75))#数据整形，然后输
    input_test = input_test.reshape((-1,50,75))
    lable_train = ntu.change_into_muti_dim(lable_train)
    lable_test = ntu.change_into_muti_dim(lable_test)

    frames_list,label_list = change_into_predict_data(input_train,frames_length=frames_length)

    # 构建模型/网络
    sequence_input = layers.Input(shape=(frames_length, 75))
    # # x = layers.Flatten()(sequence_input)
    # x = layers.LSTM(units=25,batch_input_shape=(None,frames_length,75),return_sequences=True)(sequence_input)
    x = layers.LSTM(units=75,return_sequences=False)(sequence_input)
    # x = layers.LSTM(units=25, return_sequences=True)(x)
    # x = layers.LSTM(units=25, return_sequences=False)(x)
    x = layers.Dense(units=75)(x)
    model = models.Model(inputs=sequence_input,outputs=x)

    model.summary()
    model.compile(optimizer='rmsprop',
                  # loss='binary_crossentropy',
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    # 开始
    history = model.fit(
        frames_list,label_list,
        epochs=1,
        batch_size=16,
        validation_split=0.2,
    )

    plot_model(model, to_file="model_test_png", show_layer_names=True, show_shapes=True)

    # 开始

    localtime = time.asctime(time.localtime(time.time()))
    localtime = localtime.replace(" ", "_")
    model.save("model_backup/" + "predict_net" + ".h5")

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


    '''
    loss记录: lstm 大概是在0.0044 换成256之后0.0041  虽然这里不存在长时运算，只有俩帧，但是lstm的记忆结构可以帮助预测，这是我对lstm表现的理解
            rnn 128unit 0.0079
            rnn 256unit 0.0116
            dense 0.0059
            
                
    '''

    # save frames date, label