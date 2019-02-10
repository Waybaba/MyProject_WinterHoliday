# encoding: utf-8
from is_for_import.finish_env import frameChooseEnv
from is_for_import.DQN_test import DQN
import is_for_import.ntu_date_preprocess_2 as ntu
import os,time
from keras import models
from keras import layers
from keras.utils import plot_model
from keras import optimizers
from keras import losses
from keras import metrics
import is_for_import.ntu_date_preprocess_2 as ntu
import is_for_import.ntu_skeleton_visualization as ntusee
import matplotlib.pyplot as plt
import os,time
import numpy as np

os.environ['PATH']=os.environ['PATH']+":/Users/Waybaba/anaconda3/envs/winter2/bin"  #修改环境变量，因为绘图的时候要调用一个底层的命令，而那个命令因为一些错误没有装在系统命令下，所以在这里提前把路径加上，这是在winter2的conda环境下面，如果删除环境，也会导致出错


class ModelChoose:
    def __init__(self):
        self.model_path = "model_backup/Model_Choice.h5"
        self.model = self.__build_net__()
        self.target_length = 20
    def __build_net__(self):
        # 构建模型/网络
        model = models.Sequential()
        # model.add(layers.SimpleRNN(
        #     # unroll=True,
        #     units=16,
        #     batch_input_shape=(None, 50, 75),#75相当于特征数量，或者也可以叫维度，这里有3*25个坐标点，要提前整形
        #     return_sequences=0
        # ))
        model.add(layers.LSTM(
            units=64,
            batch_input_shape=(None, 50, 75),  # 75相当于特征数量，或者也可以叫维度，这里有3*25个坐标点，要提前整形
            return_sequences=0  # 是否返回整个列表，用于再嵌套LSTM
        ))
        # model.add(layers.Dense(16, activation='relu'))  # 这个shape好像是抛去了数据的第一维度，认为这个
        # model.add(layers.Dense(16, activation='relu'))
        # model.add(layers.Dense(8, activation='relu'))
        # model.add(layers.Dense(4, activation='relu'))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(50, activation='sigmoid'))
        model.summary()
        # plot_model(model, to_file='model.png')
        # 编译步骤
        model.compile(optimizer='rmsprop',
                      # loss='binary_crossentropy',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        plot_model(model, to_file="model_image/choose_model.png", show_layer_names=True, show_shapes=True)
        return model

    def choose_frame(self,data):
        predict_result = self.model.predict(x=data)
        data_20 = np.empty(shape=(data.shape[0], 20, 75))
        choose_mask = np.empty(shape=(data.shape[0],50))
        for date_count in range(data.shape[0]):
            # to 01
            current_date = data[date_count]
            current_predict = predict_result[date_count]
            # index = np.random.choice(a=np.arange(50), size=20, replace=False)
            # # index_01 = np.zeros(shape=(50))
            # # index_01[index] = 1.0
            # date_20[date_count] = current_date[index]
            b = np.argsort(current_predict)
            index_max = b[-3:]
            index_01 = np.zeros(shape=(50))
            index_01[index_max] = 1.0
            result = np.empty(shape=(self.target_length,75))
            index_record = 0
            for i in range(50):
                if index_01[i] == 1.0:
                    result[index_record] = current_date[i]
                    index_record += 1
            data_20[date_count] = result
            choose_mask[date_count] = index_01
            # choose from 01
        return data_20,choose_mask

    def choose_frame_from_mask(self,frames,current_mask):
        result = np.empty(shape=(self.target_length, 75))
        index_record = 0
        for i in range(50):
            if current_mask[i] == 1.0:
                result[index_record] = frames[i]
                index_record += 1
        return result

    def train(self,date,label,epotch=5):
        self.model.fit(x=date,y=label,batch_size=32,epochs=epotch)




class ModelPredict20:
    def __init__(self):
        self.model_path = "model_backup/Model_Predict20.h5"
        self.model = self.__build_net__()
        self.target_length = 20
    def __build_net__(self):
        # 构建模型/网络
        model = models.Sequential()
        # model.add(layers.SimpleRNN(
        #     # unroll=True,
        #     units=16,
        #     batch_input_shape=(None, 50, 75),#75相当于特征数量，或者也可以叫维度，这里有3*25个坐标点，要提前整形
        #     return_sequences=0
        # ))
        model.add(layers.LSTM(
            units=64,
            batch_input_shape=(None, 20, 75),  # 75相当于特征数量，或者也可以叫维度，这里有3*25个坐标点，要提前整形
            return_sequences=0  # 是否返回整个列表，用于再嵌套LSTM
        ))
        # model.add(layers.Dense(16, activation='relu'))  # 这个shape好像是抛去了数据的第一维度，认为这个
        # model.add(layers.Dense(16, activation='relu'))
        # model.add(layers.Dense(8, activation='relu'))
        # model.add(layers.Dense(4, activation='relu'))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(40, activation='softmax'))
        model.summary()
        # plot_model(model, to_file='model.png')
        # 编译步骤
        model.compile(optimizer='rmsprop',
                      # loss='binary_crossentropy',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        plot_model(model, to_file="model_image/predic_model.png", show_layer_names=True, show_shapes=True)
        return model

    def get_right_flag(self,data,label):
        self.model.evaluate(x=data,y=label)
        predict = self.model.predict(x=data)
        predict_max = predict.argmax(axis=1)
        label_max = label.argmax(axis=1)
        total_num = label.shape[0]
        right_flag = np.zeros(shape=(total_num))
        for i in range(total_num):
            if predict_max[i] == label_max[i]:
                right_flag[i] = 1.0
            else:
                right_flag[i] = -1.0
        return right_flag

    def train_by_random_choose(self,data,label,epotch=5):
        date_20 = np.empty(shape=(data.shape[0],20,75))
        for date_count in range(data.shape[0]):
            current_date = data[date_count]
            index = np.random.choice(a=np.arange(50), size=20, replace=False)
            # index_01 = np.zeros(shape=(50))
            # index_01[index] = 1.0
            date_20[date_count] = current_date[index]
        self.train(date_20,label,epotch)

    def evalue_by_random_choose(self,data,label):
        date_20 = np.empty(shape=(data.shape[0], 20, 75))
        for date_count in range(data.shape[0]):
            current_date = data[date_count]
            index = np.random.choice(a=np.arange(50), size=20, replace=False)
            # index_01 = np.zeros(shape=(50))
            # index_01[index] = 1.0
            date_20[date_count] = current_date[index]
        self.model.evaluate(date_20, label)

    def train(self,date,label,epotch=5):
        self.model.fit(x=date,y=label,batch_size=32,epochs=epotch,validation_split=0.1)





if __name__ == "__main__":
    # load date
    input_train, lable_train, input_test, lable_test = ntu.load_date("40_actions")
    input_train = input_train.reshape((-1, 50, 75))
    input_test = input_test.reshape((-1, 50, 75))
    lable_train = ntu.change_into_muti_dim(lable_train)
    lable_test = ntu.change_into_muti_dim(lable_test)
    print('input_train shape:', input_train.shape)
    print('input_test shape:', input_test.shape)

    # init
    model_choose = ModelChoose()
    model_predict_20 = ModelPredict20()

    # train model_predict randomly
    model_predict_20.train_by_random_choose(input_train,lable_train,epotch=1)
    model_predict_20.train_by_random_choose(input_train, lable_train, epotch=1)
    eval_result = model_predict_20.evalue_by_random_choose(input_train,lable_train)
    print(eval_result)
    # turn 50 to 20, get mask_now
    data_20,mask_now = model_choose.choose_frame(input_train)
    # get right flag
    right_flag = model_predict_20.get_right_flag(data_20,label=lable_train)
    print(right_flag)
    # date_50 is input, mask_now is label, right flag help to rejust loss
    # train model_choose

    # round_of_dule = 2
    # for i in range(round_of_dule):
        # turn 50 to 20, get date_20
        # train model_predict with date_20 and label

        # turn 50 to 20, get mask_now
        # get right flag
        # date_50 is input, mask_now is label, right flag help to rejust loss
        # train model_choose

