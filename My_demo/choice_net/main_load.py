# encoding: utf-8
from is_for_import.finish_env import frameChooseEnv
from is_for_import.DQN_test import DQN
import is_for_import.ntu_date_preprocess_2 as ntu
import os,time
from keras import models
from keras import layers
import keras.backend as K
from keras import Model
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
        self.model_path = "model_backup/"
        self.model_choose,self.model_for_train = self.__build_net__()
        self.target_length = 20

    def __build_net__(self):
        # 构建模型/网络
        inputs_data = layers.Input(shape=(50,75,))
        inputs_reward = layers.Input(shape=(1,))
        inputs_target = layers.Input(shape=(50,))

        x = layers.LSTM(units=128,batch_input_shape=(None, 50, 75), return_sequences=0)(inputs_data)
        x = layers.Dense(128, activation='relu')(x)
        predictions = layers.Dense(50, activation='sigmoid')(x)

        model_choose = Model(inputs=inputs_data,outputs=predictions)

        def lambda_fuc_1(inputx):
            predictions,inputs_target,inputs_reward = inputx
            result = K.mean(K.square(predictions-inputs_target))*(inputs_reward)
            # return K.flatten(inputs_reward+1)
            return result
        # output_loss = layers.Lambda(function=lambda_fuc_1)([predictions, inputs_target, inputs_reward])
        output_loss = predictions
        # output_loss = layers.Lambda(lambda x: ((x[0]-x[1])**2)*x[2])([predictions, inputs_target, inputs_reward])

        model_for_train = Model(inputs=[inputs_data,inputs_target,inputs_reward],outputs=[output_loss,predictions])


        # choose_index = choose_index_model(input_whole)

        model_choose.summary()
        model_for_train.summary()

        plot_model(model_choose, to_file="model_image/choose_model.png", show_layer_names=True, show_shapes=True)
        plot_model(model_for_train, to_file="model_image/choose_model_for_trian.png", show_layer_names=True, show_shapes=True)

        # 编译步骤
        def lambda_x(y_true, y_pred):
            return y_pred
        def lambda_y(y_true, y_pred):
            result = K.zeros_like(y_pred)
            return result
        model_choose.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy']
                      )
        losses = [
            lambda_x,  # loss is computed in Lambda layer
            lambda_y # we only include this for the metrics
        ]
        model_for_train.compile(
            # optimizer='rmsprop',
            optimizer=optimizers.Adam(lr=0.01),
                      loss = losses,
                    # metrics=['accuracy']
                     )
        return model_choose,model_for_train

    def choose_frame(self,data):
        predict_result = self.model_for_train.predict(x=[data,np.zeros(shape=(data.shape[0],50)),np.zeros(shape=(data.shape[0],1))])[1]
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
            index_max = b[-20:]
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

    def train(self,data,target,reward,epotch=5):
        print("train model_choose ...",end="")
        self.model_for_train.fit(x=[data,target,reward],y=[reward,target],batch_size=32,epochs=epotch,verbose=main_verbose)
        print("  funish!")

    def end(self,save_name=""):
        self.model_choose.save(filepath=self.model_path+save_name+"model_choose.h5")
        self.model_for_train.save(filepath=self.model_path+save_name+"model_for_train.h5")
        print("")
        print("Model choose save successfully !")
        print("model_choose save at : "+self.model_path+save_name+"model_choose.h5")
        print("model_for_train save at : "+self.model_path +save_name+"model_for_train.h5")
    def load_model(self,load_name=""):
        def lambda_x(y_true, y_pred):
            return y_pred
        def lambda_y(y_true, y_pred):
            result = K.zeros_like(y_pred)
            return result
        self.model_for_train = models.load_model(filepath=self.model_path + load_name + "model_for_train.h5",custom_objects={"lambda_x":lambda_x,"lambda_y":lambda_y})
        self.model_choose = models.load_model(filepath=self.model_path+load_name+"model_choose.h5")
        print("load choose model from: " + self.model_path + load_name + "model_choose.h5")
        print("load choose_for_trian model from: " + self.model_path + load_name + "model_choose.h5")

class ModelPredict20:
    def __init__(self):
        self.model_path = "model_backup/"
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
            units=128,
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
        predict = self.model.predict(x=data)
        predict_max = np.argmax(predict,axis=-1) # -1或者1都可以，-1代表最后一轴
        label_max = np.argmax(label,axis=-1)
        total_num = label.shape[0]
        right_flag = np.zeros(shape=(total_num))
        right_num = 0
        for i in range(total_num):
            if predict_max[i] == label_max[i]:
                right_flag[i] = 1.0
                right_num += 1
            else:
                right_flag[i] = -1.0
        print("Right rate is: ",end="")
        print(right_num/total_num)
        return right_flag
    def train_by_random_choose(self,data,label,epotch=5):
        data_20 = np.empty(shape=(data.shape[0],20,75))
        for date_count in range(data.shape[0]):
            current_date = data[date_count]
            index = np.random.choice(a=np.arange(50), size=20, replace=False)
            # index_01 = np.zeros(shape=(50))
            # index_01[index] = 1.0
            data_20[date_count] = current_date[index]
        self.train(data_20,label,epotch)

    def train(self,data,label,epotch=5):
        print("")
        print("train model_predict ...",end="")
        history = self.model.fit(x=data,y=label,batch_size=32,epochs=epotch,validation_split=0.1,verbose=main_verbose)###
        print("  funish!")
        print("loss: ",end="")
        print(history.history['loss'][epotch-1],end="   ")
        print("accuracy: ",end="")
        print(history.history['acc'][epotch-1])

    def evalue_random_50choose(self,data,label):
        date_20 = np.empty(shape=(data.shape[0], 20, 75))
        for date_count in range(data.shape[0]):
            current_date = data[date_count]
            index = np.random.choice(a=np.arange(50), size=20, replace=False)
            # index_01 = np.zeros(shape=(50))
            # index_01[index] = 1.0
            date_20[date_count] = current_date[index]
        random_result =  self.model.evaluate(date_20, label,verbose=main_verbose)
        print("------------------Random Evalue------------------")
        print("Loss: ", end="")
        print(random_result[0], end="    ")
        print("Accuracy: ", end="")
        print(random_result[1])
    def evalue(self,data,label):
        result = self.model.evaluate(data,label,verbose=main_verbose)
        print("------------------Choose Evalue------------------")
        print("Loss: ",end="")
        print(result[0],end="    ")
        print("Accuracy: ",end="")
        print(result[1])

    def end(self,save_name=""):
        self.model.save(filepath=self.model_path+save_name+"model_predict.h5")
        print("model_for_train save at : "+self.model_path +save_name+ "model_predict.h5")
    def load_model(self,load_name=""):
        self.model = models.load_model(filepath=self.model_path+load_name+"model_predict.h5")
        print("load predict model from: " + self.model_path+load_name+"model_predict.h5")


def get_random_data(data,label,size):
    random_start = int(np.random.random()*(data.shape[0]-size-1))
    random_over = random_start+size
    return data[random_start:random_over],label[random_start:random_over]

choose_train_once_num = 256
choose_train_times = 20
main_verbose = False
load_name = "test"
save_name = "test"

if __name__ == "__main__":
    # load date
    # input_train, lable_train, input_test, lable_test = ntu.load_date("40_actions")
    # input_train = input_train.reshape((-1, 50, 75))
    # input_test = input_test.reshape((-1, 50, 75))
    # lable_train = ntu.change_into_muti_dim(lable_train)
    # lable_test = ntu.change_into_muti_dim(lable_test)
    # print('input_train shape:', input_train.shape)
    # print('input_test shape:', input_test.shape)
    #
    # # init
    model_choose = ModelChoose()
    model_predict_20 = ModelPredict20()
    model_choose.load_model(load_name)
    model_predict_20.load_model(load_name)
    #
    # round_of_dule = 3
    # for round_count in range(round_of_dule):
    #     # train choose_model for ... rounds
    #     for i in range(choose_train_times):
    #         # choose data
    #         chose_data,chose_label = get_random_data(input_train,lable_train,size=choose_train_once_num)
    #         # turn to 20, get flag, train
    #         chose_data_20,chose_mask_now = model_choose.choose_frame(data=chose_data)
    #         chose_right_flag = model_predict_20.get_right_flag(data=chose_data_20,label=chose_label)
    #         model_choose.train(data=chose_data,target=chose_mask_now,reward=chose_right_flag,epotch=1)
    #     # trian predict model
    #     data_20,_ = model_choose.choose_frame(input_train)
    #     # model_predict_20.train(data=data_20,label=lable_train,epotch=2)
    #
    #     # eval
    #     print("--------------------------------------------")
    #     print("Duel Train Round "+str(round_count)+" Over !")
    #     model_predict_20.evalue(data=data_20,label=lable_train)
    #     model_predict_20.evalue_random_50choose(data=input_train,label=lable_train)
    model_choose.end(save_name)
    model_predict_20.end(save_name)



    # evaluate
    # model_predict_20.evalue_random_50choose(input_train, lable_train)
    # model_predict_20.evalue(data_20,label=lable_train)

