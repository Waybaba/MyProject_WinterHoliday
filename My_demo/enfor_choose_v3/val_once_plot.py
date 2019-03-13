# encoding: utf-8
"""
把数据列表和标签列表放到evaluate里面就可以得到准确度
"""
import math
import numpy as np
from keras import models
import is_for_import.ntu_date_preprocess_2 as ntu
import os
from My_demo.enfor_choose_v2.env import frameChooseEnv
from My_demo.enfor_choose_v2.DQN import DQN
import matplotlib.pyplot as plt
import copy



# Para
step_num = 20
DQN_model_path = "/Users/Waybaba/PycharmProjects/Machine_learning/MyProject/My_demo/0203_reinforcement_demo/test/model_backup/model1.h5"
predict_30_model_path="/Users/Waybaba/PycharmProjects/Machine_learning/MyProject/My_demo/enfor_choose_v2/model_backup/model1.h5"



# Function
def evaluate(date_input,lable):
    env = frameChooseEnv()
    RL = DQN()
    date_num = date_input.shape[0]
    right_number = 0
    for i in range(date_num):
        predict = predict_once(input_frames=date_input[i],input_lable=lable[i],env=env,RL=RL)
        # print(predict)
        print("Predict argmax is : ", end="")
        print(np.argmax(predict))
        print("Lable argmax is : ", end="")
        print(np.argmax(lable[i]))
        if np.argmax(predict) == np.argmax(lable[i]):
            right_number += 1
    # print summary
    print("-------------------------")
    print("Date number :", end="")
    print(date_num,end="   ")
    print("Right number :", end="")
    print(right_number)
    print("Accuracy :", end="")
    print(right_number/date_num)
    print("-------------------------")
def predict_once(input_frames,input_lable,env,RL):
    # init
    observation = env.creat_new_epotch(whole_frames=input_frames,lable=input_lable)
    for i in range(step_num):
        # RL choose action
        action = RL.choose_action(observation)

        # Env take action
        observation, a, reward, observation_ = env.step(action)

    # end of game
    predict = env.predict(chose_frames=observation[1])
    return predict


if __name__ == "__main__":
    np.random.seed(1)
    os.environ['PATH'] = os.environ['PATH'] + ":/Users/Waybaba/anaconda3/envs/winter2/bin"  # 修改环境变量，因为绘图的时候要调用一个底层的命令，而那个命令因为一些错误没有装在系统命令下，所以在这里提前把路径加上，这是在winter2的conda环境下面，如果删除环境，也会导致出错
    # load_date
    input_train,lable_train,input_test,lable_test =ntu.load_date("40_actions")
    # ntusee.show_gif(input_train[1])
    input_train = input_train.reshape((-1,50,75))#数据整形，然后输
    input_test = input_test.reshape((-1,50,75))
    lable_train = ntu.change_into_muti_dim(lable_train)
    lable_test = ntu.change_into_muti_dim(lable_test)
    print('input_train shape:', input_train.shape)
    print('input_test shape:', input_test.shape)

    i = 204
    env = frameChooseEnv()
    RL = DQN()
    predict = predict_once(input_frames=input_train[i], input_lable=lable_train[i], env=env, RL=RL)

    print(predict)

    x_axis = np.linspace(0,40, 40)
    print(lable_train[i].argmax())
    predict = predict.flatten()
    plt.plot(x_axis, predict, 'r-',label="prob")
    # plt.plot(xnew, power_smooth, 'b-', label='loss')
    # plt.plot(x,show_activity_rate(old_sequence),"g-",label="activity")
    plt.title('Prob - Action')
    plt.xlabel('Action')
    plt.ylabel('Prob')
    plt.legend()
    plt.show()
