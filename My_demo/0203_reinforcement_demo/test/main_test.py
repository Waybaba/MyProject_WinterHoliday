# encoding: utf-8
from is_for_import.env_test import frameChooseEnv
from is_for_import.DQN_test import DQN
import time

import math
import numpy as np
from keras import models
from keras import models
from keras import layers
from keras.utils import plot_model
from keras import optimizers
from keras import losses
from keras import metrics
from keras.layers import Input,Dense
from keras.models import Model
from keras.layers import TimeDistributed
import is_for_import.ntu_date_preprocess_2 as ntu
import is_for_import.ntu_skeleton_visualization as ntusee
import matplotlib.pyplot as plt
import os,time
import copy
os.environ['PATH']=os.environ['PATH']+":/Users/Waybaba/anaconda3/envs/winter2/bin"  #修改环境变量，因为绘图的时候要调用一个底层的命令，而那个命令因为一些错误没有装在系统命令下，所以在这里提前把路径加上，这是在winter2的conda环境下面，如果删除环境，也会导致出错


def run_env():
    step = 0
    # 300个episodes，从开始到最优解
    for episode in range(300):
        # initial observation
        observation = env.creat_new_epotch(whole_frames=input_train[episode],lable=lable_train[episode])
        time_count = 0
        while True:
            # fresh env
            # env.render()

            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            # observation_, reward, done = env.step(action)
            observation,a,reward,observation_ = env.step(action)
            if observation_[0].shape !=(50,75):
                print(209)
            RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 20 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            time_count += 1
            if time_count>50:
                time_count = 0
                break
            step += 1

    # end of game
    print('over')


if __name__ == "__main__":
    # load_date
    input_train,lable_train,input_test,lable_test =ntu.load_date("40_actions")
    # ntusee.show_gif(input_train[1])
    input_train = input_train.reshape((-1,50,75))#数据整形，然后输
    input_test = input_test.reshape((-1,50,75))
    lable_train = ntu.change_into_muti_dim(lable_train)
    lable_test = ntu.change_into_muti_dim(lable_test)
    print('input_train shape:', input_train.shape)
    print('input_test shape:', input_test.shape)

    # maze game
    env = frameChooseEnv()
    RL = DQN(learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=10,
                      memory_size=4000,
                      # output_graph=True
                      )
    # env.after(100, run_maze)
    # env.mainloop()
    run_env()
    localtime = time.asctime(time.localtime(time.time()))
    localtime = localtime.replace(" ", "_")
    model.save("model_backup/" + localtime + ".h5")
    # RL.plot_cost()