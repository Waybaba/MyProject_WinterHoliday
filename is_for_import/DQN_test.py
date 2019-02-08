# encoding: utf-8
import math
import numpy as np
from keras import models
from keras import models
from keras import layers
from keras.utils import plot_model
from keras import optimizers
from keras import losses
from keras import metrics
from keras.layers import Input, Dense
from keras.models import Model
from keras.layers import TimeDistributed
import is_for_import.ntu_date_preprocess_2 as ntu
import is_for_import.ntu_skeleton_visualization as ntusee
import matplotlib.pyplot as plt
import os, time
import copy
import numpy as np
from keras.layers import Dense, Input, LSTM, RepeatVector,Concatenate
from keras.models import Model
import keras
from keras.optimizers import RMSprop ,adam
from keras import backend as K

np.random.seed(1)


os.environ['PATH'] = os.environ[
                         'PATH'] + ":/Users/Waybaba/anaconda3/envs/winter2/bin"  # 修改环境变量，因为绘图的时候要调用一个底层的命令，而那个命令因为一些错误没有装在系统命令下，所以在这里提前把路径加上，这是在winter2的conda环境下面，如果删除环境，也会导致出错



class DQN:
    def __init__(
            self,
            n_actions=30*3, # 最后的输出是n_actions维
            n_features=3,
            learning_rate=0.01, # 这是model网络的参数了。
            reward_decay=0.9, # 下一步的rewar的衰减值
            e_greedy=0.9, # e是按网络选择的概率，相当于探索新路和按网络走的比值了。然后这里有两个设置方法A：递增，那么e_greedy就是最大值，increment就是增长率，每次learn之后都会增加 B：固定值，如果increment是None，那么就固定在e_greedy
            e_greedy_increment=None,
            replace_target_iter=300, # 慢更的iter阀值次数
            memory_size=500,# 总的存储状态数
            batch_size=32,#每次抽的数量
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.learn_step_counter = 0

        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        self._build_net()

    def target_replace_op(self):
        v1 = self.model2.get_weights()
        self.model1.set_weights(v1)
        print("params has changed")

    def _build_net(self):
        # 构建evaluation网络
        eval_inputs = [Input(shape=(50,75,)),Input(shape=[30,75,]),Input(shape=[50,])]
        x = LSTM(units=64,return_sequences=0)(eval_inputs[0])
        x1 = Dense(units=50,activation='relu')(x)
        x = LSTM(units=64, return_sequences=0)(eval_inputs[1])
        x2 = Dense(units=50, activation='relu')(x)
        x_whole = Concatenate()([x1,x2,eval_inputs[2]])
        x_repeat = RepeatVector(n=30)(x_whole)
        self.q_eval = Dense(units=3,activation='softmax')(x_repeat)

        # 构建target网络，注意这个target层输出是q_next而不是，算法中的q_target
        target_inputs = [Input(shape=(50, 75,)), Input(shape=[30, 75, ]), Input(shape=[50, ])]
        x = LSTM(units=64, return_sequences=0)(target_inputs[0])
        x1 = Dense(units=50, activation='relu')(x)
        x = LSTM(units=64, return_sequences=0)(target_inputs[1])
        x2 = Dense(units=50, activation='relu')(x)
        x_whole = Concatenate()([x1, x2, target_inputs[2]])
        x_repeat = RepeatVector(30)(x_whole)
        self.q_next = Dense(units=3, activation='softmax')(x_repeat)

        self.model1 = Model(eval_inputs, self.q_eval)
        self.model2 = Model(target_inputs, self.q_next)
        rmsprop = RMSprop(lr=self.lr)
        self.model1.compile(loss='mean_squared_error', optimizer=rmsprop, metrics=['accuracy'])
        self.model2.compile(loss='mean_squared_error', optimizer=rmsprop, metrics=['accuracy'])
        plot_model(self.model1, to_file="model1_eval.png", show_layer_names=True, show_shapes=True)
        plot_model(self.model2, to_file="model1_target.png", show_layer_names=True, show_shapes=True)

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition  # memory是一个二维列表
        self.memory_counter += 1

    def choose_action(self, observation):
        observation = np.array(observation)
        observation = observation[np.newaxis, :]
        actions_value = self.model1.predict(observation)
        action_index = np.argmax(actions_value, axis=1)
        action = np.zeros(shape=(30,3))
        for i in range(action.shape[0]):
            if np.random.uniform() < self.epsilon:
                action[i][action_index[i]] = 1.0
            else:
                action[i][np.random.randint(0, 3)] = 1.0
        return action

    def learn(self):
        # 通过这个learn，可以总结出来，网络的搭建要求是
        # 1、输入是state（根据环境不同有不同的feature个数）
        # 2、内部有两个model，形式一样，一个快更新，一个慢更新
        # 3、有出来替换参数的接口
        # 4、输出是value表

        # 更新target参数（2是快更新的，1是要替换的暂存的）
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_replace_op()
            print('\ntarget_params_replaced\n')

        # 抽样
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)#choice可以在第一个para范围内抽出pata_2个数作为列表
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)#这里的选择是在没有超出时，就只在已经有的个数里面选
        batch_memory = self.memory[sample_index, :]
        s = batch_memory[:,0]
        a = batch_memory[:,1]
        r = batch_memory[:,2]
        s_= batch_memory[:,3]
        # 通过两个state分别计算出value表，q_next is the value of next,
        q_next = self.model1.predict(s_) # q_next是下一次的各动作value表，唯一的用处是结合reward制作出本次的target
        q_eval = self.model2.predict(s) # 因为要进行更新的就是eval快更新网络，所以要把当时的state放入制作出基准
        q_target = q_eval.copy()# 相当于是把model2（快更新的）的作为基准进行修改

        # 制作标签，精准定位修改
        batch_index = np.arange(self.batch_size, dtype=np.int32)# 这句和下句做出来两个列表，一个是次序索引，一个是行为索引，通过这两个，就可以精准更改一个history里面一个action对应的值（这个arrange类似于range，arrange是numpy里面的，而range是python里的）
        eval_act_index = a。.astype(int) #这一列是action，也就是说，记忆中的action决定了target要替换的位置
        reward = batch_memory[:, self.n_features + 1]# 这一列是reward，抽取reward
        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)#这句十分关键，总结起来是一句话：用下一时刻的慢更新网络的最大action期望（就是value表的最大值）来修正本时刻快更新网络的生成值。

        # 正常的拟合
        self.model2.fit(batch_memory[:, :self.n_features], q_target, epochs=10)
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
tem_net = DQN()
tem_net._build_net()