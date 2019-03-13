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
os.environ['PATH'] = os.environ['PATH'] + ":/Users/Waybaba/anaconda3/envs/winter2/bin"  # 修改环境变量，因为绘图的时候要调用一个底层的命令，而那个命令因为一些错误没有装在系统命令下，所以在这里提前把路径加上，这是在winter2的conda环境下面，如果删除环境，也会导致出错

# Parameters
DQN_model_save_path = "model_backup/model1.h5"

class DQN:
    def __init__(
            self,
            n_actions=30*2, # 最后的输出是n_actions维
            n_features=3,
            learning_rate=0.01, # 这是model网络的参数了。
            reward_decay=0.9, # 下一步的rewar的衰减值
            e_greedy=0.9, # e是按网络选择的概率，相当于探索新路和按网络走的比值了。然后这里有两个设置方法A：递增，那么e_greedy就是最大值，increment就是增长率，每次learn之后都会增加 B：固定值，如果increment是None，那么就固定在e_greedy
            e_greedy_increment=None,
            replace_target_iter=5, # 慢更的iter阀值次数
            memory_size=5000,# 总的存储状态数
            batch_size=1024,#每次抽的数量
            train_epotch = 5
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
        self.memory = [None]*self.memory_size
        self.train_epotch = train_epotch
        self._build_net()
        self.__build_train_fn()

    def target_replace_op(self):
        v1 = self.model1.get_weights()
        self.model2.set_weights(v1)
        print("params has changed")

    def _build_net(self):
        # evaluation网络 快更新
        eval_inputs = [Input(shape=(50,75,)),Input(shape=[30,75,]),Input(shape=[50,])]
        x = LSTM(units=50,return_sequences=0)(eval_inputs[0])
        x1 = Dense(units=50,activation='relu')(x)
        # x1 = Dense(units=50,activation='relu')(x1)
        x = LSTM(units=30, return_sequences=0)(eval_inputs[1])
        x2 = Dense(units=50, activation='relu')(x)
        # x2 = Dense(units=50,activation='relu')(x2)
        x_whole = Concatenate()([x1,x2,eval_inputs[2]])
        # x_repeat = RepeatVector(n=30)(x_whole)
        # x_whole = layers.RepeatVector(n=30)(x_whole)
        x_whole = layers.Dense(units=90,activation=None)(x_whole)
        x_whole = layers.Reshape(target_shape=(30,3,))(x_whole)
        self.q_eval = layers.Dense(units=3,activation='softmax',use_bias=False)(x_whole)

        # target网络---注意这个target层输出是q_next而不是，算法中的q_target，慢更新
        target_inputs = [Input(shape=(50, 75,)), Input(shape=[30, 75, ]), Input(shape=[50, ])]
        x = LSTM(units=50, return_sequences=0)(target_inputs[0])
        x1 = Dense(units=50, activation='relu')(x)
        # x1 = Dense(units=50, activation='relu')(x1)
        x = LSTM(units=30, return_sequences=0)(target_inputs[1])
        x2 = Dense(units=50, activation='relu')(x)
        # x2 = Dense(units=50, activation='relu')(x2)
        x_whole = Concatenate()([x1, x2, target_inputs[2]])
        # x_repeat = RepeatVector(30)(x_whole)
        # x_whole = layers.RepeatVector(n=30)(x_whole)
        x_whole = layers.Dense(units=90, activation=None)(x_whole)
        x_whole = layers.Reshape(target_shape=(30, 3,))(x_whole)
        self.q_next = layers.Dense(units=3, activation='softmax', use_bias=False)(x_whole)


        self.model1 = Model(eval_inputs, self.q_eval) # 快
        self.model2 = Model(target_inputs, self.q_next) # 慢
        rmsprop = RMSprop(lr=self.lr)
        self.model1.compile(loss='mean_squared_error', optimizer=rmsprop, metrics=['accuracy'])
        self.model2.compile(loss='mean_squared_error', optimizer=rmsprop, metrics=['accuracy'])

        plot_model(self.model1, to_file="model1_eval.png", show_layer_names=True, show_shapes=True)
        plot_model(self.model2, to_file="model1_target.png", show_layer_names=True, show_shapes=True)

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = [s, a, r, s_]
        index = self.memory_counter % self.memory_size
        self.memory[index] = transition  # memory是一个二维列表
        if self.memory[index][3][0].shape != (50, 75):
            print(2000)
        self.memory_counter += 1

    def choose_action_v2(self, observation):
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

    def choose_action(self, observation,random_switch=True):
        # 返回动作，形状是30*3，1表示要执行的动作
        # observation = np.array(observation)
        # 如果不需要随机
        if random_switch == False:
            observation[0] = observation[0][np.newaxis, :]
            observation[1] = observation[1][np.newaxis, :]
            observation[2] = observation[2][np.newaxis, :]
            actions_value = self.model1.predict(x=observation) # 快
            action_index = np.argmax(actions_value, axis=-1)
            action_index = action_index[0]
            action = np.zeros(shape=(30, 3))
            for i in range(action_index.__len__()):
                action[i][action_index[i]] = 1
            return action
        if np.random.uniform() < self.epsilon:
            observation[0]=observation[0][np.newaxis, :]
            observation[1] = observation[1][np.newaxis, :]
            observation[2] = observation[2][np.newaxis, :]
            actions_value = self.model1.predict(x=observation)
            action_index = np.argmax(actions_value,axis=-1)
            action_index = action_index[0]
            action = np.zeros(shape=(30, 3))
            for i in range(action_index.__len__()):
                action[i][action_index[i]] = 1
        else:
            # 随机生成一个动作
            action = np.zeros(shape=(30,3))
            for i in range(30):
                action[i][np.random.randint(0,3)]=1
        return action

    # 输入是 input,action,reward，这不是一个完整的函数，只是通过这个在self里面创建了一个可以调用的self.__build_train_fn()
    def __build_train_fn(self):
        """Create a train function
        It replaces `model.fit(X, y)` because we use the output of model and use it for training.
        For example, we need action placeholder
        called `action_one_hot` that stores, which action we took at state `s`.
        Hence, we can update the same action.
        This function will create
        `self.train_fn([state, action_one_hot, discount_reward])`
        which would train the model.
        """
        action_prob_placeholder = self.model1.output
        action_onehot_placeholder = K.placeholder(shape=(None, 30,3),
                                                  name="action_label")
        discount_reward_placeholder = K.placeholder(shape=(None,),
                                                    name="discount_reward")

        action_prob = K.sum(K.sum(action_prob_placeholder * action_onehot_placeholder, axis=1),axis=1) # 这个没看懂
        log_action_prob = K.log(action_prob)

        loss = - log_action_prob * discount_reward_placeholder
        loss = K.mean(loss)

        adam = optimizers.Adam()

        updates = adam.get_updates(params=self.model1.trainable_weights,
                                   # constraints=[],
                                   loss=loss)

        self.train_fn = K.function(inputs=[self.model1.inputs[0],
                                           self.model1.inputs[1],
                                           self.model1.inputs[2],
                                           action_onehot_placeholder,
                                           discount_reward_placeholder],
                                   outputs=self.model1.outputs,
                                   updates=updates)



    def learn(self):
        # 通过这个learn，可以总结出来，网络的搭建要求是
        # 1、输入是state（根据环境不同有不同的feature个数）
        # 2、内部有两个model，形式一样，一个快更新，一个慢更新
        # 3、有出来替换参数的接口
        # 4、输出是value表

        # 更新target参数（2是快更新的，1是要替换的暂存的）
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_replace_op()
            print('target_params_replaced')

        # 抽样
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)#choice可以在第一个para范围内抽出pata_2个数作为列表
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)#这里的选择是在没有超出时，就只在已经有的个数里面选
        batch_memory = [None]*self.batch_size
        # 提取记忆里的数据
        s = []
        a = []
        r = []
        s_ = []
        s_whole = []
        s_part=[]
        s_mask = []
        s__whole=[]
        s__part=[]
        s__mask=[]
        for i in range(self.batch_size):
            batch_memory[i]=self.memory[sample_index[i]]
        for i in range(self.batch_size):
            # 莫名其妙，不知道为什么要进行的矫正
            if batch_memory[i][3][0].shape == (1,50,75):
                batch_memory[i][3][0] = batch_memory[i][3][0][0]
            if batch_memory[i][3][1].shape == (1,30,75):
                batch_memory[i][3][1] = batch_memory[i][3][1][0]
            if batch_memory[i][3][2].shape == (1,50):
                batch_memory[i][3][2] = batch_memory[i][3][2][0]
            s.append(batch_memory[i][0])
            a.append(batch_memory[i][1])
            r.append(batch_memory[i][2])
            s_.append(batch_memory[i][3])

            s_whole.append(batch_memory[i][0][0])
            s_part.append(batch_memory[i][0][1])
            s_mask.append(batch_memory[i][0][2])

            s__whole.append(batch_memory[i][3][0])
            s__part.append(batch_memory[i][3][1])
            s__mask.append(batch_memory[i][3][2])
        s_whole = np.array(s_whole)
        s_part = np.array(s_part)
        s_mask = np.array(s_mask)
        s__whole = np.array(s__whole)
        s__part = np.array(s__part)
        s__mask = np.array(s__mask)
        a=np.array(a)
        r=np.array(r)

        # 用数据制作出input
        reward = r  # 这一列是reward，抽取reward
        action = a
        # 正常的拟合
        self.train_fn([s_whole,s_part,s_mask, action, reward])
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def end(self):
        self.model1.save(DQN_model_save_path)


# tem_net = DQN()
# tem_net._build_net()