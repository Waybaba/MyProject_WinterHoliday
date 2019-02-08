import numpy as np
from keras.layers import Dense, Input
from keras.models import Model
import keras
from keras.optimizers import RMSprop ,adam
from keras import backend as K

np.random.seed(1)


class DQN:
    def __init__(
            self,
            n_actions, # 最后的输出是n_actions维
            n_features,
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
        eval_inputs = Input(shape=(self.n_features,))
        x = Dense(64, activation='relu')(eval_inputs)
        x = Dense(64, activation='relu')(x)
        self.q_eval = Dense(self.n_actions)(x)
        # 构建target网络，注意这个target层输出是q_next而不是，算法中的q_target
        target_inputs = Input(shape=(self.n_features,))
        x = Dense(64, activation='relu')(target_inputs)
        x = Dense(64, activation='relu')(x)
        self.q_next = Dense(self.n_actions)(x)

        self.model1 = Model(target_inputs, self.q_next)
        self.model2 = Model(eval_inputs, self.q_eval)
        rmsprop = RMSprop(lr=self.lr)
        self.model1.compile(loss='mean_squared_error', optimizer=rmsprop, metrics=['accuracy'])
        self.model2.compile(loss='mean_squared_error', optimizer=rmsprop, metrics=['accuracy'])

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
        if np.random.uniform() < self.epsilon:
            actions_value = self.model1.predict(observation)
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
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

        # 通过两个state分别计算出value表，q_next is the value of next,
        q_next = self.model1.predict(batch_memory[:, -self.n_features:]) # q_next是下一次的各动作value表，唯一的用处是结合reward制作出本次的target
        q_eval = self.model2.predict(batch_memory[:, :self.n_features] ) # 因为要进行更新的就是eval快更新网络，所以要把当时的state放入制作出基准
        q_target = q_eval.copy()# 相当于是把model2（快更新的）的作为基准进行修改

        # 制作标签，精准定位修改
        batch_index = np.arange(self.batch_size, dtype=np.int32)# 这句和下句做出来两个列表，一个是次序索引，一个是行为索引，通过这两个，就可以精准更改一个history里面一个action对应的值（这个arrange类似于range，arrange是numpy里面的，而range是python里的）
        eval_act_index = batch_memory[:, self.n_features].astype(int) #这一列是action，也就是说，记忆中的action决定了target要替换的位置
        reward = batch_memory[:, self.n_features + 1]# 这一列是reward，抽取reward
        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)#这句十分关键，总结起来是一句话：用下一时刻的慢更新网络的最大action期望（就是value表的最大值）来修正本时刻快更新网络的生成值。

        # 正常的拟合
        self.model2.fit(batch_memory[:, :self.n_features], q_target, epochs=10)
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
