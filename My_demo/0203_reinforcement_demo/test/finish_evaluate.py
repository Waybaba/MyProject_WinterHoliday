# encoding: utf-8
"""
把数据列表和标签列表放到evaluate里面就可以得到准确度
"""
import math
import numpy as np
from keras import models
import is_for_import.ntu_date_preprocess_2 as ntu
import os
import copy



# Para
step_num = 50
DQN_model_path = "/Users/Waybaba/PycharmProjects/Machine_learning/MyProject/My_demo/0203_reinforcement_demo/test/model_backup/model1.h5"
predict_30_model_path="/Users/Waybaba/PycharmProjects/Machine_learning/MyProject/My_demo/0203_reinforcement_demo/test/model_backup/30_predict_model.h5"

# Net and Env
class frameChooseEnv:
    # -------------Env for Action Chose Recognization--------------------
    # in __init__, you can change settings such as lengh, personal model
    # main API:
    # creat_new_epotch(self,whole_frames,lable): when given date and lable, return state
    # step(self,action): excute action, return s,a,r,s_
    # -------------------------------------------------------------------

    def __init__(self, whole_length=50, target_length=30):
        self.whole_length = whole_length
        self.target_length = target_length
        self.predict_model = models.load_model(predict_30_model_path)

    ### new epotch, return state
    def creat_new_epotch(self, whole_frames, lable):
        self.whole_frames = whole_frames
        self.current_mask = self.uniform_mask()
        self.chose_frames = self.choose_frame_from_mask(whole_frames=self.whole_frames, mask=self.current_mask)
        self.predict_target = lable
        state = [self.whole_frames, self.chose_frames, self.current_mask]
        return state

    ### important! excute action and return s,action,reward,s_,action format 30x3
    def step(self, action):
        # change action format
        action_2D = np.zeros(shape=(self.target_length,3))
        x_axis = 0 if action % 2 ==0 else 2
        y_axis = int(action/2)
        action_2D[y_axis][x_axis] = 1.0
        s = copy.deepcopy([self.whole_frames, self.chose_frames, self.current_mask])  # this state #深拷贝才能完全复制，不过比较耗时，这好像是拷贝的唯一方法
        # mask update
        self.current_mask = self.update_mask_with_action(self.current_mask, action_2D)
        # produce chose_frames
        self.chose_frames = self.choose_frame_from_mask(self.whole_frames, self.current_mask)
        # give back rewards
        reward = self.get_reward_from_frames(self.chose_frames)
        s_ = [self.whole_frames, self.chose_frames, self.current_mask]  # next state
        return s, action, reward, s_

    # tools
    def choose_frame_from_mask(self, whole_frames, mask):
        result = np.empty(shape=(self.target_length, 75))
        index_record = 0
        for i in range(self.whole_length):
            if self.current_mask[i] == 1.0:
                result[index_record] = whole_frames[i]
                index_record += 1
        return result

    def update_mask_with_action_v1_old(self, mask, action):
        # 老的版本，跳的比较远
        # get list
        old_index = np.array([])
        for i in np.arange(self.whole_length):
            if mask[i] == 1.0:
                old_index = np.append(old_index, i)
        new_index = old_index.__deepcopy__()
        # a middle one: 0,1,2 dim 1 point to dif action
        for i in range(self.target_length):
            # normal
            if i > 0 and i < self.target_length - 1:
                if action[i][0] == 1:  # back
                    if ((index_i + old_index[i - 1]) / 2) - math.ceil((index_i + old_index[i - 1]) / 2) != 0:
                        new_index[i] = math.ceil((index_i + old_index[i - 1]) / 2)
                elif action[i][1] == 1:  # stay
                    new_index[i] = index_i
                elif action[i][2] == 1:  # forward
                    new_index[i] = math.floor((index_i + old_index[i + 1]) / 2)
            # first
            if i == 0:
                if action[i][0] == 1:  # back
                    new_index[i] = math.ceil((index_i) / 2)
                elif action[i][1] == 1:  # stay
                    new_index[i] = index_i
                elif action[i][2] == 1:  # forward
                    new_index[i] = math.floor((index_i + old_index[i + 1]) / 2)
            # last
            if i == self.target_length - 1:
                if action[i][0] == 1:  # back
                    if ((index_i + old_index[i - 1]) / 2) - math.ceil((index_i + old_index[i - 1]) / 2) != 0:
                        new_index[i] = math.ceil((index_i + old_index[i - 1]) / 2)
                elif action[i][1] == 1:  # stay
                    new_index[i] = index_i
                elif action[i][2] == 1:  # forward
                    new_index[i] = math.floor((index_i + self.whole_length - 1) / 2)
        # generate new mask
        new_mask = np.zeros(shape=(50))
        for each in new_index:
            new_mask[int(each)] = 1
        return new_mask

    def update_mask_with_action(self, mask, action):
        # 写的有点臃肿，这是每次移一格的，我觉得这样更好点，况且移动速度应该不是大问题。
        # get list
        old_index = np.array([])
        for i in np.arange(self.whole_length):
            if mask[i] == 1.0:
                old_index = np.append(old_index, i)
        new_index = old_index.copy()
        # a middle one: 0,1,2 dim 1 point to dif action
        for i in range(self.target_length):
            # normal
            index_i = int(old_index[i])
            if i > 0 and i < self.target_length - 1:
                if action[i][0] == 1:  # back
                    if mask[index_i - 1] == 0:
                        mask[index_i] = 0
                        mask[index_i - 1] = 1
                elif action[i][1] == 1:  # stay
                    new_index[i] = index_i
                elif action[i][2] == 1:  # forward
                    if mask[index_i + 1] == 0:
                        mask[index_i] = 0
                        mask[index_i + 1] = 1
            # first
            if i == 0:
                if index_i != 0:
                    if action[i][0] == 1:  # back
                        if mask[index_i - 1] == 0:
                            mask[index_i] = 0
                            mask[index_i - 1] = 1
                    elif action[i][1] == 1:  # stay
                        new_index[i] = index_i
                    elif action[i][2] == 1:  # forward
                        if mask[index_i + 1] == 0:
                            mask[index_i] = 0
                            mask[index_i + 1] = 1
                else:
                    if action[i][1] == 1:  # stay
                        new_index[i] = index_i
                    elif action[i][2] == 1:  # forward
                        if mask[index_i + 1] == 0:
                            mask[index_i] = 0
                            mask[index_i + 1] = 1
            # last
            if i == self.target_length - 1:
                if index_i != self.whole_length - 1:
                    if action[i][0] == 1:  # back
                        if mask[index_i - 1] == 0:
                            mask[index_i] = 0
                            mask[index_i - 1] = 1
                    elif action[i][1] == 1:  # stay
                        new_index[i] = index_i
                    elif action[i][2] == 1:  # forward
                        if mask[index_i + 1] == 0:
                            mask[index_i] = 0
                            mask[index_i + 1] = 1
                else:
                    if action[i][0] == 1:  # back
                        if mask[index_i - 1] == 0:
                            mask[index_i] = 0
                            mask[index_i - 1] = 1
                    elif action[i][1] == 1:  # stay
                        new_index[i] = index_i
        # generate new mask
        # new_mask = np.zeros(shape=(50))
        # for each in new_index:
        # new_mask[int(each)] = 1
        return mask

    def get_reward_from_frames(self, chose_frames):
        predict = self.predict_model.predict(x=chose_frames[np.newaxis, :])
        if np.argmax(predict) == np.argmax(self.predict_target):
            return 1.0
        else:
            return -1.0


    # others
    def uniform_mask(self):
        choose_index = np.arange(self.target_length)
        init_mask = np.zeros(shape=(50))
        for i in np.arange(self.target_length):
            choose_index[i] = math.floor(choose_index[i] * self.whole_length / self.target_length)
        for each in choose_index:
            init_mask[each] = 1
        return init_mask

    def predict(self,chose_frames):
        predict = self.predict_model.predict(x=chose_frames[np.newaxis, :])
        return predict
class DQN:
    def __init__(
            self,
            n_actions=30*2, # 最后的输出是n_actions维
            n_features=3,
            learning_rate=0.01, # 这是model网络的参数了。
            reward_decay=0.9, # 下一步的rewar的衰减值
            e_greedy=0.9, # e是按网络选择的概率，相当于探索新路和按网络走的比值了。然后这里有两个设置方法A：递增，那么e_greedy就是最大值，increment就是增长率，每次learn之后都会增加 B：固定值，如果increment是None，那么就固定在e_greedy
            e_greedy_increment=None,
            replace_target_iter=300, # 慢更的iter阀值次数
            memory_size=5000,# 总的存储状态数
            batch_size=1024,#每次抽的数量
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

        self.memory = [None]*self.memory_size

        self._build_net()

    def _build_net(self):
        self.model1 = models.load_model(filepath=DQN_model_path)

    def choose_action(self, observation):
        observation[0]=observation[0][np.newaxis, :]
        observation[1] = observation[1][np.newaxis, :]
        observation[2] = observation[2][np.newaxis, :]
        actions_value = self.model1.predict(x=observation)
        action = np.argmax(actions_value)
        return action

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
        # get action
        action = RL.choose_action(observation)
        # act
        observation,_ = env.step(action)
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

    evaluate(input_train[600:700],lable_train[600:700])
