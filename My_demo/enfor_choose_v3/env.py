# encoding: utf-8
import math
import numpy as np
from keras import models
import copy

# 这里load的model是上一次的0203 test里面的

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
        self.predict_model = models.load_model(
            "/Users/Waybaba/PycharmProjects/Machine_learning/MyProject/My_demo/enfor_choose_v3/model_backup/30_predict_model.h5")

    ### new epotch, return state
    def creat_new_epotch(self, whole_frames, lable):
        self.whole_frames = whole_frames
        self.current_mask = self.uniform_mask()
        self.chose_frames = self.choose_frame_from_mask(whole_frames=self.whole_frames, mask=self.current_mask)
        self.predict_target = lable
        # 清空上一次的list
        self.data_list_without_reward = []
        self.data_list = []
        self.reward_list =[]
        self.prob_list = []
        self.rigth_flag_list = []
        state = [self.whole_frames, self.chose_frames, self.current_mask]


        ## store: caculate prob, store data and prob, 第一步不需要存data，因为data里面要包含动作和下一次的state，第二步再存
        self.prob_list.append(self.get_prob_now(self.chose_frames,label=self.predict_target)) # 第一个prob是一开始的uniform的prob，但是这里没有存date_without_reward,要注意他两差了一位，最后理论上prob比data多一个
        self.rigth_flag_list.append(self.get_right_flag_now(self.chose_frames)) # save right flag
        return state

    ### important! excute action and return s,action,reward,s_,action format 30x3
    def step(self, action):
        # s is old state, s_ is new state
        # backup old: copy to create old state
        s = copy.deepcopy(
            [self.whole_frames, self.chose_frames, self.current_mask])  # this state #深拷贝才能完全复制，不过比较耗时，这好像是拷贝的唯一方法
        # update new: mask update, get new frames, create new state
        self.current_mask = self.update_mask_with_action(self.current_mask, action)
        self.chose_frames = self.choose_frame_from_mask(self.whole_frames, self.current_mask)
        s_ = [self.whole_frames, self.chose_frames, self.current_mask]  # next state
        # store: caculate prob, store data and prob
        self.data_list_without_reward.append([s, action, s_]) # store data without reward
        self.prob_list.append(self.get_prob_now(self.chose_frames,label=self.predict_target)) # store prob
        self.rigth_flag_list.append(self.get_right_flag_now(self.chose_frames))  # save right flag
        return s, action, s_

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

    def update_mask_with_action_v2_old(self, mask, action):
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

    
    # version 3
    def get_prob_now(self,frames,label):
        predict = self.predict_model.predict(x=frames[np.newaxis, :])[0]
        prob = predict[np.argmax(label)]
        return prob
    def get_right_flag_now(self, chose_frames):
        predict = self.predict_model.predict(x=chose_frames[np.newaxis, :]) # 获取到最后一层的输出
        a = np.argmax(predict)
        b = np.argmax(self.predict_target)
        if np.argmax(predict) == np.argmax(self.predict_target):
            return 1.0
        else:
            return 0

    def get_reward_from_prob(self):
        stimilation = 5.0
        for i in range(len(self.data_list_without_reward)): #这里用这个来推算总的长度，因为对象内部没有step总个数
            # 有突变
            # 转正确
            if self.rigth_flag_list[i+1]==1 and self.rigth_flag_list[i]==0:
                self.reward_list.append(stimilation)
            # 转错误
            elif self.rigth_flag_list[i + 1] == 0 and self.rigth_flag_list[i] == 1:
                self.reward_list.append(-stimilation)
            # # 正确维持
            # elif self.rigth_flag_list[i + 1] == 1 and self.rigth_flag_list[i] == 1:
            #     self.reward_list.append(1.0)
            # 无跳变
            # 增加
            elif self.prob_list[i+1] > self.prob_list[i]:
                self.reward_list.append(1.0)
            # 减少
            elif self.prob_list[i+1] < self.prob_list[i]:
                self.reward_list.append(-1.0)
            # 不变
            else:
                self.reward_list.append(0.0)
        return self.reward_list


        pass

    def get_data_list(self):
        self.reward_list = self.get_reward_from_prob() # turn prob list into reward list
        for i in range(len(self.data_list_without_reward)): # iterate the whole list
            self.data_list.append([self.data_list_without_reward[i][0],
                                   self.data_list_without_reward[i][1],
                                   self.reward_list[i],
                                   self.data_list_without_reward[i][2]])
        return self.data_list




    def predict(self, chose_frames):
        predict = self.predict_model.predict(x=chose_frames[np.newaxis, :])
        return predict

    # others
    def uniform_mask(self):
        choose_index = np.arange(self.target_length)
        init_mask = np.zeros(shape=(50))
        for i in np.arange(self.target_length):
            choose_index[i] = math.floor(choose_index[i] * self.whole_length / self.target_length)
        for each in choose_index:
            init_mask[each] = 1
        return init_mask

# -----------------env test------------------------------
# input_train,lable_train,input_test,lable_test =ntu.load_date("4_actions")
# # ntusee.show_gif(input_train[1])
# input_train = input_train.reshape((-1,50,75))#数据整形，然后输
# input_test = input_test.reshape((-1,50,75))
# lable_train = ntu.change_into_muti_dim(lable_train)
# lable_test = ntu.change_into_muti_dim(lable_test)
# print('input_train shape:', input_train.shape)
# print('input_test shape:', input_test.shape)
# env = frameChooseEnv(whole_length=50,target_length=30)
# state = env.creat_new_epotch(whole_frames=input_train[0],lable=lable_train[0])
# temp_action = 59
# s,a,r,s_ = env.step(temp_action)
# print(s)
# print(a)
# print(r)
# print(s_)

