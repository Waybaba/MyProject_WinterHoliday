# encoding: utf-8
import math
import numpy as np
from keras import models
import copy


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
            "/Users/Waybaba/PycharmProjects/Machine_learning/MyProject/My_demo/0203_reinforcement_demo/test/model_backup/30_predict_model.h5")

    ### new epotch, return state
    def creat_new_epotch(self, whole_frames, lable):
        self.whole_frames = whole_frames
        self.current_mask = self.uniform_mask()
        self.chose_frames = self.choose_frame_from_mask(whole_frames=self.whole_frames, mask=self.current_mask)
        self.predict_target = lable
        self.history = []
        self.solo_reward_list = []
        self.history_pred_prob = []
        state = [self.whole_frames, self.chose_frames, self.current_mask]
        return state

    ### important! excute action and return s,action,reward,s_,action format 30x3
    def step(self, action):
        # change action format
        s = copy.deepcopy(
            [self.whole_frames, self.chose_frames, self.current_mask])  # this state #深拷贝才能完全复制，不过比较耗时，这好像是拷贝的唯一方法
        # mask update
        self.current_mask = self.update_mask_with_action(self.current_mask, action)
        # produce chose_frames
        self.chose_frames = self.choose_frame_from_mask(self.whole_frames, self.current_mask)
        # give back rewards
        reward = self.get_reward_from_frames(self.chose_frames)
        s_ = [self.whole_frames, self.chose_frames, self.current_mask]  # next state
        self.history.append([s, action, reward, s_])
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

