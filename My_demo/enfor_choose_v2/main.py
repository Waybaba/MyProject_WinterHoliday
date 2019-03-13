# encoding: utf-8
from My_demo.enfor_choose_v2.env import frameChooseEnv
from My_demo.enfor_choose_v2.DQN import DQN
import is_for_import.ntu_date_preprocess_2 as ntu
import os,time
import numpy as np

os.environ['PATH']=os.environ['PATH']+":/Users/Waybaba/anaconda3/envs/winter2/bin"  #修改环境变量，因为绘图的时候要调用一个底层的命令，而那个命令因为一些错误没有装在系统命令下，所以在这里提前把路径加上，这是在winter2的conda环境下面，如果删除环境，也会导致出错

episode_total_num= 200000 # 一共考虑几个数据
step_in_each_episode = 30 # 每个数据步进几次

step_for_learning = step_in_each_episode*100 # 多少步学一次
step_for_validating = step_for_learning*10 # 多少步检测一次

memory_size = step_for_learning*3 # 总的存数据量，学习量的3倍吧，每次更新1/3

replace_target_iter = 2 # 学几次更新target网络




def run_env():
    total_step = 0
    # 300个episodes，从开始到最优解
    for episode in range(episode_total_num):
        # initial observation
        observation = env.creat_new_epotch(whole_frames=input_train[episode],lable=lable_train[episode])
        print("Pocessing on "+str(episode)+" data ...")
        for now_round_count_use in range(step_in_each_episode):

            # RL choose action
            action = RL.choose_action(observation)

            # Env take action
            observation,a,reward,observation_ = env.step(action)

            # Store
            RL.store_transition(observation, action, reward, observation_)

            # learning
            if (total_step > 1) and (total_step % step_for_learning == 0):
                print("learning ...",end='')
                RL.learn()
                print("FUNISH !!!")
            # validating
            if (total_step > 1) and (total_step % step_for_validating == 0):
                print("Validating ...",end='')
                validation(input_test[:100],lable_test[:100])
                print("FUNISH !!!")

            # swap observation
            observation = observation_
            total_step += 1

    # end of game
    print('over')

def validation(date_input,lable):
    env = frameChooseEnv()
    RL = DQN()
    date_num = date_input.shape[0]
    right_number = 0
    for i in range(date_num):
        predict = predict_once(input_frames=date_input[i], input_lable=lable[i], env=env, RL=RL)
        # print(predict)
        print("Predict argmax is : ", end="")
        print(np.argmax(predict))
        print("Lable argmax is : ", end="")
        print(np.argmax(lable_train[i]))
        if np.argmax(predict) == np.argmax(lable_train[i]):
            right_number += 1
    # print summary
    print("-------------------------")
    print("Date number :", end="")
    print(date_num, end="   ")
    print("Right number :", end="")
    print(right_number)
    print("Accuracy :", end="")
    print(right_number / date_num)
    print("-------------------------")
def predict_once(input_frames, input_lable, env, RL,step_num = 30):
    # init
    observation = env.creat_new_epotch(whole_frames=input_frames, lable=input_lable)
    for i in range(step_num):
        # get action # 记得去掉随机选择
        action = RL.choose_action(observation,random_switch = False)
        # act
        observation,d,e,f = env.step(action)
    # end of game
    predict = env.predict(chose_frames=observation[1])
    return predict


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
    RL = DQN(n_actions=30*2, # 最后的输出是n_actions维
            learning_rate=0.01, # 这是model网络的参数了。
            reward_decay=0.9, # 下一步的rewar的衰减值
            e_greedy=0.9, # e是按网络选择的概率，相当于探索新路和按网络走的比值了。然后这里有两个设置方法A：递增，那么e_greedy就是最大值，increment就是增长率，每次learn之后都会增加 B：固定值，如果increment是None，那么就固定在e_greedy
            e_greedy_increment=0.05,
            replace_target_iter=replace_target_iter, # 慢更的iter阀值次数
            memory_size=memory_size,# 总的存储状态数
            batch_size=200,#每次抽的数量
                      )
    # env.after(100, run_maze)
    # env.mainloop()
    run_env()
    RL.end()

    # RL.plot_cost()

    # validation
    validation(input_test[:100],lable_test[:100])