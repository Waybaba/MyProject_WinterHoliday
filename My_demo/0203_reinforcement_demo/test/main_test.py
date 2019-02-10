# encoding: utf-8
from is_for_import.finish_env import frameChooseEnv
from is_for_import.DQN_test import DQN
import is_for_import.ntu_date_preprocess_2 as ntu
import os,time

os.environ['PATH']=os.environ['PATH']+":/Users/Waybaba/anaconda3/envs/winter2/bin"  #修改环境变量，因为绘图的时候要调用一个底层的命令，而那个命令因为一些错误没有装在系统命令下，所以在这里提前把路径加上，这是在winter2的conda环境下面，如果删除环境，也会导致出错


def run_env():
    step = 0
    # 300个episodes，从开始到最优解
    for episode in range(100):
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

            RL.store_transition(observation, action, reward, observation_)

            if (step > 6400) and (step % 1000 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            time_count += 1
            if time_count>200:
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
    RL = DQN(n_actions=30*2, # 最后的输出是n_actions维
            learning_rate=0.01, # 这是model网络的参数了。
            reward_decay=0.9, # 下一步的rewar的衰减值
            e_greedy=0.9, # e是按网络选择的概率，相当于探索新路和按网络走的比值了。然后这里有两个设置方法A：递增，那么e_greedy就是最大值，increment就是增长率，每次learn之后都会增加 B：固定值，如果increment是None，那么就固定在e_greedy
            e_greedy_increment=0.05,
            replace_target_iter=100, # 慢更的iter阀值次数
            memory_size=200*10,# 总的存储状态数
            batch_size=200,#每次抽的数量
                      )
    # env.after(100, run_maze)
    # env.mainloop()
    run_env()
    RL.end()
    # RL.plot_cost()