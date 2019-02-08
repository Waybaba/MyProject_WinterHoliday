from My_demo.Tensorflow_demo.reinforcement_demo.maze_env import Maze
from My_demo.Tensorflow_demo.reinforcement_demo.blogs import DQN
import time

def run_maze():
    step = 0
    # 300个episodes，从开始到最优解
    for episode in range(300):
        # initial observation
        observation = env.reset()

        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 10 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1

    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    # maze game
    env = Maze()
    RL = DQN(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=10,
                      memory_size=4000,
                      # output_graph=True
                      )
    env.after(100, run_maze)
    env.mainloop()
    localtime = time.asctime(time.localtime(time.time()))
    localtime = localtime.replace(" ", "_")
    model.save("model_backup/" + localtime + ".h5")
    # RL.plot_cost()