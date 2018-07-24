from env import MarketEnv
from agents.DQN import Agent
import matplotlib.pyplot as plt
import datetime
import os

def main():
    nb_actions = 3
    obs_size = 9
    batch_size = 32
    stock = "INTC"


    agent = Agent(obs_size, 10, nb_actions, 0.95, 1.0, 0.995, 0.01, 0.001, 1000, stock_name=stock)
    env = MarketEnv(stock, window_size=10, train_test_split=.8)


    for i in range(500):
        state = env.reset()

        for time in range(env.l):
            action = agent.act(state)

            next_state, action, reward, done = env.step(action, time)

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if len(agent.memory) > batch_size:
                agent.replay(batch_size, time, i)

        model_name = "{}-{}".format(stock, str(i))
        path = "models/{}/{}/".format(stock, model_name)

        if not os.path.exists(path):
            os.makedirs(path)

        with open(os.path.join(path, 'LTYP.mif'), 'w'):
            pass
            
        agent.saver.save(agent.sess, path + model_name, global_step = i)
        print("\nEpisode " + str(i) + " finished")
        

        if i == 500:
            prices = [line[-2] for line in env.prices]
            dates = [i for i in range(len(env.prices))]
            plt.plot(dates, prices)

            for line in env.buy:
                plt.plot(line[0], line[1], 'ro', color="r", markersize=2)

            for line in env.sell:
                plt.plot(line[0], line[1], "ro", color="g", markersize=2)

            plt.show()

main()

