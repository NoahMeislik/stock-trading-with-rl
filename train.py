from env import MarketEnv
from agents.LSTM import Agent
import matplotlib.pyplot as plt
import datetime
import os

def main():
    nb_actions = 3
    obs_size = 7
    batch_size = 256
    stock = "WTW"


    agent = Agent(state_size=obs_size, window_size=1, action_size=nb_actions, batch_size=batch_size, gamma=0.8, epsilon=.9, epsilon_decay=0.95, epsilon_min=0.001, learning_rate=0.001, stock_name=stock)
    env = MarketEnv(stock, window_size = 1, state_size=obs_size, shares_to_buy = 1, train_test_split=.8)


    for i in range(5000):
        state = env.reset()

        for time in range(env.l):
            if time % 100 == 0:
                print(time)
            action = agent.act(state)

            next_state, action, reward, done = env.step(action, time)

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if len(agent.memory) % batch_size == 0:
                
                agent.replay(time, i)
                if i % 10 == 0:
                    agent.q_values.append(agent.target)

        model_name = "{}-{}".format(stock, str(i))
        path = "models/{}/{}/".format(stock, model_name)

        if i % 5 == 0:    
            if not os.path.exists(path):
                os.makedirs(path)

            with open(os.path.join(path, 'LTYP.mif'), 'w'):
                pass
            agent.saver.save(agent.sess, path + model_name, global_step = i)

        print("\nEpisode " + str(i) + " finished")
        

        if i == 2000:
            prices = [line[-2] for line in env.prices]
            dates = [i for i in range(len(env.prices))]
            plt.plot(dates, prices)

            for line in env.buy:
                plt.plot(line[0], line[1], 'ro', color="r", markersize=2)

            for line in env.sell:
                plt.plot(line[0], line[1], "ro", color="g", markersize=2)

            plt.plot(agent.q_values)

            plt.show()

            plt.plot(agent.q_values)

            plt.show()

main()

