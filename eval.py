from env import MarketEnv
from agents.DQN import Agent
import matplotlib.pyplot as plt
import datetime
import os

def main():
    nb_actions = 3
    obs_size = 7
    batch_size = 32
    stock = "WTW"
    episode = 300

    agent = Agent(obs_size, 5, nb_actions, 0.95, 1.0, 0.995, 0.01, 0.001, 1000, stock_name=stock, episode=episode, is_eval=True)
    env = MarketEnv(stock, window_size=5 ,is_eval=True, train_test_split=.8)


    
    state = env.reset()

    for time in range(env.l):
        action = agent.act(state)

        next_state, action, reward, done = env.step(action, time)

        agent.remember(state, action, reward, next_state, done)
        state = next_state

    prices = [line[-2] for line in env.prices]
    dates = [i for i in range(len(env.prices))]
    plt.plot(dates, prices)

    for line in env.buy:
        plt.plot(line[0], line[1], 'ro', color="g", markersize=2)

    for line in env.sell:
        plt.plot(line[0], line[1], "ro", color="r", markersize=2)
    print("Profitable Trades: " + str(env.profitable_trades))
    print("UnProfitable Trades: " + str(env.unprofitable_trades))
    plt.show()

main()
