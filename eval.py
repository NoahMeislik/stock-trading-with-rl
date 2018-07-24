from env import MarketEnv
from agents.DQN import Agent
import matplotlib.pyplot as plt
import datetime
import os

def main():
    nb_actions = 3
    obs_size = 9
    batch_size = 32
    stock = "AAPL"
    episode = 320

    agent = Agent(obs_size, 10, nb_actions, 0.95, 1.0, 0.995, 0.01, 0.001, 1000, stock_name="AAPL", episode=episode, is_eval=True)
    env = MarketEnv(stock, window_size=10 ,is_eval=True, train_test_split=0.8)


    
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
        plt.plot(line[0], line[1], 'ro', color="r", markersize=2)

    for line in env.sell:
        plt.plot(line[0], line[1], "ro", color="g", markersize=2)

    plt.show()

main()
