from env import MarketEnv
from agents.LSTM import Agent
import matplotlib.pyplot as plt
import datetime
import os

def main():
    nb_actions = 3
    obs_size = 7
    window_size = 1
    batch_size = 256
    stock = "WTW"
    episode = 20

    agent = Agent(state_size = obs_size, window_size = window_size, action_size = nb_actions, batch_size = batch_size, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, learning_rate=0.001, is_eval=True, stock_name=stock, episode=episode)
    env = MarketEnv(stock, window_size=window_size, state_size = obs_size, is_eval=True, shares_to_buy = 1, train_test_split=None)


    
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
