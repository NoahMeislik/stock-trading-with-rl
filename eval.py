from env import MarketEnv
from agents.LIN import Agent
import matplotlib.pyplot as plt
import datetime
import os

def main():
    nb_actions = 3
    obs_size = 9
    window_size = 10
    batch_size = 256
    stock = "BAC"
    episode = 75
    total_spent = 0
    total_sold = 0

    agent = Agent(state_size = obs_size, window_size = window_size, action_size = nb_actions, batch_size = batch_size, gamma=0.95, epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.001, learning_rate=0.001, is_eval=True, stock_name=stock, episode=episode)
    env = MarketEnv(stock, window_size=window_size, state_size = obs_size, account_balance = 1000000, is_eval=True, shares_to_buy = 10, train_test_split=.8)


    
    state = env.reset()

    for time in range(env.l):
        action = agent.act(state)
        
        next_state, action, reward, done = env.step(action, time)

        agent.remember(state, action, reward, next_state, done)
        state = next_state




    prices = [line[3] for line in env.prices]
    dates = [i for i in range(len(env.prices))]
    plt.plot(dates, prices)

    for line in env.buy:
        plt.plot(line[0], line[1], 'ro', color="g", markersize=2)
        total_spent += line[1]

    for line in env.sell:
        plt.plot(line[0], line[1], "ro", color="r", markersize=2)
        total_sold += line[1]

    percentage_gain = ((env.account_balance - env.starting_balance) / env.starting_balance) * 100

    print("Profitable Trades: " + str(env.profitable_trades))
    print("Unprofitable Trades: " + str(env.unprofitable_trades))
    print("Percentage Gain: " + str(percentage_gain))
    print("Amount Spent: " + str(total_spent))
    print("Amount Sold: " + str(total_sold))

    plt.show()
    plt.savefig("models/{}/{}-{}/{}".format(stock, stock, str(episode), stock))

main()
