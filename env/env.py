import numpy
import math
from utils.utils import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class MarketEnv():
    def __init__(self, stock_name, window_size = 1, state_size = 7, account_balance = 1000000, shares_to_buy = 1, commision_per_share = .005, min_commision = 1.00, train_test_split = None, print_report = True, is_eval = False):
        """
        stock_name: Symbol of stock being examined
        window_size: Number of days being represented in each state (current place in time - number of days)
        account_balance: Starting balance in USD
        """
        self.stock_name = stock_name
        self.window_size = window_size
        self.state_size = state_size
        self.action_size = 1
        self.action_bound = np.array([-10., 10.])
        self.starting_balance = account_balance
        self.account_balance = account_balance
        self.shares_to_buy = shares_to_buy
        self.commision_per_share = commision_per_share
        self.min_commision = min_commision
        self.train_test_split = train_test_split
        self.print_report = print_report
        self.inventory = []
        self.state = None
        self.done = True # Start off by calling MarketEnv.reset() will change this to false
        self.is_eval = is_eval

        

        self._get_data()
        self.l = len(self.data) - 1 - self.window_size

    def _get_data(self):
        scaler = StandardScaler() # Normalize time series data
        data = getStockDataVec(self.stock_name).fillna(0)
        size = data.values.shape[0] # Shape (x, y) x is num_examples y is data in each example
        if self.train_test_split == None:
            self.data = scaler.fit_transform(data.values)
            self.prices = data.values
        else:
            if not self.is_eval:
                self.prices = data.values[:int(np.floor(self.train_test_split * size))]
                self.data = scaler.fit_transform(data.values[:int(np.floor(self.train_test_split * size))])
            elif self.is_eval:
                self.prices = data.values[int(np.floor(self.train_test_split * size)):]
                self.data = scaler.fit_transform(data.values[int(np.floor(self.train_test_split * size)):])

    def _flatten(self):
        for price in self.inventory:
            self.episode_profit += price - self.prices[-1][3] # Change this for real data
            self.inventory.remove(price)

    def reset(self):
        self.inventory = []
        self.account_balance = self.starting_balance
        self.episode_profit = 0.0
        self.unrealized_gain = 0.0
        self.reward = 0.0
        self.done = False
        self.action = 0.0
        self.buy = []
        self.sell = []
        self.dates = []
        self.profitable_trades = 0
        self.unprofitable_trades = 0

        self.state = getState(self.data, 0, self.window_size, self.unrealized_gain, self.account_balance).reshape(1, self.window_size, self.state_size)

        return self.state

    def step(self, action, time):
        if self.done:
            raise ValueError("Done, call reset to start again!")
        if action == 0:
            self.reward = 0

        if action == 1 and self.account_balance > 0 + self.prices[time][3]: #and len(self.inventory) <= 10: # buy
            price = (self.prices[time][3] * self.shares_to_buy) + (self.min_commision + (self.commision_per_share * (self.shares_to_buy - 1)))
            self.inventory.append(price) # Change -2 to wherever the close is
            self.account_balance -= price
            print("Buy: " + str(self.prices[time][3]  * self.shares_to_buy))
            self.buy.append((time, self.prices[time][3]))
            self.reward = 0

        elif action == 2 and len(self.inventory) > 0:
            bought_price = self.inventory.pop(0)
            price = (self.prices[time][3] * self.shares_to_buy) - (self.min_commision + (self.commision_per_share * (self.shares_to_buy - 1)))
            profit = price - bought_price
            if profit > 0:
                self.profitable_trades += 1
                self.reward = max(profit, 0) # Change to 0 to reset to normal
            else:
                self.unprofitable_trades += 1
                self.reward = -1
            self.episode_profit += profit
            self.account_balance += profit
            print("Sell: " + str(self.prices[time][3]  * self.shares_to_buy) + " | Profit: " + str(profit))
            self.sell.append((time, self.prices[time][3]))

        self.done = True if time == self.l - 1 else False

        if self.done:
            self._flatten()
            print("--------------------------------")
            print("Total Profit: " + str(self.episode_profit))
            print("--------------------------------")

        self.unrealized_gain = self.data[time][3]  * self.shares_to_buy - self.inventory[0] if len(self.inventory) > 0 else 0.            
        self.state = getState(self.data, time, self.window_size, self.unrealized_gain, self.account_balance).reshape(1, self.window_size, self.state_size)

        return self.state, self.reward, self.done