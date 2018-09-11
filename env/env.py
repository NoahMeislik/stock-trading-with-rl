import numpy
import math
from utils.utils import *
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class MarketEnv():
    def __init__(self, stock_name, buy_position, window_size = 1, account_balance = 1000000, shares_to_buy = 1000, max_positions = 1,  min_commision = 1.00, commision_per_share = .005, max_episode_len=10000, train_test_split = None, print_report = True, is_eval = False):
        """Market Environment base class. Please be careful, a strategy tested using this environment may have completely different results in practice.

        Args:
            stock_name (str): Name of the file that contains data, place this file in a folder named data. For example, data/AAPL.csv
            buy_position (int): The position of the price to use for purchasing a share of a stock in the data. For example, position 4 could be the close price of a day of trading.
            window_size (int): The number of days that data in each step will contain. For example, a window size of 10 returns the data at a timestep with ten steps back included (data[t:t-10]) Default: 1
            account_balance (int): The amount of money allowed to trade with. The episode will automatically end if the account_balance goes below 0 with no positions held. Default: $1000000
            shares_to_buy (int): The number of shares bought per trade. Default: 1 share per position held
            max_positions (int): The max number of positions that are allowed to be held. Default: 3 positions can be held before the first need to be sold
            min_commision (float): The base price reduction taken from one position. Default: $1 per position held
            commision_per_share (float): The amount of money added to the min_commision per share. Default: $.005 per share
            max_episode_len (int): The max number of data points before an episode will end. Set to a number larger than the amount of data points to go through all data. Default: 100000 data points
            train_test_split (float between 0-1): Split the data so unseen data can be used for evaluation. Default: Set to None, change to a number between 0-1 (.8 for 80% train)
            print_report (bool): Whether or not to print stats after and episode finishes. Default: True
            is_eval (bool): Whether or not the environment is being used for evaluation. If yes and train_test_split is not None than the test split of data will be used. Default: False
            
        Functions:
            _get_data: Read, scale and reshape data.
            _flatten: Sell off all positions at the end of an episode.
            reset: Call before and after an episode. Resets all environment variables.
            step: Run through one time step.
        """
        self.stock_name = stock_name
        self.window_size = window_size
        self.action_size = 3
        self.action_bound = np.array([-10., 10.])
        self.action_space = "discrete"
        self.starting_balance = account_balance
        self.account_balance = account_balance
        self.shares_to_buy = shares_to_buy
        self.commision_per_share = commision_per_share
        self.min_commision = min_commision
        self.max_episode_len = max_episode_len
        self.max_positions = max_positions
        self.buy_position = buy_position
        self.train_test_split = train_test_split
        self.print_report = print_report
        self.inventory = []
        self.state = None
        self.done = True # Start off by calling MarketEnv.reset() will change this to false
        self.is_eval = is_eval

        self._get_data()
        self.l = len(self.data) - 1 - self.window_size # add the random start to the end of the self.l
    
        

    def _get_data(self):
        """Read, scale and reshape data
        """
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
        self.state_size = self.data.shape[1]
        

    def _flatten(self, time):
        """Sell off all positions at the end of an episode
        """
        for price in self.inventory:
            
            adj_price = (self.prices[time][self.buy_position] * self.shares_to_buy) - (self.min_commision + (self.commision_per_share * (self.shares_to_buy - 1)))
            self.account_balance += adj_price
            profit = adj_price - price
            self.episode_profit += profit
            if profit > 0:
                self.profitable_trades += 1
            else:
                self.unprofitable_trades += 1
            
            self.reward += profit
            self.inventory.remove(price)

            
    def reset(self):
        """Call before and after an episode. Resets all environment variables. Will reset account_balance to the starting amount.
        """
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

        self.random_start = random.randint(1, self.l) - self.window_size if not self.is_eval else 0
        self.state = getState(self.data, 0, self.window_size, self.random_start, self.unrealized_gain, self.account_balance).reshape(self.state_size)

        return self.state

    def step(self, action, time):
        """Run through one time step. Buy, Sell, Hold.
        
        Args:
            action (int): Whether a position should be acquired, sold, or held at the current timestep. (0 for hold, 1 for buy, 2 for sell)
            time (int): The current timestep in the data. 
        
        Output:
            state (array): Data from the next timestep.
            reward (float): Whether the action taken was good or bad. If a position is sold, the reward is the difference between the price bought and price sold, if a positions is bought or held, reward is 0.


        """
        if self.done:
            raise ValueError("Done, call reset to start again!")
        if self.action == 0:
            profit = 0

        if action == 1 and self.account_balance > 0 + self.prices[time][self.buy_position] and len(self.inventory) < self.max_positions: # buy
            price = (self.prices[time][self.buy_position] * self.shares_to_buy) + (self.min_commision + (self.commision_per_share * (self.shares_to_buy - 1)))
            self.inventory.append(price) # Change -2 to wherever the close is
            self.account_balance -= price
            print("Buy: " + str(self.prices[time][self.buy_position]  * self.shares_to_buy))
            self.buy.append((time, self.prices[time][self.buy_position]))
            profit = 0
            

        elif action == 2 and len(self.inventory) > 0:
            bought_price = self.inventory.pop(0)
            price = (self.prices[time][self.buy_position] * self.shares_to_buy) - (self.min_commision + (self.commision_per_share * (self.shares_to_buy - 1)))
            profit = price - bought_price
            if profit > 0:
                self.profitable_trades += 1
            else:
                self.unprofitable_trades += 1
                
            self.account_balance += price
            print("Sell: " + str(self.prices[time][self.buy_position]  * self.shares_to_buy) + " | Profit: " + str(profit))
            print("Account Balance: " + str(self.account_balance))
            self.sell.append((time, self.prices[time][self.buy_position]))

        self.reward = profit
        self.episode_profit += float(profit)
        profit = 0

        if time + self.random_start == self.l or time == self.max_episode_len:
            self.done = True
        elif self.account_balance <  (self.prices[time][self.buy_position] * self.shares_to_buy) + (self.min_commision + (self.commision_per_share * (self.shares_to_buy - 1))) and len(self.inventory) < 1:
            self.done = True
        else:
            self.done = False

        if self.done:
            self._flatten(time)
            if self.print_report:
                print("--------------------------------")
                print("Total Profit: " + str(self.episode_profit))
                print("Account Balance: " + str(self.account_balance))
                print("--------------------------------")

        self.unrealized_gain = self.data[time][self.buy_position]  * self.shares_to_buy - self.inventory[0] if len(self.inventory) > 0 else 0.
        self.state = getState(self.data, time, self.window_size, self.random_start, self.unrealized_gain, self.account_balance).reshape(self.state_size)
        return self.state, self.reward, self.done