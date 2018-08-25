from env import MarketEnv
from agents.A2C import Agent
import matplotlib.pyplot as plt
import datetime
import os
import time
import atexit
import numpy as np

start_time = time.clock()
def exit_handler():
    time_done = time.clock() - start_time
    print("The program took: " + str(time_done) + " seconds" )
    
def main():
    obs_size = 9
    window_size = 10
    batch_size = 2048
    episodes = 10000
    stock = "BAC"

    args = {'tau': .001, 'gamma': .99, 'lr_actor': .0001, 'lr_critic': .001, 'batch_size': batch_size}


    env = MarketEnv(stock, window_size = window_size, state_size=obs_size, account_balance = 1000000, shares_to_buy = 10, train_test_split=.8)
    agent = Agent(args, state_size=env.state_size, window_size=env.window_size, action_size=env.action_size, action_bound=env.action_bound[1], is_eval=False, stock_name=stock)


    for i in range(10000):
        state = env.reset()
        episode_ave_max_q = 0
        ep_reward = 0

        for time in range(env.l):
            
            action = agent.act(state)[0]
            
            if action < 0:
                choice = 2
            elif action > 0 and action[0] < 1:
                choice = 0
            elif action > 1:
                choice = 1

            next_state, reward, done = env.step(choice, time)
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if agent.replay_buffer.size() % batch_size == 0:
                episode_ave_max_q += agent.replay(time, i, episode_ave_max_q)
            
            ep_reward += reward

            if env.account_balance < 0 and len(env.inventory) == 0:
                break


        model_name = "{}-{}".format(stock, str(i))
        path = "models/{}/{}/".format(stock, model_name)

        if i % 5 == 0:    
            if not os.path.exists(path):
                os.makedirs(path)

            with open(os.path.join(path, 'LTYP.mif'), 'w'):
                pass
            agent.saver.save(agent.sess, path + model_name, global_step = i)


        summary_str = agent.sess.run(agent.summary_ops, feed_dict={agent.summary_vars[0]: ep_reward, agent.summary_vars[1]: episode_ave_max_q})

        agent.writer.add_summary(summary_str, i)
        agent.writer.flush()

        print('| Reward: {:d} | Episode: {:d} | Qmax: {:.4f}'.format(int(ep_reward), i, (episode_ave_max_q)))
        
        

atexit.register(exit_handler)
main()

