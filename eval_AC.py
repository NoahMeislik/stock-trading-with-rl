import tensorflow as tf
import numpy as np
from utils import utils as U
import argparse
from scipy import signal
import os 
from env import MarketEnv
from agents import AC as pol
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--seed", default=12321, type=int)  
parser.add_argument("--model_dir", type=str)   #where to save checkpoint
parser.add_argument("--frames", default=1, type=int)    #how many recent frames to send to model 

class Framer(object):
    """
    Creates the augmentd obs features from the bare observations. Any obs fed to Actor & Critics nets must go through Framer. 
    Currently it simply concatenates a few (frame_num) recent bare obs together. 
    So ob_dim = env.observation_space.shape[0] * frame_num
    Members:
      last: given the current stack of obs creates the last feature for t=len(obs)
      full: same as last but gives features for the whole whole history t= 0, 1, ..., len(obs)
    """
    def __init__(self, frame_num):
        self.frame_num =  frame_num
    def _extend(self, obs):
        obs = list(obs)
        init = [obs[0]] * (self.frame_num-1)
        return init + obs

    def last(self, obs):
        obs = self._extend(obs)
        li = [obs[i] for i in range(-self.frame_num, 0)]
        return np.concatenate(li)
    def full(self, obs):
        
        obs = self._extend(obs)
        frames = []
        for i in range(len(obs)-self.frame_num+1):
            li = [obs[i+j] for j in range(self.frame_num)]
            frames.append(np.concatenate(li))
        return frames

def rollout(env, sess, policy, framer, max_path_length=100):
    """
    Gather an episode of experiences by running the environment. Continues until env.done is True
    or length of episode exceeds max_path_length
    """
    t = 0
    ob = env.reset()
    obs = [ob]
    logps = []
    rews = []
    acs = []
    sum_ents = 0
    done = False
    while t < max_path_length and not done:
        t += 1
        ac, logp, ent = policy(framer.last(obs), sess=sess)
        
        ob, rew, done = env.step(ac, t)
        obs.append(ob)
        rews.append(rew)
        acs.append(ac)
        sum_ents += ent
        logps.append(logp)
    path = {'rews': rews, 'obs':obs, 'acs':acs, 'terminated': done, 'logps':logps, 'entropy':sum_ents}
    return path

def get_roll_params():
    """
    Creates environment and sets up the rollout params.
    """
    env = MarketEnv("BAC", 3, is_eval=True, max_positions=1)
    max_path_length, ep_length_stop = env.l, env.l
    
    print('\nMAX PATH LENGTH, EP LENGTH STEP: {}, {}\n'.format(max_path_length, ep_length_stop))
    return env, max_path_length, ep_length_stop


def test_process(random_seed, stack_frames, model_path, num_episodes):
    env, MAX_PATH_LENGTH, _ = get_roll_params()
    framer = Framer(frame_num=stack_frames)
    ob_dim = env.state_size * stack_frames
    
    if env.action_space == "discrete":
        act_type = 'disc'
        ac_dim, ac_scale = env.action_size, None
        print('Discrete Action Space. Numer of actions is {}.'.format(env.action_size))
    else:
        act_type = 'cont'
        ac_dim, ac_scale = env.action_size, env.action_bound[1]
        print('Continuous Action Space. Action Scale is {}.'.format(ac_scale))
    actor = pol.Actor(num_ob_feat=ob_dim, ac_dim=ac_dim, act_type=act_type, ac_scale=ac_scale) 
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess=sess, save_path=tf.train.latest_checkpoint(model_path))
        avg_rew = 0
        for i in range(num_episodes):
            path = rollout(env=env, sess= sess, policy=actor.act, max_path_length=MAX_PATH_LENGTH, framer=framer)
            rew = np.sum(path['rews'])
            print("Iteration {}".format(i))
            print("Reward {}".format(rew))
            print("Episode Length {}\n".format(len(path['rews'])))
            avg_rew += rew/float(num_episodes)
        print('Average reward over {} was {}'.format(num_episodes, avg_rew))

        prices = [line[3] for line in env.prices]
        dates = [i for i in range(len(env.prices))]
        plt.plot(dates, prices)

        for line in env.buy:
            plt.plot(line[0], line[1], 'ro', color="g", markersize=2)

        for line in env.sell:
            plt.plot(line[0], line[1], "ro", color="r", markersize=2)

        percentage_gain = ((env.account_balance - env.starting_balance) / env.starting_balance) * 100

        print("Profitable Trades: " + str(env.profitable_trades))
        print("Unprofitable Trades: " + str(env.unprofitable_trades))
        print("Percentage Gain: " + str(percentage_gain))
        

        plt.show()

if __name__ == "__main__":
    args = parser.parse_args()
    test_process(args.seed, args.frames, args.model_dir, 1)
    
    
