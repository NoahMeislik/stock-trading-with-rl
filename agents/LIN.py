import numpy as np 
import tensorflow as tf 
import random
from collections import deque
from .agent import Agent

class Agent(Agent):
    def __init__(self, state_size, window_size, action_size, gamma, epsilon, epsilon_decay, epsilon_min, learning_rate, episodes, batch_size = 32, is_eval=False, model_name="", stock_name="", episode=1):
        """
        state_size: Size of the state coming from the environment
        action_size: How many decisions the algo will make in the end
        gamma: Decay rate to discount future reward
        epsilon: Rate of randomly decided action
        epsilon_decay: Rate of decrease in epsilon
        epsilon_min: The lowest epsilon can get (limit to the randomness)
        learning_rate: Progress of neural net in each iteration
        episodes: How many times data will be run through
        """
        self.state_size = state_size
        self.window_size = window_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.episodes = episodes
        self.is_eval = is_eval
        self.model_name = model_name
        self.stock_name = stock_name
        self.q_values = []

        self.layers = []
        tf.reset_default_graph()
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement = True))
        self.memory = deque(maxlen=2000)
        if self.is_eval:
            self._model_init()
            model_name = stock_name + "-" + str(episode)
            self.saver = tf.train.Saver()
            self.saver.restore(self.sess, tf.train.latest_checkpoint("models/{}/{}".format(stock_name, model_name)))
        else:
            self._model_init()
            self.saver = tf.train.Saver()
            self.sess.run(self.init)
            path = "models/{}/1".format(self.stock_name)
            self.writer = tf.summary.FileWriter(path)
            self.writer.add_graph(self.sess.graph)

    def _model_init(self):
        """
        Init tensorflow graph vars
        """
        # (1,10,9)
        with tf.device("/device:GPU:0"):
            self.X_input = tf.placeholder(tf.float32, shape=(None))
            self.Y_input = tf.placeholder(tf.float32, shape=(1, self.action_size))
            # self.dropout_keep_prob = tf.placeholder(tf.float32) For dropout

            # Can be changed to another initializer
            self.initializer = tf.initializers.truncated_normal(seed=1)

            self.neurons = self.X_input
            self.weights = tf.Variable(self.initializer((self.state_size, self.action_size)), dtype=tf.float32)
            self.biases = tf.Variable(tf.zeros((self.action_size), dtype=tf.float32))

            self.neurons = tf.add(tf.matmul(self.weights, self.neurons), self.biases)

            self.logits = self.neurons

            # Mean-Squared-Error
            with tf.name_scope("accuracy"):
                self.loss_op = tf.losses.mean_squared_error(self.Y_input, self.logits))
                tf.summary.scalar("accuracy", self.loss_op)


            optimizer = tf.train.RMSPropOptimizer(learning_rate = self.learning_rate, decay=.99)

            with tf.name_scope("train"):
                self.train_op = optimizer.minimize(self.loss_op)

            self.summ = tf.summary.merge_all()

            self.init = tf.global_variables_initializer()


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon and not self.is_eval:
            return random.randrange(self.action_size)
        state = state.reshape((1, self.window_size * self.state_size))
        act_values = self.sess.run(self.logits, feed_dict={self.X_input: state})
        return np.argmax(act_values[0])

    def replay(self, batch_size, time, episode):
        mini_batch = []
        l = len(self.memory)
        for i in range(l - batch_size + 1, l):
            
            mini_batch.append(self.memory[i])

        for state, action, reward, next_state, done in mini_batch:
            target = reward
            state = state.reshape((1, self.window_size * self.state_size))
            next_state = next_state.reshape((1, self.window_size * self.state_size))
            if not done:
                self.target = reward + self.gamma * np.amax(self.sess.run(self.logits, feed_dict = {self.X_input: next_state})[0])
                
            target_f = self.sess.run(self.logits, feed_dict={self.X_input: state})
            target_f[0][action] = target
            _, c, s = self.sess.run([self.train_op, self.loss_op, self.summ], feed_dict={self.X_input: state, self.Y_input: target_f}) # Add self.summ into the sess.run for tensorboard
            self.writer.add_summary(s, (episode + 1) * time)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay 