import numpy as np 
import tensorflow as tf 
import random
from collections import deque
from .agent import Agent

class Agent(Agent):
    def __init__(self, state_size, window_size, action_size, gamma, epsilon, epsilon_decay, epsilon_min, learning_rate, episodes, is_eval=False, model_name="", stock_name="", episode=1):
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

        self.layers = [150, 150, 150]
        tf.reset_default_graph()
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement = True))
        self.memory = deque(maxlen=2000)
        if self.is_eval:
            model_name = stock_name + "-" + str(episode)
            self.saver = tf.train.import_meta_graph("models/{}/{}/{}".format(stock_name, model_name, model_name + "-" + str(episode) + ".meta"))
            self.saver.restore(self.sess, "models/{}/{}/{}".format(stock_name, model_name, model_name + "-" + str(episode)))
        else:
            self._model_init()
            self.saver = tf.train.Saver()
            self.sess.run(self.init)
            path = "tmp/{}/1".format(self.stock_name)
            self.writer = tf.summary.FileWriter(path)
            self.writer.add_graph(self.sess.graph)

    def _model_init(self):
        """
        Init tensorflow graph vars
        """
        # (1,10,9)
        with tf.device("/device:GPU:0"):

            with tf.name_scope("Inputs"):
                self.X_input = tf.placeholder(tf.float32, [None, self.state_size, 1])
                self.Y_input = tf.placeholder(tf.float32, [self.window_size, self.action_size])

            self.lstm_cells = [tf.contrib.rnn.BasicLSTMCell(num_units=self.layers[layer])
                        for layer in range(len(self.layers))]

            #lstm_cell = tf.contrib.rnn.LSTMCell(num_units=n_neurons, use_peepholes=True)
            #gru_cell = tf.contrib.rnn.GRUCell(num_units=n_neurons)

            self.multi_cell = tf.contrib.rnn.MultiRNNCell(self.lstm_cells)
            self.outputs, self.states = tf.nn.dynamic_rnn(self.multi_cell, self.X_input, dtype=tf.float32)
            self.top_layer_h_state = self.states[-1][1]

            with tf.name_scope("Output"):
                self.logits = tf.layers.dense(self.top_layer_h_state, self.action_size, name="softmax")

            with tf.name_scope("Cross_Entropy"):
                self.loss_op = tf.reduce_mean(tf.squared_difference(self.logits, self.Y_input))
                self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
                self.train_op = self.optimizer.minimize(self.loss_op)
            # self.correct = tf.nn.in_top_k(self.logits, self.Y_input, 1)

            # self.accuracy = tf.reduce_mean(tf.cast(self., tf.float32))

            # Merge all of the summaries
            self.summ = tf.summary.merge_all()
            # Initiate all vars
            self.init = tf.global_variables_initializer()


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.sess.run(self.logits, feed_dict={self.X_input: state})
        return np.argmax(act_values[0])

    def replay(self, batch_size, time, episode):
        mini_batch = []
        l = len(self.memory)
        for i in range(l - batch_size + 1, l):
            
            mini_batch.append(self.memory[i])

        for state, action, reward, next_state, done in mini_batch:
            target = reward
            if not done:
                self.target = reward + self.gamma * np.amax(self.sess.run(self.logits, feed_dict = {self.X_input: next_state})[0])
            target_f = self.sess.run(self.logits, feed_dict={self.X_input: state})
            target_f[0][action] = self.target
            _, c = self.sess.run([self.train_op, self.loss_op], feed_dict={self.X_input: state, self.Y_input: target_f}) # Add self.summ into the sess.run for tensorboard
            # self.writer.add_summary(s, (episode + 1) * time)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay 