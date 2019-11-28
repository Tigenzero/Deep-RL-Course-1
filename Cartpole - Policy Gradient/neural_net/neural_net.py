import tensorflow as tf


class NeuralNet(object):
    def __init__(self):
        self.input_ = None
        self.loss = None
        self.mean_reward_ = None
        self.action_distribution = None
        self.writer = None
        self.write_op = None
        self.train_opt = None
        self.actions = None
        self.discounted_episode_rewards_ = None

    def initialize(self, state_size, action_size, learning_rate):
        with tf.name_scope("inputs"):
            self.input_ = tf.placeholder(tf.float32, [None, state_size], name="input_")
            self.actions = tf.placeholder(tf.int32, [None, action_size], name="actions")
            self.discounted_episode_rewards_ = tf.placeholder(tf.float32, [None, ], name="discounted_episode_rewards")

            self.mean_reward_ = tf.placeholder(tf.float32, name="mean_reward")

            with tf.name_scope("fcl"):
                fc1 = self.create_fully_connected_layer(self.input_, 10, tf.nn.relu)

            with tf.name_scope("fc2"):
                fc2 = self.create_fully_connected_layer(fc1, action_size, tf.nn.relu)

            with tf.name_scope("fc3"):
                fc3 = self.create_fully_connected_layer(fc2, action_size, None)

            with tf.name_scope("softmax"):
                self.action_distribution = tf.nn.softmax(fc3)

            with tf.name_scope("loss"):
                neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc3, labels=self.actions)
                self.loss = tf.reduce_mean(neg_log_prob * self.discounted_episode_rewards_)

            with tf.name_scope("train"):
                self.train_opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

    def create_fully_connected_layer(self, inputs, outputs, activation_fn):
        return tf.contrib.layers.fully_connected(inputs=inputs,
                                                 num_outputs=outputs,
                                                 activation_fn=activation_fn,
                                                 weights_initializer=tf.contrib.layers.xavier_initializer())

    def setup_tensorboard(self, foldername):
        self.writer = tf.summary.FileWriter(foldername)
        tf.summary.scalar("Loss", self.loss)
        tf.summary.scalar("Reward_mean", self.mean_reward_)
        self.write_op = tf.summary.merge_all()

    def get_action_distribution(self):
        return self.action_distribution
