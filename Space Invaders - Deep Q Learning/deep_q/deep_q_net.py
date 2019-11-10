import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np


class DeepQnet(object):
    def __init__(self, state_size, action_size, learning_rate, name="DeepQNet"):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.name = name

        # Initialize model values
        self.inputs_ = None
        self.actions_ = None
        self.target_Q = None
        self.output = None
        self.optimizer = None
        self.loss = None

        # Tensorboard
        self.writer = None
        self.write_op = None

    def initialize(self):
        """Creates the net's placeholders for later use
        :return:
        """
        with tf.variable_creator_scope(self.name):

            self.inputs_ = tf.placeholder(tf.float32, [None, *self.state_size], name="inputs_")
            self.actions_ = tf.placeholder(tf.float32, [None, self.action_size], name="actions_")

            # Target Q is R(s,a) + ymax Qhat(s', a')
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")

            conv1 = self.create_conv_net(self.inputs_, 32, [8, 8], [4, 4], "conv1")
            conv1_out = self.create_elu(conv1, "conv1_out")

            conv2 = self.create_conv_net(conv1_out, 64, [4, 4], [2, 2], "conv2")
            conv2_out = self.create_elu(conv2, name="conv2_out")

            conv3 = self.create_conv_net(conv2_out, 64, [3, 3], [2, 2], "conv3")
            conv3_out = self.create_elu(conv3, "conv3_out")

            fc = self.flatten_model(conv3_out)

            self.output = self.output_model(fc, self.action_size)

            # Q value is our predicted Q value
            q_value = tf.reduce_sum(tf.multiply(self.output, self.actions_))

            # Loss is the difference between our predicted Q values and the Q target
            self.loss = tf.reduce_mean(tf.square(self.target_Q - q_value))

            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def create_conv_net(self, inputs, filters, kernal_size, strides, name):
        return tf.layers.conv2d(inputs=inputs, filters=filters, kernal_size=kernal_size, strides=strides,
                                padding="VALID", kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                name=name)

    def create_elu(self, inputs, name):
        return tf.nn.elu(inputs, name)

    def flatten_model(self, inputs):
        flatten = tf.contrib.layers.flatten(inputs)
        return tf.layers.dense(inputs=flatten, units=512, activation=tf.nn.elu,
                               kernel_initializer=tf.contrib.layers.xavier_initializer(), name="fcl")

    def output_model(self, inputs, action_size):
        return tf.layers.dense(inputs=inputs, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               units=action_size, activation=None)

    def setup_tensorboard_writer(self, folder_name):
        self.writer = tf.summary.FileWriter(folder_name)
        tf.summary.scalar("Loss", self.loss)
        self.write_op = tf.summary.merge_all()

    def learn(self, batch, session, gamma, episode):
        states_mb = np.array([each[0] for each in batch], ndmin=3)
        actions_mb = np.array([each[1] for each in batch])
        rewards_mb = np.array([each[2] for each in batch])
        next_states_mb = np.array([each[3] for each in batch], ndmin=3)
        dones_mb = np.array([each[4] for each in batch])

        target_Qs_batch = []

        Qs_next_state = session.run(self.output, feed_dict={self.inputs_: next_states_mb})

        # set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma*maxQ(s', a')
        for i in range(0, len(batch)):
            terminal = dones_mb[i]
            # if in terminal state, it only equals reward
            if terminal:
                target_Qs_batch.append(rewards_mb[i])
            else:
                target = rewards_mb[i] + gamma * np.max(Qs_next_state[i])
                target_Qs_batch.append(target)
            targets_mb = np.array([each for each in target_Qs_batch])

            loss, _ = session.run([self.loss, self.optimizer],
                                  feed_dict={self.inputs_: states_mb,
                                             self.target_Q: targets_mb,
                                             self.actions_: actions_mb})

            # write TF summaries
            summary = session.run(self.write_op,
                                  feed_dict={self.inputs_: states_mb,
                                             self.target_Q: targets_mb,
                                             self.actions_: actions_mb})
            self.writer.add_summary(summary, episode)
            self.writer.flush()
