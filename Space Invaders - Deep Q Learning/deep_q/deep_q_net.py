import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


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

    # @tf.function
    def initialize(self):
        """Creates the net's placeholders for later use
        :return:
        """
        with tf.variable_creator_scope(self.name):
            # Changing tf.placeholder to tf.Variable

            self.inputs_ = tf.placeholder(tf.float32, [None, *self.state_size], name="inputs_")
            # self.actions_ = tf.placeholder(tf.float32, [None, self.action_size], name="actions_")
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

