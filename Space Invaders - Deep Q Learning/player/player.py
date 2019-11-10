import numpy as np
import tensorflow.compat.v1 as tf
import random
import logging
from deep_q.deep_q_net import DeepQnet
tf.disable_v2_behavior()

MODEL_PATH = "./models/model.ckpt"


class Player(object):
    def __init__(self, explore_start, explore_stop, decay_rate, possible_actions, action_size, state_size,
                 learning_rate):
        self.explore_start = explore_start
        self.explore_stop = explore_stop
        self.decay_rate = decay_rate
        self.decay_step = 0
        self.possible_actions = possible_actions
        self.deep_q_net = DeepQnet(state_size=state_size, action_size=action_size, learning_rate=learning_rate)
        self.saver = None
        self.training_rewards_list = []
        self.test_rewards_list = []

    def predict_action(self, state, session):
        exp_exp_tradeoff = np.random.rand()

        explore_probability = self.explore_stop + (self.explore_start - self.explore_stop) * \
                              np.exp(-self.decay_rate * self.decay_step)

        if explore_probability > exp_exp_tradeoff:
            action = self.get_random_action()
        else:
            action = self.get_exploit_action(state, session)

        return action, explore_probability

    def get_random_action(self):
        choice = random.randint(1, len(self.possible_actions)) - 1
        return self.possible_actions[choice]

    def get_exploit_action(self, state, session):
        Qs = session.run(self.deep_q_net.output, feed_dict={self.deep_q_net.inputs_: state.reshape((1, *state.shape))})
        choice = np.argmax(Qs)
        return self.possible_actions[choice]

    def reset_decay_step(self):
        self.decay_step = 0

    def get_decay_step(self):
        return self.decay_step

    def increase_decay_step(self):
        self.decay_step += 1

    def add_reward_to_list(self, episode, total_reward, training=False):
        if training:
            self.training_rewards_list.append({episode, total_reward})
        else:
            self.test_rewards_list.append({episode, total_reward})


    def init_saver(self):
        self.saver = tf.train.Saver()

    def save_model(self, session):
        self.saver.save(session, MODEL_PATH)
        logging.info("Model Saved")

    def load_model(self, session):
        self.saver.restore(session, MODEL_PATH)

    @classmethod
    def create_class_with_param_object(cls, training_params, possible_actions, action_size):
        return cls(explore_start=training_params.explore_start,
                   explore_stop=training_params.explore_stop,
                   decay_rate=training_params.decay_rate,
                   possible_actions=possible_actions,
                   action_size=action_size,
                   state_size=training_params.state_size,
                   learning_rate=training_params.learning_rate)
