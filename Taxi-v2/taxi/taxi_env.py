import logging
import random

import gym
import numpy as np

TAXI_NAME = "Taxi-v3"
MODE = 'ansi'


class Taxi:
    def __init__(self, episode_params, exploration_params):
        self.env = gym.make(TAXI_NAME)
        self.render_env()
        self.q_table = None
        self.create_q_table()
        self.episode_params = episode_params
        self.exploration_params = exploration_params

    def render_env(self):
        self.env.render(mode=MODE)

    def reset_env(self):
        return self.env.reset()

    """
    Creates a Q-table and initializes it.
    No need to determine the size, Taxi-v2 already has the default size provided
    """
    def create_q_table(self):
        action_size = self.env.action_space.n
        state_size = self.env.observation_space.n
        logging.debug("Action size {}".format(action_size))
        logging.debug("State size {}".format(state_size))

        self.q_table = np.zeros((state_size, action_size))
        logging.info(self.q_table)

    def update_q_table(self, state, new_state, action, reward):
        return self.q_table[state, action] + self.episode_params.learning_rate * (reward + self.episode_params.gamma *
                                                                                  np.max(self.q_table[new_state, :]) -
                                                                                  self.q_table[state, action])

    def execute_episode(self, epsilon, max_steps=100):
        state = self.reset_env()
        for step in range(max_steps):
            action = self.get_action_decision(state, epsilon)
            new_state, reward, done, info = self.take_action(action)
            self.q_table[state, action] = self.update_q_table(state, new_state, action, reward)
            state = new_state
            if done is True:
                break

    """
    Choose an action in the current state
    """
    def get_action_decision(self, state, epsilon):
        exp_exp_tradeoff = random.uniform(0, 1)
        # if the tradeoff is greater than epsilon, Exploit
        if exp_exp_tradeoff > epsilon:
            return np.argmax(self.q_table[state, :])
        # otherwise, Explore
        else:
            return self.env.action_space.sample()

    def take_action(self, action):
        return self.env.step(action)

    def reduce_exploration(self, episode):
        return self.exploration_params.min_epsilon + \
               (self.exploration_params.max_epsilon - self.exploration_params.min_epsilon) * np.exp(-self.exploration_params.decay_rate*episode)
