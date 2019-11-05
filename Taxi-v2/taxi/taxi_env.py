import logging
import random

import gym
import numpy as np

TAXI_NAME = "Taxi-v3"
MODE = 'human'


class Taxi:
    def __init__(self, episode_params, epsilon):
        self.env = gym.make(TAXI_NAME)
        self._render_env()
        self._reset_env()
        self.q_table = None
        self._create_q_table()
        self.episode_params = episode_params
        self.epsilon = epsilon

    def _render_env(self):
        self.env.render(mode=MODE)

    def _reset_env(self):
        return self.env.reset()

    def _create_q_table(self):
        """Creates a Q-table and initializes it.
        No need to determine the size, Taxi-v3 already has the default size provided

        :return: None
        """
        action_size = self.env.action_space.n
        state_size = self.env.observation_space.n
        logging.debug("Action size {}".format(action_size))
        logging.debug("State size {}".format(state_size))

        self.q_table = np.zeros((state_size, action_size))
        logging.info(self.q_table)

    def _update_q_table(self, state, new_state, action, reward):
        """Updates the Q table state and action based on the new state and reward.

        :param state: previous state
        :param new_state: new state
        :param action: action taken
        :param reward: reward received
        :return: New state,action value for Q table
        """
        return self.q_table[state, action] + self.episode_params.learning_rate * (reward + self.episode_params.gamma *
                                                                                  np.max(self.q_table[new_state, :]) -
                                                                                  self.q_table[state, action])

    def execute_training_episode(self, max_steps=100):
        """ Completes a full training episode with the intent to train the Q table

        :param max_steps: amount of steps the program has to complete the episode.
        A default is set and can be changed in main.
        :return: None
        """
        state = self._reset_env()
        for step in range(max_steps):
            action = self._get_action_decision(state, self.epsilon)
            new_state, reward, done, info = self._take_action(action)
            self.q_table[state, action] = self._update_q_table(state, new_state, action, reward)
            if done is True:
                break
            state = new_state

    def execute_test_episode(self, max_steps=100):
        """ Completes a full test episode with the intent to test the current Q table

        :param max_steps: amount of steps the program has to complete the episode.
        A default is set and can be changed in main.
        :return: total rewards the episode has achieved
        """
        state = self._reset_env()
        total_rewards = 0
        for step in range(max_steps):
            self._render_env()
            action = self._get_exploit_action(state)
            new_state, reward, done, info = self._take_action(action)
            total_rewards += reward

            if done:
                logging.info("Total Reward: {}".format(total_rewards))
                return total_rewards
            state = new_state


    """
    Choose an action in the current state
    """
    def _get_action_decision(self, state, epsilon):
        exp_exp_tradeoff = random.uniform(0, 1)
        # if the tradeoff is greater than epsilon, Exploit
        if exp_exp_tradeoff > epsilon:
            return self._get_exploit_action(state)
        # otherwise, Explore
        else:
            return self.env.action_space.sample()

    def _get_exploit_action(self, state):
        # logging.info("exploit action engaged with state type: {}".format(type(state)))
        return np.argmax(self.q_table[state, :])

    def _take_action(self, action):
        return self.env.step(action)

    def save_q_table(self, filename):
        np.savetxt(filename, self.q_table)

    def load_q_table(self, filename):
        self.q_table = np.loadtxt(filename)

    @staticmethod
    def reduce_exploration(episode, min_epsilon, max_epsilon, decay_rate):
        return min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate*episode)
