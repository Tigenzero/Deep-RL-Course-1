import os
import logging.config
import retro
import matplotlib.pyplot as plt
import random
import numpy as np
import warnings
import tensorflow.compat.v1 as tf
from experience_replay.memory import Memory
from processing.frame_process import FramePreprocessor
from processing.frame_stack import FrameStacker
from player.player import Player

tf.disable_v2_behavior()
from deep_q.deep_q_net import DeepQnet
from deep_q_params import ProcessingParams, TrainingParams

RETRO_GAME = 'SpaceInvaders-Atari2600'


class Environment(object):
    def __init__(self):
        self._env = retro.make(game=RETRO_GAME)
        self._state = None
        self.reset_environment()
        self.possible_actions = np.array(np.identity(environment.get_action_space(), dtype=int).tolist())
        self.processing_params = ProcessingParams()
        self.frame_preprocessor = FramePreprocessor.create_class_with_param_object(self.processing_params)
        self.frame_stacker = FrameStacker.create_class_with_param_object(self.processing_params)
        self.memory = None
        self.initialize_memory()

    def get_action_space(self):
        return self._env.action_space.n

    def get_possible_actions(self):
        return self.possible_actions

    def get_observation_space(self):
        return self._env.observation_space

    def get_state(self):
        return self._state

    def reset_environment(self):
        self._state = self._env.reset()

    def render_environment(self):
        self._env.render()

    def get_random_action(self):
        choice = random.randint(1, len(self.possible_actions)) - 1
        return self.possible_actions[choice]

    def take_action(self, action):
        return self._env(action)

    def init_stack_frames(self, state):
        """
        Process state and stack frames in a new environment
        Nothing is needed since the state will be pulled from the class state
        :return: stacked state, stacked frames
        """
        processed_frame = self.frame_preprocessor.preprocess_frame(state)
        return self.frame_stacker.stack_frames(processed_frame)

    def stack_frames(self, stacked_frames, state):
        processed_frame = self.frame_preprocessor.preprocess_frame(state)
        return self.frame_stacker.stack_frames(stacked_frames, processed_frame)

    def initialize_memory(self):
        state = environment.get_state()
        self.memory = Memory(self.processing_params.memory_size)
        stacked_state, stacked_frames = self.init_stack_frames(state)
        for _ in range(self.processing_params.pretrain_length):
            action = self.get_random_action()
            next_state, reward, done, _ = self.take_action(action)
            # self.render_environment()
            next_state, stacked_frames = self.stack_frames(stacked_frames, next_state)
            if done:
                next_state = np.zeros(state.shape)
                self.memory.add((state, action, reward, next_state))
                self.reset_environment()
                state, stacked_frames = self.init_stack_frames(self.get_state())
            else:
                self.memory.add((state, action, reward, next_state))
                state = next_state


def setup_tensorboard_writer(folder_name, loss):
    writer = tf.summary.FileWriter(folder_name)

    tf.summary.scalar("Loss", loss)
    write_op = tf.summary.merge_all()
    return writer, write_op


if __name__ == "__main__":
    logfile = os.path.join('logs', 'logging_file.log')
    print(logfile)
    if not os.path.isdir('logs'):
        os.makedirs('logs')
    logging.config.fileConfig('logging.conf', defaults={'logfile': logfile})
    logging.debug("Starting main")
    warnings.filterwarnings('ignore')

    environment = Environment()
    training_params = TrainingParams()
    player = Player.create_class_with_param_object(training_params=training_params,
                                                   possible_actions=environment.get_possible_actions())
    logging.info("Size of Frame: {}".format(environment.get_observation_space()))
    logging.info("Action Size: {}".format(environment.get_action_space()))

    tf.reset_default_graph
    deep_q_net = DeepQnet(training_params.state_size, environment.get_action_space(), training_params.learning_rate)
    deep_q_net.initialize()
    # To Launch Tensorboard: tensorboard --logidr=/tensorboard/dqn/1
    writer, write_op = setup_tensorboard_writer("/tensorboard/dqn/1", deep_q_net.loss)
