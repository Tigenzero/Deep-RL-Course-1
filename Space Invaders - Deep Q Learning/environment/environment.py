import os
import retro
import numpy as np
import tensorflow.compat.v1 as tf
import imageio
from experience_replay.memory import Memory
from processing.frame_process import FramePreprocessor
from processing.frame_stack import FrameStacker
from deep_q_params import ProcessingParams
tf.disable_v2_behavior()

RETRO_GAME = 'SpaceInvaders-Atari2600'
CUR_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(CUR_FILE_PATH, "images")


class Environment(object):
    def __init__(self):
        self._env = retro.make(game=RETRO_GAME)
        self._state = None
        self.reset_environment()
        self.possible_actions = np.array(np.identity(self.get_action_size(), dtype=int).tolist())
        self.processing_params = ProcessingParams()
        self.frame_preprocessor = FramePreprocessor.create_class_with_param_object(self.processing_params)
        self.frame_stacker = FrameStacker.create_class_with_param_object(self.processing_params)
        self.memory = None

    def get_action_size(self):
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

    def take_action(self, action):
        return self._env.step(action)

    def env_init_stack_frames(self, state):
        """
        Process state and stack frames in a new cartpole_environment
        Nothing is needed since the state will be pulled from the class state
        :return: stacked state, stacked frames
        """
        processed_frame = self.frame_preprocessor.preprocess_frame(state)
        return self.frame_stacker.stack_frames(None, processed_frame)

    def env_stack_frames(self, stacked_frames, state):
        processed_frame = self.frame_preprocessor.preprocess_frame(state)
        return self.frame_stacker.stack_frames(stacked_frames, processed_frame)

    def create_zero_state(self, state):
        return np.zeros(state.shape)

    def initialize_memory(self, player):
        self.memory = Memory(self.processing_params.memory_size)
        stacked_state, stacked_frames = self.env_init_stack_frames(self.get_state())
        for _ in range(self.processing_params.pretrain_length):
            action = player.get_random_action()
            next_state, reward, done, _ = self.take_action(action)
            # self.render_environment()
            next_state, stacked_frames = self.env_stack_frames(stacked_frames, next_state)
            if done:
                next_state = self.create_zero_state(stacked_state)
                self.memory.add((stacked_state, action, reward, next_state, done))
                self.reset_environment()
                stacked_state, stacked_frames = self.env_init_stack_frames(self.get_state())
            else:
                self.memory.add((stacked_state, action, reward, next_state, done))
                stacked_state = next_state

    def save_frame(self, frame, filename):
        if not os.path.exists(IMAGE_PATH):
            os.mkdir(IMAGE_PATH)
        frame_path = os.path.join(IMAGE_PATH, filename)
        print(frame_path)
        imageio.imsave(frame_path, frame)
