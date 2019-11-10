from collections import deque
import numpy as np


class Memory(object):
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                 size=batch_size,
                                 replace=False)
        return [self.buffer[i] for i in index]

    @classmethod
    def create_initialized_memory(cls, memory_size, pretrain_len, state, stacked_frames):
        """returns a Memory object with initialized memory already in place
        NOTE: reset the state and get the state and stacked_frames from FrameStacker.stack_frames prior to using them
        in this function.

        :param memory_size: memory size allowed. Parameter pulled from and altered at deep_q_params.py under TrainingParams.memory_size
        :param pretrain_len: how many times to pretrain the memory, relates to batch_size. Parameter pulled from and altered at deep_q_params.py under TrainingParams.batch_size
        :param state: current state of the environment
        :param stacked_frames: array of stacked processed frames
        :return:
        """
        state = None




