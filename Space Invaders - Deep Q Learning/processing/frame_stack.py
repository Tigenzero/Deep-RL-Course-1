from collections import deque
import numpy as np
import logging


class FrameStacker(object):
    def __init__(self, stack_size, frame_size_tuple):
        """
        params pulled from deep_q_params.py
        :param stack_size: Parameter pulled from and altered at ProcessingParams.stack_size
        :param frame_size_tuple: Parameter pulled from and altered at
        (ProcessingParams.resize_width, ProcessingParams.resize_height)
        """
        self.stack_size = stack_size
        self.frame_size_tuple = frame_size_tuple

    def init_stacked_frames(self):
        return deque([np.zeros(self.frame_size_tuple, dtype=np.int) for i in range(self.stack_size)],
                     maxlen=self.stack_size)

    def stack_frames(self, processed_frame):
        stacked_frames = self.init_stacked_frames()
        for _ in range(self.stack_size - 1):
            stacked_frames.append(processed_frame)
        return self.stack_frames(stacked_frames, processed_frame)

    # process frame before using this function
    def stack_frames(self, stacked_frames, processed_frame):
        stacked_frames.append(processed_frame)
        stacked_state = np.stack(stacked_frames, axis=2)
        return stacked_state, stacked_frames

    @classmethod
    def create_class_with_param_object(cls, processing_params):
        return cls(processing_params.stack_size, (processing_params.resize_width, processing_params.resize_height))