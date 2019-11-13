BATCH_SIZE = 64

class ProcessingParams(object):
    def __init__(self):
        self.top_crop = 8
        self.bottom_crop = 12
        self.left_crop = 4
        self.right_crop = 12
        self.normalize = 255.0
        self.resize_width = 110
        self.resize_height = 84
        self.stack_size = 4

        # Memory
        self.pretrain_length = BATCH_SIZE
        self.memory_size = 10 ** 6


class TrainingParams(object):
    def __init__(self):
        # Model
        self.state_size = [110, 84, 4]
        self.learning_rate = 0.00025

        # Training
        self.total_episodes = 30
        self.max_steps = 50000
        self.batch_size = BATCH_SIZE

        # Exploration
        self.explore_start = 1.0
        self.explore_stop = 0.01
        self.decay_rate = 0.00001

        # Q learning
        self.gamma = 0.9

        # To see trained agent, set to False
        self.training = True
        self.use_existing_model = True

        # To see the environment, set to True
        self.episode_render = False
