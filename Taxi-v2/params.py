class EpisodeParams(object):
    def __init__(self):
        self.total_episodes = 50000
        self.total_test_episodes = 100
        # Perhaps too many steps
        self.max_steps = 99
        # what are these?
        self.learning_rate = 0.7
        self.gamma = 0.618


class ExplorationParams(object):
    def __init__(self, epsilon=1.0, max_epsilon=1.0, min_epsilon=0.01, decay_rate=0.01):
        self.epsilon = epsilon
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
