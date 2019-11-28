class EpisodeParams(object):
    def __init__(self, total_training_episodes=50000, total_test_episodes=100, max_steps=100, learning_rate=0.7, gamma=0.618):
        self.total_training_episodes = total_training_episodes
        self.total_test_episodes = total_test_episodes
        # Perhaps too many steps
        self.max_steps = max_steps
        # what are these?
        self.learning_rate = learning_rate
        self.gamma = gamma


class ExplorationParams(object):
    def __init__(self, epsilon=1.0, max_epsilon=1.0, min_epsilon=0.01, decay_rate=0.01):
        self.epsilon = epsilon
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
