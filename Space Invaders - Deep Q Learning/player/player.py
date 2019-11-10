class Player(object):
    def __init__(self, explore_start, explore_stop, decay_rate, actions):
        self.explore_start = explore_start
        self.explore_stop = explore_stop
        self.decay_rate = decay_rate
        self.actions = actions

    @classmethod
    def create_class_with_param_object(cls, training_params, possible_actions):
        return cls(explore_start=training_params.explore_start,
                   explore_stop=training_params.explore_stop,
                   decay_rate=training_params.decay_rate,
                   actions=possible_actions)