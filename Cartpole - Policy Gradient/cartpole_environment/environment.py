import gym

CARTPOLE_PATH = "CartPole-v0"


class Environment(object):
    def __init__(self):
        self.cartpole_path = CARTPOLE_PATH
        self.env = None
        self.action_size = None
        self.init_env()
        self.init_action_space()

    def init_env(self):
        self.env = gym.make(self.cartpole_path)
        self.env = self.env.unwrapped
        self.env.seed(1)

    def init_action_space(self):
        self.action_size = self.env.action_space.n

    def reset_environment(self):
        return self.env.reset()

    def render_environment(self):
        self.env.render()

    def take_action(self, action):
        return self.env.step(action)

    def get_action_size(self):
        return self.action_size