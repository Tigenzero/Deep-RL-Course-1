class NeuralNetParams(object):
    def __init__(self):
        self.state_size = 4


class TrainingParams(object):
    def __init__(self):
        self.max_training_episodes = 300
        self.max_testing_episodes = 10
        self.learning_rate = 0.01
        self.gamma = 0.95
        self.model_path = "./models/model.ckpt"
        self.tensorboard_folder = "./tensorboard/pg/1"
        self.train = True
        self.test = True
        self.load_model = True
        self.render_environment = False
