import os
import logging
import logging.config
import warnings
from cartpole_params import NeuralNetParams, TrainingParams
from cartpole_environment.environment import Environment
from neural_net.neural_net import NeuralNet
from cartpole_trainer.trainer import Trainer

if __name__ == "__main__":
    logfile = os.path.join('logs', 'logging_file.log')
    print(logfile)
    if not os.path.isdir('logs'):
        os.makedirs('logs')
    logging.config.fileConfig('logging.conf', defaults={'logfile': logfile})
    logging.debug("Starting main")
    warnings.filterwarnings('ignore')
    neural_net_params = NeuralNetParams()
    training_params = TrainingParams()
    environment = Environment(training_params.render_environment)

    neural_net = NeuralNet()
    neural_net.initialize(neural_net_params.state_size,
                          environment.get_action_size(),
                          training_params.learning_rate)
    neural_net.setup_tensorboard(training_params.tensorboard_folder)
    trainer = Trainer(training_params.model_path)
    if training_params.train:
        trainer.train_model(environment, neural_net, training_params.max_training_episodes, training_params.gamma)
    if training_params.test:
        trainer.test_model(environment, neural_net, training_params.max_testing_episodes)
