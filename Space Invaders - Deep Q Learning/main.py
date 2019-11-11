import os
import logging.config
import warnings
import tensorflow.compat.v1 as tf
from player.player import Player
from deep_q_params import TrainingParams
from trainer.trainer import Trainer
from environment.environment import Environment
tf.disable_v2_behavior()

TEST_EPISODES = 2

if __name__ == "__main__":
    logfile = os.path.join('logs', 'logging_file.log')
    print(logfile)
    if not os.path.isdir('logs'):
        os.makedirs('logs')
    logging.config.fileConfig('logging.conf', defaults={'logfile': logfile})
    logging.debug("Starting main")
    warnings.filterwarnings('ignore')
    trainer = Trainer()
    environment = Environment()
    training_params = TrainingParams()
    player = Player.create_class_with_param_object(training_params=training_params,
                                                   possible_actions=environment.get_possible_actions(),
                                                   action_size=environment.get_action_size())
    logging.info("Size of Frame: {}".format(environment.get_observation_space()))
    logging.info("Action Size: {}".format(environment.get_action_size()))

    tf.reset_default_graph
    player.deep_q_net.initialize()
    environment.initialize_memory(player)
    # To Launch Tensorboard: tensorboard --logidr=/tensorboard/dqn/1
    player.deep_q_net.setup_tensorboard_writer("./tensorboard/dqn/1")

    player.init_saver()

    if training_params.training:
        trainer.train_model(environment=environment,
                            player=player,
                            training_params=training_params)
    trainer.test_model(environment=environment,
                       player=player,
                       episodes=2)
