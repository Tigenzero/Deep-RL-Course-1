import os
import logging.config
from taxi_params import ExplorationParams, EpisodeParams
from taxi.taxi_env import Taxi

LOAD_Q_TABLE = False
Q_TABLE_FILENAME = 'qtable.txt'

if __name__ == "__main__":
    logfile = os.path.join('logs', 'logging_file.log')
    print(logfile)
    if not os.path.isdir('logs'):
        os.makedirs('logs')
    logging.config.fileConfig('logging.conf', defaults={'logfile': logfile})
    logging.debug("Starting main")

    episode_params = EpisodeParams()
    exploration_params = ExplorationParams()

    taxi_game = Taxi(episode_params, exploration_params.epsilon)

    if not LOAD_Q_TABLE:
        for training_episode in range(episode_params.total_training_episodes):
            taxi_game.execute_training_episode(episode_params.max_steps)
            taxi_game.epsilon = taxi_game.reduce_exploration(training_episode,
                                                             exploration_params.min_epsilon,
                                                             exploration_params.max_epsilon,
                                                             exploration_params.decay_rate)

        taxi_game.save_q_table(Q_TABLE_FILENAME)
    else:
        taxi_game.load_q_table(Q_TABLE_FILENAME)
    rewards = []
    for test_episode in range(episode_params.total_test_episodes):
        logging.info("***************************")
        logging.info("EPISODE {}".format(test_episode))
        total_rewards = taxi_game.execute_test_episode(episode_params.max_steps)
        if total_rewards is not None:
            rewards.append(total_rewards)
    taxi_game.env.close()
    logging.info("Score over time: {}".format(str(sum(rewards)/episode_params.total_test_episodes)))
