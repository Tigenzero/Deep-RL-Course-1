import os
import logging.config
# from params import EpisodeParams, ExplorationParams
from params import ExplorationParams, EpisodeParams
from taxi.taxi_env import Taxi


if __name__ == "__main__":
    logfile = os.path.join('logs', 'logging_file.log')
    print(logfile)
    if not os.path.isdir('logs'):
        os.makedirs('logs')
    logging.config.fileConfig('logging.conf', defaults={'logfile': logfile})
    logging.debug("Starting main")

    episode_params = EpisodeParams()
    exploration_params = ExplorationParams()

    taxi_game = Taxi(episode_params, exploration_params)


    # get tax_env
    # create for loop of episodes
        # execute episode
        # reduce exploration
