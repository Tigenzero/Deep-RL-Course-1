
import os
import logging
from .params import EpisodeParams, ExplorationParams


if __name__ == "__main__":
    logfile = os.path.join('logs', 'logging_file.log')
    print(logfile)
    if not os.path.isdir('logs'):
        os.makedirs('logs')
    logging.config.fileConfig('logging.conf')
    logging.debug("Starting main")

    episode_params = EpisodeParams()
    exploration_params = ExplorationParams()


    # get tax_env
    # create for loop of episodes
        # execute episode
        # reduce exploratoin
