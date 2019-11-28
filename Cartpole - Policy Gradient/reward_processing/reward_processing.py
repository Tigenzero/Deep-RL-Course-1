import numpy as np


class RewardProcessing(object):
    def discount_rewards(self, episode_rewards, gamma):
        discounted_rewards = np.zeros_like(episode_rewards)
        cumulative = 0.0
        for i in reversed(range(len(episode_rewards))):
            cumulative = cumulative * gamma + episode_rewards[i]
            discounted_rewards[i] = cumulative
        return discounted_rewards

    def normalize_rewards(self, episode_rewards):
        mean = np.mean(episode_rewards)
        std = np.std(episode_rewards)
        return (episode_rewards - mean) / std

    def discount_and_normalize_rewards(self, episode_rewards, gamma):
        discounted_rewards = self.discount_rewards(episode_rewards, gamma)
        return self.normalize_rewards(discounted_rewards)
