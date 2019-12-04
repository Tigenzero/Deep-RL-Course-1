import tensorflow as tf
import numpy as np
import logging
from reward_processing.reward_processing import RewardProcessing


class Trainer(object):
    def __init__(self, model_path, episode_save_num=100):
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.saver = None
        self.model_path = model_path
        self.episode_save_num = episode_save_num
        self.init_saver()

    def init_saver(self):
        self.saver = tf.train.Saver()

    def reset_episode_variables(self):
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []

    def save_model(self, episode, sess):
        if episode % self.episode_save_num == 0:
            self.saver.save(sess, self.model_path)
            logging.info("Model Saved")

    def load_model(self, sess):
        self.saver.restore(sess, self.model_path)
        logging.info("Model Loaded")

    def train_model(self, environment, neural_net, max_episodes, gamma):
        reward_processing = RewardProcessing()
        all_rewards = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for episode in range(max_episodes):

                state = environment.reset_environment()
                environment.render_environment()

                while True:
                    action_prob_distribution = sess.run(neural_net.action_distribution, feed_dict={neural_net.input_:
                                                                                                   state.reshape([1, 4])})
                    action = np.random.choice(range(action_prob_distribution.shape[1]), p=action_prob_distribution.ravel())
                    # TODO: Consider adding these from here to before "if done:" to function
                    new_state, reward, done, info = environment.take_action(action)
                    self.episode_states.append(state)
                    # we need [0., 1.] (if we take right) not just the index
                    # I hate using the same variable name and using _
                    action_ = np.zeros(environment.get_action_size())
                    action_[action] = 1

                    self.episode_actions.append(action_)
                    self.episode_rewards.append(reward)

                    if done:
                        # calculate sum reward
                        episode_rewards_sum = np.sum(self.episode_rewards)
                        all_rewards.append(episode_rewards_sum)
                        total_rewards = np.sum(all_rewards)

                        # mean reward
                        mean_reward = np.divide(total_rewards, episode+1)

                        max_reward_recorded = np.amax(all_rewards)

                        logging.info("==================")
                        logging.info("Episode: {}".format(episode))
                        logging.info("Reward: {}".format(episode_rewards_sum))
                        logging.info("Mean Reward: {}".format(mean_reward))
                        logging.info("Max reward so far: {}".format(max_reward_recorded))

                        # calculate discounted reward
                        discounted_episode_rewards = reward_processing.discount_and_normalize_rewards(self.episode_rewards, gamma)
                        # Feedforward, gradient, and backpropagation
                        loss_, _ = sess.run([neural_net.loss, neural_net.train_opt],
                                            feed_dict={neural_net.input_: np.vstack(np.array(self.episode_states)),
                                                       neural_net.actions: np.vstack(np.array(self.episode_actions)),
                                                       neural_net.discounted_episode_rewards_: discounted_episode_rewards})
                        # Write TF Summaries
                        summary = sess.run(neural_net.write_op,
                                           feed_dict={neural_net.input_: np.vstack(np.array(self.episode_states)),
                                                      neural_net.actions: np.vstack(np.array(self.episode_actions)),
                                                      neural_net.discounted_episode_rewards_: discounted_episode_rewards,
                                                      neural_net.mean_reward_: mean_reward})

                        neural_net.writer.add_summary(summary, episode)
                        neural_net.writer.flush()

                        self.reset_episode_variables()
                        break

                    state = new_state

                # Save model
                self.save_model(episode, sess)

    def test_model(self, environment, neural_net, max_episodes):
        with tf.Session() as sess:
            environment.reset_environment()
            rewards = []
            self.load_model(sess)
            for episode in range(max_episodes):
                state = environment.reset_environment()
                environment.render_environment()
                total_rewards = 0
                logging.info("*****************************************")
                logging.info("EPISODE {}".format(episode))

                while True:
                    # Choose action a, remember we aren't in a deterministic environment, we are output probabilities.
                    action_probability_distribution = sess.run(neural_net.action_distribution, feed_dict={
                        neural_net.input_: state.reshape([1, 4])})
                    logging.debug(action_probability_distribution)
                    # select action w.r.t the actions prob
                    action = np.random.choice(range(action_probability_distribution.shape[1]),
                                              p=action_probability_distribution.ravel())
                    new_state, reward, done, info = environment.take_action(action)
                    total_rewards += reward

                    if done:
                        rewards.append(total_rewards)
                        logging.info("Score: {}".format(total_rewards))
                        break
                    state = new_state
            logging.info("Score over time: {}".format(sum(rewards)/10))

