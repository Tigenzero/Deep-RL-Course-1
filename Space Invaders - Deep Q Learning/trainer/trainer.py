import tensorflow.compat.v1 as tf
import numpy as np
import logging
tf.disable_v2_behavior()


class Trainer(object):
    def train_model(self, environment, player, training_params):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            player.reset_decay_step()
            for episode in range(training_params.total_episodes):
                step = 0
                episode_rewards = []
                environment.reset_environment()
                state, stacked_frames = environment.init_stack_frames(environment.get_state())

                while step < training_params.max_steps:
                    step += 1
                    player.increase_decay_step()
                    action, explore_probability = player.predict_action(state, sess)
                    next_state, reward, done, _ = environment.take_action(action)

                    if training_params.episode_render:
                        environment.render_environment()

                    episode_rewards.append(reward)

                    if done:
                        next_state = environment.create_zero_state(state)
                        next_state, stacked_frames = environment.stack_frames(stacked_frames, next_state)
                        # end the episode by maxing the steps
                        step = training_params.max_steps
                        total_reward = np.sum(episode_rewards)

                        logging.info("Episode: {}".format(episode),
                                     "Total reward: {}".format(total_reward),
                                     "Explore P {:.4f}".format(explore_probability),
                                     "Training Loss {:.4f}".format(player.deep_q_net.loss))

                        player.add_reward_to_list(episode, total_reward, training=True)
                        # Odd, previous memory adds did not add 'done' to the list. Keep an eye out if issues arise
                        environment.memory.add((state, action, reward, next_state, done))

                    else:
                        next_state, stacked_frames = environment.stack_frames(stacked_frames, next_state)
                        # same here
                        environment.memory.add((state, action, reward, next_state, done))
                        state = next_state

                    player.deep_q_net.learn(batch=environment.memory.sample(training_params.batch_size),
                                            session=sess,
                                            gamma=training_params.gamma,
                                            episode=episode)
                if episode % 5 == 0:
                    player.save_model(sess)
            # Save model just in case we pick a number not divisible by 5
            player.save_model(sess)

    def test_model(self, environment, player, episodes):
        with tf.Session() as sess:
            player.load_model(sess)
            for episode in range(episodes):
                total_rewards = 0
                environment.reset_environment()
                state, stacked_frames = environment.init_stack_frames(environment.get_state())
                logging.info("******************************************************")
                logging.info("EPISODE: ", episode)

                while True:
                    action = player.get_exploit_action(state, sess)
                    next_state, reward, done, _ = environment.take_action(action)
                    environment.render_environment()

                    total_rewards += reward

                    if done:
                        logging.info("Score: {}".format(total_rewards))
                        player.add_reward_to_list(episode, total_rewards)
                        break

                    next_state, stacked_frames = environment.stack_frames(stacked_frames, next_state)
                    state = next_state
