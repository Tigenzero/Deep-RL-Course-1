import tensorflow.compat.v1 as tf
import numpy as np
import logging
tf.disable_v2_behavior()


class Trainer(object):
    def train_model(self, environment, player, training_params):
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            sess.run(tf.global_variables_initializer())
            if training_params.use_existing_model:
                player.load_model(sess)
            player.reset_decay_step()
            for episode in range(training_params.total_episodes):
                step = 0
                episode_rewards = []
                environment.reset_environment()
                state, stacked_frames = environment.env_init_stack_frames(environment.get_state())

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
                        next_state, stacked_frames = environment.env_stack_frames(stacked_frames, next_state)
                        # end the episode by maxing the steps
                        step = training_params.max_steps
                        total_reward = np.sum(episode_rewards)

                        logging.info("Episode: {}".format(episode))
                        logging.info("Total reward: {}".format(total_reward))
                        logging.info("Explore P: {:.4f}".format(explore_probability))
                        logging.info("Training Loss: {:.4f}".format(loss))

                        player.add_reward_to_list(episode, total_reward, training=True)
                        environment.memory.add((state, action, reward, next_state, done))

                    else:
                        next_state, stacked_frames = environment.env_stack_frames(stacked_frames, next_state)
                        # same here
                        environment.memory.add((state, action, reward, next_state, done))
                        state = next_state

                    loss = player.deep_q_net.learn(batch=environment.memory.sample(training_params.batch_size),
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
                # frame_num = 1
                total_rewards = 0
                environment.reset_environment()
                state, stacked_frames = environment.env_init_stack_frames(environment.get_state())
                logging.info("******************************************************")
                logging.info("EPISODE: {}".format(episode))

                while True:
                    action = player.get_exploit_action(state, sess)
                    next_state, reward, done, _ = environment.take_action(action)
                    environment.render_environment()

                    total_rewards += reward

                    if done:
                        logging.info("Score: {}".format(total_rewards))
                        player.add_reward_to_list(episode, total_rewards)
                        break

                    next_state, stacked_frames = environment.env_stack_frames(stacked_frames, next_state)
                    #  For Testing that frames look like they are supposed to
                    # if frame_num % 10 == 0:
                    #     frame_filename = "E{}F{}.png".format(episode, frame_num)
                    #     environment.save_frame(stacked_frames[-1], frame_filename)
                    # frame_num += 1
                    state = next_state
