from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from pysc2.lib import actions
from pysc2.lib import features

from agents.network import build_net
import utils as U


class A3CAgent(object):
    """An agent specifically for solving the mini-game maps."""

    def __init__(self, training, ssize, name='A3C/A3CAgent'):
        self.name = name
        self.training = training
        self.summary = []
        self.ssize = ssize
        # self.isize = len(actions.FUNCTIONS)
        self.legal_action_ids = [actions.FUNCTIONS.Attack_screen.id]
        self.num_legal_actions = len(self.legal_action_ids)

    def initialize(self, sess, summary_writer):
        self.sess = sess
        self.summary_writer = summary_writer
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

    def reset(self):
        # Epsilon schedule
        self.epsilon = [0.05, 0.2]

    def build_model(self, reuse, dev, ntype):
        with tf.variable_scope(self.name) and tf.device(dev):
            if reuse:
                tf.get_variable_scope().reuse_variables()
                assert tf.get_variable_scope().reuse

            # Set inputs of networks
            # self.minimap = tf.placeholder(tf.float32, [None, U.minimap_channel(), self.msize, self.msize], name='minimap')
            self.screen = tf.placeholder(tf.float32, [None, U.screen_channel(), self.ssize, self.ssize], name='screen')
            # self.info = tf.placeholder(tf.float32, [None, self.isize], name='info')

            # Build networks
            # num_actions = len(actions.FUNCTIONS)
            net = build_net(self.screen, self.ssize, self.num_legal_actions, ntype)
            self.spatial_action, self.non_spatial_action, self.value = net

            # Set targets and masks
            self.valid_spatial_action = tf.placeholder(tf.float32, [None], name='valid_spatial_action')
            self.spatial_action_selected = tf.placeholder(tf.float32, [None, self.ssize ** 2],
                                                          name='spatial_action_selected')
            self.valid_non_spatial_action = tf.placeholder(tf.float32, [None, self.num_legal_actions],
                                                           name='valid_non_spatial_action')
            self.non_spatial_action_selected = tf.placeholder(tf.float32, [None, self.num_legal_actions],
                                                              name='non_spatial_action_selected')
            self.value_target = tf.placeholder(tf.float32, [None], name='value_target')

            # Compute log probability
            spatial_action_prob = tf.reduce_sum(self.spatial_action * self.spatial_action_selected, axis=1)
            spatial_action_log_prob = tf.log(tf.clip_by_value(spatial_action_prob, 1e-10, 1.))
            non_spatial_action_prob = tf.reduce_sum(self.non_spatial_action * self.non_spatial_action_selected, axis=1)
            valid_non_spatial_action_prob = tf.reduce_sum(self.non_spatial_action * self.valid_non_spatial_action,
                                                          axis=1)
            valid_non_spatial_action_prob = tf.clip_by_value(valid_non_spatial_action_prob, 1e-10, 1.)
            non_spatial_action_prob = non_spatial_action_prob / valid_non_spatial_action_prob
            non_spatial_action_log_prob = tf.log(tf.clip_by_value(non_spatial_action_prob, 1e-10, 1.))
            self.summary.append(tf.summary.histogram('spatial_action_prob', spatial_action_prob))
            self.summary.append(tf.summary.histogram('non_spatial_action_prob', non_spatial_action_prob))

            # Compute losses, more details in https://arxiv.org/abs/1602.01783
            # Policy loss and value loss
            action_log_prob = self.valid_spatial_action * spatial_action_log_prob + non_spatial_action_log_prob
            advantage = tf.stop_gradient(self.value_target - self.value)
            policy_loss = - tf.reduce_mean(action_log_prob * advantage)
            value_loss = - tf.reduce_mean(self.value * advantage)
            self.summary.append(tf.summary.scalar('policy_loss', policy_loss))
            self.summary.append(tf.summary.scalar('value_loss', value_loss))

            # TODO: policy penalty
            loss = policy_loss + value_loss

            # Build the optimizer
            self.learning_rate = tf.placeholder(tf.float32, None, name='learning_rate')
            opt = tf.train.RMSPropOptimizer(self.learning_rate, decay=0.99, epsilon=1e-10)
            grads = opt.compute_gradients(loss)
            cliped_grad = []
            for grad, var in grads:
                self.summary.append(tf.summary.histogram(var.op.name, var))
                self.summary.append(tf.summary.histogram(var.op.name + '/grad', grad))
                grad = tf.clip_by_norm(grad, 10.0)
                cliped_grad.append([grad, var])
            self.train_op = opt.apply_gradients(cliped_grad)
            self.summary_op = tf.summary.merge(self.summary)

            self.saver = tf.train.Saver(max_to_keep=100)

    def step(self, observation):
        # minimap = np.array(obs.observation['minimap'], dtype=np.float32)
        # minimap = np.expand_dims(U.preprocess_minimap(minimap), axis=0)  # MANNSI: scale inputs and then flatten to 1D
        # screen = np.array(obs.observation['screen'], dtype=np.float32)
        feature_index = features.SCREEN_FEATURES.player_id.index
        feature_values = observation['screen'][feature_index:feature_index+1]
        screen = np.array(feature_values, dtype=np.float32)
        screen = np.expand_dims(U.preprocess_screen(screen), axis=0)
        # TODO: only use available actions
        # info = np.zeros([1, self.isize], dtype=np.float32)
        # info[0, obs.observation['available_actions']] = 1

        # feed = {self.minimap: minimap,
        #         self.screen: screen,
        #         self.info: info}

        able_to_attack = actions.FUNCTIONS.Attack_screen.id in observation['available_actions']
        if not able_to_attack:
            actions.FunctionCall(actions.FUNCTIONS.select_army, [0])


        feed = {self.screen: screen}
        non_spatial_action, spatial_action = self.sess.run(
            [self.non_spatial_action, self.spatial_action],
            feed_dict=feed)

        # Select an action and a spatial target
        non_spatial_action = non_spatial_action.ravel()  # MANNSI: Convert from (1,x) shape to (x,) shape.
        spatial_action = spatial_action.ravel()

        # valid_actions = obs.observation['available_actions']
        # act_id = valid_actions[np.argmax(non_spatial_action[valid_actions])]
        act_id = self.legal_action_ids[np.argmax(non_spatial_action)]

        target = np.argmax(spatial_action)
        target = [int(target // self.ssize), int(target % self.ssize)]

        # Epsilon greedy exploration
        if self.training and np.random.rand() < self.epsilon[0]:
            act_id = np.random.choice(self.legal_action_ids)
        if self.training and np.random.rand() < self.epsilon[1]:
            dy = np.random.randint(-4, 5)
            target[0] = int(max(0, min(self.ssize - 1, target[0] + dy)))
            dx = np.random.randint(-4, 5)
            target[1] = int(max(0, min(self.ssize - 1, target[1] + dx)))

        # Set act_id and act_args
        act_args = []
        for arg in actions.FUNCTIONS[act_id].args:
            if arg.name in ('screen', 'minimap', 'screen2'):
                act_args.append([target[1], target[0]])
            else:
                act_args.append([0])  # TODO: Be careful
        return actions.FunctionCall(act_id, act_args)

    def update(self, replay_buffer, discount, lr, episode_counter):
        """
        
        :param replay_buffer: list of tuples containing [(prev_timestep, action, next_time_step)] 
        :param discount: Discount factor
        :param lr: Learning rate
        :param episode_counter: Episode counter
        :return: 
        """

        # Compute R, which is value of the last observation
        obs = replay_buffer[-1][-1]  # latest timestep
        if obs.last():
            R = 0
        else:
            # minimap = np.array(obs.observation['minimap'], dtype=np.float32)
            # minimap = np.expand_dims(U.preprocess_minimap(minimap), axis=0)

            feature_index = features.SCREEN_FEATURES.player_id.index
            feature_values = obs.observation['screen'][feature_index:feature_index + 1]
            # screen = np.array(obs.observation['screen'], dtype=np.float32)
            screen = np.array(feature_values, dtype=np.float32)
            screen = np.expand_dims(U.preprocess_screen(screen), axis=0)
            # info = np.zeros([1, self.isize], dtype=np.float32)
            # info[0, obs.observation['available_actions']] = 1

            # feed = {self.minimap: minimap,
            #         self.screen: screen,
            #         self.info: info}

            feed = {self.screen: screen}

            R = self.sess.run(self.value, feed_dict=feed)[0]

        # Compute targets and masks
        # minimaps = []
        screens = []
        # infos = []

        value_target = np.zeros([len(replay_buffer)], dtype=np.float32)
        value_target[-1] = R

        valid_spatial_action = np.zeros([len(replay_buffer)], dtype=np.float32)
        spatial_action_selected = np.zeros([len(replay_buffer), self.ssize ** 2], dtype=np.float32)
        valid_non_spatial_action = np.zeros([len(replay_buffer), self.num_legal_actions], dtype=np.float32)
        non_spatial_action_selected = np.zeros([len(replay_buffer), self.num_legal_actions], dtype=np.float32)

        replay_buffer.reverse()
        for i, [obs, action, next_obs] in enumerate(replay_buffer):
            # minimap = np.array(obs.observation['minimap'], dtype=np.float32)
            # minimap = np.expand_dims(U.preprocess_minimap(minimap), axis=0)

            feature_index = features.SCREEN_FEATURES.player_id.index
            feature_values = obs.observation['screen'][feature_index:feature_index + 1]
            # screen = np.array(obs.observation['screen'], dtype=np.float32)
            screen = np.array(feature_values, dtype=np.float32)
            screen = np.expand_dims(U.preprocess_screen(screen), axis=0)

            # screen = np.array(obs.observation['screen'], dtype=np.float32)
            # screen = np.expand_dims(U.preprocess_screen(screen), axis=0)
            # info = np.zeros([1, self.isize], dtype=np.float32)
            # info[0, obs.observation['available_actions']] = 1

            # minimaps.append(minimap)
            screens.append(screen)
            # infos.append(info)

            reward = obs.reward
            act_id = action.function  # MANNSI TODO:
            act_index = self.action_index_to_array_indices([act_id])[0]
            act_args = action.arguments

            value_target[i] = reward + discount * value_target[i - 1]

            # valid_actions = obs.observation["available_actions"]
            # valid_non_spatial_action[i, valid_actions] = 1
            legal_action_indices = self.action_index_to_array_indices(self.legal_action_ids)
            assert len(legal_action_indices) == self.num_legal_actions
            valid_non_spatial_action[i, legal_action_indices] = 1
            non_spatial_action_selected[i, act_index] = 1

            args = actions.FUNCTIONS[act_id].args
            for arg, act_arg in zip(args, act_args):
                if arg.name in ('screen', 'minimap', 'screen2'):
                    ind = act_arg[1] * self.ssize + act_arg[0]
                    valid_spatial_action[i] = 1
                    spatial_action_selected[i, ind] = 1

        # minimaps = np.concatenate(minimaps, axis=0)
        screens = np.concatenate(screens, axis=0)
        # infos = np.concatenate(infos, axis=0)

        # Train
        # feed = {self.minimap: minimaps,
        #         self.screen: screens,
        #         self.info: infos,
        #         self.value_target: value_target,
        #         self.valid_spatial_action: valid_spatial_action,
        #         self.spatial_action_selected: spatial_action_selected,
        #         self.valid_non_spatial_action: valid_non_spatial_action,
        #         self.non_spatial_action_selected: non_spatial_action_selected,
        #         self.learning_rate: lr}

        # MANNSI TODO: Don't understand valid_spatial_action param and valid_non_spatial_action param
        feed = {self.screen: screens,
                self.value_target: value_target,
                self.valid_spatial_action: valid_spatial_action,
                self.spatial_action_selected: spatial_action_selected,
                self.valid_non_spatial_action: valid_non_spatial_action,
                self.non_spatial_action_selected: non_spatial_action_selected,
                self.learning_rate: lr}
        _, summary = self.sess.run([self.train_op, self.summary_op], feed_dict=feed)
        self.summary_writer.add_summary(summary, episode_counter)

    def action_index_to_array_indices(self, action_indexes):
        return [self.legal_action_ids.index(x) for x in action_indexes]

    def save_model(self, path, count):
        self.saver.save(self.sess, path + '/model.pkl', count)

    def load_model(self, path):
        ckpt = tf.train.get_checkpoint_state(path)
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        return int(ckpt.model_checkpoint_path.split('-')[-1])
