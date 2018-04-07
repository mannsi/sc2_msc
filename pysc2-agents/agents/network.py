from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.layers as layers


def build_net(screen, ssize, num_action, ntype):
    if ntype == 'atari':
        return build_atari(screen, ssize, num_action)
    elif ntype == 'fcn':
        return build_fcn(screen, num_action)
    else:
        raise 'FLAGS.net must be atari or fcn'


def build_atari(screen, ssize, num_action):
    # Extract features
    sconv1 = layers.conv2d(tf.transpose(screen, [0, 2, 3, 1]),
                           num_outputs=16,
                           kernel_size=8,
                           stride=4,
                           scope='sconv1')
    sconv2 = layers.conv2d(sconv1,
                           num_outputs=32,
                           kernel_size=4,
                           stride=2,
                           scope='sconv2')

    # Compute spatial actions, non spatial actions and value
    feat_fc = tf.concat([layers.flatten(sconv2)], axis=1)
    feat_fc = layers.fully_connected(feat_fc,
                                     num_outputs=256,
                                     activation_fn=tf.nn.relu,
                                     scope='feat_fc')

    spatial_action_x = layers.fully_connected(feat_fc,
                                              num_outputs=ssize,
                                              activation_fn=tf.nn.softmax,
                                              scope='spatial_action_x')
    spatial_action_y = layers.fully_connected(feat_fc,
                                              num_outputs=ssize,
                                              activation_fn=tf.nn.softmax,
                                              scope='spatial_action_y')
    spatial_action_x = tf.reshape(spatial_action_x, [-1, 1, ssize])
    spatial_action_x = tf.tile(spatial_action_x, [1, ssize, 1])
    spatial_action_y = tf.reshape(spatial_action_y, [-1, ssize, 1])
    spatial_action_y = tf.tile(spatial_action_y, [1, 1, ssize])
    spatial_action = layers.flatten(spatial_action_x * spatial_action_y)

    non_spatial_action = layers.fully_connected(feat_fc,
                                                num_outputs=num_action,
                                                activation_fn=tf.nn.softmax,
                                                scope='non_spatial_action')
    value = tf.reshape(layers.fully_connected(feat_fc,
                                              num_outputs=1,
                                              activation_fn=None,
                                              scope='value'), [-1])

    return spatial_action, non_spatial_action, value


def build_fcn(screen, num_action):
    # Extract features
    sconv1 = layers.conv2d(tf.transpose(screen, [0, 2, 3, 1]),
                           num_outputs=16,
                           kernel_size=5,
                           stride=1,
                           scope='sconv1')
    sconv2 = layers.conv2d(sconv1,
                           num_outputs=32,
                           kernel_size=3,
                           stride=1,
                           scope='sconv2')

    # Compute spatial actions
    feat_conv = tf.concat([sconv2], axis=3)
    spatial_action = layers.conv2d(feat_conv,
                                   num_outputs=1,
                                   kernel_size=1,
                                   stride=1,
                                   activation_fn=None,
                                   scope='spatial_action')
    spatial_action = tf.nn.softmax(layers.flatten(spatial_action))

    # Compute non spatial actions and value
    feat_fc = tf.concat([layers.flatten(sconv2)], axis=1)
    feat_fc = layers.fully_connected(feat_fc,
                                     num_outputs=256,
                                     activation_fn=tf.nn.relu,
                                     scope='feat_fc')
    non_spatial_action = layers.fully_connected(feat_fc,
                                                num_outputs=num_action,
                                                activation_fn=tf.nn.softmax,
                                                scope='non_spatial_action')
    value = tf.reshape(layers.fully_connected(feat_fc,
                                              num_outputs=1,
                                              activation_fn=None,
                                              scope='value'), [-1])

    return spatial_action, non_spatial_action, value
