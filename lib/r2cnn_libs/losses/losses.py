# -*- coding: utf-8 -*-
"""
@author: jemmy li
@contact: zengarden2009@gmail.com
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os
import numpy as np

this_dir = os.path.dirname(__file__)
lib_path = os.path.join(this_dir, '..')#,'..','libs')
sys.path.insert(0, lib_path)

import tensorflow as tf
from configs import cfgs


def _smooth_l1_loss_base(bbox_pred, bbox_targets, sigma=1.0):
    '''

    :param bbox_pred: [-1, 4] in RPN. [-1, cls_num+1, 4] or [-1, cls_num+1, 5] in Fast-rcnn
    :param bbox_targets: shape is same as bbox_pred
    :param sigma:
    :return:
    '''
    sigma_2 = sigma**2

    box_diff = bbox_pred - bbox_targets

    abs_box_diff = tf.abs(box_diff)

    smoothL1_sign = tf.stop_gradient(
        tf.to_float(tf.less(abs_box_diff, 1. / sigma_2)))
    loss_box = tf.pow(box_diff, 2) * (sigma_2 / 2.0) * smoothL1_sign \
               + (abs_box_diff - (0.5 / sigma_2)) * (1.0 - smoothL1_sign)
    return loss_box


def smooth_l1_loss_rpn(bbox_pred, bbox_targets, label, sigma=1.0):
    '''

    :param bbox_pred: [-1, 4]
    :param bbox_targets: [-1, 4]
    :param label: [-1]
    :param sigma:
    :return:
    '''
    value = _smooth_l1_loss_base(bbox_pred, bbox_targets, sigma=sigma)

    # value = tf.reduce_mean(value, axis=1)  # to sum in axis 1
    # rpn_select = tf.reshape(tf.where(tf.greater_equal(label, 0)), [-1])

    value = tf.reduce_sum(value, axis=1)  # to sum in axis 1
    rpn_select = tf.where(tf.greater(label, 0))

    # rpn_select = tf.stop_gradient(rpn_select) # to avoid
    selected_value = tf.gather(value, rpn_select)

    non_ignored_mask = tf.stop_gradient(
        1.0 - tf.to_float(tf.equal(label, -1)))  # positve is 1.0 others is 0.0

    bbox_loss = tf.reduce_sum(selected_value) / tf.maximum(1.0, tf.reduce_sum(non_ignored_mask))

    return bbox_loss


def smooth_l1_loss_rcnn_h(bbox_pred, bbox_targets, label, num_classes, sigma=1.0):
    '''

    :param bbox_pred: [-1, (cfgs.CLS_NUM +1) * 4]
    :param bbox_targets:[-1, (cfgs.CLS_NUM +1) * 4]
    :param label:[-1]
    :param num_classes:
    :param sigma:
    :return:
    '''

    outside_mask = tf.stop_gradient(tf.to_float(tf.greater(label, 0)))

    bbox_pred = tf.reshape(bbox_pred, [-1, num_classes, 4])
    bbox_targets = tf.reshape(bbox_targets, [-1, num_classes, 4])

    value = _smooth_l1_loss_base(bbox_pred,
                                 bbox_targets,
                                 sigma=sigma)
    value = tf.reduce_sum(value, 2)
    value = tf.reshape(value, [-1, num_classes])

    inside_mask = tf.one_hot(tf.reshape(label, [-1, 1]),
                             depth=num_classes, axis=1)

    inside_mask = tf.stop_gradient(
        tf.to_float(tf.reshape(inside_mask, [-1, num_classes])))

    normalizer = tf.to_float(tf.shape(bbox_pred)[0])
    bbox_loss = tf.reduce_sum(
        tf.reduce_sum(value * inside_mask, 1)*outside_mask) / normalizer

    return bbox_loss


def smooth_l1_loss_rcnn_r(bbox_pred, bbox_targets, label, num_classes, sigma=1.0):
    '''

    :param bbox_pred: [-1, (cfgs.CLS_NUM +1) * 5]
    :param bbox_targets:[-1, (cfgs.CLS_NUM +1) * 5]
    :param label:[-1]
    :param num_classes:
    :param sigma:
    :return:
    '''

    outside_mask = tf.stop_gradient(tf.to_float(tf.greater(label, 0)))

    bbox_pred = tf.reshape(bbox_pred, [-1, num_classes, 5])
    bbox_targets = tf.reshape(bbox_targets, [-1, num_classes, 5])

    value = _smooth_l1_loss_base(bbox_pred,
                                 bbox_targets,
                                 sigma=sigma)
    value = tf.reduce_sum(value, 2)
    value = tf.reshape(value, [-1, num_classes])

    inside_mask = tf.one_hot(tf.reshape(label, [-1, 1]),
                             depth=num_classes, axis=1)

    inside_mask = tf.stop_gradient(
        tf.to_float(tf.reshape(inside_mask, [-1, num_classes])))

    normalizer = tf.to_float(tf.shape(bbox_pred)[0])
    bbox_loss = tf.reduce_sum(
        tf.reduce_sum(value * inside_mask, 1)*outside_mask) / normalizer

    return bbox_loss


def sum_ohem_loss(cls_score, label, bbox_pred, bbox_targets,
                  nr_ohem_sampling, nr_classes, sigma=1.0):

    raise NotImplementedError('Not implement now.')


def build_attention_loss(mask, featuremap):
    # shape = mask.get_shape().as_list()
    shape = tf.shape(mask)
    featuremap = tf.image.resize_bilinear(featuremap, [shape[0], shape[1]])
    # shape = tf.shape(featuremap)
    # mask = tf.expand_dims(mask, axis=0)
    # mask = tf.image.resize_bilinear(mask, [shape[1], shape[2]])
    # mask = tf.squeeze(mask, axis=0)

    mask = tf.cast(mask, tf.int32)
    mask = tf.reshape(mask, [-1, ])
    featuremap = tf.reshape(featuremap, [-1, 2])
    attention_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=mask, logits=featuremap)
    attention_loss = tf.reduce_mean(attention_loss)
    return attention_loss
