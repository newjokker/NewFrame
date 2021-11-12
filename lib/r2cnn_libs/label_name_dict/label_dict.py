# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import sys
this_dir = os.path.dirname(__file__)
lib_path = os.path.join(this_dir, '..')
sys.path.insert(0, lib_path)

from configs import cfgs

if cfgs.DATASET_NAME == 'ship':
    NAME_LABEL_MAP = {
        'back_ground': 0,
        'ship': 1
    }
elif cfgs.DATASET_NAME == 'FDDB':
    NAME_LABEL_MAP = {
        'back_ground': 0,
        'face': 1
    }
elif cfgs.DATASET_NAME == 'ICDAR2015':
    NAME_LABEL_MAP = {
        'back_ground': 0,
        'text': 1
    }
elif cfgs.DATASET_NAME.startswith('DOTA'):
    NAME_LABEL_MAP = {
        'back_ground': 0,
        'roundabout': 1,
        'tennis-court': 2,
        'swimming-pool': 3,
        'storage-tank': 4,
        'soccer-ball-field': 5,
        'small-vehicle': 6,
        'ship': 7,
        'plane': 8,
        'large-vehicle': 9,
        'helicopter': 10,
        'harbor': 11,
        'ground-track-field': 12,
        'bridge': 13,
        'basketball-court': 14,
        'baseball-diamond': 15
    }
elif cfgs.DATASET_NAME == 'pascal':
    NAME_LABEL_MAP = {
        'back_ground': 0,
        'jyz2':1,
    }
elif cfgs.DATASET_NAME == 'connection':
    NAME_LABEL_MAP = {
        'back_ground': 0,
        'PGuaBan':1,
        'UGuaHuan':2,
        'ZGuaBan':3,
        'Sanjiaoban':4,
        'Zhongchui':5,
        'XuanChuiXianJia':6
    }
else:
    assert 'please set label dict!'


def get_label_name_map():
    reverse_dict = {}
    for name, label in NAME_LABEL_MAP.items():
        reverse_dict[label] = name
    return reverse_dict

LABEl_NAME_MAP = get_label_name_map()
