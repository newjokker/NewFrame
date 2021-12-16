# -*- coding: utf-8  -*-
# -*- author: jokker -*-

# todo add random array

import random
import numpy as np


class RandomUtil(object):

    @staticmethod
    def choice(data):
        """Choose a random element from a non-empty sequence."""
        return random.choice(data)

    @staticmethod
    def randint(min_number, max_number):
        """Return random integer in range [a, b], including both end points."""
        return random.randint(min_number, max_number)

    @staticmethod
    def random():
        """random() -> x in the interval [0, 1)."""
        return random.random()

    @staticmethod
    def shuffle(data):
        """random=random.random -> shuffle list x in place; return None.
        Optional arg random is a 0-argument function returning a random
        float in [0.0, 1.0); by default, the standard random.random."""
        return random.shuffle(data)

    @staticmethod
    def randrange(start, stop=None, step=1, _int=int):
        """Choose a random item from range(start, stop[, step])"""
        return random.randrange(start, stop, step=step, _int=_int)

    @staticmethod
    def rand_range_float(minute, maxute, keep_decimal=3):
        """返回一定范围内的一个保留一定位数的随机浮点数"""
        return round(float(minute) + random.random() * (float(maxute) - float(minute)), keep_decimal)


class RandomNumpyUtil(object):
    """获取随机的数组"""


