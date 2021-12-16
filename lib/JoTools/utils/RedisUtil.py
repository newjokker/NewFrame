# -*- coding: utf-8  -*-
# -*- author: jokker -*-


import redis
import pickle
import time
import cv2
import numpy as np


class RedisUtil(object):

    def __init__(self, host, port):
        self.r = redis.StrictRedis(host=host, port=port)

    def clear_all(self):
        """删除所有的信息"""
        for each_key in r.keys():
            self.r.delete(each_key)

    def insert_image(self, frame_id, img_path):
        # 将图片序列化存入redis中
        start_time = time.time()
        frame = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), 1)
        after_read = time.time()
        b = pickle.dumps(frame)  # frame is numpy.ndarray
        self.r.set(frame_id, b)
        after_insert = time.time()
        return after_read - start_time, after_insert - after_read

    def get_image(self, frame_id):
        # 从redis中取出序列化的图片并进行反序列化
        start_time = time.time()
        a = pickle.loads(self.r.get(frame_id))
        print(a.shape)
        return time.time() - start_time

    def set(self, key, value):
        """插入数据"""
        self.r.set(key, value)

    def get(self, key):
        """获取数据"""
        return self.r.get(key)

if __name__ == "__main__":

    a = RedisUtil("192.168.3.185", "6379")

    a.set('name', 'jokker')

    print(a.get('name'))




