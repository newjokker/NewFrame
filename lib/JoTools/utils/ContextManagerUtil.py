# -*- coding: utf-8  -*-
# -*- author: jokker -*-

import time
from contextlib import contextmanager



@contextmanager
def time_this(label):
    """函数计时器"""
    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        print('{0}: {1}'.format(label, end-start))



class time_this_002():

    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        end = time.time()
        print('{0}: {1}'.format(self.label, end - self.start))





if __name__ == "__main__":


    with time_this_002('counting'):
        for i in range(10):
            time.sleep(0.1)

    with time_this('counting'):
        for i in range(10):
            time.sleep(0.1)













