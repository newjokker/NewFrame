# -*- coding: utf-8  -*-
# -*- author: jokker -*-

import time
import os
from functools import wraps


class DecoratorUtil(object):

    @staticmethod
    def buddha_bless_me(func):
        """ 佛祖保佑 """

        @wraps(func)
        def wrapper(*args, **kwargs):
            print(""" 

                        _ooOoo_
                       o8888888o
                       88" . "88
                       (| -_- |)
                       O\  =  /O
                    ____/`---'\____
                  .'  \|     |  `.
                 /  \|||||  :  |||  |
                /  _||||| -:- |||||-  |
                |   | \  -  / |   |
                | \_|  ''\---/''  |   |
                \  .-\__  `-`  ___/-. /
              ___`. .'  /--.--\  `. . __
           ."" '<  `.___\_<|>_/___.'  >'"".
          | | :  `- \`.;`\ _ /`;.`/ - ` : | |
          \  \ `-.   \_ __\ /__ _/   .-` /  /
     ======`-.____`-.___\_____/___.-`____.-'======
                        `=---='
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                  Buddha Bless, No Bug !

             """)
            result = func(*args, **kwargs)
            return result

        return wrapper

    @staticmethod
    def sleipmon_bless_me(func):
        """ 神兽羊驼保佑 """

        @wraps(func)
        def wrapper(*args, **kwargs):
            print(""" 

                  ┏┛ ┻━━━━━┛ ┻┓
                  ┃　　　　　　 ┃
                  ┃　　　━　　　┃
                  ┃　┳┛　  ┗┳　┃
                  ┃　　　　　　 ┃
                  ┃　　　┻　　　┃
                  ┃　　　　　　 ┃
                  ┗━┓　　　┏━━━┛
                    ┃　　　┃   神兽保佑
                    ┃　　　┃   代码无BUG！
                    ┃　　　┗━━━━━━━━━┓
                    ┃　　　　　　　    ┣┓
                    ┃　　　　         ┏┛
                    ┗━┓ ┓ ┏━━━┳ ┓ ┏━┛
                      ┃ ┫ ┫   ┃ ┫ ┫
                      ┗━┻━┛   ┗━┻━┛

                    """)
            result = func(*args, **kwargs)
            return result

        return wrapper

    @staticmethod
    def tangbohu(func):
        """ 佛祖保佑 """

        @wraps(func)
        def wrapper(*args, **kwargs):
            print(""" 

      佛曰:
              写字楼里写字间，写字间里程序员；
              程序人员写程序，又拿程序换酒钱。
              酒醒只在网上坐，酒醉还来网下眠；
              酒醉酒醒日复日，网上网下年复年。
              但愿老死电脑间，不愿鞠躬老板前；
              奔驰宝马贵者趣，公交自行程序员。
              别人笑我忒疯癫，我笑自己命太贱；
              不见满街漂亮妹，哪个归得程序员？

             """)
            result = func(*args, **kwargs)
            return result

        return wrapper

    @staticmethod
    def no_bug_forever(func):
        """永无bug，一个永无 bug 的文字瀑布在程序运行的时候慢慢晃过"""

        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            # 得到当前文件地址
            now_dir = os.path.dirname(os.path.realpath(__file__))
            # 辅助文件地址
            aux_dir = os.path.join(now_dir, 'Auxdata')
            # 将辅助文件生成在辅助文件夹下面
            if not os.path.exists(aux_dir):
                os.makedirs(aux_dir)
            return result

        return wrapper

    @staticmethod
    def time_this(func):
        """计算函数运行的时间"""

        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            print('{0} --> {1}(s)'.format(func.__name__, end - start))
            return result

        return wrapper

    @staticmethod
    def log(func):
        pass

