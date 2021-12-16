# -*- coding: utf-8  -*-
# -*- author: jokker -*-

import argparse

# todo 完善一下这个函数，直接在其他地方调用就行，不用再重新写一遍了
# todo 完善这个 util 当做参考文档


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='run model')
    parser.add_argument('--host', dest='host',type=str,default='192.168.3.101')
    parser.add_argument('--port',dest='port',type=str,default='8084')

    # 传入多个参数中间使用逗号隔开，可以使用参数 nargs 指定参数个数，为 ‘+’ 代表可以任意参数个数
    parser.add_argument('--more', dest='more', nargs='+', help='list')


    args = parser.parse_args()
    return args


if __name__ == "__main__":





    pass
