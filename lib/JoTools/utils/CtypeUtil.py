# -*- coding: utf-8  -*-
# -*- author: jokker -*-

import ctypes

"""
# 可以使用 C++ 替换 python 的一些函数

* opencv 中的裁剪小图
* parse write xml
* 获取多边形的 iou
* nms center_nms
*  

"""



class CtypesUtil(object):

    def __init__(self, so_path):
        self.lib = ctypes.CDLL(so_path)

    def get_func(self, func_name):
        """获取函数"""
        return getattr(self.lib, func_name, None)



if __name__ == "__main__":

    a = CtypesUtil(r"test001.so")
    func = a.get_func("main")
    # 运行函数
    func()
