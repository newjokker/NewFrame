# -*- coding: utf-8  -*-
# -*- author: jokker -*-

import argparse


class PrintUtil(object):

    @staticmethod
    def print(data):
        """用我习惯的方式进行打印"""

        if isinstance(data, list):
            PrintUtil._print_list(data)
        elif isinstance(data, argparse.Namespace):
            PrintUtil._print_args(data)
        elif isinstance(data, dict):
            PrintUtil._print_dict(data)
        else:
            raise TypeError("type : {0}  not support".format(type(data)))

    @staticmethod
    def _print_list(data):
        """打印列表"""
        for index, each in enumerate(data):
            print(index, each)

    @staticmethod
    def _print_args(args):
        """打印解析传入的参数信息"""
        print("*" + "-" * 30)
        for each in args.__dict__:
            print("{0} : {1} , [{2}]".format(each, args.__dict__[each], type(args.__dict__[each])))
        print("-" * 30 + "*")

    @staticmethod
    def _print_dict(data):
        """打印字典"""
        keys = sorted(data.keys())
        for each in keys:
            print("{0} : {1}".format(each, data[each]))







